"""Pure tests for the offline normalization-transfer artifact audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.single_uav_ppo_a_norm_audit import (
    _Audit,
    audit_artifacts,
    challenge_release_gate,
    normal_release_gate,
    recompute_gae,
    recompute_metrics,
    parse_vector,
)


def test_parse_vector_and_gae_are_pure() -> None:
    assert parse_vector("[1, 2, 3]") == [1.0, 2.0, 3.0]
    assert parse_vector("[1, 2]") is None
    records = [
        {"record_index": 0, "reward": 1.0, "old_value": 0.0, "next_value": 0.0, "done": False, "terminated": False, "truncated": False},
        {"record_index": 1, "reward": 2.0, "old_value": 0.0, "next_value": 0.0, "done": True, "terminated": True, "truncated": False},
    ]
    result = recompute_gae(records, gamma=1.0, gae_lambda=1.0)
    assert result["count"] == 2
    np.testing.assert_allclose(result["raw_advantages"], [3.0, 2.0])
    assert result["max_residual"] == 0.0


def test_normal_gate_uses_floor_one_percent_limits() -> None:
    train = {"all": {"episode_count": 6, "actual_steps": 120, "task_success_count": 4, "collision_episode_count": 0, "physical_contact_steps": 0, "boundary_attempt_steps": 1, "boundary_projection_steps": 1}}
    validation = {"all": {"episode_count": 8, "actual_steps": 120, "task_success_count": 5, "median_final_distance_m": 1.0, "collision_episode_count": 0, "physical_contact_steps": 0, "boundary_attempt_steps": 1, "boundary_projection_steps": 1}}
    assert normal_release_gate(train, validation)["passed"] is True
    train["all"]["boundary_attempt_steps"] = 2
    assert normal_release_gate(train, validation)["passed"] is False


def test_challenge_gate_requires_both_challenge_types() -> None:
    metrics = {"all": {"episode_count": 4, "actual_steps": 480, "task_success_count": 3, "collision_episode_count": 0, "physical_contact_steps": 0, "boundary_attempt_steps": 0, "boundary_projection_steps": 0}}
    episodes = [
        {"challenge_type": "stationary", "task_success": 1},
        {"challenge_type": "stationary", "task_success": 0},
        {"challenge_type": "fast", "task_success": 1},
        {"challenge_type": "fast", "task_success": 0},
    ]
    assert challenge_release_gate(metrics, episodes)["passed"] is True
    episodes[2]["task_success"] = 0
    assert challenge_release_gate(metrics, episodes)["passed"] is False


def test_metrics_are_recomputed_without_environment() -> None:
    rows = [
        {"steps": 3, "task_success": 1, "collision": 0, "physical_contact_steps": 0, "boundary_attempt_steps": 0, "boundary_projection_steps": 0, "final_actual_goal_distance_m": 0.5},
        {"steps": 3, "task_success": 0, "collision": 0, "physical_contact_steps": 0, "boundary_attempt_steps": 1, "boundary_projection_steps": 1, "final_actual_goal_distance_m": 2.5},
    ]
    metrics = recompute_metrics(rows)
    assert metrics["all"]["actual_steps"] == 6
    assert metrics["all"]["task_success_count"] == 1
    assert metrics["all"]["median_final_distance_m"] == 1.5


def test_missing_future_artifacts_are_pending_and_output_is_exclusive(tmp_path: Path) -> None:
    data = tmp_path / "data" / "synthetic" / "single_uav_ppo_a_norm_transfer_v2_seed2026_20261011"
    logs = tmp_path / "logs" / "preliminary" / "single_uav_ppo_a_norm_transfer_v2_seed2026_20261011"
    data.mkdir(parents=True)
    logs.mkdir(parents=True)
    (data / "run_specification.lock.json").write_text(json.dumps({"schema_version": "test"}), encoding="utf-8")
    output = tmp_path / "audit.json"
    report = audit_artifacts(logs, project_root=tmp_path, output=output)
    assert report["status"] in {"failed", "pending"}
    assert report["env_steps_executed"] == 0
    assert report["ppo_updates_executed"] == 0
    with pytest.raises(FileExistsError):
        audit_artifacts(logs, project_root=tmp_path, output=output)


def test_audit_collector_has_no_environment_or_checkpoint_side_effects(tmp_path: Path) -> None:
    audit = _Audit(tmp_path, tmp_path / "logs", tmp_path / "data")
    assert audit.log_root == (tmp_path / "logs").resolve()
    assert audit.data_root == (tmp_path / "data").resolve()
