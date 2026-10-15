"""Focused tests for the read-only paired smoke artifact audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.single_uav_ppo_smoke_curriculum import build_smoke_curriculum

from src.analysis.single_uav_ppo_smoke_audit import (
    _Audit,
    _check_actor_contract,
    audit_smoke_artifacts,
    parse_vector,
    recompute_episode_metrics,
    total_correction,
)


def test_vector_and_correction_helpers_require_finite_three_vectors() -> None:
    np.testing.assert_allclose(parse_vector("[1, 2, 3]"), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(total_correction([0.1, 0.2, 0.3], [0.2, 0.0, 0.4]), [0.1, -0.2, 0.1])
    try:
        parse_vector("[1, 2]")
    except ValueError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("short vectors must be rejected")


def test_recompute_metrics_keeps_normal_physical_denominators_na() -> None:
    rows = [
        {
            "kind": "normal", "steps": 4, "stable_arrival": "1", "task_success": "1",
            "physical_bypass": "", "collision": "0", "timeout_failure": "0",
            "time_limit_reached_raw": "0", "rollout_window_cut": "0",
            "raw_action_danger_steps": 0, "raw_action_physical_denominator": 0,
            "safety_projection_steps": 1, "boundary_projection_steps": 0,
        },
        {
            "kind": "single", "steps": 4, "stable_arrival": "0", "task_success": "0",
            "physical_bypass": "1", "collision": "0", "timeout_failure": "1",
            "time_limit_reached_raw": "1", "rollout_window_cut": "0",
            "raw_action_danger_steps": 2, "raw_action_physical_denominator": 4,
            "safety_projection_steps": 0, "boundary_projection_steps": 0,
        },
    ]
    metrics = recompute_episode_metrics(rows)
    assert metrics["all"]["physical_bypass_rate"] == 1.0
    assert metrics["all"]["raw_action_danger_rate"] == 0.5
    assert metrics["normal"]["physical_bypass_rate"] is None
    assert metrics["normal"]["raw_action_danger_rate"] is None


def test_audit_reports_intentional_budget_mismatch_without_writing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    data_root.mkdir()
    log_root.mkdir()
    (data_root / "config.lock.json").write_text(json.dumps({"smoke_specification": {"calibration_steps_per_arm": 287}}), encoding="utf-8")
    report = audit_smoke_artifacts(data_root, log_root)
    assert report["passed"] is False
    assert report["failure_count"] >= 1
    assert any(item["check"] == "artifacts_available" or item["check"] == "curriculum_manifest" for item in report["failures"])
    assert not (log_root / "smoke_audit.json").exists()


def test_audit_rejects_mutated_train_validation_scene_seed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    data_root.mkdir()
    log_root.mkdir()
    curriculum = build_smoke_curriculum(2026)
    curriculum["validation_pool"][0]["noise_seed"] = curriculum["train_pool"][0]["noise_seed"]
    (data_root / "curriculum.json").write_text(json.dumps(curriculum), encoding="utf-8")
    report = audit_smoke_artifacts(data_root, log_root)
    assert report["passed"] is False
    details = [item["details"] for item in report["failures"] if item["check"] == "curriculum_manifest"]
    assert details and "noise_seed" in details[0]


def test_actor_contract_rejects_nonfinite_sensitivity(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    data_root.mkdir()
    log_root.mkdir()
    feature_map = {
        "estimated_velocity": [3, 6],
        "goal_delta": [30, 33],
        "obstacle_body_direction": [39, 42],
    }
    def scene() -> dict[str, object]:
        return {
            "actual_actor_node_feature_dim": 42,
            "normalized_actor_node_feature_dim": 42,
            "normalization_status": "fitted",
            "feature_index_map": feature_map,
            "normalized_feature_sensitivity": {
                name: {"context_change_l2": 1.0, "pre_tanh_mean_change_l2": 1.0}
                for name in ("estimated_velocity", "goal_delta", "obstacle_body_direction")
            },
        }
    payload = {
        "environment_steps": 0,
        "training_or_optimizer_steps": 0,
        "real_encoder_input": {"actor_encoder_input_shape": [1, 42]},
        "scenes": [scene(), scene(), scene()],
    }
    payload["scenes"][1]["normalized_feature_sensitivity"]["goal_delta"]["context_change_l2"] = float("nan")
    (log_root / "A").mkdir()
    (log_root / "A" / "actor_observation_contract.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="not finite and positive"):
        _check_actor_contract(_Audit(data_root, log_root), "A")
