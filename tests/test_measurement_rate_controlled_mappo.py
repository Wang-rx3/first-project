from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.experiments.measurement_rate_controlled_mappo import (
    _fit_measurement_rate_targets,
)
from src.learning.mappo import MAPPOConfig, MAPPOTrainer
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig


def _write_actions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uav_id", "measurement_uav_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_rate_targets_use_baseline_train_actions_only(tmp_path: Path) -> None:
    baseline_scale = tmp_path / "baseline" / "scale_3"
    _write_actions(
        baseline_scale / "train" / "episode_000001" / "actions.csv",
        [
            {"uav_id": "uav_00", "measurement_uav_id": "uav_01"},
            {"uav_id": "uav_00", "measurement_uav_id": "none"},
            {"uav_id": "uav_01", "measurement_uav_id": "uav_00"},
            {"uav_id": "uav_01", "measurement_uav_id": "none"},
            {"uav_id": "uav_02", "measurement_uav_id": "none"},
        ],
    )
    targets, rows = _fit_measurement_rate_targets(baseline_scale, uav_count=3)
    assert targets == {"uav_00": 0.5, "uav_01": 0.5, "uav_02": 0.0}
    assert len(rows) == 3
    assert {row["source_split"] for row in rows} == {"train"}
    assert {row["source_branch"] for row in rows} == {"baseline_mappo"}


def test_rate_constraint_requires_fitted_targets() -> None:
    with pytest.raises(ValueError, match="measurement_rate_target_by_uav"):
        MAPPOConfig(measurement_rate_constraint_enabled=True)
    with pytest.raises(ValueError, match="measurement_rate_tolerance"):
        MAPPOConfig(measurement_rate_tolerance=1.1)


def test_rate_constraint_training_signal_is_finite() -> None:
    environment = MultiUAVJointEnv(SimulationConfig(max_steps=2, gnss_mode="denied"))
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=2026,
        gat_hidden_dim=8,
        gat_output_dim=8,
        gat_heads=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        max_obstacles=4,
        policy_epochs=1,
        minibatch_size=4,
        measurement_rate_constraint_enabled=True,
        measurement_rate_constraint_coef=0.5,
        measurement_rate_tolerance=0.05,
        measurement_rate_target_by_uav={
            "uav_00": 0.5,
            "uav_01": 0.5,
            "uav_02": 0.5,
        },
    )
    observations, info = environment.reset(seed=2026)
    decision = trainer.act(observations, info["central_state"])
    next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
    trainer.add_transition(
        decision,
        rewards,
        next_info["central_state"],
        done=bool(terminated or truncated),
    )
    stats = trainer.update()
    assert stats["measurement_rate_constraint_enabled"] == 1.0
    assert "measurement_rate_constraint_weighted_loss" in stats
    assert "measurement_rate_expected_mean" in stats
    for uav_id in ("uav_00", "uav_01", "uav_02"):
        assert f"measurement_rate_target_{uav_id}" in stats
        assert f"measurement_rate_expected_{uav_id}" in stats
        assert f"measurement_rate_upper_residual_{uav_id}" in stats
        assert f"measurement_rate_lower_residual_{uav_id}" in stats
    assert np.isfinite(list(stats.values())).all()


def test_rate_constraint_disabled_keeps_legacy_path() -> None:
    config = MAPPOConfig()
    assert config.measurement_rate_constraint_enabled is False
    assert config.measurement_rate_target_by_uav is None
    assert config.measurement_rate_dual_enabled is False


def test_adaptive_rate_dual_updates_per_uav_and_stays_bounded() -> None:
    environment = MultiUAVJointEnv(SimulationConfig(max_steps=2, gnss_mode="denied"))
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=2026,
        gat_hidden_dim=8,
        gat_output_dim=8,
        gat_heads=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        max_obstacles=4,
        policy_epochs=1,
        minibatch_size=16,
        measurement_rate_constraint_enabled=True,
        measurement_rate_constraint_coef=0.0,
        measurement_rate_tolerance=0.0,
        measurement_rate_target_by_uav={
            "uav_00": 0.0,
            "uav_01": 0.0,
            "uav_02": 0.0,
        },
        measurement_rate_dual_enabled=True,
        measurement_rate_dual_initial=0.0,
        measurement_rate_dual_step_size=0.25,
        measurement_rate_dual_max=5.0,
    )
    observations, info = environment.reset(seed=2026)
    decision = trainer.act(observations, info["central_state"])
    _next_observations, rewards, terminated, truncated, next_info = environment.step(decision.actions)
    trainer.add_transition(
        decision,
        rewards,
        next_info["central_state"],
        done=bool(terminated or truncated),
    )
    stats = trainer.update()
    assert stats["measurement_rate_dual_enabled"] == 1.0
    assert np.isfinite(stats["measurement_rate_dual_loss"])
    violations = [
        float(stats[f"measurement_rate_violation_{uav_id}"])
        for uav_id in ("uav_00", "uav_01", "uav_02")
    ]
    duals = [
        float(trainer.measurement_rate_dual_state_dict()[uav_id])
        for uav_id in ("uav_00", "uav_01", "uav_02")
    ]
    assert any(value > 0.0 for value in violations)
    assert any(value > 0.0 for value in duals)
    assert all(0.0 <= value <= 5.0 for value in duals)
    for uav_id in ("uav_00", "uav_01", "uav_02"):
        assert f"measurement_rate_dual_{uav_id}" in stats
        assert f"measurement_rate_dual_update_{uav_id}" in stats


def test_adaptive_rate_dual_configuration_is_validated_and_restorable() -> None:
    target_map = {"uav_00": 0.0, "uav_01": 0.0, "uav_02": 0.0}
    with pytest.raises(ValueError, match="dual_enabled"):
        MAPPOConfig(
            measurement_rate_dual_enabled=True,
            measurement_rate_target_by_uav=target_map,
        )
    with pytest.raises(ValueError, match="must not exceed"):
        MAPPOConfig(
            measurement_rate_constraint_enabled=True,
            measurement_rate_target_by_uav=target_map,
            measurement_rate_dual_enabled=True,
            measurement_rate_dual_initial=2.0,
            measurement_rate_dual_max=1.0,
        )

    environment = MultiUAVJointEnv(SimulationConfig(max_steps=1, gnss_mode="denied"))
    trainer = MAPPOTrainer.from_environment(
        environment,
        measurement_rate_constraint_enabled=True,
        measurement_rate_target_by_uav=target_map,
        measurement_rate_dual_enabled=True,
        measurement_rate_dual_initial=0.75,
        measurement_rate_dual_step_size=0.25,
        measurement_rate_dual_max=5.0,
    )
    state = trainer.measurement_rate_dual_state_dict()
    trainer.measurement_rate_dual_by_uav = {key: 0.0 for key in state}
    trainer.load_measurement_rate_dual_state_dict(state)
    assert trainer.measurement_rate_dual_state_dict() == state
