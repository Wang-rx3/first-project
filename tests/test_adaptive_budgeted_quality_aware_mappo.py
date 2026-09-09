from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.adaptive_budgeted_quality_aware_mappo import (
    ADAPTIVE_BRANCH,
    BRANCHES,
    AdaptiveMeasurementRateMAPPOConfig,
    _formal_config,
)


def test_adaptive_experiment_has_three_expected_branches_and_fixed_defaults(tmp_path: Path) -> None:
    config = AdaptiveMeasurementRateMAPPOConfig(output_dir=tmp_path / "run")
    assert BRANCHES == (
        "baseline_mappo",
        "budgeted_quality_aware_mappo",
        ADAPTIVE_BRANCH,
    )
    assert config.measurement_overhead_enabled is True
    assert config.measurement_rate_dual_initial == 0.0
    assert config.measurement_rate_dual_step_size == 0.25
    assert config.measurement_rate_dual_max == 5.0


def test_adaptive_formal_config_enables_only_adaptive_dual_control(tmp_path: Path) -> None:
    config = AdaptiveMeasurementRateMAPPOConfig(output_dir=tmp_path / "run")
    ekf = {
        "vo_position_std_m": 0.03,
        "process_position_std_m": 0.08,
        "process_velocity_std_mps": 0.15,
        "nlos_variance_multiplier": 2.0,
    }
    adaptive = _formal_config(
        config,
        seed=2026,
        output_dir=tmp_path / "adaptive",
        ekf_parameters=ekf,
        treatment=ADAPTIVE_BRANCH,
        budgets={"uav_00": 1, "uav_01": 1, "uav_02": 1},
        targets={"uav_00": 0.1, "uav_01": 0.2, "uav_02": 0.3},
    )
    assert adaptive.mappo_variant == ADAPTIVE_BRANCH
    assert adaptive.measurement_budget_enabled is True
    assert adaptive.measurement_rate_constraint_enabled is True
    assert adaptive.measurement_rate_dual_enabled is True
    assert adaptive.measurement_rate_constraint_coef == 0.0
    assert adaptive.measurement_rate_target_by_uav == {
        "uav_00": 0.1,
        "uav_01": 0.2,
        "uav_02": 0.3,
    }


def test_adaptive_config_rejects_invalid_multiplier_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="initial"):
        AdaptiveMeasurementRateMAPPOConfig(
            output_dir=tmp_path / "run",
            measurement_rate_dual_initial=6.0,
            measurement_rate_dual_max=5.0,
        )
    with pytest.raises(ValueError, match="step"):
        AdaptiveMeasurementRateMAPPOConfig(
            output_dir=tmp_path / "run",
            measurement_rate_dual_step_size=0.0,
        )
