from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.proxy_guided_adaptive_budgeted_quality_aware_mappo import (
    ADAPTIVE_BRANCH,
    BASELINE_BRANCH,
    PROXY_BRANCH,
    ProxyGuidedAdaptiveMAPPOConfig,
    _formal_config,
)


def test_proxy_guided_config_uses_fixed_teacher_defaults() -> None:
    config = ProxyGuidedAdaptiveMAPPOConfig(output_dir=Path("tmp-proxy-test"), seeds=(2026,))
    assert config.proxy_teacher_coef == 0.25
    assert config.proxy_temperature == 0.25
    assert config.communication_cost_weight == 1.0
    assert config.measurement_overhead_enabled is True


@pytest.mark.parametrize(
    ("branch", "proxy_enabled"),
    [
        (BASELINE_BRANCH, False),
        (ADAPTIVE_BRANCH, False),
        (PROXY_BRANCH, True),
    ],
)
def test_formal_branch_flags_isolate_proxy_teacher(
    branch: str, proxy_enabled: bool
) -> None:
    config = ProxyGuidedAdaptiveMAPPOConfig(output_dir=Path("tmp-proxy-test"), seeds=(2026,))
    targets = {"uav_00": 0.1, "uav_01": 0.1, "uav_02": 0.1}
    budgets = {"uav_00": 5, "uav_01": 5, "uav_02": 5}
    model = {"source_split": "train", "scales": {"ig_q05": 0.0, "ig_q95": 1.0}}
    formal = _formal_config(
        config,
        seed=2026,
        output_dir=Path("tmp-proxy-test") / branch,
        ekf_parameters={
            "vo_position_std_m": 0.03,
            "process_position_std_m": 0.08,
            "process_velocity_std_mps": 0.15,
            "nlos_variance_multiplier": 2.0,
        },
        treatment=branch,
        budgets=budgets if branch != BASELINE_BRANCH else None,
        targets=targets if branch != BASELINE_BRANCH else None,
        proxy_model=model if proxy_enabled else None,
    )
    assert formal.measurement_proxy_features_enabled is proxy_enabled
    assert formal.measurement_proxy_guidance_enabled is proxy_enabled
    assert formal.measurement_rate_dual_enabled is (branch != BASELINE_BRANCH)
    assert formal.measurement_budget_enabled is (branch != BASELINE_BRANCH)
    if proxy_enabled:
        assert formal.measurement_proxy_model == model
