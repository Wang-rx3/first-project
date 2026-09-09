"""Contracts for the ordered report-direct and delta-scale protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_core_framework_v8_templates_apply_the_two_stage_contract(tmp_path: Path) -> None:
    from src.experiments.mainline_baseline_comparison import (
        MainlineBaselineComparisonConfig,
    )
    from src.experiments.mainline_corrected_stages import _template

    reference = MainlineBaselineComparisonConfig(
        protocol="core-framework-v8",
        stage="smoke",
        experiment_arm="reference",
        output_dir=tmp_path / "reference-data",
        logs_dir=tmp_path / "reference-logs",
        plots_dir=tmp_path / "reference-plots",
    )
    direct = _template(reference, "baseline_mappo", tmp_path / "reference-data" / "seed_42")
    assert direct.learning_reward_mode == "report_direct"
    assert direct.learning_feedback_mode == "v3"
    assert direct.episode_objective == "finite_horizon"
    assert direct.include_remaining_time is True
    assert direct.bootstrap_time_limits is False
    assert direct.rollout_episodes_per_update == 4
    assert direct.ppo_action_logprob_mode == "sampled"
    assert direct.learning_potential_coef == 0.0

    progress = MainlineBaselineComparisonConfig(
        protocol="core-framework-v8",
        stage="smoke",
        experiment_arm="loc-progress-v2",
        output_dir=tmp_path / "progress-data",
        logs_dir=tmp_path / "progress-logs",
        plots_dir=tmp_path / "progress-plots",
    )
    shaped = _template(progress, "baseline_mappo", tmp_path / "progress-data" / "seed_42")
    assert shaped.learning_reward_mode == "report_plus_feedback"
    assert shaped.learning_feedback_mode == "loc_progress_v2"
    assert shaped.loc_progress_coef == 0.1
    assert shaped.learning_potential_coef == 0.0
    assert shaped.episode_objective == "finite_horizon"


def test_report_plus_feedback_keeps_the_physical_collision_term_unclipped() -> None:
    from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig

    statistics = {
        "u_center": 0.0,
        "u_scale": 1.0,
        "c_center": 0.0,
        "c_scale": 1.0,
        "delta_u_scale": 0.5,
    }
    env = MultiUAVJointEnv(
        SimulationConfig(
            max_steps=1,
            gnss_mode="denied",
            reward=RewardConfig(
                learning_reward_mode="report_plus_feedback",
                learning_feedback_mode="loc_progress_v2",
                potential_statistics=statistics,
                learning_potential_coef=0.0,
                loc_progress_coef=0.1,
                learning_reward_clip=1.0,
            ),
        )
    )
    env.reset(seed=42)
    env._previous_position_logdet = env._position_logdet_uncertainty() + 2.0
    actions = {
        uav_id: {
            "measurement_uav_id": "none",
            "bandwidth": {},
            "delta_v": [0.0, 0.0, 0.0],
        }
        for uav_id in env.uav_ids
    }
    execution = {
        uav_id: {"delta_v_applied": [0.0, 0.0, 0.0]}
        for uav_id in env.uav_ids
    }
    reporting, info = env._rewards(1, actions, execution)
    components = info["learning_reward_components"]
    report_value = float(next(iter(reporting.values())))
    learning_value = float(next(iter(info["learning_rewards"].values())))

    assert np.isfinite(learning_value)
    assert learning_value > report_value
    assert float(components["collision_cost_learning"]) == np.float64(10.0)
    assert np.isclose(
        learning_value - report_value,
        float(components["loc_progress_reward"]),
    )

