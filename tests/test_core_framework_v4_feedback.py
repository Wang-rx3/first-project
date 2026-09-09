"""Regression tests for the v4 training-only feedback channels."""

from __future__ import annotations

import json

import numpy as np


def _zero_actions(environment):
    return {
        uav_id: {
            "measurement_uav_id": "none",
            "bandwidth": {},
            "delta_v": [0.0, 0.0, 0.0],
        }
        for uav_id in environment.uav_ids
    }


def _zero_execution(environment):
    return {
        uav_id: {"delta_v_applied": [0.0, 0.0, 0.0]}
        for uav_id in environment.uav_ids
    }


def _progress_statistics():
    return {"u_center": 0.0, "u_scale": 1.0, "c_center": 0.0, "c_scale": 1.0}


def _progress_statistics_v2(delta_u_scale=0.5):
    return {
        **_progress_statistics(),
        "delta_u_scale": float(delta_u_scale),
        "delta_u_median": 0.0,
        "delta_u_mad": 0.337,
        "delta_u_samples": 10,
        "delta_u_scale_floor_applied": False,
        "delta_u_scale_source": "warmup_adjacent_window_mad",
        "delta_u_definition": "u_previous_minus_u_current_same_episode",
    }


def test_v4_location_progress_has_the_expected_sign_and_preserves_reporting_reward():
    from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig

    environment = MultiUAVJointEnv(
        SimulationConfig(
            max_steps=1,
            gnss_mode="denied",
            reward=RewardConfig(
                learning_reward_mode="progress_potential_v2",
                potential_statistics=_progress_statistics(),
                learning_feedback_mode="loc_progress_v1",
                loc_progress_coef=0.1,
            ),
        )
    )
    environment.reset(seed=42)
    environment._previous_position_logdet = environment._position_logdet_uncertainty() + 1.0
    actions = _zero_actions(environment)
    reporting, info = environment._rewards(0, actions, _zero_execution(environment))

    components = info["learning_reward_components"]
    assert float(components["loc_progress_delta"]) > 0.0
    assert float(components["loc_progress_reward"]) > 0.0
    assert components["actual_measurement_count"] == 0
    assert components["actual_measurement_bytes"] == 0
    assert info["reporting_rewards"] == reporting
    assert all(np.isfinite(float(value)) for value in info["learning_rewards"].values())


def test_v4_location_progress_v2_uses_delta_scale_and_clips_feedback():
    from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig

    environment = MultiUAVJointEnv(
        SimulationConfig(
            max_steps=1,
            gnss_mode="denied",
            reward=RewardConfig(
                learning_reward_mode="progress_potential_v2",
                potential_statistics=_progress_statistics_v2(delta_u_scale=0.5),
                learning_feedback_mode="loc_progress_v2",
                loc_progress_coef=0.1,
            ),
        )
    )
    environment.reset(seed=42)
    current = environment._position_logdet_uncertainty()
    environment._previous_position_logdet = current + 2.0
    reporting, info = environment._rewards(
        0, _zero_actions(environment), _zero_execution(environment)
    )

    components = info["learning_reward_components"]
    assert np.isclose(float(components["loc_progress_delta_raw"]), 4.0)
    assert np.isclose(float(components["loc_progress_delta"]), 3.0)
    assert np.isclose(float(components["loc_progress_scale"]), 0.5)
    assert components["loc_progress_scale_source"] == "warmup_adjacent_window_mad"
    assert np.isclose(float(components["loc_progress_reward"]), 0.3)
    assert info["reporting_rewards"] == reporting


def test_v4_location_progress_v2_rejects_missing_delta_scale():
    import pytest

    from src.simulation.environment import RewardConfig

    with pytest.raises(ValueError, match="delta_u_scale"):
        RewardConfig(
            learning_reward_mode="progress_potential_v2",
            learning_feedback_mode="loc_progress_v2",
            potential_statistics=_progress_statistics(),
        )


def test_v4_location_progress_v2_scale_uses_warmup_adjacent_mad_and_not_episode_boundary():
    from src.experiments.formal import (
        _fit_adjacent_uncertainty_delta_statistics,
        _within_episode_uncertainty_deltas,
    )

    # The two episodes are intentionally passed separately.  The helper only
    # creates within-episode pairs, so no artificial boundary delta is fitted.
    first = _within_episode_uncertainty_deltas([3.0, 2.0, 1.0])
    second = _within_episode_uncertainty_deltas([10.0, 9.0])
    assert first == [1.0, 1.0]
    assert second == [1.0]
    assert _fit_adjacent_uncertainty_delta_statistics(first + second) == {
        "delta_u_scale": 1.0e-3,
        "delta_u_median": 1.0,
        "delta_u_mad": 0.0,
        "delta_u_samples": 3,
        "delta_u_scale_floor_applied": True,
        "delta_u_scale_source": "warmup_adjacent_window_mad",
        "delta_u_definition": "u_previous_minus_u_current_same_episode",
    }


def test_v4_actual_load_counts_only_non_none_measurements_and_closes_512_bytes():
    from src.simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig

    environment = MultiUAVJointEnv(
        SimulationConfig(
            max_steps=1,
            gnss_mode="denied",
            reward=RewardConfig(
                learning_feedback_mode="actual_load_v1",
                actual_measurement_cost_enabled=True,
                actual_measurement_cost_target=0.0,
                actual_measurement_cost_coef=0.1,
            ),
        )
    )
    environment.reset(seed=42)
    actions = _zero_actions(environment)
    actions["uav_00"]["measurement_uav_id"] = "uav_01"
    _reporting, info = environment._rewards(0, actions, _zero_execution(environment))

    components = info["learning_reward_components"]
    assert components["actual_measurement_count"] == 1
    assert components["actual_measurement_bytes"] == 512
    assert np.isclose(float(components["actual_measurement_rate"]), 1.0 / 3.0)
    assert float(components["actual_measurement_load_excess"]) > 0.0
    assert float(components["actual_measurement_load_reward"]) < 0.0


def test_v4_protocol_uses_proposed_method_and_three_rule_references(tmp_path):
    from src.experiments.mainline_baseline_comparison import MainlineBaselineComparisonConfig
    from src.experiments.mainline_corrected_stages import _learners_for, _methods_for, _template

    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v4",
        stage="smoke",
        experiment_arm="loc-progress",
        output_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        plots_dir=tmp_path / "plots",
    )
    assert _methods_for(config) == (
        "local_only_fixed",
        "random_cooperative",
        "budgeted_greedy_ig",
        "baseline_mappo",
    )
    assert _learners_for(config) == ("baseline_mappo",)
    formal = _template(config, "baseline_mappo", tmp_path / "data" / "seed_42")
    assert formal.learning_feedback_mode == "loc_progress_v1"
    assert formal.loc_progress_coef == 0.1
    assert formal.actual_measurement_cost_enabled is False
    assert formal.ppo_action_logprob_mode == "sampled"


def test_v4_delta_scale_protocol_requires_v2_feedback_and_uses_new_root(tmp_path):
    from src.experiments.mainline_baseline_comparison import MainlineBaselineComparisonConfig
    from src.experiments.mainline_corrected_stages import _roots, _template

    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v4-delta-scale",
        stage="smoke",
        experiment_arm="loc-progress-v2",
        output_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        plots_dir=tmp_path / "plots",
    )
    assert _roots(config)[0].name == "mainline_core_framework_v4_delta_scale_seed42"
    formal = _template(config, "baseline_mappo", tmp_path / "data" / "seed_42")
    assert formal.learning_feedback_mode == "loc_progress_v2"
    assert formal.loc_progress_coef == 0.1


def test_v4_feedback_configuration_rejects_missing_actual_load_target():
    import pytest

    from src.simulation.environment import RewardConfig

    with pytest.raises(ValueError, match="actual_measurement_cost_target"):
        RewardConfig(
            learning_feedback_mode="actual_load_v1",
            actual_measurement_cost_enabled=True,
        )


def test_v4_actual_load_target_uses_per_uav_window_units(tmp_path, monkeypatch):
    """The fitted target must use the same units as environment ``b_t``."""

    from src.experiments import mainline_corrected_stages as stages

    scale = tmp_path / "seed_42" / "baseline_mappo" / "scale_3"
    train_episode = scale / "train" / "episode_000001"
    train_episode.mkdir(parents=True)
    (scale / "manifest.json").write_text(
        json.dumps(
            {"episodes": [{"split": "train", "episode_id": "episode_000001"}]}
        ),
        encoding="utf-8",
    )
    # Six rows = two windows for three UAVs.  Three selected measurements
    # therefore give 1.5 total measurements/window, or 0.5/UAV/window.
    stages._write_table(
        train_episode / "actions.csv",
        [
            {"measurement_uav_id": "uav_01"},
            {"measurement_uav_id": "none"},
            {"measurement_uav_id": "none"},
            {"measurement_uav_id": "uav_02"},
            {"measurement_uav_id": "uav_00"},
            {"measurement_uav_id": "none"},
        ],
    )
    monkeypatch.setattr(stages, "V3_REFERENCE_ROOT", tmp_path)

    fitted = stages._fit_v4_actual_measurement_target()

    assert np.isclose(float(fitted["target"]), 0.5)
    assert fitted["target_definition"] == (
        "mean_episode_measurement_rate_per_uav_window"
    )
