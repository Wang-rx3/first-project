from __future__ import annotations

from pathlib import Path

from src.experiments.mainline_baseline_comparison import MainlineBaselineComparisonConfig
from src.experiments.mainline_corrected_stages import _template


def test_core_framework_v7_encodes_the_seven_stabilization_contract(tmp_path: Path) -> None:
    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v7",
        stage="smoke",
        output_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        plots_dir=tmp_path / "plots",
        figure_profile="compact-proposed-v4",
    )
    formal = _template(config, "baseline_mappo", tmp_path / "data" / "seed_42")

    assert formal.rollout_episodes_per_update == 8
    assert formal.episode_objective == "finite_horizon"
    assert formal.include_remaining_time is True
    assert formal.bootstrap_time_limits is False
    assert formal.actor_learning_rate_scale == 0.5
    assert formal.learning_reward_mode == "bounded_potential"
    assert formal.learning_collision_penalty_cap == 1.0
    assert formal.learning_potential_coef == 0.0
    assert formal.ppo_action_logprob_mode == "sampled"
    assert formal.gradient_clip_scope == "actor_critic"
    assert formal.measurement_head_mode == "action_conditioned_v2"
