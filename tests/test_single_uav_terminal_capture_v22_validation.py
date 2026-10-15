"""Pure/mock contracts for the single-UAV terminal-capture v2.2 evaluator."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import pytest

from src.experiments import single_uav_goal_recovery_correction_evaluation as module
from src.experiments.formal import FormalExperimentConfig


def _episode(index: int = 0) -> dict[str, object]:
    return {
        "episode_index": index,
        "episode_id": f"episode_{index:02d}",
        "scenario_seed": 100 + index,
        "noise_seed": 200 + index,
        "scenario_hash": f"scenario-{index}",
    }


def test_v22_config_defaults_off_and_radii_are_validated():
    config = FormalExperimentConfig(output_dir=Path("C:/tmp/v22"))
    assert config.goal_recovery_terminal_capture_v22_enabled is False
    assert config.goal_recovery_terminal_hold_enter_radius_m == pytest.approx(0.98)
    assert config.goal_recovery_terminal_hold_release_radius_m == pytest.approx(1.05)
    assert config.goal_recovery_terminal_hold_restorative_enabled is True
    with pytest.raises(ValueError):
        FormalExperimentConfig(
            output_dir=Path("C:/tmp/v22-invalid"),
            goal_recovery_terminal_hold_enter_radius_m=1.01,
        )


def test_v22_cli_locks_on_formal_and_fresh_roots():
    parser = module.build_parser()
    assert module.FORMAL_EPISODES == 20
    assert module.HORIZON_BASELINE_MAX_STEPS == 100
    assert parser.parse_args([]).terminal_capture_v22_validation is False
    argv = ["--terminal-capture-v22-validation"]
    resolved = module._resolve_cli_mode(parser.parse_args(argv), argv)
    assert resolved == {
        "condition": "on",
        "formal": True,
        "data_root": module.TERMINAL_CAPTURE_V22_DATA_ROOT,
        "logs_root": module.TERMINAL_CAPTURE_V22_LOG_ROOT,
        "plots_root": module.TERMINAL_CAPTURE_V22_PLOT_ROOT,
    }
    for suffix in (("--smoke",), ("--paired",), ("--condition", "off"),
                   ("--terminal-capture-v21-validation",)):
        args = ["--terminal-capture-v22-validation", *suffix]
        with pytest.raises(ValueError):
            module._resolve_cli_mode(parser.parse_args(args), args)


def test_v22_episode_summary_is_pure_and_uses_canonical_post_step_fields():
    rows = [
        {
            "step": 0, "terminal_v22_hold_latched_before": 0,
            "terminal_v22_hold_latched": 1, "terminal_v22_hold_enter_event": 1,
            "terminal_v22_hold_release_event": 0,
            "terminal_v22_restorative_active": 1,
            "terminal_v22_restorative_desired_radial_speed_mps": 0.25,
            "terminal_v22_inside_canonical_goal": 1, "goal_distance_m": 0.97,
            "post_step_actual_speed_mps": 0.31,
            "stable_hold_counter": 1, "goal_distance_ok": 1, "goal_speed_ok": 1,
        },
        {
            "step": 1, "terminal_v22_hold_latched_before": 1,
            "terminal_v22_hold_latched": 1, "terminal_v22_hold_enter_event": 0,
            "terminal_v22_hold_release_event": 0,
            "terminal_v22_restorative_active": 0,
            "terminal_v22_restorative_desired_radial_speed_mps": 0.0,
            "terminal_v22_inside_canonical_goal": 0, "goal_distance_m": 1.06,
            "post_step_actual_speed_mps": 0.4,
            "stable_hold_counter": 0, "goal_distance_ok": 0, "goal_speed_ok": 1,
        },
        {
            "step": 2, "terminal_v22_hold_latched_before": 1,
            "terminal_v22_hold_latched": 0, "terminal_v22_hold_enter_event": 0,
            "terminal_v22_hold_release_event": 1,
            "terminal_v22_restorative_active": 0,
            "terminal_v22_restorative_desired_radial_speed_mps": 0.0,
            "terminal_v22_inside_canonical_goal": 1, "goal_distance_m": 0.99,
            "post_step_actual_speed_mps": 0.2,
            "stable_hold_counter": 1, "goal_distance_ok": 1, "goal_speed_ok": 1,
        },
    ]
    summary = module._aggregate_v22_episode_fields(rows)
    assert summary["v22_hold_entered"] == 1
    assert summary["first_hold_enter_step"] == 0
    assert summary["goal_distance_at_hold_enter"] == pytest.approx(0.97)
    assert summary["actual_speed_at_hold_enter"] == pytest.approx(0.31)
    assert summary["hold_release_count"] == 1
    assert summary["hold_reentry_count"] == 0
    assert summary["restorative_active_steps"] == 1
    assert summary["mean_restorative_desired_radial_speed_mps"] == pytest.approx(0.25)
    assert summary["mean_restorative_desired_radial_speed"] == pytest.approx(0.25)
    assert summary["max_restorative_desired_radial_speed"] == pytest.approx(0.25)
    assert summary["maximum_distance_while_hold_latched"] == pytest.approx(1.06)
    assert summary["inside_outside_crossings_after_hold"] == 1
    assert summary["stable_counter_reset_count"] == 1
    assert summary["stable_distance_reset_count"] == 1
    assert summary["stable_speed_reset_count"] == 0


def test_v22_evaluation_aggregate_distinguishes_global_and_mean_episode_maxima():
    summaries = [
        {
            "mean_restorative_desired_radial_speed": 0.20,
            "max_restorative_desired_radial_speed": 1.0,
            "maximum_distance_while_hold_latched": 2.0,
        },
        {
            "mean_restorative_desired_radial_speed": 0.40,
            "max_restorative_desired_radial_speed": 3.0,
            "maximum_distance_while_hold_latched": 4.0,
        },
    ]

    aggregate = module._aggregate_episode_summaries(summaries)

    assert aggregate["mean_restorative_desired_radial_speed"] == pytest.approx(0.30)
    assert aggregate["mean_restorative_desired_radial_speed_mps"] == pytest.approx(0.30)
    assert aggregate["global_max_restorative_desired_radial_speed"] == pytest.approx(3.0)
    assert aggregate["global_max_restorative_desired_radial_speed_mps"] == pytest.approx(3.0)
    assert aggregate["mean_episode_max_restorative_desired_radial_speed"] == pytest.approx(2.0)
    assert aggregate["mean_episode_max_restorative_desired_radial_speed_mps"] == pytest.approx(2.0)
    assert aggregate["global_maximum_distance_while_hold_latched"] == pytest.approx(4.0)
    assert aggregate["mean_episode_maximum_distance_while_hold_latched"] == pytest.approx(3.0)

    # These names previously hid a mean-of-maxima reduction and must not be
    # emitted by the evaluation-level schema.
    assert "max_restorative_desired_radial_speed" not in aggregate
    assert "max_restorative_desired_radial_speed_mps" not in aggregate
    assert "maximum_distance_while_hold_latched" not in aggregate


def test_v22_evaluation_aggregate_ignores_nan_invalid_and_empty_maxima():
    summaries = [
        {
            "mean_restorative_desired_radial_speed": float("nan"),
            "max_restorative_desired_radial_speed": float("nan"),
            "maximum_distance_while_hold_latched": float("nan"),
        },
        {
            "mean_restorative_desired_radial_speed_mps": 0.50,
            "max_restorative_desired_radial_speed_mps": 2.0,
            "maximum_distance_while_hold_latched": 1.0,
        },
        {
            "mean_restorative_desired_radial_speed": "invalid",
            "max_restorative_desired_radial_speed": None,
            "maximum_distance_while_hold_latched": "invalid",
        },
    ]

    aggregate = module._aggregate_episode_summaries(summaries)
    assert aggregate["mean_restorative_desired_radial_speed"] == pytest.approx(0.50)
    assert aggregate["global_max_restorative_desired_radial_speed"] == pytest.approx(2.0)
    assert aggregate["mean_episode_max_restorative_desired_radial_speed"] == pytest.approx(2.0)
    assert aggregate["global_maximum_distance_while_hold_latched"] == pytest.approx(1.0)
    assert aggregate["mean_episode_maximum_distance_while_hold_latched"] == pytest.approx(1.0)

    empty = module._aggregate_episode_summaries([])
    for field in (
        "mean_restorative_desired_radial_speed",
        "global_max_restorative_desired_radial_speed",
        "mean_episode_max_restorative_desired_radial_speed",
        "global_maximum_distance_while_hold_latched",
        "mean_episode_maximum_distance_while_hold_latched",
    ):
        assert math.isnan(empty[field])


def test_v22_identity_normalizes_only_v22_fields():
    base = FormalExperimentConfig(
        output_dir=Path("C:/tmp/v22-base"), max_steps=100,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
    )
    candidate = replace(
        base,
        output_dir=Path("C:/tmp/v22-candidate"),
        goal_recovery_terminal_capture_v22_enabled=True,
        goal_recovery_terminal_hold_enter_radius_m=0.98,
        goal_recovery_terminal_hold_release_radius_m=1.05,
        goal_recovery_terminal_hold_restorative_enabled=True,
    )
    episodes = [_episode()]
    baseline = {
        "checkpoint": {"resolved_path": "C:/tmp/frozen_actor.pt", "sha256": "sha", "policy_hash": "policy"},
        "scenario_manifest_hash": "manifest",
        "results": {"on": {"episodes": episodes}},
    }
    baseline_summary = {"candidate_config": module._effective_correction_config(base, enabled=True),
                        "stable_distance_threshold": 1.0, "stable_speed_threshold": 0.5,
                        "stable_hold_steps": 5}
    checks = module._v22_only_identity_checks(
        baseline_evaluation=baseline, baseline_summary=baseline_summary,
        baseline_config=base, candidate_config=candidate,
        checkpoint_path=Path("C:/tmp/frozen_actor.pt"), checkpoint_sha256="sha",
        policy_hash="policy", scenario_manifest_hash="manifest",
        current_episodes=episodes,
    )
    assert checks["only_v22_fields_changed"] is True
    assert checks["same_reward"] is True
    assert checks["same_dynamics"] is True
    assert checks["same_action_transform"] is True
    assert checks["same_stable_criterion"] is True


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("single_uav_goal_hold_reward", 1.0),
        ("process_position_std_m", 0.09),
        ("dynamic_obstacle_motion_features_enabled", True),
    ),
)
def test_v22_identity_fails_fast_on_reward_dynamics_or_obstacle_change(
    field_name, changed_value
):
    base = FormalExperimentConfig(
        output_dir=Path("C:/tmp/v22-base"), max_steps=100,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
    )
    candidate = replace(
        base,
        output_dir=Path("C:/tmp/v22-candidate"),
        goal_recovery_terminal_capture_v22_enabled=True,
        **{field_name: changed_value},
    )
    episodes = [_episode()]
    baseline = {
        "checkpoint": {
            "resolved_path": "C:/tmp/frozen_actor.pt",
            "sha256": "sha",
            "policy_hash": "policy",
        },
        "scenario_manifest_hash": "manifest",
        "results": {"on": {"episodes": episodes}},
    }
    baseline_summary = {
        "candidate_config": module._effective_correction_config(base, enabled=True),
        "stable_distance_threshold": 1.0,
        "stable_speed_threshold": 0.5,
        "stable_hold_steps": 5,
    }
    with pytest.raises(ValueError, match="v2.2-only identity mismatch"):
        module._v22_only_identity_checks(
            baseline_evaluation=baseline,
            baseline_summary=baseline_summary,
            baseline_config=base,
            candidate_config=candidate,
            checkpoint_path=Path("C:/tmp/frozen_actor.pt"),
            checkpoint_sha256="sha",
            policy_hash="policy",
            scenario_manifest_hash="manifest",
            current_episodes=episodes,
        )


def test_v22_source_contains_reset_and_required_logging_artifacts():
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset"
        for node in ast.walk(tree)
    )
    source = source_path.read_text(encoding="utf-8")
    for field in (
        "terminal_v22_enabled", "terminal_v22_hold_latched_before",
        "terminal_v22_hold_latched", "terminal_v22_hold_enter_event",
        "terminal_v22_hold_release_event", "terminal_v22_hold_mode",
        "terminal_v22_restorative_active", "terminal_v22_distance_outside_stable_m",
        "terminal_v22_restore_fraction", "terminal_v22_restorative_desired_radial_speed_mps",
        "terminal_v22_restorative_radial_error_mps", "terminal_v22_restorative_delta_v_mps",
        "terminal_v22_inside_canonical_goal",
    ):
        assert f'"{field}"' in source
    assert "terminal_capture_v22_validation_summary.json" in source
    assert "terminal_capture_v22_validation_episodes.csv" in source
