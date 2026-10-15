"""Schema and identity contracts for the locked v2.3 validation entry point."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import pytest

from src.experiments import single_uav_goal_recovery_correction_evaluation as module
from src.experiments.formal import FormalExperimentConfig


def _row(
    step: int,
    active: bool,
    pre: bool,
    post: bool,
    desired: float,
    distance: float,
    **extra: object,
):
    row = {
        "step": step,
        "terminal_v23_guard_active": active,
        "terminal_v23_preboundary_active": pre,
        "terminal_v23_postboundary_active": post,
        "terminal_v23_desired_radial_speed_mps": desired,
        "goal_distance_m": distance,
    }
    row.update(extra)
    return row


def _episode(index: int = 0) -> dict[str, object]:
    return {
        "episode_index": index,
        "episode_id": f"episode_{index:02d}",
        "scenario_seed": 100 + index,
        "noise_seed": 200 + index,
        "scenario_hash": f"scenario-{index}",
    }


def test_01_formal_defaults_off_and_target():
    config = FormalExperimentConfig(output_dir=Path("C:/tmp/v23"))
    assert config.goal_recovery_terminal_capture_v23_enabled is False
    assert config.goal_recovery_terminal_hold_target_radius_m == pytest.approx(0.96)


def test_02_formal_target_relation_validation():
    with pytest.raises(ValueError):
        FormalExperimentConfig(output_dir=Path("C:/tmp/v23"), goal_recovery_terminal_hold_target_radius_m=0.98)


def test_03_cli_defaults_off():
    args = module.build_parser().parse_args([])
    assert args.terminal_capture_v23_validation is False


def test_04_cli_resolves_fresh_v23_roots():
    parser = module.build_parser()
    resolved = module._resolve_cli_mode(parser.parse_args(["--terminal-capture-v23-validation"]), ["--terminal-capture-v23-validation"])
    assert resolved == {
        "condition": "on",
        "formal": True,
        "data_root": module.TERMINAL_CAPTURE_V23_DATA_ROOT,
        "logs_root": module.TERMINAL_CAPTURE_V23_LOG_ROOT,
        "plots_root": module.TERMINAL_CAPTURE_V23_PLOT_ROOT,
    }


def test_05_cli_rejects_v22_conflict():
    parser = module.build_parser()
    args = ["--terminal-capture-v23-validation", "--terminal-capture-v22-validation"]
    with pytest.raises(ValueError):
        module._resolve_cli_mode(parser.parse_args(args), args)


def test_06_cli_rejects_override():
    parser = module.build_parser()
    args = ["--terminal-capture-v23-validation", "--terminal-capture-radius", "7"]
    with pytest.raises(ValueError):
        module._resolve_cli_mode(parser.parse_args(args), args)


def test_07_aggregate_counts_and_mean():
    summary = module._aggregate_v23_episode_fields([
        _row(0, True, True, False, 0.1, 0.97),
        _row(1, True, False, True, 0.2, 1.02),
        _row(2, False, False, False, 0.0, 1.05),
    ])
    assert summary["v23_guard_active_steps"] == 2
    assert summary["v23_preboundary_guard_steps"] == 1
    assert summary["v23_postboundary_restore_steps"] == 1
    assert summary["mean_v23_desired_radial_speed"] == pytest.approx(0.15)
    assert summary["global_max_v23_desired_radial_speed"] == pytest.approx(0.2)
    assert summary["first_v23_guard_step"] == 0
    assert summary["distance_at_first_v23_guard"] == pytest.approx(0.97)


def test_08_aggregate_empty_uses_nan_mean_and_neutral_max():
    summary = module._aggregate_v23_episode_fields([])
    assert math.isnan(summary["mean_v23_desired_radial_speed"])
    assert summary["global_max_v23_desired_radial_speed"] == 0.0
    assert summary["first_v23_guard_step"] == ""


def test_09_episode_summary_augmentation_is_pure():
    original = {"episode_id": "e"}
    result = module._augment_v23_episode_summary(original, [_row(2, True, False, True, 0.2, 1.02)])
    assert original == {"episode_id": "e"}
    assert result["v23_guard_active_steps"] == 1


def test_10_evaluation_aggregate_uses_global_max_not_mean_max():
    result = module._aggregate_episode_summaries([
        {"global_max_v23_desired_radial_speed": 0.2, "mean_v23_desired_radial_speed": 0.1},
        {"global_max_v23_desired_radial_speed": 0.8, "mean_v23_desired_radial_speed": 0.3},
    ])
    assert result["global_max_v23_desired_radial_speed"] == pytest.approx(0.8)
    assert result["mean_v23_desired_radial_speed"] == pytest.approx(0.2)


def test_11_evaluation_aggregate_ignores_bad_maxima():
    result = module._aggregate_episode_summaries([
        {"global_max_v23_desired_radial_speed": float("nan"), "mean_v23_desired_radial_speed": float("nan")},
        {"global_max_v23_desired_radial_speed": 0.4, "mean_v23_desired_radial_speed": 0.4},
    ])
    assert result["global_max_v23_desired_radial_speed"] == pytest.approx(0.4)
    assert result["mean_v23_desired_radial_speed"] == pytest.approx(0.4)


def test_12_effective_config_contains_only_new_v23_fields():
    config = FormalExperimentConfig(output_dir=Path("C:/tmp/v23"), goal_recovery_terminal_capture_v23_enabled=True)
    effective = module._effective_correction_config(config, enabled=True)
    assert effective["terminal_capture_v23_enabled"] is True
    assert effective["terminal_hold_target_radius_m"] == pytest.approx(0.96)


def test_13_v23_identity_accepts_only_v23_transition():
    base = FormalExperimentConfig(
        output_dir=Path("C:/tmp/v23-base"),
        max_steps=100,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        goal_recovery_terminal_capture_v22_enabled=True,
    )
    candidate = replace(base, output_dir=Path("C:/tmp/v23-candidate"), goal_recovery_terminal_capture_v23_enabled=True)
    episodes = [_episode()]
    baseline = {
        "checkpoint": {"resolved_path": "C:/tmp/frozen.pt", "sha256": "sha", "policy_hash": "policy"},
        "scenario_manifest_hash": "manifest",
        "results": {"on": {"episodes": episodes}},
    }
    summary = {
        "candidate_config": module._effective_correction_config(base, enabled=True),
        "stable_distance_threshold": 1.0, "stable_speed_threshold": 0.5, "stable_hold_steps": 5,
    }
    checks = module._v23_only_identity_checks(
        baseline_evaluation=baseline, baseline_summary=summary,
        baseline_config=base, candidate_config=candidate,
        checkpoint_path=Path("C:/tmp/frozen.pt"), checkpoint_sha256="sha", policy_hash="policy",
        scenario_manifest_hash="manifest", current_episodes=episodes,
    )
    assert checks["only_v23_fields_changed"]


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("single_uav_goal_hold_reward", 1.0),
        ("process_position_std_m", 0.09),
        ("dynamic_obstacle_motion_features_enabled", True),
    ),
)
def test_14_v23_identity_rejects_non_v23_behavior_change(field_name, changed_value):
    base = FormalExperimentConfig(
        output_dir=Path("C:/tmp/v23-base"),
        max_steps=100,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        goal_recovery_terminal_capture_v22_enabled=True,
    )
    candidate = replace(
        base,
        output_dir=Path("C:/tmp/v23-candidate"),
        goal_recovery_terminal_capture_v23_enabled=True,
        **{field_name: changed_value},
    )
    episodes = [_episode()]
    baseline = {"checkpoint": {"resolved_path": "C:/tmp/frozen.pt", "sha256": "sha", "policy_hash": "policy"}, "scenario_manifest_hash": "manifest", "results": {"on": {"episodes": episodes}}}
    summary = {"candidate_config": module._effective_correction_config(base, enabled=True), "stable_distance_threshold": 1.0, "stable_speed_threshold": 0.5, "stable_hold_steps": 5}
    with pytest.raises(ValueError, match="v2.3-only identity mismatch"):
        module._v23_only_identity_checks(
            baseline_evaluation=baseline, baseline_summary=summary, baseline_config=base, candidate_config=candidate,
            checkpoint_path=Path("C:/tmp/frozen.pt"), checkpoint_sha256="sha", policy_hash="policy", scenario_manifest_hash="manifest", current_episodes=episodes,
        )


def test_15_source_has_required_step_schema_fields():
    source = Path(module.__file__).read_text(encoding="utf-8")
    for field in (
        "terminal_v23_enabled", "terminal_v23_hold_target_radius_m",
        "terminal_v23_guard_active", "terminal_v23_guard_fraction",
        "terminal_v23_desired_radial_speed_mps", "terminal_v23_preboundary_active",
        "terminal_v23_postboundary_active", "terminal_v23_distance_to_target_band_m",
        "terminal_v23_radial_error_mps",
    ):
        assert f'"{field}"' in source


def test_16_source_has_required_episode_schema_fields():
    source = Path(module.__file__).read_text(encoding="utf-8")
    for field in (
        "v23_guard_active_steps", "v23_preboundary_guard_steps",
        "v23_postboundary_restore_steps", "mean_v23_desired_radial_speed",
        "global_max_v23_desired_radial_speed", "first_v23_guard_step",
        "distance_at_first_v23_guard",
    ):
        assert f'"{field}"' in source


def test_17_source_has_fresh_artifact_names():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "terminal_capture_v23_validation_summary.json" in source
    assert "terminal_capture_v23_validation_episodes.csv" in source


def test_18_no_environment_step_in_pure_aggregation():
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_aggregate_v23_episode_fields")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "step" for node in ast.walk(fn))


def test_19_v23_summary_is_evaluation_only():
    assert module.EVALUATION_ONLY_FLAGS["training_performed"] is False
    assert module.EVALUATION_ONLY_FLAGS["reward_modified"] is False
    assert module.EVALUATION_ONLY_FLAGS["dynamics_modified"] is False


def test_20_v23_flag_is_in_parser_help_surface():
    assert "v23" in module.build_parser().format_help()


def test_21_v23_runtime_baseline_keeps_v21_and_v22_enabled():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "or terminal_capture_v23_validation" in source
    assert "goal_recovery_terminal_capture_v22_enabled=bool(" in source
    assert "goal_recovery_terminal_capture_v23_enabled=False" in source


def test_22_authority_repair_cli_is_registered_but_defaults_off():
    args = module.build_parser().parse_args([])
    assert args.terminal_capture_v23_authority_repair_validation is False
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "--terminal-capture-v23-authority-repair-validation" in source
    assert "terminal_capture_v23_authority_repair_validation_summary.json" in source
    assert "terminal_capture_v23_authority_repair_validation_episodes.csv" in source


def test_23_authority_repair_roots_are_independent_from_v23_baseline():
    assert module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_DATA_ROOT != module.TERMINAL_CAPTURE_V23_DATA_ROOT
    assert module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT != module.TERMINAL_CAPTURE_V23_LOG_ROOT
    assert module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_PLOT_ROOT != module.TERMINAL_CAPTURE_V23_PLOT_ROOT
    assert module.TERMINAL_CAPTURE_V23_FORMAL_BASELINE_ROOT == module.TERMINAL_CAPTURE_V23_LOG_ROOT


def test_24_authority_aggregate_counts_and_radial_means():
    summary = module._aggregate_v23_episode_fields([
        _row(
            0, True, True, False, 0.2, 0.98,
            terminal_v23_positive_radial_error=True,
            terminal_v23_radial_propulsion_allowed=True,
            terminal_v23_radial_propulsion_suppressed=False,
            terminal_v23_authority_active=True,
            terminal_v23_preboundary_authority_repair_active=True,
            terminal_v23_requested_propulsion_radial_mps=0.2,
            terminal_v23_executed_propulsion_radial_mps=0.2,
        ),
        _row(
            1, True, True, False, 0.1, 0.99,
            terminal_v23_positive_radial_error=True,
            terminal_v23_radial_propulsion_allowed=False,
            terminal_v23_radial_propulsion_suppressed=True,
            terminal_v23_authority_active=False,
            terminal_v23_preboundary_authority_repair_active=False,
            terminal_v23_requested_propulsion_radial_mps=0.1,
            terminal_v23_executed_propulsion_radial_mps=0.0,
        ),
    ])
    assert summary["v23_positive_radial_error_steps"] == 2
    assert summary["v23_radial_propulsion_allowed_steps"] == 1
    assert summary["v23_radial_propulsion_suppressed_steps"] == 1
    assert summary["v23_authority_active_steps"] == 1
    assert summary["v23_preboundary_authority_repair_active_steps"] == 1
    assert summary["mean_v23_requested_propulsion_radial_mps"] == pytest.approx(0.2)
    assert summary["mean_v23_executed_propulsion_radial_mps"] == pytest.approx(0.2)


def _authority_identity_fixture() -> tuple[
    FormalExperimentConfig,
    FormalExperimentConfig,
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    baseline_config = FormalExperimentConfig(
        output_dir=Path("C:/tmp/v23-authority-base"),
        max_steps=100,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        goal_recovery_terminal_capture_v22_enabled=True,
        goal_recovery_terminal_capture_v23_enabled=True,
    )
    candidate_config = replace(
        baseline_config,
        output_dir=Path("C:/tmp/v23-authority-candidate"),
    )
    episodes = [_episode()]
    baseline = {
        "checkpoint": {
            "resolved_path": "C:/tmp/frozen.pt",
            "sha256": "sha",
            "policy_hash": "policy",
        },
        "scenario_manifest_hash": "manifest",
        "results": {"on": {"episodes": episodes}},
    }
    summary = {
        "candidate_config": module._effective_correction_config(
            baseline_config,
            enabled=True,
        ),
        "stable_distance_threshold": 1.0,
        "stable_speed_threshold": 0.5,
        "stable_hold_steps": 5,
    }
    return baseline_config, candidate_config, baseline, summary, episodes


def test_25_authority_repair_identity_accepts_source_only_change():
    base, candidate, baseline, summary, episodes = _authority_identity_fixture()
    checks = module._v23_authority_repair_identity_checks(
        baseline_evaluation=baseline,
        baseline_summary=summary,
        baseline_config=base,
        candidate_config=candidate,
        checkpoint_path=Path("C:/tmp/frozen.pt"),
        checkpoint_sha256="sha",
        policy_hash="policy",
        scenario_manifest_hash="manifest",
        current_episodes=episodes,
    )
    assert checks["only_v23_authority_repair_changed"]


def test_26_authority_repair_identity_rejects_parameter_change():
    base, candidate, baseline, summary, episodes = _authority_identity_fixture()
    candidate = replace(candidate, goal_recovery_terminal_hold_target_radius_m=0.95)
    with pytest.raises(ValueError, match="authority-repair identity mismatch"):
        module._v23_authority_repair_identity_checks(
            baseline_evaluation=baseline,
            baseline_summary=summary,
            baseline_config=base,
            candidate_config=candidate,
            checkpoint_path=Path("C:/tmp/frozen.pt"),
            checkpoint_sha256="sha",
            policy_hash="policy",
            scenario_manifest_hash="manifest",
            current_episodes=episodes,
        )


def test_27_source_has_additive_authority_schema_fields():
    source = Path(module.__file__).read_text(encoding="utf-8")
    for field in (
        "terminal_v23_positive_radial_error",
        "terminal_v23_radial_propulsion_allowed",
        "terminal_v23_radial_propulsion_suppressed",
        "terminal_v23_radial_propulsion_suppression_reason",
        "terminal_v23_requested_propulsion_radial_mps",
        "terminal_v23_executed_propulsion_radial_mps",
        "terminal_v23_authority_active",
        "terminal_v23_preboundary_authority_repair_active",
        "terminal_v23_net_tracking_radial_before_v21_mps",
        "terminal_v23_net_tracking_radial_after_v21_mps",
    ):
        assert f'"{field}"' in source
