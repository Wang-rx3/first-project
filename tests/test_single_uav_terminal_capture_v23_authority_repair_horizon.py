"""Pure preparation contracts for the repaired v2.3 horizon diagnostic.

These tests exercise parser/identity/post-processing helpers only.  They do not
invoke the evaluator, an environment, ``environment.step``, or any rollout.
"""

import ast
import importlib
from pathlib import Path

import pytest


FLAG = "--terminal-capture-v23-authority-repair-horizon120-diagnostic"


def _module():
    return importlib.import_module(
        "src.experiments.single_uav_goal_recovery_correction_evaluation"
    )


def _episode_identity():
    return [
        {
            "episode_index": 1,
            "episode_id": "test_r99_episode_0001",
            "scenario_seed": 4021774,
            "noise_seed": 4221062,
            "scenario_hash": "scenario-1",
        }
    ]


def _identity_fixture(module):
    common = dict(
        single_uav_stable_hold_steps=5,
        max_steps=module.HORIZON_BASELINE_MAX_STEPS,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
        goal_recovery_terminal_capture_v22_enabled=True,
        goal_recovery_terminal_hold_enter_radius_m=0.98,
        goal_recovery_terminal_hold_release_radius_m=1.05,
        goal_recovery_terminal_hold_restorative_enabled=True,
        goal_recovery_terminal_capture_v23_enabled=True,
        goal_recovery_terminal_hold_target_radius_m=0.96,
        goal_recovery_max_delta_v_mps=0.6,
        goal_recovery_terminal_effective_deceleration_mps2=1.697726,
        goal_recovery_terminal_stopping_margin_m=1.328009,
    )
    baseline_config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/frozen-v23-repair-formal100"), **common
    )
    diagnostic_config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/frozen-v23-repair-horizon120"),
        **dict(common, max_steps=module.HORIZON_DIAGNOSTIC_MAX_STEPS),
    )
    episodes = _episode_identity()
    checkpoint = {
        "resolved_path": "C:/tmp/frozen_actor.pt",
        "sha256": "sha",
        "policy_hash": "policy",
        "observation_dimension": int(baseline_config.node_feature_dim),
    }
    baseline_evaluation = {
        "checkpoint": checkpoint,
        "scenario_manifest_hash": "scenario",
        "results": {"on": {"episodes": episodes}},
    }
    baseline_summary = {
        "candidate_config": module._effective_correction_config(
            baseline_config, enabled=True
        ),
        "candidate_config_hash": module._canonical_hash(
            module._effective_correction_config(baseline_config, enabled=True)
        ),
        "stable_distance_threshold": 1.0,
        "stable_speed_threshold": 0.5,
        "stable_hold_steps": 5,
        "terminal_capture_v23_authority_repaired": True,
        "only_v23_authority_repair_changed": True,
    }
    return (
        baseline_evaluation,
        baseline_summary,
        baseline_config,
        diagnostic_config,
        episodes,
    )


def _synthetic_rows(module):
    rows = []
    for index in range(module.HORIZON_DIAGNOSTIC_MAX_STEPS):
        row = {
            "step": index,
            "goal_distance_ok": int(index >= 100),
            "goal_speed_ok": 0,
            "stable_candidate_step": int(index >= 104),
            "stable_goal": int(index >= 104),
            "stable_hold_counter": max(0, index - 103) if index >= 104 else 0,
            "goal_distance_for_stable_m": 10.0 - 0.03 * index,
            "goal_distance_after_m": 10.0 - 0.03 * index,
            "post_step_actual_speed_mps": 0.2,
            "post_step_actual_radial_velocity_mps": 0.4,
            "post_step_actual_tangential_velocity_mps": 0.1,
            "terminal_v21_actor_opposing_radial_mps": 0.2 if index >= 100 else 0.0,
            "terminal_v21_radial_conflict_active": int(index >= 100),
            "radial_propulsion_delta_v_mps": [1.0, 0.0, 0.0]
            if index % 2 == 0
            else [0.0, 0.0, 0.0],
            "radial_braking_delta_v_mps": [0.0, 0.0, 0.0]
            if index % 2 == 0
            else [1.0, 0.0, 0.0],
            "collision": int(index == 0 or index == 100),
            "safety_projection_applied": int(index in {1, 101}),
            "boundary_projection_applied": int(index in {2, 102}),
            "action_clipped": int(index in {3, 103}),
            "terminal_correction_clipped": int(index in {4, 104}),
        }
        rows.append(row)
    return rows


def test_new_cli_is_formal_on_only_and_uses_independent_roots():
    module = _module()
    parser = module.build_parser()
    args = parser.parse_args([FLAG])
    resolved = module._resolve_cli_mode(args, [FLAG])
    assert args.__dict__[FLAG[2:].replace("-", "_")] is True
    assert resolved["condition"] == "on"
    assert resolved["formal"] is True
    assert resolved["data_root"] == module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_DATA_ROOT
    assert resolved["logs_root"] == module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_LOG_ROOT
    assert resolved["plots_root"] == module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_PLOT_ROOT
    assert module.FORMAL_EPISODES == 20
    assert module.HORIZON_BASELINE_MAX_STEPS == 100
    assert module.HORIZON_DIAGNOSTIC_MAX_STEPS == 120
    assert resolved["logs_root"] != module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT


def test_old_authority_repair_formal100_cli_and_root_are_unchanged():
    module = _module()
    parser = module.build_parser()
    old_flag = "--terminal-capture-v23-authority-repair-validation"
    args = parser.parse_args([old_flag])
    resolved = module._resolve_cli_mode(args, [old_flag])
    assert resolved["condition"] == "on"
    assert resolved["formal"] is True
    assert resolved["logs_root"] == module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT
    assert "horizon120" not in str(resolved["logs_root"])
    assert module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_LOG_ROOT != (
        module.TERMINAL_CAPTURE_V23_AUTHORITY_REPAIR_HORIZON_LOG_ROOT
    )


def test_horizon_augmentation_has_correct_zero_based_extension_boundaries():
    module = _module()
    augmented = module._augment_horizon_episode_summary(
        {"episode_id": "test_r99_episode_0001"}, _synthetic_rows(module)
    )
    assert augmented["baseline_valid_step_index_start"] == 0
    assert augmented["baseline_valid_step_index_end"] == 99
    assert augmented["extension_valid_step_index_start"] == 100
    assert augmented["extension_valid_step_index_end"] == 119
    assert augmented["entered1_before_or_at_100"] == 0
    assert augmented["entered1_in_extension_101_120"] == 1
    assert augmented["never_entered1_by_120"] == 0
    assert augmented["stable_before_or_at_100"] == 0
    assert augmented["stable_in_extension_101_120"] == 1
    assert augmented["never_stable_by_120"] == 0
    assert augmented["goal_distance_at_step_100"] == pytest.approx(10.0 - 0.03 * 99)
    assert augmented["goal_distance_at_step_120"] == pytest.approx(10.0 - 0.03 * 119)
    assert augmented["distance_reduction_during_extension_m"] == pytest.approx(0.6)
    assert augmented["extension_steps_observed"] == 20
    assert augmented["extension_actor_opposing_fraction"] == pytest.approx(1.0)
    assert augmented["extension_v21_conflict_fraction"] == pytest.approx(1.0)
    assert augmented["extension_propulsion_braking_switch_count"] == 19
    assert augmented["first100_collision_count"] == 1
    assert augmented["extension_collision_count"] == 1
    assert augmented["first100_safety_projection_steps"] == 1
    assert augmented["extension_safety_projection_steps"] == 1
    assert augmented["first100_boundary_projection_steps"] == 1
    assert augmented["extension_boundary_projection_steps"] == 1
    assert augmented["first100_action_clipping_steps"] == 1
    assert augmented["extension_action_clipping_steps"] == 1
    assert augmented["first100_terminal_saturation_steps"] == 1
    assert augmented["extension_terminal_saturation_steps"] == 1


def test_horizon_identity_locks_every_requested_invariant():
    module = _module()
    baseline, summary, baseline_config, diagnostic_config, episodes = _identity_fixture(
        module
    )
    checks = module._v23_authority_repair_horizon_identity_checks(
        baseline_evaluation=baseline,
        baseline_summary=summary,
        baseline_config=baseline_config,
        diagnostic_config=diagnostic_config,
        checkpoint_path=Path("C:/tmp/frozen_actor.pt"),
        checkpoint_sha256="sha",
        policy_hash="policy",
        baseline_normalized_manifest_hash="scenario",
        current_episodes=episodes,
    )
    required = {
        "same_checkpoint",
        "same_checkpoint_sha256",
        "same_policy_hash",
        "same_scenario_manifest",
        "same_ordered_episode_identity",
        "same_reward",
        "same_dynamics",
        "same_observation",
        "same_action_transform",
        "same_stable_criterion",
        "same_max_correction",
        "same_a_eff",
        "same_stopping_margin",
        "same_v21_gain",
        "same_v22_parameters",
        "same_v23_parameters",
        "same_authority_repair_behavior",
        "same_controller_config",
        "only_horizon_changed",
    }
    assert required <= checks.keys()
    assert all(checks[key] is True for key in required)
    changed = module.FormalExperimentConfig(
        output_dir=diagnostic_config.output_dir,
        **{
            field.name: getattr(diagnostic_config, field.name)
            for field in module.fields(diagnostic_config)
            if field.name not in {"output_dir", "single_uav_goal_hold_reward"}
        },
        single_uav_goal_hold_reward=0.25,
    )
    with pytest.raises(ValueError, match="same_reward"):
        module._v23_authority_repair_horizon_identity_checks(
            baseline_evaluation=baseline,
            baseline_summary=summary,
            baseline_config=baseline_config,
            diagnostic_config=changed,
            checkpoint_path=Path("C:/tmp/frozen_actor.pt"),
            checkpoint_sha256="sha",
            policy_hash="policy",
            baseline_normalized_manifest_hash="scenario",
            current_episodes=episodes,
        )


def test_horizon_artifact_contract_and_protected_controller_are_static_only():
    root = Path(__file__).resolve().parents[1]
    evaluator_source = (
        root / "src" / "experiments" / "single_uav_goal_recovery_correction_evaluation.py"
    ).read_text(encoding="utf-8")
    controller_source = (
        root / "src" / "control" / "goal_recovery_correction.py"
    ).read_text(encoding="utf-8")
    for token in (
        "HORIZON120_DIAGNOSTIC_READY",
        FLAG,
        "terminal_capture_v23_authority_repair_horizon120_summary.json",
        "terminal_capture_v23_authority_repair_horizon120_recovery_episodes.csv",
        "focus_recovery_episodes.csv",
        "entered1_in_extension_101_120",
        "extension_v21_conflict_fraction",
        "first100_terminal_saturation_steps",
    ):
        # The readiness marker is in the preparation report/README; the other
        # tokens belong to the evaluator source.
        if token == "HORIZON120_DIAGNOSTIC_READY":
            continue
        assert token in evaluator_source
    report = (
        root
        / "sections"
        / "single_uav_near_goal_capture_horizon120_diagnostic_preparation.md"
    )
    assert report.read_text(encoding="utf-8").splitlines()[0] == (
        "HORIZON120_DIAGNOSTIC_READY"
    )
    assert FLAG not in controller_source
    assert "optimizer.step(" not in evaluator_source
    assert "trainer.update(" not in evaluator_source


def test_preparation_tests_do_not_call_rollout_environment_or_training():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_names & {
        "run_goal_recovery_correction_evaluation",
        "_rollout_frozen_actor_episode",
        "run_external_recovery_method_study",
    }
    assert not called_attributes & {"step", "reset", "collect_rollout"}
