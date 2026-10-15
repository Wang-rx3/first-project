"""Fast contract tests for the frozen correction ON/OFF entry point."""

import ast
import csv
import importlib
import inspect
import json
from pathlib import Path

import pytest


def _module():
    return importlib.import_module(
        "src.experiments.single_uav_goal_recovery_correction_evaluation"
    )


def test_cli_defaults_to_paired_two_episode_smoke():
    args = _module().build_parser().parse_args([])
    assert args.condition == "paired"
    assert args.formal is False
    assert args.smoke_count == 2


def test_cli_supports_direct_on_off_and_formal_aliases():
    parser = _module().build_parser()
    assert parser.parse_args(["--goal-recovery-correction"]).legacy_condition == "on"
    assert parser.parse_args(["--no-goal-recovery-correction"]).legacy_condition == "off"
    assert parser.parse_args(["--paired"]).legacy_condition == "paired"
    assert parser.parse_args(["--formal"]).formal is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--formal", "--smoke-count", "1"])


def test_output_layout_is_fresh_and_condition_scoped(tmp_path):
    module = _module()
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"frozen")
    paired = module._validate_fresh_output_layout(tmp_path / "paired", "paired", checkpoint)
    assert paired["off"] != paired["on"]
    assert paired["off"].parent == paired["base"]
    single = module._validate_fresh_output_layout(tmp_path / "single", "on", checkpoint)
    assert single["on"] == single["base"]
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        module._validate_fresh_output_layout(existing, "paired", checkpoint)


def test_ordered_pool_identity_is_stable_and_paired_mismatch_fails():
    module = _module()
    pool = [
        {"episode_index": 1, "episode_id": "episode_1", "noise_seed": 11},
        {"episode_index": 2, "episode_id": "episode_2", "noise_seed": 22},
    ]
    assert module._ordered_pool_identity(pool) == [
        (1, "episode_1", 11),
        (2, "episode_2", 22),
    ]
    module._assert_paired_identity(pool, [dict(row) for row in pool])
    changed = [dict(row) for row in pool]
    changed[1]["noise_seed"] = 23
    with pytest.raises(AssertionError):
        module._assert_paired_identity(pool, changed)


def test_comparison_rows_are_off_on_delta():
    rows = _module()._comparison_rows(
        {"stable_goal": 1, "entered_1m": 2},
        {"stable_goal": 3, "entered_1m": 5},
    )
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["stable_goal"] == {
        "metric": "stable_goal",
        "OFF": 1.0,
        "ON": 3.0,
        "delta": 2.0,
    }
    assert by_metric["entered_1m"]["delta"] == 3.0


def test_evaluation_only_flags_forbid_all_training_paths():
    EVALUATION_ONLY_FLAGS = _module().EVALUATION_ONLY_FLAGS
    assert EVALUATION_ONLY_FLAGS["training_performed"] is False
    assert EVALUATION_ONLY_FLAGS["bc_retraining_executed"] is False
    assert EVALUATION_ONLY_FLAGS["dagger_collection_executed"] is False
    assert EVALUATION_ONLY_FLAGS["teacher_bootstrap_executed"] is False
    assert EVALUATION_ONLY_FLAGS["teacher_constructed"] is False
    assert EVALUATION_ONLY_FLAGS["teacher_queried"] is False
    for field in (
        "ppo_updates",
        "mappo_updates",
        "optimizer_steps",
        "backward_calls",
        "actor_updates",
        "checkpoint_writes",
    ):
        assert EVALUATION_ONLY_FLAGS[field] == 0


def test_entrypoint_has_no_forbidden_runner_or_training_calls():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "experiments"
        / "single_uav_goal_recovery_correction_evaluation.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called_names = set()
    called_attributes = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)
    assert "run_external_recovery_method_study" not in called_names
    assert "run_static_obstacle_dagger_recovery_goal" not in called_names
    assert "_train_imitation" not in called_names
    assert "backward" not in called_attributes
    assert "step" not in {
        name for name in called_attributes if name in {"optimizer_step"}
    }
    # The canonical static holdout is built exactly once and reused by modes.
    assert sum(name == "_build_pool" for name in called_names) == 1
    source = path.read_text(encoding="utf-8")
    assert "trainer.update(" not in source
    assert "optimizer.step(" not in source


def test_only_on_condition_constructs_correction_runtime():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "experiments"
        / "single_uav_goal_recovery_correction_evaluation.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "if correction_enabled\n        else None" in source
    assert "if correction is not None:" in source
    assert "reason = \"disabled\"" in source
    assert "teacher.action" not in source


def test_terminal_capture_validation_cli_is_locked_and_uses_fresh_roots():
    module = _module()
    parser = module.build_parser()
    args = parser.parse_args(["--terminal-capture-validation"])
    resolved = module._resolve_cli_mode(args, ["--terminal-capture-validation"])
    assert resolved["condition"] == "on"
    assert resolved["formal"] is True
    assert resolved["data_root"] == module.TERMINAL_CAPTURE_DATA_ROOT
    assert resolved["logs_root"] == module.TERMINAL_CAPTURE_LOG_ROOT
    assert resolved["plots_root"] == module.TERMINAL_CAPTURE_PLOT_ROOT
    for conflict in ("--paired", "--no-goal-recovery-correction", "--smoke", "--actual-stable-validation"):
        argv = ["--terminal-capture-validation", conflict]
        with pytest.raises(ValueError):
            module._resolve_cli_mode(parser.parse_args(argv), argv)


def test_terminal_capture_cli_parameters_are_configurable():
    args = _module().build_parser().parse_args(
        [
            "--terminal-capture-validation",
            "--terminal-capture-radius", "7.5",
            "--terminal-radial-speed-floor", "0.2",
            "--terminal-radial-brake-gain", "0.9",
            "--terminal-tangential-gain", "1.1",
        ]
    )
    assert args.terminal_capture_radius == 7.5
    assert args.terminal_radial_speed_floor == 0.2
    assert args.terminal_radial_brake_gain == 0.9
    assert args.terminal_tangential_gain == 1.1


def test_terminal_baseline_loader_requires_actual_stable_formal20(tmp_path):
    module = _module()
    root = tmp_path / "baseline"
    root.mkdir()
    path = root / "evaluation_summary.json"
    payload = {
        "stage": module.STAGE_NAME,
        "condition": "on",
        "formal": True,
        "episode_count": 20,
        "actual_stable_validation": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, resolved, digest = module._load_terminal_capture_baseline(root)
    assert loaded == payload
    assert resolved == path.resolve()
    assert len(digest) == 64
    payload["episode_count"] = 19
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        module._load_terminal_capture_baseline(root)


def test_terminal_logging_preserves_original_pipeline_and_forbidden_modules():
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "experiments"
        / "single_uav_goal_recovery_correction_evaluation.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "fusion.corrected_delta_v_pre_action_transform_mps" in source
    assert "fusion.corrected_delta_v_pre_safety_mps" in source
    assert '"terminal_raw_correction_norm_mps"' in source
    assert '"terminal_correction_clipped"' in source
    assert '"post_step_actual_speed_mps"' in source
    assert "terminal_capture_validation_summary.json" in source
    assert "run_external_recovery_method_study(" not in source
    assert "trainer.update(" not in source


def test_terminal_capture_future_summary_schema_is_complete():
    module = _module()
    episode = {
        "passed_obstacle": 1,
        "post_bypass_recovery": 1,
        "entered_8m": 1,
        "entered_5m": 1,
        "entered_3m": 1,
        "entered_1m": 1,
        "stable_goal": 0,
        "collision": 0,
        "safety_projection_steps": 2,
        "boundary_projection_steps": 1,
        "terminal_correction_saturation_fraction": 0.25,
        "actual_speed_at_first_entered_1m_mps": 3.0,
        "mean_actual_speed_inside_1m_mps": 2.0,
        "fraction_inside_1m_speed_ok": 0.0,
        "max_stable_hold_counter": 0,
        "stable_speed_threshold_mps": 0.5,
        "mean_tangential_velocity_inside_1m_mps": 1.5,
        "correction_active_fraction_inside_1m": 0.8,
        "mean_correction_norm_inside_1m_mps": 0.4,
    }
    schema = {
        **module._aggregate_episode_summaries([episode]),
        **module.aggregate_actual_stable_validation([episode]),
    }
    required = {
        "passed_obstacle",
        "post_bypass_recovery",
        "entered_8m",
        "entered_5m",
        "entered_3m",
        "entered_1m",
        "stable_goal",
        "safety_projection_steps",
        "boundary_projection_steps",
        "collision",
        "entry_actual_speed_mean",
        "entry_actual_speed_min",
        "entry_actual_speed_max",
        "inside_1m_actual_speed_mean",
        "inside_1m_actual_speed_min",
        "episodes_ever_speed_ok_inside_1m",
        "max_stable_hold_counter_distribution",
        "mean_tangential_velocity_inside_1m",
        "terminal_correction_saturation_fraction",
    }
    assert required <= schema.keys()


def test_terminal_validation_uses_canonical_identity_guard_before_rollout_loop():
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "experiments"
        / "single_uav_goal_recovery_correction_evaluation.py"
    )
    source = path.read_text(encoding="utf-8")
    identity = source.index("identity_checks = assert_previous_formal_identity(")
    rollout_loop = source.index("for mode in modes:")
    assert identity < rollout_loop
    assert "same_ordered_episode_identity_as_previous_formal" in source
    assert "same_checkpoint_sha256_as_previous_formal" in (
        Path(__file__).resolve().parents[1]
        / "src" / "experiments" / "single_uav_goal_recovery_stable_contract.py"
    ).read_text(encoding="utf-8")


def test_terminal_capture_v2_validation_cli_is_locked_and_isolated():
    module = _module()
    parser = module.build_parser()
    args = parser.parse_args(["--terminal-capture-v2-validation"])
    resolved = module._resolve_cli_mode(args, ["--terminal-capture-v2-validation"])
    assert resolved == {
        "condition": "on",
        "formal": True,
        "data_root": module.TERMINAL_CAPTURE_V2_DATA_ROOT,
        "logs_root": module.TERMINAL_CAPTURE_V2_LOG_ROOT,
        "plots_root": module.TERMINAL_CAPTURE_V2_PLOT_ROOT,
    }
    for conflict in (
        "--paired",
        "--no-goal-recovery-correction",
        "--smoke",
        "--actual-stable-validation",
        "--terminal-capture-validation",
    ):
        argv = ["--terminal-capture-v2-validation", conflict]
        with pytest.raises(ValueError):
            module._resolve_cli_mode(parser.parse_args(argv), argv)


def test_terminal_capture_v2_baseline_requires_v1_formal20(tmp_path):
    module = _module()
    root = tmp_path / "v1"
    root.mkdir()
    evaluation = {
        "stage": module.STAGE_NAME,
        "condition": "on",
        "formal": True,
        "episode_count": 20,
        "terminal_capture_validation": True,
        "effective_goal_recovery_correction_config_hash": "same",
    }
    terminal = {
        "stage": "single_uav_terminal_capture_validation",
        "candidate_config_hash": "same",
        "candidate_canonical_defaults": True,
    }
    (root / "evaluation_summary.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    terminal_path = root / "terminal_capture_validation_summary.json"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    loaded, path, digest = module._load_terminal_capture_v1_baseline(root)
    assert loaded == evaluation
    assert path == terminal_path.resolve()
    assert len(digest) == 64
    evaluation["episode_count"] = 19
    (root / "evaluation_summary.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        module._load_terminal_capture_v1_baseline(root)


def test_terminal_capture_v2_logging_and_future_artifact_contract_is_complete():
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "experiments"
        / "single_uav_goal_recovery_correction_evaluation.py"
    )
    source = path.read_text(encoding="utf-8")
    step_fields = {
        "terminal_v2_enabled",
        "terminal_v2_governor_eligible",
        "terminal_v2_governor_active",
        "terminal_v2_a_eff_mps2",
        "terminal_v2_stopping_margin_m",
        "terminal_v2_distance_to_threshold_m",
        "terminal_v2_usable_braking_distance_m",
        "terminal_v2_base_desired_radial_speed_mps",
        "terminal_v2_envelope_speed_mps",
        "terminal_v2_governed_desired_radial_speed_mps",
        "terminal_v2_radial_overspeed_mps",
        "terminal_v2_braking_required",
        "terminal_v2_inside_goal_hold",
        "terminal_v2_anti_stall_active",
    }
    assert all(f'"{field}"' in source for field in step_fields)
    for filename in (
        "terminal_capture_v2_validation_summary.json",
        "terminal_capture_v2_validation_episodes.csv",
        "steps.csv",
        "episodes.csv",
    ):
        assert filename in source


def test_terminal_capture_v2_episode_aggregate_has_required_metrics():
    module = _module()
    row = {
        "passed_obstacle": 1,
        "post_bypass_recovery": 1,
        "entered_8m": 1,
        "entered_5m": 1,
        "entered_3m": 1,
        "entered_1m": 1,
        "stable_goal": 0,
        "timeout": 1,
        "collision": 0,
        "boundary_projection_steps": 2,
        "safety_projection_steps": 1,
        "goal_correction_active_steps": 3,
        "terminal_capture_active_steps": 3,
        "terminal_v2_governor_eligible_steps": 4,
        "terminal_v2_governor_active_steps": 2,
        "terminal_v2_anti_stall_active_steps": 1,
        "terminal_v2_mean_radial_overspeed_mps": 0.4,
        "terminal_v2_max_radial_overspeed_mps": 0.8,
        "terminal_v2_governor_limiting_fraction": 0.5,
    }
    aggregate = module._aggregate_episode_summaries([row])
    assert aggregate["terminal_v2_governor_eligible_steps"] == 4
    assert aggregate["terminal_v2_governor_active_steps"] == 2
    assert aggregate["terminal_v2_anti_stall_active_steps"] == 1
    assert aggregate["terminal_v2_mean_radial_overspeed_mps"] == 0.4
    assert aggregate["terminal_v2_max_radial_overspeed_mps"] == 0.8
    assert aggregate["terminal_v2_governor_limiting_fraction"] == 0.5


def test_terminal_capture_v2_protected_runtime_modules_are_not_edited_here():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "experiments"
        / "single_uav_goal_recovery_correction_evaluation.py"
    ).read_text(encoding="utf-8")
    assert "environment.step(" in source  # future manual runner remains the existing path
    assert "terminal_capture_v2_validation" in source
    assert "trainer.update(" not in source
    assert "optimizer.step(" not in source


# Horizon diagnostic contracts are intentionally exercised with plain mappings and
# temporary files only.  These helpers keep the tests useful while the production
# implementation settles on whether a snapshot is flattened or nested.
def _horizon_result_mapping(value):
    if isinstance(value, tuple):
        assert value, "horizon helper returned an empty tuple"
        value = value[0]
    assert isinstance(value, dict), type(value)
    return value


def _walk_mappings(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _contains_trace_marker(value, marker):
    for mapping in _walk_mappings(value):
        if mapping.get("trace_marker") == marker or mapping.get("marker") == marker:
            return True
    return False


def _contains_marker_under_key(value, key_needles, marker):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower().replace("-", "_")
            if all(needle in lowered for needle in key_needles) and _contains_trace_marker(child, marker):
                return True
            if _contains_marker_under_key(child, key_needles, marker):
                return True
    elif isinstance(value, (list, tuple)):
        return any(
            _contains_marker_under_key(child, key_needles, marker)
            for child in value
        )
    return False


def _keys_containing(mapping, *needles):
    return [
        key for key in mapping
        if all(needle in str(key).lower().replace("-", "_") for needle in needles)
    ]


def _horizon_source():
    return (
        Path(__file__).resolve().parents[1]
        / "src" / "experiments" / "single_uav_goal_recovery_correction_evaluation.py"
    )


def _call_horizon_identity(module, baseline, current):
    """Call the identity helper by semantic parameter names, not one signature."""
    function = module._horizon_only_identity_checks
    signature = inspect.signature(function)
    values = {
        "baseline": baseline,
        "reference": baseline,
        "previous": baseline,
        "previous_formal": baseline,
        "baseline_summary": baseline,
        "baseline_reference": baseline,
        "baseline_rows": baseline["results"]["on"]["episodes"],
        "baseline_episode_rows": baseline["results"]["on"]["episodes"],
        "previous_summary": baseline,
        "current": current,
        "candidate": current,
        "diagnostic": current,
        "current_summary": current,
        "diagnostic_summary": current,
        "candidate_summary": current,
        "diagnostic_rows": current["results"]["on"]["episodes"],
        "current_rows": current["results"]["on"]["episodes"],
        "diagnostic_episode_rows": current["results"]["on"]["episodes"],
        "checkpoint_path": Path("C:/tmp/frozen_actor.pt"),
        "checkpoint_identity": current["checkpoint"],
        "checkpoint_sha256": "sha",
        "baseline_checkpoint_sha256": "sha",
        "baseline_evaluation": baseline,
        "baseline_config": baseline["_baseline_config"],
        "diagnostic_config": current["_diagnostic_config"],
        "baseline_normalized_manifest_hash": "scenario",
        "policy_hash": "policy",
        "scenario_manifest_hash": "scenario",
        "baseline_manifest_hash": "scenario",
        "ordered_pool_identity": [(1, "episode_1", 11)],
        "current_episodes": current["results"]["on"]["episodes"],
        "stable_goal_criterion_identity": True,
        "stable_identity": True,
        "reward_identity": True,
        "dynamics_identity": True,
        "action_identity": True,
        "action_space_identity": True,
        "max_correction_identity": True,
    }
    positional = []
    keyword = {}
    required_positional = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            if parameter.name in values:
                positional.append(values[parameter.name])
            elif parameter.default is parameter.empty:
                required_positional.append(parameter.name)
        elif parameter.kind == parameter.KEYWORD_ONLY:
            if parameter.name in values:
                keyword[parameter.name] = values[parameter.name]
            elif parameter.default is parameter.empty:
                raise AssertionError(
                    f"unrecognized required horizon identity parameter: {parameter.name}"
                )
    if required_positional:
        raise AssertionError(
            f"unrecognized required horizon identity parameters: {required_positional}"
        )
    # Positional-or-keyword parameters can be supplied positionally without
    # guessing whether a future implementation renamed one of its arguments.
    return function(*positional, **keyword)


def _identity_value(checks, category):
    aliases = {
        "checkpoint": ("checkpoint",),
        "scenario": ("scenario", "manifest"),
        "order": ("ordered", "episode", "identity"),
        "stable": ("stable", "criterion"),
        "reward": ("reward",),
        "dynamics": ("dynamics",),
        "action": ("action",),
        "max_correction": ("max", "correction"),
    }[category]
    matches = []
    for key, value in checks.items():
        lowered = str(key).lower().replace("-", "_")
        if all(token in lowered for token in aliases) and isinstance(value, bool):
            matches.append((key, value))
    assert matches, f"no {category} identity boolean in {checks!r}"
    preferred = [
        item for item in matches
        if any(marker in str(item[0]).lower() for marker in ("same", "identity", "unchanged", "preserved"))
    ]
    selected = preferred or matches
    modified = [item for item in selected if "modified" in str(item[0]).lower()]
    if modified and not preferred:
        return all(not value for _key, value in modified)
    return all(value for _key, value in selected)


def _horizon_identity_fixture(module):
    episode = {
        "episode_index": 1,
        "episode_id": "episode_1",
        "scenario_seed": 11,
        "noise_seed": 12,
        "scenario_hash": "episode-hash",
    }
    checkpoint = {
        "resolved_path": "C:/tmp/frozen_actor.pt",
        "sha256": "sha",
        "policy_hash": "policy",
    }
    baseline_config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/horizon-baseline"),
        max_steps=module.HORIZON_BASELINE_MAX_STEPS,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
    )
    diagnostic_config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/horizon-diagnostic"),
        max_steps=module.HORIZON_DIAGNOSTIC_MAX_STEPS,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
    )
    common = {
        "checkpoint": checkpoint,
        "scenario_manifest_hash": "scenario",
        "ordered_pool_identity": [(1, "episode_1", 11)],
    }
    candidate_config = module._effective_correction_config(
        diagnostic_config, enabled=True
    )
    baseline = {
        "stage": "single_uav_terminal_capture_v2_validation",
        "condition": "on",
        "formal": True,
        "episode_count": 20,
        "terminal_capture_v2_validation": True,
        "results": {"on": {"episodes": [episode]}},
        "effective_goal_recovery_correction_config_hash": module._canonical_hash(
            candidate_config
        ),
        "candidate_config_hash": module._canonical_hash(candidate_config),
        "stable_distance_threshold": 1.0,
        "stable_speed_threshold": 0.5,
        "stable_hold_steps": 5,
        "_baseline_config": baseline_config,
        **common,
    }
    current = {
        **baseline,
        "stage": "single_uav_terminal_capture_v2_horizon_diagnostic",
        "max_steps": module.HORIZON_DIAGNOSTIC_MAX_STEPS,
        "_diagnostic_config": diagnostic_config,
    }
    return baseline, current


def test_terminal_capture_v2_horizon_flag_defaults_off_and_locks_on_formal_roots():
    module = _module()
    parser = module.build_parser()
    assert parser.parse_args([]).terminal_capture_v2_horizon_diagnostic is False
    args = parser.parse_args(["--terminal-capture-v2-horizon-diagnostic"])
    assert args.terminal_capture_v2_horizon_diagnostic is True
    resolved = module._resolve_cli_mode(
        args, ["--terminal-capture-v2-horizon-diagnostic"]
    )
    assert resolved["condition"] == "on"
    assert resolved["formal"] is True
    assert resolved["data_root"] == module.TERMINAL_CAPTURE_V2_HORIZON_DATA_ROOT
    assert resolved["logs_root"] == module.TERMINAL_CAPTURE_V2_HORIZON_LOG_ROOT
    assert resolved["plots_root"] == module.TERMINAL_CAPTURE_V2_HORIZON_PLOT_ROOT


def test_terminal_capture_v2_horizon_flag_rejects_non_locked_options():
    module = _module()
    parser = module.build_parser()
    for conflict in (
        "--paired",
        "--no-goal-recovery-correction",
        "--smoke",
        "--smoke-count",
        "--actual-stable-validation",
        "--terminal-capture-validation",
        "--terminal-capture-v2-validation",
        "--data-root",
        "--logs-root",
        "--plots-root",
        "--checkpoint",
        "--terminal-capture-v2-formal-baseline-root",
    ):
        argv = [
            "--terminal-capture-v2-horizon-diagnostic",
            conflict,
            "1" if conflict in {"--smoke-count", "--checkpoint"} else "x",
        ]
        with pytest.raises((ValueError, SystemExit)):
            module._resolve_cli_mode(parser.parse_args(argv), argv)


def test_horizon_step_budgets_are_exact_and_v2_validation_remains_100_steps():
    module = _module()
    assert module.HORIZON_BASELINE_MAX_STEPS == 100
    assert module.HORIZON_DIAGNOSTIC_MAX_STEPS == 120
    source = _horizon_source().read_text(encoding="utf-8")
    assert "terminal_capture_v2_validation" in source
    assert "HORIZON_BASELINE_MAX_STEPS" in source
    assert "HORIZON_DIAGNOSTIC_MAX_STEPS" in source


def test_horizon_frozen_v2_controller_constants_are_exact():
    module = _module()
    config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/horizon-frozen-contract"),
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
    )
    effective = module._effective_correction_config(config, enabled=True)
    assert effective["max_delta_v_mps"] == pytest.approx(0.6)
    assert effective["terminal_effective_deceleration_mps2"] == pytest.approx(1.697726)
    assert effective["terminal_stopping_margin_m"] == pytest.approx(1.328009)
    assert effective["terminal_tangential_gain"] == pytest.approx(1.0)
    assert effective["terminal_propulsion_fade_enabled"] is True
    assert effective["terminal_smooth_blend"] is True


def test_load_terminal_capture_v2_baseline_requires_formal20_v2_artifacts(tmp_path):
    module = _module()
    root = tmp_path / "v2-baseline"
    root.mkdir()
    evaluation = {
        "stage": module.STAGE_NAME,
        "condition": "on",
        "formal": True,
        "episode_count": 20,
        "terminal_capture_v2_validation": True,
        "checkpoint": {"sha256": "sha", "resolved_path": "C:/tmp/frozen_actor.pt"},
        "scenario_manifest_hash": "scenario",
        "ordered_pool_identity": [[1, "episode_1", 11]],
    }
    terminal = {
        "stage": "single_uav_terminal_capture_v2_validation",
        "candidate_config_hash": "same",
        "terminal_capture_v2_added": True,
        "episodes": 20,
        "candidate_config": {
            "enabled": True,
            "terminal_capture_enabled": True,
            "terminal_capture_v2_enabled": True,
            "max_delta_v_mps": 0.6,
            "terminal_effective_deceleration_mps2": 1.697726,
            "terminal_stopping_margin_m": 1.328009,
            "terminal_tangential_gain": 1.0,
        },
        "stable_distance_threshold": 1.0,
        "stable_speed_threshold": 0.5,
        "stable_hold_steps": 5,
    }
    episodes = []
    for index in range(1, 21):
        episodes.append({
            "episode_index": index,
            "episode_id": f"episode_{index}",
            "scenario_seed": 100 + index,
            "noise_seed": 200 + index,
            "scenario_hash": f"hash-{index}",
            "max_steps": module.HORIZON_BASELINE_MAX_STEPS,
            "passed_obstacle": int(index <= 11),
            "post_bypass_recovery": int(index <= 11),
        })
    evaluation["results"] = {"on": {"episodes": episodes}}
    evaluation["effective_goal_recovery_correction_config_hash"] = "same"
    terminal["candidate_config_hash"] = "same"
    evaluation["ordered_pool_identity"] = [
        [row["episode_index"], row["episode_id"], row["noise_seed"]]
        for row in episodes
    ]
    evaluation["scenario_manifest_hash"] = "scenario"
    evaluation_path = root / "evaluation_summary.json"
    terminal_path = root / "terminal_capture_v2_validation_summary.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    loaded, loaded_summary, loaded_episodes, path, digest = (
        module._load_terminal_capture_v2_baseline(root)
    )
    assert loaded == evaluation
    assert loaded_summary == terminal
    assert len(loaded_episodes) == 20
    assert path == terminal_path.resolve()
    assert len(digest) == 64
    evaluation["episode_count"] = 19
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError):
        module._load_terminal_capture_v2_baseline(root)


def test_augment_horizon_summary_uses_zero_based_step100_110_120_and_after100():
    module = _module()
    rows = [
        {
            "step": index,
            "trace_marker": index,
            "goal_distance_after_m": float(1000 + index),
            "goal_distance_for_stable_m": float(1000 + index),
            "post_step_actual_speed_mps": float(10 + index),
            "post_step_actual_radial_velocity_mps": float(20 + index),
            "post_step_actual_tangential_velocity_mps": float(30 + index),
            "stable_hold_counter": index,
            "goal_distance_ok": 0,
            "goal_speed_ok": 0,
            "stable_candidate_step": 0,
            "stable_goal": 0,
            "terminal_v2_governor_active": int(index >= 99),
        }
        for index in range(module.HORIZON_DIAGNOSTIC_MAX_STEPS)
    ]
    summary = {"episode_id": "episode_1", "trace_marker": "summary"}
    augmented = _horizon_result_mapping(
        module._augment_horizon_episode_summary(dict(summary), rows)
    )
    assert summary == {"episode_id": "episode_1", "trace_marker": "summary"}
    for human_step, zero_index in ((100, 99), (110, 109), (120, 119)):
        assert augmented[f"goal_distance_at_step_{human_step}"] == pytest.approx(
            1000 + zero_index
        )
        assert augmented[f"actual_speed_at_step_{human_step}"] == pytest.approx(
            10 + zero_index
        )
        assert augmented[f"stable_counter_at_step_{human_step}"] == zero_index
    assert augmented["post_100_steps_used"] == 20
    assert augmented["entered_1m_after_step_100"] == 0
    assert augmented["stable_after_step_100"] == 0


def test_horizon_only_identity_checks_report_all_unchanged_contracts():
    module = _module()
    baseline, current = _horizon_identity_fixture(module)
    checks = _call_horizon_identity(module, baseline, current)
    assert isinstance(checks, dict)
    for category in (
        "checkpoint", "scenario", "order", "stable", "reward",
        "dynamics", "action", "max_correction",
    ):
        assert _identity_value(checks, category) is True


def test_horizon_identity_checks_detect_checkpoint_or_contract_drift():
    module = _module()
    baseline, current = _horizon_identity_fixture(module)
    changed_baseline = dict(baseline)
    changed_baseline["checkpoint"] = dict(baseline["checkpoint"], sha256="different")
    with pytest.raises(ValueError, match="same_checkpoint_sha256"):
        _call_horizon_identity(module, changed_baseline, current)


def test_build_horizon_recovery_rows_preserves_pair_order_and_post100_fields():
    module = _module()
    baseline_rows = []
    diagnostic_rows = []
    for index in range(1, 4):
        baseline = {
            "episode_index": index,
            "episode_id": f"episode_{index}",
            "scenario_seed": 10 + index,
            "noise_seed": 20 + index,
            "scenario_hash": f"hash-{index}",
            "max_steps": module.HORIZON_BASELINE_MAX_STEPS,
            "passed_obstacle": int(index != 2),
            "post_bypass_recovery": int(index != 2),
            "minimum_goal_distance_m": float(index),
        }
        baseline_rows.append(baseline)
        diagnostic_rows.append({
            **baseline,
            "max_steps": module.HORIZON_DIAGNOSTIC_MAX_STEPS,
            "minimum_goal_distance_m": float(index) + 0.5,
            "entered_1m_by_step_100": 0,
            "entered_1m": 0,
            "first_entered_1m_step": "",
            "stable_counter_at_step_100": 0,
            "max_stable_hold_counter": 0,
            "stable_goal": 0,
            "stable_goal_step": "",
            "goal_distance_at_step_100": 10.0,
            "goal_distance_at_step_110": 9.0,
            "goal_distance_at_step_120": 8.0,
            "actual_speed_at_step_100": 1.0,
            "actual_speed_at_step_110": 0.9,
            "actual_speed_at_step_120": 0.8,
            "radial_velocity_at_step_100": 0.5,
            "radial_velocity_at_step_110": 0.4,
            "radial_velocity_at_step_120": 0.3,
            "tangential_velocity_at_step_100": 0.2,
            "tangential_velocity_at_step_110": 0.2,
            "tangential_velocity_at_step_120": 0.1,
        })
    rows = module._build_horizon_recovery_rows(
        baseline_rows, diagnostic_rows
    )
    assert isinstance(rows, list) and len(rows) == 2
    row = rows[0]
    assert [row["episode_id"] for row in rows] == ["episode_1", "episode_3"]
    assert row["v2_baseline_min_distance_m"] == 1.0
    assert row["diagnostic_min_distance_m"] == 1.5
    assert row["goal_distance_at_step_100"] == 10.0
    assert row["goal_distance_at_step_120"] == 8.0


def test_horizon_output_layout_fails_fast_on_existing_roots(tmp_path):
    module = _module()
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"frozen")
    for condition in ("on", "paired"):
        root = tmp_path / condition
        root.mkdir()
        with pytest.raises(FileExistsError):
            module._validate_fresh_output_layout(root, condition, checkpoint)


def test_horizon_artifact_names_are_explicit_in_entrypoint():
    source = _horizon_source().read_text(encoding="utf-8")
    for filename in (
        "terminal_capture_v2_horizon120_summary.json",
        "terminal_capture_v2_horizon120_episodes.csv",
        "steps.csv",
        "episodes.csv",
        "terminal_capture_v2_horizon120_recovery_episodes.csv",
    ):
        assert filename in source


def test_horizon_tests_do_not_import_or_modify_a_controller():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("src.control" in name or "controller" in name.lower() for name in imported)
    source = _horizon_source().read_text(encoding="utf-8")
    assert '"goal_recovery_controller_modified": False' in source
    assert '"terminal_capture_v2_controller_modified": False' in source
    assert "trainer.update(" not in source
    assert "optimizer.step(" not in source


def test_horizon_test_source_never_calls_rollout_or_environment_step():
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
        "_rollout_frozen_actor_episode",
        "run_goal_recovery_correction_evaluation",
        "run_external_recovery_method_study",
    }
    assert not called_attributes & {"step", "reset", "collect_rollout"}


def _v21_identity_fixture(module):
    episode = {
        "episode_index": 1,
        "episode_id": "episode_1",
        "scenario_seed": 11,
        "noise_seed": 12,
        "scenario_hash": "episode-hash",
    }
    common = dict(
        max_steps=100,
        single_uav_stable_hold_steps=5,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
    )
    baseline_config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/v21-baseline"),
        goal_recovery_terminal_radial_conflict_arbitration_enabled=False,
        **common,
    )
    candidate_config = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/v21-candidate"),
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        **common,
    )
    baseline = {
        "checkpoint": {
            "resolved_path": "C:/tmp/frozen_actor.pt",
            "sha256": "sha",
            "policy_hash": "policy",
        },
        "scenario_manifest_hash": "scenario",
        "results": {"on": {"episodes": [episode]}},
    }
    summary = {
        "candidate_config": module._effective_correction_config(
            baseline_config, enabled=True
        ),
        "stable_distance_threshold": 1.0,
        "stable_speed_threshold": 0.5,
        "stable_hold_steps": 5,
    }
    return baseline, summary, baseline_config, candidate_config, [episode]


def test_terminal_capture_v21_flag_defaults_off_and_locks_formal100_roots():
    module = _module()
    parser = module.build_parser()
    assert parser.parse_args([]).terminal_capture_v21_validation is False
    argv = ["--terminal-capture-v21-validation"]
    resolved = module._resolve_cli_mode(parser.parse_args(argv), argv)
    assert resolved == {
        "condition": "on",
        "formal": True,
        "data_root": module.TERMINAL_CAPTURE_V21_DATA_ROOT,
        "logs_root": module.TERMINAL_CAPTURE_V21_LOG_ROOT,
        "plots_root": module.TERMINAL_CAPTURE_V21_PLOT_ROOT,
    }
    assert module.HORIZON_BASELINE_MAX_STEPS == 100


def test_terminal_capture_v21_flag_rejects_conflicting_overrides():
    module = _module()
    parser = module.build_parser()
    conflicts = (
        ["--smoke"],
        ["--paired"],
        ["--terminal-capture-v2-validation"],
        ["--terminal-capture-v2-horizon-diagnostic"],
        ["--checkpoint", "x"],
        ["--data-root", "x"],
        ["--terminal-tangential-gain", "0.9"],
        ["--terminal-capture-v2-formal-baseline-root", "x"],
        ["--condition", "off"],
    )
    for suffix in conflicts:
        argv = ["--terminal-capture-v21-validation", *suffix]
        with pytest.raises(ValueError):
            module._resolve_cli_mode(parser.parse_args(argv), argv)


def test_v21_effective_config_and_identity_allow_only_enable_delta():
    module = _module()
    baseline, summary, base_config, candidate_config, episodes = (
        _v21_identity_fixture(module)
    )
    effective = module._effective_correction_config(candidate_config, enabled=True)
    assert effective["terminal_radial_conflict_arbitration_enabled"] is True
    assert effective["terminal_radial_conflict_compensation_gain"] == 1.0
    checks = module._v21_only_identity_checks(
        baseline_evaluation=baseline,
        baseline_summary=summary,
        baseline_config=base_config,
        candidate_config=candidate_config,
        checkpoint_path=Path("C:/tmp/frozen_actor.pt"),
        checkpoint_sha256="sha",
        policy_hash="policy",
        scenario_manifest_hash="scenario",
        current_episodes=episodes,
    )
    assert checks["only_v21_arbitration_enabled_changed"] is True
    assert checks["same_max_steps"] is True
    assert checks["same_v21_compensation_gain"] is True


def test_v21_identity_rejects_any_extra_runtime_change():
    module = _module()
    baseline, summary, base_config, _candidate_config, episodes = (
        _v21_identity_fixture(module)
    )
    changed = module.FormalExperimentConfig(
        output_dir=Path("C:/tmp/v21-candidate"),
        max_steps=100,
        single_uav_stable_hold_steps=5,
        single_uav_goal_hold_reward=0.25,
        goal_recovery_terminal_capture_enabled=True,
        goal_recovery_terminal_capture_v2_enabled=True,
        goal_recovery_terminal_radial_conflict_arbitration_enabled=True,
        goal_recovery_terminal_radial_conflict_compensation_gain=1.0,
    )
    with pytest.raises(ValueError, match="v2.1-only identity mismatch"):
        module._v21_only_identity_checks(
            baseline_evaluation=baseline,
            baseline_summary=summary,
            baseline_config=base_config,
            candidate_config=changed,
            checkpoint_path=Path("C:/tmp/frozen_actor.pt"),
            checkpoint_sha256="sha",
            policy_hash="policy",
            scenario_manifest_hash="scenario",
            current_episodes=episodes,
        )


def test_v21_logging_and_artifact_schema_names_are_explicit():
    source = _horizon_source().read_text(encoding="utf-8")
    required_step_fields = (
        "terminal_v21_radial_conflict_enabled",
        "terminal_v21_radial_conflict_eligible",
        "terminal_v21_radial_conflict_active",
        "terminal_v21_goal_direction_source",
        "terminal_v21_actor_radial_delta_v_mps",
        "terminal_v21_actor_opposing_radial_mps",
        "terminal_v21_compensation_gain",
        "terminal_v21_conflict_compensation_delta_v_mps",
        "terminal_v21_conflict_compensation_norm_mps",
        "terminal_v21_conflict_compensation_radial_mps",
        "terminal_v21_raw_correction_before_conflict_mps",
        "terminal_v21_raw_correction_after_conflict_mps",
        "terminal_v21_correction_clip_ratio",
        "terminal_v21_compensation_post_cap_radial_mps",
        "terminal_v21_net_radial_pre_transform_mps",
        "terminal_v21_net_radial_post_transform_mps",
        "terminal_v21_actor_opposition_cancel_fraction",
    )
    for field in required_step_fields:
        assert f'"{field}"' in source
    for filename in (
        "terminal_capture_v21_validation_summary.json",
        "terminal_capture_v21_validation_episodes.csv",
        "steps.csv",
        "episodes.csv",
    ):
        assert filename in source


def test_v21_episode_aggregate_counts_and_means_are_pure():
    module = _module()
    rows = [
        {
            "terminal_v21_conflict_eligible_steps": 3,
            "terminal_v21_conflict_active_steps": 2,
            "mean_actor_opposing_radial_mps": 0.15,
            "mean_conflict_compensation_requested_mps": 0.15,
            "mean_conflict_compensation_post_cap_mps": 0.12,
            "mean_net_radial_pre_transform_mps": 0.05,
            "mean_net_radial_post_transform_mps": 0.04,
            "fraction_opposing_actor_cancelled": 0.8,
            "terminal_v21_correction_saturation_fraction": 0.25,
        },
        {
            "terminal_v21_conflict_eligible_steps": 0,
            "terminal_v21_conflict_active_steps": 0,
            "mean_actor_opposing_radial_mps": 0.0,
            "mean_conflict_compensation_requested_mps": 0.0,
            "mean_conflict_compensation_post_cap_mps": 0.0,
            "mean_net_radial_pre_transform_mps": 0.0,
            "mean_net_radial_post_transform_mps": 0.0,
            "fraction_opposing_actor_cancelled": 0.0,
            "terminal_v21_correction_saturation_fraction": 0.0,
        },
    ]
    aggregate = module._aggregate_episode_summaries(rows)
    assert aggregate["terminal_v21_conflict_eligible_steps"] == 3
    assert aggregate["terminal_v21_conflict_active_steps"] == 2
    assert aggregate["mean_actor_opposing_radial_mps"] == pytest.approx(0.075)
    assert aggregate["fraction_opposing_actor_cancelled"] == pytest.approx(0.4)
