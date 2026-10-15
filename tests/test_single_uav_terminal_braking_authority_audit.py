"""Pure offline tests for the terminal braking-authority audit."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "experiments" / "single_uav_terminal_braking_authority_audit.py"
)
_SPEC = importlib.util.spec_from_file_location("terminal_braking_authority_audit", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _row(**updates):
    row = {
        "episode_id": "episode_1",
        "episode_index": "1",
        "step": "3",
        "post_bypass_recovery": "1",
        "terminal_capture_active": "1",
        "radial_velocity_error_mps": "-1.0",
        "radial_braking_delta_v_mps": "[-0.8,0.0,0.0]",
        "terminal_tangential_damping_delta_v_mps": "[0.0,-0.6,0.0]",
        "terminal_raw_correction_delta_v_mps": "[-0.8,-0.6,0.0]",
        "terminal_raw_correction_norm_mps": "1.0",
        "terminal_correction_clipped": "1",
        "terminal_correction_clip_ratio": "0.6",
        "goal_recovery_correction_delta_v_mps": "[-0.48,-0.36,0.0]",
        "goal_recovery_applied_delta_v_mps": "[-0.24,-0.18,0.0]",
        "actor_raw_delta_v_mps": "[0.1,0.0,0.0]",
        "corrected_delta_v_pre_transform_mps": "[-0.14,-0.18,0.0]",
        "corrected_delta_v_pre_safety_mps": "[-0.14,-0.18,0.0]",
        "applied_delta_v_mps": "[-0.12,-0.18,0.0]",
        "safety_projection_applied": "0",
        "boundary_projection_applied": "0",
        "action_clipped": "0",
        "collision": "0",
        "goal_distance_m": "4.0",
        "goal_distance_after_m": "3.5",
        "current_radial_velocity_mps": "3.0",
        "current_tangential_velocity_mps": "1.0",
        "terminal_desired_radial_velocity_mps": "2.0",
        "radial_propulsion_delta_v_mps": "[0.0,0.0,0.0]",
        "radial_propulsion_scale": "0.5",
        "terminal_blend": "0.5",
        "post_step_actual_radial_velocity_mps": "2.8",
        "post_step_actual_tangential_velocity_mps": "0.8",
        "post_step_actual_speed_mps": "2.91",
    }
    row.update({key: str(value) for key, value in updates.items()})
    return row


def test_vector_string_parsing_is_finite_and_three_dimensional():
    assert audit.parse_vector("[1, 2.5, -3]") == (1.0, 2.5, -3.0)
    assert audit.parse_vector("[1, 2]") is None
    assert audit.parse_vector("[1, NaN, 3]") is None
    assert audit.parse_vector("not a vector") is None


def test_goal_direction_reconstruction_uses_outward_braking_vector():
    assert audit.recover_goal_direction((-2.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))


def test_radial_projection_sign_is_positive_for_braking():
    direction = (1.0, 0.0, 0.0)
    assert audit.braking_projection((-0.3, 0.1, 0.0), direction) == pytest.approx(0.3)
    assert audit.braking_projection((0.3, 0.1, 0.0), direction) == 0.0


def test_stage_mapping_and_retention_are_exact():
    sample = audit.derive_braking_sample(_row())
    assert sample is not None
    assert sample["requested_radial_braking_mps"] == pytest.approx(0.8)
    assert sample["terminal_raw_combined_radial_braking_mps"] == pytest.approx(0.8)
    assert sample["post_correction_cap_radial_braking_mps"] == pytest.approx(0.48)
    assert sample["applied_correction_radial_braking_mps"] == pytest.approx(0.24)
    assert sample["pre_transform_radial_braking_mps"] == pytest.approx(0.14)
    assert sample["post_transform_radial_braking_mps"] == pytest.approx(0.14)
    assert sample["final_applied_radial_braking_mps"] == pytest.approx(0.12)
    assert sample["correction_cap_retention"] == pytest.approx(0.6)
    assert sample["transform_retention"] == pytest.approx(1.0)
    assert sample["final_retention"] == pytest.approx(0.15)
    assert sample["command_side_effective_radial_deceleration_mps2"] == pytest.approx(0.6)


def test_actor_classification_does_not_assume_conflict():
    assert audit.classify_actor_radial(-0.1) == "assist"
    assert audit.classify_actor_radial(0.1) == "oppose"
    assert audit.classify_actor_radial(0.0) == "neutral"
    sample = audit.derive_braking_sample(_row(actor_raw_delta_v_mps="[-0.1,0,0]"))
    assert sample is not None and sample["actor_relation"] == "assist"


def test_zero_vector_and_zero_denominator_do_not_divide_by_zero():
    assert audit.recover_goal_direction((0.0, 0.0, 0.0)) is None
    assert audit.safe_ratio(1.0, 0.0) is None
    row = _row(radial_braking_delta_v_mps="[0,0,0]", radial_velocity_error_mps="-1")
    assert audit.derive_braking_sample(row) is None


def test_nan_values_fail_closed():
    assert audit.finite_float("nan") is None
    with pytest.raises(ValueError):
        audit.derive_braking_sample(_row(applied_delta_v_mps="[NaN,0,0]"))


def test_episode_grouping_preserves_cluster_level_statistics():
    samples = []
    for episode, values in {"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]}.items():
        for index, value in enumerate(values):
            sample = audit.derive_braking_sample(
                _row(
                    episode_id=episode,
                    step=index,
                    applied_delta_v_mps=f"[-{value},0,0]",
                )
            )
            assert sample is not None
            samples.append(sample)
    grouped = audit.aggregate_episode_rows(samples)
    assert len(grouped) == 2
    assert grouped[0]["saturated_clean_braking_samples"] == 3
    assert grouped[0]["eligible_for_conservative_estimate"] == 1
    assert grouped[0]["saturated_clean_effective_deceleration_median_mps2"] == pytest.approx(1.0)
    assert grouped[1]["saturated_clean_effective_deceleration_median_mps2"] == pytest.approx(2.5)


def test_quantile_uses_linear_interpolation():
    assert audit.quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    stats = audit.descriptive_statistics([1.0, 2.0, 3.0, 4.0])
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p25"] == pytest.approx(1.75)


def test_conservative_candidates_do_not_use_maximum():
    values = [1.0, 1.1, 1.2, 1.3, 10.0]
    p10 = audit.quantile(values, 0.10)
    p25 = audit.quantile(values, 0.25)
    lower = audit.robust_median_lower_bound(values, replicates=1000, seed=7)
    assert p10 is not None and p10 < max(values)
    assert p25 is not None and p25 < max(values)
    assert lower is not None and lower < max(values)


def test_existing_output_root_fails_before_input_read(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        audit.run_audit(tmp_path / "missing-input", output)


def test_source_has_no_environment_step_or_runtime_imports():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports |= {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith(("src.simulation", "src.control", "src.learning")) for name in imports)
    step_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
    ]
    assert step_calls == []


def test_audit_source_does_not_reference_or_modify_controller_file():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("src.control") for name in imported)


def test_cli_defaults_point_to_independent_existing_log_and_new_output_roots():
    args = audit.build_parser().parse_args([])
    assert args.terminal_root == audit.DEFAULT_TERMINAL_ROOT
    assert args.output_root == audit.DEFAULT_OUTPUT_ROOT
    assert args.terminal_root != args.output_root
