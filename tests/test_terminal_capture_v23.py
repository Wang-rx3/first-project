"""Pure contracts for the default-off terminal-capture v2.3 guard."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.control.goal_recovery_correction import (
    GoalRecoveryCorrection,
    GoalRecoveryCorrectionConfig,
    compute_terminal_v23_decision,
    fuse_actor_action,
)


def _controller(**overrides):
    values = dict(
        enabled=True,
        max_delta_v_mps=10.0,
        alpha_ramp_enabled=False,
        terminal_capture_v2_enabled=True,
        terminal_radial_conflict_arbitration_enabled=True,
        terminal_capture_v22_enabled=True,
        terminal_capture_v23_enabled=True,
    )
    values.update(overrides)
    return GoalRecoveryCorrection(GoalRecoveryCorrectionConfig(**values))


def _compute(controller, distance, velocity=(0.0, 0.0, 0.0), **overrides):
    values = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=velocity,
        goal_position_m=[distance, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        action_provenance="evaluation",
    )
    values.update(overrides)
    return controller.compute(**values)


def test_01_default_off():
    config = GoalRecoveryCorrectionConfig()
    assert config.terminal_capture_v23_enabled is False
    assert config.terminal_hold_target_radius_m == pytest.approx(0.96)


def test_02_target_relation_is_strict():
    with pytest.raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_hold_target_radius_m=0.98)
    with pytest.raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_hold_target_radius_m=1.0)


def test_03_non_boolean_flag_rejected():
    with pytest.raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_capture_v23_enabled=1)


def test_04_formula_below_target_zero():
    decision = compute_terminal_v23_decision(0.95, hold_latched=True)
    assert decision.desired_radial_speed_mps == 0.0
    assert not decision.guard_active


def test_05_formula_at_target_zero():
    decision = compute_terminal_v23_decision(0.96, hold_latched=True)
    assert decision.desired_radial_speed_mps == 0.0
    assert not decision.guard_active


def test_06_formula_preboundary():
    decision = compute_terminal_v23_decision(0.97, hold_latched=True)
    assert decision.guard_active and decision.preboundary_active
    assert decision.desired_radial_speed_mps == pytest.approx(0.25 * 0.01 / 0.09)


def test_07_formula_boundary_continuity():
    left = compute_terminal_v23_decision(1.0 - 1e-8, hold_latched=True)
    right = compute_terminal_v23_decision(1.0, hold_latched=True)
    assert abs(left.desired_radial_speed_mps - right.desired_radial_speed_mps) < 1e-7


def test_08_formula_postboundary_flag():
    decision = compute_terminal_v23_decision(1.01, hold_latched=True)
    assert decision.guard_active and decision.postboundary_active
    assert not decision.preboundary_active


def test_09_formula_reaches_cap_before_release():
    decision = compute_terminal_v23_decision(1.049999, hold_latched=True)
    assert decision.desired_radial_speed_mps < 0.25
    assert decision.desired_radial_speed_mps >= 0.0


def test_10_formula_release_not_guard():
    decision = compute_terminal_v23_decision(1.05, hold_latched=True)
    assert not decision.guard_active
    assert decision.desired_radial_speed_mps == 0.0


def test_11_formula_unlatched_is_inactive():
    decision = compute_terminal_v23_decision(0.97, hold_latched=False)
    assert not decision.guard_active
    assert decision.desired_radial_speed_mps == 0.0


def test_12_formula_band_distance_clamps():
    assert compute_terminal_v23_decision(0.90).distance_to_target_band_m == 0.0
    assert compute_terminal_v23_decision(1.01).distance_to_target_band_m == pytest.approx(0.05)


def test_13_formula_nan_rejected():
    with pytest.raises(ValueError):
        compute_terminal_v23_decision(np.nan)


def test_14_formula_radius_validation():
    with pytest.raises(ValueError):
        compute_terminal_v23_decision(1.0, target_radius_m=1.01)


def test_15_gate_requires_v2():
    result = _compute(_controller(terminal_capture_v2_enabled=False), 0.97)
    assert not result.terminal_v23_guard_active
    assert result.terminal_v23_guard_fraction == 0.0


def test_16_gate_requires_v21():
    result = _compute(_controller(terminal_radial_conflict_arbitration_enabled=False), 0.97)
    assert not result.terminal_v23_guard_active


def test_17_gate_requires_v22():
    result = _compute(_controller(terminal_capture_v22_enabled=False), 0.97)
    assert not result.terminal_v23_guard_active


def test_18_controller_enters_v22_before_guard():
    controller = _controller()
    captured = _compute(controller, 0.98)
    assert captured.terminal_v22_hold_enter_event
    result = _compute(controller, 0.97)
    assert result.terminal_v23_guard_active


def test_19_controller_preboundary_desired():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 0.97)
    assert result.terminal_v23_preboundary_active
    assert result.desired_radial_velocity_mps == pytest.approx(0.25 / 9.0)


def test_20_controller_postboundary_desired():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.01)
    assert result.terminal_v23_postboundary_active
    assert result.desired_radial_velocity_mps == pytest.approx(0.25 * 0.05 / 0.09)


def test_21_release_is_same_call_and_v23_not_active():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.05)
    assert result.terminal_v22_hold_release_event
    assert not result.terminal_v23_guard_active
    assert result.terminal_v22_hold_latched is False


def test_22_v23_off_is_exact_v22_path():
    kwargs = dict(distance=1.02, velocity=(0.0, 1.2, 0.0))
    off = _controller(terminal_capture_v23_enabled=False)
    legacy_v22 = GoalRecoveryCorrection(
        GoalRecoveryCorrectionConfig(
            enabled=True,
            max_delta_v_mps=10.0,
            alpha_ramp_enabled=False,
            terminal_capture_v2_enabled=True,
            terminal_radial_conflict_arbitration_enabled=True,
            terminal_capture_v22_enabled=True,
        )
    )
    _compute(off, 0.98)
    _compute(legacy_v22, 0.98)
    left, right = _compute(off, **kwargs), _compute(legacy_v22, **kwargs)
    for name in ("correction_delta_v_mps", "terminal_raw_correction_delta_v_mps", "tangential_correction_delta_v_mps"):
        assert np.array_equal(getattr(left, name), getattr(right, name))


def test_23_v21_arbitration_is_available_preboundary():
    controller = _controller()
    _compute(controller, 0.98)
    actor = np.asarray([-2.0, 0.0, 0.0])
    before = actor.copy()
    result = _compute(controller, 0.97, actor_raw_delta_v_mps=actor)
    assert result.terminal_v23_guard_active
    assert result.terminal_v21_radial_conflict_eligible
    assert result.terminal_v21_radial_conflict_active
    assert np.array_equal(actor, before)


def test_24_existing_shared_cap_applies():
    controller = _controller(max_delta_v_mps=0.6)
    _compute(controller, 0.98)
    result = _compute(controller, 0.97, actor_raw_delta_v_mps=[-2.0, 0.0, 0.0])
    assert np.linalg.norm(result.correction_delta_v_mps) <= 0.6 + 1e-12


def test_25_actor_tangential_component_is_not_modified():
    baseline_controller = _controller()
    conflict_controller = _controller()
    _compute(baseline_controller, 0.98)
    _compute(conflict_controller, 0.98)
    baseline = _compute(
        baseline_controller, 0.97, velocity=(0.0, 1.5, 0.0),
        actor_raw_delta_v_mps=[0.0, 0.3, 0.0],
    )
    conflict = _compute(
        conflict_controller, 0.97, velocity=(0.0, 1.5, 0.0),
        actor_raw_delta_v_mps=[-0.2, 0.3, 0.0],
    )
    assert conflict.terminal_v21_radial_conflict_active
    assert np.array_equal(
        conflict.tangential_correction_delta_v_mps,
        baseline.tangential_correction_delta_v_mps,
    )


def test_26_action_transform_still_runs_once():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 0.97)
    calls = []
    fuse_actor_action([0.1, 0.2, 0.0], result, lambda value: calls.append(value.copy()) or {"constrained_delta_v_mps": value})
    assert len(calls) == 1


def test_27_radial_error_is_signed_desired_minus_current():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 0.97, velocity=(0.05, 0.0, 0.0))
    assert result.terminal_v23_radial_error_mps == pytest.approx(
        result.terminal_v23_desired_radial_speed_mps - 0.05
    )


def test_28_floor_config_is_used_as_guard_cap():
    controller = _controller(terminal_radial_speed_floor_mps=0.4)
    _compute(controller, 0.98)
    result = _compute(controller, 1.04)
    assert result.terminal_v23_desired_radial_speed_mps == pytest.approx(0.4 * 0.08 / 0.09)


def test_29_reset_clears_v22_and_v23_state():
    controller = _controller()
    _compute(controller, 0.98)
    controller.reset()
    result = _compute(controller, 0.99)
    assert not result.terminal_v22_hold_latched
    assert not result.terminal_v23_guard_active


def test_30_invalid_motion_does_not_mutate_latch():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, np.nan)
    assert not result.active
    assert controller.terminal_hold_latched


def test_31_source_has_no_rollout_or_environment_step_dependency():
    source = (Path(__file__).resolve().parents[1] / "src" / "control" / "goal_recovery_correction.py").read_text(encoding="utf-8")
    assert "environment.step" not in source
    assert "rollout" not in source


def test_32_result_telemetry_types_are_exact():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 0.97)
    assert isinstance(result.terminal_v23_enabled, bool)
    assert isinstance(result.terminal_v23_guard_active, bool)
    assert isinstance(result.terminal_v23_preboundary_active, bool)
    assert isinstance(result.terminal_v23_postboundary_active, bool)
    assert isinstance(result.terminal_v23_guard_fraction, float)


def test_33_frozen_v2_v22_and_stable_constants_are_unchanged():
    config = GoalRecoveryCorrectionConfig()
    assert config.goal_threshold_m == pytest.approx(1.0)
    assert config.terminal_hold_enter_radius_m == pytest.approx(0.98)
    assert config.terminal_hold_release_radius_m == pytest.approx(1.05)
    assert config.terminal_effective_deceleration_mps2 == pytest.approx(1.697726)
    assert config.terminal_stopping_margin_m == pytest.approx(1.328009)
    assert config.max_delta_v_mps == pytest.approx(0.6)


@pytest.mark.parametrize("distance", (0.97, 0.98, 0.99))
def test_34_preboundary_positive_error_restores_existing_propulsion(distance):
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    result = _compute(controller, distance, velocity=(-0.1, 0.0, 0.0))

    assert result.terminal_v22_hold_latched
    assert result.terminal_v23_guard_active
    assert result.terminal_v23_preboundary_active
    assert result.terminal_v23_radial_error_mps > 1.0e-9
    assert result.radial_propulsion_scale == pytest.approx(1.0)
    assert np.dot(result.radial_propulsion_delta_v_mps, result.goal_direction) > 0.0
    assert np.array_equal(result.radial_braking_delta_v_mps, np.zeros(3))
    assert result.terminal_v23_positive_radial_error
    assert result.terminal_v23_radial_propulsion_allowed
    assert not result.terminal_v23_radial_propulsion_suppressed
    assert result.terminal_v23_radial_propulsion_suppression_reason == "none"
    assert result.terminal_v23_authority_active
    assert result.terminal_v23_preboundary_authority_repair_active
    assert result.terminal_v23_executed_propulsion_radial_mps == pytest.approx(
        result.terminal_v23_requested_propulsion_radial_mps
    )


def test_35_preboundary_negative_error_keeps_propulsion_braking_exclusive():
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    result = _compute(controller, 0.97, velocity=(0.2, 0.0, 0.0))

    assert result.terminal_v23_radial_error_mps < 0.0
    assert np.array_equal(result.radial_propulsion_delta_v_mps, np.zeros(3))
    assert np.dot(result.radial_braking_delta_v_mps, result.goal_direction) < 0.0
    assert not result.terminal_v23_radial_propulsion_allowed
    assert not result.terminal_v23_authority_active
    assert result.terminal_v23_radial_propulsion_suppression_reason == (
        "radial_error_non_positive"
    )


@pytest.mark.parametrize("distance", (0.95, 0.96))
def test_36_deep_hold_keeps_zero_target_without_repair_authority(distance):
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    result = _compute(controller, distance, velocity=(-0.1, 0.0, 0.0))

    assert result.terminal_v23_desired_radial_speed_mps == 0.0
    assert not result.terminal_v23_guard_active
    assert not result.terminal_v23_authority_active
    assert np.array_equal(result.radial_propulsion_delta_v_mps, np.zeros(3))


def test_37_postboundary_restorative_propulsion_remains_available():
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    result = _compute(controller, 1.01, velocity=(-0.1, 0.0, 0.0))

    assert result.terminal_v23_postboundary_active
    assert not result.terminal_v23_preboundary_authority_repair_active
    assert result.terminal_v2_anti_stall_active
    assert result.radial_propulsion_scale == pytest.approx(1.0)
    assert np.dot(result.radial_propulsion_delta_v_mps, result.goal_direction) > 0.0


def test_38_v21_compensation_preserves_v23_tracking_authority():
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    actor = np.asarray([-0.2, 0.3, 0.0])
    before = actor.copy()
    result = _compute(
        controller,
        0.97,
        velocity=(-0.1, 1.5, 0.0),
        actor_raw_delta_v_mps=actor,
    )

    assert result.terminal_v23_authority_active
    assert result.terminal_v23_preboundary_authority_repair_active
    assert result.terminal_v21_radial_conflict_active
    assert result.terminal_v23_net_tracking_radial_before_v21_mps > 0.0
    assert (
        result.terminal_v23_net_tracking_radial_after_v21_mps
        > result.terminal_v23_net_tracking_radial_before_v21_mps
    )
    assert np.array_equal(actor, before)


def test_39_nonopposing_actor_needs_no_compensation_but_keeps_propulsion():
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    result = _compute(
        controller,
        0.97,
        velocity=(-0.1, 0.0, 0.0),
        actor_raw_delta_v_mps=[0.2, 0.0, 0.0],
    )

    assert result.terminal_v23_authority_active
    assert result.terminal_v23_preboundary_authority_repair_active
    assert not result.terminal_v21_radial_conflict_active
    assert result.terminal_v23_net_tracking_radial_before_v21_mps == pytest.approx(
        result.terminal_v23_net_tracking_radial_after_v21_mps
    )


def test_40_stage0_defect_fixture_now_has_explained_nonzero_propulsion():
    controller = _controller(terminal_capture_enabled=True)
    _compute(controller, 0.98)
    distance = 0.9781516024978172
    desired = 0.25 * (distance - 0.96) / (1.05 - 0.96)
    radial_error = 0.1725111916
    current_radial = desired - radial_error
    result = _compute(
        controller,
        distance,
        velocity=(current_radial, 0.0, 0.0),
        actor_raw_delta_v_mps=[-0.1188, 0.0, 0.0],
    )

    assert result.terminal_v23_preboundary_active
    assert result.terminal_v23_radial_error_mps == pytest.approx(radial_error)
    assert result.terminal_v23_requested_propulsion_radial_mps > 0.0
    assert result.terminal_v23_executed_propulsion_radial_mps > 0.0
    assert not result.terminal_v23_radial_propulsion_suppressed
    assert result.terminal_v23_radial_propulsion_suppression_reason == "none"


def test_41_shared_cap_and_transform_remain_downstream_of_repair():
    controller = _controller(
        terminal_capture_enabled=True,
        max_delta_v_mps=0.6,
    )
    _compute(controller, 0.98)
    result = _compute(
        controller,
        0.99,
        velocity=(-1.0, 2.0, 0.0),
        actor_raw_delta_v_mps=[-2.0, 0.0, 0.0],
    )
    assert result.terminal_v23_authority_active
    assert result.terminal_correction_clipped
    assert np.linalg.norm(result.correction_delta_v_mps) <= 0.6 + 1e-12
    assert 0.0 < result.terminal_v23_executed_propulsion_radial_mps <= 0.6
    assert (
        result.terminal_v23_executed_propulsion_radial_mps
        < result.terminal_v23_requested_propulsion_radial_mps
    )
    calls = []
    fused = fuse_actor_action(
        [0.1, 0.2, 0.0],
        result,
        lambda value: calls.append(value.copy())
        or {"constrained_delta_v_mps": 0.5 * value},
    )
    assert len(calls) == 1
    assert fused.transform_applied
    assert np.allclose(fused.corrected_delta_v_pre_safety_mps, 0.5 * calls[0])


def test_42_v23_off_keeps_new_authority_fields_neutral():
    controller = _controller(
        terminal_capture_enabled=True,
        terminal_capture_v23_enabled=False,
    )
    _compute(controller, 0.98)
    result = _compute(controller, 0.97, velocity=(-0.1, 0.0, 0.0))
    assert not result.terminal_v23_positive_radial_error
    assert not result.terminal_v23_radial_propulsion_allowed
    assert not result.terminal_v23_radial_propulsion_suppressed
    assert not result.terminal_v23_authority_active
    assert not result.terminal_v23_preboundary_authority_repair_active
    assert result.terminal_v23_radial_propulsion_suppression_reason == "not_v23_guard"
