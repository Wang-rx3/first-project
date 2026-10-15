import numpy as np
import pytest

from src.control.goal_recovery_correction import (
    GoalRecoveryCorrection,
    GoalRecoveryCorrectionConfig,
    compute_terminal_v22_decision,
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


def test_v22_defaults_are_opt_in_and_have_hysteretic_thresholds():
    config = GoalRecoveryCorrectionConfig()
    assert config.terminal_capture_v22_enabled is False
    assert config.terminal_hold_enter_radius_m == 0.98
    assert config.terminal_hold_release_radius_m == 1.05
    assert config.terminal_hold_restorative_enabled is True
    for kwargs in (
        {"terminal_capture_v22_enabled": 1},
        {"terminal_hold_restorative_enabled": 0},
        {"terminal_hold_enter_radius_m": -0.01},
        {"terminal_hold_release_radius_m": 1.0},
        {"terminal_hold_release_radius_m": np.inf},
    ):
        with pytest.raises(ValueError):
            GoalRecoveryCorrectionConfig(**kwargs)


def test_pure_v22_decision_covers_capture_hold_restore_and_release():
    outside = compute_terminal_v22_decision(0.99, latched_before=False)
    assert outside.mode == "inactive"
    assert outside.latched_after is False

    capture = compute_terminal_v22_decision(0.98, latched_before=False)
    assert capture.mode == "capture"
    assert capture.enter_event and capture.latched_after
    assert capture.inside_canonical_goal

    inside = compute_terminal_v22_decision(0.97, latched_before=True)
    assert inside.mode == "inside_canonical"
    assert inside.restorative_desired_radial_speed_mps == 0.0

    restorative = compute_terminal_v22_decision(1.02, latched_before=True)
    assert restorative.mode == "restorative"
    assert restorative.restore_fraction == pytest.approx(0.4)
    assert restorative.restorative_desired_radial_speed_mps == pytest.approx(0.10)

    released = compute_terminal_v22_decision(1.05, latched_before=True)
    assert released.mode == "released_capture"
    assert released.release_event and released.latched_after is False


def test_v22_controller_latches_at_enter_and_uses_fixed_restorative_profile():
    controller = _controller()
    assert _compute(controller, 0.99).terminal_v22_hold_latched is False
    captured = _compute(controller, 0.98)
    assert captured.terminal_v22_hold_enter_event
    assert captured.desired_radial_velocity_mps == 0.0
    assert captured.terminal_v22_inside_canonical_goal

    for distance, expected in ((0.97, 0.0), (1.01, 0.05), (1.02, 0.10), (1.04, 0.20)):
        result = _compute(controller, distance)
        assert result.desired_radial_velocity_mps == pytest.approx(expected)
        if distance > 1.0:
            assert result.terminal_v22_restorative_active
            assert result.terminal_v22_restorative_radial_error_mps == pytest.approx(expected)


def test_v22_release_is_same_call_and_cross_episode_reset_isolated():
    controller = _controller()
    _compute(controller, 0.98)
    released = _compute(controller, 1.05)
    assert released.terminal_v22_hold_release_event
    assert released.terminal_v22_hold_latched is False
    assert released.desired_radial_velocity_mps > 0.0
    assert released.terminal_v22_hold_mode == "released_capture"

    _compute(controller, 0.98)
    controller.reset()
    assert controller.terminal_hold_latched is False
    fresh = _compute(controller, 0.99)
    assert fresh.terminal_v22_hold_enter_event is False
    assert fresh.terminal_v22_hold_latched is False


def test_v22_gate_closed_is_exact_v2_path_and_new_fields_are_neutral():
    kwargs = dict(distance=3.0, velocity=(0.0, 1.0, 0.0))
    v2 = _controller(terminal_capture_v22_enabled=False)
    v22 = _controller(terminal_capture_v22_enabled=True, terminal_radial_conflict_arbitration_enabled=False)
    left = _compute(v2, **kwargs)
    right = _compute(v22, **kwargs)
    for name in (
        "correction_delta_v_mps",
        "radial_correction_delta_v_mps",
        "tangential_correction_delta_v_mps",
        "desired_radial_velocity_mps",
        "terminal_raw_correction_delta_v_mps",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name)), name
    assert right.terminal_v22_enabled is True
    assert right.terminal_v22_hold_mode == "inactive"
    assert right.terminal_v22_hold_enter_event is False
    assert right.terminal_v22_restorative_active is False


def test_v22_restoration_reaches_v21_arbitration_and_shared_cap():
    controller = _controller(max_delta_v_mps=0.6)
    _compute(controller, 0.98)
    result = _compute(controller, 1.02, actor_raw_delta_v_mps=[-2.0, 0.0, 0.0])
    assert result.terminal_v22_restorative_radial_error_mps > 0.0
    assert result.terminal_v21_radial_conflict_active
    assert np.linalg.norm(result.correction_delta_v_mps) <= 0.6 + 1.0e-12

    actor = np.asarray([-0.2, 0.3, 0.0])
    before = actor.copy()
    calls = []
    fused = fuse_actor_action(
        actor,
        result,
        lambda value: calls.append(value.copy()) or {"constrained_delta_v_mps": value},
    )
    assert len(calls) == 1
    assert np.array_equal(actor, before)
    assert np.allclose(
        fused.corrected_delta_v_pre_action_transform_mps,
        actor + result.correction_alpha * result.correction_delta_v_mps,
    )


def test_invalid_motion_is_finite_zero_and_does_not_mutate_latch():
    controller = _controller()
    _compute(controller, 0.98)
    invalid = _compute(controller, np.nan)
    assert not invalid.active
    assert np.array_equal(invalid.correction_delta_v_mps, np.zeros(3))
    assert controller.terminal_hold_latched is True
    invalid_inf = _compute(controller, np.inf)
    assert not invalid_inf.active
    assert controller.terminal_hold_latched is True


def test_v22_no_latch_at_0_99():
    result = _compute(_controller(), 0.99)
    assert result.terminal_v22_hold_latched is False
    assert result.terminal_v22_hold_enter_event is False
    assert result.terminal_v22_hold_mode == "inactive"


def test_v22_latches_at_0_98():
    result = _compute(_controller(), 0.98)
    assert result.terminal_v22_hold_latched is True
    assert result.terminal_v22_hold_latched_before is False
    assert result.terminal_v22_hold_enter_event is True
    assert result.terminal_v22_hold_mode == "capture"


def test_v22_latched_0_97_has_zero_desired_speed():
    result = _compute(_controller(), 0.97)
    assert result.terminal_v22_hold_latched is True
    assert result.desired_radial_velocity_mps == 0.0
    assert result.terminal_v22_hold_mode == "capture"


def test_v22_latched_1_00_is_inside_canonical_goal_and_zero():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.00)
    assert result.terminal_v22_hold_mode == "inside_canonical"
    assert result.terminal_v22_inside_canonical_goal
    assert result.desired_radial_velocity_mps == 0.0


def test_v22_restorative_1_01_is_0_05_mps():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.01)
    assert result.terminal_v22_hold_mode == "restorative"
    assert result.terminal_v22_restore_fraction == pytest.approx(0.2)
    assert result.desired_radial_velocity_mps == pytest.approx(0.05)


def test_v22_restorative_1_02_is_0_10_mps():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.02)
    assert result.desired_radial_velocity_mps == pytest.approx(0.10)
    assert result.terminal_v22_restorative_desired_radial_speed_mps == pytest.approx(0.10)


def test_v22_restorative_1_04_is_0_20_mps():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.04)
    assert result.desired_radial_velocity_mps == pytest.approx(0.20)
    assert result.terminal_v22_restorative_radial_error_mps == pytest.approx(0.20)
    assert result.terminal_v22_restorative_delta_v_mps[0] == pytest.approx(0.16)


def test_v22_1_05_release_preserves_same_call_normal_v2_speed():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.05)
    baseline = _compute(_controller(terminal_capture_v22_enabled=False), 1.05)
    assert result.terminal_v22_hold_release_event is True
    assert result.terminal_v22_hold_mode == "released_capture"
    assert result.terminal_v22_hold_latched is False
    assert result.desired_radial_velocity_mps == baseline.desired_radial_velocity_mps


def test_v22_latch_persists_across_threshold_crossing():
    controller = _controller()
    _compute(controller, 0.98)
    restored = _compute(controller, 1.02)
    assert restored.terminal_v22_hold_latched is True
    inside_again = _compute(controller, 0.99)
    assert inside_again.terminal_v22_hold_latched is True
    assert inside_again.terminal_v22_hold_mode == "inside_canonical"
    assert inside_again.desired_radial_velocity_mps == 0.0


def test_v22_release_then_recapture_is_explicit():
    controller = _controller()
    _compute(controller, 0.98)
    _compute(controller, 1.05)
    recaptured = _compute(controller, 0.98)
    assert recaptured.terminal_v22_hold_enter_event
    assert recaptured.terminal_v22_hold_latched


def test_v22_restorative_disabled_keeps_zero_inside_band():
    controller = _controller(terminal_hold_restorative_enabled=False)
    _compute(controller, 0.98)
    result = _compute(controller, 1.02)
    assert result.terminal_v22_hold_mode == "restorative"
    assert result.terminal_v22_restorative_active is False
    assert result.terminal_v22_restorative_desired_radial_speed_mps == 0.0
    assert result.desired_radial_velocity_mps == 0.0


def test_v22_uses_existing_terminal_radial_floor_config():
    controller = _controller(terminal_radial_speed_floor_mps=0.5)
    _compute(controller, 0.98)
    result = _compute(controller, 1.02)
    assert result.desired_radial_velocity_mps == pytest.approx(0.2)
    assert result.terminal_v22_hold_release_radius_m == pytest.approx(1.05)


def test_v22_shared_cap_is_preserved_after_actor_compensation():
    controller = _controller(max_delta_v_mps=0.6)
    _compute(controller, 0.98)
    result = _compute(controller, 1.02, actor_raw_delta_v_mps=[-2.0, 0.0, 0.0])
    assert result.terminal_correction_clipped
    assert np.linalg.norm(result.correction_delta_v_mps) <= 0.6 + 1.0e-12


def test_v22_actor_tangential_component_and_raw_path_are_preserved():
    kwargs = dict(distance=1.02, velocity=(0.0, 1.5, 0.0))
    base_controller = _controller()
    _compute(base_controller, 0.98)
    base = _compute(base_controller, **kwargs, actor_raw_delta_v_mps=[0.0, 0.0, 0.0])
    conflict_controller = _controller()
    _compute(conflict_controller, 0.98)
    conflict = _compute(conflict_controller, **kwargs, actor_raw_delta_v_mps=[-0.2, 0.3, 0.0])
    assert np.array_equal(conflict.tangential_correction_delta_v_mps, base.tangential_correction_delta_v_mps)
    assert np.array_equal(conflict.terminal_v21_raw_correction_before_conflict_mps[1:], base.terminal_v21_raw_correction_before_conflict_mps[1:])
    assert conflict.terminal_v21_radial_conflict_active


def test_v22_action_transform_is_called_once_after_fusion():
    controller = _controller()
    _compute(controller, 0.98)
    result = _compute(controller, 1.02)
    actor = np.asarray([0.1, 0.2, 0.0])
    calls = []
    fused = fuse_actor_action(
        actor,
        result,
        lambda value: calls.append(value.copy()) or {"constrained_delta_v_mps": value},
    )
    assert len(calls) == 1
    assert np.allclose(calls[0], actor + result.correction_delta_v_mps)
    assert fused.transform_applied


def test_v22_reset_clears_latch_without_leaking_between_controllers():
    first = _controller()
    second = _controller()
    _compute(first, 0.98)
    assert first.terminal_hold_latched is True
    assert second.terminal_hold_latched is False
    second_result = _compute(second, 0.99)
    assert second_result.terminal_v22_hold_latched is False
    first.reset()
    assert first.terminal_hold_latched is False


def test_v22_nan_inf_distance_helper_is_rejected_without_a_gain_escape():
    for distance in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            compute_terminal_v22_decision(distance, latched_before=False)
    with pytest.raises(ValueError):
        compute_terminal_v22_decision(1.02, latched_before=False, radial_speed_floor_mps=np.nan)


def test_v22_default_v2_constants_and_hold_config_are_stable():
    config = GoalRecoveryCorrectionConfig()
    assert config.terminal_effective_deceleration_mps2 == pytest.approx(1.697726)
    assert config.terminal_stopping_margin_m == pytest.approx(1.328009)
    assert config.terminal_radial_speed_floor_mps == pytest.approx(0.25)
    assert config.terminal_hold_enter_radius_m <= config.goal_threshold_m < config.terminal_hold_release_radius_m


def test_v22_result_reports_neutral_telemetry_when_gate_is_closed():
    result = _compute(
        _controller(
            terminal_capture_v22_enabled=True,
            terminal_radial_conflict_arbitration_enabled=False,
        ),
        0.98,
    )
    assert result.terminal_v22_hold_latched is False
    assert result.terminal_v22_hold_enter_event is False
    assert result.terminal_v22_hold_release_event is False
    assert result.terminal_v22_restorative_delta_v_mps.tolist() == [0.0, 0.0, 0.0]


def test_v22_source_has_no_environment_step_or_rollout_dependency():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "control" / "goal_recovery_correction.py").read_text(encoding="utf-8")
    assert "environment.step" not in source
    assert "rollout" not in source
