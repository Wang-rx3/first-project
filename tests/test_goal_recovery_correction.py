import inspect
import math
from pathlib import Path

import numpy as np

from src.control.goal_recovery_correction import (
    GoalRecoveryCorrection,
    GoalRecoveryCorrectionConfig,
    compute_desired_goal_speed,
    compute_terminal_blend,
    compute_terminal_desired_radial_speed,
    compute_terminal_v2_envelope,
    decompose_goal_relative_motion,
    fuse_actor_action,
)


def _active(**overrides):
    values = dict(enabled=True, max_delta_v_mps=10.0, alpha_ramp_enabled=False)
    values.update(overrides)
    return GoalRecoveryCorrection(GoalRecoveryCorrectionConfig(**values))


def _compute(controller, velocity, **overrides):
    values = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=velocity,
        goal_position_m=[10.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        action_provenance="evaluation",
    )
    values.update(overrides)
    return controller.compute(**values)


def test_matching_radial_velocity_has_near_zero_correction():
    desired = compute_desired_goal_speed(10.0, max_acceleration_mps2=3.0, cruise_speed_mps=6.0)
    result = _compute(_active(), [desired, 0.0, 0.0])
    assert np.allclose(result.correction_delta_v_mps, 0.0)


def test_low_or_negative_radial_velocity_is_corrected_toward_goal():
    controller = _active()
    assert _compute(controller, [0.0, 0.0, 0.0]).radial_correction_delta_v_mps[0] > 0.0
    assert _compute(controller, [-2.0, 0.0, 0.0]).radial_correction_delta_v_mps[0] > 0.0


def test_tangential_correction_opposes_tangential_velocity():
    result = _compute(_active(), [0.0, 2.0, -1.0])
    tangent = result.current_tangential_velocity_vector_mps
    assert np.dot(result.tangential_correction_delta_v_mps, tangent) < 0.0
    assert np.isclose(result.tangential_correction_delta_v_mps[0], 0.0)


def test_off_and_ineligible_paths_are_exact_and_skip_transform():
    calls = []
    actor = np.asarray([0.1, -0.2, 0.3])
    off = _compute(GoalRecoveryCorrection(GoalRecoveryCorrectionConfig()), [0.0, 0.0, 0.0])
    fused = fuse_actor_action(actor, off, lambda value: calls.append(value))
    assert np.array_equal(fused.corrected_delta_v_pre_safety_mps, actor)
    assert not calls
    ineligible = _compute(_active(), [0.0, 0.0, 0.0], release_latched=False)
    assert not ineligible.active and ineligible.correction_alpha == 0.0


def test_recovery_alpha_is_positive_and_correction_is_norm_limited():
    result = _compute(
        _active(max_delta_v_mps=0.2, alpha_ramp_enabled=True),
        [-4.0, 4.0, 0.0],
        pass_progress=0.75,
    )
    assert 0.0 < result.correction_alpha < 1.0
    assert np.linalg.norm(result.correction_delta_v_mps) <= 0.2 + 1.0e-12
    assert result.correction_norm_limited


def test_invalid_and_zero_distance_inputs_return_finite_zero():
    controller = _active()
    for result in (
        _compute(controller, [np.nan, 0.0, 0.0]),
        _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[0.0, 0.0, 0.0]),
    ):
        assert not result.active
        assert np.isfinite(result.correction_delta_v_mps).all()
        assert np.array_equal(result.correction_delta_v_mps, np.zeros(3))


def test_active_fusion_calls_transform_once_with_actor_plus_scaled_correction():
    result = _compute(_active(max_delta_v_mps=0.2), [0.0, 1.0, 0.0])
    actor = np.asarray([0.1, 0.2, 0.0])
    calls = []

    def transform(value):
        calls.append(value.copy())
        return {"constrained_delta_v_mps": value.tolist()}

    fused = fuse_actor_action(actor, result, transform)
    expected = actor + result.correction_alpha * result.correction_delta_v_mps
    assert len(calls) == 1
    assert np.allclose(calls[0], expected)
    assert np.allclose(fused.corrected_delta_v_pre_safety_mps, expected)


def test_diagnostic_helper_is_read_only_and_estimated_state_only():
    class ForbiddenTruth:
        @property
        def states(self):
            raise AssertionError("truth state must not be accessed")

    guarded = ForbiddenTruth()
    motion = decompose_goal_relative_motion([0, 0, 0], [1, 2, 0], [10, 0, 0])
    assert motion.valid
    assert guarded is not None
    parameters = inspect.signature(GoalRecoveryCorrection.compute).parameters
    assert "environment" not in parameters
    assert "state" not in parameters
    assert "passed_before" not in parameters


def test_teacher_goal_speed_helper_is_numerically_equivalent():
    for distance in (0.0, 1.0, 1.01, 3.0, 20.0, 100.0):
        legacy = float(
            np.clip(
                math.sqrt(max(0.0, 2.0 * 3.0 * max(distance - 1.0, 0.0))),
                0.35,
                6.0,
            )
        )
        shared = compute_desired_goal_speed(
            distance,
            max_acceleration_mps2=3.0,
            goal_threshold_m=1.0,
            minimum_speed_mps=0.35,
            cruise_speed_mps=6.0,
        )
        assert shared == legacy


def test_alpha_ramp_is_monotonic_after_release():
    controller = _active(alpha_ramp_enabled=True)
    alphas = [
        _compute(controller, [0.0, 0.0, 0.0], pass_progress=value).correction_alpha
        for value in (0.51, 0.6, 0.75, 0.9, 1.0)
    ]
    assert 0.0 < alphas[0] <= alphas[-1] <= 1.0
    assert alphas == sorted(alphas)


def test_formal_config_defaults_goal_recovery_off():
    from src.experiments.formal import FormalExperimentConfig

    config = FormalExperimentConfig(output_dir=Path("unused"))
    assert config.goal_recovery_correction_enabled is False
    assert config.goal_recovery_activation_mode == "post_bypass_only"
    assert config.goal_recovery_terminal_capture_enabled is False
    assert config.goal_recovery_terminal_capture_v2_enabled is False
    assert config.goal_recovery_terminal_effective_deceleration_mps2 == 1.697726
    assert config.goal_recovery_terminal_stopping_margin_m == 1.328009
    assert config.goal_recovery_terminal_hold_inside_goal_enabled is True


def test_terminal_capture_defaults_off_and_max_budget_unchanged():
    config = GoalRecoveryCorrectionConfig()
    assert config.terminal_capture_enabled is False
    assert config.terminal_capture_radius_m == 8.0
    assert config.terminal_radial_speed_floor_mps == 0.25
    assert config.terminal_tangential_gain == 1.0
    assert config.max_delta_v_mps == 0.6


def test_terminal_disabled_and_outside_radius_are_legacy_equivalent():
    legacy = _compute(_active(), [2.0, 1.0, 0.0])
    disabled = _compute(_active(terminal_capture_enabled=False), [2.0, 1.0, 0.0])
    outside = _compute(
        _active(terminal_capture_enabled=True, terminal_capture_radius_m=8.0),
        [2.0, 1.0, 0.0],
    )
    assert np.array_equal(disabled.correction_delta_v_mps, legacy.correction_delta_v_mps)
    assert np.array_equal(outside.correction_delta_v_mps, legacy.correction_delta_v_mps)
    assert not outside.terminal_capture_active


def test_terminal_speed_envelope_is_bounded_and_monotone_inward():
    distances = np.linspace(8.0, 0.0, 81)
    values = []
    for distance in distances:
        base = compute_desired_goal_speed(distance, max_acceleration_mps2=3.0)
        value, blend = compute_terminal_desired_radial_speed(
            distance,
            base_desired_speed_mps=base,
            terminal_capture_radius_m=8.0,
            goal_threshold_m=1.0,
            terminal_radial_speed_floor_mps=0.25,
            base_speed_at_terminal_radius_mps=6.0,
        )
        assert value <= base + 1.0e-12
        assert 0.0 <= blend <= 1.0
        values.append(value)
    assert all(left >= right - 1.0e-12 for left, right in zip(values, values[1:]))
    assert values[-1] == 0.25


def test_terminal_component_branches_fade_only_propulsion():
    controller = _active(
        terminal_capture_enabled=True,
        terminal_capture_radius_m=8.0,
        terminal_radial_brake_gain=0.8,
        terminal_tangential_gain=1.0,
    )
    near_goal = dict(goal_position_m=[1.0, 0.0, 0.0])
    propulsion = _compute(controller, [0.0, 0.0, 0.0], **near_goal)
    assert propulsion.terminal_capture_active
    assert propulsion.radial_propulsion_scale == 0.0
    assert np.allclose(propulsion.radial_propulsion_delta_v_mps, 0.0)
    braking = _compute(controller, [3.0, 0.0, 0.0], **near_goal)
    assert np.linalg.norm(braking.radial_braking_delta_v_mps) > 0.0
    assert braking.radial_propulsion_scale == 0.0


def test_terminal_tangential_damping_and_gain_are_directional():
    controller = _active(
        terminal_capture_enabled=True,
        terminal_capture_radius_m=8.0,
        tangential_gain=0.7,
        terminal_tangential_gain=1.0,
    )
    zero = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[1.0, 0.0, 0.0])
    assert np.allclose(zero.terminal_tangential_damping_delta_v_mps, 0.0)
    tangent = _compute(controller, [0.0, 2.0, 0.0], goal_position_m=[1.0, 0.0, 0.0])
    assert np.dot(
        tangent.terminal_tangential_damping_delta_v_mps,
        tangent.current_tangential_velocity_vector_mps,
    ) < 0.0
    outer = _compute(controller, [0.0, 2.0, 0.0], goal_position_m=[8.0, 0.0, 0.0])
    assert outer.effective_terminal_tangential_gain <= tangent.effective_terminal_tangential_gain


def test_terminal_combined_correction_is_capped_and_inputs_are_read_only():
    actor = np.asarray([0.1, -0.2, 0.3])
    actor_before = actor.copy()
    result = _compute(
        _active(terminal_capture_enabled=True, max_delta_v_mps=0.6),
        [5.0, 5.0, 0.0],
        goal_position_m=[2.0, 0.0, 0.0],
    )
    fuse_actor_action(actor, result, lambda value: {"constrained_delta_v_mps": value})
    assert np.linalg.norm(result.correction_delta_v_mps) <= 0.6 + 1.0e-12
    assert result.terminal_correction_clipped
    assert np.array_equal(actor, actor_before)


def test_terminal_config_and_blend_validation():
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_capture_radius_m=1.0)
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_tangential_gain=-0.1)
    assert compute_terminal_blend(
        8.0, terminal_capture_radius_m=8.0, goal_threshold_m=1.0
    ) == 0.0
    assert compute_terminal_blend(
        1.0, terminal_capture_radius_m=8.0, goal_threshold_m=1.0
    ) == 1.0


def test_terminal_propulsion_is_positive_and_fades_monotonically_inward():
    controller = _active(
        terminal_capture_enabled=True,
        terminal_capture_radius_m=8.0,
        terminal_propulsion_fade_enabled=True,
    )
    outer = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[7.0, 0.0, 0.0])
    inner = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[3.0, 0.0, 0.0])
    boundary = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[1.0, 0.0, 0.0])
    assert outer.radial_velocity_error_mps > 0.0
    assert outer.radial_propulsion_delta_v_mps[0] > 0.0
    assert outer.radial_propulsion_scale > inner.radial_propulsion_scale > boundary.radial_propulsion_scale
    assert boundary.radial_propulsion_scale == 0.0


def test_terminal_fade_and_smooth_blend_switches_are_actually_used():
    faded = _compute(
        _active(terminal_capture_enabled=True, terminal_propulsion_fade_enabled=True),
        [0.0, 0.0, 0.0],
        goal_position_m=[7.0, 0.0, 0.0],
    )
    unfaded = _compute(
        _active(terminal_capture_enabled=True, terminal_propulsion_fade_enabled=False),
        [0.0, 0.0, 0.0],
        goal_position_m=[7.0, 0.0, 0.0],
    )
    assert faded.radial_propulsion_scale < 1.0
    assert unfaded.radial_propulsion_scale == 1.0
    smooth = compute_terminal_blend(
        7.0, terminal_capture_radius_m=8.0, goal_threshold_m=1.0, smooth=True
    )
    linear = compute_terminal_blend(
        7.0, terminal_capture_radius_m=8.0, goal_threshold_m=1.0, smooth=False
    )
    assert smooth != linear


def test_terminal_compute_does_not_mutate_estimated_state_inputs():
    position = np.asarray([0.0, 0.0, 0.0])
    velocity = np.asarray([1.0, 2.0, 0.0])
    goal = np.asarray([4.0, 0.0, 0.0])
    before = tuple(value.copy() for value in (position, velocity, goal))
    _compute(
        _active(terminal_capture_enabled=True),
        velocity,
        estimated_position_m=position,
        goal_position_m=goal,
    )
    for value, expected in zip((position, velocity, goal), before):
        assert np.array_equal(value, expected)


def test_terminal_control_has_no_reward_environment_or_obstacle_mutation_imports():
    import ast

    path = Path(__file__).resolve().parents[1] / "src" / "control" / "goal_recovery_correction.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        token in module
        for module in imported_modules
        for token in ("environment", "reward", "obstacle", "simulation")
    )


def test_terminal_v2_defaults_and_scalar_validation():
    config = GoalRecoveryCorrectionConfig()
    assert config.terminal_capture_v2_enabled is False
    assert config.terminal_effective_deceleration_mps2 == 1.697726
    assert config.terminal_stopping_margin_m == 1.328009
    assert config.terminal_hold_inside_goal_enabled is True
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_capture_v2_enabled=1)
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_hold_inside_goal_enabled=0)
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_stopping_margin_m=-1.0)
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_effective_deceleration_mps2=np.nan)


def test_terminal_v2_envelope_formula_monotonic_and_capped():
    args = dict(
        base_desired_speed_mps=6.0,
        effective_deceleration_mps2=1.697726,
        stopping_margin_m=1.328009,
        goal_threshold_m=1.0,
        radial_speed_floor_mps=0.25,
    )
    distances = np.linspace(1.0, 20.0, 100)
    speeds = [compute_terminal_v2_envelope(distance, **args) for distance in distances]
    assert all(left <= right + 1.0e-12 for left, right in zip(speeds, speeds[1:]))
    assert all(speed <= args["base_desired_speed_mps"] + 1.0e-12 for speed in speeds)
    distance = 5.0
    expected = min(
        args["base_desired_speed_mps"],
        math.sqrt(
            args["radial_speed_floor_mps"] ** 2
            + 2.0
            * args["effective_deceleration_mps2"]
            * max(distance - args["goal_threshold_m"] - args["stopping_margin_m"], 0.0)
        ),
    )
    assert compute_terminal_v2_envelope(distance, **args) == expected


def test_terminal_v2_margin_and_deceleration_change_envelope_in_expected_direction():
    common = dict(
        goal_distance_m=5.0,
        base_desired_speed_mps=6.0,
        goal_threshold_m=1.0,
        radial_speed_floor_mps=0.25,
    )
    faster = compute_terminal_v2_envelope(
        **common, effective_deceleration_mps2=2.0, stopping_margin_m=1.0
    )
    slower = compute_terminal_v2_envelope(
        **common, effective_deceleration_mps2=1.0, stopping_margin_m=1.0
    )
    wider_margin = compute_terminal_v2_envelope(
        **common, effective_deceleration_mps2=2.0, stopping_margin_m=2.0
    )
    assert faster > slower
    assert wider_margin < faster


def test_terminal_v2_gates_hold_overspeed_and_anti_stall_diagnostics():
    controller = _active(
        terminal_capture_enabled=True,
        terminal_capture_v2_enabled=True,
        terminal_propulsion_fade_enabled=True,
    )
    not_released = _compute(controller, [0.0, 0.0, 0.0], release_latched=False)
    not_post_bypass = _compute(controller, [0.0, 0.0, 0.0], post_bypass_recovery=False)
    for result in (not_released, not_post_bypass):
        assert result.terminal_v2_enabled
        assert not result.terminal_v2_governor_eligible
        assert not result.terminal_v2_governor_active
        assert result.terminal_v2_governed_desired_radial_speed_mps == 0.0
        assert not result.terminal_v2_braking_required
    anti_stall = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[1.17, 0.0, 0.0])
    assert anti_stall.terminal_v2_governor_eligible
    assert anti_stall.terminal_v2_anti_stall_active
    assert anti_stall.desired_radial_velocity_mps > 0.0
    assert anti_stall.radial_propulsion_scale == 1.0
    hold = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[1.0, 0.0, 0.0])
    assert hold.terminal_v2_inside_goal_hold
    assert hold.terminal_v2_governed_desired_radial_speed_mps == 0.0
    overspeed = _compute(controller, [2.0, 0.0, 0.0], goal_position_m=[3.0, 0.0, 0.0])
    assert overspeed.terminal_v2_radial_overspeed_mps > 0.0
    assert overspeed.terminal_v2_braking_required
    assert np.linalg.norm(overspeed.correction_delta_v_mps) <= controller.config.max_delta_v_mps


def test_terminal_v2_is_not_limited_to_legacy_eight_meter_radius_and_fusion_is_unchanged():
    controller = _active(terminal_capture_v2_enabled=True)
    far = _compute(controller, [0.0, 0.0, 0.0], goal_position_m=[12.0, 0.0, 0.0])
    assert far.terminal_v2_governor_eligible
    assert far.goal_distance_m > controller.config.terminal_capture_radius_m
    assert far.terminal_capture_active is False
    actor = np.asarray([0.1, -0.2, 0.3])
    actor_before = actor.copy()
    fused = fuse_actor_action(actor, far, lambda value: {"constrained_delta_v_mps": value})
    assert np.array_equal(fused.actor_raw_delta_v_mps, actor_before)
    assert np.array_equal(actor, actor_before)


def test_terminal_v2_off_keeps_legacy_action_and_zeroes_v2_diagnostics():
    legacy = _compute(_active(terminal_capture_enabled=True), [2.0, 1.0, 0.0], goal_position_m=[3.0, 0.0, 0.0])
    assert legacy.terminal_v2_enabled is False
    assert legacy.terminal_v2_governor_eligible is False
    assert legacy.terminal_v2_governor_active is False
    assert legacy.terminal_v2_base_desired_radial_speed_mps == 0.0
    assert legacy.terminal_v2_envelope_speed_mps == 0.0
    assert legacy.terminal_v2_governed_desired_radial_speed_mps == 0.0
    assert legacy.terminal_v2_radial_overspeed_mps == 0.0


def test_terminal_v2_off_is_exactly_equivalent_across_legacy_control_paths():
    cases = (
        ([0.0, 0.0, 0.0], [7.0, 0.0, 0.0]),
        ([4.0, 2.0, 0.0], [3.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [1.17, 0.0, 0.0]),
    )
    legacy_fields = (
        "correction_delta_v_mps",
        "radial_correction_delta_v_mps",
        "tangential_correction_delta_v_mps",
        "desired_radial_velocity_mps",
        "terminal_blend",
        "radial_velocity_error_mps",
        "radial_propulsion_delta_v_mps",
        "radial_propulsion_scale",
        "radial_braking_delta_v_mps",
        "effective_terminal_tangential_gain",
        "terminal_raw_correction_delta_v_mps",
        "terminal_raw_correction_norm_mps",
        "terminal_correction_clipped",
    )
    for velocity, goal in cases:
        implicit = _compute(
            _active(terminal_capture_enabled=True, max_delta_v_mps=0.6),
            velocity,
            goal_position_m=goal,
        )
        explicit = _compute(
            _active(
                terminal_capture_enabled=True,
                terminal_capture_v2_enabled=False,
                max_delta_v_mps=0.6,
            ),
            velocity,
            goal_position_m=goal,
        )
        for name in legacy_fields:
            left, right = getattr(implicit, name), getattr(explicit, name)
            if isinstance(left, np.ndarray):
                assert np.array_equal(left, right), name
            else:
                assert left == right, name


def _v21_active(**overrides):
    values = dict(
        enabled=True,
        max_delta_v_mps=10.0,
        alpha_ramp_enabled=False,
        terminal_capture_v2_enabled=True,
        terminal_radial_conflict_arbitration_enabled=True,
    )
    values.update(overrides)
    return GoalRecoveryCorrection(GoalRecoveryCorrectionConfig(**values))


def test_terminal_v21_defaults_and_validation_are_opt_in():
    config = GoalRecoveryCorrectionConfig()
    assert config.terminal_radial_conflict_arbitration_enabled is False
    assert config.terminal_radial_conflict_compensation_gain == 1.0
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_radial_conflict_arbitration_enabled=1)
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_radial_conflict_compensation_gain=-1.0)
    with np.testing.assert_raises(ValueError):
        GoalRecoveryCorrectionConfig(terminal_radial_conflict_compensation_gain=np.nan)

    from src.experiments.formal import FormalExperimentConfig

    formal = FormalExperimentConfig(output_dir=Path("unused"))
    assert formal.goal_recovery_terminal_radial_conflict_arbitration_enabled is False
    assert formal.goal_recovery_terminal_radial_conflict_compensation_gain == 1.0
    with np.testing.assert_raises(ValueError):
        FormalExperimentConfig(
            output_dir=Path("unused"),
            goal_recovery_terminal_radial_conflict_arbitration_enabled=1,
        )
    with np.testing.assert_raises(ValueError):
        FormalExperimentConfig(
            output_dir=Path("unused"),
            goal_recovery_terminal_radial_conflict_compensation_gain=np.inf,
        )


def test_terminal_v21_off_does_not_inspect_actor_and_preserves_v2_exactly():
    class ForbiddenActor:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Actor must not be inspected while v2.1 is off")

    controller = _active(terminal_capture_v2_enabled=True)
    kwargs = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 1.0, 0.0],
        goal_position_m=[10.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
    )
    omitted = controller.compute(**kwargs)
    malformed = controller.compute(**kwargs, actor_raw_delta_v_mps=ForbiddenActor())
    for name in (
        "correction_delta_v_mps",
        "radial_correction_delta_v_mps",
        "tangential_correction_delta_v_mps",
        "terminal_raw_correction_delta_v_mps",
    ):
        assert np.array_equal(getattr(omitted, name), getattr(malformed, name)), name
    assert omitted.terminal_v21_radial_conflict_enabled is False
    assert malformed.terminal_v21_inactive_reason == "disabled"


def test_terminal_v21_missing_or_nan_actor_keeps_base_correction_exact():
    kwargs = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 1.0, 0.0],
        goal_position_m=[10.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
    )
    base = _active(terminal_capture_v2_enabled=True).compute(**kwargs)
    for actor, reason in ((None, "actor_missing"), ([np.nan, 0.0, 0.0], "actor_non_finite")):
        result = _v21_active().compute(**kwargs, actor_raw_delta_v_mps=actor)
        assert np.array_equal(result.correction_delta_v_mps, base.correction_delta_v_mps)
        assert np.array_equal(
            result.terminal_v21_raw_correction_before_conflict_mps,
            result.terminal_v21_raw_correction_after_conflict_mps,
        )
        assert result.terminal_v21_radial_conflict_eligible is False
        assert result.terminal_v21_radial_conflict_active is False
        assert result.terminal_v21_inactive_reason == reason


def test_terminal_v21_gate_order_requires_v2_and_approach_state():
    common = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 0.0, 0.0],
        goal_position_m=[10.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        actor_raw_delta_v_mps=[-2.0, 0.0, 0.0],
    )
    no_v2 = _active(
        terminal_radial_conflict_arbitration_enabled=True,
    ).compute(**common)
    assert no_v2.terminal_v21_radial_conflict_enabled
    assert not no_v2.terminal_v21_radial_conflict_eligible
    assert no_v2.terminal_v21_inactive_reason == "v2_disabled"

    inside = _v21_active().compute(**{**common, "goal_position_m": [1.0, 0.0, 0.0]})
    assert not inside.terminal_v21_radial_conflict_eligible
    assert inside.terminal_v21_inactive_reason == "distance_not_above_threshold"

    overspeed = _v21_active().compute(
        **{**common, "estimated_velocity_mps": [6.0, 0.0, 0.0]}
    )
    assert not overspeed.terminal_v21_radial_conflict_eligible
    assert overspeed.terminal_v21_inactive_reason == "radial_error_non_positive"

    non_conflicting = _v21_active().compute(
        **{**common, "actor_raw_delta_v_mps": [1.0, 0.0, 0.0]}
    )
    assert non_conflicting.terminal_v21_radial_conflict_eligible
    assert not non_conflicting.terminal_v21_radial_conflict_active
    assert non_conflicting.terminal_v21_inactive_reason == "actor_not_opposing"


def test_terminal_v21_axis_and_diagonal_compensation_uses_goal_direction():
    axis = _v21_active().compute(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 0.0, 0.0],
        goal_position_m=[10.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        actor_raw_delta_v_mps=[-2.0, 3.0, 0.0],
    )
    assert axis.terminal_v21_actor_radial_delta_v_mps == -2.0
    assert axis.terminal_v21_actor_opposing_radial_mps == 2.0
    assert np.array_equal(
        axis.terminal_v21_conflict_compensation_delta_v_mps,
        np.asarray([2.0, 0.0, 0.0]),
    )
    assert axis.terminal_v21_goal_direction_source == (
        "estimated_goal_direction_used_by_radial_propulsion"
    )

    diagonal = _v21_active().compute(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 0.0, 0.0],
        goal_position_m=[10.0, 10.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        actor_raw_delta_v_mps=[-3.0, 0.0, 0.0],
    )
    expected = np.asarray([1.5, 1.5, 0.0])
    assert np.allclose(
        diagonal.terminal_v21_conflict_compensation_delta_v_mps,
        expected,
    )
    assert np.isclose(diagonal.terminal_v21_conflict_compensation_norm_mps, np.sqrt(4.5))
    assert np.isclose(diagonal.terminal_v21_conflict_compensation_radial_mps, 3.0 / np.sqrt(2.0))


def test_terminal_v21_preserves_tangential_component_and_attributes_shared_cap():
    kwargs = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 1.0, 0.0],
        goal_position_m=[10.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
    )
    base = _active(
        terminal_capture_v2_enabled=True,
        max_delta_v_mps=0.5,
    ).compute(**kwargs)
    result = _v21_active(max_delta_v_mps=0.5).compute(
        **kwargs,
        actor_raw_delta_v_mps=[-2.0, 0.0, 0.0],
    )
    assert np.array_equal(
        result.tangential_correction_delta_v_mps,
        base.tangential_correction_delta_v_mps,
    )
    assert np.allclose(
        result.terminal_v21_raw_correction_after_conflict_mps,
        result.terminal_v21_raw_correction_before_conflict_mps
        + result.terminal_v21_conflict_compensation_delta_v_mps,
    )
    assert result.terminal_v21_radial_conflict_active
    assert result.terminal_v21_correction_clip_ratio < 1.0
    assert np.allclose(
        result.correction_delta_v_mps,
        result.terminal_v21_raw_correction_after_conflict_mps
        * result.terminal_v21_correction_clip_ratio,
    )
    assert np.isclose(
        result.terminal_v21_compensation_post_cap_radial_mps,
        result.terminal_v21_conflict_compensation_radial_mps
        * result.terminal_v21_correction_clip_ratio,
    )

    actor = np.asarray([-2.0, 0.0, 0.0])
    calls = []
    fused = fuse_actor_action(
        actor,
        result,
        lambda value: calls.append(value.copy())
        or {"constrained_delta_v_mps": value},
    )
    assert len(calls) == 1
    assert np.allclose(
        fused.corrected_delta_v_pre_action_transform_mps,
        actor + result.correction_alpha * result.correction_delta_v_mps,
    )
    assert np.array_equal(actor, np.asarray([-2.0, 0.0, 0.0]))


def test_terminal_v21_base_eligibility_gates_never_compensate():
    controller = _v21_active(alpha_ramp_enabled=True)
    common = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 0.0, 0.0],
        goal_position_m=[3.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        action_provenance="evaluation",
        actor_raw_delta_v_mps=[-0.2, 0.3, 0.0],
    )
    cases = (
        {"release_latched": False},
        {"post_bypass_recovery": False},
        {"pass_progress": 0.5},
        {"action_provenance": "training"},
    )
    for override in cases:
        result = controller.compute(**{**common, **override})
        assert not result.terminal_v21_radial_conflict_active
        assert not result.terminal_v21_radial_conflict_eligible
        assert np.array_equal(
            result.terminal_v21_conflict_compensation_delta_v_mps,
            np.zeros(3),
        )


def test_terminal_v21_braking_branch_and_v2_governor_are_exactly_unchanged():
    common = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[3.0, 0.4, 0.0],
        goal_position_m=[3.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
    )
    base = _active(
        terminal_capture_enabled=True,
        terminal_capture_v2_enabled=True,
        max_delta_v_mps=0.6,
    ).compute(**common)
    candidate = _v21_active(
        terminal_capture_enabled=True,
        max_delta_v_mps=0.6,
    ).compute(**common, actor_raw_delta_v_mps=[-0.5, 0.2, 0.0])
    assert base.terminal_v2_braking_required
    assert not candidate.terminal_v21_radial_conflict_active
    for name in (
        "correction_delta_v_mps",
        "radial_correction_delta_v_mps",
        "tangential_correction_delta_v_mps",
        "terminal_raw_correction_delta_v_mps",
    ):
        assert np.array_equal(getattr(candidate, name), getattr(base, name)), name
    for name in (
        "desired_radial_velocity_mps",
        "terminal_v2_envelope_speed_mps",
        "terminal_v2_governed_desired_radial_speed_mps",
        "terminal_v2_radial_overspeed_mps",
        "terminal_v2_braking_required",
    ):
        assert getattr(candidate, name) == getattr(base, name), name


def test_terminal_v21_gain_scales_only_requested_radial_compensation():
    common = dict(
        estimated_position_m=[0.0, 0.0, 0.0],
        estimated_velocity_mps=[0.0, 0.5, 0.0],
        goal_position_m=[3.0, 0.0, 0.0],
        release_latched=True,
        post_bypass_recovery=True,
        pass_progress=1.0,
        actor_raw_delta_v_mps=[-0.2, 0.7, 0.0],
    )
    full = _v21_active(terminal_radial_conflict_compensation_gain=1.0).compute(**common)
    half = _v21_active(terminal_radial_conflict_compensation_gain=0.5).compute(**common)
    assert np.allclose(
        half.terminal_v21_conflict_compensation_delta_v_mps,
        0.5 * full.terminal_v21_conflict_compensation_delta_v_mps,
    )
    assert np.array_equal(
        half.tangential_correction_delta_v_mps,
        full.tangential_correction_delta_v_mps,
    )
