from __future__ import annotations

import inspect
import math
from pathlib import Path
import unittest

from src.experiments.single_uav_radial_intent_persistence_diagnostic import (
    ESTIMATED_LATERAL_EXIT_CONTIGUOUS_STEPS,
    ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS,
    RADIAL_COMMAND_DEADBAND_MPS,
    TARGET_FLOOR_MPS,
    RadialIntentPersistenceAssistance,
    assert_radial_intent_exclusive,
    transform_radial_assisted_delta_v_once,
)


def _assert_approx(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(f"{actual!r} is not approximately {expected!r}")


def _helper(*, enabled: bool = True, cap: float = 0.05) -> RadialIntentPersistenceAssistance:
    return RadialIntentPersistenceAssistance(enabled=enabled, cap_mps=cap)


def _step(
    helper: RadialIntentPersistenceAssistance,
    step: int,
    actor_radial: float = -0.02,
    **overrides: object,
):
    values: dict[str, object] = {
        "step": step,
        "actor_raw_delta_v_mps": (actor_radial, 0.3, -0.4),
        "radial_unit": (1.0, 0.0, 0.0),
        "estimated_lateral_velocity_mps": 0.25,
        "frame_frozen": True,
        "relevant_obstacle_match": True,
        "release_latched": False,
        "post_bypass_recovery": False,
        "passed_obstacle": False,
        "goal_recovery_active": False,
        "lateral_assist_active": False,
        "done": False,
    }
    values.update(overrides)
    return helper.evaluate(**values)  # type: ignore[arg-type]


def test_goalward_radial_sign_uses_frozen_encounter_axis() -> None:
    helper = _helper(enabled=False)
    positive = _step(helper, 0, actor_radial=0.2)
    negative = _step(helper, 1, actor_radial=-0.1)
    _assert_approx(positive.actor_radial_delta_v_mps, 0.2)
    _assert_approx(negative.actor_radial_delta_v_mps, -0.1)
    assert positive.radial_unit == (1.0, 0.0, 0.0)


def test_estimated_lateral_exit_requires_three_contiguous_steps_at_threshold() -> None:
    helper = _helper(enabled=False)
    first = _step(helper, 0, estimated_lateral_velocity_mps=ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS)
    second = _step(helper, 1, estimated_lateral_velocity_mps=-ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS)
    third = _step(helper, 2, estimated_lateral_velocity_mps=0.1)
    assert first.estimated_lateral_exit_run == 1
    assert second.estimated_lateral_exit_run == 2
    assert third.estimated_lateral_exit_run == ESTIMATED_LATERAL_EXIT_CONTIGUOUS_STEPS
    assert third.estimated_lateral_exit_ready


def test_estimated_lateral_exit_run_resets_on_threshold_or_step_gap() -> None:
    helper = _helper(enabled=False)
    _step(helper, 0, estimated_lateral_velocity_mps=0.2)
    _step(helper, 1, estimated_lateral_velocity_mps=0.8)
    after_threshold_break = _step(helper, 2, estimated_lateral_velocity_mps=0.1)
    after_step_gap = _step(helper, 4, estimated_lateral_velocity_mps=0.1)
    assert after_threshold_break.estimated_lateral_exit_run == 1
    assert after_step_gap.estimated_lateral_exit_run == 1
    assert not after_step_gap.estimated_lateral_exit_ready


def test_runtime_helper_has_no_simulator_truth_parameters_or_reads() -> None:
    signature = inspect.signature(RadialIntentPersistenceAssistance.evaluate)
    assert not any("actual" in name.lower() for name in signature.parameters)
    source = inspect.getsource(RadialIntentPersistenceAssistance.evaluate)
    assert "actual_" not in source
    assert "environment" not in source


def test_actor_persistence_uses_full_five_step_contiguous_history_and_four_threshold() -> None:
    deficient = _helper()
    outcomes = [
        _step(deficient, 0, actor_radial=0.1),
        _step(deficient, 1, actor_radial=0.1),
        _step(deficient, 2, actor_radial=0.1),
        _step(deficient, 3, actor_radial=-0.01),
        _step(deficient, 4, actor_radial=-0.02),
    ]
    assert outcomes[-2].actor_goalward_history_5 == (1, 1, 1, 0)
    assert outcomes[-1].actor_goalward_history_5 == (1, 1, 1, 0, 0)
    assert outcomes[-1].actor_goalward_count_5 == 3
    assert outcomes[-1].actor_radial_persistence_deficient
    assert outcomes[-1].active

    adequate = _helper()
    result = None
    for step, radial in enumerate((0.1, -0.01, 0.2, 0.1, -0.02)):
        result = _step(adequate, step, actor_radial=radial)
    assert result is not None
    assert result.actor_goalward_count_5 == 3
    assert result.actor_radial_persistence_deficient
    assert result.active
    four_goalward = _helper()
    for step, radial in enumerate((0.1, 0.1, 0.2, 0.1, -0.02)):
        result = _step(four_goalward, step, actor_radial=radial)
    assert result is not None
    assert result.actor_goalward_count_5 == 4
    assert not result.actor_radial_persistence_deficient
    assert not result.active


def test_assistance_is_radial_only_and_respects_the_frozen_floor_and_cap() -> None:
    helper = _helper(cap=0.05)
    decision = None
    for step in range(5):
        decision = _step(helper, step, actor_radial=-0.2)
    assert decision is not None and decision.active
    _assert_approx(decision.requested_radial_assist_mps, TARGET_FLOOR_MPS + 0.2)
    _assert_approx(decision.applied_radial_assist_mps, 0.05)
    for actual, expected in zip(decision.applied_radial_assist_delta_v_mps, (0.05, 0.0, 0.0)):
        _assert_approx(actual, expected)
    assert decision.applied_radial_assist_delta_v_mps[1] == 0.0
    assert decision.applied_radial_assist_delta_v_mps[2] == 0.0


def test_assistance_stops_immediately_on_pass_or_release_latch() -> None:
    helper = _helper()
    for step in range(5):
        decision = _step(helper, step)
    assert decision.active
    post_pass = _step(helper, 5, release_latched=True, passed_obstacle=True)
    assert not post_pass.active
    assert post_pass.applied_radial_assist_mps == 0.0
    assert post_pass.applied_radial_assist_delta_v_mps == (0.0, 0.0, 0.0)


def test_overlap_gates_prevent_assistance_and_active_overlap_asserts() -> None:
    for overlap in (
        {"goal_recovery_active": True},
        {"lateral_assist_active": True},
    ):
        helper = _helper()
        for step in range(4):
            _step(helper, step)
        decision = _step(helper, 4, **overlap)
        assert not decision.active
        assert decision.applied_radial_assist_mps == 0.0
        with unittest.TestCase().assertRaises(AssertionError):
            assert_radial_intent_exclusive(
                radial_intent_active=True,
                goal_recovery_active=overlap.get("goal_recovery_active", False),
                lateral_assist_active=overlap.get("lateral_assist_active", False),
            )


def test_frozen_frame_obstacle_prepass_done_and_inactive_gates_are_enforced() -> None:
    cases = (
        {"frame_frozen": False},
        {"relevant_obstacle_match": False},
        {"post_bypass_recovery": True},
        {"passed_obstacle": True},
        {"done": True},
    )
    for overrides in cases:
        helper = _helper()
        for step in range(4):
            _step(helper, step)
        decision = _step(helper, 4, **overrides)
        assert not decision.active
        assert decision.applied_radial_assist_mps == 0.0


def test_off_assistance_is_zero_and_radial_transform_hook_calls_once() -> None:
    helper = _helper(enabled=False)
    decision = None
    for step in range(5):
        decision = _step(helper, step, actor_radial=-0.2)
    assert decision is not None
    assert decision.requested_radial_assist_mps == 0.0
    assert decision.applied_radial_assist_mps == 0.0
    assert decision.applied_radial_assist_delta_v_mps == (0.0, 0.0, 0.0)

    calls: list[tuple[float, ...]] = []

    def transform(vector, limit):
        calls.append(tuple(vector))
        return {"constrained_delta_v_mps": tuple(vector)}

    combined, constrained = transform_radial_assisted_delta_v_once(
        base_delta_v_mps=(0.1, 0.2, -0.3),
        radial_assist_delta_v_mps=(0.04, 0.0, 0.0),
        action_transform=transform,
        action_limit_mps=0.6,
    )
    assert calls == [(0.14, 0.2, -0.3)]
    for actual, expected in zip(combined, (0.14, 0.2, -0.3)):
        _assert_approx(actual, expected)
    for actual, expected in zip(constrained, combined):
        _assert_approx(actual, expected)


def test_evaluator_keeps_radial_transform_behind_active_gate_and_legacy_off_route() -> None:
    source = Path(
        "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
    ).read_text(encoding="utf-8")
    active_branch = "if radial_intent_decision is not None and radial_intent_decision.active:"
    assert active_branch in source
    branch = source.split(active_branch, maxsplit=1)[1].split("elif assist_decision", maxsplit=1)[0]
    assert "transform_radial_assisted_delta_v_once" in branch
    assert "elif correction_result is not None:" in source


class RadialIntentPersistenceAssistanceTests(unittest.TestCase):
    """Expose the focused pure-helper checks to unittest discovery."""

    def test_goalward_radial_sign_uses_frozen_encounter_axis(self) -> None:
        test_goalward_radial_sign_uses_frozen_encounter_axis()

    def test_estimated_lateral_exit_requires_three_contiguous_steps_at_threshold(self) -> None:
        test_estimated_lateral_exit_requires_three_contiguous_steps_at_threshold()

    def test_estimated_lateral_exit_run_resets_on_threshold_or_step_gap(self) -> None:
        test_estimated_lateral_exit_run_resets_on_threshold_or_step_gap()

    def test_runtime_helper_has_no_simulator_truth_parameters_or_reads(self) -> None:
        test_runtime_helper_has_no_simulator_truth_parameters_or_reads()

    def test_actor_persistence_uses_full_five_step_contiguous_history_and_four_threshold(self) -> None:
        test_actor_persistence_uses_full_five_step_contiguous_history_and_four_threshold()

    def test_assistance_is_radial_only_and_respects_the_frozen_floor_and_cap(self) -> None:
        test_assistance_is_radial_only_and_respects_the_frozen_floor_and_cap()

    def test_assistance_stops_immediately_on_pass_or_release_latch(self) -> None:
        test_assistance_stops_immediately_on_pass_or_release_latch()

    def test_overlap_gates_prevent_assistance_and_active_overlap_asserts(self) -> None:
        test_overlap_gates_prevent_assistance_and_active_overlap_asserts()

    def test_frozen_frame_obstacle_prepass_done_and_inactive_gates_are_enforced(self) -> None:
        test_frozen_frame_obstacle_prepass_done_and_inactive_gates_are_enforced()

    def test_off_assistance_is_zero_and_radial_transform_hook_calls_once(self) -> None:
        test_off_assistance_is_zero_and_radial_transform_hook_calls_once()

    def test_evaluator_keeps_radial_transform_behind_active_gate_and_legacy_off_route(self) -> None:
        test_evaluator_keeps_radial_transform_behind_active_gate_and_legacy_off_route()
