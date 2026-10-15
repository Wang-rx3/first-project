"""Pure PASS_COMMIT action composition and decomposition checks.

These helpers operate only on action vectors and a supplied transform callable.
They do not import or construct an environment, simulator, or controller.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable


Vector3 = tuple[float, float, float]


def _vector3(value: Iterable[float], *, name: str) -> Vector3:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result  # type: ignore[return-value]


def compose_pre_transform_action(
    *,
    actor_base_candidate_mps: Iterable[float],
    goal_recovery_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    lateral_exit_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    pass_commit_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    radial_intent_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    pass_commit_enabled: bool = False,
    pass_commit_active: bool = False,
    goal_recovery_active: bool = False,
    lateral_exit_active: bool = False,
    radial_intent_active: bool = False,
) -> Vector3:
    """Sum only actually applied components and guard PASS_COMMIT exclusivity."""

    actor = _vector3(actor_base_candidate_mps, name="actor_base_candidate_mps")
    goal = _vector3(goal_recovery_applied_mps, name="goal_recovery_applied_mps")
    lateral = _vector3(lateral_exit_applied_mps, name="lateral_exit_applied_mps")
    pass_commit = _vector3(pass_commit_applied_mps, name="pass_commit_applied_mps")
    radial = _vector3(radial_intent_applied_mps, name="radial_intent_applied_mps")

    if pass_commit_active and not pass_commit_enabled:
        raise ValueError("PASS_COMMIT cannot be active while disabled")
    if pass_commit_active and (goal_recovery_active or lateral_exit_active or radial_intent_active):
        raise ValueError("PASS_COMMIT cannot overlap Goal Recovery, lateral, or radial assistance")
    if pass_commit_active and not any(component != 0.0 for component in pass_commit):
        raise ValueError("active PASS_COMMIT requires a nonzero applied component")
    if not pass_commit_active and any(component != 0.0 for component in pass_commit):
        raise ValueError("inactive PASS_COMMIT must contribute a zero applied component")

    return tuple(
        actor[index] + goal[index] + lateral[index] + pass_commit[index] + radial[index]
        for index in range(3)
    )  # type: ignore[return-value]


def pre_transform_decomposition_residual_mps(
    *,
    combined_pre_transform_action_mps: Iterable[float],
    actor_base_candidate_mps: Iterable[float],
    goal_recovery_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    lateral_exit_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    pass_commit_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    radial_intent_applied_mps: Iterable[float] = (0.0, 0.0, 0.0),
    pass_commit_enabled: bool = False,
    pass_commit_active: bool = False,
    goal_recovery_active: bool = False,
    lateral_exit_active: bool = False,
    radial_intent_active: bool = False,
) -> float:
    """Measure the Euclidean residual against the applied-component sum."""

    combined = _vector3(
        combined_pre_transform_action_mps,
        name="combined_pre_transform_action_mps",
    )
    reconstructed = compose_pre_transform_action(
        actor_base_candidate_mps=actor_base_candidate_mps,
        goal_recovery_applied_mps=goal_recovery_applied_mps,
        lateral_exit_applied_mps=lateral_exit_applied_mps,
        pass_commit_applied_mps=pass_commit_applied_mps,
        radial_intent_applied_mps=radial_intent_applied_mps,
        pass_commit_enabled=pass_commit_enabled,
        pass_commit_active=pass_commit_active,
        goal_recovery_active=goal_recovery_active,
        lateral_exit_active=lateral_exit_active,
        radial_intent_active=radial_intent_active,
    )
    return math.sqrt(sum((combined[index] - reconstructed[index]) ** 2 for index in range(3)))


def post_transform_decomposition_residual_mps(
    *,
    action_transform_input_mps: Iterable[float],
    action_transform_output_mps: Iterable[float],
    transformed_reconstruction_mps: Iterable[float],
) -> float:
    """Compare the actual post-transform vector with a pure reconstruction."""

    # Validate the recorded input too, so malformed transform telemetry fails closed.
    _vector3(action_transform_input_mps, name="action_transform_input_mps")
    actual = _vector3(action_transform_output_mps, name="action_transform_output_mps")
    expected = _vector3(transformed_reconstruction_mps, name="transformed_reconstruction_mps")
    return math.sqrt(sum((actual[index] - expected[index]) ** 2 for index in range(3)))


class SingleCallActionTransform:
    """Transparent transform proxy that fails if a diagnostic step calls twice."""

    def __init__(self, transform: Callable[..., Any], action_limit_mps: float) -> None:
        limit = float(action_limit_mps)
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("action_limit_mps must be positive and finite")
        self._transform = transform
        self._action_limit_mps = limit
        self.call_count = 0

    def __call__(self, action_delta_v_mps: Iterable[float], action_limit_mps: float | None = None) -> Any:
        if self.call_count >= 1:
            raise AssertionError("PASS_COMMIT action transform called more than once")
        if action_limit_mps is not None and abs(float(action_limit_mps) - self._action_limit_mps) > 1.0e-12:
            raise ValueError("action transform limit changed within a PASS_COMMIT step")
        action = _vector3(action_delta_v_mps, name="action_delta_v_mps")
        self.call_count += 1
        return self._transform(action, self._action_limit_mps)

    def assert_call_count(self, expected: int) -> None:
        if self.call_count != int(expected):
            raise AssertionError(
                f"PASS_COMMIT action transform call count {self.call_count} != {int(expected)}"
            )


def post_safety_execution_residual_mps(
    *,
    pre_safety_action_mps: Iterable[float],
    projection_delta_mps: Iterable[float],
    final_applied_action_mps: Iterable[float],
) -> float:
    """Preserve the evaluator's existing post-safety execution identity check."""

    pre_safety = _vector3(pre_safety_action_mps, name="pre_safety_action_mps")
    projection = _vector3(projection_delta_mps, name="projection_delta_mps")
    final = _vector3(final_applied_action_mps, name="final_applied_action_mps")
    return math.sqrt(
        sum((pre_safety[index] + projection[index] - final[index]) ** 2 for index in range(3))
    )


__all__ = [
    "SingleCallActionTransform",
    "compose_pre_transform_action",
    "post_safety_execution_residual_mps",
    "post_transform_decomposition_residual_mps",
    "pre_transform_decomposition_residual_mps",
]
