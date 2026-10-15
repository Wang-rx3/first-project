"""Pure estimated-state PASS_COMMIT non-regression controller.

The controller consumes only the pre-step EKF state, static pass geometry,
existing obstacle-context gates, and the baseline policy command.  It does not
accept simulator truth and never applies an action transform itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


PASS_COMMIT_TRIGGER_DEADBAND_M = 1.0e-8
PASS_COMMIT_CLEARANCE_FLOOR_M = 2.5
PASS_COMMIT_HARD_CAP_MPS = 0.075
PASS_COMMIT_LATERAL_EXIT_THRESHOLD_MPS = 0.5
PASS_COMMIT_LATERAL_EXIT_STEPS = 3
PASS_COMMIT_DIRECTION_SOURCE = "current_ekf_goal_unit_from_pass_geometry"


class PassCommitGeometryError(ValueError):
    """Raised when the existing estimated pass geometry cannot be verified."""


def _vector3(value: Iterable[float], *, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result  # type: ignore[return-value]


def _dot(left: Iterable[float], right: Iterable[float]) -> float:
    a = tuple(left)
    b = tuple(right)
    return sum(a[index] * b[index] for index in range(3))


@dataclass(frozen=True)
class PassGeometry:
    direction_unit: tuple[float, float, float]
    along_obstacle_m: float
    obstacle_extent_m: float
    passed_margin_m: float


def derive_pass_geometry(
    *,
    estimated_position_m: Iterable[float],
    goal_position_m: Iterable[float],
    obstacle_center_m: Iterable[float],
    obstacle_extent_m: float,
    pass_buffer_m: float,
    logged_along_obstacle_m: float | None = None,
    logged_passed_margin_m: float | None = None,
    tolerance: float = 1.0e-8,
) -> PassGeometry:
    """Recompute the evaluator's pass geometry and verify its logged values.

    This is the same geometry as the existing release helper:
    ``along=(center-position)·unit(goal-position)`` and
    ``passed_margin=-(along+extent+buffer)``.
    """

    position = _vector3(estimated_position_m, name="estimated_position_m")
    goal = _vector3(goal_position_m, name="goal_position_m")
    center = _vector3(obstacle_center_m, name="obstacle_center_m")
    extent = float(obstacle_extent_m)
    buffer = float(pass_buffer_m)
    tol = float(tolerance)
    if not all(math.isfinite(item) for item in (extent, buffer, tol)):
        raise PassCommitGeometryError("non-finite pass geometry parameter")
    if extent < 0.0 or buffer < 0.0 or tol < 0.0:
        raise PassCommitGeometryError("negative pass geometry parameter")
    goal_delta = tuple(goal[index] - position[index] for index in range(3))
    goal_norm = math.sqrt(_dot(goal_delta, goal_delta))
    if not math.isfinite(goal_norm) or goal_norm <= 1.0e-12:
        raise PassCommitGeometryError("degenerate estimated goal direction")
    direction = tuple(value / goal_norm for value in goal_delta)
    obstacle_delta = tuple(center[index] - position[index] for index in range(3))
    along = _dot(obstacle_delta, direction)
    margin = -(along + extent + buffer)
    if not math.isfinite(along) or not math.isfinite(margin):
        raise PassCommitGeometryError("non-finite recomputed pass geometry")
    if logged_along_obstacle_m is not None:
        logged_along = float(logged_along_obstacle_m)
        if not math.isfinite(logged_along) or abs(logged_along - along) > tol:
            raise PassCommitGeometryError("logged along-obstacle geometry mismatch")
    if logged_passed_margin_m is not None:
        logged_margin = float(logged_passed_margin_m)
        if not math.isfinite(logged_margin) or abs(logged_margin - margin) > tol:
            raise PassCommitGeometryError("logged passed-margin geometry mismatch")
    return PassGeometry(
        direction_unit=direction,
        along_obstacle_m=along,
        obstacle_extent_m=extent,
        passed_margin_m=margin,
    )


@dataclass(frozen=True)
class PassCommitDecision:
    enabled: bool
    eligible: bool
    triggered: bool
    latched: bool
    active: bool
    inactive_reason: str
    direction_source: str
    direction_unit: tuple[float, float, float]
    passed_margin_m: float
    running_best_margin_m: float
    margin_delta_m: float
    regression_run: int
    estimated_lateral_exit_ready: bool
    pass_progress_velocity_before_mps: float
    pass_progress_velocity_after_mps: float
    requested_delta_v_mps: float
    applied_delta_v_mps: float
    requested_vector_mps: tuple[float, float, float]
    applied_vector_mps: tuple[float, float, float]
    permanently_ended: bool


class PassCommitNonRegression:
    """Stateful two-regression trigger followed by capped zero-target correction."""

    def __init__(self, *, enabled: bool, cap_mps: float) -> None:
        cap = float(cap_mps)
        if not math.isfinite(cap) or cap < 0.0 or cap > PASS_COMMIT_HARD_CAP_MPS:
            raise ValueError("cap_mps must be finite and within the 0.075 m/s authority")
        self.enabled = bool(enabled)
        self.cap_mps = cap
        self.reset()

    def reset(self) -> None:
        self._last_step: int | None = None
        self._lateral_exit_run = 0
        self._lateral_exit_ready = False
        self._previous_margin: float | None = None
        self._running_best: float | None = None
        self._regression_run = 0
        self._latched = False
        self._permanently_ended = False
        self._end_reason = ""

    def _end(self, reason: str) -> None:
        self._permanently_ended = True
        self._end_reason = reason
        self._latched = False
        self._regression_run = 0

    def evaluate(
        self,
        *,
        step: int,
        estimated_position_m: Iterable[float],
        goal_position_m: Iterable[float],
        obstacle_center_m: Iterable[float],
        obstacle_extent_m: float,
        pass_buffer_m: float,
        logged_along_obstacle_m: float,
        logged_passed_margin_m: float,
        estimated_velocity_mps: Iterable[float],
        baseline_candidate_delta_v_mps: Iterable[float],
        estimated_lateral_velocity_mps: float,
        encounter_context_established: bool,
        frozen_obstacle_matches: bool,
        relevant_obstacle_available: bool,
        relevant_obstacle_clearance_m: float,
        passed_obstacle: bool,
        passed_condition_now: bool,
        release_latched: bool,
        post_bypass_recovery: bool,
        goal_recovery_active: bool,
        lateral_exit_assist_active: bool,
        past_closest_approach: bool,
        clearance_increasing: bool,
        episode_done: bool,
    ) -> PassCommitDecision:
        """Evaluate one pre-action row. Simulator-actual state is not accepted."""

        current_step = int(step)
        if self._last_step is not None and current_step != self._last_step + 1:
            self._lateral_exit_run = 0
            self._lateral_exit_ready = False
            self._previous_margin = None
            self._running_best = None
            self._regression_run = 0
            self._latched = False
        self._last_step = current_step
        zero = (0.0, 0.0, 0.0)
        margin = float("nan")
        delta = float("nan")
        direction = zero
        before = float("nan")
        after = float("nan")
        requested = 0.0
        applied = 0.0
        requested_vector = zero
        applied_vector = zero
        triggered_now = False

        # Before a frozen encounter exists there is no PASS_COMMIT episode yet.
        # Once one exists, any loss of its obstacle identity permanently fails closed.
        if self._permanently_ended:
            reason = self._end_reason or "permanently_ended"
        elif not encounter_context_established:
            self._lateral_exit_run = 0
            reason = "encounter_not_established"
        elif not relevant_obstacle_available or not frozen_obstacle_matches:
            self._end("frozen_obstacle_unavailable_or_mismatch")
            reason = self._end_reason
        elif bool(episode_done):
            self._end("episode_done")
            reason = self._end_reason
        elif bool(passed_obstacle) or bool(passed_condition_now) or bool(release_latched):
            self._end("exact_pass_or_release_latched")
            reason = self._end_reason
        elif bool(post_bypass_recovery):
            self._end("post_bypass_recovery")
            reason = self._end_reason
        elif bool(goal_recovery_active):
            self._end("goal_recovery_active")
            reason = self._end_reason
        elif not math.isfinite(float(relevant_obstacle_clearance_m)):
            self._end("relevant_obstacle_clearance_unavailable")
            reason = self._end_reason
        elif float(relevant_obstacle_clearance_m) < PASS_COMMIT_CLEARANCE_FLOOR_M:
            self._end("clearance_below_2p5_m")
            reason = self._end_reason
        else:
            try:
                geometry = derive_pass_geometry(
                    estimated_position_m=estimated_position_m,
                    goal_position_m=goal_position_m,
                    obstacle_center_m=obstacle_center_m,
                    obstacle_extent_m=obstacle_extent_m,
                    pass_buffer_m=pass_buffer_m,
                    logged_along_obstacle_m=logged_along_obstacle_m,
                    logged_passed_margin_m=logged_passed_margin_m,
                )
                velocity = _vector3(estimated_velocity_mps, name="estimated_velocity_mps")
                candidate = _vector3(
                    baseline_candidate_delta_v_mps,
                    name="baseline_candidate_delta_v_mps",
                )
                lateral = float(estimated_lateral_velocity_mps)
                if not math.isfinite(lateral):
                    raise PassCommitGeometryError("non-finite estimated lateral velocity")
                margin = geometry.passed_margin_m
                direction = geometry.direction_unit
                if abs(lateral) <= PASS_COMMIT_LATERAL_EXIT_THRESHOLD_MPS:
                    self._lateral_exit_run += 1
                else:
                    self._lateral_exit_run = 0
                if self._lateral_exit_run >= PASS_COMMIT_LATERAL_EXIT_STEPS:
                    self._lateral_exit_ready = True
                if self._lateral_exit_ready:
                    if self._previous_margin is not None:
                        delta = margin - self._previous_margin
                        if delta < -PASS_COMMIT_TRIGGER_DEADBAND_M:
                            self._regression_run += 1
                        else:
                            self._regression_run = 0
                        if self._regression_run >= 2:
                            triggered_now = not self._latched
                            self._latched = True
                    else:
                        self._running_best = margin
                    self._running_best = (
                        margin if self._running_best is None
                        else max(self._running_best, margin)
                    )
                    self._previous_margin = margin
                else:
                    self._previous_margin = None
                    self._running_best = None
                    self._regression_run = 0
                before = _dot(
                    (velocity[index] + candidate[index] for index in range(3)),
                    direction,
                )
                after = before
                if not math.isfinite(before):
                    raise PassCommitGeometryError("non-finite pass-progress component")
            except (PassCommitGeometryError, TypeError, ValueError, OverflowError):
                self._end("pass_geometry_fail_closed")
                reason = self._end_reason
                self._lateral_exit_ready = False

        if not self._permanently_ended:
            eligible_gates = {
                "estimated_lateral_exit_not_ready": self._lateral_exit_ready,
                "pass_commit_not_latched": self._latched,
                "not_past_closest_approach": bool(past_closest_approach),
                "clearance_not_increasing": bool(clearance_increasing),
                "lateral_exit_assistance_active": not bool(lateral_exit_assist_active),
                "episode_done": not bool(episode_done),
                "exact_pass_or_release_latched": not bool(
                    passed_obstacle or passed_condition_now or release_latched
                ),
                "goal_recovery_active": not bool(goal_recovery_active),
                "clearance_below_2p5_m": (
                    math.isfinite(float(relevant_obstacle_clearance_m))
                    and float(relevant_obstacle_clearance_m) >= PASS_COMMIT_CLEARANCE_FLOOR_M
                ),
            }
            eligible = all(eligible_gates.values())
            if eligible and before < 0.0:
                requested = -before
                requested_vector = tuple(requested * item for item in direction)
                if self.enabled:
                    applied = min(requested, self.cap_mps)
                    applied_vector = tuple(applied * item for item in direction)
                    after = before + applied
            if not self.enabled:
                reason = "disabled_off_condition"
            elif lateral_exit_assist_active:
                reason = "lateral_exit_assistance_active"
            elif not self._lateral_exit_ready:
                reason = "estimated_lateral_exit_not_ready"
            elif not self._latched:
                reason = "two_step_margin_regression_not_latched"
            elif not past_closest_approach:
                reason = "not_past_closest_approach"
            elif not clearance_increasing:
                reason = "clearance_not_increasing"
            elif before >= 0.0:
                reason = "candidate_pass_progress_nonnegative"
            elif applied <= 0.0:
                reason = "zero_authority"
            elif not eligible:
                reason = next(
                    name for name, passed in eligible_gates.items() if not passed
                )
            else:
                reason = "active"
        else:
            eligible = False
            reason = self._end_reason
        active = bool(self.enabled and eligible and applied > 0.0)
        return PassCommitDecision(
            enabled=self.enabled,
            eligible=bool(eligible),
            triggered=bool(triggered_now),
            latched=bool(self._latched),
            active=active,
            inactive_reason=str(reason),
            direction_source=PASS_COMMIT_DIRECTION_SOURCE,
            direction_unit=direction,
            passed_margin_m=margin,
            running_best_margin_m=(
                float(self._running_best)
                if self._running_best is not None else float("nan")
            ),
            margin_delta_m=delta,
            regression_run=int(self._regression_run),
            estimated_lateral_exit_ready=bool(self._lateral_exit_ready),
            pass_progress_velocity_before_mps=before,
            pass_progress_velocity_after_mps=after,
            requested_delta_v_mps=requested,
            applied_delta_v_mps=applied,
            requested_vector_mps=requested_vector,
            applied_vector_mps=applied_vector,
            permanently_ended=bool(self._permanently_ended),
        )


def pass_margin_direction_proof(
    *,
    estimated_position_m: Iterable[float],
    goal_position_m: Iterable[float],
    obstacle_center_m: Iterable[float],
    obstacle_extent_m: float,
    pass_buffer_m: float,
    epsilon_m: float = 1.0e-5,
) -> tuple[float, float]:
    """Return the margin before/after a small step along the derived direction."""

    geometry = derive_pass_geometry(
        estimated_position_m=estimated_position_m,
        goal_position_m=goal_position_m,
        obstacle_center_m=obstacle_center_m,
        obstacle_extent_m=obstacle_extent_m,
        pass_buffer_m=pass_buffer_m,
    )
    epsilon = float(epsilon_m)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon_m must be positive and finite")
    position = _vector3(estimated_position_m, name="estimated_position_m")
    shifted = tuple(
        position[index] + epsilon * geometry.direction_unit[index]
        for index in range(3)
    )
    after = derive_pass_geometry(
        estimated_position_m=shifted,
        goal_position_m=goal_position_m,
        obstacle_center_m=obstacle_center_m,
        obstacle_extent_m=obstacle_extent_m,
        pass_buffer_m=pass_buffer_m,
    )
    return geometry.passed_margin_m, after.passed_margin_m


__all__ = [
    "PASS_COMMIT_CLEARANCE_FLOOR_M",
    "PASS_COMMIT_DIRECTION_SOURCE",
    "PASS_COMMIT_HARD_CAP_MPS",
    "PASS_COMMIT_LATERAL_EXIT_STEPS",
    "PASS_COMMIT_LATERAL_EXIT_THRESHOLD_MPS",
    "PASS_COMMIT_TRIGGER_DEADBAND_M",
    "PassCommitDecision",
    "PassCommitGeometryError",
    "PassCommitNonRegression",
    "PassGeometry",
    "derive_pass_geometry",
    "pass_margin_direction_proof",
]
