"""Estimated-state-only post-bypass goal-recovery action correction.

This module is intentionally independent of the simulator and learned policy.
It never receives a UAV truth state and never mutates position, velocity, yaw,
or a next state.  Its only control output is a bounded physical delta-v that a
caller may fuse upstream of the existing action/safety/dynamics pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class GoalRecoveryCorrectionConfig:
    enabled: bool = False
    radial_gain: float = 0.8
    tangential_gain: float = 0.7
    max_delta_v_mps: float = 0.6
    activation_mode: str = "post_bypass_only"
    alpha_ramp_enabled: bool = True
    max_acceleration_mps2: float = 3.0
    goal_threshold_m: float = 1.0
    minimum_goal_speed_mps: float = 0.35
    cruise_speed_mps: float = 6.0
    terminal_capture_enabled: bool = False
    terminal_capture_radius_m: float = 8.0
    terminal_radial_speed_floor_mps: float = 0.25
    terminal_radial_brake_gain: float = 0.8
    terminal_tangential_gain: float = 1.0
    terminal_propulsion_fade_enabled: bool = True
    terminal_smooth_blend: bool = True
    terminal_capture_v2_enabled: bool = False
    terminal_effective_deceleration_mps2: float = 1.697726
    terminal_stopping_margin_m: float = 1.328009
    terminal_hold_inside_goal_enabled: bool = True
    terminal_radial_conflict_arbitration_enabled: bool = False
    terminal_radial_conflict_compensation_gain: float = 1.0
    # v2.2 is deliberately opt-in and has no tunable restorative gain.  The
    # fixed 0.25 m/s floor is part of the terminal-hold contract.
    terminal_capture_v22_enabled: bool = False
    terminal_hold_enter_radius_m: float = 0.98
    terminal_hold_release_radius_m: float = 1.05
    terminal_hold_restorative_enabled: bool = True
    # v2.3 is deliberately opt-in.  It adds a fixed target-radius guard to
    # the existing v2.2 latch without changing any v2/v2.1/v2.2 arithmetic.
    terminal_capture_v23_enabled: bool = False
    terminal_hold_target_radius_m: float = 0.96
    epsilon: float = 1.0e-9

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if self.activation_mode != "post_bypass_only":
            raise ValueError("activation_mode must be post_bypass_only")
        if not isinstance(self.alpha_ramp_enabled, bool):
            raise ValueError("alpha_ramp_enabled must be boolean")
        if not isinstance(self.terminal_capture_enabled, bool):
            raise ValueError("terminal_capture_enabled must be boolean")
        if not isinstance(self.terminal_propulsion_fade_enabled, bool):
            raise ValueError("terminal_propulsion_fade_enabled must be boolean")
        if not isinstance(self.terminal_smooth_blend, bool):
            raise ValueError("terminal_smooth_blend must be boolean")
        if not isinstance(self.terminal_capture_v2_enabled, bool):
            raise ValueError("terminal_capture_v2_enabled must be boolean")
        if not isinstance(self.terminal_hold_inside_goal_enabled, bool):
            raise ValueError("terminal_hold_inside_goal_enabled must be boolean")
        if not isinstance(self.terminal_radial_conflict_arbitration_enabled, bool):
            raise ValueError(
                "terminal_radial_conflict_arbitration_enabled must be boolean"
            )
        if not isinstance(self.terminal_capture_v22_enabled, bool):
            raise ValueError("terminal_capture_v22_enabled must be boolean")
        if not isinstance(self.terminal_hold_restorative_enabled, bool):
            raise ValueError("terminal_hold_restorative_enabled must be boolean")
        if not isinstance(self.terminal_capture_v23_enabled, bool):
            raise ValueError("terminal_capture_v23_enabled must be boolean")
        for name in (
            "radial_gain",
            "tangential_gain",
            "max_delta_v_mps",
            "max_acceleration_mps2",
            "goal_threshold_m",
            "minimum_goal_speed_mps",
            "cruise_speed_mps",
            "terminal_capture_radius_m",
            "terminal_radial_speed_floor_mps",
            "terminal_radial_brake_gain",
            "terminal_tangential_gain",
            "terminal_effective_deceleration_mps2",
            "terminal_stopping_margin_m",
            "terminal_radial_conflict_compensation_gain",
            "terminal_hold_enter_radius_m",
            "terminal_hold_release_radius_m",
            "terminal_hold_target_radius_m",
            "epsilon",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_delta_v_mps <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("max_delta_v_mps and epsilon must be positive")
        if self.cruise_speed_mps < self.minimum_goal_speed_mps:
            raise ValueError("cruise_speed_mps must be at least minimum_goal_speed_mps")
        if self.terminal_capture_radius_m <= self.goal_threshold_m:
            raise ValueError("terminal_capture_radius_m must exceed goal_threshold_m")
        if not (
            0.0 <= self.terminal_hold_enter_radius_m <= self.goal_threshold_m
            < self.terminal_hold_release_radius_m
        ):
            raise ValueError(
                "terminal hold radii must satisfy "
                "0 <= enter <= goal_threshold < release"
            )
        if not (
            0.0 <= self.terminal_hold_target_radius_m
            < self.terminal_hold_enter_radius_m
            < self.terminal_hold_release_radius_m
        ):
            raise ValueError(
                "terminal hold target, enter, and release radii must satisfy "
                "0 <= target < enter < release"
            )


@dataclass(frozen=True)
class GoalRelativeMotion:
    goal_distance_m: float
    goal_direction: np.ndarray
    current_radial_velocity_mps: float
    current_tangential_velocity_vector_mps: np.ndarray
    current_tangential_velocity_mps: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class GoalRecoveryCorrectionResult:
    correction_delta_v_mps: np.ndarray
    radial_correction_delta_v_mps: np.ndarray
    tangential_correction_delta_v_mps: np.ndarray
    correction_alpha: float
    goal_distance_m: float
    goal_direction: np.ndarray
    current_radial_velocity_mps: float
    current_tangential_velocity_vector_mps: np.ndarray
    current_tangential_velocity_mps: float
    desired_radial_velocity_mps: float
    active: bool
    eligible: bool
    activation_reason: str
    correction_norm_limited: bool
    state_source: str = "ekf_estimated_position_velocity"
    terminal_capture_enabled: bool = False
    terminal_capture_active: bool = False
    terminal_capture_radius_m: float = 0.0
    terminal_blend: float = 0.0
    base_desired_radial_velocity_mps: float = 0.0
    terminal_desired_radial_velocity_mps: float = 0.0
    radial_velocity_error_mps: float = 0.0
    radial_propulsion_delta_v_mps: np.ndarray | None = None
    radial_propulsion_scale: float = 1.0
    radial_braking_delta_v_mps: np.ndarray | None = None
    effective_terminal_radial_brake_gain: float = 0.0
    base_tangential_gain: float = 0.0
    effective_terminal_tangential_gain: float = 0.0
    terminal_tangential_damping_delta_v_mps: np.ndarray | None = None
    terminal_raw_correction_delta_v_mps: np.ndarray | None = None
    terminal_raw_correction_norm_mps: float = 0.0
    terminal_correction_clipped: bool = False
    terminal_correction_clip_ratio: float = 1.0
    terminal_v2_enabled: bool = False
    terminal_v2_governor_eligible: bool = False
    terminal_v2_governor_active: bool = False
    terminal_v2_a_eff_mps2: float = 0.0
    terminal_v2_stopping_margin_m: float = 0.0
    terminal_v2_distance_to_threshold_m: float = 0.0
    terminal_v2_usable_braking_distance_m: float = 0.0
    terminal_v2_base_desired_radial_speed_mps: float = 0.0
    terminal_v2_envelope_speed_mps: float = 0.0
    terminal_v2_governed_desired_radial_speed_mps: float = 0.0
    terminal_v2_radial_overspeed_mps: float = 0.0
    terminal_v2_braking_required: bool = False
    terminal_v2_inside_goal_hold: bool = False
    terminal_v2_anti_stall_active: bool = False
    terminal_v21_radial_conflict_enabled: bool = False
    terminal_v21_radial_conflict_eligible: bool = False
    terminal_v21_radial_conflict_active: bool = False
    terminal_v21_inactive_reason: str = "disabled"
    terminal_v21_goal_direction_source: str = "estimated_goal_direction"
    terminal_v21_actor_radial_delta_v_mps: float = 0.0
    terminal_v21_actor_opposing_radial_mps: float = 0.0
    terminal_v21_compensation_gain: float = 1.0
    terminal_v21_conflict_compensation_delta_v_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    terminal_v21_conflict_compensation_norm_mps: float = 0.0
    terminal_v21_conflict_compensation_radial_mps: float = 0.0
    terminal_v21_raw_correction_before_conflict_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    terminal_v21_raw_correction_after_conflict_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    terminal_v21_correction_clip_ratio: float = 1.0
    terminal_v21_compensation_post_cap_radial_mps: float = 0.0
    # v2.2 terminal-hold diagnostics.  Defaults are intentionally neutral so
    # disabled and ineligible paths remain structurally compatible.
    terminal_v22_enabled: bool = False
    terminal_v22_hold_latched: bool = False
    terminal_v22_hold_latched_before: bool = False
    terminal_v22_hold_enter_radius_m: float = 0.0
    terminal_v22_hold_release_radius_m: float = 0.0
    terminal_v22_hold_enter_event: bool = False
    terminal_v22_hold_release_event: bool = False
    terminal_v22_hold_mode: str = "inactive"
    terminal_v22_restorative_active: bool = False
    terminal_v22_distance_outside_stable_m: float = 0.0
    terminal_v22_restore_fraction: float = 0.0
    terminal_v22_restorative_desired_radial_speed_mps: float = 0.0
    terminal_v22_restorative_radial_error_mps: float = 0.0
    terminal_v22_restorative_delta_v_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    terminal_v22_inside_canonical_goal: bool = False
    # v2.3 target-radius guard diagnostics.  These fields are neutral unless
    # the complete v2/v2.1/v2.2/v2.3 gate is open.
    terminal_v23_enabled: bool = False
    terminal_v23_hold_target_radius_m: float = 0.0
    terminal_v23_guard_active: bool = False
    terminal_v23_guard_fraction: float = 0.0
    terminal_v23_desired_radial_speed_mps: float = 0.0
    terminal_v23_preboundary_active: bool = False
    terminal_v23_postboundary_active: bool = False
    terminal_v23_distance_to_target_band_m: float = 0.0
    terminal_v23_radial_error_mps: float = 0.0
    terminal_v23_positive_radial_error: bool = False
    terminal_v23_radial_propulsion_allowed: bool = False
    terminal_v23_radial_propulsion_suppressed: bool = False
    terminal_v23_radial_propulsion_suppression_reason: str = "not_v23_guard"
    terminal_v23_requested_propulsion_radial_mps: float = 0.0
    terminal_v23_executed_propulsion_radial_mps: float = 0.0
    terminal_v23_authority_active: bool = False
    terminal_v23_preboundary_authority_repair_active: bool = False
    terminal_v23_net_tracking_radial_before_v21_mps: float = 0.0
    terminal_v23_net_tracking_radial_after_v21_mps: float = 0.0


@dataclass(frozen=True)
class GoalRecoveryActionFusion:
    actor_raw_delta_v_mps: np.ndarray
    corrected_delta_v_pre_action_transform_mps: np.ndarray
    corrected_delta_v_pre_safety_mps: np.ndarray
    transform_applied: bool


def compute_desired_goal_speed(
    goal_distance_m: float,
    *,
    max_acceleration_mps2: float,
    goal_threshold_m: float = 1.0,
    minimum_speed_mps: float = 0.35,
    cruise_speed_mps: float = 6.0,
) -> float:
    """Return the existing Teacher's braking-distance goal-speed profile."""

    distance = float(goal_distance_m)
    acceleration = float(max_acceleration_mps2)
    if not np.isfinite(distance) or not np.isfinite(acceleration):
        return 0.0
    raw = np.sqrt(max(0.0, 2.0 * acceleration * max(distance - float(goal_threshold_m), 0.0)))
    return float(np.clip(raw, float(minimum_speed_mps), float(cruise_speed_mps)))


def decompose_goal_relative_motion(
    estimated_position_m: Any,
    estimated_velocity_mps: Any,
    goal_position_m: Any,
    *,
    epsilon: float = 1.0e-9,
) -> GoalRelativeMotion:
    """Read-only decomposition of EKF-estimated motion relative to a goal."""

    zero = np.zeros(3, dtype=float)
    try:
        position = np.asarray(estimated_position_m, dtype=float).reshape(3)
        velocity = np.asarray(estimated_velocity_mps, dtype=float).reshape(3)
        goal = np.asarray(goal_position_m, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return GoalRelativeMotion(0.0, zero, 0.0, zero, 0.0, False, "invalid_vector_shape")
    if not (np.isfinite(position).all() and np.isfinite(velocity).all() and np.isfinite(goal).all()):
        return GoalRelativeMotion(0.0, zero, 0.0, zero, 0.0, False, "non_finite_estimated_state")
    error = goal - position
    distance = float(np.linalg.norm(error))
    if not np.isfinite(distance) or distance <= float(epsilon):
        return GoalRelativeMotion(max(0.0, distance) if np.isfinite(distance) else 0.0, zero, 0.0, zero, 0.0, False, "goal_distance_degenerate")
    direction = error / distance
    radial = float(np.dot(velocity, direction))
    tangential = velocity - radial * direction
    return GoalRelativeMotion(
        distance,
        direction,
        radial,
        tangential,
        float(np.linalg.norm(tangential)),
        True,
        "valid",
    )


def _smooth_alpha(pass_progress: float, enabled: bool) -> float:
    if not enabled:
        return 1.0
    progress = float(pass_progress)
    if not np.isfinite(progress):
        return 1.0
    normalized = float(np.clip((progress - 0.5) / 0.5, 0.0, 1.0))
    return float(normalized * normalized * (3.0 - 2.0 * normalized))


def compute_terminal_blend(
    goal_distance_m: float,
    *,
    terminal_capture_radius_m: float,
    goal_threshold_m: float,
    smooth: bool = True,
) -> float:
    """Return a continuous near-goal blend: zero at radius, one at goal."""

    distance = float(goal_distance_m)
    radius = float(terminal_capture_radius_m)
    threshold = float(goal_threshold_m)
    if not all(np.isfinite(value) for value in (distance, radius, threshold)):
        raise ValueError("terminal blend inputs must be finite")
    if radius <= threshold:
        raise ValueError("terminal_capture_radius_m must exceed goal_threshold_m")
    inward = float(np.clip((radius - distance) / (radius - threshold), 0.0, 1.0))
    if smooth:
        inward = inward * inward * (3.0 - 2.0 * inward)
    return float(inward)


def compute_terminal_desired_radial_speed(
    goal_distance_m: float,
    *,
    base_desired_speed_mps: float,
    terminal_capture_radius_m: float,
    goal_threshold_m: float,
    terminal_radial_speed_floor_mps: float,
    base_speed_at_terminal_radius_mps: float,
    smooth: bool = True,
) -> tuple[float, float]:
    """Cap the legacy desired speed with a smooth, inward-decreasing envelope."""

    base = float(base_desired_speed_mps)
    floor = float(terminal_radial_speed_floor_mps)
    boundary_speed = float(base_speed_at_terminal_radius_mps)
    if not all(np.isfinite(value) and value >= 0.0 for value in (base, floor, boundary_speed)):
        raise ValueError("terminal desired-speed inputs must be finite and non-negative")
    blend = compute_terminal_blend(
        goal_distance_m,
        terminal_capture_radius_m=terminal_capture_radius_m,
        goal_threshold_m=goal_threshold_m,
        smooth=smooth,
    )
    envelope = floor + (boundary_speed - floor) * (1.0 - blend)
    return float(min(base, max(0.0, envelope))), blend


def compute_terminal_v2_envelope(
    goal_distance_m: float,
    *,
    base_desired_speed_mps: float,
    goal_threshold_m: float,
    effective_deceleration_mps2: float | None = None,
    stopping_margin_m: float | None = None,
    radial_speed_floor_mps: float | None = None,
    hold_inside_goal_enabled: bool = True,
    terminal_effective_deceleration_mps2: float | None = None,
    terminal_stopping_margin_m: float | None = None,
    terminal_radial_speed_floor_mps: float | None = None,
    terminal_hold_inside_goal_enabled: bool | None = None,
) -> float:
    """Return the pure v2 stopping envelope, capped by the base speed.

    The envelope leaves a configurable stopping margin between the point at
    which braking is planned and the goal threshold.  Inside the threshold a
    zero-speed hold is used when enabled; just outside it the radial-speed
    floor prevents a discontinuous stall.
    """

    if effective_deceleration_mps2 is None:
        effective_deceleration_mps2 = terminal_effective_deceleration_mps2
    if stopping_margin_m is None:
        stopping_margin_m = terminal_stopping_margin_m
    if radial_speed_floor_mps is None:
        radial_speed_floor_mps = terminal_radial_speed_floor_mps
    if terminal_hold_inside_goal_enabled is not None:
        hold_inside_goal_enabled = terminal_hold_inside_goal_enabled
    if (
        effective_deceleration_mps2 is None
        or stopping_margin_m is None
        or radial_speed_floor_mps is None
    ):
        raise ValueError("v2 envelope deceleration, margin, and floor are required")
    distance = float(goal_distance_m)
    base = float(base_desired_speed_mps)
    acceleration = float(effective_deceleration_mps2)
    margin = float(stopping_margin_m)
    threshold = float(goal_threshold_m)
    floor = float(radial_speed_floor_mps)
    if not all(
        np.isfinite(value) and value >= 0.0
        for value in (distance, base, acceleration, margin, threshold, floor)
    ):
        raise ValueError("v2 envelope inputs must be finite and non-negative")
    if not isinstance(hold_inside_goal_enabled, bool):
        raise ValueError("hold_inside_goal_enabled must be boolean")
    if hold_inside_goal_enabled and distance <= threshold:
        return 0.0
    usable_distance = max(distance - threshold - margin, 0.0)
    envelope = np.sqrt(max(floor * floor + 2.0 * acceleration * usable_distance, 0.0))
    return float(min(base, envelope))


# Descriptive aliases keep the helper discoverable alongside the legacy
# terminal helpers while retaining one pure implementation.
compute_terminal_v2_envelope_speed = compute_terminal_v2_envelope
compute_terminal_capture_v2_envelope = compute_terminal_v2_envelope
compute_terminal_capture_v2_envelope_speed = compute_terminal_v2_envelope


TERMINAL_V22_RESTORATIVE_FLOOR_MPS = 0.25
TERMINAL_HOLD_RESTORATIVE_FLOOR_MPS = TERMINAL_V22_RESTORATIVE_FLOOR_MPS


@dataclass(frozen=True)
class TerminalV22Decision:
    """Pure state transition for the v2.2 canonical terminal hold.

    The controller owns the episode-local latch.  This helper only evaluates
    the transition and the associated speed profile, so it is straightforward
    to exercise without a simulator or an action pipeline.
    """

    latched_before: bool
    latched_after: bool
    enter_event: bool
    release_event: bool
    mode: str
    inside_canonical_goal: bool
    distance_outside_stable_m: float
    restore_fraction: float
    restorative_desired_radial_speed_mps: float
    restorative_active: bool = False

    # Naming aliases make the pure result as explicit as the controller
    # telemetry without duplicating state in the dataclass.
    @property
    def hold_latched_before(self) -> bool:
        return self.latched_before

    @property
    def hold_latched_after(self) -> bool:
        return self.latched_after

    @property
    def hold_enter_event(self) -> bool:
        return self.enter_event

    @property
    def hold_release_event(self) -> bool:
        return self.release_event


def compute_terminal_v22_decision(
    goal_distance_m: float,
    *,
    latched_before: bool,
    enter_radius_m: float = 0.98,
    goal_threshold_m: float = 1.0,
    release_radius_m: float = 1.05,
    restorative_enabled: bool = True,
    radial_speed_floor_mps: float = TERMINAL_V22_RESTORATIVE_FLOOR_MPS,
) -> TerminalV22Decision:
    """Evaluate one pure v2.2 terminal-hold transition.

    Entering is hysteretic (``distance <= enter_radius_m``), while release is
    inclusive (``distance >= release_radius_m``).  The restorative profile
    reuses the existing terminal radial-speed floor; there is intentionally no
    additional tunable gain.
    """

    distance = float(goal_distance_m)
    enter = float(enter_radius_m)
    threshold = float(goal_threshold_m)
    release = float(release_radius_m)
    floor = float(radial_speed_floor_mps)
    if not all(
        np.isfinite(value)
        for value in (distance, enter, threshold, release, floor)
    ):
        raise ValueError("v2.2 decision inputs must be finite")
    if distance < 0.0 or floor < 0.0:
        raise ValueError(
            "goal_distance_m and radial_speed_floor_mps must be non-negative"
        )
    if not (0.0 <= enter <= threshold < release):
        raise ValueError(
            "terminal hold radii must satisfy "
            "0 <= enter <= goal_threshold < release"
        )
    if not isinstance(latched_before, bool):
        raise ValueError("latched_before must be boolean")
    if not isinstance(restorative_enabled, bool):
        raise ValueError("restorative_enabled must be boolean")

    before = latched_before
    after = before
    enter_event = False
    release_event = False
    if not before and distance <= enter:
        after = True
        enter_event = True
        mode = "capture"
    elif before and distance >= release:
        after = False
        release_event = True
        mode = "released_capture"
    elif after and distance <= threshold:
        mode = "inside_canonical"
    elif after:
        mode = "restorative"
    else:
        mode = "inactive"

    inside_canonical = bool(after and distance <= threshold)
    distance_outside = max(distance - threshold, 0.0)
    restore_fraction = float(
        np.clip((distance - threshold) / (release - threshold), 0.0, 1.0)
    )
    restorative_active = bool(mode == "restorative" and restorative_enabled)
    restorative_desired = float(
        floor * restore_fraction
        if restorative_active
        else 0.0
    )
    return TerminalV22Decision(
        latched_before=before,
        latched_after=after,
        enter_event=enter_event,
        release_event=release_event,
        mode=mode,
        inside_canonical_goal=inside_canonical,
        distance_outside_stable_m=float(distance_outside),
        restore_fraction=restore_fraction,
        restorative_desired_radial_speed_mps=restorative_desired,
        restorative_active=restorative_active,
    )


# Keep the naming discoverable for callers that refer to the feature as a
# terminal hold rather than by its version number.
compute_terminal_hold_v22_decision = compute_terminal_v22_decision
TerminalHoldV22Decision = TerminalV22Decision
compute_terminal_capture_v22_decision = compute_terminal_v22_decision
TerminalCaptureV22Decision = TerminalV22Decision


@dataclass(frozen=True)
class TerminalV23Decision:
    """Pure v2.3 target-radius restorative guard profile."""

    guard_active: bool
    guard_fraction: float
    desired_radial_speed_mps: float
    preboundary_active: bool
    postboundary_active: bool
    distance_to_target_band_m: float


def compute_terminal_v23_decision(
    goal_distance_m: float,
    *,
    target_radius_m: float = 0.96,
    goal_threshold_m: float = 1.0,
    release_radius_m: float = 1.05,
    hold_latched: bool = True,
    restorative_speed_cap_mps: float = TERMINAL_V22_RESTORATIVE_FLOOR_MPS,
) -> TerminalV23Decision:
    """Evaluate one pure v2.3 guard decision.

    The guard is intentionally a single linear profile over ``[target,
    release)``.  It is only active while the v2.2 latch remains set; v2.2's
    inclusive release transition therefore owns the release-boundary call.
    """

    distance = float(goal_distance_m)
    target = float(target_radius_m)
    threshold = float(goal_threshold_m)
    release = float(release_radius_m)
    cap = float(restorative_speed_cap_mps)
    if not all(np.isfinite(value) for value in (distance, target, threshold, release, cap)):
        raise ValueError("v2.3 decision inputs must be finite")
    if distance < 0.0 or target < 0.0 or cap < 0.0:
        raise ValueError("v2.3 distance, target, and cap must be non-negative")
    if not (0.0 <= target < threshold < release):
        raise ValueError("v2.3 radii must satisfy 0 <= target < threshold < release")
    if not isinstance(hold_latched, bool):
        raise ValueError("hold_latched must be boolean")
    distance_to_target = max(distance - target, 0.0)
    # A guard is active only when it has positive inward authority.  At or
    # below the target the commanded speed is exactly zero and telemetry must
    # remain inactive; the v2.2 canonical hold still owns that region.
    active = bool(hold_latched and target < distance < release)
    fraction = float(
        np.clip((distance - target) / (release - target), 0.0, 1.0)
        if active
        else 0.0
    )
    desired = float(cap * fraction if active else 0.0)
    preboundary = bool(active and target < distance < threshold)
    postboundary = bool(active and threshold <= distance < release)
    return TerminalV23Decision(
        guard_active=active,
        guard_fraction=fraction,
        desired_radial_speed_mps=desired,
        preboundary_active=preboundary,
        postboundary_active=postboundary,
        distance_to_target_band_m=float(distance_to_target),
    )


def compute_terminal_v23_desired_radial_speed(
    goal_distance_m: float,
    *,
    target_radius_m: float = 0.96,
    release_radius_m: float = 1.05,
    hold_latched: bool = True,
    radial_speed_cap_mps: float = TERMINAL_V22_RESTORATIVE_FLOOR_MPS,
) -> float:
    """Return only the v2.3 unified desired inward radial speed."""

    return compute_terminal_v23_decision(
        goal_distance_m,
        target_radius_m=target_radius_m,
        release_radius_m=release_radius_m,
        hold_latched=hold_latched,
        restorative_speed_cap_mps=radial_speed_cap_mps,
    ).desired_radial_speed_mps


# Descriptive aliases for callers/tests that use the feature name instead of
# its version number.
compute_terminal_capture_v23_decision = compute_terminal_v23_decision
compute_terminal_hold_v23_decision = compute_terminal_v23_decision
compute_terminal_capture_v23_desired_radial_speed = compute_terminal_v23_desired_radial_speed
TerminalCaptureV23Decision = TerminalV23Decision


def _zero_result(
    reason: str,
    *,
    eligible: bool = False,
    config: GoalRecoveryCorrectionConfig | None = None,
) -> GoalRecoveryCorrectionResult:
    zero = np.zeros(3, dtype=float)
    return GoalRecoveryCorrectionResult(
        correction_delta_v_mps=zero.copy(),
        radial_correction_delta_v_mps=zero.copy(),
        tangential_correction_delta_v_mps=zero.copy(),
        correction_alpha=0.0,
        goal_distance_m=0.0,
        goal_direction=zero.copy(),
        current_radial_velocity_mps=0.0,
        current_tangential_velocity_vector_mps=zero.copy(),
        current_tangential_velocity_mps=0.0,
        desired_radial_velocity_mps=0.0,
        active=False,
        eligible=eligible,
        activation_reason=reason,
        correction_norm_limited=False,
        terminal_capture_enabled=bool(config and config.terminal_capture_enabled),
        terminal_capture_radius_m=float(config.terminal_capture_radius_m) if config else 0.0,
        terminal_v2_enabled=bool(config and config.terminal_capture_v2_enabled),
        terminal_v2_a_eff_mps2=(
            float(config.terminal_effective_deceleration_mps2) if config else 0.0
        ),
        terminal_v2_stopping_margin_m=(float(config.terminal_stopping_margin_m) if config else 0.0),
        base_tangential_gain=float(config.tangential_gain) if config else 0.0,
        effective_terminal_tangential_gain=float(config.tangential_gain) if config else 0.0,
        radial_propulsion_delta_v_mps=zero.copy(),
        radial_braking_delta_v_mps=zero.copy(),
        terminal_tangential_damping_delta_v_mps=zero.copy(),
        terminal_raw_correction_delta_v_mps=zero.copy(),
        terminal_v21_radial_conflict_enabled=bool(
            config and config.terminal_radial_conflict_arbitration_enabled
        ),
        terminal_v21_radial_conflict_eligible=False,
        terminal_v21_radial_conflict_active=False,
        terminal_v21_inactive_reason=(
            reason
            if config and config.terminal_radial_conflict_arbitration_enabled
            else "disabled"
        ),
        terminal_v21_goal_direction_source="unavailable",
        terminal_v21_actor_radial_delta_v_mps=0.0,
        terminal_v21_actor_opposing_radial_mps=0.0,
        terminal_v21_compensation_gain=(
            float(config.terminal_radial_conflict_compensation_gain)
            if config
            else 1.0
        ),
        terminal_v21_conflict_compensation_delta_v_mps=zero.copy(),
        terminal_v21_conflict_compensation_norm_mps=0.0,
        terminal_v21_conflict_compensation_radial_mps=0.0,
        terminal_v21_raw_correction_before_conflict_mps=zero.copy(),
        terminal_v21_raw_correction_after_conflict_mps=zero.copy(),
        terminal_v21_correction_clip_ratio=1.0,
        terminal_v21_compensation_post_cap_radial_mps=0.0,
        terminal_v22_enabled=bool(config and config.terminal_capture_v22_enabled),
        terminal_v22_hold_latched=False,
        terminal_v22_hold_latched_before=False,
        terminal_v22_hold_enter_radius_m=(
            float(config.terminal_hold_enter_radius_m) if config else 0.0
        ),
        terminal_v22_hold_release_radius_m=(
            float(config.terminal_hold_release_radius_m) if config else 0.0
        ),
        terminal_v22_hold_enter_event=False,
        terminal_v22_hold_release_event=False,
        terminal_v22_hold_mode="inactive",
        terminal_v22_restorative_active=False,
        terminal_v22_distance_outside_stable_m=0.0,
        terminal_v22_restore_fraction=0.0,
        terminal_v22_restorative_desired_radial_speed_mps=0.0,
        terminal_v22_restorative_radial_error_mps=0.0,
        terminal_v22_restorative_delta_v_mps=zero.copy(),
        terminal_v22_inside_canonical_goal=False,
        terminal_v23_enabled=bool(config and config.terminal_capture_v23_enabled),
        terminal_v23_hold_target_radius_m=(
            float(config.terminal_hold_target_radius_m) if config else 0.0
        ),
        terminal_v23_guard_active=False,
        terminal_v23_guard_fraction=0.0,
        terminal_v23_desired_radial_speed_mps=0.0,
        terminal_v23_preboundary_active=False,
        terminal_v23_postboundary_active=False,
        terminal_v23_distance_to_target_band_m=0.0,
        terminal_v23_radial_error_mps=0.0,
    )


class GoalRecoveryCorrection:
    """Compute bounded radial recovery plus tangential damping."""

    def __init__(self, config: GoalRecoveryCorrectionConfig) -> None:
        self.config = config
        # Episode-local state: callers must invoke reset() at episode start.
        self.terminal_hold_latched = False

    def reset(self) -> None:
        """Clear v2.2 terminal-hold state for a new episode."""

        self.terminal_hold_latched = False

    def compute(
        self,
        *,
        estimated_position_m: Any,
        estimated_velocity_mps: Any,
        goal_position_m: Any,
        release_latched: bool,
        post_bypass_recovery: bool,
        pass_progress: float = float("nan"),
        action_provenance: str = "evaluation",
        actor_raw_delta_v_mps: Any = None,
    ) -> GoalRecoveryCorrectionResult:
        if not self.config.enabled:
            return _zero_result("disabled", config=self.config)
        if action_provenance != "evaluation":
            return _zero_result("non_evaluation_action", config=self.config)
        if not bool(release_latched):
            return _zero_result("bypass_not_released", config=self.config)
        if not bool(post_bypass_recovery):
            return _zero_result("not_post_bypass_recovery", config=self.config)
        alpha = _smooth_alpha(pass_progress, self.config.alpha_ramp_enabled)
        if alpha <= self.config.epsilon:
            return _zero_result("alpha_zero", eligible=True, config=self.config)
        motion = decompose_goal_relative_motion(
            estimated_position_m,
            estimated_velocity_mps,
            goal_position_m,
            epsilon=self.config.epsilon,
        )
        if not motion.valid:
            return _zero_result(motion.reason, config=self.config)
        desired = compute_desired_goal_speed(
            motion.goal_distance_m,
            max_acceleration_mps2=self.config.max_acceleration_mps2,
            goal_threshold_m=self.config.goal_threshold_m,
            minimum_speed_mps=self.config.minimum_goal_speed_mps,
            cruise_speed_mps=self.config.cruise_speed_mps,
        )
        base_desired = desired
        v2_governor_eligible = False
        v2_governor_active = False
        v2_distance_to_threshold = 0.0
        v2_usable_braking_distance = 0.0
        v2_envelope_speed = 0.0
        v2_governed_speed = 0.0
        v2_radial_overspeed = 0.0
        v2_braking_required = False
        v2_inside_goal_hold = False
        v2_anti_stall_active = False
        v22_gate = bool(
            self.config.terminal_capture_v2_enabled
            and self.config.terminal_radial_conflict_arbitration_enabled
            and self.config.terminal_capture_v22_enabled
        )
        v22_decision: TerminalV22Decision | None = None
        v22_latched_before = False
        v22_restorative_active = False
        v22_distance_outside_stable = 0.0
        v22_restore_fraction = 0.0
        v22_restorative_desired = 0.0
        v22_restorative_error = 0.0
        v22_restorative_delta = np.zeros(3, dtype=float)
        v22_inside_canonical = False
        v23_decision: TerminalV23Decision | None = None
        v23_radial_error = 0.0
        v23_positive_radial_error = False
        v23_preboundary_authority_repair_active = False
        v23_requested_propulsion_radial = 0.0
        v23_executed_propulsion_radial = 0.0
        v23_radial_propulsion_allowed = False
        v23_radial_propulsion_suppressed = False
        v23_radial_propulsion_suppression_reason = "not_v23_guard"
        v23_authority_active = False
        v23_gate = bool(
            v22_gate and self.config.terminal_capture_v23_enabled
        )
        terminal_active = bool(
            self.config.terminal_capture_enabled
            and motion.goal_distance_m <= self.config.terminal_capture_radius_m
        )
        terminal_blend = 0.0
        propulsion_scale = 1.0
        brake_gain = self.config.radial_gain
        tangential_gain = self.config.tangential_gain
        if terminal_active:
            base_at_radius = compute_desired_goal_speed(
                self.config.terminal_capture_radius_m,
                max_acceleration_mps2=self.config.max_acceleration_mps2,
                goal_threshold_m=self.config.goal_threshold_m,
                minimum_speed_mps=self.config.minimum_goal_speed_mps,
                cruise_speed_mps=self.config.cruise_speed_mps,
            )
            desired, terminal_blend = compute_terminal_desired_radial_speed(
                motion.goal_distance_m,
                base_desired_speed_mps=base_desired,
                terminal_capture_radius_m=self.config.terminal_capture_radius_m,
                goal_threshold_m=self.config.goal_threshold_m,
                terminal_radial_speed_floor_mps=self.config.terminal_radial_speed_floor_mps,
                base_speed_at_terminal_radius_mps=base_at_radius,
                smooth=self.config.terminal_smooth_blend,
            )
            if self.config.terminal_propulsion_fade_enabled:
                propulsion_scale = 1.0 - terminal_blend
            brake_gain = self.config.radial_gain + terminal_blend * (
                self.config.terminal_radial_brake_gain - self.config.radial_gain
            )
            tangential_gain = self.config.tangential_gain + terminal_blend * (
                self.config.terminal_tangential_gain - self.config.tangential_gain
            )
        if self.config.terminal_capture_v2_enabled:
            # The v2 governor is eligible after all existing correction gates
            # and state validation, independent of the legacy 8 m radius.
            v2_governor_eligible = True
            v2_distance_to_threshold = max(
                motion.goal_distance_m - self.config.goal_threshold_m,
                0.0,
            )
            v2_usable_braking_distance = max(
                motion.goal_distance_m
                - self.config.goal_threshold_m
                - self.config.terminal_stopping_margin_m,
                0.0,
            )
            v2_inside_goal_hold = bool(
                self.config.terminal_hold_inside_goal_enabled
                and motion.goal_distance_m <= self.config.goal_threshold_m
            )
            if v2_inside_goal_hold:
                v2_envelope_speed = 0.0
                v2_governed_speed = 0.0
            else:
                v2_envelope_speed = float(
                    np.sqrt(
                        max(
                            self.config.terminal_radial_speed_floor_mps**2
                            + 2.0
                            * self.config.terminal_effective_deceleration_mps2
                            * v2_usable_braking_distance,
                            0.0,
                        )
                    )
                )
                # Use the pure helper for the base-capped desired speed.  The
                # legacy v1 terminal speed cap is intentionally not stacked.
                v2_governed_speed = compute_terminal_v2_envelope(
                    motion.goal_distance_m,
                    base_desired_speed_mps=base_desired,
                    effective_deceleration_mps2=self.config.terminal_effective_deceleration_mps2,
                    stopping_margin_m=self.config.terminal_stopping_margin_m,
                    goal_threshold_m=self.config.goal_threshold_m,
                    radial_speed_floor_mps=self.config.terminal_radial_speed_floor_mps,
                    hold_inside_goal_enabled=self.config.terminal_hold_inside_goal_enabled,
                )
            desired = float(v2_governed_speed)
            v2_governor_active = bool(desired < base_desired - self.config.epsilon)
            v2_radial_overspeed = max(
                motion.current_radial_velocity_mps - desired,
                0.0,
            )
            v2_braking_required = bool(v2_radial_overspeed > self.config.epsilon)
        if v22_gate:
            # The helper is called only after finite motion and all existing
            # bypass gates have passed.  Updating the latch here makes invalid
            # motion and all early returns state-preserving by construction.
            v22_latched_before = bool(self.terminal_hold_latched)
            v22_decision = compute_terminal_v22_decision(
                motion.goal_distance_m,
                latched_before=v22_latched_before,
                enter_radius_m=self.config.terminal_hold_enter_radius_m,
                goal_threshold_m=self.config.goal_threshold_m,
                release_radius_m=self.config.terminal_hold_release_radius_m,
                restorative_enabled=self.config.terminal_hold_restorative_enabled,
                radial_speed_floor_mps=self.config.terminal_radial_speed_floor_mps,
            )
            self.terminal_hold_latched = v22_decision.latched_after
            v22_restorative_active = bool(
                v22_decision.mode == "restorative"
                and self.config.terminal_hold_restorative_enabled
            )
            v22_distance_outside_stable = (
                v22_decision.distance_outside_stable_m
            )
            v22_restore_fraction = v22_decision.restore_fraction
            v22_restorative_desired = (
                v22_decision.restorative_desired_radial_speed_mps
            )
            v22_inside_canonical = v22_decision.inside_canonical_goal
            if v22_decision.mode in ("capture", "inside_canonical"):
                desired = 0.0
            elif v22_decision.mode == "restorative":
                desired = float(v22_restorative_desired)
            if v22_restorative_active:
                # Keep v2.2 telemetry tied to its own restorative profile even
                # when v2.3 later supplies the unified desired speed.
                v22_restorative_error = float(
                    v22_restorative_desired
                    - motion.current_radial_velocity_mps
                )
                if v22_restorative_error > 0.0:
                    v22_restorative_delta = (
                        self.config.radial_gain
                        * v22_restorative_error
                        * motion.goal_direction
                    )
            # released_capture and inactive deliberately preserve the normal
            # v2 desired speed computed immediately above.
        if v23_gate and self.terminal_hold_latched:
            # v2.2 owns the inclusive release boundary.  If it released on
            # this call, the latch is false and v2.3 intentionally does not
            # override the v2.2 same-call release/capture result.
            v23_decision = compute_terminal_v23_decision(
                motion.goal_distance_m,
                target_radius_m=self.config.terminal_hold_target_radius_m,
                goal_threshold_m=self.config.goal_threshold_m,
                release_radius_m=self.config.terminal_hold_release_radius_m,
                hold_latched=True,
                restorative_speed_cap_mps=self.config.terminal_radial_speed_floor_mps,
            )
            desired = float(v23_decision.desired_radial_speed_mps)
        radial_error = desired - motion.current_radial_velocity_mps
        if v23_decision is not None:
            v23_radial_error = float(radial_error)
            v23_positive_radial_error = bool(
                v23_decision.guard_active
                and radial_error > self.config.epsilon
            )
            v23_preboundary_authority_repair_active = bool(
                v23_gate
                and self.terminal_hold_latched
                and v23_decision.guard_active
                and v23_decision.preboundary_active
                and radial_error > self.config.epsilon
            )
            v23_requested_propulsion_radial = float(
                self.config.radial_gain * max(radial_error, 0.0)
                if v23_decision.guard_active
                else 0.0
            )
        propulsion = (
            self.config.radial_gain
            * max(radial_error, 0.0)
            * propulsion_scale
            * motion.goal_direction
        )
        v2_anti_stall_active = bool(
            v2_governor_eligible
            and motion.goal_distance_m > self.config.goal_threshold_m
            and radial_error > 0.0
        )
        if v2_anti_stall_active or v23_preboundary_authority_repair_active:
            # v2 owns the near-goal speed envelope; do not apply v1's
            # propulsion fade on top of it.  The v2.3 repair reuses the same
            # existing authority only while its latched pre-boundary guard has
            # a positive tracking error; all other paths retain the legacy
            # fade arithmetic.
            propulsion_scale = 1.0
            propulsion = (
                self.config.radial_gain
                * max(radial_error, 0.0)
                * motion.goal_direction
            )
        braking = brake_gain * min(radial_error, 0.0) * motion.goal_direction
        v23_radial_propulsion_allowed = bool(v23_positive_radial_error)
        if terminal_active:
            radial = propulsion + braking
            tangential = -tangential_gain * motion.current_tangential_velocity_vector_mps
        else:
            # Preserve the historical arithmetic path bit-for-bit whenever the
            # optional terminal mode is disabled or outside its radius.
            radial = self.config.radial_gain * radial_error * motion.goal_direction
            tangential = -self.config.tangential_gain * motion.current_tangential_velocity_vector_mps
        raw = radial + tangential
        raw_before_conflict = raw.copy()
        raw_after_conflict = raw_before_conflict.copy()
        v21_enabled = bool(
            self.config.terminal_radial_conflict_arbitration_enabled
        )
        v21_eligible = False
        v21_active = False
        v21_inactive_reason = "disabled" if not v21_enabled else "v2_disabled"
        v21_actor_radial = 0.0
        v21_actor_opposing_radial = 0.0
        v21_compensation = np.zeros(3, dtype=float)
        v21_compensation_norm = 0.0
        v21_compensation_radial = 0.0
        v21_goal_direction_source = (
            "estimated_goal_direction_used_by_radial_propulsion"
        )
        v23_conflict_authority = bool(
            v23_decision is not None
            and v23_decision.guard_active
            and radial_error > self.config.epsilon
        )
        if v21_enabled:
            # v2.1 is deliberately downstream of every existing controller
            # gate.  In particular, do not inspect the Actor action until v2
            # has admitted an anti-stall approach state.
            if not v2_governor_eligible:
                v21_inactive_reason = "v2_disabled"
            elif (
                motion.goal_distance_m <= self.config.goal_threshold_m
                and not v23_conflict_authority
            ):
                v21_inactive_reason = "distance_not_above_threshold"
            elif radial_error <= 0.0:
                v21_inactive_reason = "radial_error_non_positive"
            elif not v2_anti_stall_active and not v23_conflict_authority:
                v21_inactive_reason = "anti_stall_inactive"
            elif actor_raw_delta_v_mps is None:
                v21_inactive_reason = "actor_missing"
            else:
                try:
                    actor = np.asarray(actor_raw_delta_v_mps, dtype=float).reshape(3)
                except (TypeError, ValueError, OverflowError):
                    v21_inactive_reason = "actor_invalid"
                else:
                    if not np.isfinite(actor).all():
                        v21_inactive_reason = "actor_non_finite"
                    else:
                        with np.errstate(over="ignore", invalid="ignore"):
                            v21_actor_radial = float(
                                np.dot(actor, motion.goal_direction)
                            )
                        if not np.isfinite(v21_actor_radial):
                            v21_actor_radial = 0.0
                            v21_inactive_reason = "actor_non_finite"
                        else:
                            # Eligibility records a valid candidate after all
                            # base gates; activation is the opposing-radial
                            # sign test.
                            v21_eligible = True
                            v21_actor_opposing_radial = max(
                                0.0, -v21_actor_radial
                            )
                            if v21_actor_radial < 0.0:
                                v21_active = True
                                v21_inactive_reason = ""
                                v21_compensation = (
                                    self.config.terminal_radial_conflict_compensation_gain
                                    * v21_actor_opposing_radial
                                    * motion.goal_direction
                                )
                                if not np.isfinite(v21_compensation).all():
                                    # The configuration is validated, but a
                                    # finite Actor value can still overflow a
                                    # dot/product.  Keep the base correction in
                                    # that pathological case.
                                    v21_active = False
                                    v21_eligible = False
                                    v21_compensation = np.zeros(3, dtype=float)
                                    v21_actor_opposing_radial = 0.0
                                    v21_inactive_reason = "actor_non_finite"
                                else:
                                    raw_after_conflict = (
                                        raw_before_conflict + v21_compensation
                                    )
                                    v21_compensation_norm = float(
                                        np.linalg.norm(v21_compensation)
                                    )
                                    v21_compensation_radial = float(
                                        np.dot(
                                            v21_compensation,
                                            motion.goal_direction,
                                        )
                                    )
                            else:
                                v21_inactive_reason = "actor_not_opposing"
        norm = float(np.linalg.norm(raw_after_conflict))
        limited = bool(norm > self.config.max_delta_v_mps)
        correction = raw_after_conflict.copy()
        if limited and norm > self.config.epsilon:
            correction *= self.config.max_delta_v_mps / norm
        clip_ratio = float(np.linalg.norm(correction) / norm) if norm > self.config.epsilon else 1.0
        if not np.isfinite(correction).all():
            return _zero_result("non_finite_correction", config=self.config)
        # Attribute the shared correction cap to propulsion before reporting
        # "executed" authority.  Actor fusion and its transform remain
        # downstream and intentionally cannot be decomposed back into a
        # unique propulsion contribution.
        v23_executed_propulsion_radial = float(
            np.dot(propulsion * clip_ratio, motion.goal_direction)
        )
        v23_authority_active = bool(
            v23_radial_propulsion_allowed
            and v23_executed_propulsion_radial > self.config.epsilon
        )
        v23_radial_propulsion_suppressed = bool(
            v23_radial_propulsion_allowed and not v23_authority_active
        )
        if v23_authority_active:
            v23_radial_propulsion_suppression_reason = "none"
        elif v23_radial_propulsion_allowed:
            v23_radial_propulsion_suppression_reason = (
                "numeric_tolerance"
                if v23_requested_propulsion_radial <= self.config.epsilon
                else "other_explicit_reason"
            )
        elif v23_decision is not None and v23_decision.guard_active:
            v23_radial_propulsion_suppression_reason = "radial_error_non_positive"
        v21_compensation_post_cap_radial = float(
            np.dot(
                v21_compensation * clip_ratio,
                motion.goal_direction,
            )
        )
        reported_radial = radial + v21_compensation if v21_active else radial
        return GoalRecoveryCorrectionResult(
            correction_delta_v_mps=correction,
            radial_correction_delta_v_mps=reported_radial,
            tangential_correction_delta_v_mps=tangential,
            correction_alpha=alpha,
            goal_distance_m=motion.goal_distance_m,
            goal_direction=motion.goal_direction,
            current_radial_velocity_mps=motion.current_radial_velocity_mps,
            current_tangential_velocity_vector_mps=motion.current_tangential_velocity_vector_mps,
            current_tangential_velocity_mps=motion.current_tangential_velocity_mps,
            desired_radial_velocity_mps=desired,
            active=True,
            eligible=True,
            activation_reason="post_bypass_recovery",
            correction_norm_limited=limited,
            terminal_capture_enabled=bool(self.config.terminal_capture_enabled),
            terminal_capture_active=terminal_active,
            terminal_capture_radius_m=float(self.config.terminal_capture_radius_m),
            terminal_blend=float(terminal_blend),
            base_desired_radial_velocity_mps=float(base_desired),
            terminal_desired_radial_velocity_mps=float(desired),
            radial_velocity_error_mps=float(radial_error),
            radial_propulsion_delta_v_mps=propulsion,
            radial_propulsion_scale=float(propulsion_scale),
            radial_braking_delta_v_mps=braking,
            effective_terminal_radial_brake_gain=float(brake_gain),
            base_tangential_gain=float(self.config.tangential_gain),
            effective_terminal_tangential_gain=float(tangential_gain),
            terminal_tangential_damping_delta_v_mps=tangential,
            terminal_raw_correction_delta_v_mps=raw_after_conflict,
            terminal_raw_correction_norm_mps=norm,
            terminal_correction_clipped=limited,
            terminal_correction_clip_ratio=clip_ratio,
            terminal_v2_enabled=bool(self.config.terminal_capture_v2_enabled),
            terminal_v2_governor_eligible=v2_governor_eligible,
            terminal_v2_governor_active=v2_governor_active,
            terminal_v2_a_eff_mps2=float(self.config.terminal_effective_deceleration_mps2),
            terminal_v2_stopping_margin_m=float(self.config.terminal_stopping_margin_m),
            terminal_v2_distance_to_threshold_m=float(v2_distance_to_threshold),
            terminal_v2_usable_braking_distance_m=float(v2_usable_braking_distance),
            terminal_v2_base_desired_radial_speed_mps=(
                float(base_desired) if v2_governor_eligible else 0.0
            ),
            terminal_v2_envelope_speed_mps=(
                float(v2_envelope_speed) if v2_governor_eligible else 0.0
            ),
            terminal_v2_governed_desired_radial_speed_mps=(
                float(v2_governed_speed) if v2_governor_eligible else 0.0
            ),
            terminal_v2_radial_overspeed_mps=(
                float(v2_radial_overspeed) if v2_governor_eligible else 0.0
            ),
            terminal_v2_braking_required=v2_braking_required,
            terminal_v2_inside_goal_hold=v2_inside_goal_hold,
            terminal_v2_anti_stall_active=v2_anti_stall_active,
            terminal_v21_radial_conflict_enabled=v21_enabled,
            terminal_v21_radial_conflict_eligible=v21_eligible,
            terminal_v21_radial_conflict_active=v21_active,
            terminal_v21_inactive_reason=v21_inactive_reason,
            terminal_v21_goal_direction_source=v21_goal_direction_source,
            terminal_v21_actor_radial_delta_v_mps=float(v21_actor_radial),
            terminal_v21_actor_opposing_radial_mps=float(
                v21_actor_opposing_radial
            ),
            terminal_v21_compensation_gain=float(
                self.config.terminal_radial_conflict_compensation_gain
            ),
            terminal_v21_conflict_compensation_delta_v_mps=v21_compensation,
            terminal_v21_conflict_compensation_norm_mps=float(
                v21_compensation_norm
            ),
            terminal_v21_conflict_compensation_radial_mps=float(
                v21_compensation_radial
            ),
            terminal_v21_raw_correction_before_conflict_mps=raw_before_conflict,
            terminal_v21_raw_correction_after_conflict_mps=raw_after_conflict,
            terminal_v21_correction_clip_ratio=clip_ratio,
            terminal_v21_compensation_post_cap_radial_mps=(
                v21_compensation_post_cap_radial
            ),
            terminal_v22_enabled=bool(self.config.terminal_capture_v22_enabled),
            terminal_v22_hold_latched=bool(self.terminal_hold_latched)
            if v22_gate
            else False,
            terminal_v22_hold_latched_before=(
                bool(v22_latched_before) if v22_gate else False
            ),
            terminal_v22_hold_enter_radius_m=(
                float(self.config.terminal_hold_enter_radius_m)
            ),
            terminal_v22_hold_release_radius_m=(
                float(self.config.terminal_hold_release_radius_m)
            ),
            terminal_v22_hold_enter_event=(
                bool(v22_decision.enter_event)
                if v22_decision is not None
                else False
            ),
            terminal_v22_hold_release_event=(
                bool(v22_decision.release_event)
                if v22_decision is not None
                else False
            ),
            terminal_v22_hold_mode=(
                v22_decision.mode if v22_decision is not None else "inactive"
            ),
            terminal_v22_restorative_active=v22_restorative_active,
            terminal_v22_distance_outside_stable_m=float(
                v22_distance_outside_stable
            ),
            terminal_v22_restore_fraction=float(v22_restore_fraction),
            terminal_v22_restorative_desired_radial_speed_mps=float(
                v22_restorative_desired
            ),
            terminal_v22_restorative_radial_error_mps=float(v22_restorative_error),
            terminal_v22_restorative_delta_v_mps=v22_restorative_delta,
            terminal_v22_inside_canonical_goal=v22_inside_canonical,
            terminal_v23_enabled=bool(self.config.terminal_capture_v23_enabled),
            terminal_v23_hold_target_radius_m=float(
                self.config.terminal_hold_target_radius_m
            ),
            terminal_v23_guard_active=bool(
                v23_decision.guard_active if v23_decision is not None else False
            ),
            terminal_v23_guard_fraction=float(
                v23_decision.guard_fraction if v23_decision is not None else 0.0
            ),
            terminal_v23_desired_radial_speed_mps=float(
                v23_decision.desired_radial_speed_mps
                if v23_decision is not None else 0.0
            ),
            terminal_v23_preboundary_active=bool(
                v23_decision.preboundary_active
                if v23_decision is not None else False
            ),
            terminal_v23_postboundary_active=bool(
                v23_decision.postboundary_active
                if v23_decision is not None else False
            ),
            terminal_v23_distance_to_target_band_m=float(
                v23_decision.distance_to_target_band_m
                if v23_decision is not None else 0.0
            ),
            terminal_v23_radial_error_mps=float(v23_radial_error),
            terminal_v23_positive_radial_error=bool(v23_positive_radial_error),
            terminal_v23_radial_propulsion_allowed=bool(
                v23_radial_propulsion_allowed
            ),
            terminal_v23_radial_propulsion_suppressed=bool(
                v23_radial_propulsion_suppressed
            ),
            terminal_v23_radial_propulsion_suppression_reason=(
                v23_radial_propulsion_suppression_reason
            ),
            terminal_v23_requested_propulsion_radial_mps=float(
                v23_requested_propulsion_radial
            ),
            terminal_v23_executed_propulsion_radial_mps=float(
                v23_executed_propulsion_radial
            ),
            terminal_v23_authority_active=bool(v23_authority_active),
            terminal_v23_preboundary_authority_repair_active=bool(
                v23_preboundary_authority_repair_active
                and v23_authority_active
            ),
            terminal_v23_net_tracking_radial_before_v21_mps=float(
                np.dot(raw_before_conflict, motion.goal_direction)
            ),
            terminal_v23_net_tracking_radial_after_v21_mps=float(
                np.dot(raw_after_conflict, motion.goal_direction)
            ),
        )


def fuse_actor_action(
    actor_raw_delta_v_mps: Any,
    correction: GoalRecoveryCorrectionResult,
    action_transform: Callable[[np.ndarray], Mapping[str, Any]],
) -> GoalRecoveryActionFusion:
    """Fuse an active correction, structurally bypassing transform otherwise."""

    actor = np.asarray(actor_raw_delta_v_mps, dtype=float).reshape(3)
    if not correction.active or correction.correction_alpha <= 0.0:
        return GoalRecoveryActionFusion(actor.copy(), actor.copy(), actor.copy(), False)
    corrected = actor + correction.correction_alpha * correction.correction_delta_v_mps
    transformed = action_transform(corrected)
    output = np.asarray(transformed["constrained_delta_v_mps"], dtype=float).reshape(3)
    if not np.isfinite(output).all():
        raise ValueError("action transform returned a non-finite delta-v")
    return GoalRecoveryActionFusion(actor.copy(), corrected, output, True)
