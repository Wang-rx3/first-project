"""Action-driven three-dimensional UAV kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Iterable

import numpy as np


BRAKING_DIAGNOSTIC_TOLERANCE_MPS = 1e-4
BRAKING_DISTANCE_SPEED_PROFILE_LEGACY = "legacy"
BRAKING_DISTANCE_SPEED_PROFILE_EARLY_SMOOTH = "early_smooth"
BRAKING_DISTANCE_SPEED_PROFILES = (
    BRAKING_DISTANCE_SPEED_PROFILE_LEGACY,
    BRAKING_DISTANCE_SPEED_PROFILE_EARLY_SMOOTH,
)

# The normalized anchor fractions are deliberately fixed and auditable.  With
# the current 1 m goal radius and 8 m near-goal boundary they correspond to
# 1, 3, 5, 6, and 8 m.  The profile keeps a non-zero low-speed target around
# 3 m, starts reducing the admissible speed before 5 m, and reaches the cruise
# target continuously at the near-goal boundary.
_EARLY_SMOOTH_DISTANCE_FRACTIONS = (0.0, 2.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 1.0)
_EARLY_SMOOTH_SPEED_FRACTIONS = (0.0, 0.16, 0.35, 0.53, 1.0)


def stopping_distance_m(
    speed_mps: float,
    goal_speed_mps: float,
    braking_acceleration_mps2: float,
) -> float:
    """Return the constant-deceleration distance needed to reach a target speed.

    The calculation is used by the single-UAV controller as a conservative
    planning signal.  It is deliberately independent of simulator truth and
    therefore can be evaluated from the EKF-estimated velocity.
    """

    speed = max(0.0, float(speed_mps))
    target = max(0.0, float(goal_speed_mps))
    acceleration = float(braking_acceleration_mps2)
    if not np.isfinite(speed) or not np.isfinite(target):
        return 0.0
    if not np.isfinite(acceleration) or acceleration <= 0.0:
        raise ValueError("braking_acceleration_mps2 must be finite and positive")
    return float(max(0.0, speed * speed - target * target) / (2.0 * acceleration))


def braking_distance_target_speed(
    distance_m: float,
    *,
    goal_threshold_m: float,
    near_goal_radius_m: float = 8.0,
    stable_speed_mps: float,
    braking_acceleration_mps2: float,
    braking_margin_m: float,
    far_target_speed_mps: float,
    max_speed_mps: float,
    speed_profile: str = BRAKING_DISTANCE_SPEED_PROFILE_LEGACY,
) -> float:
    """Return a speed target that leaves room for braking before the goal.

    ``braking_margin_m`` is a safety buffer outside the goal radius.  The
    target is capped by the configured cruise and physical speed limits, so
    the policy is encouraged to fly quickly when there is room and to start
    reducing speed before the stopping distance would cross the goal region.

    ``legacy`` is the original stopping-distance envelope.  ``early_smooth``
    is a single-factor alternative used by the terminal-approach experiment:
    it interpolates a fixed, normalized set of distance/speed anchors with a
    smoothstep easing function.  It changes only the target-speed curve; the
    braking/goal reward coefficients and physical dynamics remain untouched.
    """

    distance = max(0.0, float(distance_m))
    goal_radius = max(0.0, float(goal_threshold_m))
    near_radius = max(goal_radius, float(near_goal_radius_m))
    stable_speed = max(0.0, float(stable_speed_mps))
    margin = max(0.0, float(braking_margin_m))
    far_speed = max(stable_speed, float(far_target_speed_mps))
    speed_limit = max(stable_speed, float(max_speed_mps))
    acceleration = float(braking_acceleration_mps2)
    profile = str(speed_profile)
    if profile not in BRAKING_DISTANCE_SPEED_PROFILES:
        raise ValueError(
            "speed_profile must be one of "
            + ", ".join(BRAKING_DISTANCE_SPEED_PROFILES)
        )
    if not np.isfinite(distance) or not np.isfinite(goal_radius) or not np.isfinite(near_radius):
        return stable_speed
    if not np.isfinite(acceleration) or acceleration <= 0.0:
        raise ValueError("braking_acceleration_mps2 must be finite and positive")
    if not np.isfinite(margin) or not np.isfinite(far_speed) or not np.isfinite(speed_limit):
        raise ValueError("braking-distance speed parameters must be finite")
    if profile == BRAKING_DISTANCE_SPEED_PROFILE_EARLY_SMOOTH:
        if distance <= goal_radius:
            return float(np.clip(stable_speed, stable_speed, speed_limit))
        if near_radius <= goal_radius:
            return float(np.clip(far_speed, stable_speed, speed_limit))
        span = near_radius - goal_radius
        anchor_distances = np.asarray(
            [goal_radius + fraction * span for fraction in _EARLY_SMOOTH_DISTANCE_FRACTIONS],
            dtype=float,
        )
        anchor_speeds = np.asarray(
            [
                stable_speed + fraction * (far_speed - stable_speed)
                for fraction in _EARLY_SMOOTH_SPEED_FRACTIONS
            ],
            dtype=float,
        )
        if distance >= near_radius:
            return float(np.clip(far_speed, stable_speed, speed_limit))
        interval = int(np.searchsorted(anchor_distances, distance, side="right") - 1)
        interval = max(0, min(interval, len(anchor_distances) - 2))
        left_distance = float(anchor_distances[interval])
        right_distance = float(anchor_distances[interval + 1])
        left_speed = float(anchor_speeds[interval])
        right_speed = float(anchor_speeds[interval + 1])
        if right_distance <= left_distance:
            return float(np.clip(left_speed, stable_speed, speed_limit))
        progress = float(
            np.clip(
                (distance - left_distance) / (right_distance - left_distance),
                0.0,
                1.0,
            )
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        return float(
            np.clip(left_speed + eased * (right_speed - left_speed), stable_speed, speed_limit)
        )

    usable_distance = max(0.0, distance - goal_radius - margin)
    stopping_limited_speed = float(
        np.sqrt(max(0.0, stable_speed * stable_speed + 2.0 * acceleration * usable_distance))
    )
    return float(np.clip(min(far_speed, stopping_limited_speed), stable_speed, speed_limit))


@dataclass(frozen=True)
class MotionConfig:
    """Configuration for the bounded translational motion model."""

    dt_s: float = 0.2
    max_speed_mps: float = 12.0
    max_acceleration_mps2: float = 3.0
    bounds_min_m: tuple[float, float, float] = (0.0, 0.0, 2.0)
    bounds_max_m: tuple[float, float, float] = (200.0, 200.0, 60.0)
    safety_distance_m: float = 2.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0 or self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            raise ValueError("motion time and limits must be positive")
        if any(lo >= hi for lo, hi in zip(self.bounds_min_m, self.bounds_max_m)):
            raise ValueError("motion bounds must be strictly increasing")

    def action_contract(self) -> dict[str, object]:
        """Return the explicit motion-action and braking contract."""

        max_delta_v = float(self.max_acceleration_mps2 * self.dt_s)
        return {
            "type": "continuous_3d_velocity_increment",
            "vector_dimension": 3,
            "units": "m/s per decision window",
            "max_norm_mps": max_delta_v,
            "max_speed_mps": float(self.max_speed_mps),
            "max_acceleration_mps2": float(self.max_acceleration_mps2),
            "decision_interval_s": float(self.dt_s),
            "negative_delta_v_allowed": True,
            "braking_supported": True,
            "braking_signal": (
                "negative projection of applied delta_v onto pre-action velocity"
            ),
            "braking_tolerance_mps": BRAKING_DIAGNOSTIC_TOLERANCE_MPS,
            "speed_update": "clip(pre_velocity + applied_delta_v, max_speed_mps)",
        }


@dataclass
class UAVKinematicState:
    """Minimal state required by the Python environment."""

    position_m: np.ndarray
    velocity_mps: np.ndarray
    attitude_rad: np.ndarray
    acceleration_mps2: np.ndarray
    angular_velocity_rps: np.ndarray

    def copy(self) -> "UAVKinematicState":
        return UAVKinematicState(
            position_m=self.position_m.copy(),
            velocity_mps=self.velocity_mps.copy(),
            attitude_rad=self.attitude_rad.copy(),
            acceleration_mps2=self.acceleration_mps2.copy(),
            angular_velocity_rps=self.angular_velocity_rps.copy(),
        )


def clip_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """Project a vector into an l2 ball without changing its direction."""

    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= limit or norm == 0.0:
        return value.copy()
    return value * (limit / norm)


def velocity_increment_diagnostics(
    previous_velocity_mps: Iterable[float],
    next_velocity_mps: Iterable[float],
    delta_v_commanded_mps: Iterable[float],
    delta_v_applied_mps: Iterable[float],
    *,
    tolerance_mps: float = BRAKING_DIAGNOSTIC_TOLERANCE_MPS,
    speed_limit_mps: float | None = None,
) -> dict[str, float | int]:
    """Quantify whether a velocity-increment action actually brakes.

    ``delta_v`` is a continuous three-dimensional velocity increment.  Its
    component along the *pre-action* velocity is the physically meaningful
    braking signal: a negative projection opposes the current motion.  The
    returned fields separate the policy/motion command from the executed
    increment after speed and safety projection, and compare the pre/post
    speeds so a diagnostic can distinguish a braking command from effective
    speed reduction.
    """

    previous = np.asarray(tuple(previous_velocity_mps), dtype=float).reshape(3)
    next_velocity = np.asarray(tuple(next_velocity_mps), dtype=float).reshape(3)
    commanded = np.asarray(tuple(delta_v_commanded_mps), dtype=float).reshape(3)
    applied = np.asarray(tuple(delta_v_applied_mps), dtype=float).reshape(3)
    previous_speed = float(np.linalg.norm(previous))
    next_speed = float(np.linalg.norm(next_velocity))
    tolerance = max(0.0, float(tolerance_mps))

    if previous_speed > tolerance:
        previous_direction = previous / previous_speed
        commanded_longitudinal = float(np.dot(previous_direction, commanded))
        applied_longitudinal = float(np.dot(previous_direction, applied))
    else:
        # There is no well-defined direction from rest, so a first action is
        # not labelled as braking even if one of its components is negative.
        commanded_longitudinal = 0.0
        applied_longitudinal = 0.0

    speed_change = next_speed - previous_speed
    speed_reduction = max(0.0, -speed_change)
    commanded_braking = max(0.0, -commanded_longitudinal)
    applied_braking = max(0.0, -applied_longitudinal)
    braking_action = int(applied_braking > tolerance)
    brake_effective = int(
        braking_action and speed_reduction > tolerance
    )
    brake_execution_loss = max(0.0, commanded_braking - applied_braking)
    speed_limit_limited = int(
        speed_limit_mps is not None
        and np.linalg.norm(previous + commanded) > float(speed_limit_mps) + tolerance
        and next_speed >= float(speed_limit_mps) - tolerance
    )
    brake_blocked = int(
        commanded_braking > tolerance
        and (
            applied_braking + tolerance < commanded_braking
            or bool(speed_limit_limited)
        )
    )
    return {
        "pre_speed_mps": previous_speed,
        "post_speed_mps": next_speed,
        "speed_change_mps": speed_change,
        "speed_reduction_mps": speed_reduction,
        "delta_v_commanded_norm_mps": float(np.linalg.norm(commanded)),
        "delta_v_applied_norm_mps": float(np.linalg.norm(applied)),
        "commanded_longitudinal_delta_v_mps": commanded_longitudinal,
        "applied_longitudinal_delta_v_mps": applied_longitudinal,
        "commanded_braking_delta_v_mps": commanded_braking,
        "braking_delta_v_mps": applied_braking,
        "braking_action": braking_action,
        "brake_effective": brake_effective,
        "brake_execution_loss_mps": brake_execution_loss,
        "brake_blocked": brake_blocked,
        "speed_limit_limited": speed_limit_limited,
    }


def apply_velocity_increment(
    state: UAVKinematicState,
    delta_v_mps: Iterable[float],
    config: MotionConfig,
    *,
    safety_projector=None,
) -> tuple[UAVKinematicState, dict[str, object]]:
    """Apply one high-level velocity-increment action.

    The increment is bounded by ``a_max * dt``.  The returned position uses
    constant acceleration over the interval.  A caller may provide a safety
    projector that clips the proposed position against the current scene.
    """

    original_delta = np.asarray(tuple(delta_v_mps), dtype=float).reshape(3)
    delta = clip_vector_norm(
        original_delta,
        config.max_acceleration_mps2 * config.dt_s,
    )
    next_velocity = clip_vector_norm(state.velocity_mps + delta, config.max_speed_mps)
    # Use the velocity after speed clipping in the kinematic integration.
    # Integrating the unclipped command here would make the logged position,
    # velocity and IMU mutually inconsistent whenever the speed limit was
    # active, which can look like an EKF failure even with correct updates.
    executed_delta = next_velocity - state.velocity_mps
    next_position = (
        state.position_m
        + state.velocity_mps * config.dt_s
        + 0.5 * executed_delta * config.dt_s
    )
    clipped_boundary = False
    lower = np.asarray(config.bounds_min_m, dtype=float)
    upper = np.asarray(config.bounds_max_m, dtype=float)
    bounded_position = np.clip(next_position, lower, upper)
    if not np.allclose(next_position, bounded_position):
        clipped_boundary = True
        next_position = bounded_position

    safety_projected = False
    if safety_projector is not None:
        projected = np.asarray(safety_projector(next_position), dtype=float).reshape(3)
        safety_projected = not np.allclose(projected, next_position)
        next_position = projected

    # A hard boundary/obstacle projection changes the displacement selected
    # by the nominal acceleration command. Reconcile the kinematic velocity
    # with the actually executed displacement; otherwise a stopped position
    # would still report the pre-projection velocity and generate a false
    # relative-VO innovation in the EKF. This is an explicit safety reset,
    # and its larger braking increment is recorded for diagnostics/energy.
    if clipped_boundary or safety_projected:
        displacement = next_position - state.position_m
        projected_velocity = clip_vector_norm(displacement / config.dt_s, config.max_speed_mps)
        next_position = state.position_m + projected_velocity * config.dt_s
        next_velocity = projected_velocity
        executed_delta = next_velocity - state.velocity_mps

    yaw = atan2(float(next_velocity[1]), float(next_velocity[0])) if np.linalg.norm(next_velocity[:2]) > 1e-9 else float(state.attitude_rad[2])
    next_attitude = np.array([0.0, 0.0, yaw], dtype=float)
    next_state = UAVKinematicState(
        position_m=next_position,
        velocity_mps=next_velocity,
        attitude_rad=next_attitude,
        acceleration_mps2=executed_delta / config.dt_s,
        angular_velocity_rps=np.array([0.0, 0.0, (yaw - state.attitude_rad[2]) / config.dt_s], dtype=float),
    )
    info = {
        "delta_v_applied": executed_delta.copy(),
        "delta_v_commanded": delta.copy(),
        "action_clipped": bool(
            not np.allclose(delta, original_delta)
            or clipped_boundary
            or safety_projected
        ),
        "boundary_projected": clipped_boundary,
        "safety_projected": safety_projected,
    }
    info.update(
        velocity_increment_diagnostics(
            state.velocity_mps,
            next_state.velocity_mps,
            delta,
            executed_delta,
            speed_limit_mps=config.max_speed_mps,
        )
    )
    return next_state, info


def pairwise_distances(states: dict[str, UAVKinematicState]) -> dict[tuple[str, str], float]:
    """Return directed pairwise distances for a state mapping."""

    result: dict[tuple[str, str], float] = {}
    ids = sorted(states)
    for source in ids:
        for target in ids:
            if source == target:
                continue
            result[(source, target)] = float(
                np.linalg.norm(states[source].position_m - states[target].position_m)
            )
    return result


__all__ = [
    "MotionConfig",
    "UAVKinematicState",
    "BRAKING_DIAGNOSTIC_TOLERANCE_MPS",
    "apply_velocity_increment",
    "clip_vector_norm",
    "pairwise_distances",
    "velocity_increment_diagnostics",
]
