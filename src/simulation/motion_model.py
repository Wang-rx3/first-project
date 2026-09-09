"""Action-driven three-dimensional UAV kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Iterable

import numpy as np


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
    "apply_velocity_increment",
    "clip_vector_norm",
    "pairwise_distances",
]
