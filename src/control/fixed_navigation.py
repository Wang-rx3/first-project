"""Deterministic fixed navigation driven only by a local state estimate.

The controller in this module is intentionally small and auditable.  It consumes
the first six values of an estimated position/velocity vector and a static
reference position, then produces a bounded three dimensional velocity
increment.  It has no simulator, truth state, reward, graph, or learned-model
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Mapping

import numpy as np

from ..simulation.motion_model import clip_vector_norm


@dataclass(frozen=True)
class FixedNavigationConfig:
    """Frozen gains and limits for the fixed navigation controller."""

    position_gain_s_inv: float = 0.6
    velocity_gain_s_inv: float = 1.6
    cruise_speed_mps: float = 2.5
    waypoint_radius_m: float = 0.8
    max_delta_v_norm_mps: float = 0.6
    dt_s: float = 0.2

    def __post_init__(self) -> None:
        names = (
            "position_gain_s_inv",
            "velocity_gain_s_inv",
            "cruise_speed_mps",
            "waypoint_radius_m",
            "max_delta_v_norm_mps",
            "dt_s",
        )
        for name in names:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and positive") from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            # Keep the public config numerically stable even when callers pass
            # integer-like or NumPy scalar values.
            object.__setattr__(self, name, value)

    @property
    def component_limit_mps(self) -> float:
        """Per-axis limit implied by the configured three-dimensional norm."""

        return float(self.max_delta_v_norm_mps / sqrt(3.0))


def _finite_flat(value: Any, *, name: str, minimum_size: int | None = None, exact_size: int | None = None) -> np.ndarray:
    """Convert an input to a finite one-dimensional float array."""

    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if minimum_size is not None and array.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} values")
    if exact_size is not None and array.size != exact_size:
        raise ValueError(f"{name} must contain exactly {exact_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def fixed_navigation_command(
    estimate: Iterable[float],
    reference: Iterable[float],
    config: FixedNavigationConfig | None = None,
) -> dict[str, object]:
    """Compute one bounded fixed-navigation command.

    Only ``estimate[:3]`` and ``estimate[3:6]`` are consumed.  The requested
    acceleration first becomes a raw velocity increment, then receives an
    axis-aligned component bound followed by the configured three-dimensional
    norm bound.
    """

    controller_config = config if config is not None else FixedNavigationConfig()
    if not isinstance(controller_config, FixedNavigationConfig):
        raise TypeError("config must be a FixedNavigationConfig")

    estimate_array = _finite_flat(estimate, name="estimate", minimum_size=6)
    reference_array = _finite_flat(reference, name="reference", exact_size=3)
    estimated_position = estimate_array[:3].copy()
    estimated_velocity = estimate_array[3:6].copy()
    reference_position = reference_array.copy()

    desired_velocity = clip_vector_norm(
        controller_config.position_gain_s_inv
        * (reference_position - estimated_position),
        controller_config.cruise_speed_mps,
    )
    requested_acceleration = controller_config.velocity_gain_s_inv * (
        desired_velocity - estimated_velocity
    )
    raw_delta_v = requested_acceleration * controller_config.dt_s
    axis_limited_delta_v = np.clip(
        raw_delta_v,
        -controller_config.component_limit_mps,
        controller_config.component_limit_mps,
    )
    command_delta_v = clip_vector_norm(
        axis_limited_delta_v,
        controller_config.max_delta_v_norm_mps,
    )
    controller_limited = bool(
        not np.array_equal(axis_limited_delta_v, raw_delta_v)
        or not np.array_equal(command_delta_v, axis_limited_delta_v)
    )

    return {
        "estimated_position_m": estimated_position,
        "estimated_velocity_mps": estimated_velocity,
        "reference_position_m": reference_position,
        "desired_velocity_mps": desired_velocity,
        "requested_acceleration_mps2": requested_acceleration,
        "raw_delta_v_mps": raw_delta_v,
        "axis_limited_delta_v_mps": axis_limited_delta_v.copy(),
        "command_delta_v_mps": command_delta_v,
        "command_norm": float(np.linalg.norm(command_delta_v)),
        "controller_limited": controller_limited,
    }


def local_observation_command(
    local_obs: Mapping[str, Any],
    reference: Iterable[float] | None = None,
    config: FixedNavigationConfig | None = None,
) -> dict[str, object]:
    """Compute a command from a local observation and optional static target.

    The observation is deliberately narrowed to ``estimate`` and, when no
    explicit reference is supplied, ``goal``.  Every other observation field is
    ignored, including truth, graph, noise, and reward-like fields.
    """

    if not isinstance(local_obs, Mapping):
        raise TypeError("local_obs must be a mapping")
    estimate = local_obs["estimate"]
    target = local_obs["goal"] if reference is None else reference
    return fixed_navigation_command(estimate, target, config)


class FixedWaypointNavigator:
    """Stateful waypoint wrapper around :func:`fixed_navigation_command`."""

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        config: FixedNavigationConfig | None = None,
    ) -> None:
        self.config = config if config is not None else FixedNavigationConfig()
        if not isinstance(self.config, FixedNavigationConfig):
            raise TypeError("config must be a FixedNavigationConfig")
        self._references = self._copy_references(points)
        self._index = 0
        self._waypoint_switch_count = 0

    @staticmethod
    def _copy_references(points: Iterable[Iterable[float]]) -> tuple[tuple[float, float, float], ...]:
        try:
            values = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("points must contain finite 3D vectors") from exc
        if values.ndim == 1:
            if values.size != 3:
                raise ValueError("points must contain finite 3D vectors")
            values = values.reshape(1, 3)
        elif values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must contain finite 3D vectors")
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("points must contain at least one finite 3D vector")
        return tuple(tuple(float(component) for component in row) for row in values)

    @property
    def index(self) -> int:
        """Current waypoint index."""

        return int(self._index)

    @property
    def references(self) -> tuple[np.ndarray, ...]:
        """Return defensive copies of the frozen waypoint references."""

        return tuple(np.asarray(point, dtype=float).copy() for point in self._references)

    @property
    def reference_count(self) -> int:
        return len(self._references)

    @property
    def waypoint_switch_count(self) -> int:
        return int(self._waypoint_switch_count)

    def reset(self) -> None:
        """Start a fresh episode at the first waypoint."""

        self._index = 0
        self._waypoint_switch_count = 0

    def command(self, estimate: Iterable[float]) -> dict[str, object]:
        """Advance using only estimated position and return the current command."""

        estimate_array = _finite_flat(estimate, name="estimate", minimum_size=6)
        while self._index < len(self._references) - 1:
            reference = np.asarray(self._references[self._index], dtype=float)
            distance = float(np.linalg.norm(estimate_array[:3] - reference))
            if distance > self.config.waypoint_radius_m:
                break
            self._index += 1
            self._waypoint_switch_count += 1

        command = fixed_navigation_command(
            estimate_array,
            self._references[self._index],
            self.config,
        )
        command.update(
            {
                "reference_index": int(self._index),
                "reference_count": int(len(self._references)),
                "waypoint_switch_count": int(self._waypoint_switch_count),
            }
        )
        return command


__all__ = [
    "FixedNavigationConfig",
    "FixedWaypointNavigator",
    "fixed_navigation_command",
    "local_observation_command",
]
