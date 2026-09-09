"""Synthetic IMU, UWB, visual-odometry, and GNSS observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .motion_model import UAVKinematicState


@dataclass(frozen=True)
class SensorConfig:
    imu_accel_std_mps2: float = 0.035
    imu_gyro_std_rps: float = 0.01
    accel_bias_rw_std: float = 0.0002
    gyro_bias_rw_std: float = 0.00005
    uwb_los_std_m: float = 0.08
    uwb_nlos_std_m: float = 0.65
    uwb_nlos_bias_m: float = 0.8
    # Calibration sweep parameter.  The simulator samples with
    # ``uwb_nlos_std_m`` and the EKF uses this multiplier for the declared
    # measurement covariance, so both sources remain explicit in logs.
    uwb_nlos_variance_multiplier: float = 1.0
    uwb_max_range_m: float = 100.0
    # UWB measurements may be sampled for link diagnostics every decision
    # window, but the local EKF consumes a paired range/neighbor-state update
    # only at this physical sensor update period.
    uwb_update_period_s: float = 0.8
    vo_position_std_m: float = 0.015
    vo_yaw_std_rad: float = 0.005
    gnss_nominal_std_m: float = 0.5
    gnss_degraded_std_m: float = 3.0
    gnss_denied_std_m: float = 20.0

    def __post_init__(self) -> None:
        if self.uwb_nlos_variance_multiplier <= 0.0:
            raise ValueError("uwb_nlos_variance_multiplier must be positive")
        if self.uwb_update_period_s <= 0.0:
            raise ValueError("uwb_update_period_s must be positive")


def _noise(rng: np.random.Generator, std: float, size: int = 3) -> np.ndarray:
    return rng.normal(0.0, std, size=size).astype(float)


def sample_imu(
    state: UAVKinematicState,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    rng: np.random.Generator,
    config: SensorConfig,
) -> dict[str, Any]:
    """Generate world-frame IMU values after the interface conversion."""

    accel_bias += _noise(rng, config.accel_bias_rw_std)
    gyro_bias += _noise(rng, config.gyro_bias_rw_std)
    return {
        "accel": state.acceleration_mps2 + accel_bias + _noise(rng, config.imu_accel_std_mps2),
        "gyro": state.angular_velocity_rps + gyro_bias + _noise(rng, config.imu_gyro_std_rps),
        "bias_accel": accel_bias.copy(),
        "bias_gyro": gyro_bias.copy(),
        "valid": 1,
    }


def sample_uwb(
    source: UAVKinematicState,
    target: UAVKinematicState,
    *,
    los: bool,
    rng: np.random.Generator,
    config: SensorConfig,
) -> dict[str, Any]:
    distance = float(np.linalg.norm(source.position_m - target.position_m))
    available = int(distance <= config.uwb_max_range_m)
    if not available:
        return {
            "true_range": distance,
            "measured_range": None,
            "los": int(los),
            "measurement_std": None,
            "valid": 0,
            "nlos_reason": "out_of_range",
        }
    std = config.uwb_los_std_m if los else config.uwb_nlos_std_m
    bias = 0.0 if los else config.uwb_nlos_bias_m
    valid_probability = 0.97 if los else 0.78
    valid = int(rng.random() < valid_probability)
    measured = distance + bias + float(rng.normal(0.0, std)) if valid else None
    return {
        "true_range": distance,
        "measured_range": measured,
        "los": int(los),
        "measurement_std": std,
        "valid": valid,
        "nlos_reason": "obstacle" if not los else "",
    }


def sample_visual_odometry(
    state: UAVKinematicState,
    dt_s: float,
    rng: np.random.Generator,
    config: SensorConfig,
    *,
    reference_position: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sample a local visual-odometry displacement.

    ``state`` is the post-motion state for the current decision window.  When
    the caller supplies ``reference_position``, the displacement is computed
    from the actual pre-motion position, including any boundary or safety
    projection.  The analytical fallback is retained for callers that only
    have the current state and assumes constant acceleration over ``dt_s``.
    """

    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if reference_position is None:
        # v_t * dt - 1/2 a_t * dt^2 equals the constant-acceleration
        # displacement v_{t-1} * dt + 1/2 a_t * dt^2.
        displacement = state.velocity_mps * dt_s - 0.5 * state.acceleration_mps2 * dt_s * dt_s
    else:
        displacement = state.position_m - np.asarray(reference_position, dtype=float).reshape(3)

    valid = int(rng.random() > 0.04)
    if not valid:
        return {"delta_position": None, "delta_yaw": None, "valid": 0, "quality": 0.0}
    return {
        "delta_position": displacement + _noise(rng, config.vo_position_std_m),
        "delta_yaw": float(state.angular_velocity_rps[2] * dt_s + rng.normal(0.0, config.vo_yaw_std_rad)),
        "valid": 1,
        "quality": float(np.clip(rng.normal(0.85, 0.05), 0.0, 1.0)),
    }


def sample_gnss(
    state: UAVKinematicState,
    mode: str,
    rng: np.random.Generator,
    config: SensorConfig,
) -> dict[str, Any]:
    normalized = mode.lower()
    if normalized in {"nominal", "normal"}:
        std, valid = config.gnss_nominal_std_m, 1
        fix_type = "3D"
    elif normalized in {"degraded", "intermittent"}:
        std, valid = config.gnss_degraded_std_m, int(rng.random() > 0.25)
        fix_type = "3D" if valid else "NO_FIX"
    else:
        std, valid = config.gnss_denied_std_m, 0
        fix_type = "NO_FIX"
    measurement = state.position_m + _noise(rng, std) if valid else None
    return {
        "position": measurement,
        "std": std,
        "valid": valid,
        "fix_type": fix_type,
        "mode": normalized,
    }


__all__ = [
    "SensorConfig",
    "sample_gnss",
    "sample_imu",
    "sample_uwb",
    "sample_visual_odometry",
]
