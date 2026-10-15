"""Stationary five-step IMU-bias acquisition for the local EKF.

The startup arm is deliberately small and isolated from :class:`LocalEKF`.
It assumes only the declared startup fixture (zero commanded motion) and uses
the IMU values as a bias observation.  After the five prediction calls the
parent EKF implementation is used unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..simulation.ekf import (
    ACCEL_BIAS_SLICE,
    ATTITUDE_SLICE,
    GYRO_BIAS_SLICE,
    POSITION_SLICE,
    STATE_DIM,
    VELOCITY_SLICE,
    EKFConfig,
    LocalEKF,
)
from ..simulation.sensor_model import SensorConfig


_STARTUP_STEPS = 5
_CROSS_TOLERANCE = 1.0e-12


class StartupLocalEKF(LocalEKF):
    """A ``LocalEKF`` with a bounded stationary bias-acquisition prefix.

    The parent constructor is intentionally left in charge of state and
    covariance initialization.  During the first five calls to
    :meth:`predict`, the state transition contains only position from the
    existing velocity, while a Joseph-form IMU bias observation estimates the
    six bias components.  Position/velocity/attitude and bias covariance
    blocks are kept separate throughout this prefix so that the subsequent
    temporal VO update has the same 30-state interpretation as the native
    implementation.
    """

    def __init__(
        self,
        initial_state: Iterable[float],
        config: EKFConfig | None = None,
        *,
        sensor_config: SensorConfig | None = None,
        startup_steps: int = _STARTUP_STEPS,
        vo_motion_limit: float = 16.27,
    ) -> None:
        if (
            isinstance(startup_steps, bool)
            or not isinstance(startup_steps, (int, np.integer))
            or int(startup_steps) != _STARTUP_STEPS
        ):
            raise ValueError("startup_steps is fixed at five prediction calls")
        try:
            motion_limit = float(vo_motion_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("vo_motion_limit must be positive and finite") from exc
        if not np.isfinite(motion_limit) or motion_limit <= 0.0:
            raise ValueError("vo_motion_limit must be positive and finite")
        super().__init__(initial_state, config)
        self.sensor_config = sensor_config or SensorConfig()
        self.startup_steps = _STARTUP_STEPS
        self.vo_motion_limit = motion_limit
        self._prediction_call_count = 0
        self._last_predict_was_startup = False
        self.startup_valid_vo_count = 0
        self.startup_bias_update_count = 0
        # Aliases make the counters explicit to diagnostic callers without
        # changing LocalEKF's VO/GNSS observation counters.
        self.valid_startup_vo_count = 0
        self.valid_bias_update_count = 0
        self.last_startup_update: dict[str, Any] = {}

    @staticmethod
    def _cross_block(covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
        return np.concatenate(
            (
                covariance[:9, 9:15].reshape(-1),
                covariance[9:15, :9].reshape(-1),
            )
        )

    def _require_startup_covariance(self, covariance: np.ndarray | None = None, *, where: str) -> None:
        """Reject invalid startup covariance instead of silently repairing it."""

        matrix = self.covariance if covariance is None else np.asarray(covariance, dtype=float)
        matrix = np.asarray(matrix, dtype=float).reshape(STATE_DIM, STATE_DIM)
        if not np.isfinite(matrix).all():
            raise FloatingPointError(f"startup covariance is non-finite {where}")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"startup covariance is not symmetric {where}")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
        if minimum_eigenvalue < -1.0e-10:
            raise ValueError(f"startup covariance is materially non-PSD {where}")
        cross = self._cross_block(matrix)
        if cross.size and float(np.max(np.abs(cross))) > _CROSS_TOLERANCE:
            raise ValueError(f"startup covariance has position/bias cross-covariance {where}")

    @staticmethod
    def _finite_vector(value: object, name: str) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=float).reshape(3)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite 3-D vector") from exc
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must be finite")
        return vector

    def _native_process_covariance(self, dt_s: float) -> np.ndarray:
        """Extract the native Q without sharing state or covariance storage."""

        temporary = LocalEKF(self.state.copy(), self.config)
        temporary.covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        temporary.predict({"accel": np.zeros(3), "gyro": np.zeros(3)}, dt_s)
        process = np.asarray(temporary.covariance, dtype=float).copy()
        if not np.isfinite(process).all():
            raise FloatingPointError("native startup process covariance is non-finite")
        return process

    def _startup_bias_observation(self, accel: np.ndarray, gyro: np.ndarray) -> None:
        """Apply one uncensored Joseph-form IMU bias observation."""

        self._require_startup_covariance(where="before bias observation")
        accel_std = float(self.sensor_config.imu_accel_std_mps2)
        gyro_std = float(self.sensor_config.imu_gyro_std_rps)
        if (
            not np.isfinite(accel_std)
            or not np.isfinite(gyro_std)
            or accel_std <= 0.0
            or gyro_std <= 0.0
        ):
            raise ValueError("IMU sensor standard deviations must be positive and finite")

        measurement = np.concatenate((accel, gyro)).astype(float, copy=False)
        jacobian = np.zeros((6, STATE_DIM), dtype=float)
        jacobian[:3, ACCEL_BIAS_SLICE] = np.eye(3)
        jacobian[3:, GYRO_BIAS_SLICE] = np.eye(3)
        noise = np.diag(
            np.r_[np.full(3, accel_std**2), np.full(3, gyro_std**2)]
        ).astype(float)
        prior_state = self.state.copy()
        prior_covariance = self.covariance.copy()
        innovation = measurement - jacobian @ prior_state
        innovation_covariance = jacobian @ prior_covariance @ jacobian.T + noise
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        if not np.isfinite(innovation_covariance).all():
            raise FloatingPointError("startup bias innovation covariance is non-finite")
        minimum = float(np.min(np.linalg.eigvalsh(innovation_covariance)))
        if minimum <= 0.0:
            raise FloatingPointError("startup bias innovation covariance is not positive definite")
        inverse_innovation_covariance = np.linalg.inv(innovation_covariance)
        nis = float(innovation @ inverse_innovation_covariance @ innovation.T)
        if not np.isfinite(nis):
            raise FloatingPointError("startup bias NIS is non-finite")

        gain = prior_covariance @ jacobian.T @ inverse_innovation_covariance
        identity = np.eye(STATE_DIM, dtype=float)
        self.state = prior_state + gain @ innovation
        residual_map = identity - gain @ jacobian
        self.covariance = (
            residual_map @ prior_covariance @ residual_map.T
            + gain @ noise @ gain.T
        )
        self._stabilize_covariance()
        self._require_startup_covariance(where="after bias observation")
        self.last_nis = nis
        self.startup_bias_update_count += 1
        self.valid_bias_update_count = self.startup_bias_update_count
        self.last_startup_update = {
            "active": True,
            "accepted": True,
            "sensor": "imu_bias",
            "measurement": measurement.tolist(),
            "calibration_measurement": measurement.tolist(),
            "innovation": innovation.tolist(),
            "innovation_covariance_trace": float(np.trace(innovation_covariance)),
            "nis": nis,
            "measurement_noise_diagonal": np.diag(noise).tolist(),
            "bias_gain": gain[9:15].tolist(),
            "gain_norm": float(np.linalg.norm(gain[9:15])),
            "weight": np.diag(gain[9:15, :]).tolist(),
            "prior_bias": prior_state[9:15].tolist(),
            "posterior_bias": self.state[9:15].tolist(),
            "prior_covariance_trace": float(np.trace(prior_covariance)),
            "posterior_covariance_trace": float(np.trace(self.covariance)),
            "prediction_index": int(self._prediction_call_count),
        }

    def predict(self, imu: dict[str, object], dt_s: float) -> np.ndarray:
        """Predict, acquiring a stationary IMU bias during the first five calls."""

        try:
            dt = float(dt_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt_s must be positive and finite") from exc
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if not isinstance(imu, dict) or "accel" not in imu or "gyro" not in imu:
            raise ValueError("IMU measurements must provide accel and gyro")
        accel = self._finite_vector(imu["accel"], "IMU accel")
        gyro = self._finite_vector(imu["gyro"], "IMU gyro")

        startup = self._prediction_call_count < self.startup_steps
        if not startup:
            transition = super().predict({"accel": accel, "gyro": gyro}, dt)
            self._prediction_call_count += 1
            self._last_predict_was_startup = False
            if self.last_startup_update:
                self.last_startup_update = {
                    **self.last_startup_update,
                    "active": False,
                    "latest_calibration_index": int(self.startup_bias_update_count),
                }
            return transition

        self._require_startup_covariance(where="before startup prediction")
        if not np.isfinite(self.state).all():
            raise FloatingPointError("startup state is non-finite")
        previous_state = self.state.copy()
        previous_covariance = self.covariance.copy()
        transition = np.eye(STATE_DIM, dtype=float)
        transition[POSITION_SLICE, VELOCITY_SLICE] = np.eye(3, dtype=float) * dt
        process = self._native_process_covariance(dt)

        # Stationary startup intentionally does not use IMU acceleration or
        # angular velocity as motion.  The raw values enter exactly once as
        # the bias observation below.
        self.state[POSITION_SLICE] = previous_state[POSITION_SLICE] + previous_state[VELOCITY_SLICE] * dt
        self.state[VELOCITY_SLICE] = previous_state[VELOCITY_SLICE]
        self.state[ATTITUDE_SLICE] = previous_state[ATTITUDE_SLICE]
        self.state[9:15] = previous_state[9:15]
        self.covariance = transition @ previous_covariance @ transition.T + process
        self._stabilize_covariance()
        self._require_startup_covariance(where="after startup FQ")

        # Keep the nominal prediction context before conditioning the current
        # bias block.  Native relative VO consumes this prior/F pair.
        self.last_prediction_context = {
            "previous_state": previous_state.copy(),
            "previous_covariance": previous_covariance.copy(),
            "predicted_state": self.state.copy(),
            "predicted_covariance": self.covariance.copy(),
            "nominal_predicted_state": self.state.copy(),
            "nominal_predicted_covariance": self.covariance.copy(),
            "transition_matrix": transition.copy(),
            "dt_s": dt,
        }
        self._prediction_call_count += 1
        self._last_predict_was_startup = True
        self._startup_bias_observation(accel, gyro)
        return transition

    def update_visual_delta(
        self,
        delta_position: np.ndarray,
        covariance: np.ndarray | None = None,
        *,
        previous_state: np.ndarray | None = None,
        previous_covariance: np.ndarray | None = None,
        transition_matrix: np.ndarray | None = None,
        reference_position: np.ndarray | None = None,
        dt_s: float = 0.2,
        previous_time_s: float | None = None,
        current_time_s: float | None = None,
    ) -> bool:
        """Apply native relative VO, with a stationary-motion startup guard."""

        if not self._last_predict_was_startup:
            return super().update_visual_delta(
                delta_position,
                covariance=covariance,
                previous_state=previous_state,
                previous_covariance=previous_covariance,
                transition_matrix=transition_matrix,
                reference_position=reference_position,
                dt_s=dt_s,
                previous_time_s=previous_time_s,
                current_time_s=current_time_s,
            )

        self._require_startup_covariance(where="before startup VO")
        displacement = self._finite_vector(delta_position, "relative VO displacement")
        nominal_variance = float(self.sensor_config.vo_position_std_m) ** 2
        if not np.isfinite(nominal_variance) or nominal_variance <= 0.0:
            raise ValueError("stationary_start_unsupported: VO covariance is invalid")
        nominal_noise = np.eye(3, dtype=float) * nominal_variance
        if covariance is None:
            noise = nominal_noise.copy()
        else:
            try:
                noise = np.asarray(covariance, dtype=float).reshape(3, 3)
            except (TypeError, ValueError) as exc:
                raise ValueError("stationary_start_unsupported: VO covariance must be 3x3") from exc
            if not np.isfinite(noise).all():
                raise ValueError("stationary_start_unsupported: VO covariance must be finite")
            if not np.allclose(noise, noise.T, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("stationary_start_unsupported: VO covariance must be symmetric")
            if float(np.min(np.linalg.eigvalsh(noise))) <= 0.0:
                raise ValueError("stationary_start_unsupported: VO covariance must be positive definite")
            if not np.allclose(noise, nominal_noise, rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("stationary_start_unsupported: VO covariance must equal SensorConfig nominal covariance")
        if not np.isfinite(noise).all() or float(np.min(np.linalg.eigvalsh(noise))) <= 0.0:
            raise ValueError("stationary_start_unsupported: VO covariance must be positive definite")
        quadratic = float(displacement @ np.linalg.solve(noise, displacement))
        if not np.isfinite(quadratic) or quadratic > self.vo_motion_limit:
            raise ValueError(
                "stationary_start_unsupported: relative VO motion exceeds startup limit"
            )
        self.startup_valid_vo_count += 1
        self.valid_startup_vo_count = self.startup_valid_vo_count
        try:
            return super().update_visual_delta(
                displacement,
                covariance=noise,
                previous_state=previous_state,
                previous_covariance=previous_covariance,
                transition_matrix=transition_matrix,
                reference_position=reference_position,
                dt_s=dt_s,
                previous_time_s=previous_time_s,
                current_time_s=current_time_s,
            )
        finally:
            self._require_startup_covariance(where="after startup VO")

    def update_position(
        self,
        position: np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> bool:
        """Use native GNSS and enforce startup block independence."""

        if not self._last_predict_was_startup:
            return super().update_position(position, covariance)
        self._require_startup_covariance(where="before startup GNSS")
        try:
            accepted = super().update_position(position, covariance)
        finally:
            self._require_startup_covariance(where="after startup GNSS")
        return accepted

    def check_startup_evidence(self) -> None:
        """Require exactly five startup calls and at least one valid VO packet."""

        if self._prediction_call_count != self.startup_steps:
            raise ValueError("startup evidence requires exactly five completed predictions")
        if self.startup_valid_vo_count < 1:
            raise ValueError("startup evidence requires at least one valid VO packet")


__all__ = ["StartupLocalEKF"]
