"""Local EKF and conservative relative-range fusion for the Python environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


STATE_DIM = 15
POSITION_SLICE = slice(0, 3)
VELOCITY_SLICE = slice(3, 6)
ATTITUDE_SLICE = slice(6, 9)
ACCEL_BIAS_SLICE = slice(9, 12)
GYRO_BIAS_SLICE = slice(12, 15)


@dataclass(frozen=True)
class EKFConfig:
    process_position_std_m: float = 0.03
    process_velocity_std_mps: float = 0.08
    process_attitude_std_rad: float = 0.01
    process_bias_std: float = 0.001
    # A boundary or obstacle projection is a non-smooth part of the physical
    # transition that is not represented by the constant-acceleration EKF
    # model.  These terms are added only when such a projection is actually
    # applied; they are not a generic covariance inflation or a truth reset.
    projection_position_std_m: float = 1.0
    projection_velocity_std_mps: float = 1.0
    gnss_position_std_m: float = 0.5
    vo_position_std_m: float = 0.08
    uwb_range_std_m: float = 0.1
    stale_message_limit_s: float = 0.6
    innovation_gate_threshold: float = 9.0


def covariance_intersection(
    mean_one: np.ndarray,
    covariance_one: np.ndarray,
    mean_two: np.ndarray,
    covariance_two: np.ndarray,
    *,
    grid_size: int = 51,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuse two estimates with unknown cross-correlation conservatively."""

    covariance_one = np.asarray(covariance_one, dtype=float)
    covariance_two = np.asarray(covariance_two, dtype=float)
    mean_one = np.asarray(mean_one, dtype=float)
    mean_two = np.asarray(mean_two, dtype=float)
    # The precision matrices do not depend on the grid weight.  Computing
    # their pseudoinverses once is materially faster during the EKF calibration
    # sweep while preserving the same 51-point conservative search.
    precision_one = np.linalg.pinv(covariance_one)
    precision_two = np.linalg.pinv(covariance_two)
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for weight in np.linspace(0.0, 1.0, max(2, grid_size)):
        precision = weight * precision_one + (1.0 - weight) * precision_two
        covariance = np.linalg.pinv(precision)
        mean = covariance @ (
            weight * precision_one @ mean_one
            + (1.0 - weight) * precision_two @ mean_two
        )
        score = float(np.trace(covariance))
        if best is None or score < best[0]:
            best = (score, mean, covariance, float(weight))
    assert best is not None
    return best[1], best[2], best[3]


class LocalEKF:
    """A deterministic local EKF with position, velocity, attitude, and biases."""

    def __init__(self, initial_state: Iterable[float], config: EKFConfig | None = None) -> None:
        vector = np.asarray(tuple(initial_state), dtype=float).reshape(STATE_DIM)
        self.config = config or EKFConfig()
        self.state = vector
        self.covariance = np.eye(STATE_DIM, dtype=float)
        self.covariance[:3, :3] *= 4.0
        self.covariance[3:6, 3:6] *= 1.0
        self.valid_observation_count = 0
        self.valid_neighbor_count = 0
        self.last_neighbor_timestamp: float | None = None
        self.last_nis: float | None = None
        self.rejected_observation_count = 0
        # The environment saves this prediction context before applying the
        # relative VO update.  Keeping it on the filter makes it possible to
        # reject a measurement that is paired with a different time window or
        # a different prior covariance.
        self.last_prediction_context: dict[str, Any] | None = None
        self.last_visual_update: dict[str, Any] = {}
        # Read-only mechanism traces.  They expose calculations already made
        # by the filter and never participate in a later update.
        self.last_linear_update: dict[str, Any] = {}
        self.last_relative_range_update: dict[str, Any] = {}
        self._step_diagnostics: dict[str, Any] = {}
        self.begin_step()

    @property
    def position(self) -> np.ndarray:
        return self.state[POSITION_SLICE]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[VELOCITY_SLICE]

    @property
    def attitude(self) -> np.ndarray:
        return self.state[ATTITUDE_SLICE]

    def predict(self, imu: dict[str, object], dt_s: float) -> np.ndarray:
        """Run the IMU prediction and return its linear transition matrix."""

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        accel = np.asarray(imu["accel"], dtype=float).reshape(3)
        gyro = np.asarray(imu["gyro"], dtype=float).reshape(3)
        if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
            raise ValueError("IMU measurements must be finite")
        previous_state = self.state.copy()
        previous_covariance = self.covariance.copy()
        accel_bias = self.state[ACCEL_BIAS_SLICE]
        gyro_bias = self.state[GYRO_BIAS_SLICE]
        corrected_accel = accel - accel_bias
        corrected_gyro = gyro - gyro_bias
        self.state[POSITION_SLICE] += self.state[VELOCITY_SLICE] * dt_s + 0.5 * corrected_accel * dt_s * dt_s
        self.state[VELOCITY_SLICE] += corrected_accel * dt_s
        self.state[ATTITUDE_SLICE] += corrected_gyro * dt_s

        transition = np.eye(STATE_DIM, dtype=float)
        transition[0:3, 3:6] = np.eye(3) * dt_s
        # Linearization of the same constant-acceleration model used by the
        # motion simulator.  The bias terms must be present here; otherwise
        # bias uncertainty is not propagated into position and velocity.
        transition[0:3, 9:12] = -0.5 * np.eye(3) * dt_s * dt_s
        transition[3:6, 9:12] = -np.eye(3) * dt_s
        transition[6:9, 12:15] = -np.eye(3) * dt_s

        # Discrete white-acceleration covariance.  ``process_velocity_std`` is
        # interpreted as the one-window velocity standard deviation.  The
        # resulting position/velocity cross block is required by the same
        # constant-acceleration transition and is important for correlated VO.
        acceleration_variance = (
            self.config.process_velocity_std_mps / max(float(dt_s), 1e-12)
        ) ** 2
        process = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        process[0:3, 0:3] += np.eye(3) * (
            self.config.process_position_std_m**2
            + 0.25 * dt_s**4 * acceleration_variance
        )
        process[0:3, 3:6] += np.eye(3) * (0.5 * dt_s**3 * acceleration_variance)
        process[3:6, 0:3] += np.eye(3) * (0.5 * dt_s**3 * acceleration_variance)
        process[3:6, 3:6] += np.eye(3) * (dt_s**2 * acceleration_variance)
        process[6:9, 6:9] = np.eye(3) * self.config.process_attitude_std_rad**2
        process[9:15, 9:15] = np.eye(6) * self.config.process_bias_std**2
        self.covariance = transition @ self.covariance @ transition.T + process
        self._stabilize_covariance()
        self.last_prediction_context = {
            "previous_state": previous_state,
            "previous_covariance": previous_covariance,
            "predicted_state": self.state.copy(),
            "predicted_covariance": self.covariance.copy(),
            "transition_matrix": transition.copy(),
            "dt_s": float(dt_s),
        }
        return transition

    def begin_step(self) -> None:
        """Reset per-decision-window sensor update diagnostics."""

        self._step_diagnostics = {
            "nis_vo_values": [],
            "nis_vo_accepted_values": [],
            "nis_vo_rejected_values": [],
            "nis_uwb_values": [],
            "nis_uwb_accepted_values": [],
            "nis_uwb_rejected_values": [],
            "nis_gnss_values": [],
            "nis_gnss_accepted_values": [],
            "nis_gnss_rejected_values": [],
            "vo_update_attempted": 0,
            "vo_update_accepted": 0,
            "vo_update_rejected": 0,
            "uwb_update_attempted": 0,
            "uwb_update_accepted": 0,
            "uwb_update_rejected": 0,
            "gnss_update_attempted": 0,
            "gnss_update_accepted": 0,
            "gnss_update_rejected": 0,
            "rejected_observation_count": 0,
        }

    def _stabilize_covariance(self) -> None:
        """Keep the covariance finite, symmetric, and numerically PSD."""

        covariance = 0.5 * (np.asarray(self.covariance, dtype=float) + np.asarray(self.covariance, dtype=float).T)
        if not np.isfinite(covariance).all():
            raise FloatingPointError("EKF covariance became non-finite")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if minimum_eigenvalue < 1e-9:
            covariance = covariance + np.eye(STATE_DIM) * (1e-9 - minimum_eigenvalue)
        self.covariance = 0.5 * (covariance + covariance.T)

    def inflate_motion_uncertainty(
        self,
        *,
        position_std_m: float | None = None,
        velocity_std_mps: float | None = None,
    ) -> None:
        """Add uncertainty for an externally projected motion transition.

        The simulator may clip a proposed motion at a flight-volume or safety
        boundary.  The EKF does not receive the true position used by that
        projection, so the non-smooth transition is represented as additional
        process uncertainty.  This method intentionally changes only the
        covariance; it never injects a ground-truth state or measurement.
        """

        position_std = float(
            self.config.projection_position_std_m
            if position_std_m is None
            else position_std_m
        )
        velocity_std = float(
            self.config.projection_velocity_std_mps
            if velocity_std_mps is None
            else velocity_std_mps
        )
        if position_std < 0.0 or velocity_std < 0.0:
            raise ValueError("projection uncertainty standard deviations must be non-negative")
        self.covariance[POSITION_SLICE, POSITION_SLICE] += np.eye(3) * position_std**2
        self.covariance[VELOCITY_SLICE, VELOCITY_SLICE] += np.eye(3) * velocity_std**2
        self._stabilize_covariance()

    def _record_update(self, sensor: str, accepted: bool, nis: float) -> None:
        sensor = str(sensor).lower()
        if sensor not in {"vo", "uwb", "gnss"}:
            return
        self._step_diagnostics[f"{sensor}_update_attempted"] += 1
        if sensor == "vo":
            self._step_diagnostics["nis_vo_values"].append(float(nis))
            self._step_diagnostics[
                "nis_vo_accepted_values" if accepted else "nis_vo_rejected_values"
            ].append(float(nis))
        elif sensor == "uwb":
            self._step_diagnostics["nis_uwb_values"].append(float(nis))
            self._step_diagnostics[
                "nis_uwb_accepted_values" if accepted else "nis_uwb_rejected_values"
            ].append(float(nis))
        else:
            self._step_diagnostics["nis_gnss_values"].append(float(nis))
            self._step_diagnostics[
                "nis_gnss_accepted_values" if accepted else "nis_gnss_rejected_values"
            ].append(float(nis))
        if accepted:
            self._step_diagnostics[f"{sensor}_update_accepted"] += 1
        else:
            self._step_diagnostics[f"{sensor}_update_rejected"] += 1
            self._step_diagnostics["rejected_observation_count"] += 1

    def diagnostics_snapshot(self) -> dict[str, float | int]:
        """Return current-window NIS and update counters for logging only."""

        def _summary(key: str, reducer: str = "mean") -> float:
            values = np.asarray(self._step_diagnostics.get(key, []), dtype=float)
            # ``-1`` is the schema sentinel for a missing/invalid NIS and
            # must not be included in a mean or maximum.
            values = values[np.isfinite(values) & (values >= 0.0)]
            if values.size == 0:
                return -1.0
            return float(np.max(values) if reducer == "max" else np.mean(values))

        return {
            "nis_vo": _summary("nis_vo_values"),
            "nis_uwb_mean": _summary("nis_uwb_values"),
            "nis_uwb_max": _summary("nis_uwb_values", "max"),
            "nis_gnss": _summary("nis_gnss_values"),
            "nis_vo_accepted": _summary("nis_vo_accepted_values"),
            "nis_vo_rejected": _summary("nis_vo_rejected_values"),
            "nis_uwb_accepted_mean": _summary("nis_uwb_accepted_values"),
            "nis_uwb_rejected_mean": _summary("nis_uwb_rejected_values"),
            "nis_gnss_accepted": _summary("nis_gnss_accepted_values"),
            "nis_gnss_rejected": _summary("nis_gnss_rejected_values"),
            "vo_update_attempted": int(self._step_diagnostics["vo_update_attempted"]),
            "vo_update_accepted": int(self._step_diagnostics["vo_update_accepted"]),
            "vo_update_rejected": int(self._step_diagnostics["vo_update_rejected"]),
            "uwb_update_attempted": int(self._step_diagnostics["uwb_update_attempted"]),
            "uwb_update_accepted": int(self._step_diagnostics["uwb_update_accepted"]),
            "uwb_update_rejected": int(self._step_diagnostics["uwb_update_rejected"]),
            "gnss_update_attempted": int(self._step_diagnostics["gnss_update_attempted"]),
            "gnss_update_accepted": int(self._step_diagnostics["gnss_update_accepted"]),
            "gnss_update_rejected": int(self._step_diagnostics["gnss_update_rejected"]),
            # This field is per decision window.  The cumulative counter is
            # retained internally for lifetime diagnostics but is not mixed
            # into a per-step CSV record.
            "rejected_observation_count": int(self._step_diagnostics["rejected_observation_count"]),
            "vo_innovation_norm": float(self.last_visual_update.get("innovation_norm", -1.0)),
            "vo_innovation_covariance_trace": float(
                self.last_visual_update.get("innovation_covariance_trace", -1.0)
            ),
            "vo_measurement_covariance_trace": float(
                self.last_visual_update.get("measurement_covariance_trace", -1.0)
            ),
            "vo_measurement_covariance_scale": float(
                self.last_visual_update.get("measurement_covariance_scale", 1.0)
            ),
            "vo_time_delta_s": float(self.last_visual_update.get("time_delta_s", -1.0)),
        }

    def _linear_update(
        self,
        measurement: np.ndarray,
        expected: np.ndarray,
        jacobian: np.ndarray,
        noise: np.ndarray,
        *,
        gate_threshold: float | None = None,
        sensor: str = "unknown",
    ) -> bool:
        measurement = np.asarray(measurement, dtype=float).reshape(-1)
        expected = np.asarray(expected, dtype=float).reshape(-1)
        jacobian = np.asarray(jacobian, dtype=float)
        noise = np.asarray(noise, dtype=float)
        innovation = measurement - expected
        innovation_covariance = jacobian @ self.covariance @ jacobian.T + noise
        self.last_nis = float(innovation @ np.linalg.pinv(innovation_covariance) @ innovation.T)
        prior_state = self.state.copy()
        prior_covariance = self.covariance.copy()
        if gate_threshold is not None and self.last_nis > float(gate_threshold):
            self.last_linear_update = {
                "accepted": False,
                "innovation": innovation.tolist(),
                "innovation_covariance": innovation_covariance.tolist(),
                "nis": float(self.last_nis),
                "gain_norm": 0.0,
                "state_correction": np.zeros(STATE_DIM, dtype=float).tolist(),
                "prior_state": prior_state.tolist(),
                "posterior_state": prior_state.tolist(),
                "prior_covariance_trace": float(np.trace(prior_covariance)),
                "posterior_covariance_trace": float(np.trace(prior_covariance)),
                "prior_position_covariance_trace": float(
                    np.trace(prior_covariance[POSITION_SLICE, POSITION_SLICE])
                ),
                "posterior_position_covariance_trace": float(
                    np.trace(prior_covariance[POSITION_SLICE, POSITION_SLICE])
                ),
            }
            self.rejected_observation_count += 1
            self._record_update(sensor, False, self.last_nis)
            return False
        gain = self.covariance @ jacobian.T @ np.linalg.pinv(innovation_covariance)
        identity = np.eye(STATE_DIM)
        self.state = self.state + gain @ innovation
        self.covariance = (
            (identity - gain @ jacobian)
            @ self.covariance
            @ (identity - gain @ jacobian).T
            + gain @ noise @ gain.T
        )
        self._stabilize_covariance()
        self.last_linear_update = {
            "accepted": True,
            "innovation": innovation.tolist(),
            "innovation_covariance": innovation_covariance.tolist(),
            "nis": float(self.last_nis),
            "gain_norm": float(np.linalg.norm(gain)),
            "state_correction": (self.state - prior_state).tolist(),
            "prior_state": prior_state.tolist(),
            "posterior_state": self.state.tolist(),
            "prior_covariance_trace": float(np.trace(prior_covariance)),
            "posterior_covariance_trace": float(np.trace(self.covariance)),
            "prior_position_covariance_trace": float(
                np.trace(prior_covariance[POSITION_SLICE, POSITION_SLICE])
            ),
            "posterior_position_covariance_trace": float(
                np.trace(self.covariance[POSITION_SLICE, POSITION_SLICE])
            ),
        }
        self._record_update(sensor, True, self.last_nis)
        self.valid_observation_count += 1
        return True

    def update_position(self, position: np.ndarray, covariance: np.ndarray | None = None) -> bool:
        measurement = np.asarray(position, dtype=float).reshape(3)
        jacobian = np.zeros((3, STATE_DIM), dtype=float)
        jacobian[:, POSITION_SLICE] = np.eye(3)
        noise = np.asarray(covariance if covariance is not None else np.eye(3) * self.config.gnss_position_std_m**2, dtype=float)
        return self._linear_update(
            measurement,
            self.position,
            jacobian,
            noise,
            gate_threshold=self.config.innovation_gate_threshold,
            sensor="gnss",
        )

    def project_position_bounds(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> bool:
        """Project the estimated position into the configured flight volume.

        The physical simulator already constrains true motion. A nonlinear
        relative-range update can nevertheless produce a temporary EKF mean
        outside that volume, especially when several delayed neighbor
        estimates are correlated. The projection is an explicit constrained
        estimate step, and its count is exposed by the environment for
        diagnostics rather than silently hidden.
        """

        lower = np.asarray(lower_bounds, dtype=float).reshape(3)
        upper = np.asarray(upper_bounds, dtype=float).reshape(3)
        if np.any(lower >= upper):
            raise ValueError("lower_bounds must be strictly below upper_bounds")
        original = self.position.copy()
        projected = np.clip(original, lower, upper)
        changed = not np.allclose(original, projected)
        if changed:
            self.state[POSITION_SLICE] = projected
            correction = projected - original
            # Preserve a conservative uncertainty record for the constrained
            # correction instead of reporting an unchanged overconfident mean.
            self.covariance[POSITION_SLICE, POSITION_SLICE] += np.outer(correction, correction)
            self._stabilize_covariance()
        return changed

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
        """Update position from a local VO displacement.

        VO supplies a displacement between two consecutive states.  It is
        therefore processed as a correlated relative measurement, never as an
        independent absolute position observation.  The cross-covariance
        between the previous and current EKF states is retained in the
        innovation covariance.
        """

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if (previous_time_s is None) != (current_time_s is None):
            raise ValueError("previous_time_s and current_time_s must be provided together")
        time_delta_s: float | None = None
        if previous_time_s is not None and current_time_s is not None:
            time_delta_s = float(current_time_s) - float(previous_time_s)
            tolerance = max(1e-9, 1e-6 * float(dt_s))
            if not np.isfinite(time_delta_s) or abs(time_delta_s - float(dt_s)) > tolerance:
                raise ValueError(
                    "relative VO timestamps do not match the prediction interval: "
                    f"delta={time_delta_s!r}, dt_s={dt_s!r}"
                )
        displacement = np.asarray(delta_position, dtype=float).reshape(3)
        if not np.isfinite(displacement).all():
            raise ValueError("relative VO displacement must be finite")
        current_state = self.state.copy()
        current_covariance = self.covariance.copy()
        explicit_temporal_context = previous_state is not None
        if previous_state is None:
            # Compatibility path for direct low-level callers.  It still uses
            # a relative measurement and never constructs an absolute VO
            # position observation.
            previous_state = current_state.copy()
            previous_state[POSITION_SLICE] = (
                np.asarray(reference_position, dtype=float).reshape(3)
                if reference_position is not None
                else current_state[POSITION_SLICE] - current_state[VELOCITY_SLICE] * dt_s
            )
        elif reference_position is not None:
            raise ValueError(
                "reference_position is only allowed for the compatibility path; "
                "the environment must pass the previous EKF state"
            )
        previous_state = np.asarray(previous_state, dtype=float).reshape(STATE_DIM)
        if previous_covariance is None:
            if explicit_temporal_context and self.last_prediction_context is not None:
                raise ValueError(
                    "previous_covariance is required when a prediction context exists"
                )
            previous_covariance = current_covariance.copy()
        previous_covariance = np.asarray(previous_covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
        if transition_matrix is None:
            if explicit_temporal_context and self.last_prediction_context is not None:
                raise ValueError(
                    "transition_matrix is required when a prediction context exists"
                )
            transition_matrix = np.eye(STATE_DIM, dtype=float)
            transition_matrix[POSITION_SLICE, VELOCITY_SLICE] = np.eye(3) * dt_s
        transition_matrix = np.asarray(transition_matrix, dtype=float).reshape(STATE_DIM, STATE_DIM)
        if not np.isfinite(previous_state).all() or not np.isfinite(previous_covariance).all():
            raise ValueError("relative VO prior state and covariance must be finite")
        if not np.isfinite(transition_matrix).all():
            raise ValueError("relative VO transition matrix must be finite")

        # If this filter has just predicted, a full temporal context is
        # mandatory and must refer to that exact prior/prediction pair.  This
        # prevents an accidentally shifted VO packet from shrinking the
        # covariance with the wrong cross-covariance.
        context = self.last_prediction_context
        if context is not None and explicit_temporal_context:
            if not np.allclose(previous_state, context["previous_state"], rtol=1e-8, atol=1e-10):
                raise ValueError("relative VO previous_state does not match the last EKF prediction")
            if not np.allclose(
                previous_covariance,
                context["previous_covariance"],
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError("relative VO previous_covariance does not match the last EKF prediction")
            if not np.allclose(
                transition_matrix,
                context["transition_matrix"],
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError("relative VO transition_matrix does not match the last EKF prediction")
            if abs(float(context["dt_s"]) - float(dt_s)) > max(1e-9, 1e-6 * float(dt_s)):
                raise ValueError("relative VO dt_s does not match the last EKF prediction")
        selector = np.zeros((3, STATE_DIM), dtype=float)
        selector[:, POSITION_SLICE] = np.eye(3)
        cross_previous_current = previous_covariance @ transition_matrix.T
        innovation = displacement - (
            current_state[POSITION_SLICE] - previous_state[POSITION_SLICE]
        )
        noise = np.asarray(
            covariance if covariance is not None else np.eye(3) * self.config.vo_position_std_m**2,
            dtype=float,
        ).reshape(3, 3)
        nominal_trace = max(1e-12, 3.0 * float(self.config.vo_position_std_m) ** 2)
        measurement_covariance_trace = float(np.trace(noise))
        measurement_covariance_scale = measurement_covariance_trace / nominal_trace
        if not np.isfinite(measurement_covariance_scale) or measurement_covariance_scale < 0.0:
            raise FloatingPointError("visual-odometry measurement covariance scale became non-finite")
        innovation_covariance = (
            selector @ current_covariance @ selector.T
            + selector @ previous_covariance @ selector.T
            - selector @ cross_previous_current @ selector.T
            - selector @ cross_previous_current.T @ selector.T
            + noise
        )
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        if not np.isfinite(innovation_covariance).all():
            raise FloatingPointError("relative VO innovation covariance became non-finite")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(innovation_covariance)))
        if minimum_eigenvalue < 1e-9:
            innovation_covariance = innovation_covariance + np.eye(3) * (1e-9 - minimum_eigenvalue)
        self.last_nis = float(innovation @ np.linalg.pinv(innovation_covariance) @ innovation.T)
        if not np.isfinite(self.last_nis):
            raise FloatingPointError("relative VO NIS became non-finite")
        self.last_visual_update = {
            "innovation_norm": float(np.linalg.norm(innovation)),
            "innovation_covariance_trace": float(np.trace(innovation_covariance)),
            "measurement_covariance_trace": measurement_covariance_trace,
            "measurement_covariance_scale": float(measurement_covariance_scale),
            "time_delta_s": float(time_delta_s if time_delta_s is not None else dt_s),
            "accepted": False,
        }
        accepted = self.last_nis <= float(self.config.innovation_gate_threshold)
        self.last_visual_update["accepted"] = bool(accepted)
        self._record_update("vo", accepted, self.last_nis)
        if not accepted:
            self.rejected_observation_count += 1
            return False
        cross_current_innovation = (
            current_covariance @ selector.T
            - cross_previous_current.T @ selector.T
        )
        gain = cross_current_innovation @ np.linalg.pinv(innovation_covariance)
        self.state = current_state + gain @ innovation
        self.covariance = current_covariance - gain @ innovation_covariance @ gain.T
        self._stabilize_covariance()
        self.valid_observation_count += 1
        return True

    def update_relative_range(
        self,
        measured_range_m: float,
        neighbor_position: np.ndarray,
        neighbor_covariance: np.ndarray | None = None,
        *,
        measurement_variance: float | None = None,
        nlos_bias_m: float = 0.0,
        timestamp: float | None = None,
    ) -> bool:
        neighbor_position = np.asarray(neighbor_position, dtype=float).reshape(3)
        difference = self.position - neighbor_position
        distance = float(np.linalg.norm(difference))
        if distance < 1e-8:
            self.last_relative_range_update = {
                "accepted": False,
                "rejection_reason": "degenerate_predicted_range",
            }
            self._record_update("uwb", False, -1.0)
            self.rejected_observation_count += 1
            return False
        direction = difference / distance
        jacobian = np.zeros((1, STATE_DIM), dtype=float)
        jacobian[0, POSITION_SLICE] = direction
        neighbor_variance = 0.0
        if neighbor_covariance is not None:
            neighbor_covariance = np.asarray(neighbor_covariance, dtype=float)
            neighbor_variance = float(direction @ neighbor_covariance[:3, :3] @ direction.T)
        # The local and neighbor estimates are not independent.  CI protects
        # repeated estimates of the same neighbor, but it does not remove the
        # unknown cross-correlation between this UAV and that neighbor.  Use
        # the Cauchy-Schwarz upper bound for the difference variance so a
        # relative-range update cannot become overconfident merely because the
        # cross-correlation was omitted from the state.
        local_variance = float(direction @ self.covariance[:3, :3] @ direction.T)
        unknown_cross_bound = 2.0 * float(
            np.sqrt(max(0.0, local_variance) * max(0.0, neighbor_variance))
        )
        measurement_variance_base = float(
            measurement_variance
            if measurement_variance is not None
            else self.config.uwb_range_std_m**2
        )
        effective_measurement_variance = max(
            1e-9,
            measurement_variance_base
            + max(0.0, neighbor_variance)
            + unknown_cross_bound,
        )
        # Save the local prior before the scalar update.  The neighbor packet
        # is an estimate produced by another filter, so the local state and
        # the packet can have an unknown cross-covariance.  Treating the
        # packet as an independent measurement would make repeated UWB range
        # updates progressively overconfident even when the relative range is
        # informative only about a shared/common-mode error.
        prior_covariance = self.covariance.copy()
        accepted = self._linear_update(
            np.array([float(measured_range_m)]),
            np.array([distance + nlos_bias_m]),
            jacobian,
            np.array([[effective_measurement_variance]], dtype=float),
            gate_threshold=self.config.innovation_gate_threshold,
            sensor="uwb",
        )
        if accepted:
            # The range update above is allowed to correct the local mean,
            # but it must not reduce the local absolute-state covariance when
            # the neighbor estimate has an unknown cross-correlation with the
            # local filter.  A range constrains a relative state; without a
            # maintained joint covariance, the fully correlated common-mode
            # case provides no guaranteed reduction of the local marginal
            # covariance.  Retaining the pre-range covariance is therefore a
            # conservative covariance-union bound, not a hand-tuned scale.
            # The neighbor uncertainty and the correlation upper bound have
            # already been included in the innovation variance used for the
            # NIS and mean gain above.
            if neighbor_covariance is not None:
                self.covariance = prior_covariance
                self._stabilize_covariance()
            self.valid_neighbor_count += 1
            self.last_neighbor_timestamp = timestamp
        linear = dict(self.last_linear_update)
        correction = np.asarray(linear.get("state_correction", np.zeros(STATE_DIM)), dtype=float)
        innovation_values = np.asarray(linear.get("innovation", [0.0]), dtype=float)
        innovation_covariance_values = np.asarray(
            linear.get("innovation_covariance", [[local_variance + effective_measurement_variance]]),
            dtype=float,
        )
        self.last_relative_range_update = {
            "accepted": bool(accepted),
            "rejection_reason": "none" if accepted else "innovation_gate",
            "measured_range_m": float(measured_range_m),
            "predicted_range_m": float(distance),
            "nlos_bias_m": float(nlos_bias_m),
            "expected_measurement_m": float(distance + nlos_bias_m),
            "innovation_m": float(innovation_values.reshape(-1)[0]),
            "nis": float(linear.get("nis", self.last_nis if self.last_nis is not None else -1.0)),
            "measurement_variance_m2": float(measurement_variance_base),
            "local_direction_variance_m2": float(local_variance),
            "neighbor_direction_variance_m2": float(max(0.0, neighbor_variance)),
            "unknown_cross_bound_m2": float(unknown_cross_bound),
            "effective_measurement_variance_m2": float(effective_measurement_variance),
            "innovation_variance_m2": float(innovation_covariance_values.reshape(-1)[0]),
            "kalman_gain_norm": float(linear.get("gain_norm", 0.0)),
            "state_correction_norm": float(np.linalg.norm(correction)),
            "position_correction_norm_m": float(np.linalg.norm(correction[POSITION_SLICE])),
            "position_correction_x_m": float(correction[0]),
            "position_correction_y_m": float(correction[1]),
            "position_correction_z_m": float(correction[2]),
            "state_before": linear.get("prior_state", prior_covariance.shape[0] * [0.0]),
            "state_after": linear.get("posterior_state", prior_covariance.shape[0] * [0.0]),
            "state_correction": linear.get("state_correction", prior_covariance.shape[0] * [0.0]),
            "position_covariance_trace_before_m2": float(np.trace(prior_covariance[POSITION_SLICE, POSITION_SLICE])),
            "position_covariance_trace_after_linear_m2": float(
                linear.get(
                    "posterior_position_covariance_trace",
                    np.trace(self.covariance[POSITION_SLICE, POSITION_SLICE]),
                )
            ),
            "position_covariance_trace_after_final_m2": float(np.trace(self.covariance[POSITION_SLICE, POSITION_SLICE])),
            "covariance_restored": bool(accepted and neighbor_covariance is not None),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
        }
        return accepted


__all__ = [
    "EKFConfig",
    "LocalEKF",
    "STATE_DIM",
    "covariance_intersection",
]
