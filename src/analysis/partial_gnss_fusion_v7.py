"""Measurement covariance-intersection fusion for the partial-GNSS v7 audit.

The classes in this module are intentionally small adapters around the frozen
v6/local EKF interfaces.  ``MeasurementCIEKF`` performs a scalar range update
using a conservative unknown-cross-covariance gate and a measurement CI
information update.  ``PartialGNSSJointEKF`` retains the v6 joint prediction
and VO path while allowing GNSS packets for the two permitted agents.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from src.analysis.cooperative_fusion_estimators_v6 import (
    JOINT_DIM,
    UIDS,
    JointEKF,
)
from src.simulation.ekf import LocalEKF, POSITION_SLICE, STATE_DIM


PSD_TOL = 1.0e-8
_CI_WEIGHTS = np.arange(1, 21, dtype=float) / 20.0


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _symmetric(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return 0.5 * (array + array.T)


def _positive_definite(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = _finite_array(value, shape, name)
    if not np.allclose(array, array.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    array = _symmetric(array)
    minimum = float(np.min(np.linalg.eigvalsh(array)))
    if minimum <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return array


def _positive_semidefinite(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = _finite_array(value, shape, name)
    if not np.allclose(array, array.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    array = _symmetric(array)
    minimum = float(np.min(np.linalg.eigvalsh(array)))
    if minimum < -PSD_TOL:
        raise ValueError(f"{name} must be positive semidefinite")
    # Do not floor, inflate, or rescale a covariance.  The tolerance only
    # permits ordinary roundoff at the PSD boundary; the matrix itself is
    # retained as supplied.
    return array


def _finite_scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def conditional_linear_error_bound(
    prior_covariance: np.ndarray,
    jacobian: np.ndarray,
    weight: float,
    gain: np.ndarray,
    pseudo_observation_variance: float,
) -> np.ndarray:
    """Return the fixed-H conditional linear error bound.

    For an unbiased prior and an unbiased scalar pseudo-observation, the
    covariance associated with a fixed linear gain is bounded by

    ``A P A.T / w + K (q + R) K.T / (1 - w)``, ``A = I - K H``.

    The identity with the CI posterior is checked by the pure tests and is
    exposed in update diagnostics.  This is a fixed-linearization result; it
    makes no universal claim for the nonlinear range function.
    """

    P = np.asarray(prior_covariance, dtype=float)
    H = np.asarray(jacobian, dtype=float)
    K = np.asarray(gain, dtype=float)
    w = float(weight)
    variance = float(pseudo_observation_variance)
    if P.shape != (STATE_DIM, STATE_DIM):
        raise ValueError("prior_covariance must be 15x15")
    if H.shape != (1, STATE_DIM):
        raise ValueError("jacobian must be 1x15")
    if K.shape != (STATE_DIM, 1):
        raise ValueError("gain must be 15x1")
    if not 0.0 < w < 1.0:
        raise ValueError("conditional bound requires 0 < weight < 1")
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("pseudo_observation_variance must be positive and finite")
    A = np.eye(STATE_DIM) - K @ H
    bound = A @ P @ A.T / w + K @ K.T * variance / (1.0 - w)
    return _symmetric(bound)


class MeasurementCIEKF(LocalEKF):
    """Local EKF with scalar measurement covariance-intersection range fusion."""

    def _range_event(
        self,
        *,
        accepted: bool,
        reason: str,
        measured_range: float,
        distance: float,
        nlos_bias: float,
        innovation: float,
        nis: float,
        weight: float,
        gain: np.ndarray,
        correction: np.ndarray,
        state_before: np.ndarray,
        state_after: np.ndarray,
        covariance_before: np.ndarray,
        covariance_after: np.ndarray,
        direction: np.ndarray,
        H: np.ndarray,
        R: float,
        q: float,
        p: float,
        S: float,
        conditional_bound: np.ndarray | None = None,
    ) -> dict[str, Any]:
        before_position_trace = float(np.trace(covariance_before[POSITION_SLICE, POSITION_SLICE]))
        after_position_trace = float(np.trace(covariance_after[POSITION_SLICE, POSITION_SLICE]))
        correction = np.asarray(correction, dtype=float).reshape(STATE_DIM)
        gain = np.asarray(gain, dtype=float).reshape(STATE_DIM, 1)
        direction = np.asarray(direction, dtype=float).reshape(3)
        event: dict[str, Any] = {
            "sensor": "uwb",
            "accepted": bool(accepted),
            "rejection_reason": str(reason),
            "measured_range_m": float(measured_range),
            "predicted_range_m": float(distance),
            "nlos_bias_m": float(nlos_bias),
            "expected_measurement_m": float(distance + nlos_bias),
            "innovation_m": float(innovation),
            "innovation": [float(innovation)],
            "nis": float(nis),
            "measurement_variance_m2": float(R),
            "local_direction_variance_m2": float(p),
            "neighbor_direction_variance_m2": float(q),
            "unknown_cross_bound_m2": float(2.0 * np.sqrt(max(0.0, p) * max(0.0, q))),
            "effective_measurement_variance_m2": float(q + R + 2.0 * np.sqrt(max(0.0, p) * max(0.0, q))),
            "innovation_variance_m2": float(S),
            "R": float(R),
            "q": float(q),
            "p": float(p),
            "S": float(S),
            "weight": float(weight),
            "ci_weight": float(weight),
            "gain": gain.tolist(),
            "H": H.tolist(),
            "kalman_gain_norm": float(np.linalg.norm(gain)),
            "gain_norm": float(np.linalg.norm(gain)),
            "state_correction": correction.tolist(),
            "state_correction_norm": float(np.linalg.norm(correction)),
            "correction_norm": float(np.linalg.norm(correction)),
            "position_correction_norm_m": float(np.linalg.norm(correction[POSITION_SLICE])),
            "position_correction_x_m": float(correction[0]),
            "position_correction_y_m": float(correction[1]),
            "position_correction_z_m": float(correction[2]),
            "state_before": state_before.tolist(),
            "state_after": state_after.tolist(),
            "position_covariance_trace_before_m2": before_position_trace,
            "position_covariance_trace_after_linear_m2": after_position_trace,
            "position_covariance_trace_after_final_m2": after_position_trace,
            "trace_before": before_position_trace,
            "trace_after": after_position_trace,
            "prior_covariance_trace": float(np.trace(covariance_before)),
            "posterior_covariance_trace": float(np.trace(covariance_after)),
            "prior_position_covariance_trace": before_position_trace,
            "posterior_position_covariance_trace": after_position_trace,
            "P_trace": float(np.trace(covariance_after)),
            "prior_trace": float(np.trace(covariance_before)),
            "post_trace": float(np.trace(covariance_after)),
            "full_position_trace": after_position_trace,
            "prior_full_position_covariance_trace": before_position_trace,
            "posterior_full_position_covariance_trace": after_position_trace,
            "covariance_restored": False,
            "covariance_restored_from_prior": False,
            "direction": direction.tolist(),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
            "state_covariance_after": covariance_after.tolist(),
        }
        if conditional_bound is not None:
            bound = np.asarray(conditional_bound, dtype=float)
            event.update(
                {
                    "conditional_linear_error_bound": bound.tolist(),
                    "conditional_bound_trace_m2": float(np.trace(bound)),
                    "conditional_bound_matches_posterior": bool(
                        np.allclose(bound, covariance_after, rtol=2.0e-8, atol=2.0e-10)
                    ),
                    "conditional_bound_residual_norm": float(np.linalg.norm(bound - covariance_after)),
                    "conditional_bound_scope": "fixed_H_linear_unbiased_prior_pseudo_observation",
                    "nonlinear_range_guarantee": False,
                }
            )
        else:
            event.update(
                {
                    "conditional_linear_error_bound": None,
                    "conditional_bound_trace_m2": float("nan"),
                    "conditional_bound_matches_posterior": False,
                    "conditional_bound_residual_norm": float("nan"),
                    "conditional_bound_scope": "fixed_H_linear_unbiased_prior_pseudo_observation",
                    "nonlinear_range_guarantee": False,
                }
            )
        return event

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
        """Apply a gated scalar range update with measurement covariance intersection."""

        # Validate every input and the current local covariance before changing
        # any field.  This is deliberately stricter than the native helper:
        # the v7 contract requires R > 0 and a positive-definite local P.
        measured_range = _finite_scalar(measured_range_m, "measured_range_m")
        if measured_range < 0.0:
            raise ValueError("measured_range_m must be non-negative")
        position = _finite_array(neighbor_position, (3,), "neighbor_position")
        nlos_bias = _finite_scalar(nlos_bias_m, "nlos_bias_m")
        if timestamp is not None:
            timestamp_value = _finite_scalar(timestamp, "timestamp")
        else:
            timestamp_value = None
        state_before = _finite_array(self.state, (STATE_DIM,), "local state")
        covariance_before = _positive_definite(self.covariance, (STATE_DIM, STATE_DIM), "local covariance")

        reference_covariance: np.ndarray | None
        if neighbor_covariance is None:
            reference_covariance = None
        else:
            raw_reference = np.asarray(neighbor_covariance, dtype=float)
            if raw_reference.shape == (3, 3):
                reference_covariance = _positive_semidefinite(
                    raw_reference, (3, 3), "neighbor position covariance"
                )
            elif raw_reference.shape == (STATE_DIM, STATE_DIM):
                reference_covariance = _positive_semidefinite(
                    raw_reference, (STATE_DIM, STATE_DIM), "neighbor covariance"
                )
            else:
                raise ValueError("neighbor_covariance must be 3x3 or 15x15")

        if measurement_variance is None:
            R = float(self.config.uwb_range_std_m) ** 2
        else:
            R = _finite_scalar(measurement_variance, "measurement_variance")
        if not np.isfinite(R) or R <= 0.0:
            raise ValueError("measurement_variance must be finite and strictly positive")

        difference = state_before[POSITION_SLICE] - position
        distance = float(np.linalg.norm(difference))
        if not np.isfinite(distance):
            raise FloatingPointError("predicted range became non-finite")

        # A coincident reference has no defined range Jacobian.  It is a valid
        # physical attempt and therefore updates rejection diagnostics, while
        # leaving the estimate itself unchanged.
        if distance < 1.0e-8:
            direction = np.zeros(3, dtype=float)
            H = np.zeros((1, STATE_DIM), dtype=float)
            q = 0.0
            p = 0.0
            S = R
            innovation = measured_range - nlos_bias
            nis = -1.0
            event = self._range_event(
                accepted=False,
                reason="degenerate_predicted_range",
                measured_range=measured_range,
                distance=distance,
                nlos_bias=nlos_bias,
                innovation=innovation,
                nis=nis,
                weight=1.0,
                gain=np.zeros((STATE_DIM, 1)),
                correction=np.zeros(STATE_DIM),
                state_before=state_before,
                state_after=state_before,
                covariance_before=covariance_before,
                covariance_after=covariance_before,
                direction=direction,
                H=H,
                R=R,
                q=q,
                p=p,
                S=S,
            )
            self.last_nis = float(nis)
            self.last_relative_range_update = event
            self._record_update("uwb", False, nis)
            self.rejected_observation_count += 1
            return False

        direction = difference / distance
        H = np.zeros((1, STATE_DIM), dtype=float)
        H[0, POSITION_SLICE] = direction
        p = float((H @ covariance_before @ H.T).item())
        if reference_covariance is None:
            q = 0.0
        elif reference_covariance.shape == (3, 3):
            q = float(direction @ reference_covariance @ direction.T)
        else:
            q = float(direction @ reference_covariance[:3, :3] @ direction.T)
        if q < -PSD_TOL:
            raise ValueError("neighbor direction variance must be non-negative")
        q = max(0.0, q)
        cross_bound = 2.0 * np.sqrt(max(0.0, p) * q)
        S = float(p + q + cross_bound + R)
        innovation = float(measured_range - distance - nlos_bias)
        nis = float(innovation * innovation / S)
        if not np.isfinite(nis):
            raise FloatingPointError("range innovation NIS became non-finite")

        # The conservative unknown-cross-covariance gate remains the native
        # NIS threshold 9. This is an innovation bound, not a calibrated
        # nine-sigma or chi-square claim. Rejection leaves state/P unchanged.
        gate_threshold = float(self.config.innovation_gate_threshold)
        if nis > gate_threshold:
            event = self._range_event(
                accepted=False,
                reason="innovation_gate",
                measured_range=measured_range,
                distance=distance,
                nlos_bias=nlos_bias,
                innovation=innovation,
                nis=nis,
                weight=1.0,
                gain=np.zeros((STATE_DIM, 1)),
                correction=np.zeros(STATE_DIM),
                state_before=state_before,
                state_after=state_before,
                covariance_before=covariance_before,
                covariance_after=covariance_before,
                direction=direction,
                H=H,
                R=R,
                q=q,
                p=p,
                S=S,
            )
            self.last_nis = nis
            self.last_relative_range_update = event
            self._record_update("uwb", False, nis)
            self.rejected_observation_count += 1
            return False

        prior_precision = np.linalg.inv(covariance_before)
        pseudo_variance = float(q + R)
        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for weight in _CI_WEIGHTS:
            precision = weight * prior_precision + (1.0 - weight) * (H.T @ H) / pseudo_variance
            posterior = _symmetric(np.linalg.inv(precision))
            score = float(np.trace(posterior[POSITION_SLICE, POSITION_SLICE]))
            gain = posterior @ H.T * ((1.0 - weight) / pseudo_variance)
            if best is None:
                best = (score, float(weight), posterior, gain)
                continue
            best_score, best_weight = best[0], best[1]
            tie = np.isclose(score, best_score, rtol=1.0e-12, atol=1.0e-12)
            if score < best_score - 1.0e-12 or (tie and float(weight) > best_weight):
                best = (score, float(weight), posterior, gain)
        assert best is not None
        _, weight, posterior, gain = best
        posterior = _symmetric(posterior)
        correction = (gain[:, 0] * innovation).reshape(STATE_DIM)
        state_after = state_before + correction

        # w=1 is retained in the prescribed grid.  It is a real attempted
        # measurement but carries no information from the pseudo-observation.
        if weight >= 1.0 - 1.0e-15 or np.linalg.norm(gain) <= 1.0e-15:
            event = self._range_event(
                accepted=False,
                reason="no_information_gain",
                measured_range=measured_range,
                distance=distance,
                nlos_bias=nlos_bias,
                innovation=innovation,
                nis=nis,
                weight=weight,
                gain=np.zeros((STATE_DIM, 1)),
                correction=np.zeros(STATE_DIM),
                state_before=state_before,
                state_after=state_before,
                covariance_before=covariance_before,
                covariance_after=covariance_before,
                direction=direction,
                H=H,
                R=R,
                q=q,
                p=p,
                S=S,
            )
            self.last_nis = nis
            self.last_relative_range_update = event
            self._record_update("uwb", False, nis)
            self.rejected_observation_count += 1
            return False

        minimum_posterior = float(np.min(np.linalg.eigvalsh(posterior)))
        if minimum_posterior < -PSD_TOL:
            raise FloatingPointError("CI posterior covariance is not PSD")
        if not np.isfinite(posterior).all() or not np.isfinite(state_after).all():
            raise FloatingPointError("CI range update became non-finite")
        conditional_bound = conditional_linear_error_bound(
            covariance_before, H, weight, gain, pseudo_variance
        )
        conditional_residual = float(np.linalg.norm(conditional_bound - posterior))
        event = self._range_event(
            accepted=True,
            reason="none",
            measured_range=measured_range,
            distance=distance,
            nlos_bias=nlos_bias,
            innovation=innovation,
            nis=nis,
            weight=weight,
            gain=gain,
            correction=correction,
            state_before=state_before,
            state_after=state_after,
            covariance_before=covariance_before,
            covariance_after=posterior,
            direction=direction,
            H=H,
            R=R,
            q=q,
            p=p,
            S=S,
            conditional_bound=conditional_bound,
        )
        event["conditional_bound_residual_norm"] = conditional_residual
        event["conditional_bound_matches_posterior"] = bool(
            np.allclose(conditional_bound, posterior, rtol=2.0e-8, atol=2.0e-10)
        )

        # Commit only after all calculations and diagnostics are finite.  No
        # covariance floor or post-update rescaling is applied here.
        self.state = state_after
        self.covariance = posterior
        self.last_nis = nis
        self.last_relative_range_update = event
        self._record_update("uwb", True, nis)
        self.valid_observation_count += 1
        self.valid_neighbor_count += 1
        self.last_neighbor_timestamp = timestamp_value
        return True


class PartialGNSSJointEKF(JointEKF):
    """v6 joint estimator with GNSS allowed for uav_00 and uav_01 only."""

    @staticmethod
    def _validate_gnss_packets(
        sensor_packets: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not isinstance(sensor_packets, Mapping):
            raise TypeError("sensor_packets must be a mapping")
        valid: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for uid, raw in sensor_packets.items():
            uid = str(uid)
            if uid not in UIDS:
                raise ValueError(f"unknown uid {uid!r}; expected one of {UIDS}")
            if not isinstance(raw, Mapping):
                raise TypeError(f"sensor packet for {uid!r} must be a mapping")
            gnss = raw.get("gnss", {"valid": False})
            if not isinstance(gnss, Mapping):
                raise TypeError(f"{uid!r} gnss packet must be a mapping")
            if not bool(gnss.get("valid", False)):
                continue
            if "position" not in gnss or "std" not in gnss:
                raise ValueError(f"valid {uid!r} GNSS packet requires position and std")
            position = _finite_array(gnss["position"], (3,), f"{uid} GNSS position")
            std_array = np.asarray(gnss["std"], dtype=float)
            if std_array.ndim == 0:
                std_array = np.repeat(float(std_array), 3)
            std = _finite_array(std_array, (3,), f"{uid} GNSS std")
            if np.any(std <= 0.0):
                raise ValueError("GNSS standard deviations must be positive")
            valid[uid] = (position, std)
        return valid

    def predict_and_local_updates(
        self,
        sensor_packets: Mapping[str, Mapping[str, Any]],
        dt_s: float,
        anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        valid_gnss = self._validate_gnss_packets(sensor_packets)
        if "uav_02" in valid_gnss:
            raise ValueError("GNSS for uav_02 is denied")

        # The frozen v6 sanitizer intentionally admits only uav_00 GNSS.  Run
        # the complete prediction/VO path with all GNSS marked invalid, then
        # apply the two permitted local GNSS observations through its existing
        # event/counter path.
        packets = copy.deepcopy(sensor_packets)
        for uid in UIDS:
            if uid in packets and isinstance(packets[uid], Mapping):
                packet = dict(packets[uid])
                packet["gnss"] = {"valid": False}
                packets[uid] = packet
        super().predict_and_local_updates(packets, float(dt_s), anchor=None)
        for uid in ("uav_00", "uav_01"):
            if uid not in valid_gnss:
                continue
            position, std = valid_gnss[uid]
            standard_deviation = np.repeat(float(std[0]), 3) if np.all(std == std[0]) else std
            self._update_position_observation(uid, position, standard_deviation, "gnss")
        return self.diagnostics_snapshot()


__all__ = [
    "MeasurementCIEKF",
    "PartialGNSSJointEKF",
    "conditional_linear_error_bound",
]
