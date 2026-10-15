"""Small, truth-free cooperative estimator references used by the v6 audit.

This module intentionally contains estimators rather than an environment or a
training runner.  :class:`YoungBoundEKF` is a conservative wrapper around the
local filter, while :class:`JointEKF` keeps the covariance of three local
filters in one 45-dimensional state.  The latter is useful for checking the
difference between a genuine current joint update and a distributed, stale
neighbor envelope.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from src.simulation.ekf import (
    ATTITUDE_SLICE,
    EKFConfig,
    LocalEKF,
    POSITION_SLICE,
    STATE_DIM,
    VELOCITY_SLICE,
)
from .cooperative_ekf_reliability_reference_v5 import fixed_gain_covariance_bound


UIDS = ("uav_00", "uav_01", "uav_02")
JOINT_DIM = len(UIDS) * STATE_DIM
POSITION_DIM = len(UIDS) * 3
PSD_TOL = 1.0e-8

# Explicit alias used by the v6 audit text.  The implementation is the locked
# v5 helper, so there is one source of the Young inequality calculation.
v5fixed_gain_covariance_bound = fixed_gain_covariance_bound
fixed_gain_covariance_bound_v5 = fixed_gain_covariance_bound


def _sym(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return 0.5 * (value + value.T)


def _finite_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _validated_psd(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    result = _finite_matrix(value, shape, name)
    result = _sym(result)
    minimum = float(np.min(np.linalg.eigvalsh(result)))
    if minimum < -PSD_TOL:
        raise ValueError(f"{name} must be positive semidefinite")
    if minimum < 0.0:
        result = result + np.eye(shape[0]) * (-minimum)
    return _sym(result)


def _stabilize_full(value: Any) -> np.ndarray:
    result = _sym(np.asarray(value, dtype=float))
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(result).all():
        raise FloatingPointError("covariance became non-finite")
    minimum = float(np.min(np.linalg.eigvalsh(result)))
    if minimum < -PSD_TOL:
        raise FloatingPointError(f"covariance is not PSD (minimum eigenvalue {minimum:g})")
    if minimum < 0.0:
        result = result + np.eye(result.shape[0]) * (-minimum + 1.0e-12)
    return _sym(result)


def _safe_fraction(before: float, after: float) -> float:
    return float((before - after) / before) if abs(before) > 1.0e-15 else 0.0


def _logdet_gain(before: np.ndarray, after: np.ndarray) -> float:
    """Return ``log det(before) - log det(after)`` with PSD rank handling."""

    b = _sym(np.asarray(before, dtype=float))
    a = _sym(np.asarray(after, dtype=float))
    eb = np.linalg.eigvalsh(b)
    ea = np.linalg.eigvalsh(a)
    # A rank-zero or singular covariance carries no finite ordinary log-det
    # information.  Do not manufacture a value by flooring eigenvalues.
    if eb.size == 0 or ea.size == 0:
        return 0.0
    if float(np.min(eb)) <= 1.0e-15 or float(np.min(ea)) <= 1.0e-15:
        return 0.0
    sign_b, log_b = np.linalg.slogdet(b)
    sign_a, log_a = np.linalg.slogdet(a)
    if sign_b <= 0.0 or sign_a <= 0.0 or not np.isfinite(log_b) or not np.isfinite(log_a):
        return 0.0
    return float(log_b - log_a)


def _position_selector(uid_index: int, dimension: int = JOINT_DIM) -> np.ndarray:
    selector = np.zeros((3, dimension), dtype=float)
    start = uid_index * STATE_DIM
    selector[:, start : start + 3] = np.eye(3)
    return selector


def _clone_selector() -> np.ndarray:
    selector = np.zeros((POSITION_DIM, JOINT_DIM), dtype=float)
    for i in range(len(UIDS)):
        selector[i * 3 : i * 3 + 3, i * STATE_DIM : i * STATE_DIM + 3] = np.eye(3)
    return selector


class YoungBoundEKF(LocalEKF):
    """Local EKF using the fixed-gain Young covariance upper envelope.

    The local filter still computes the online range innovation, gate, gain,
    and mean.  The envelope is applied only after an accepted range packet
    that includes neighbor covariance; this keeps the deployed local rule
    unchanged when that packet is unavailable.
    """

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
        prior_covariance = np.asarray(self.covariance, dtype=float).copy()
        prior_covariance = _validated_psd(prior_covariance, (STATE_DIM, STATE_DIM), "local covariance")
        # Validate the packet before invoking the native update.  The native
        # method changes the mean/counters before a subclass can inspect its
        # covariance, so delayed validation would leave a partially mutated
        # filter after a malformed neighbor packet.
        neighbor_covariance_array = np.asarray(neighbor_covariance, dtype=float) if neighbor_covariance is not None else None
        validated_neighbor_position: np.ndarray | None = None
        if neighbor_covariance_array is not None:
            if neighbor_covariance_array.shape == (3, 3):
                validated_neighbor_position = _validated_psd(
                    neighbor_covariance_array, (3, 3), "neighbor position covariance"
                )
            elif neighbor_covariance_array.shape == (STATE_DIM, STATE_DIM):
                neighbor_covariance_array = _validated_psd(
                    neighbor_covariance_array, (STATE_DIM, STATE_DIM), "neighbor covariance"
                )
                validated_neighbor_position = neighbor_covariance_array[:3, :3]
            else:
                raise ValueError("neighbor_covariance must be 3x3 or 15x15")
        accepted = super().update_relative_range(
            measured_range_m,
            neighbor_position,
            neighbor_covariance,
            measurement_variance=measurement_variance,
            nlos_bias_m=nlos_bias_m,
            timestamp=timestamp,
        )
        # A missing neighbor covariance deliberately follows the native local
        # behavior exactly, including its diagnostic schema.
        if not accepted or neighbor_covariance is None:
            return accepted

        record = dict(self.last_relative_range_update)
        assert neighbor_covariance_array is not None
        if neighbor_covariance_array.shape == (3, 3):
            neighbor_position_covariance = validated_neighbor_position
        elif neighbor_covariance_array.shape == (STATE_DIM, STATE_DIM):
            neighbor_position_covariance = validated_neighbor_position
        else:
            raise ValueError("neighbor_covariance must be 3x3 or 15x15")
        assert neighbor_position_covariance is not None
        if {"direction_x", "direction_y", "direction_z"}.issubset(record):
            direction = np.asarray(
                [record["direction_x"], record["direction_y"], record["direction_z"]], dtype=float
            )
        else:
            direction = np.asarray(neighbor_position, dtype=float).reshape(3)
            difference = self.position - direction
            norm = float(np.linalg.norm(difference))
            if norm < 1.0e-8:
                return accepted
            direction = difference / norm
        if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1.0e-8:
            raise FloatingPointError("range direction became invalid")
        direction = direction / np.linalg.norm(direction)
        jacobian = np.zeros((1, STATE_DIM), dtype=float)
        jacobian[0, POSITION_SLICE] = direction
        native_innovation_variance = float(record.get("innovation_variance_m2", np.nan))
        if not np.isfinite(native_innovation_variance) or native_innovation_variance <= 0.0:
            raise FloatingPointError("native range innovation variance is invalid")
        gain = prior_covariance @ jacobian.T / native_innovation_variance
        q = float(direction @ neighbor_position_covariance @ direction.T)
        range_variance = float(
            self.config.uwb_range_std_m**2
            if measurement_variance is None
            else measurement_variance
        )
        if not np.isfinite(range_variance) or range_variance < 0.0:
            raise ValueError("measurement_variance must be finite and non-negative")
        envelope = fixed_gain_covariance_bound(
            prior_covariance,
            jacobian,
            gain,
            q,
            range_variance,
        )
        bound = _validated_psd(
            envelope["bound_matrix"], (STATE_DIM, STATE_DIM), "Young covariance bound"
        )
        self.covariance = bound
        self._stabilize_covariance()
        final_trace = float(np.trace(self.covariance[POSITION_SLICE, POSITION_SLICE]))
        record.update(
            {
                "position_covariance_trace_after_final_m2": final_trace,
                "covariance_restored": False,
                "young_weight": float(envelope["weight"]),
                "young_bound_weight": float(envelope["weight"]),
                "young_bound_method": "v5_fixed_gain_covariance_bound",
                "young_weight_method": "trace_optimal_young_v5",
                "weight_method": "trace_optimal_young_v5",
                "young_bound_trace_m2": float(np.trace(bound)),
                "young_A_trace": float(envelope["trace_A"]),
                "young_B_trace": float(envelope["trace_B"]),
                "H": jacobian.tolist(),
                "K": gain.tolist(),
                "neighbor_direction_variance_m2": q,
                "range_variance_m2": range_variance,
            }
        )
        self.last_relative_range_update = record
        return accepted


class JointEKF:
    """A three-agent current-time EKF with explicit cross-agent covariance."""

    def __init__(
        self,
        initial_states: Mapping[str, Iterable[float]],
        ekf_config: EKFConfig | None = None,
        initial_cross_position: float = 2.0,
    ) -> None:
        if not isinstance(initial_states, Mapping):
            raise TypeError("initial_states must be a mapping from uid to 15-state vectors")
        unknown = set(initial_states) - set(UIDS)
        missing = set(UIDS) - set(initial_states)
        if unknown or missing:
            raise ValueError(f"initial_states must contain exactly UIDS={UIDS}")
        if not np.isfinite(initial_cross_position):
            raise ValueError("initial_cross_position must be finite")
        if ekf_config is None:
            self.config = EKFConfig()
        elif isinstance(ekf_config, EKFConfig):
            self.config = ekf_config
        elif isinstance(ekf_config, Mapping):
            self.config = EKFConfig(**dict(ekf_config))
        else:
            # A few audit runners keep the simulation module under the short
            # name ``ekf``; accepting that module preserves the constructor's
            # compact positional API without coupling this reference to an
            # environment object.
            self.config = (
                ekf_config.EKFConfig()
                if hasattr(ekf_config, "EKFConfig") and not hasattr(ekf_config, "innovation_gate_threshold")
                else ekf_config
            )
        self._uid_index = {uid: i for i, uid in enumerate(UIDS)}
        self.state = np.concatenate(
            [_finite_matrix(initial_states[uid], (STATE_DIM,), f"initial state {uid}") for uid in UIDS]
        )
        self.covariance = np.zeros((JOINT_DIM, JOINT_DIM), dtype=float)
        for i, uid in enumerate(UIDS):
            local = LocalEKF(initial_states[uid], self.config)
            start = i * STATE_DIM
            self.covariance[start : start + STATE_DIM, start : start + STATE_DIM] = local.covariance
        cross = float(initial_cross_position) * np.eye(3)
        for i in range(len(UIDS)):
            for j in range(i + 1, len(UIDS)):
                i0, j0 = i * STATE_DIM, j * STATE_DIM
                self.covariance[i0 : i0 + 3, j0 : j0 + 3] = cross
                self.covariance[j0 : j0 + 3, i0 : i0 + 3] = cross
        self.covariance = _stabilize_full(self.covariance)
        self._consumed_measurement_ids: set[str] = set()
        self.step_diagnostics: dict[str, list[dict[str, Any]]] = {}
        self.events: list[dict[str, Any]] = []
        self.vo_update_events: list[dict[str, Any]] = []
        self.gnss_update_events: list[dict[str, Any]] = []
        self.range_update_events: list[dict[str, Any]] = []
        self._last_augmented_covariance: np.ndarray | None = None
        self.begin_step()

    @property
    def mean(self) -> np.ndarray:
        return self.state

    @mean.setter
    def mean(self, value: Any) -> None:
        self.state = _finite_matrix(value, (JOINT_DIM,), "joint mean")

    @property
    def state_vector(self) -> np.ndarray:
        return self.state

    @state_vector.setter
    def state_vector(self, value: Any) -> None:
        self.mean = value

    @property
    def P(self) -> np.ndarray:
        return self.covariance

    @property
    def x(self) -> np.ndarray:
        return self.state

    @property
    def matrix_storage_bytes(self) -> int:
        """Bytes for the full covariance and, when retained, the 54-state VO matrix."""

        total = int(np.asarray(self.covariance).nbytes)
        if self._last_augmented_covariance is not None:
            total += int(self._last_augmented_covariance.nbytes)
        return total

    def storage_bytes(self) -> int:
        return self.matrix_storage_bytes

    def _index(self, uid: str) -> int:
        try:
            return self._uid_index[str(uid)]
        except KeyError as exc:
            raise ValueError(f"unknown uid {uid!r}; expected one of {UIDS}") from exc

    def local_state(self, uid: str) -> np.ndarray:
        start = self._index(uid) * STATE_DIM
        return self.state[start : start + STATE_DIM].copy()

    def local_covariance(self, uid: str) -> np.ndarray:
        start = self._index(uid) * STATE_DIM
        return self.covariance[start : start + STATE_DIM, start : start + STATE_DIM].copy()

    def position_covariance_9(self) -> np.ndarray:
        selector = _clone_selector()
        return _sym(selector @ self.covariance @ selector.T)

    def relative_covariance(self, uid: str, source: str) -> np.ndarray:
        own = _position_selector(self._index(uid))
        peer = _position_selector(self._index(source))
        h = own - peer
        return _sym(h @ self.covariance @ h.T)

    def begin_step(self) -> None:
        self.step_diagnostics = {"vo": [], "gnss": [], "range": []}
        self.events = []
        self.vo_update_events = self.step_diagnostics["vo"]
        self.gnss_update_events = self.step_diagnostics["gnss"]
        self.range_update_events = self.step_diagnostics["range"]

    def start_step(self) -> None:
        """Public spelling for resetting the per-step diagnostic lists."""

        self.begin_step()

    def _linear_update(
        self,
        measurement: np.ndarray,
        expected: np.ndarray,
        jacobian: np.ndarray,
        noise: np.ndarray,
        *,
        sensor: str,
    ) -> tuple[bool, dict[str, Any]]:
        z = np.asarray(measurement, dtype=float).reshape(-1)
        mu = np.asarray(expected, dtype=float).reshape(-1)
        H = np.asarray(jacobian, dtype=float)
        R = np.asarray(noise, dtype=float)
        if H.shape != (z.size, JOINT_DIM):
            raise ValueError(f"jacobian must have shape {(z.size, JOINT_DIM)}, got {H.shape}")
        if R.shape != (z.size, z.size):
            raise ValueError(f"noise must have shape {(z.size, z.size)}, got {R.shape}")
        if not np.isfinite(z).all() or not np.isfinite(mu).all() or not np.isfinite(H).all():
            raise ValueError("linear observation values must be finite")
        if not np.isfinite(self.state).all() or not np.isfinite(self.covariance).all():
            raise FloatingPointError("joint mean or covariance became non-finite")
        R = _validated_psd(R, R.shape, "measurement covariance")
        innovation = z - mu
        prior_state = self.state.copy()
        prior_covariance = self.covariance.copy()
        S = _sym(H @ prior_covariance @ H.T + R)
        minimum = float(np.min(np.linalg.eigvalsh(S)))
        if minimum < -PSD_TOL:
            raise FloatingPointError("innovation covariance is not PSD")
        if minimum < 1.0e-12:
            S = S + np.eye(z.size) * (1.0e-12 - minimum)
        Sinv = np.linalg.pinv(S)
        nis = float(innovation @ Sinv @ innovation.T)
        if not np.isfinite(nis):
            raise FloatingPointError("innovation NIS became non-finite")
        K = prior_covariance @ H.T @ Sinv
        accepted = nis <= float(self.config.innovation_gate_threshold)
        if accepted:
            self.state = prior_state + K @ innovation
            identity = np.eye(JOINT_DIM)
            self.covariance = _stabilize_full(
                (identity - K @ H) @ prior_covariance @ (identity - K @ H).T + K @ R @ K.T
            )
        else:
            self.state = prior_state
            self.covariance = prior_covariance
        event = {
            "sensor": sensor,
            "accepted": bool(accepted),
            "nis": float(nis),
            "innovation": innovation.tolist(),
            "innovation_covariance": S.tolist(),
            "H": H.tolist(),
            "gain_norm": float(np.linalg.norm(K)) if accepted else 0.0,
            "prior_covariance_trace": float(np.trace(prior_covariance)),
            "posterior_covariance_trace": float(np.trace(self.covariance)),
            "prior_position_covariance_trace": float(np.trace(_clone_selector() @ prior_covariance @ _clone_selector().T)),
            "posterior_position_covariance_trace": float(np.trace(_clone_selector() @ self.covariance @ _clone_selector().T)),
        }
        event["P_trace"] = event["posterior_covariance_trace"]
        event["prior_trace"] = event["prior_covariance_trace"]
        event["post_trace"] = event["posterior_covariance_trace"]
        event["full_position_trace"] = event["posterior_position_covariance_trace"]
        event["prior_full_position_covariance_trace"] = event["prior_position_covariance_trace"]
        event["posterior_full_position_covariance_trace"] = event["posterior_position_covariance_trace"]
        return accepted, event

    def _sanitize_packets(self, sensor_packets: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        if not isinstance(sensor_packets, Mapping):
            raise TypeError("sensor_packets must be a mapping")
        packets: dict[str, dict[str, Any]] = {}
        for uid, raw in sensor_packets.items():
            self._index(uid)
            if not isinstance(raw, Mapping):
                raise TypeError(f"sensor packet for {uid!r} must be a mapping")
            packet = dict(raw)
            if "imu" not in packet:
                raise ValueError(f"sensor packet {uid!r} requires imu")
            imu = packet["imu"]
            if not isinstance(imu, Mapping) or "accel" not in imu or "gyro" not in imu:
                raise ValueError(f"sensor packet {uid!r} requires imu.accel and imu.gyro")
            accel = _finite_matrix(imu["accel"], (3,), f"{uid} imu accel")
            gyro = _finite_matrix(imu["gyro"], (3,), f"{uid} imu gyro")
            packet["imu"] = {"accel": accel, "gyro": gyro}
            vo = packet.get("vo", {"valid": False})
            if not isinstance(vo, Mapping):
                raise TypeError(f"{uid!r} vo packet must be a mapping")
            vo = dict(vo)
            valid_vo = bool(vo.get("valid", False))
            if valid_vo:
                if "delta_position" not in vo:
                    raise ValueError(f"valid {uid!r} VO packet requires delta_position")
                vo["delta_position"] = _finite_matrix(vo["delta_position"], (3,), f"{uid} VO delta_position")
                if vo.get("covariance") is not None:
                    vo["covariance"] = _validated_psd(vo["covariance"], (3, 3), f"{uid} VO covariance")
            vo["valid"] = valid_vo
            packet["vo"] = vo
            gnss = packet.get("gnss", {"valid": False})
            if not isinstance(gnss, Mapping):
                raise TypeError(f"{uid!r} gnss packet must be a mapping")
            gnss = dict(gnss)
            valid_gnss = bool(gnss.get("valid", False))
            if valid_gnss:
                if str(uid) != "uav_00":
                    raise ValueError("centralized all-local GNSS is denied; only uav_00 may carry GNSS")
                if "position" not in gnss or "std" not in gnss:
                    raise ValueError(f"valid {uid!r} GNSS packet requires position and std")
                gnss["position"] = _finite_matrix(gnss["position"], (3,), f"{uid} GNSS position")
                std = np.asarray(gnss["std"], dtype=float)
                if std.ndim == 0:
                    std = np.repeat(float(std), 3)
                std = _finite_matrix(std, (3,), f"{uid} GNSS std")
                if np.any(std <= 0.0):
                    raise ValueError("GNSS standard deviations must be positive")
                gnss["std"] = std
            gnss["valid"] = valid_gnss
            packet["gnss"] = gnss
            packets[str(uid)] = packet
        if set(packets) != set(UIDS):
            raise ValueError(f"sensor_packets must contain exactly UIDS={UIDS}")
        return packets

    def predict_and_local_updates(
        self,
        sensor_packets: Mapping[str, Mapping[str, Any]],
        dt_s: float,
        anchor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict all local blocks and apply temporal VO, GNSS, and anchor data."""

        if not np.isfinite(dt_s) or float(dt_s) <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        packets = self._sanitize_packets(sensor_packets)
        self.begin_step()
        prior_state = self.state.copy()
        prior_covariance = self.covariance.copy()
        F = np.zeros((JOINT_DIM, JOINT_DIM), dtype=float)
        Q = np.zeros((JOINT_DIM, JOINT_DIM), dtype=float)
        predicted_state = np.zeros(JOINT_DIM, dtype=float)
        for i, uid in enumerate(UIDS):
            packet = packets.get(uid, {"imu": {"accel": np.zeros(3), "gyro": np.zeros(3)}, "vo": {"valid": False}, "gnss": {"valid": False}})
            local = LocalEKF(prior_state[i * STATE_DIM : (i + 1) * STATE_DIM], self.config)
            local.covariance = prior_covariance[i * STATE_DIM : (i + 1) * STATE_DIM, i * STATE_DIM : (i + 1) * STATE_DIM].copy()
            local.predict(packet["imu"], float(dt_s))
            predicted_state[i * STATE_DIM : (i + 1) * STATE_DIM] = local.state
            F[i * STATE_DIM : (i + 1) * STATE_DIM, i * STATE_DIM : (i + 1) * STATE_DIM] = local.last_prediction_context["transition_matrix"]
            local_prior_covariance = prior_covariance[i * STATE_DIM : (i + 1) * STATE_DIM, i * STATE_DIM : (i + 1) * STATE_DIM]
            q_block = local.covariance - local.last_prediction_context["transition_matrix"] @ local_prior_covariance @ local.last_prediction_context["transition_matrix"].T
            Q[i * STATE_DIM : (i + 1) * STATE_DIM, i * STATE_DIM : (i + 1) * STATE_DIM] = _sym(q_block)
        predicted_covariance = _stabilize_full(F @ prior_covariance @ F.T + Q)
        clone_selector = _clone_selector()
        augmented_state = np.concatenate((predicted_state, clone_selector @ prior_state))
        cross_current_past = F @ prior_covariance @ clone_selector.T
        augmented_covariance = _stabilize_full(
            np.block(
                [
                    [predicted_covariance, cross_current_past],
                    [cross_current_past.T, clone_selector @ prior_covariance @ clone_selector.T],
                ]
            )
        )
        # Keep one 54-state matrix for memory accounting and diagnostics; no
        # per-agent duplicate clone covariance is allocated.
        self._last_augmented_covariance = augmented_covariance.copy()

        for i, uid in enumerate(UIDS):
            packet = packets.get(uid)
            if packet is None or not packet["vo"].get("valid", False):
                continue
            H = np.zeros((3, JOINT_DIM + POSITION_DIM), dtype=float)
            H[:, i * STATE_DIM : i * STATE_DIM + 3] = np.eye(3)
            H[:, JOINT_DIM + i * 3 : JOINT_DIM + i * 3 + 3] = -np.eye(3)
            vo_covariance = packet["vo"].get("covariance")
            R = np.asarray(
                vo_covariance if vo_covariance is not None else np.eye(3) * self.config.vo_position_std_m**2,
                dtype=float,
            )
            measurement = packet["vo"]["delta_position"]
            expected = H @ augmented_state
            innovation = measurement - expected
            S = _sym(H @ augmented_covariance @ H.T + R)
            minimum = float(np.min(np.linalg.eigvalsh(S)))
            if minimum < 1.0e-12:
                S = S + np.eye(3) * (1.0e-12 - minimum)
            K = augmented_covariance @ H.T @ np.linalg.pinv(S)
            nis = float(innovation @ np.linalg.pinv(S) @ innovation.T)
            if not np.isfinite(nis):
                raise FloatingPointError("VO innovation NIS became non-finite")
            prior_aug_covariance = augmented_covariance.copy()
            accepted = nis <= float(self.config.innovation_gate_threshold)
            if accepted:
                augmented_state = augmented_state + K @ innovation
                identity = np.eye(JOINT_DIM + POSITION_DIM)
                augmented_covariance = _stabilize_full(
                    (identity - K @ H) @ augmented_covariance @ (identity - K @ H).T + K @ R @ K.T
                )
            vo_event = {
                    "sensor": "vo",
                    "uid": uid,
                    "accepted": bool(accepted),
                    "nis": nis,
                    "H": H.tolist(),
                    "gain_norm": float(np.linalg.norm(K)) if accepted else 0.0,
                    "prior_covariance_trace": float(np.trace(prior_aug_covariance[:JOINT_DIM, :JOINT_DIM])),
                    "posterior_covariance_trace": float(np.trace(augmented_covariance[:JOINT_DIM, :JOINT_DIM])),
                    "prior_position_covariance_trace": float(np.trace(clone_selector @ prior_aug_covariance[:JOINT_DIM, :JOINT_DIM] @ clone_selector.T)),
                    "posterior_position_covariance_trace": float(np.trace(clone_selector @ augmented_covariance[:JOINT_DIM, :JOINT_DIM] @ clone_selector.T)),
                    "innovation": innovation.tolist(),
                    "innovation_covariance": S.tolist(),
                }
            vo_event["P_trace"] = vo_event["posterior_covariance_trace"]
            vo_event["prior_trace"] = vo_event["prior_covariance_trace"]
            vo_event["post_trace"] = vo_event["posterior_covariance_trace"]
            vo_event["full_position_trace"] = vo_event["posterior_position_covariance_trace"]
            vo_event["prior_full_position_covariance_trace"] = vo_event["prior_position_covariance_trace"]
            vo_event["posterior_full_position_covariance_trace"] = vo_event["posterior_position_covariance_trace"]
            self.vo_update_events.append(vo_event)
            self.events.append(vo_event)
        self.state = augmented_state[:JOINT_DIM].copy()
        self.covariance = _stabilize_full(augmented_covariance[:JOINT_DIM, :JOINT_DIM])

        for uid, packet in packets.items():
            gnss = packet["gnss"]
            if not gnss.get("valid", False):
                continue
            if uid != "uav_00":
                raise ValueError("centralized all-local GNSS is denied; only uav_00 may carry GNSS")
            self._update_position_observation(uid, gnss["position"], gnss["std"], "gnss")
        if anchor is not None:
            if not isinstance(anchor, Mapping):
                raise TypeError("anchor must be a mapping")
            if bool(anchor.get("valid", True)):
                anchor_uid = str(anchor.get("uav_id", "uav_00"))
                if anchor_uid != "uav_00":
                    raise ValueError("anchor noisy observation is restricted to uav_00")
                if "position" not in anchor or "std" not in anchor:
                    raise ValueError("valid anchor requires position and std")
                position = _finite_matrix(anchor["position"], (3,), "anchor position")
                std = np.asarray(anchor["std"], dtype=float)
                if std.ndim == 0:
                    std = np.repeat(float(std), 3)
                std = _finite_matrix(std, (3,), "anchor std")
                if np.any(std <= 0.0):
                    raise ValueError("anchor standard deviations must be positive")
                self._update_position_observation(anchor_uid, position, std, "anchor")
        self._last_augmented_covariance = augmented_covariance.copy()
        return self.diagnostics_snapshot()

    def _update_position_observation(self, uid: str, position: Any, std: Any, sensor: str) -> bool:
        i = self._index(uid)
        z = _finite_matrix(position, (3,), f"{sensor} position")
        standard_deviation = _finite_matrix(std, (3,), f"{sensor} std")
        if np.any(standard_deviation <= 0.0):
            raise ValueError(f"{sensor} standard deviations must be positive")
        H = _position_selector(i)
        accepted, event = self._linear_update(
            z,
            H @ self.state,
            H,
            np.diag(standard_deviation**2),
            sensor=sensor,
        )
        event["uid"] = uid
        self.gnss_update_events.append(event)
        self.events.append(event)
        return accepted

    def update_range(
        self,
        target: str,
        source: str,
        measurement: float,
        variance: float,
        nlos_bias: float = 0.0,
        measurement_id: str | None = None,
    ) -> bool:
        target_index = self._index(target)
        source_index = self._index(source)
        if target_index == source_index:
            raise ValueError("target and source must be different")
        if measurement_id is not None:
            key = str(measurement_id)
            if key in self._consumed_measurement_ids:
                raise ValueError(f"duplicate measurement_id {key!r}")
            self._consumed_measurement_ids.add(key)
        measurement_value = float(measurement)
        variance_value = float(variance)
        nlos_bias_value = float(nlos_bias)
        if not np.isfinite(measurement_value) or not np.isfinite(variance_value) or not np.isfinite(nlos_bias_value):
            raise ValueError("range measurement, variance, and nlos_bias must be finite")
        if measurement_value < 0.0 or variance_value < 0.0:
            raise ValueError("range measurement and variance must be non-negative")
        target_position = self.local_state(target)[POSITION_SLICE]
        source_position = self.local_state(source)[POSITION_SLICE]
        difference = target_position - source_position
        distance = float(np.linalg.norm(difference))
        if not np.isfinite(distance):
            raise FloatingPointError("predicted range became non-finite")
        H = _position_selector(target_index) - _position_selector(source_index)
        prior_covariance = self.covariance.copy()
        if distance < 1.0e-12:
            event = {
                "sensor": "range",
                "target": target,
                "source": source,
                "measurement_id": measurement_id,
                "accepted": False,
                "nis": -1.0,
                "rejection_reason": "degenerate_predicted_range",
                "H": H.tolist(),
                "relative_H": H.tolist(),
                "gain_norm": 0.0,
                "prior_covariance_trace": float(np.trace(prior_covariance)),
                "posterior_covariance_trace": float(np.trace(prior_covariance)),
            }
            event["prior_trace"] = event["prior_covariance_trace"]
            event["post_trace"] = event["posterior_covariance_trace"]
            self.range_update_events.append(event)
            self.events.append(event)
            return False
        direction = difference / distance
        scalar_H = direction.reshape(1, 3) @ H.reshape(3, JOINT_DIM)
        expected = np.array([distance + nlos_bias_value], dtype=float)
        accepted, event = self._linear_update(
            np.array([measurement_value]),
            expected,
            scalar_H,
            np.array([[variance_value]], dtype=float),
            sensor="range",
        )
        event.update(
            {
                "target": target,
                "source": source,
                "measurement_id": measurement_id,
                "predicted_range_m": distance,
                "expected_measurement_m": float(distance + nlos_bias_value),
                "direction": direction.tolist(),
                "relative_H": H.tolist(),
                "rejection_reason": "none" if accepted else "innovation_gate",
            }
        )
        self.range_update_events.append(event)
        self.events.append(event)
        return accepted

    def measurement_information(self, uid: str, source: str, R: float | np.ndarray) -> dict[str, Any]:
        """Return current, cross-aware scalar and relative-vector information."""

        own_index = self._index(uid)
        source_index = self._index(source)
        if own_index == source_index:
            raise ValueError("uid and source must be different")
        measurement_noise = np.asarray(R, dtype=float)
        scalar_noise_from_matrix: np.ndarray | None = None
        if measurement_noise.ndim == 0:
            scalar_noise = float(measurement_noise)
        elif measurement_noise.shape == (1, 1):
            scalar_noise = float(measurement_noise[0, 0])
        elif measurement_noise.shape == (3, 3):
            # A matrix R is accepted only as a noise covariance to be
            # projected onto the *same scalar range direction* below.  It is
            # never treated as a fictitious three-axis relative-position
            # observation.
            scalar_noise_from_matrix = _validated_psd(measurement_noise, (3, 3), "R")
            scalar_noise = float(np.max(np.diag(scalar_noise_from_matrix)))
        else:
            raise ValueError("R must be scalar, 1x1, or 3x3")
        if not np.isfinite(scalar_noise) or scalar_noise < 0.0:
            raise ValueError("R must be finite and non-negative")
        own_selector = _position_selector(own_index)
        source_selector = _position_selector(source_index)
        relative_selector = own_selector - source_selector
        relative_prior = _sym(relative_selector @ self.covariance @ relative_selector.T)
        own_prior = _sym(own_selector @ self.covariance @ own_selector.T)
        own_position = self.local_state(uid)[POSITION_SLICE]
        source_position = self.local_state(source)[POSITION_SLICE]
        difference = own_position - source_position
        distance = float(np.linalg.norm(difference))
        rank = 0 if distance < 1.0e-12 else 1
        direction = np.zeros(3) if rank == 0 else difference / distance
        if scalar_noise_from_matrix is not None:
            scalar_noise = float(direction @ scalar_noise_from_matrix @ direction.T)
        H_scalar = direction.reshape(1, 3) @ relative_selector.reshape(3, JOINT_DIM)
        predicted_s = float((H_scalar @ self.covariance @ H_scalar.T).item() + scalar_noise)
        scalar_gain = np.zeros((JOINT_DIM, 1), dtype=float)
        own_post = own_prior.copy()
        relative_post = relative_prior.copy()
        relative_gain = np.zeros((3, 1), dtype=float)
        if rank:
            scalar_s = max(predicted_s, 1.0e-12)
            if scalar_noise_from_matrix is not None:
                scalar_noise = float(direction @ scalar_noise_from_matrix @ direction.T)
                predicted_s = float((H_scalar @ self.covariance @ H_scalar.T).item() + scalar_noise)
                scalar_s = max(predicted_s, 1.0e-12)
            scalar_gain = self.covariance @ H_scalar.T / scalar_s
            own_post = _sym(own_prior - (own_selector @ scalar_gain) @ (own_selector @ scalar_gain).T * scalar_s)
            # This is the relative 3-D state covariance after one scalar
            # range observation.  Only the measured direction is reduced;
            # the transverse directions remain in the covariance.
            relative_cross = relative_selector @ self.covariance @ H_scalar.T
            relative_gain = relative_cross / scalar_s
            relative_post = _sym(relative_prior - relative_cross @ relative_cross.T / scalar_s)
        own_before_trace = float(np.trace(own_prior))
        own_after_trace = float(np.trace(own_post))
        relative_before_trace = float(np.trace(relative_prior))
        relative_after_trace = float(np.trace(relative_post))
        result = {
            "uid": uid,
            "source": source,
            "predicted_innovation_variance": predicted_s,
            "scalar_predicted_S": predicted_s,
            "scalar_jacobian_rank": rank,
            "scalar_jacobian": H_scalar.tolist(),
            "own_position_covariance_trace_before": own_before_trace,
            "own_position_covariance_trace_after": own_after_trace,
            "own_position_trace_gain_fraction": _safe_fraction(own_before_trace, own_after_trace) if rank else 0.0,
            "relative_position_covariance_trace_before": relative_before_trace,
            "relative_position_covariance_trace_after": relative_after_trace,
            "relative_vector3d_prior_trace": relative_before_trace,
            "relative_vector3d_post_trace": relative_after_trace,
            "relative_vector3d_trace_gain_fraction": _safe_fraction(relative_before_trace, relative_after_trace),
            "relative_vector3d_logdet_gain": _logdet_gain(relative_prior, relative_post),
            "relative_jacobian_rank": rank,
            "relative_prior_covariance": relative_prior.tolist(),
            "relative_posterior_covariance": relative_post.tolist(),
            "gain_norm": float(np.linalg.norm(scalar_gain)) if rank else 0.0,
            "relative_gain_norm": float(np.linalg.norm(relative_gain)) if rank else 0.0,
        }
        # Short aliases keep diagnostic consumers independent of the exact
        # prose used for the long-form schema fields.
        result.update(
            {
                "predicted_S": predicted_s,
                "own_pos_prior_trace": own_before_trace,
                "own_pos_post_trace": own_after_trace,
                "own_pos_trace_gain_fraction": result["own_position_trace_gain_fraction"],
                "relative_vector_prior_trace": relative_before_trace,
                "relative_vector_post_trace": relative_after_trace,
                "relative_vector_trace_gain_fraction": result["relative_vector3d_trace_gain_fraction"],
                "relative_logdet_gain": result["relative_vector3d_logdet_gain"],
                "rank": rank,
            }
        )
        return result

    def diagnostics_snapshot(self) -> dict[str, Any]:
        return {
            "vo_events": [dict(event) for event in self.vo_update_events],
            "gnss_events": [dict(event) for event in self.gnss_update_events],
            "range_events": [dict(event) for event in self.range_update_events],
            "vo_update_attempted": len(self.vo_update_events),
            "vo_update_accepted": sum(bool(e.get("accepted")) for e in self.vo_update_events),
            "gnss_update_attempted": len(self.gnss_update_events),
            "gnss_update_accepted": sum(bool(e.get("accepted")) for e in self.gnss_update_events),
            "range_update_attempted": len(self.range_update_events),
            "range_update_accepted": sum(bool(e.get("accepted")) for e in self.range_update_events),
            "matrix_storage_bytes": self.matrix_storage_bytes,
        }


__all__ = [
    "UIDS",
    "JOINT_DIM",
    "YoungBoundEKF",
    "JointEKF",
    "fixed_gain_covariance_bound",
    "v5fixed_gain_covariance_bound",
    "fixed_gain_covariance_bound_v5",
]
