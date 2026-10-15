"""Scoped dynamic GNSS/radio quality adapter for the v13 audit.

The adapter deliberately delegates motion, filtering, queue service, packet
expiry, and episode control to the v7/v10 environment.  The only experimental
changes are the spatial quality field, the corresponding sensor/channel
parameters, and packet quality metadata used by the audit logger.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from . import neighbor_selection_execution_v11 as prior
from . import partial_gnss_baseline_v7 as v7
from . import partial_gnss_execution_v10 as native
from ..simulation import environment as environment_module
from ..simulation import sensor_model
from ..simulation import topology as topology_module
from ..synthetic import channel_model


# This reference must be captured before a caller lets v10 temporarily replace
# v7.PartialGNSSEnv with its environment factory.
ORIGINALCLASS = v7.PartialGNSSEnv
_ORIGINAL_PARTIAL_GNSS_ENV = ORIGINALCLASS
_ORIGINAL_SAMPLE_GNSS = sensor_model.sample_gnss
_ORIGINAL_SAMPLE_IMU = sensor_model.sample_imu
_ORIGINAL_SAMPLE_VO = sensor_model.sample_visual_odometry

PACKET_SIZE_BYTES = 512
QUALITY_HEADER_BYTES = 16
QUALITY_PAYLOAD_BYTES = PACKET_SIZE_BYTES - QUALITY_HEADER_BYTES
V13_PACKET_SIZE_BYTES = PACKET_SIZE_BYTES
V13_QUALITY_HEADER_BYTES = QUALITY_HEADER_BYTES
V13_BASE_PAYLOAD_BYTES = QUALITY_PAYLOAD_BYTES
assert QUALITY_PAYLOAD_BYTES + QUALITY_HEADER_BYTES == PACKET_SIZE_BYTES


def _quality_level(phase: float) -> tuple[float, float, float, str]:
    """Return ``(GNSS std, radio loss, access wait, label)`` for a phase."""

    if phase < 5.0:
        return 0.5, 0.0, 0.02, "clear"
    if phase < 10.0:
        return 1.5, 6.0, 0.22, "mild"
    if phase < 15.0:
        return 3.0, 12.0, 0.42, "severe"
    return 1.5, 6.0, 0.22, "mild"


def quality_at(position: Any, offset: float = 0) -> dict[str, Any]:
    """Evaluate the frozen spatial quality field at one world position.

    The returned ``position`` is intentionally not included.  Callers that
    write evaluation logs may retain their own copy of the world coordinate;
    actor-facing code receives no field coordinates or cell labels.
    """

    point = np.asarray(position, dtype=float).reshape(-1)
    if point.size != 3 or not np.isfinite(point).all():
        raise ValueError("position must be a finite 3-D vector")
    offset_value = float(offset)
    if not np.isfinite(offset_value):
        raise ValueError("offset must be finite")
    phase = float(
        (
            point[0]
            - 40.0
            + 0.1 * (point[1] - 125.0)
            + 0.15 * (point[2] - 28.0)
            + offset_value
        )
        % 20.0
    )
    if point[2] > 50.0:
        std, loss, wait, level = 0.5, 0.0, 0.02, "clear"
    else:
        std, loss, wait, level = _quality_level(phase)
    return {
        "phase": phase,
        "gnss_std_m": float(std),
        "excess_loss_db": float(loss),
        "access_wait_s": float(wait),
        "level": level,
    }


class _FirstGaussWithLoss:
    """Proxy preserving the native RNG call sequence while shifting shadowing."""

    def __init__(self, rng: Any, loss_db: float, expected_sigma: float) -> None:
        self._rng = rng
        self._loss_db = float(loss_db)
        self._expected_sigma = float(expected_sigma)
        self._first = True

    def gauss(self, mu: float, sigma: float) -> float:
        value = self._rng.gauss(mu, sigma)
        if self._first:
            self._first = False
            if not np.isclose(float(mu), 0.0, rtol=0.0, atol=1e-12) or not np.isclose(
                float(sigma), self._expected_sigma, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError("v13 channel first Gaussian signature changed")
            return float(value + self._loss_db)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rng, name)


def channel_metrics(
    kwargs: Mapping[str, Any],
    loss: float,
    wait: float,
    enabled: bool = True,
) -> channel_model.LinkMetrics:
    """Call the native channel model and append one access delay when enabled.

    A disabled call is intentionally a direct native call, including its RNG
    object, so both the returned metrics and RNG state remain bit-equivalent.
    """

    call_kwargs = dict(kwargs)
    if not enabled:
        return channel_model.compute_link_metrics(**call_kwargs)
    rng = call_kwargs.get("rng")
    if rng is None:
        raise TypeError("channel metric kwargs must contain rng")
    config = call_kwargs.get("config") or channel_model.ChannelConfig()
    expected_sigma = config.shadowing_std_los_db if bool(call_kwargs.get("los")) else config.shadowing_std_nlos_db
    base = channel_model.compute_link_metrics(
        **{
            **call_kwargs,
            "rng": _FirstGaussWithLoss(rng, float(loss), float(expected_sigma)),
        }
    )
    base_total_delay = max(0.0, float(base.raw_delay_s))
    total_delay = base_total_delay + max(0.0, float(wait))
    max_delay = float(config.max_delay_s)
    clipped_delay = min(max_delay, total_delay)
    if total_delay > max_delay:
        # Reapply the unchanged raw-delay candidate condition to the total.
        return replace(
            base,
            raw_delay_s=float(total_delay),
            delay_s=float(clipped_delay),
            candidate_edge=0,
            effective_edge=0,
            packet_success=0,
            drop_reason="v13_delay_limit",
            link_state="unavailable",
        )
    return replace(base, raw_delay_s=float(total_delay), delay_s=float(clipped_delay))


def _empty_quality() -> dict[str, Any]:
    return {
        "gnss_std_m": None,
        "gnss_report_time_s": None,
        "packet_id": None,
    }


def _received_quality_for(env: Any, message: Any) -> dict[str, Any]:
    result = _empty_quality()
    if message is None or env is None:
        return result
    packet_id = getattr(message, "attr_v13_packet_id", None)
    envelope = getattr(message, "v13_packet_envelope", None)
    if packet_id is None or envelope is None:
        return result
    registry = getattr(env, "v13_packet_registry", {})
    entry = registry.get(packet_id)
    if not isinstance(entry, Mapping) or entry.get("message") is not message:
        return result
    try:
        current_envelope = (
            str(source := getattr(message, "source")),
            str(target := getattr(message, "target")),
            float(getattr(message, "send_time_s")),
            int(getattr(message, "arrival_step")),
            int(tuple(packet_id)[4]),
        )
    except (IndexError, TypeError, ValueError, AttributeError):
        return result
    if tuple(envelope) != tuple(entry.get("envelope", ())) or tuple(envelope) != current_envelope:
        return result
    arrived = getattr(env, "arrived_messages", {})
    if arrived.get((source, target)) is not message:
        return result
    result.update(
        {
            "gnss_std_m": entry.get("gnss_std_m"),
            "gnss_report_time_s": entry.get("gnss_report_time_s"),
            "packet_id": packet_id,
        }
    )
    return result


def received_quality(message: Any) -> dict[str, Any]:
    """Return quality metadata only for the actual arrived packet object."""

    return _received_quality_for(getattr(message, "_v13_owner", None), message)


class DynamicEnv(prior.NeighborAuditEnv):
    """v11 audit environment with scoped dynamic sensor/radio adapters."""

    def __init__(
        self,
        *args: Any,
        quality_offset: float = 0,
        dynamic_gnss: bool = False,
        dynamic_radio: bool = False,
        prefix_steps: int = 6,
        precheck_guard: bool = True,
        **kwargs: Any,
    ) -> None:
        # Reset/initialization may call _refresh_topology, so every attribute
        # consumed by an override exists before native construction starts.
        self.v13_quality_offset = float(quality_offset)
        self.v13_dynamic_gnss = bool(dynamic_gnss)
        self.v13_dynamic_radio = bool(dynamic_radio)
        self.v13_prefix_steps = int(prefix_steps)
        self.v13_precheck_guard = bool(precheck_guard)
        self.v13_quality_log: list[dict[str, Any]] = []
        self.v13_radio_log: list[dict[str, Any]] = []
        self.v13_metadata_events: list[dict[str, Any]] = []
        self.v13_failure_probes: list[dict[str, Any]] = []
        self.v13_packet_registry: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.v13_prefix_guard_calls: list[int] = []
        self.v13_prefix_limit_marker: dict[str, Any] | None = None
        self._v13_last_gnss: dict[str, dict[str, Any]] = {}
        self._v13_pre_gnss: dict[str, dict[str, Any]] = {}
        self._v13_pre_payloads: dict[str, dict[str, np.ndarray]] = {}
        self._v13_sensor_packets: dict[str, dict[str, Any]] = {}
        self._v13_serial = 0
        super().__init__(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.v13_quality_log.clear()
        self.v13_radio_log.clear()
        self.v13_metadata_events.clear()
        self.v13_failure_probes.clear()
        self.v13_packet_registry.clear()
        self.v13_prefix_guard_calls.clear()
        self.v13_prefix_limit_marker = None
        self._v13_last_gnss = {}
        self._v13_pre_gnss = {}
        self._v13_pre_payloads = {}
        self._v13_sensor_packets = {}
        self._v13_serial = 0
        return super().reset(*args, **kwargs)

    def step(self, actions: Mapping[str, Any]) -> Any:
        if self.v13_precheck_guard and int(getattr(self, "timestep", 0)) >= self.v13_prefix_steps:
            current = int(getattr(self, "timestep", 0))
            self.v13_prefix_guard_calls.append(current)
            self.v13_prefix_limit_marker = {
                "timestep": current,
                "prefix_steps": int(self.v13_prefix_steps),
                "reason": "v13_prefix_limit",
            }
            raise RuntimeError("v13_prefix_limit")
        self._v13_pre_gnss = copy.deepcopy(self._v13_last_gnss)
        self._v13_pre_payloads = {
            uid: {
                "state": np.asarray(ekf.state, dtype=float).copy(),
                "covariance": np.asarray(ekf.covariance, dtype=float).copy(),
            }
            for uid, ekf in getattr(self, "ekfs", {}).items()
        }
        self._v13_sensor_packets = {}
        try:
            return super().step(actions)
        except BaseException as exc:
            filters = {
                uid: {
                    "state": np.asarray(ekf.state, dtype=float).copy(),
                    "covariance": np.asarray(ekf.covariance, dtype=float).copy(),
                }
                for uid, ekf in getattr(self, "ekfs", {}).items()
            }
            self.v13_failure_probes.append(
                {
                    "step": int(getattr(self, "timestep", 0)) + 1,
                    "exception": {"type": type(exc).__name__, "reason": str(exc)},
                    "sensor_packets": copy.deepcopy(self._v13_sensor_packets),
                    "filter_state": filters,
                }
            )
            raise

    def _refresh_topology(
        self,
        bandwidth: dict[str, dict[str, float]],
        *,
        measurement_arrivals: Mapping[tuple[str, str], float] | None = None,
    ) -> Any:
        if not self.v13_dynamic_radio:
            return super()._refresh_topology(
                bandwidth, measurement_arrivals=measurement_arrivals
            )

        ids = sorted(self.states)
        max_range = float(self.config.topology.max_comm_range_m)
        pairs: list[tuple[str, str]] = []
        for source in ids:
            for target in ids:
                if source == target:
                    continue
                distance = float(
                    np.linalg.norm(
                        self.states[source].position_m - self.states[target].position_m
                    )
                )
                if distance <= max_range:
                    pairs.append((source, target))
        pair_iter = iter(pairs)
        now = (int(getattr(self, "timestep", 0)) + 1) * float(self.config.motion.dt_s)

        def wrapped_compute(**kwargs: Any) -> channel_model.LinkMetrics:
            try:
                source, target = next(pair_iter)
            except StopIteration as exc:
                raise RuntimeError("v13 topology metric pair underflow") from exc
            expected_distance = float(
                np.linalg.norm(
                    self.states[source].position_m - self.states[target].position_m
                )
            )
            if abs(float(kwargs.get("distance_m")) - expected_distance) > 1e-8:
                raise RuntimeError("v13 topology metric pair mismatch")
            midpoint = (
                np.asarray(self.states[source].position_m, dtype=float)
                + np.asarray(self.states[target].position_m, dtype=float)
            ) / 2.0
            quality = quality_at(midpoint, self.v13_quality_offset + 5.0)
            base_total_delay = float(kwargs.get("queue_length_packets", 0.0))
            metrics = channel_metrics(
                kwargs,
                quality["excess_loss_db"],
                quality["access_wait_s"],
                enabled=True,
            )
            self.v13_radio_log.append(
                {
                    "time_s": float(now),
                    "source": source,
                    "target": target,
                    "pair": (source, target),
                    "level": quality["level"],
                    "regime": quality["level"],
                    "excess_loss_db": float(quality["excess_loss_db"]),
                    "loss_db": float(quality["excess_loss_db"]),
                    "access_wait_s": float(quality["access_wait_s"]),
                    "base_totaldelay": float(metrics.raw_delay_s - quality["access_wait_s"]),
                    "base_totaldelay_s": float(metrics.raw_delay_s - quality["access_wait_s"]),
                    "raw_totaldelay_s": float(metrics.raw_delay_s),
                    "edge_delay_s": float(metrics.delay_s),
                    "rssi_dbm": float(metrics.rssi_dbm),
                    "packet_loss": float(metrics.packet_loss),
                    "loss": float(metrics.packet_loss),
                    "rate_bps": float(metrics.rate_bps),
                    "candidate_edge": int(metrics.candidate_edge),
                    "effective_edge": int(metrics.effective_edge),
                    "packet_success": int(metrics.packet_success),
                    "drop_reason": str(metrics.drop_reason),
                    "arrival_slots": max(
                        1,
                        int(
                            math.ceil(
                                float(metrics.delay_s)
                                / max(float(self.config.motion.dt_s), 1e-9)
                            )
                        ),
                    ),
                }
            )
            return metrics

        with patch.object(topology_module, "compute_link_metrics", wrapped_compute):
            snapshot = super()._refresh_topology(
                bandwidth, measurement_arrivals=measurement_arrivals
            )
        try:
            next(pair_iter)
        except StopIteration:
            return snapshot
        raise RuntimeError("v13 topology metric pair overflow")

    def _enqueue_messages(
        self,
        transmitted_edges: set[tuple[str, str]],
        sender_payloads: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        before_ids = {id(message) for message in self.pending_messages}
        super()._enqueue_messages(transmitted_edges, sender_payloads)
        created = [message for message in self.pending_messages if id(message) not in before_ids]
        quality_enabled = bool(self.v13_dynamic_gnss or self.v13_dynamic_radio)
        for message in created:
            self._v13_serial += 1
            packet_id = (
                str(message.source),
                str(message.target),
                float(message.send_time_s),
                int(message.arrival_step),
                int(self._v13_serial),
            )
            envelope = tuple(packet_id)
            setattr(message, "attr_v13_packet_id", packet_id)
            setattr(message, "v13_packet_envelope", envelope)
            setattr(message, "_v13_owner", self)
            source_quality = self._v13_pre_gnss.get(str(message.source), {}) if quality_enabled else {}
            gnss_std = source_quality.get("gnss_std_m")
            report_time = source_quality.get("report_time_s")
            self.v13_packet_registry[packet_id] = {
                "message": message,
                "envelope": envelope,
                "gnss_std_m": None if gnss_std is None else float(gnss_std),
                "gnss_report_time_s": None if report_time is None else float(report_time),
                "state_time_s": None if report_time is None else float(report_time),
                "pre_payload_state": self._v13_pre_payloads.get(str(message.source), {}).get("state"),
                "pre_payload_covariance": self._v13_pre_payloads.get(str(message.source), {}).get("covariance"),
            }
            quality_header = QUALITY_HEADER_BYTES if quality_enabled else 0
            payload_bytes = PACKET_SIZE_BYTES - quality_header
            assert payload_bytes + quality_header == PACKET_SIZE_BYTES
            self.v13_metadata_events.append(
                {
                    "event": "enqueue",
                    "packet_id": packet_id,
                    "source": str(message.source),
                    "target": str(message.target),
                    "send_time_s": float(message.send_time_s),
                    "arrival_step": int(message.arrival_step),
                    "serial": int(self._v13_serial),
                    "packet_size_bytes": PACKET_SIZE_BYTES,
                    "payload_bytes": int(payload_bytes),
                    "quality_header_bytes": int(quality_header),
                    "header_bytes": int(quality_header),
                    "quality_metadata_enabled": int(quality_enabled),
                    "gnss_std_m": None if gnss_std is None else float(gnss_std),
                    "gnss_report_time_s": None if report_time is None else float(report_time),
                    "state_time_s": None if report_time is None else float(report_time),
                }
            )

    def _release_messages(self, next_timestep: int) -> None:
        before = {id(message) for message in self.arrived_messages.values()}
        super()._release_messages(next_timestep)
        for message in self.arrived_messages.values():
            if id(message) in before:
                continue
            quality = self.received_quality(message)
            self.v13_metadata_events.append({"event": "visible_arrival", "step": next_timestep,
                "packet_id": quality["packet_id"], "source": message.source, "target": message.target,
                "send_time_s": message.send_time_s, "arrival_step": message.arrival_step,
                "gnss_std_m": quality["gnss_std_m"], "gnss_report_time_s": quality["gnss_report_time_s"],
                "valid_quality": quality["gnss_std_m"] is not None,
                "scope": "new packet visible in arrived buffer, not total physical deliveries overwritten in same step"})
        live = {
            getattr(message, "attr_v13_packet_id", None)
            for message in [*self.pending_messages, *self.arrived_messages.values()]
        }
        self.v13_packet_registry = {
            packet_id: entry
            for packet_id, entry in self.v13_packet_registry.items()
            if packet_id in live and entry.get("message") is not None
        }

    def received_quality(self, message: Any) -> dict[str, Any]:
        return _received_quality_for(self, message)


def _dynamic_gnss_step(self: DynamicEnv, actions: Mapping[str, Any]) -> Any:
    """Patched v7 method: preserve its UID policy and native transition."""

    def gnss_sensor(state: Any, mode: str, rng: Any, config: Any) -> dict[str, Any]:
        uid = next(uid for uid, value in self.states.items() if value is state)
        report_time = (int(getattr(self, "timestep", 0)) + 1) * float(self.config.motion.dt_s)
        if uid in v7.PEERS:
            if self.v13_dynamic_gnss:
                quality = quality_at(state.position_m, self.v13_quality_offset)
                dynamic_config = replace(
                    config,
                    gnss_nominal_std_m=float(quality["gnss_std_m"]),
                )
                packet = _ORIGINAL_SAMPLE_GNSS(state, "nominal", rng, dynamic_config)
                level = quality["level"]
            else:
                packet = _ORIGINAL_SAMPLE_GNSS(state, "nominal", rng, config)
                level = "baseline"
        else:
            # The denied target uses the native denied branch, including its
            # zero-draw behavior and fixed 20 m declared standard deviation.
            packet = _ORIGINAL_SAMPLE_GNSS(state, "denied", rng, config)
            level = "denied"
        self._v13_sensor_packets.setdefault(uid, {})["gnss"] = copy.deepcopy(packet)
        if self.v13_dynamic_gnss or self.v13_dynamic_radio:
            self.v13_quality_log.append(
                {
                    "uid": str(uid),
                    "time_s": float(report_time),
                    "gnss_std_m": float(packet["std"]),
                    "level": level,
                    "position": np.asarray(state.position_m, dtype=float).copy(),
                    "world_position_m": np.asarray(state.position_m, dtype=float).copy(),
                }
            )
        self._v13_last_gnss[str(uid)] = {
            "gnss_std_m": float(packet["std"]),
            "report_time_s": float(report_time),
        }
        return packet

    def imu_capture(state: Any, accel_bias: Any, gyro_bias: Any, rng: Any, config: Any) -> dict[str, Any]:
        packet = _ORIGINAL_SAMPLE_IMU(state, accel_bias, gyro_bias, rng, config)
        uid = next(uid for uid, value in self.states.items() if value is state)
        self._v13_sensor_packets.setdefault(uid, {})["imu"] = copy.deepcopy(packet)
        return packet

    def vo_capture(state: Any, dt_s: float, rng: Any, config: Any, *, reference_position: Any = None) -> dict[str, Any]:
        packet = _ORIGINAL_SAMPLE_VO(
            state,
            dt_s,
            rng,
            config,
            reference_position=reference_position,
        )
        uid = next(uid for uid, value in self.states.items() if value is state)
        self._v13_sensor_packets.setdefault(uid, {})["vo"] = copy.deepcopy(packet)
        return packet

    with patch.object(environment_module, "sample_imu", imu_capture), patch.object(
        environment_module, "sample_visual_odometry", vo_capture
    ), patch.object(environment_module, "sample_gnss", gnss_sensor):
        # ORIGINALCLASS is the original v7 adapter.  Its next MRO entry is
        # DiagnosticEnv, so the original v7 closure is bypassed while every
        # native transition below it remains intact.
        return super(ORIGINALCLASS, self).step(actions)


@contextmanager
def adapters(case: Mapping[str, Any], spec: Mapping[str, Any]) -> Iterator[type[DynamicEnv]]:
    """Yield a configured environment class for the root v10 patch scope."""

    if not isinstance(case, Mapping) or not isinstance(spec, Mapping):
        raise TypeError("case and spec must be mappings")
    quality_offset = float(case.get("quality_offset", 0))
    if quality_offset not in (0.0, 10.0):
        raise ValueError("quality_offset must be 0 or 10")
    dynamic_gnss = bool(spec.get("dynamic_gnss", False))
    dynamic_radio = bool(spec.get("dynamic_radio", False))
    prefix_explicit = "prefix_steps" in spec
    prefix_steps = int(spec.get("prefix_steps", 6))
    precheck_guard = bool(spec.get("precheck_guard", True)) if prefix_explicit else False

    class ConfiguredDynamicEnv(DynamicEnv):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                *args,
                quality_offset=quality_offset,
                dynamic_gnss=dynamic_gnss,
                dynamic_radio=dynamic_radio,
                prefix_steps=prefix_steps,
                precheck_guard=precheck_guard,
                **kwargs,
            )

    ConfiguredDynamicEnv.__name__ = "ConfiguredDynamicEnv"
    with patch.object(ORIGINALCLASS, "step", _dynamic_gnss_step):
        yield ConfiguredDynamicEnv


__all__ = [
    "DynamicEnv",
    "ORIGINALCLASS",
    "adapters",
    "channel_metrics",
    "quality_at",
    "received_quality",
]
