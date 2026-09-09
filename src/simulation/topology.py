"""Potential/candidate/effective graph construction for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..synthetic.channel_model import ChannelConfig, LinkMetrics, compute_link_metrics
from .obstacle_model import Obstacle, line_of_sight
from .motion_model import UAVKinematicState
from .sensor_model import SensorConfig, sample_uwb


@dataclass(frozen=True)
class TopologyConfig:
    max_comm_range_m: float = 75.0
    uwb_range_m: float = 100.0
    message_arrivals_per_window: float = 1.0
    default_bandwidth_ratio: float = 0.0


@dataclass
class TopologySnapshot:
    potential_edges: set[tuple[str, str]] = field(default_factory=set)
    candidate_edges: set[tuple[str, str]] = field(default_factory=set)
    effective_edges: set[tuple[str, str]] = field(default_factory=set)
    uwb_edges: set[tuple[str, str]] = field(default_factory=set)
    metrics: dict[tuple[str, str], LinkMetrics] = field(default_factory=dict)
    uwb_measurements: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    queue_next: dict[tuple[str, str], float] = field(default_factory=dict)


def normalize_bandwidth_allocations(
    source: str,
    allocations: dict[str, float] | None,
    potential_targets: list[str],
    total_bandwidth_ratio: float = 1.0,
) -> dict[str, float]:
    """Normalize a per-source allocation vector to a feasible simplex."""

    if not potential_targets:
        return {}
    raw = {
        target: max(0.0, float((allocations or {}).get(target, 0.0)))
        for target in potential_targets
    }
    if not any(raw.values()):
        # An omitted/zero action means that no bandwidth is allocated.  The
        # environment therefore cannot silently turn a zero action into an
        # equal-share transmission policy.
        return {target: 0.0 for target in potential_targets}
    total = sum(raw.values())
    scale = min(1.0, max(0.0, float(total_bandwidth_ratio)) / total)
    return {target: value * scale for target, value in raw.items()}


def build_topology(
    states: dict[str, UAVKinematicState],
    obstacles: list[Obstacle],
    bandwidth_allocations: dict[str, dict[str, float]],
    queue_state: dict[tuple[str, str], float],
    aoi_state: dict[tuple[str, str], float],
    *,
    rng: np.random.Generator,
    dt_s: float,
    channel_config: ChannelConfig | None = None,
    topology_config: TopologyConfig | None = None,
    sensor_config: SensorConfig | None = None,
    aoi_reset_on_success: bool = True,
    measurement_arrivals: Mapping[tuple[str, str], float] | None = None,
    rng_factory=None,
) -> TopologySnapshot:
    """Build graphs and independent UWB measurements for one environment step."""

    channel = channel_config or ChannelConfig()
    topology = topology_config or TopologyConfig()
    sensors = sensor_config or SensorConfig(uwb_max_range_m=topology.uwb_range_m)
    measurement_arrivals = measurement_arrivals or {}
    snapshot = TopologySnapshot()
    ids = sorted(states)
    for source in ids:
        for target in ids:
            if source == target:
                continue
            key = (source, target)
            source_state = states[source]
            target_state = states[target]
            distance = float(np.linalg.norm(source_state.position_m - target_state.position_m))
            los = line_of_sight(source_state.position_m, target_state.position_m, obstacles)

            # UWB availability is evaluated independently of the radio
            # communication range and bandwidth allocation.
            if distance <= topology.uwb_range_m:
                uwb = sample_uwb(
                    source_state,
                    target_state,
                    los=los,
                    rng=rng_factory("uwb", source, target) if rng_factory is not None else rng,
                    config=sensors,
                )
                if uwb["valid"]:
                    snapshot.uwb_edges.add(key)
                snapshot.uwb_measurements[key] = uwb

            if distance > topology.max_comm_range_m:
                continue
            snapshot.potential_edges.add(key)

        potential_targets = [
            target for target in ids
            if target != source and (source, target) in snapshot.potential_edges
        ]
        allocation = normalize_bandwidth_allocations(
            source,
            bandwidth_allocations.get(source),
            potential_targets,
        )
        for target in potential_targets:
            key = (source, target)
            source_state = states[source]
            target_state = states[target]
            distance = float(np.linalg.norm(source_state.position_m - target_state.position_m))
            los = line_of_sight(source_state.position_m, target_state.position_m, obstacles)
            metrics = compute_link_metrics(
                distance_m=distance,
                los=los,
                bandwidth_ratio=allocation.get(target, 0.0),
                queue_length_packets=(
                    max(0.0, float(queue_state.get(key, 0.0)))
                    + max(0.0, float(measurement_arrivals.get(key, 0.0)))
                ),
                previous_aoi_s=aoi_state.get(key, 0.0),
                dt_s=dt_s,
                rng=_as_random(rng_factory("channel", source, target) if rng_factory is not None else rng),
                config=channel,
                max_comm_range_m=topology.max_comm_range_m,
                aoi_reset_on_success=aoi_reset_on_success,
            )
            snapshot.metrics[key] = metrics
            if metrics.candidate_edge:
                snapshot.candidate_edges.add(key)
            if metrics.effective_edge:
                snapshot.effective_edges.add(key)
            aoi_state[key] = metrics.aoi_s
            arrivals = max(0.0, topology.message_arrivals_per_window)
            service = metrics.rate_bps * dt_s / (channel.packet_size_bytes * 8.0) if metrics.rate_bps > 0 else 0.0
            queue_before = max(0.0, float(queue_state.get(key, 0.0)))
            measurement_load = max(0.0, float(measurement_arrivals.get(key, 0.0)))
            snapshot.queue_next[key] = max(0.0, queue_before + measurement_load + arrivals - service)
    return snapshot


def _as_random(rng: np.random.Generator):
    """Adapt NumPy generator draws to the legacy channel interface."""

    import random

    seed = int(rng.integers(0, 2**32 - 1))
    return random.Random(seed)


__all__ = [
    "TopologyConfig",
    "TopologySnapshot",
    "build_topology",
    "normalize_bandwidth_allocations",
]
