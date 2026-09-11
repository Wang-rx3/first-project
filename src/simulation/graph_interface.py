"""GAT/MAPPO observation contract without a policy trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .motion_model import UAVKinematicState
from .topology import TopologySnapshot


@dataclass
class GraphBatch:
    """NumPy graph tensors with an optional PyTorch conversion."""

    node_features: np.ndarray
    effective_edge_index: np.ndarray
    effective_edge_features: np.ndarray
    candidate_edge_index: np.ndarray
    candidate_edge_features: np.ndarray
    potential_edge_index: np.ndarray
    potential_edge_features: np.ndarray
    action_masks: dict[str, np.ndarray]
    bandwidth_masks: dict[str, np.ndarray]
    local_observations: dict[str, dict[str, Any]]

    def to_torch(self, device: str = "cpu") -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("PyTorch is required for graph tensor conversion") from exc
        return {
            "node_features": torch.as_tensor(self.node_features, dtype=torch.float32, device=device),
            "effective_edge_index": torch.as_tensor(self.effective_edge_index, dtype=torch.long, device=device),
            "effective_edge_features": torch.as_tensor(self.effective_edge_features, dtype=torch.float32, device=device),
            "candidate_edge_index": torch.as_tensor(self.candidate_edge_index, dtype=torch.long, device=device),
            "candidate_edge_features": torch.as_tensor(self.candidate_edge_features, dtype=torch.float32, device=device),
            "potential_edge_index": torch.as_tensor(self.potential_edge_index, dtype=torch.long, device=device),
            "potential_edge_features": torch.as_tensor(self.potential_edge_features, dtype=torch.float32, device=device),
            "action_masks": {
                key: torch.as_tensor(value, dtype=torch.bool, device=device)
                for key, value in self.action_masks.items()
            },
            "bandwidth_masks": {
                key: torch.as_tensor(value, dtype=torch.bool, device=device)
                for key, value in self.bandwidth_masks.items()
            },
        }


def _edge_feature(
    source: str,
    target: str,
    estimates: dict[str, np.ndarray],
    snapshot: TopologySnapshot,
    *,
    use_estimated_geometry: bool = True,
    expose_physical_los: bool = True,
    estimate_overrides: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    metrics = snapshot.metrics[(source, target)]
    if use_estimated_geometry:
        overrides = estimate_overrides or {}
        source_estimate = np.asarray(overrides.get(source, estimates[source]), dtype=float)
        target_estimate = np.asarray(overrides.get(target, estimates[target]), dtype=float)
        relative = target_estimate[:3] - source_estimate[:3]
        distance = float(np.linalg.norm(relative))
    else:
        # A candidate edge need not have delivered a neighbor state.  The
        # actor therefore sees only a measured UWB range when one exists and
        # an explicit missing-value sentinel otherwise; the hidden simulator
        # distance is never copied into the local observation.
        uwb = snapshot.uwb_measurements.get((source, target), {})
        relative = np.zeros(3, dtype=float)
        distance = float(uwb["measured_range"]) if uwb.get("valid") else -1.0
    return np.asarray(
        [
            *relative.tolist(),
            distance,
            float(metrics.los) if expose_physical_los else -1.0,
            metrics.rssi_dbm,
            metrics.snr_db,
            metrics.packet_loss,
            metrics.delay_s,
            metrics.aoi_s,
            metrics.queue_length_packets,
        ],
        dtype=float,
    )


def build_graph_batch(
    states: dict[str, UAVKinematicState],
    estimates: dict[str, np.ndarray],
    covariances: dict[str, np.ndarray],
    goals: dict[str, np.ndarray],
    local_obstacle_features: dict[str, np.ndarray],
    snapshot: TopologySnapshot,
    visible_messages: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    extra_node_features: Mapping[str, np.ndarray] | None = None,
) -> GraphBatch:
    """Build actor-visible local observations and critic-compatible tensors."""

    ids = sorted(states)
    index = {uav_id: position for position, uav_id in enumerate(ids)}
    visible_messages = visible_messages or {}
    extra_node_features = extra_node_features or {}
    extra_vectors = {
        str(uav_id): np.asarray(features, dtype=float).reshape(-1)
        for uav_id, features in extra_node_features.items()
    }
    extra_dimensions = {int(features.size) for features in extra_vectors.values()}
    if len(extra_dimensions) > 1:
        raise ValueError("all extra node feature vectors must have the same dimension")
    extra_feature_dim = next(iter(extra_dimensions), 0)
    obstacle_vectors = {
        str(uav_id): np.asarray(features, dtype=float).reshape(-1)
        for uav_id, features in local_obstacle_features.items()
    }
    obstacle_dimensions = {int(features.size) for features in obstacle_vectors.values()}
    if len(obstacle_dimensions) > 1:
        raise ValueError("all local obstacle feature vectors must have the same dimension")
    obstacle_feature_dim = next(iter(obstacle_dimensions), 4)
    if obstacle_feature_dim < 1:
        raise ValueError("local obstacle feature vectors must be non-empty")

    def visible_estimate(target: str, source: str) -> tuple[np.ndarray, np.ndarray]:
        message = visible_messages.get((target, source))
        if message is None:
            return np.asarray(estimates[source], dtype=float), np.asarray(covariances[source], dtype=float)
        return (
            np.asarray(message.get("state", estimates[source]), dtype=float),
            np.asarray(message.get("covariance", covariances[source]), dtype=float),
        )

    def node_row(uav_id: str, estimate: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        goal_delta = np.asarray(goals[uav_id], dtype=float) - np.asarray(estimate, dtype=float)[:3]
        obstacle = obstacle_vectors.get(uav_id, np.zeros(obstacle_feature_dim, dtype=float))
        extra = extra_vectors.get(uav_id, np.zeros(extra_feature_dim, dtype=float))
        return np.concatenate(
            [
                np.asarray(estimate, dtype=float),
                np.diag(np.asarray(covariance, dtype=float)),
                goal_delta,
                obstacle,
                extra,
            ]
        )

    node_rows: list[np.ndarray] = []
    for uav_id in ids:
        node_rows.append(node_row(uav_id, np.asarray(estimates[uav_id], dtype=float), np.asarray(covariances[uav_id], dtype=float)))

    def build_edges(edges: set[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        ordered = sorted(edges)
        edge_index = np.asarray(
            [[index[source], index[target]] for source, target in ordered],
            dtype=np.int64,
        ).T if ordered else np.empty((2, 0), dtype=np.int64)
        features = np.asarray(
            [_edge_feature(source, target, estimates, snapshot) for source, target in ordered],
            dtype=float,
        ) if ordered else np.empty((0, 11), dtype=float)
        return edge_index, features

    effective_index, effective_features = build_edges(snapshot.effective_edges)
    candidate_index, candidate_features = build_edges(snapshot.candidate_edges)
    potential_index, potential_features = build_edges(snapshot.potential_edges)
    masks: dict[str, np.ndarray] = {}
    bandwidth_masks: dict[str, np.ndarray] = {}
    local: dict[str, dict[str, Any]] = {}
    for uav_id in ids:
        potential_targets = [target for source, target in sorted(snapshot.potential_edges) if source == uav_id]
        candidate_targets = [target for source, target in sorted(snapshot.candidate_edges) if source == uav_id]
        potential_sources = [
            source
            for source, target in sorted(snapshot.potential_edges)
            if target == uav_id
        ]
        candidate_sources = [
            source
            for source, target in sorted(snapshot.candidate_edges)
            if target == uav_id
            and snapshot.uwb_measurements.get((source, uav_id), {}).get("valid", 0)
        ]
        # Measurement actions are receiver-side requests for a source UAV.
        # Keep the actor mask explicitly bounded by potential incoming edges;
        # the candidate/UWB check is a stricter execution-time eligibility
        # filter, not a way to expose any non-potential neighbor.
        candidate_source_set = set(candidate_sources)
        measurement_sources = [
            source for source in potential_sources if source in candidate_source_set
        ]
        candidate_sources_all = [
            source for source, target in sorted(snapshot.candidate_edges) if target == uav_id
        ]
        effective_sources = [source for source, target in sorted(snapshot.effective_edges) if target == uav_id]
        effective_targets = [target for source, target in sorted(snapshot.effective_edges) if source == uav_id]
        measurement_mask = np.zeros(len(ids) + 1, dtype=bool)  # final slot is explicit none
        for source in measurement_sources:
            measurement_mask[index[source]] = True
        measurement_mask[-1] = True
        # The final slot is an explicit ``none`` measurement choice. There is
        # intentionally no relay-action mask; bandwidth is the only outgoing
        # communication control.
        masks[uav_id] = measurement_mask
        bandwidth_mask = np.zeros(len(ids), dtype=bool)
        for target in potential_targets:
            bandwidth_mask[index[target]] = True
        bandwidth_masks[uav_id] = bandwidth_mask
        potential_edge_index = np.asarray(
            [[index[uav_id], index[target]] for target in potential_targets],
            dtype=np.int64,
        ).T if potential_targets else np.empty((2, 0), dtype=np.int64)
        candidate_node_ids = [f"candidate:{target}" for target in candidate_targets]
        local_node_ids = [uav_id] + [source for source in effective_sources if source != uav_id] + candidate_node_ids
        local_index = {node_id: local_position for local_position, node_id in enumerate(local_node_ids)}
        local_effective_edges = [
            (source, target)
            for source, target in sorted(snapshot.effective_edges)
            if target == uav_id
        ]
        local_effective_index = np.asarray(
            [[local_index[source], local_index[target]] for source, target in local_effective_edges],
            dtype=np.int64,
        ).T if local_effective_edges else np.empty((2, 0), dtype=np.int64)
        local_effective_features = np.asarray(
            [
                _edge_feature(
                    source,
                    target,
                    estimates,
                    snapshot,
                    expose_physical_los=False,
                    estimate_overrides={
                        source: visible_estimate(uav_id, source)[0],
                        target: np.asarray(estimates[target], dtype=float),
                    },
                )
                for source, target in local_effective_edges
            ],
            dtype=float,
        ) if local_effective_edges else np.empty((0, 11), dtype=float)
        local_node_features = np.asarray(
            [
                node_row(
                    node_id,
                    *visible_estimate(uav_id, node_id),
                )
                if node_id in index and node_id != uav_id and (uav_id, node_id) in visible_messages
                else node_rows[index[node_id]] if node_id in index else np.zeros_like(node_rows[0])
                for node_id in local_node_ids
            ],
            dtype=float,
        )
        candidate_edge_index = np.asarray(
            [
                [0, len(local_node_ids) - len(candidate_targets) + offset]
                for offset, _target in enumerate(candidate_targets)
            ],
            dtype=np.int64,
        ).T if candidate_targets else np.empty((2, 0), dtype=np.int64)
        local[uav_id] = {
            "estimate": estimates[uav_id].copy(),
            "covariance": covariances[uav_id].copy(),
            "goal": goals[uav_id].copy(),
            "obstacle_features": obstacle_vectors.get(
                uav_id, np.zeros(obstacle_feature_dim, dtype=float)
            ).copy(),
            "candidate_targets": candidate_targets,
            "candidate_measurement_sources": candidate_sources,
            "candidate_incoming_sources": candidate_sources_all,
            "effective_sources": effective_sources,
            "effective_targets": effective_targets,
            "neighbor_messages": {
                source: {
                    "estimate": visible_estimate(uav_id, source)[0].copy(),
                    "covariance": visible_estimate(uav_id, source)[1].copy(),
                }
                for source in effective_sources
            },
            "candidate_edge_features": {
                target: _edge_feature(
                    uav_id,
                    target,
                    estimates,
                    snapshot,
                    use_estimated_geometry=False,
                    expose_physical_los=False,
                )
                for target in candidate_targets
            },
            "candidate_incoming_edge_features": {
                source: _edge_feature(
                    source,
                    uav_id,
                    estimates,
                    snapshot,
                    use_estimated_geometry=False,
                    expose_physical_los=False,
                )
                for source in candidate_sources_all
            },
            "potential_targets": potential_targets,
            "potential_edge_index": potential_edge_index,
            "potential_edge_features": {
                target: _edge_feature(
                    uav_id,
                    target,
                    estimates,
                    snapshot,
                    use_estimated_geometry=False,
                    expose_physical_los=False,
                )
                for target in potential_targets
            },
            "potential_links": {
                target: _link_observation(snapshot.metrics[(uav_id, target)])
                for target in potential_targets
            },
            "candidate_links": {
                target: _link_observation(snapshot.metrics[(uav_id, target)])
                for target in candidate_targets
            },
            "candidate_incoming_links": {
                source: _link_observation(snapshot.metrics[(source, uav_id)])
                for source in candidate_sources_all
            },
            "effective_links": {
                source: _link_observation(snapshot.metrics[(source, uav_id)])
                for source in effective_sources
            },
            "uwb_available": {
                source: bool(
                    snapshot.uwb_measurements.get((source, uav_id), {}).get("valid", 0)
                )
                for source, target in sorted(snapshot.uwb_measurements)
                if target == uav_id
            },
            "uwb_ranges": {
                source: snapshot.uwb_measurements[(source, uav_id)].get("measured_range")
                for source, target in sorted(snapshot.uwb_measurements)
                if target == uav_id
            },
            "uwb_available_outgoing": {
                target: bool(
                    snapshot.uwb_measurements.get((uav_id, target), {}).get("valid", 0)
                )
                for source, target in sorted(snapshot.uwb_measurements)
                if source == uav_id
            },
            "uwb_ranges_outgoing": {
                target: snapshot.uwb_measurements[(uav_id, target)].get("measured_range")
                for source, target in sorted(snapshot.uwb_measurements)
                if source == uav_id
            },
            "node_ids": local_node_ids,
            "node_features": local_node_features,
            "effective_edge_index": local_effective_index,
            "effective_edge_features": local_effective_features,
            "candidate_edge_index": candidate_edge_index,
            "action_mask": masks[uav_id].copy(),
            "bandwidth_mask": bandwidth_masks[uav_id].copy(),
        }
    return GraphBatch(
        node_features=np.asarray(node_rows, dtype=float),
        effective_edge_index=effective_index,
        effective_edge_features=effective_features,
        candidate_edge_index=candidate_index,
        candidate_edge_features=candidate_features,
        potential_edge_index=potential_index,
        potential_edge_features=potential_features,
        action_masks=masks,
        bandwidth_masks=bandwidth_masks,
        local_observations=local,
    )


def _link_observation(metrics) -> dict[str, float | int | str]:
    """Expose channel state used by an actor without exposing true geometry."""

    return {
        "rssi_dbm": float(metrics.rssi_dbm),
        "snr_db": float(metrics.snr_db),
        "channel_bandwidth_hz": float(metrics.channel_bandwidth_hz),
        "rate_bps": float(metrics.rate_bps),
        "packet_loss": float(metrics.packet_loss),
        "delay_s": float(metrics.delay_s),
        "aoi_s": float(metrics.aoi_s),
        "queue_length_packets": float(metrics.queue_length_packets),
        "candidate_edge": int(metrics.candidate_edge),
        "effective_edge": int(metrics.effective_edge),
        "link_state": str(metrics.link_state),
    }


__all__ = ["GraphBatch", "build_graph_batch"]
