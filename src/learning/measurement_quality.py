"""Leakage-safe, actor-visible measurement-quality features.

The quality rule is deliberately kept separate from the EKF and from the
environment transition.  It consumes only fields already exposed in a local
actor observation.  Quantiles are fitted from warm-up snapshots and the
resulting model is immutable during train/validation/test.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HIGH_IS_GOOD = ("rssi_dbm", "snr_db")
LOW_IS_GOOD = (
    "packet_loss",
    "delay_s",
    "aoi_s",
    "queue_length_packets",
    "neighbor_covariance_trace_m2",
)
RISK_FIELDS = HIGH_IS_GOOD + LOW_IS_GOOD


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _risk(value: float, field: str, model: Mapping[str, Any]) -> float:
    limits = model["quantiles"][field]
    q25 = float(limits["q25"])
    q75 = float(limits["q75"])
    width = max(abs(q75 - q25), 1e-12)
    if field in HIGH_IS_GOOD:
        return float(np.clip((q75 - float(value)) / width, 0.0, 1.0))
    return float(np.clip((float(value) - q25) / width, 0.0, 1.0))


def _local_candidates(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
) -> list[str]:
    """Return only measurement sources visible through the local mask."""

    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping):
        return []
    mask = np.asarray(observation.get("action_mask", []), dtype=bool).reshape(-1)
    output: list[str] = []
    for index, source in enumerate(uav_ids):
        if index >= mask.size or not bool(mask[index]):
            continue
        if source in incoming:
            output.append(str(source))
    # Some serialized observations do not retain the mask but do retain the
    # actor-visible incoming-link dictionary.  This fallback still excludes
    # the explicit ``none`` action and never invents a source.
    if not mask.size:
        output = [str(source) for source in incoming if str(source) in set(uav_ids)]
    return sorted(set(output))


def _neighbor_covariance_trace(observation: Mapping[str, Any], source: str) -> float | None:
    messages = observation.get("neighbor_messages", {})
    if not isinstance(messages, Mapping):
        return None
    message = messages.get(source)
    if not isinstance(message, Mapping):
        return None
    covariance = np.asarray(message.get("covariance", []), dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] < 3 or covariance.shape[1] < 3:
        return None
    value = float(np.trace(covariance[:3, :3]))
    return value if np.isfinite(value) else None


def _candidate_values(observation: Mapping[str, Any], source: str) -> dict[str, float] | None:
    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping) or source not in incoming:
        return None
    link = incoming.get(source, {})
    if not isinstance(link, Mapping):
        return None
    values = {
        "rssi_dbm": _finite(link.get("rssi_dbm"), -100.0),
        "snr_db": _finite(link.get("snr_db"), -100.0),
        "packet_loss": _finite(link.get("packet_loss"), 1.0),
        "delay_s": _finite(link.get("delay_s"), 1.0),
        "aoi_s": _finite(link.get("aoi_s"), 60.0),
        "queue_length_packets": _finite(link.get("queue_length_packets"), 1.0),
    }
    covariance = _neighbor_covariance_trace(observation, source)
    values["neighbor_covariance_trace_m2"] = (
        covariance if covariance is not None else float("nan")
    )
    values["_connected"] = float(str(link.get("link_state", "unavailable")) == "connected")
    return values


def _quantile(values: list[float], *, default: tuple[float, float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        q25, q75 = default
    else:
        q25, q75 = np.quantile(finite, (0.25, 0.75))
    return {"q25": float(q25), "q75": float(q75)}


def fit_warmup_quality_model(
    observations: Iterable[Mapping[str, Mapping[str, Any]]],
    uav_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit the frozen soft-quality rule from warm-up observations only."""

    observations = list(observations)
    values: dict[str, list[float]] = {field: [] for field in RISK_FIELDS}
    connected_scores: list[float] = []
    candidate_count = 0
    connected_count = 0
    covariance_count = 0
    for snapshot in observations:
        for local in snapshot.values():
            if not isinstance(local, Mapping):
                continue
            for source in _local_candidates(local, uav_ids):
                row = _candidate_values(local, source)
                if row is None:
                    continue
                candidate_count += 1
                for field in HIGH_IS_GOOD + ("packet_loss", "delay_s", "aoi_s", "queue_length_packets"):
                    values[field].append(float(row[field]))
                covariance = float(row["neighbor_covariance_trace_m2"])
                if np.isfinite(covariance):
                    values["neighbor_covariance_trace_m2"].append(covariance)
                    covariance_count += 1
                if row["_connected"] > 0.5:
                    connected_count += 1

    quantiles = {
        field: _quantile(values[field], default=(0.0, 1.0))
        for field in RISK_FIELDS
    }
    # A missing neighbor covariance is deliberately assigned the worst risk;
    # the placeholder quantiles are only a finite denominator for this case.
    for snapshot in observations:
        for local in snapshot.values():
            if not isinstance(local, Mapping):
                continue
            for source in _local_candidates(local, uav_ids):
                row = _candidate_values(local, source)
                if row is None or row["_connected"] <= 0.5:
                    continue
                components = {
                    field: (
                        _risk(float(row[field]), field, {"quantiles": quantiles})
                        if np.isfinite(float(row[field]))
                        else 1.0
                    )
                    for field in RISK_FIELDS
                }
                connected_scores.append(float(np.mean(list(components.values()))))

    if connected_scores:
        none_threshold = float(np.quantile(np.asarray(connected_scores, dtype=float), 0.90))
    else:
        none_threshold = 1.0
    return {
        "schema_version": "measurement-quality-v1",
        "source_split": "warmup",
        "fit_scope": "actor_visible_observation_snapshots_only",
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "forbidden_inputs": ["true_position", "true_error", "physical_los_nlos", "future_state"],
        "uav_ids": [str(value) for value in uav_ids],
        "candidate_row_count": int(candidate_count),
        "connected_candidate_row_count": int(connected_count),
        "covariance_row_count": int(covariance_count),
        "quantiles": quantiles,
        "soft_score": {
            "weights": {field: 1.0 / len(RISK_FIELDS) for field in RISK_FIELDS},
            "none_threshold_quantile": 0.90,
            "none_threshold": none_threshold,
            "missing_neighbor_covariance_risk": 1.0,
            "selection_rule": "quality is an actor input; the original candidate mask and none action remain unchanged",
        },
    }


def quality_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ID-aligned risk and validity vectors for one local observation."""

    risk = np.ones(len(uav_ids), dtype=np.float32)
    valid = np.zeros(len(uav_ids), dtype=np.float32)
    candidate_set = set(_local_candidates(observation, uav_ids))
    for index, source in enumerate(uav_ids):
        if source not in candidate_set:
            continue
        row = _candidate_values(observation, source)
        if row is None or row["_connected"] <= 0.5:
            continue
        components: list[float] = []
        for field in RISK_FIELDS:
            value = float(row[field])
            components.append(
                _risk(value, field, model) if np.isfinite(value) else 1.0
            )
        risk[index] = float(np.clip(np.mean(components), 0.0, 1.0))
        valid[index] = 1.0
    return risk, valid


def add_quality_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(observation)
    risk, valid = quality_features(observation, uav_ids, model)
    output["measurement_quality_risk"] = risk
    output["measurement_quality_valid"] = valid
    return output


__all__ = [
    "HIGH_IS_GOOD",
    "LOW_IS_GOOD",
    "RISK_FIELDS",
    "add_quality_features",
    "fit_warmup_quality_model",
    "quality_features",
]
