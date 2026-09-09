"""Execution-time information-gain and communication-cost proxy features.

The functions in this module are intentionally actor-side only.  They use the
same frozen, training-fitted scales as the read-only uncertainty-proxy audit,
but operate on a live local observation.  No true position, true range,
LOS/NLOS label, post-update NIS, or future packet outcome is consumed.

The proxy is a ranking/teacher signal, not a replacement for the EKF or the
environment reward.  All returned arrays are finite and ID-aligned with the
configured UAV list so they can be stored in a rollout without changing the
episode schema.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1e-9
PROXY_FEATURE_KEYS = (
    "expected_information_gain_norm",
    "predicted_communication_cost_norm",
    "proxy_valid",
)
PERSISTENT_PROXY_FEATURE_KEYS = (
    "persistent_gain_norm",
    "proxy_cost_norm",
    "persistent_gain_valid",
)


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(float(value), lower, upper))


def _uav_sort_key(uav_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(uav_id))
    return (int(match.group(1)) if match else 10**9, str(uav_id))


def _scales(model: Mapping[str, Any]) -> Mapping[str, Any]:
    values = model.get("scales", model)
    return values if isinstance(values, Mapping) else {}


def _scale(model: Mapping[str, Any], key: str, default: float) -> float:
    value = _finite(_scales(model).get(key), default)
    return float(value) if math.isfinite(value) else float(default)


def _normalized(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value) or not math.isfinite(lower) or not math.isfinite(upper):
        return float("nan")
    if upper - lower <= EPS:
        return 0.5
    return _clip((value - lower) / (upper - lower))


def _candidate_sources(observation: Mapping[str, Any], uav_ids: Sequence[str]) -> list[str]:
    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping):
        return []
    mask = np.asarray(observation.get("action_mask", []), dtype=bool).reshape(-1)
    known = {str(source) for source in incoming}
    if mask.size:
        return [
            str(source)
            for index, source in enumerate(uav_ids)
            if index < mask.size and bool(mask[index]) and str(source) in known
        ]
    return sorted((str(source) for source in known if str(source) in set(uav_ids)), key=_uav_sort_key)


def _link_quality(link: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    """Predict link usefulness from fields visible before the action."""

    rssi = _normalized(
        _finite(link.get("rssi_dbm"), _scale(model, "rssi_q05_dbm", -100.0)),
        _scale(model, "rssi_q05_dbm", -100.0),
        _scale(model, "rssi_q95_dbm", -40.0),
    )
    snr = _normalized(
        _finite(link.get("snr_db"), _scale(model, "snr_q05_db", 0.0)),
        _scale(model, "snr_q05_db", 0.0),
        _scale(model, "snr_q95_db", 20.0),
    )
    if not math.isfinite(rssi):
        rssi = 0.0
    if not math.isfinite(snr):
        snr = 0.0
    loss = _clip(_finite(link.get("packet_loss"), 1.0))
    delay_scale = max(_scale(model, "delay_q75_s", 1e-3), EPS)
    age_scale = max(_scale(model, "age_q75_s", 1.0), EPS)
    queue_scale = max(_scale(model, "queue_q75_packets", 1.0), EPS)
    delay = max(_finite(link.get("delay_s"), delay_scale), 0.0)
    age = max(_finite(link.get("aoi_s"), age_scale), 0.0)
    queue = max(_finite(link.get("queue_length_packets"), queue_scale), 0.0)
    state = str(link.get("link_state", "unknown")).strip().lower()
    state_factor = 1.0 if state == "connected" else (0.5 if state in {"degraded", "unknown"} else 0.0)
    return _clip(
        0.5 * (rssi + snr)
        * (1.0 - loss)
        * math.exp(-delay / delay_scale)
        * math.exp(-age / age_scale)
        * math.exp(-queue / queue_scale)
        * state_factor
    )


def _own_covariance_diag(observation: Mapping[str, Any]) -> np.ndarray | None:
    try:
        node = np.asarray(observation.get("node_features", []), dtype=float)
    except (TypeError, ValueError):
        return None
    if node.ndim == 1:
        row = node.reshape(-1)
    elif node.ndim >= 2 and node.shape[0] > 0:
        row = node[0].reshape(-1)
    else:
        return None
    if row.size < 18 or not np.isfinite(row[15:18]).all():
        return None
    return np.maximum(row[15:18], 0.0)


def _neighbor_covariance_diag(observation: Mapping[str, Any], source: str) -> np.ndarray | None:
    messages = observation.get("neighbor_messages", {})
    if not isinstance(messages, Mapping):
        return None
    message = messages.get(source)
    if not isinstance(message, Mapping):
        return None
    try:
        covariance = np.asarray(message.get("covariance", []), dtype=float)
    except (TypeError, ValueError):
        return None
    if covariance.ndim != 2 or covariance.shape[0] < 3 or covariance.shape[1] < 3:
        return None
    diag = np.diag(covariance[:3, :3])
    if not np.isfinite(diag).all():
        return None
    return np.maximum(diag, 0.0)


def _direction_from_effective_edge(observation: Mapping[str, Any], source: str) -> np.ndarray | None:
    """Recover a direction from an already visible effective edge only."""

    try:
        edge_index = np.asarray(observation.get("effective_edge_index", np.empty((2, 0))), dtype=int)
        edge_features = np.asarray(observation.get("effective_edge_features", np.empty((0, 11))), dtype=float)
    except (TypeError, ValueError):
        return None
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_features.ndim != 2:
        return None
    node_ids = [str(value) for value in observation.get("node_ids", [])]
    source_index = -1
    if source in node_ids:
        source_index = node_ids.index(source)
    for offset in range(min(edge_index.shape[1], edge_features.shape[0])):
        if int(edge_index[1, offset]) != 0:
            continue
        if source_index >= 0 and int(edge_index[0, offset]) != source_index:
            continue
        relative = edge_features[offset, :3]
        if relative.size < 3 or not np.isfinite(relative).all():
            continue
        norm = float(np.linalg.norm(relative))
        if norm > EPS:
            return -relative / norm
    return None


def _measurement_std(observation: Mapping[str, Any], source: str, model: Mapping[str, Any]) -> float:
    for key in ("measurement_std_by_source", "uwb_measurement_std_by_source"):
        values = observation.get(key, {})
        if isinstance(values, Mapping) and source in values:
            value = _finite(values.get(source))
            if math.isfinite(value) and value > 0.0:
                return value
    for key in ("uwb_measurement_std_m", "measurement_std_m"):
        value = _finite(observation.get(key))
        if math.isfinite(value) and value > 0.0:
            return value
    return max(_scale(model, "measurement_std_median", 0.5), EPS)


def _raw_information_gain(
    observation: Mapping[str, Any],
    source: str,
    link: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[float, float, bool]:
    own_cov = _own_covariance_diag(observation)
    if own_cov is None:
        return float("nan"), float("nan"), False
    neighbor_cov = _neighbor_covariance_diag(observation, source)
    if neighbor_cov is None:
        trace_scale = max(_scale(model, "neighbor_covariance_trace_q95_m2", 3.0), EPS)
        neighbor_cov = np.full(3, trace_scale / 3.0, dtype=float)
    direction = _direction_from_effective_edge(observation, source)
    if direction is None:
        # The current local observation does not expose a non-effective
        # neighbor geometry estimate.  The same deterministic isotropic
        # fallback is used by the offline audit; it is finite but less
        # informative than an arrived neighbor estimate.
        direction = np.full(3, 1.0 / math.sqrt(3.0), dtype=float)
    if not np.isfinite(direction).all():
        return float("nan"), float("nan"), False
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= EPS:
        return float("nan"), float("nan"), False
    direction = direction / direction_norm
    sigma = _measurement_std(observation, source, model)
    age = max(_finite(link.get("aoi_s"), 0.0), 0.0)
    age_coefficient = max(_finite(model.get("age_variance_coefficient"), 1.0), 0.0)
    innovation_variance = float(
        np.dot(direction, (own_cov + neighbor_cov) * direction)
        + sigma**2
        + age_coefficient * age**2
    )
    numerator = float(np.dot(own_cov * direction, own_cov * direction))
    if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
        return float("nan"), float("nan"), False
    quality = _link_quality(link, model)
    raw = max(quality, 0.0) * numerator / (innovation_variance + EPS)
    ig_norm = _normalized(
        raw,
        _scale(model, "ig_q05", 0.0),
        _scale(model, "ig_q95", 1.0),
    )
    valid = bool(math.isfinite(raw) and math.isfinite(ig_norm))
    return float(raw), float(ig_norm), valid


def information_gain_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gain, communication cost and validity vectors aligned by UAV ID."""

    count = len(uav_ids)
    gain = np.zeros(count, dtype=np.float32)
    cost = np.ones(count, dtype=np.float32)
    valid = np.zeros(count, dtype=np.float32)
    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping):
        return gain, cost, valid
    uwb_available = observation.get("uwb_available", {})
    candidate_sources = _candidate_sources(observation, uav_ids)
    for index, source in enumerate(uav_ids):
        if source not in candidate_sources:
            continue
        link = incoming.get(source, {})
        if not isinstance(link, Mapping):
            continue
        if isinstance(uwb_available, Mapping) and source in uwb_available and not bool(uwb_available[source]):
            continue
        q = _link_quality(link, model)
        _raw, ig_norm, is_valid = _raw_information_gain(observation, source, link, model)
        cost[index] = float(np.clip(1.0 - q, 0.0, 1.0))
        if is_valid:
            gain[index] = float(np.clip(ig_norm, 0.0, 1.0))
            valid[index] = 1.0
    return gain, cost, valid


def measurement_communication_cost_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
    *,
    fixed_packet_cost_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate communication costs and validity aligned by UAV ID.

    Every selectable measurement has a fixed packet component and a visible
    link-risk component. The explicit ``none`` action is not part of these
    candidate vectors and therefore has cost zero at the policy layer.
    """

    fixed = float(fixed_packet_cost_fraction)
    if not math.isfinite(fixed) or not 0.0 <= fixed <= 1.0:
        raise ValueError("fixed_packet_cost_fraction must be finite and in [0, 1]")
    _gain, link_risk, _proxy_valid = information_gain_features(observation, uav_ids, model)
    action_mask = np.asarray(observation.get("action_mask", []), dtype=bool).reshape(-1)
    valid = np.zeros(len(uav_ids), dtype=np.float32)
    for index in range(len(uav_ids)):
        if index < action_mask.size and bool(action_mask[index]):
            valid[index] = 1.0
    from .measurement_cost import candidate_communication_cost
    cost = candidate_communication_cost(np.clip(link_risk, 0.0, 1.0), fixed)
    return np.asarray(cost, dtype=np.float32), valid


def add_information_gain_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy a local observation and attach the three proxy feature vectors."""

    output = dict(observation)
    gain, cost, valid = information_gain_features(observation, uav_ids, model)
    output["expected_information_gain_norm"] = gain
    output["predicted_communication_cost_norm"] = cost
    output["proxy_valid"] = valid
    return output


def _array_or_none(value: Any, length: int) -> np.ndarray | None:
    """Return a finite vector of the requested length, or ``None``."""

    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < length or not np.isfinite(array[:length]).all():
        return None
    return array[:length].astype(float, copy=True)


def _model_parameter(model: Mapping[str, Any], key: str, default: float) -> float:
    value = _finite(model.get(key), default)
    if math.isfinite(value):
        return float(value)
    return float(default)


def _persistent_direction(
    observation: Mapping[str, Any],
    source: str,
    own_state: np.ndarray | None,
    neighbor_state: np.ndarray | None,
) -> np.ndarray | None:
    """Use estimated positions or an already visible effective edge only."""

    if own_state is not None and neighbor_state is not None:
        relative = neighbor_state[:3] - own_state[:3]
        norm = float(np.linalg.norm(relative))
        if norm > EPS and np.isfinite(relative).all():
            return relative / norm
    return _direction_from_effective_edge(observation, source)


def _persistent_raw_from_components(
    *,
    own_state: np.ndarray | None,
    own_cov_diag: np.ndarray | None,
    neighbor_state: np.ndarray | None,
    neighbor_cov_diag: np.ndarray | None,
    direction: np.ndarray | None,
    link: Mapping[str, Any],
    measurement_std: float,
    age: float,
    model: Mapping[str, Any],
    horizon: int,
) -> tuple[float, float, bool]:
    """Compute a finite, execution-time three-step covariance benefit.

    The state is an estimated position/velocity state and the covariance is a
    diagonal estimate.  This deliberately does not call the EKF or mutate its
    state; it is a policy-side proxy only.
    """

    if own_state is None or own_cov_diag is None:
        return float("nan"), float("nan"), False
    if direction is None:
        return float("nan"), float("nan"), False
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if direction.size < 3 or not np.isfinite(direction[:3]).all():
        return float("nan"), float("nan"), False
    direction = direction[:3]
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        return float("nan"), float("nan"), False
    direction = direction / norm

    own_cov_diag = np.maximum(np.asarray(own_cov_diag, dtype=float).reshape(-1)[:6], 0.0)
    if own_cov_diag.size < 6 or not np.isfinite(own_cov_diag).all():
        return float("nan"), float("nan"), False
    if neighbor_cov_diag is None:
        fallback_trace = max(
            _scale(model, "neighbor_covariance_trace_q95_m2", 3.0), EPS
        )
        neighbor_cov = np.full(3, fallback_trace / 3.0, dtype=float)
    else:
        neighbor_cov = np.asarray(neighbor_cov_diag, dtype=float).reshape(-1)
        if neighbor_cov.size < 3 or not np.isfinite(neighbor_cov[:3]).all():
            fallback_trace = max(
                _scale(model, "neighbor_covariance_trace_q95_m2", 3.0), EPS
            )
            neighbor_cov = np.full(3, fallback_trace / 3.0, dtype=float)
        else:
            neighbor_cov = np.maximum(neighbor_cov[:3], 0.0)

    sigma = max(float(measurement_std), EPS)
    age = max(float(age) if math.isfinite(float(age)) else 0.0, 0.0)
    age_coefficient = max(_model_parameter(model, "age_variance_coefficient", 1.0), 0.0)
    innovation_variance = float(
        np.dot(direction, (own_cov_diag[:3] + neighbor_cov) * direction)
        + sigma**2
        + age_coefficient * age**2
    )
    if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
        return float("nan"), float("nan"), False

    # A scalar relative-range update only observes position along ``direction``.
    h = np.concatenate([direction, np.zeros(3, dtype=float)])
    p_before = np.diag(own_cov_diag)
    ph = p_before @ h
    p_linear = p_before - np.outer(ph, ph) / (innovation_variance + EPS)
    p_linear = 0.5 * (p_linear + p_linear.T)

    dt = max(_model_parameter(model, "decision_dt_s", 0.2), EPS)
    position_std = max(_model_parameter(model, "process_position_std_m", 0.08), EPS)
    velocity_std = max(_model_parameter(model, "process_velocity_std_mps", 0.15), EPS)
    transition = np.eye(6, dtype=float)
    transition[:3, 3:] = np.eye(3, dtype=float) * dt
    process = np.diag(
        np.concatenate([
            np.full(3, position_std**2, dtype=float),
            np.full(3, velocity_std**2, dtype=float),
        ])
    )
    p0_h = p_before.copy()
    p1_h = p_linear.copy()
    for _ in range(max(1, int(horizon))):
        p0_h = transition @ p0_h @ transition.T + process
        p1_h = transition @ p1_h @ transition.T + process
    # The persistent proxy is deliberately aligned with the audited
    # ``Delta P_linear`` target: it is the propagated trace reduction after a
    # fixed horizon.  Communication quality is kept separate as the cost
    # block below, so a good link cannot silently inflate the information
    # gain itself or double-count the communication term in the teacher.
    gain = float(np.trace(p0_h[:3, :3]) - np.trace(p1_h[:3, :3]))
    gain = max(gain, 0.0)
    quality = _link_quality(link, model)
    raw = gain
    return float(raw), float(quality), bool(math.isfinite(raw) and math.isfinite(quality))


def _persistent_live_state_covariance(
    observation: Mapping[str, Any], source: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Read the current own/neighbor estimated state and diagonal covariance."""

    own_state: np.ndarray | None = None
    own_cov: np.ndarray | None = None
    try:
        nodes = np.asarray(observation.get("node_features", []), dtype=float)
    except (TypeError, ValueError):
        nodes = np.empty((0, 0), dtype=float)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    if nodes.ndim >= 2 and nodes.shape[0] > 0:
        own_state = _array_or_none(nodes[0], 6)
        own_cov = _array_or_none(nodes[0][15:21], 6)

    neighbor_state: np.ndarray | None = None
    neighbor_cov: np.ndarray | None = None
    messages = observation.get("neighbor_messages", {})
    message = messages.get(source) if isinstance(messages, Mapping) else None
    if isinstance(message, Mapping):
        neighbor_state = _array_or_none(
            message.get("estimate", message.get("state", [])), 6
        )
        try:
            covariance = np.asarray(message.get("covariance", []), dtype=float)
        except (TypeError, ValueError):
            covariance = np.empty((0, 0), dtype=float)
        if covariance.ndim == 2 and covariance.shape[0] >= 6 and covariance.shape[1] >= 6:
            diagonal = np.diag(covariance[:6, :6])
            if np.isfinite(diagonal).all():
                neighbor_cov = np.maximum(diagonal, 0.0)

    # Some graph observations expose the latest received estimate as a node
    # row even when the message dictionary is compact.  It is still a current
    # estimate, not truth or a future state.
    node_ids = [str(value) for value in observation.get("node_ids", [])]
    if (neighbor_state is None or neighbor_cov is None) and source in node_ids and nodes.ndim >= 2:
        index = node_ids.index(source)
        if 0 <= index < nodes.shape[0]:
            if neighbor_state is None:
                neighbor_state = _array_or_none(nodes[index], 6)
            if neighbor_cov is None:
                neighbor_cov = _array_or_none(nodes[index][15:21], 6)
    return own_state, own_cov, neighbor_state, neighbor_cov


def _persistent_row_link(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rssi_dbm": row.get("rssi_dbm"),
        "snr_db": row.get("snr_db"),
        "packet_loss": row.get("packet_loss_pred"),
        "delay_s": row.get("delay_pred_s"),
        "aoi_s": row.get("aoi_pred_s"),
        "queue_length_packets": row.get("queue_length_pred"),
        "link_state": row.get("link_state", "unknown"),
    }


def persistent_information_gain_from_row(
    row: Mapping[str, Any], model: Mapping[str, Any], *, horizon: int | None = None
) -> tuple[float, float, bool]:
    """Materialize the persistent proxy from an offline actor-visible row."""

    own_state = _array_or_none(row.get("own_state_6"), 6)
    own_cov = _array_or_none(row.get("own_cov_diag_6"), 6)
    neighbor_state = _array_or_none(row.get("neighbor_state_6"), 6)
    neighbor_cov = _array_or_none(row.get("neighbor_cov_diag_6"), 6)
    direction = _array_or_none(row.get("u_hat"), 3)
    raw, _quality, valid = _persistent_raw_from_components(
        own_state=own_state,
        own_cov_diag=own_cov,
        neighbor_state=neighbor_state,
        neighbor_cov_diag=neighbor_cov,
        direction=direction,
        link=_persistent_row_link(row),
        measurement_std=max(_finite(row.get("measurement_std_m"), _scale(model, "measurement_std_median", 0.5)), EPS),
        age=max(_finite(row.get("message_age_s"), 0.0), 0.0),
        model=model,
        horizon=max(1, int(horizon if horizon is not None else _model_parameter(model, "persistent_horizon", 3))),
    )
    lower = _finite(model.get("persistent_ig_q05"), _scale(model, "ig_q05", 0.0))
    upper = _finite(model.get("persistent_ig_q95"), _scale(model, "ig_q95", 1.0))
    return raw, _normalized(raw, lower, upper), valid


def persistent_information_gain_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
    *,
    horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-horizon gain, cost, and validity vectors for live input."""

    count = len(uav_ids)
    gain = np.zeros(count, dtype=np.float32)
    cost = np.ones(count, dtype=np.float32)
    valid = np.zeros(count, dtype=np.float32)
    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping):
        return gain, cost, valid
    candidate_sources = _candidate_sources(observation, uav_ids)
    uwb_available = observation.get("uwb_available", {})
    own_state, own_cov, _unused_state, _unused_cov = _persistent_live_state_covariance(observation, "")
    for index, source in enumerate(uav_ids):
        if source not in candidate_sources:
            continue
        if isinstance(uwb_available, Mapping) and source in uwb_available and not bool(uwb_available[source]):
            continue
        link = incoming.get(source, {})
        if not isinstance(link, Mapping):
            continue
        _own_state, _own_cov, neighbor_state, neighbor_cov = _persistent_live_state_covariance(observation, source)
        direction = _persistent_direction(observation, source, own_state, neighbor_state)
        raw, quality, is_valid = _persistent_raw_from_components(
            own_state=own_state if own_state is not None else _own_state,
            own_cov_diag=own_cov if own_cov is not None else _own_cov,
            neighbor_state=neighbor_state,
            neighbor_cov_diag=neighbor_cov,
            direction=direction,
            link=link,
            measurement_std=_measurement_std(observation, source, model),
            age=max(_finite(link.get("aoi_s"), 0.0), 0.0),
            model=model,
            horizon=max(1, int(horizon if horizon is not None else _model_parameter(model, "persistent_horizon", 3))),
        )
        lower = _finite(model.get("persistent_ig_q05"), _scale(model, "ig_q05", 0.0))
        upper = _finite(model.get("persistent_ig_q95"), _scale(model, "ig_q95", 1.0))
        normalized = _normalized(raw, lower, upper)
        if is_valid and math.isfinite(normalized):
            gain[index] = float(np.clip(normalized, 0.0, 1.0))
            valid[index] = 1.0
        cost[index] = float(np.clip(1.0 - quality, 0.0, 1.0)) if math.isfinite(quality) else 1.0
    return gain, cost, valid


def add_persistent_information_gain_features(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
    *,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Attach persistent proxy blocks without changing the episode schema."""

    output = dict(observation)
    gain, cost, valid = persistent_information_gain_features(
        observation, uav_ids, model, horizon=horizon
    )
    utility = gain - cost
    output["persistent_information_gain_norm"] = gain
    output["persistent_communication_cost_norm"] = cost
    output["persistent_proxy_valid"] = valid
    # Short names are kept as the public action-head contract requested by the
    # experiment plan.  ``proxy_utility`` is diagnostic/teacher-only.
    output["persistent_gain_norm"] = gain
    output["proxy_cost_norm"] = cost
    output["persistent_gain_valid"] = valid
    output["proxy_utility"] = utility.astype(np.float32)
    return output


def fit_persistent_proxy_model(
    rows: Sequence[Mapping[str, Any]],
    base_model: Mapping[str, Any],
    *,
    horizon: int = 3,
    decision_dt_s: float = 0.2,
    process_position_std_m: float = 0.08,
    process_velocity_std_mps: float = 0.15,
    age_variance_coefficient: float = 1.0,
) -> dict[str, Any]:
    """Fit only the persistent raw-gain quantiles from training rows."""

    if int(horizon) < 1:
        raise ValueError("persistent horizon must be positive")
    model = {str(key): value for key, value in dict(base_model).items()}
    scales = dict(model.get("scales", {})) if isinstance(model.get("scales", {}), Mapping) else {}
    model.update({
        "persistent_horizon": int(horizon),
        "decision_dt_s": float(decision_dt_s),
        "process_position_std_m": float(process_position_std_m),
        "process_velocity_std_mps": float(process_velocity_std_mps),
        "age_variance_coefficient": float(age_variance_coefficient),
    })
    preliminary = dict(model)
    preliminary["persistent_ig_q05"] = 0.0
    preliminary["persistent_ig_q95"] = 1.0
    raw_values = [
        persistent_information_gain_from_row(row, preliminary, horizon=horizon)[0]
        for row in rows
    ]
    finite = np.asarray([value for value in raw_values if math.isfinite(value)], dtype=float)
    q05 = float(np.quantile(finite, 0.05)) if finite.size else 0.0
    q95 = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    if not math.isfinite(q05):
        q05 = 0.0
    if not math.isfinite(q95):
        q95 = max(q05 + EPS, 1.0)
    if q95 < q05:
        q05, q95 = q95, q05
    model["persistent_ig_q05"] = q05
    model["persistent_ig_q95"] = q95
    # Store the same values in scales for transparent calibration inspection.
    scales["persistent_ig_q05"] = q05
    scales["persistent_ig_q95"] = q95
    model["scales"] = scales
    return model


__all__ = [
    "EPS",
    "PROXY_FEATURE_KEYS",
    "PERSISTENT_PROXY_FEATURE_KEYS",
    "add_information_gain_features",
    "add_persistent_information_gain_features",
    "fit_persistent_proxy_model",
    "information_gain_features",
    "measurement_communication_cost_features",
    "persistent_information_gain_features",
    "persistent_information_gain_from_row",
]
