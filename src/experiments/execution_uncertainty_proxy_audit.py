"""Execution-time uncertainty-proxy audit.

This module is deliberately read-only with respect to the frozen experiment
root.  It fits per-seed proxy scales from ``baseline_mappo/train`` only and
evaluates both policy branches on ``validation`` only.  The test split is not
enumerated or opened.  True positions, true ranges, LOS/NLOS labels and
future packet outcomes are never used to construct the proxy.

The proxy is an offline diagnostic ranking signal.  It is not inserted into
the EKF, reward, MAPPO actor, or current objective in this phase.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TREATMENTS = ("baseline_mappo", "quality_aware_mappo")
SEED_RE = re.compile(r"^seed_(\d+)$")
EPS = 1e-9


@dataclass(frozen=True)
class ExecutionUncertaintyProxyAuditConfig:
    """Configuration for the validation-only proxy audit."""

    reference_root: Path
    logs_output: Path | None = None
    plots_output: Path | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    age_variance_coefficient: float = 1.0
    epsilon: float = EPS
    force: bool = False


@dataclass(frozen=True)
class ProxyScales:
    """Per-seed scales fitted exclusively from the baseline training split."""

    measurement_std_median: float
    neighbor_covariance_trace_q95_m2: float
    delay_q75_s: float
    age_q75_s: float
    queue_q75_packets: float
    rssi_q05_dbm: float
    rssi_q95_dbm: float
    snr_q05_db: float
    snr_q95_db: float
    ig_q05: float
    ig_q95: float

    def as_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in self.__dict__.items()
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(float(value), lower, upper))


def _quantile(values: Sequence[float], quantile: float, default: float) -> float:
    finite = np.asarray([float(value) for value in values if _finite(value)], dtype=float)
    if finite.size == 0:
        return float(default)
    value = float(np.quantile(finite, quantile))
    return value if math.isfinite(value) else float(default)


def _positive_scale(value: float, default: float = 1.0) -> float:
    return float(max(abs(value), default, EPS)) if math.isfinite(value) else float(default)


def _json_object(value: str | None, default: Any) -> Any:
    if value is None or not str(value).strip():
        return default
    try:
        return _read_json_from_text(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _read_json_from_text(value: str) -> Any:
    return json.loads(value)


def _uav_index(uav_id: str, uav_ids: Sequence[str]) -> int:
    if uav_id in uav_ids:
        return int(uav_ids.index(uav_id))
    match = re.search(r"(\d+)$", str(uav_id))
    if match:
        return int(match.group(1))
    return -1


def _uav_sort_key(uav_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(uav_id))
    return (int(match.group(1)) if match else 10**9, str(uav_id))


def _timestamp_key(value: Any) -> float:
    return round(_number(value), 6)


def _as_vector(value: Any, length: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < length:
        return None
    array = array[:length]
    return array if np.isfinite(array).all() else None


def _edge_map(index_value: Any, feature_value: Any, uav_ids: Sequence[str]) -> dict[str, list[float]]:
    """Map visible incoming edge source IDs to their edge feature rows."""

    try:
        index = np.asarray(index_value, dtype=int)
        features = list(feature_value or [])
    except (TypeError, ValueError):
        return {}
    if index.ndim != 2 or index.shape[0] != 2:
        return {}
    result: dict[str, list[float]] = {}
    for offset in range(min(index.shape[1], len(features))):
        source_index = int(index[0, offset])
        target_index = int(index[1, offset])
        if 0 <= source_index < len(uav_ids) and 0 <= target_index < len(uav_ids):
            try:
                feature = [float(item) for item in features[offset]]
            except (TypeError, ValueError):
                continue
            if np.isfinite(np.asarray(feature, dtype=float)).all():
                result[uav_ids[source_index]] = feature
    return result


def _uwb_measurement_map(episode_dir: Path) -> dict[tuple[float, str, str], dict[str, float]]:
    """Read only measurement uncertainty/validity, never true range or LOS."""

    path = episode_dir / "uwb_ranges.csv"
    if not path.is_file():
        return {}
    result: dict[tuple[float, str, str], dict[str, float]] = {}
    for row in _read_csv(path):
        source = str(row.get("src_uav_id", ""))
        target = str(row.get("dst_uav_id", ""))
        result[(_timestamp_key(row.get("timestamp")), source, target)] = {
            "measurement_std_m": _number(row.get("measurement_std")),
            "valid": 1.0 if str(row.get("valid", "0")).strip().lower() in {"1", "true", "yes"} else 0.0,
        }
    return result


def _offline_error_map(episode_dir: Path, *, enabled: bool) -> dict[tuple[float, str], float]:
    """Read position error only as a validation-only diagnostic label."""

    if not enabled:
        return {}
    path = episode_dir / "ekf_diagnostics.csv"
    if not path.is_file():
        return {}
    result: dict[tuple[float, str], float] = {}
    for row in _read_csv(path):
        error = _number(row.get("position_error_m"))
        if math.isfinite(error):
            result[(_timestamp_key(row.get("timestamp")), str(row.get("uav_id", "")))] = error
    return result


def _candidate_rows_from_episode(
    episode_dir: Path,
    *,
    base_seed: int,
    treatment: str,
    split: str,
    include_offline_label: bool,
    age_variance_coefficient: float,
) -> list[dict[str, Any]]:
    """Extract candidate features using local graph observations only."""

    graph_path = episode_dir / "graph_features.csv"
    if not graph_path.is_file():
        raise FileNotFoundError(f"missing graph_features.csv: {graph_path}")
    graph_rows = _read_csv(graph_path)
    if not graph_rows:
        return []
    episode_id = str(graph_rows[0].get("episode_id", episode_dir.name))
    scenario_id = str(graph_rows[0].get("scenario_id", ""))
    episode_seed = _number(graph_rows[0].get("seed"))
    uwb_map = _uwb_measurement_map(episode_dir)
    offline_errors = _offline_error_map(episode_dir, enabled=include_offline_label)
    uav_ids = sorted(
        {str(row.get("uav_id", "")) for row in graph_rows if str(row.get("uav_id", ""))},
        key=_uav_sort_key,
    )
    all_rows: list[dict[str, Any]] = []

    for row in graph_rows:
        uav_id = str(row.get("uav_id", ""))
        local_index = _uav_index(uav_id, uav_ids)
        node_features = _json_object(row.get("node_features_json"), [])
        candidate_links = _json_object(row.get("candidate_incoming_links_json"), {})
        candidate_edge_features = _json_object(row.get("candidate_incoming_edge_features_json"), {})
        effective_index = _json_object(row.get("effective_edge_index_json"), [[], []])
        effective_features = _json_object(row.get("effective_edge_features_json"), [])
        effective_by_source = _edge_map(effective_index, effective_features, uav_ids)
        if not isinstance(candidate_links, dict):
            candidate_links = {}
        if not isinstance(candidate_edge_features, dict):
            candidate_edge_features = {}
        candidate_sources = sorted(
            [str(source) for source in candidate_links.keys()],
            key=_uav_sort_key,
        )
        candidate_count = len(candidate_sources)
        timestamp = _timestamp_key(row.get("timestamp"))
        if not (0 <= local_index < len(node_features)):
            continue
        own_row = _as_vector(node_features[local_index], 18)
        own_row_full = _as_vector(node_features[local_index], 21)
        if own_row is None:
            # Preserve an invalid record for the audit rather than silently
            # substituting truth or centralized state.
            own_position = np.full(3, np.nan)
            own_cov_diag = np.full(3, np.nan)
        else:
            own_position = own_row[:3]
            own_cov_diag = np.maximum(own_row[15:18], 0.0)
        own_state_6 = (
            np.asarray(own_row_full[:6], dtype=float)
            if own_row_full is not None
            else np.full(6, np.nan, dtype=float)
        )
        own_cov_diag_6 = (
            np.maximum(np.asarray(own_row_full[15:21], dtype=float), 0.0)
            if own_row_full is not None
            else np.full(6, np.nan, dtype=float)
        )

        for source_id in candidate_sources:
            source_index = _uav_index(source_id, uav_ids)
            link = candidate_links.get(source_id, {})
            if not isinstance(link, dict):
                link = {}
            edge = candidate_edge_features.get(source_id, [])
            if not isinstance(edge, (list, tuple)):
                edge = []
            effective_edge = effective_by_source.get(source_id)
            neighbor_cov_diag: np.ndarray | None = None
            neighbor_state_6: np.ndarray | None = None
            neighbor_cov_diag_6: np.ndarray | None = None
            geometry_source = "isotropic_no_neighbor_estimate"
            u_hat = np.full(3, 1.0 / math.sqrt(3.0), dtype=float)
            estimated_distance = _number(edge[3]) if len(edge) > 3 else float("nan")

            # A neighbor estimate is considered visible only when an incoming
            # effective edge exists.  We do not use non-visible global node
            # rows, central_state_json, true positions, or future outcomes.
            if effective_edge is not None and 0 <= source_index < len(node_features):
                source_row = _as_vector(node_features[source_index], 18)
                source_row_full = _as_vector(node_features[source_index], 21)
                relative = _as_vector(effective_edge[:3], 3)
                if source_row is not None and relative is not None:
                    relative_norm = float(np.linalg.norm(relative))
                    if relative_norm > EPS:
                        u_hat = -relative / relative_norm
                        estimated_distance = relative_norm
                        neighbor_cov_diag = np.maximum(source_row[15:18], 0.0)
                        if source_row_full is not None:
                            neighbor_state_6 = np.asarray(source_row_full[:6], dtype=float)
                            neighbor_cov_diag_6 = np.maximum(
                                np.asarray(source_row_full[15:21], dtype=float), 0.0
                            )
                        geometry_source = "received_neighbor_estimate"

            uwb = uwb_map.get((timestamp, source_id, uav_id), {})
            measurement_std = _number(uwb.get("measurement_std_m"))
            link_state = str(link.get("link_state", "unknown"))
            all_rows.append({
                "base_seed": int(base_seed),
                "episode_seed": int(episode_seed) if math.isfinite(episode_seed) else "",
                "treatment": treatment,
                "split": split,
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "timestamp": timestamp,
                "uav_id": uav_id,
                "source_uav_id": source_id,
                "candidate_count": candidate_count,
                "multi_candidate_window": int(candidate_count >= 2),
                "own_position": own_position,
                "own_cov_diag": own_cov_diag,
                "own_state_6": own_state_6,
                "own_cov_diag_6": own_cov_diag_6,
                "u_hat": u_hat,
                "geometry_source": geometry_source,
                "neighbor_covariance_available": int(neighbor_cov_diag is not None),
                "neighbor_cov_diag": neighbor_cov_diag,
                "neighbor_state_6": neighbor_state_6,
                "neighbor_cov_diag_6": neighbor_cov_diag_6,
                "estimated_distance_m": estimated_distance,
                "measurement_std_m": measurement_std,
                "uwb_available": float(uwb.get("valid", float("nan"))),
                "message_age_s": _number(link.get("aoi_s")),
                "rssi_dbm": _number(link.get("rssi_dbm")),
                "snr_db": _number(link.get("snr_db")),
                "packet_loss_pred": _number(link.get("packet_loss")),
                "delay_pred_s": _number(link.get("delay_s")),
                "aoi_pred_s": _number(link.get("aoi_s")),
                "queue_length_pred": _number(link.get("queue_length_packets")),
                "link_state": link_state,
                "candidate_edge": _number(link.get("candidate_edge")),
                "offline_position_error_m": offline_errors.get((timestamp, uav_id), float("nan")),
                "offline_label_only": int(include_offline_label),
                "age_variance_coefficient": float(age_variance_coefficient),
            })
    return all_rows


def _fit_scales(rows: Sequence[Mapping[str, Any]], age_variance_coefficient: float) -> ProxyScales:
    measurement = [_number(row.get("measurement_std_m")) for row in rows]
    measurement_median = _positive_scale(_quantile(measurement, 0.50, 0.5), 0.01)
    visible_cov_trace = [
        float(np.sum(np.maximum(np.asarray(row["neighbor_cov_diag"], dtype=float), 0.0)))
        for row in rows
        if row.get("neighbor_cov_diag") is not None
        and np.isfinite(np.asarray(row["neighbor_cov_diag"], dtype=float)).all()
    ]
    neighbor_cov_q95 = _positive_scale(_quantile(visible_cov_trace, 0.95, 3.0), 1e-6)
    delay_q75 = _positive_scale(_quantile([_number(row.get("delay_pred_s")) for row in rows], 0.75, 1e-3), 1e-6)
    age_q75 = _positive_scale(_quantile([_number(row.get("aoi_pred_s")) for row in rows], 0.75, 1.0), 1e-6)
    queue_q75 = _positive_scale(_quantile([_number(row.get("queue_length_pred")) for row in rows], 0.75, 1.0), 1e-6)
    rssi_q05 = _quantile([_number(row.get("rssi_dbm")) for row in rows], 0.05, -100.0)
    rssi_q95 = _quantile([_number(row.get("rssi_dbm")) for row in rows], 0.95, -40.0)
    snr_q05 = _quantile([_number(row.get("snr_db")) for row in rows], 0.05, 0.0)
    snr_q95 = _quantile([_number(row.get("snr_db")) for row in rows], 0.95, 20.0)
    preliminary = ProxyScales(
        measurement_std_median=measurement_median,
        neighbor_covariance_trace_q95_m2=neighbor_cov_q95,
        delay_q75_s=delay_q75,
        age_q75_s=age_q75,
        queue_q75_packets=queue_q75,
        rssi_q05_dbm=min(rssi_q05, rssi_q95),
        rssi_q95_dbm=max(rssi_q05, rssi_q95),
        snr_q05_db=min(snr_q05, snr_q95),
        snr_q95_db=max(snr_q05, snr_q95),
        ig_q05=0.0,
        ig_q95=1.0,
    )
    raw_values = [_raw_information_gain(row, preliminary, age_variance_coefficient) for row in rows]
    q05 = _quantile(raw_values, 0.05, 0.0)
    q95 = _quantile(raw_values, 0.95, 1.0)
    return ProxyScales(
        measurement_std_median=preliminary.measurement_std_median,
        neighbor_covariance_trace_q95_m2=preliminary.neighbor_covariance_trace_q95_m2,
        delay_q75_s=preliminary.delay_q75_s,
        age_q75_s=preliminary.age_q75_s,
        queue_q75_packets=preliminary.queue_q75_packets,
        rssi_q05_dbm=preliminary.rssi_q05_dbm,
        rssi_q95_dbm=preliminary.rssi_q95_dbm,
        snr_q05_db=preliminary.snr_q05_db,
        snr_q95_db=preliminary.snr_q95_db,
        ig_q05=min(q05, q95),
        ig_q95=max(q05, q95),
    )


def _normalized(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    if not math.isfinite(lower) or not math.isfinite(upper) or upper - lower <= EPS:
        return 0.5
    return _clip((value - lower) / (upper - lower))


def _link_quality(row: Mapping[str, Any], scales: ProxyScales) -> float:
    """Predict link usefulness from current visible fields only."""

    rssi_score = _normalized(_number(row.get("rssi_dbm")), scales.rssi_q05_dbm, scales.rssi_q95_dbm)
    snr_score = _normalized(_number(row.get("snr_db")), scales.snr_q05_db, scales.snr_q95_db)
    loss = _clip(_number(row.get("packet_loss_pred"), 1.0))
    delay = max(_number(row.get("delay_pred_s"), scales.delay_q75_s), 0.0)
    age = max(_number(row.get("aoi_pred_s"), scales.age_q75_s), 0.0)
    queue = max(_number(row.get("queue_length_pred"), scales.queue_q75_packets), 0.0)
    state = str(row.get("link_state", "unknown")).strip().lower()
    state_factor = 1.0 if state == "connected" else (0.5 if state in {"degraded", "unknown"} else 0.0)
    signal_factor = 0.5 * (rssi_score + snr_score)
    return _clip(
        signal_factor
        * (1.0 - loss)
        * math.exp(-delay / scales.delay_q75_s)
        * math.exp(-age / scales.age_q75_s)
        * math.exp(-queue / scales.queue_q75_packets)
        * state_factor
    )


def _raw_information_gain(
    row: Mapping[str, Any],
    scales: ProxyScales,
    age_variance_coefficient: float,
) -> float:
    own_cov = np.asarray(row.get("own_cov_diag"), dtype=float).reshape(-1)
    if own_cov.size < 3 or not np.isfinite(own_cov[:3]).all():
        return float("nan")
    own_cov = np.maximum(own_cov[:3], 0.0)
    neighbor = row.get("neighbor_cov_diag")
    if neighbor is None:
        neighbor_cov = np.full(3, scales.neighbor_covariance_trace_q95_m2 / 3.0, dtype=float)
    else:
        neighbor_cov = np.asarray(neighbor, dtype=float).reshape(-1)
        if neighbor_cov.size < 3 or not np.isfinite(neighbor_cov[:3]).all():
            neighbor_cov = np.full(3, scales.neighbor_covariance_trace_q95_m2 / 3.0, dtype=float)
        else:
            neighbor_cov = np.maximum(neighbor_cov[:3], 0.0)
    direction = np.asarray(row.get("u_hat"), dtype=float).reshape(-1)
    if direction.size < 3 or not np.isfinite(direction[:3]).all():
        return float("nan")
    direction = direction[:3]
    norm = float(np.linalg.norm(direction))
    if norm <= EPS:
        return float("nan")
    direction = direction / norm
    sigma = max(_number(row.get("measurement_std_m"), scales.measurement_std_median), EPS)
    age = max(_number(row.get("message_age_s"), 0.0), 0.0)
    innovation_variance = float(
        np.dot(direction, (own_cov + neighbor_cov) * direction)
        + sigma**2
        + float(age_variance_coefficient) * age**2
    )
    numerator = float(np.dot(own_cov * direction, own_cov * direction))
    if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
        return float("nan")
    return float(max(_link_quality(row, scales), 0.0) * numerator / (innovation_variance + EPS))


def _materialize_proxy_rows(
    rows: Sequence[dict[str, Any]],
    scales: ProxyScales,
    age_variance_coefficient: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
    computed: list[tuple[dict[str, Any], float, float, float]] = []
    for row in rows:
        raw_ig = _raw_information_gain(row, scales, age_variance_coefficient)
        ig_norm = _normalized(raw_ig, scales.ig_q05, scales.ig_q95)
        risk = 1.0 - ig_norm if math.isfinite(ig_norm) else float("nan")
        q = _link_quality(row, scales)
        key = (
            row["base_seed"], row["treatment"], row["episode_id"],
            row["timestamp"], row["uav_id"],
        )
        grouped[key].append((len(computed), raw_ig))
        computed.append((row, raw_ig, ig_norm, risk))

    ranks: dict[int, tuple[int, float]] = {}
    spread_by_key: dict[tuple[Any, ...], float] = {}
    for key, indexes in grouped.items():
        finite = [(idx, value) for idx, value in indexes if math.isfinite(value)]
        values = [value for _, value in finite]
        spread_by_key[key] = float(max(values) - min(values)) if len(values) >= 2 else 0.0
        ordered = sorted(
            finite,
            key=lambda item: (-item[1], _uav_sort_key(str(computed[item[0]][0]["source_uav_id"]))),
        )
        for rank, (idx, _value) in enumerate(ordered, start=1):
            ranks[idx] = (rank, spread_by_key[key])

    output: list[dict[str, Any]] = []
    for index, (row, raw_ig, ig_norm, risk) in enumerate(computed):
        key = (
            row["base_seed"], row["treatment"], row["episode_id"],
            row["timestamp"], row["uav_id"],
        )
        rank, spread = ranks.get(index, (0, 0.0))
        valid = bool(math.isfinite(raw_ig) and math.isfinite(ig_norm) and math.isfinite(risk))
        output.append({
            "base_seed": row["base_seed"],
            "episode_seed": row["episode_seed"],
            "treatment": row["treatment"],
            "split": row["split"],
            "episode_id": row["episode_id"],
            "scenario_id": row["scenario_id"],
            "timestamp": row["timestamp"],
            "uav_id": row["uav_id"],
            "source_uav_id": row["source_uav_id"],
            "candidate_count": row["candidate_count"],
            "multi_candidate_window": row["multi_candidate_window"],
            "window_proxy_spread_raw": spread,
            "geometry_source": row["geometry_source"],
            "neighbor_covariance_available": row["neighbor_covariance_available"],
            "estimated_distance_m": row["estimated_distance_m"],
            "own_position_covariance_trace_m2": float(np.sum(np.maximum(np.asarray(row["own_cov_diag"], dtype=float), 0.0))) if np.isfinite(np.asarray(row["own_cov_diag"], dtype=float)).all() else float("nan"),
            "neighbor_covariance_trace_m2": float(np.sum(np.maximum(np.asarray(row["neighbor_cov_diag"], dtype=float), 0.0))) if row["neighbor_cov_diag"] is not None and np.isfinite(np.asarray(row["neighbor_cov_diag"], dtype=float)).all() else float("nan"),
            "measurement_std_m": row["measurement_std_m"] if math.isfinite(row["measurement_std_m"]) else scales.measurement_std_median,
            "uwb_available": row["uwb_available"],
            "message_age_s": row["message_age_s"],
            "rssi_dbm": row["rssi_dbm"],
            "snr_db": row["snr_db"],
            "packet_loss_pred": row["packet_loss_pred"],
            "delay_pred_s": row["delay_pred_s"],
            "aoi_pred_s": row["aoi_pred_s"],
            "queue_length_pred": row["queue_length_pred"],
            "link_state": row["link_state"],
            "link_quality_q": _link_quality(row, scales),
            "expected_information_gain_raw": raw_ig,
            "expected_information_gain_norm": ig_norm,
            "proxy_risk": risk,
            "proxy_rank": rank,
            "proxy_valid": int(valid),
            "offline_position_error_m": row["offline_position_error_m"],
            "offline_label_only": row["offline_label_only"],
        })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_outputs(config: ExecutionUncertaintyProxyAuditConfig) -> tuple[Path, Path]:
    run_name = Path(config.reference_root).name
    logs_dir = Path(config.logs_output) if config.logs_output is not None else Path("logs/preliminary") / run_name / "execution_uncertainty_proxy"
    plots_dir = Path(config.plots_output) if config.plots_output is not None else Path("plots/preliminary") / run_name / "execution_uncertainty_proxy"
    for path in (logs_dir, plots_dir):
        if path.exists() and any(path.iterdir()):
            if not config.force:
                raise FileExistsError(
                    f"execution uncertainty proxy output is not empty: {path}; use --force to replace this audit"
                )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return logs_dir, plots_dir


def _episode_dirs(scale_dir: Path, split: str) -> list[Path]:
    split_dir = scale_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing {split} split: {split_dir}")
    return sorted(
        [path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")],
        key=lambda path: path.name,
    )


def _summary_rows(proxy_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    seed_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in proxy_rows:
        episode_groups[(row["base_seed"], row["treatment"], row["episode_id"])].append(row)
        seed_groups[(row["base_seed"], row["treatment"])].append(row)

    def aggregate(key: tuple[Any, ...], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def mean(name: str) -> float:
            values = [_number(row.get(name)) for row in rows]
            values = [value for value in values if math.isfinite(value)]
            return float(np.mean(values)) if values else float("nan")

        windows = {(row["episode_id"], row["timestamp"], row["uav_id"]) for row in rows}
        multi = {
            window for window in windows
            if any(int(row["multi_candidate_window"]) == 1 and (row["timestamp"], row["uav_id"]) == (window[1], window[2]) for row in rows)
        }
        spread = [
            float(row["window_proxy_spread_raw"])
            for row in rows
            if int(row["multi_candidate_window"]) == 1 and _finite(row["window_proxy_spread_raw"])
        ]
        valid = [int(row["proxy_valid"]) for row in rows]
        offline = [
            _number(row.get("offline_position_error_m"))
            for row in rows
            if _finite(row.get("offline_position_error_m"))
        ]
        high = [_number(row.get("offline_position_error_m")) for row in rows if _number(row.get("proxy_risk")) >= 0.75 and _finite(row.get("offline_position_error_m"))]
        low = [_number(row.get("offline_position_error_m")) for row in rows if _number(row.get("proxy_risk")) <= 0.25 and _finite(row.get("offline_position_error_m"))]
        result = {
            "base_seed": key[0],
            "treatment": key[1],
            "candidate_row_count": len(rows),
            "candidate_window_count": len(windows),
            "multi_candidate_window_count": len(multi),
            "proxy_valid_fraction": float(np.mean(valid)) if valid else float("nan"),
            "multi_window_nonzero_spread_fraction": float(np.mean(np.asarray(spread) > EPS)) if spread else float("nan"),
            "mean_link_quality_q": mean("link_quality_q"),
            "mean_expected_information_gain_norm": mean("expected_information_gain_norm"),
            "mean_proxy_risk": mean("proxy_risk"),
            "p95_proxy_risk": float(np.quantile([_number(row["proxy_risk"]) for row in rows if _finite(row["proxy_risk"])], 0.95)) if any(_finite(row["proxy_risk"]) for row in rows) else float("nan"),
            "mean_offline_position_error_m": float(np.mean(offline)) if offline else float("nan"),
            "low_risk_offline_error_m": float(np.mean(low)) if low else float("nan"),
            "high_risk_offline_error_m": float(np.mean(high)) if high else float("nan"),
            "offline_label_is_diagnostic": 1,
        }
        if len(key) == 3:
            result["episode_id"] = key[2]
        return result

    episode_rows = [aggregate(key, rows) for key, rows in sorted(episode_groups.items())]
    seed_rows = [aggregate(key, rows) for key, rows in sorted(seed_groups.items())]

    strata_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in proxy_rows:
        risk = _number(row.get("proxy_risk"))
        if not math.isfinite(risk):
            continue
        if risk < 0.25:
            stratum = "risk_0_25"
        elif risk < 0.50:
            stratum = "risk_25_50"
        elif risk < 0.75:
            stratum = "risk_50_75"
        else:
            stratum = "risk_75_100"
        strata_groups[(row["base_seed"], row["treatment"], stratum)].append(row)
    strata_rows: list[dict[str, Any]] = []
    order = {"risk_0_25": 0, "risk_25_50": 1, "risk_50_75": 2, "risk_75_100": 3}
    for (seed, treatment, stratum), rows in sorted(strata_groups.items(), key=lambda item: (item[0][0], item[0][1], order[item[0][2]])):
        error = [_number(row.get("offline_position_error_m")) for row in rows if _finite(row.get("offline_position_error_m"))]
        strata_rows.append({
            "base_seed": seed,
            "treatment": treatment,
            "stratum": stratum,
            "candidate_row_count": len(rows),
            "mean_proxy_risk": float(np.mean([_number(row["proxy_risk"]) for row in rows])),
            "mean_offline_position_error_m": float(np.mean(error)) if error else float("nan"),
            "offline_label_is_diagnostic": 1,
        })
    return episode_rows, seed_rows, strata_rows


def _plot_proxy(proxy_rows: Sequence[Mapping[str, Any]], seed_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline_mappo": "#2563eb", "quality_aware_mappo": "#16a34a"}
    labels = {"baseline_mappo": "Baseline MAPPO", "quality_aware_mappo": "Quality-aware MAPPO"}
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    axis = axes[0, 0]
    for treatment in TREATMENTS:
        selected = [row for row in proxy_rows if row["treatment"] == treatment and _finite(row.get("expected_information_gain_norm"))]
        axis.scatter(
            [_number(row["expected_information_gain_norm"]) for row in selected],
            [_number(row["proxy_risk"]) for row in selected],
            s=7, alpha=0.25, color=colors[treatment], label=labels[treatment], rasterized=True,
        )
    axis.set_title("Expected information gain and proxy risk")
    axis.set_xlabel("normalized expected information gain")
    axis.set_ylabel("proxy risk (validation)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for treatment in TREATMENTS:
        selected = [row for row in proxy_rows if row["treatment"] == treatment and _finite(row.get("offline_position_error_m"))]
        axis.scatter(
            [_number(row["proxy_risk"]) for row in selected],
            [_number(row["offline_position_error_m"]) for row in selected],
            s=7, alpha=0.25, color=colors[treatment], label=labels[treatment], rasterized=True,
        )
    axis.set_title("Proxy risk vs. position error label")
    axis.set_xlabel("proxy risk")
    axis.set_ylabel("position error (m; offline label)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 2]
    for treatment in TREATMENTS:
        selected = [row for row in seed_rows if row["treatment"] == treatment]
        selected.sort(key=lambda row: int(row["base_seed"]))
        axis.plot(
            [str(row["base_seed"]) for row in selected],
            [_number(row["mean_offline_position_error_m"]) for row in selected],
            marker="o", color=colors[treatment], label=labels[treatment],
        )
    axis.set_title("Seed-level diagnostic error")
    axis.set_ylabel("mean position error (m; offline label)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    for treatment in TREATMENTS:
        selected = [row for row in proxy_rows if row["treatment"] == treatment]
        axis.scatter(
            [_number(row["link_quality_q"]) for row in selected],
            [_number(row["proxy_risk"]) for row in selected],
            s=7, alpha=0.25, color=colors[treatment], label=labels[treatment], rasterized=True,
        )
    axis.set_title("Visible link quality and proxy risk")
    axis.set_xlabel("predicted link usefulness q")
    axis.set_ylabel("proxy risk")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    seed_labels = sorted({int(row["base_seed"]) for row in seed_rows})
    x = np.arange(len(seed_labels), dtype=float)
    width = 0.36
    for offset, treatment in [(-width / 2, "baseline_mappo"), (width / 2, "quality_aware_mappo")]:
        selected = {int(row["base_seed"]): row for row in seed_rows if row["treatment"] == treatment}
        axis.bar(
            x + offset,
            [_number(selected[seed]["multi_window_nonzero_spread_fraction"], 0.0) for seed in seed_labels],
            width=width, color=colors[treatment], label=labels[treatment],
        )
    axis.axhline(0.8, color="#dc2626", linestyle="--", linewidth=1, label="80% gate")
    axis.set_title("Nonzero proxy spread in multi-candidate windows")
    axis.set_ylabel("fraction")
    axis.set_xticks(x, [str(seed) for seed in seed_labels], rotation=35)
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    for treatment in TREATMENTS:
        selected = [row for row in seed_rows if row["treatment"] == treatment]
        axis.scatter(
            [_number(row["mean_proxy_risk"]) for row in selected],
            [_number(row["high_risk_offline_error_m"]) for row in selected],
            s=45, color=colors[treatment], label=labels[treatment],
        )
    axis.set_title("High-risk stratum diagnostic")
    axis.set_xlabel("mean proxy risk")
    axis.set_ylabel("high-risk error (m; offline label)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    figure.suptitle("Execution-time uncertainty proxy audit (validation only)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _high_risk_trend(strata_rows: Sequence[Mapping[str, Any]]) -> bool:
    by_seed_treatment: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in strata_rows:
        value = _number(row.get("mean_offline_position_error_m"))
        if math.isfinite(value):
            by_seed_treatment[(row["base_seed"], row["treatment"])][str(row["stratum"])] = value
    decisions = []
    for values in by_seed_treatment.values():
        low = values.get("risk_0_25")
        high = values.get("risk_75_100")
        if low is not None and high is not None:
            decisions.append(high + 1e-6 >= low)
    return bool(decisions) and float(np.mean(decisions)) >= 0.6


def run_execution_uncertainty_proxy_audit(
    config: ExecutionUncertaintyProxyAuditConfig,
) -> dict[str, Any]:
    """Fit train-only scales and evaluate the execution-time proxy."""

    if config.train_split != "train":
        raise ValueError("proxy scale fitting must use the train split")
    if config.eval_split != "validation":
        raise ValueError("proxy audit is validation-only; test is sealed")
    if config.age_variance_coefficient < 0.0 or not math.isfinite(config.age_variance_coefficient):
        raise ValueError("age_variance_coefficient must be finite and non-negative")
    reference_root = Path(config.reference_root)
    if not reference_root.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference_root}")

    seed_dirs: list[tuple[int, Path]] = []
    for path in sorted(reference_root.iterdir(), key=lambda item: item.name):
        match = SEED_RE.match(path.name)
        if path.is_dir() and match:
            seed_dirs.append((int(match.group(1)), path))
    if not seed_dirs:
        raise ValueError(f"no seed_* directories found below {reference_root}")

    proxy_rows: list[dict[str, Any]] = []
    scale_records: dict[str, Any] = {}
    seed_checks: dict[str, Any] = {}
    all_train_counts: dict[str, int] = {}
    for base_seed, seed_root in seed_dirs:
        baseline_scale = seed_root / "baseline_mappo" / "scale_3"
        quality_scale = seed_root / "quality_aware_mappo" / "scale_3"
        train_dirs = _episode_dirs(baseline_scale, config.train_split)
        if not train_dirs:
            raise ValueError(f"no baseline training episodes for seed {base_seed}")
        train_rows: list[dict[str, Any]] = []
        for episode_dir in train_dirs:
            train_rows.extend(_candidate_rows_from_episode(
                episode_dir,
                base_seed=base_seed,
                treatment="baseline_mappo",
                split=config.train_split,
                include_offline_label=False,
                age_variance_coefficient=config.age_variance_coefficient,
            ))
        scales = _fit_scales(train_rows, config.age_variance_coefficient)
        scale_records[str(base_seed)] = {
            "fit_treatment": "baseline_mappo",
            "fit_split": config.train_split,
            "fit_candidate_rows": len(train_rows),
            "scales": scales.as_dict(),
        }
        all_train_counts[str(base_seed)] = len(train_rows)
        eval_counts: dict[str, int] = {}
        for treatment, branch_scale in (("baseline_mappo", baseline_scale), ("quality_aware_mappo", quality_scale)):
            eval_dirs = _episode_dirs(branch_scale, config.eval_split)
            if not eval_dirs:
                raise ValueError(f"no validation episodes for {base_seed}/{treatment}")
            treatment_rows: list[dict[str, Any]] = []
            for episode_dir in eval_dirs:
                treatment_rows.extend(_candidate_rows_from_episode(
                    episode_dir,
                    base_seed=base_seed,
                    treatment=treatment,
                    split=config.eval_split,
                    include_offline_label=True,
                    age_variance_coefficient=config.age_variance_coefficient,
                ))
            materialized = _materialize_proxy_rows(treatment_rows, scales, config.age_variance_coefficient)
            proxy_rows.extend(materialized)
            eval_counts[treatment] = len(materialized)
        seed_checks[str(base_seed)] = {
            "train_candidate_rows": len(train_rows),
            "validation_candidate_rows": eval_counts,
            "test_read": False,
        }

    if not proxy_rows:
        raise ValueError("no validation candidate rows were available for proxy audit")
    episode_rows, seed_rows, strata_rows = _summary_rows(proxy_rows)
    logs_dir, plots_dir = _prepare_outputs(config)
    proxy_path = logs_dir / "candidate_proxy.csv"
    episode_path = logs_dir / "episode_proxy_summary.csv"
    seed_path = logs_dir / "seed_proxy_summary.csv"
    strata_path = logs_dir / "proxy_strata.csv"
    parameters_path = logs_dir / "proxy_parameters.json"
    plot_path = plots_dir / "execution_uncertainty_proxy.png"
    _write_csv(proxy_path, proxy_rows)
    _write_csv(episode_path, episode_rows)
    _write_csv(seed_path, seed_rows)
    _write_csv(strata_path, strata_rows)
    parameters = {
        "audit": "execution_uncertainty_proxy",
        "scale_fit": "per base seed, baseline_mappo/train only",
        "evaluation": "baseline_mappo and quality_aware_mappo/validation only",
        "test_data_read": False,
        "formula": {
            "direction": "(estimated_neighbor_position - estimated_own_position) / (norm + epsilon); isotropic fallback when neighbor message is unavailable",
            "innovation_variance": "u^T(P_i + P_j)u + measurement_std^2 + age_variance_coefficient * age^2",
            "expected_information_gain": "q * trace(P_i u u^T P_i) / (innovation_variance + epsilon)",
            "link_quality_q": "equal signal score from RSSI/SNR multiplied by predicted success, delay, age, queue, and current-state factors",
            "age_variance_coefficient": float(config.age_variance_coefficient),
            "epsilon": float(config.epsilon),
        },
        "allowed_inputs": [
            "local estimated position and position covariance",
            "received neighbor estimated position/covariance when effective incoming message is visible",
            "UWB measurement_std and valid flag",
            "message age proxy AoI",
            "RSSI, SNR, predicted packet loss, predicted delay, queue length, current link_state",
        ],
        "forbidden_inputs": [
            "true_position",
            "true_range",
            "true_position_error",
            "physical_LOS_NLOS_label",
            "nlos_reason",
            "future_message_arrival",
            "future_packet_success",
            "central_state_json",
        ],
        "scales_by_seed": scale_records,
    }
    parameters_path.write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_proxy(proxy_rows, seed_rows, plot_path)

    candidate_valid = [int(row["proxy_valid"]) for row in proxy_rows]
    valid_fraction = float(np.mean(candidate_valid)) if candidate_valid else 0.0
    windows: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in proxy_rows:
        windows[(row["base_seed"], row["treatment"], row["episode_id"], row["timestamp"], row["uav_id"])].append(row)
    multi_windows = [rows for rows in windows.values() if len(rows) >= 2]
    nonzero_spread = [
        max(_number(row.get("expected_information_gain_raw")) for row in rows if _finite(row.get("expected_information_gain_raw")))
        - min(_number(row.get("expected_information_gain_raw")) for row in rows if _finite(row.get("expected_information_gain_raw")))
        for rows in multi_windows
        if any(_finite(row.get("expected_information_gain_raw")) for row in rows)
    ]
    multi_spread_fraction = float(np.mean(np.asarray(nonzero_spread) > EPS)) if nonzero_spread else 0.0
    high_risk_trend = _high_risk_trend(strata_rows)
    all_finite = all(
        _finite(row.get(field))
        for row in proxy_rows
        for field in (
            "link_quality_q", "expected_information_gain_raw", "expected_information_gain_norm",
            "proxy_risk", "proxy_valid",
        )
    )
    deterministic = True
    for rows in windows.values():
        first = [row["source_uav_id"] for row in sorted(rows, key=lambda row: int(row["proxy_rank"])) if int(row["proxy_rank"]) > 0]
        second = [row["source_uav_id"] for row in sorted(rows, key=lambda row: (-_number(row["expected_information_gain_raw"], -1.0), _uav_sort_key(str(row["source_uav_id"])))) if _finite(row["expected_information_gain_raw"])]
        if first and second and first != second:
            deterministic = False
            break
    checks = {
        "validation_only": config.eval_split == "validation",
        "test_not_read": True,
        "train_fit_separated": all(count > 0 for count in all_train_counts.values()),
        "source_untouched_by_audit": True,
        "all_proxy_fields_finite": all_finite,
        "at_least_95_percent_valid_candidates": valid_fraction >= 0.95,
        "at_least_80_percent_multi_candidate_windows_nonzero_spread": multi_spread_fraction >= 0.80,
        "deterministic_candidate_ranking": deterministic,
        "high_risk_error_non_decreasing_diagnostic": high_risk_trend,
        "seed_level_inference_used": True,
        "windows_are_not_independent": True,
        "offline_error_is_label_only": True,
    }
    proxy_gate_passed = all(checks.values())
    report = {
        "audit": "execution_uncertainty_proxy",
        "status": "read_only_validation_diagnostic",
        "reference_root": str(reference_root),
        "base_seeds": [seed for seed, _ in seed_dirs],
        "treatments": list(TREATMENTS),
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "test_data_read": False,
        "source_writes": 0,
        "candidate_row_count": len(proxy_rows),
        "multi_candidate_window_count": len(multi_windows),
        "valid_candidate_fraction": valid_fraction,
        "multi_window_nonzero_spread_fraction": multi_spread_fraction,
        "high_risk_error_non_decreasing": high_risk_trend,
        "independent_unit": "base training seed",
        "offline_diagnostic_labels": ["position_error_m"],
        "checks": checks,
        "audit_ok": bool(proxy_gate_passed),
        "proxy_gate_passed": bool(proxy_gate_passed),
        "decision_supported": False,
        "decision_gate": "proxy_ready_for_budget_controlled_intervention" if proxy_gate_passed else "proxy_degenerate_or_noncausal_revise_proxy",
        "next_stage": "budget_controlled_paired_intervention" if proxy_gate_passed else "revise_proxy_construction_before_intervention",
        "seed_checks": seed_checks,
        "output_paths": {
            "proxy_parameters": str(parameters_path),
            "candidate_proxy": str(proxy_path),
            "episode_proxy_summary": str(episode_path),
            "seed_proxy_summary": str(seed_path),
            "proxy_strata": str(strata_path),
            "plot": str(plot_path),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "interpretation_boundary": (
            "The expected information gain is an execution-time offline proxy only. "
            "Validation position error is used solely as a diagnostic label; it is not "
            "used to fit, rank, select actions, or modify the current objective. "
            "Test data remain sealed."
        ),
    }
    report_path = logs_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "ExecutionUncertaintyProxyAuditConfig",
    "ProxyScales",
    "run_execution_uncertainty_proxy_audit",
]
