"""Shared experiment metric aggregation rules.

The formal experiment, plotting code, and scale summaries must use exactly
the same communication aggregation.  In particular, radio quality metrics
are defined on candidate edges only; a window without candidate edges uses
the model's explicit worst-case values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_WORST_CASE_DELAY_S = 1.0
DEFAULT_WORST_CASE_AOI_S = 60.0


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _mean(values: Iterable[Any], default: float) -> float:
    numbers = np.asarray([_number(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float(default)


def position_error_metrics(
    ground_truth_rows: Iterable[Mapping[str, Any]],
    estimate_rows: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Return vector-norm position RMSE and P95 for aligned state rows.

    The formal episode summary historically reported a component-wise RMSE,
    while several diagnostic audits used the Euclidean position error.  This
    helper is the canonical diagnostic definition: one scalar error per UAV
    and timestamp, followed by RMSE/P95 over those scalar errors.  It is
    intentionally independent of the simulator and accepts already-read CSV
    rows so validation-only audits cannot accidentally inspect another split.
    """

    def _number(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return number if np.isfinite(number) else float("nan")

    truth: dict[tuple[str, float], np.ndarray] = {}
    for row in ground_truth_rows:
        uav_id = str(row.get("uav_id", ""))
        timestamp = round(_number(row.get("timestamp")), 6)
        position = np.asarray(
            [_number(row.get(axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
        if uav_id and np.isfinite(timestamp) and np.isfinite(position).all():
            truth[(uav_id, timestamp)] = position

    errors: list[float] = []
    for row in estimate_rows:
        key = (
            str(row.get("uav_id", "")),
            round(_number(row.get("timestamp")), 6),
        )
        if key not in truth:
            continue
        estimate = np.asarray(
            [_number(row.get(axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
        if not np.isfinite(estimate).all():
            continue
        error = float(np.linalg.norm(estimate - truth[key]))
        if np.isfinite(error):
            errors.append(error)

    if not errors:
        return {"rmse_norm_m": float("nan"), "p95_norm_m": float("nan"), "sample_count": 0.0}
    values = np.asarray(errors, dtype=float)
    return {
        "rmse_norm_m": float(np.sqrt(np.mean(values**2))),
        "p95_norm_m": float(np.quantile(values, 0.95)),
        "sample_count": float(values.size),
    }


def candidate_link_series(
    links: Iterable[Mapping[str, Any]],
    *,
    worst_case_delay_s: float = DEFAULT_WORST_CASE_DELAY_S,
    worst_case_aoi_s: float = DEFAULT_WORST_CASE_AOI_S,
) -> list[dict[str, float]]:
    """Aggregate packet loss, delay and AoI per decision timestamp.

    Candidate edges are the denominator for all three metrics.  Effective
    edges are intentionally not used here: they describe successful usable
    messages, whereas packet loss and delay describe candidate transmissions.
    """

    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in links:
        timestamp = round(_number(row.get("timestamp"), 0.0), 6)
        grouped[timestamp].append(row)

    series: list[dict[str, float]] = []
    for timestamp in sorted(grouped):
        rows = grouped[timestamp]
        candidate = [
            row for row in rows
            if int(round(_number(row.get("candidate_edge"), 0.0))) == 1
        ]
        if candidate:
            packet_loss = _mean((row.get("packet_loss") for row in candidate), 1.0)
            delay = _mean((row.get("delay") for row in candidate), worst_case_delay_s)
            aoi = _mean((row.get("aoi") for row in candidate), worst_case_aoi_s)
        else:
            packet_loss = 1.0
            delay = float(worst_case_delay_s)
            aoi = float(worst_case_aoi_s)
        series.append(
            {
                "timestamp": float(timestamp),
                "packet_loss": float(np.clip(packet_loss, 0.0, 1.0)),
                "delay_s": max(0.0, float(delay)),
                "aoi_s": max(0.0, float(aoi)),
                "candidate_edge_count": float(len(candidate)),
                "effective_edge_count": float(
                    sum(int(round(_number(row.get("effective_edge"), 0.0))) for row in rows)
                ),
            }
        )
    return series


def aggregate_candidate_link_metrics(
    links: Iterable[Mapping[str, Any]],
    *,
    worst_case_delay_s: float = DEFAULT_WORST_CASE_DELAY_S,
    worst_case_aoi_s: float = DEFAULT_WORST_CASE_AOI_S,
) -> dict[str, float]:
    """Return one episode-level summary using :func:`candidate_link_series`."""

    series = candidate_link_series(
        links,
        worst_case_delay_s=worst_case_delay_s,
        worst_case_aoi_s=worst_case_aoi_s,
    )
    if not series:
        return {
            "packet_loss": 1.0,
            "delay_s": float(worst_case_delay_s),
            "aoi_s": float(worst_case_aoi_s),
            "candidate_edge_rate": 0.0,
            "effective_edge_rate": 0.0,
        }
    return {
        "packet_loss": _mean((row["packet_loss"] for row in series), 1.0),
        "delay_s": _mean((row["delay_s"] for row in series), worst_case_delay_s),
        "aoi_s": _mean((row["aoi_s"] for row in series), worst_case_aoi_s),
        "candidate_edge_rate": _mean((row["candidate_edge_count"] > 0 for row in series), 0.0),
        "effective_edge_rate": _mean((row["effective_edge_count"] for row in series), 0.0),
    }


def metric_clip_fractions(
    rows: Iterable[Mapping[str, Any]],
    scales: Mapping[str, float],
) -> dict[str, float]:
    """Measure how often training-derived p95 scales clip a metric."""

    aliases = {
        "uncertainty": "uncertainty",
        "loss": "packet_loss",
        "aoi": "aoi",
        "delay": "delay",
        "connectivity": "connectivity",
    }
    counts = {key: 0 for key in aliases}
    totals = {key: 0 for key in aliases}
    for row in rows:
        for key, source in aliases.items():
            value = _number(row.get(source))
            scale = max(float(scales.get(key, 1.0)), 1e-12)
            if not np.isfinite(value):
                continue
            totals[key] += 1
            counts[key] += int(value / scale > 1.0)
    return {
        key: float(counts[key] / totals[key]) if totals[key] else 0.0
        for key in aliases
    }


__all__ = [
    "DEFAULT_WORST_CASE_AOI_S",
    "DEFAULT_WORST_CASE_DELAY_S",
    "aggregate_candidate_link_metrics",
    "candidate_link_series",
    "metric_clip_fractions",
    "position_error_metrics",
]
