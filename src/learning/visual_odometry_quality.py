"""Leakage-safe calibration of visual-odometry measurement covariance.

The simulator already exposes a bounded, execution-time ``tracking_quality``
value with each valid VO packet.  This module turns warm-up observations into
an immutable, monotone piecewise covariance scale.  It intentionally does not
use ground-truth position, true error, LOS/NLOS labels, or future packet
outcomes.

The calibration is conservative: lower tracking quality cannot receive a
smaller covariance scale than higher tracking quality.  Warm-up innovation
statistics may increase a bin's scale, but never decrease the quality-based
floor or the nominal sensor covariance.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


MODEL_SCHEMA_VERSION = "visual-odometry-quality-v1"
VO_INNOVATION_DIMENSION = 3.0
DEFAULT_BIN_SCALES = (4.0, 2.25, 1.5, 1.0)
DEFAULT_MISSING_QUALITY_SCALE = 4.0


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _quality_bin(quality: float, model: Mapping[str, Any]) -> int:
    edges = np.asarray(model.get("quality_bin_edges", [0.0, 0.25, 0.5, 0.75, 1.0]), dtype=float)
    scales = list(model.get("covariance_scale_by_bin", DEFAULT_BIN_SCALES))
    if edges.ndim != 1 or edges.size < 2 or not scales:
        return 0
    value = float(np.clip(quality, 0.0, 1.0))
    boundaries = edges[1:-1]
    index = int(np.searchsorted(boundaries, value, side="right"))
    return max(0, min(index, len(scales) - 1))


def covariance_scale_for_quality(
    quality: Any,
    model: Mapping[str, Any],
) -> tuple[float, bool, int]:
    """Return ``(scale, quality_valid, bin_index)`` for one VO packet."""

    value = _finite(quality)
    if not np.isfinite(value):
        scale = _finite(model.get("missing_quality_scale"), DEFAULT_MISSING_QUALITY_SCALE)
        return float(max(1.0, scale)), False, -1
    index = _quality_bin(value, model)
    scales = list(model.get("covariance_scale_by_bin", DEFAULT_BIN_SCALES))
    scale = _finite(scales[index] if index < len(scales) else DEFAULT_MISSING_QUALITY_SCALE, 1.0)
    return float(max(1.0, scale)), True, index


def covariance_for_quality(
    base_position_std_m: float,
    quality: Any,
    model: Mapping[str, Any],
) -> tuple[np.ndarray, float, bool, int]:
    """Build a finite isotropic VO covariance from a frozen quality model."""

    base = _finite(base_position_std_m)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base_position_std_m must be finite and positive")
    scale, valid, index = covariance_scale_for_quality(quality, model)
    covariance = np.eye(3, dtype=float) * (base * np.sqrt(scale)) ** 2
    if not np.isfinite(covariance).all():
        raise FloatingPointError("visual-odometry quality covariance became non-finite")
    return covariance, float(scale), bool(valid), int(index)


def _monotone_nonincreasing(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return []
    array = np.where(np.isfinite(array), array, 1.0)
    array = np.maximum(array, 1.0)
    # Bin 0 is the lowest-quality bin, so its scale must be at least every
    # higher-quality bin's scale.
    array = np.maximum.accumulate(array[::-1])[::-1]
    return [float(value) for value in array]


def fit_visual_odometry_quality_model(
    records: Iterable[Mapping[str, Any]],
    *,
    base_position_std_m: float,
) -> dict[str, Any]:
    """Fit an immutable piecewise VO covariance rule from warm-up records.

    ``records`` must contain only warm-up sensor/diagnostic rows.  Innovation
    NIS is an execution-time diagnostic and is used only to make a bin more
    conservative; true position/error fields are deliberately ignored even if
    a caller includes them accidentally.
    """

    base = _finite(base_position_std_m)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base_position_std_m must be finite and positive")
    quality_values: list[float] = []
    rows: list[tuple[float, float]] = []
    for record in records:
        if not isinstance(record, Mapping) or int(_finite(record.get("valid"), 0.0)) != 1:
            continue
        quality = _finite(record.get("tracking_quality"))
        if not np.isfinite(quality):
            continue
        quality = float(np.clip(quality, 0.0, 1.0))
        nis = _finite(record.get("nis_vo"))
        quality_values.append(quality)
        rows.append((quality, nis))

    if quality_values:
        q25, q50, q75 = [float(value) for value in np.quantile(np.asarray(quality_values), (0.25, 0.50, 0.75))]
    else:
        q25, q50, q75 = 0.25, 0.50, 0.75
    edges = [0.0, q25, q50, q75, 1.0]
    counts = [0, 0, 0, 0]
    nis_by_bin: list[list[float]] = [[], [], [], []]
    for quality, nis in rows:
        index = _quality_bin(quality, {"quality_bin_edges": edges, "covariance_scale_by_bin": DEFAULT_BIN_SCALES})
        counts[index] += 1
        if np.isfinite(nis) and nis >= 0.0:
            nis_by_bin[index].append(float(nis))

    quality_reference = max(q75, 0.05)
    quality_floor: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        representative = max(0.05, (float(left) + float(right)) / 2.0)
        quality_floor.append(float(np.clip((quality_reference / representative) ** 2, 1.0, 4.0)))
    nis_scale: list[float] = []
    for values in nis_by_bin:
        if values:
            nis_scale.append(float(np.clip(np.mean(values) / VO_INNOVATION_DIMENSION, 1.0, 4.0)))
        else:
            nis_scale.append(1.0)
    scales = _monotone_nonincreasing(np.maximum(quality_floor, nis_scale))
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "source_split": "warmup",
        "fit_scope": "tracking_quality_and_execution_time_vo_innovation_only",
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "forbidden_inputs": [
            "true_position",
            "true_error",
            "true_range",
            "physical_los_nlos",
            "future_state",
            "future_packet_outcome",
        ],
        "base_position_std_m": float(base),
        "quality_quantiles": {"q25": q25, "q50": q50, "q75": q75},
        "quality_bin_edges": [float(value) for value in edges],
        "covariance_scale_by_bin": scales,
        "missing_quality_scale": float(DEFAULT_MISSING_QUALITY_SCALE),
        "sample_count": int(len(rows)),
        "sample_count_by_bin": [int(value) for value in counts],
        "mean_nis_by_bin": [
            float(np.mean(values)) if values else None for values in nis_by_bin
        ],
        "calibration_rule": (
            "monotone_nonincreasing_quality_floor_max_warmup_nis_scale; "
            "nominal_covariance_is_not_deflated"
        ),
    }


__all__ = [
    "DEFAULT_BIN_SCALES",
    "DEFAULT_MISSING_QUALITY_SCALE",
    "MODEL_SCHEMA_VERSION",
    "covariance_for_quality",
    "covariance_scale_for_quality",
    "fit_visual_odometry_quality_model",
]
