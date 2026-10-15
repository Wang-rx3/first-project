"""Scoped CI weight-grid comparison for the partial-GNSS v10 audit.

This module deliberately keeps the v7 measurement CI implementation as the
single numerical kernel.  The only v10 change is the search grid selected for
one guarded call to :meth:`MeasurementCIEKF.update_relative_range`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from . import partial_gnss_fusion_v7 as v7
from .local_ekf_startup_filter_v8 import StartupLocalEKF
from .partial_gnss_fusion_v7 import MeasurementCIEKF
from .partial_gnss_replay_v9 import StartupMeasurementCI


# Capture the object itself before any wrapper can patch the v7 module.  Keep
# this object untouched: hidden/source-freeze checks rely on identity as well
# as the exact twenty values.
_ORIGINAL_ARRAY = v7._CI_WEIGHTS
if _ORIGINAL_ARRAY.shape != (20,) or not np.allclose(
    _ORIGINAL_ARRAY,
    np.arange(1, 21, dtype=float) / 20.0,
    rtol=0.0,
    atol=1.0e-15,
):
    raise RuntimeError("v7 CI weight grid is not the frozen twenty-point source")

ORIGINAL_WEIGHTS = np.asarray(_ORIGINAL_ARRAY, dtype=float).copy()
ORIGINAL_WEIGHTS.setflags(write=False)

_fine_values = np.linspace(0.05, 1.0, 191, dtype=float)
# Every original grid point must be represented by the exact original float;
# replacing these positions also avoids a decimal-rounding disagreement at
# the endpoints and at the twenty-point anchors.
_fine_values[0::10] = _ORIGINAL_ARRAY
FINE_WEIGHTS = _fine_values
FINE_WEIGHTS.setflags(write=False)

if FINE_WEIGHTS.shape != (191,) or not np.isclose(FINE_WEIGHTS.min(), 0.05) or not np.isclose(
    FINE_WEIGHTS.max(), 1.0
):
    raise RuntimeError("v10 CI fine grid does not have the required bounds")

_CI_PATCH_LOCK = threading.RLock()


def _main_thread_guard() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("v10 CI search wrappers must run on the main thread")


def _search_metadata(grid: np.ndarray, source_kind: str, elapsed_s: float) -> dict[str, Any]:
    return {
        "compute_time_ms": float(max(0.0, elapsed_s) * 1000.0),
        "search_step": float(grid[1] - grid[0]),
        "search_grid_count": int(grid.size),
        "source_kind": str(source_kind),
    }


def _guarded_kernel_call(
    estimator: MeasurementCIEKF,
    grid: np.ndarray,
    source_kind: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """Call the unchanged v7 kernel while temporarily selecting ``grid``."""

    _main_thread_guard()
    with _CI_PATCH_LOCK:
        # Timing starts after lock acquisition and covers the range call only,
        # never an episode/replay or a caller's surrounding work.
        started = time.perf_counter()
        try:
            v7._CI_WEIGHTS = grid
            result = MeasurementCIEKF.update_relative_range(estimator, *args, **kwargs)
        finally:
            # Always restore the captured object, including invalid-input and
            # numerical exceptions raised before v7 creates an event.
            elapsed = time.perf_counter() - started
            v7._CI_WEIGHTS = _ORIGINAL_ARRAY

        estimator.last_ci_compute_ms = float(max(0.0, elapsed) * 1000.0)
        event = getattr(estimator, "last_relative_range_update", None)
        if isinstance(event, dict) and event:
            event.update(_search_metadata(grid, source_kind, elapsed))
        return bool(result)


class RefinedMeasurementCIEKF(MeasurementCIEKF):
    """v7 measurement CI with the scoped 191-point weight search."""

    last_ci_compute_ms: float | None = None

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
        return _guarded_kernel_call(
            self,
            FINE_WEIGHTS,
            "refined",
            (measured_range_m, neighbor_position, neighbor_covariance),
            {
                "measurement_variance": measurement_variance,
                "nlos_bias_m": nlos_bias_m,
                "timestamp": timestamp,
            },
        )


class StartupRefinedCI(StartupLocalEKF, RefinedMeasurementCIEKF):
    """Startup-local filter using the refined v10 measurement-CI search."""


class TimedStartupOriginalCI(StartupMeasurementCI):
    """v9 startup measurement-CI filter with guarded original-grid timing."""

    last_ci_compute_ms: float | None = None

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
        return _guarded_kernel_call(
            self,
            _ORIGINAL_ARRAY,
            "original",
            (measured_range_m, neighbor_position, neighbor_covariance),
            {
                "measurement_variance": measurement_variance,
                "nlos_bias_m": nlos_bias_m,
                "timestamp": timestamp,
            },
        )


METHOD_CLASSES: dict[str, type[StartupLocalEKF]] = {
    "original_ci": TimedStartupOriginalCI,
    "refined_ci": StartupRefinedCI,
}


__all__ = [
    "FINE_WEIGHTS",
    "METHOD_CLASSES",
    "ORIGINAL_WEIGHTS",
    "RefinedMeasurementCIEKF",
    "StartupRefinedCI",
    "TimedStartupOriginalCI",
]
