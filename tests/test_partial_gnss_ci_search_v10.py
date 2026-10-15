"""Pure contracts for the scoped partial-GNSS v10 CI search wrappers."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from src.analysis import partial_gnss_fusion_v7 as v7
from src.analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
from src.analysis.partial_gnss_fusion_v7 import MeasurementCIEKF
from src.analysis.partial_gnss_replay_v9 import StartupMeasurementCI
from src.analysis.partial_gnss_ci_search_v10 import (
    FINE_WEIGHTS,
    METHOD_CLASSES,
    ORIGINAL_WEIGHTS,
    RefinedMeasurementCIEKF,
    StartupRefinedCI,
    TimedStartupOriginalCI,
)
from src.simulation.ekf import EKFConfig


def _filter(cls: type[MeasurementCIEKF] = RefinedMeasurementCIEKF) -> MeasurementCIEKF:
    state = np.zeros(15)
    return cls(state, EKFConfig())


def _range_kwargs() -> dict[str, object]:
    return {
        "measured_range_m": 5.1,
        "neighbor_position": np.array([3.0, 4.0, 0.0]),
        "neighbor_covariance": np.eye(3) * 0.01,
        "measurement_variance": 0.01,
    }


def test_frozen_source_grids_and_anchor_nesting() -> None:
    expected = np.arange(1, 21, dtype=float) / 20.0
    np.testing.assert_allclose(ORIGINAL_WEIGHTS, expected, rtol=0.0, atol=1.0e-15)
    assert ORIGINAL_WEIGHTS.shape == (20,)
    assert FINE_WEIGHTS.shape == (191,)
    np.testing.assert_array_equal(FINE_WEIGHTS[0::10], expected)
    assert np.isclose(FINE_WEIGHTS[0], 0.05)
    assert np.isclose(FINE_WEIGHTS[-1], 1.0)
    assert FINE_WEIGHTS.flags.writeable is False
    assert ORIGINAL_WEIGHTS.flags.writeable is False


def test_source_grid_bounds_and_step() -> None:
    assert 0.05 <= float(FINE_WEIGHTS.min()) <= float(FINE_WEIGHTS.max()) <= 1.0
    assert np.isclose(FINE_WEIGHTS[1] - FINE_WEIGHTS[0], 0.005)


def test_refined_event_records_search_metadata_and_parent_fields() -> None:
    estimator = _filter()
    assert estimator.update_relative_range(**_range_kwargs())
    event = estimator.last_relative_range_update
    assert event["source_kind"] == "refined"
    assert event["search_grid_count"] == 191
    assert np.isclose(event["search_step"], 0.005)
    assert np.isfinite(event["compute_time_ms"])
    assert event["accepted"] is True
    assert np.isfinite(event["nis"])
    assert np.isfinite(event["weight"])
    assert estimator.last_ci_compute_ms == event["compute_time_ms"]


def test_original_wrapper_records_twenty_point_metadata() -> None:
    estimator = _filter(TimedStartupOriginalCI)
    assert estimator.update_relative_range(**_range_kwargs())
    event = estimator.last_relative_range_update
    assert event["source_kind"] == "original"
    assert event["search_grid_count"] == 20
    assert np.isclose(event["search_step"], 0.05)
    assert np.isfinite(event["compute_time_ms"])


def test_original_wrapper_matches_v7_numeric_kernel() -> None:
    left = _filter(TimedStartupOriginalCI)
    right = _filter(MeasurementCIEKF)
    kwargs = _range_kwargs()
    assert left.update_relative_range(**kwargs) is right.update_relative_range(**kwargs)
    np.testing.assert_array_equal(left.state, right.state)
    np.testing.assert_array_equal(left.covariance, right.covariance)
    for key in ("accepted", "rejection_reason", "nis", "weight", "p", "q", "S", "H", "gain"):
        assert left.last_relative_range_update[key] == right.last_relative_range_update[key]


def test_refined_objective_is_no_worse_on_a_frozen_pure_state() -> None:
    original = _filter(TimedStartupOriginalCI)
    refined = _filter(RefinedMeasurementCIEKF)
    kwargs = _range_kwargs()
    assert original.update_relative_range(**kwargs)
    assert refined.update_relative_range(**kwargs)
    assert refined.last_relative_range_update["posterior_position_covariance_trace"] <= (
        original.last_relative_range_update["posterior_position_covariance_trace"] + 1.0e-12
    )


def test_w1_rejection_keeps_existing_nan_conditional_fields() -> None:
    estimator = _filter()
    kwargs = {
        "measured_range_m": 2.0,
        "neighbor_position": np.array([2.0, 0.0, 0.0]),
        "neighbor_covariance": np.eye(3) * 100.0,
        "measurement_variance": 1.0,
    }
    assert estimator.update_relative_range(**kwargs) is False
    event = estimator.last_relative_range_update
    assert event["weight"] == 1.0
    assert event["rejection_reason"] == "no_information_gain"
    assert np.isnan(event["conditional_bound_trace_m2"])
    assert np.isnan(event["conditional_bound_residual_norm"])
    assert event["search_grid_count"] == 191


@pytest.mark.parametrize(
    "kwargs",
    [
        {"neighbor_covariance": np.full((3, 3), np.nan), "measurement_variance": 0.1},
        {"neighbor_covariance": np.eye(3), "measurement_variance": 0.0},
        {"neighbor_covariance": -np.eye(3), "measurement_variance": 0.1},
    ],
)
def test_exception_restores_original_array_and_preserves_estimator(kwargs: dict[str, object]) -> None:
    estimator = _filter()
    before_state = estimator.state.copy()
    before_covariance = estimator.covariance.copy()
    with pytest.raises((ValueError, FloatingPointError)):
        estimator.update_relative_range(
            2.0,
            np.array([2.0, 0.0, 0.0]),
            **kwargs,
        )
    assert v7._CI_WEIGHTS is ORIGINAL_ARRAY_SOURCE
    np.testing.assert_array_equal(estimator.state, before_state)
    np.testing.assert_array_equal(estimator.covariance, before_covariance)


def test_degenerate_range_and_invalid_input_restore_source_identity() -> None:
    estimator = _filter()
    assert estimator.update_relative_range(
        0.0,
        np.zeros(3),
        np.eye(3),
        measurement_variance=0.1,
    ) is False
    assert estimator.last_relative_range_update["rejection_reason"] == "degenerate_predicted_range"
    assert v7._CI_WEIGHTS is ORIGINAL_ARRAY_SOURCE


def test_worker_thread_is_rejected_before_global_patch() -> None:
    estimator = _filter()
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            estimator.update_relative_range(**_range_kwargs())
        except BaseException as exc:  # the test asserts the guard's exact class below
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    worker.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert v7._CI_WEIGHTS is ORIGINAL_ARRAY_SOURCE


def test_startup_mros_and_prediction_compatibility() -> None:
    assert StartupRefinedCI.mro()[:5] == [
        StartupRefinedCI,
        StartupLocalEKF,
        RefinedMeasurementCIEKF,
        MeasurementCIEKF,
        StartupLocalEKF.__mro__[-2],
    ]
    assert TimedStartupOriginalCI.mro()[:4] == [
        TimedStartupOriginalCI,
        StartupMeasurementCI,
        StartupLocalEKF,
        MeasurementCIEKF,
    ]
    assert StartupRefinedCI.predict is StartupLocalEKF.predict
    assert TimedStartupOriginalCI.predict is StartupMeasurementCI.predict


def test_method_mapping_is_explicit() -> None:
    assert METHOD_CLASSES == {
        "original_ci": TimedStartupOriginalCI,
        "refined_ci": StartupRefinedCI,
    }


def test_conditional_bound_is_finite_and_matches_posterior() -> None:
    estimator = _filter()
    assert estimator.update_relative_range(**_range_kwargs())
    event = estimator.last_relative_range_update
    bound = np.asarray(event["conditional_linear_error_bound"], dtype=float)
    assert np.isfinite(bound).all()
    assert np.allclose(bound, estimator.covariance)
    assert event["conditional_bound_matches_posterior"] is True
    assert np.min(np.linalg.eigvalsh(estimator.covariance)) >= -1.0e-8


ORIGINAL_ARRAY_SOURCE = v7._CI_WEIGHTS
