"""Pure contracts for the isolated partial-GNSS v7 fusion kernel."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.analysis.cooperative_fusion_estimators_v6 import JointEKF, UIDS
from src.analysis.partial_gnss_fusion_v7 import (
    MeasurementCIEKF,
    PartialGNSSJointEKF,
    conditional_linear_error_bound,
)
from src.simulation.ekf import EKFConfig, LocalEKF


def _packets() -> dict[str, dict[str, object]]:
    return {
        uid: {
            "imu": {"accel": np.zeros(3), "gyro": np.zeros(3)},
            "vo": {"valid": False},
            "gnss": {"valid": False},
        }
        for uid in UIDS
    }


def _filter() -> MeasurementCIEKF:
    state = np.zeros(15)
    state[:3] = [0.0, 0.0, 0.0]
    return MeasurementCIEKF(state, EKFConfig())


def test_finite_jacobian_gate_and_native_event_fields() -> None:
    estimator = _filter()
    accepted = estimator.update_relative_range(
        5.1,
        np.array([3.0, 4.0, 0.0]),
        0.01 * np.eye(3),
        measurement_variance=0.01,
        timestamp=0.2,
    )
    assert accepted is True
    event = estimator.last_relative_range_update
    assert np.allclose(np.asarray(event["H"])[0, :3], [-0.6, -0.8, 0.0])
    assert np.allclose(np.asarray(event["H"])[0, 3:], 0.0)
    assert np.isclose(event["p"], 4.0)
    assert np.isclose(event["q"], 0.01)
    assert np.isclose(event["S"], 4.0 + 0.01 + 2.0 * np.sqrt(0.04) + 0.01)
    assert event["accepted"] is True
    assert event["covariance_restored"] is False
    assert event["state_before"] != event["state_after"]
    assert estimator.valid_neighbor_count == 1
    assert estimator.last_neighbor_timestamp == 0.2


@pytest.mark.parametrize("dimension", [3, 15])
def test_random_admissible_reference_covariance_and_conditional_bound(dimension: int) -> None:
    rng = np.random.default_rng(20261012 + dimension)
    estimator = _filter()
    state = rng.normal(size=15)
    state[:3] = [1.0, -2.0, 0.5]
    estimator.state = state
    factor = rng.normal(size=(dimension, dimension))
    reference = 0.002 * (factor @ factor.T) + 0.001 * np.eye(dimension)
    neighbor = state[:3] - np.array([1.5, -0.2, 0.8])
    measurement = float(np.linalg.norm(state[:3] - neighbor))
    assert estimator.update_relative_range(
        measurement,
        neighbor,
        reference,
        measurement_variance=0.1,
    )
    event = estimator.last_relative_range_update
    assert event["conditional_bound_matches_posterior"] is True
    assert np.min(np.linalg.eigvalsh(estimator.covariance)) > 0.0
    assert np.isfinite(estimator.covariance).all()
    assert np.allclose(np.asarray(event["conditional_linear_error_bound"]), estimator.covariance)


def test_ci_formula_grid_and_fixed_gain_identity() -> None:
    estimator = _filter()
    neighbor = np.array([2.0, 0.0, 0.0])
    reference = 0.2 * np.eye(3)
    R = 0.1
    assert estimator.update_relative_range(2.1, neighbor, reference, measurement_variance=R)
    event = estimator.last_relative_range_update
    P = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0] + [1.0] * 9)
    H = np.zeros((1, 15))
    H[0, 0] = 1.0
    q = 0.2
    prior_precision = np.linalg.inv(P)
    candidates = []
    for weight in np.arange(1, 21, dtype=float) / 20.0:
        posterior = np.linalg.inv(weight * prior_precision + (1.0 - weight) * (H.T @ H) / (q + R))
        candidates.append((np.trace(posterior[:3, :3]), weight, posterior))
    best_trace = min(value[0] for value in candidates)
    best_weight = max(value[1] for value in candidates if np.isclose(value[0], best_trace))
    expected = next(value[2] for value in candidates if value[1] == best_weight)
    assert np.isclose(event["weight"], best_weight)
    assert np.allclose(estimator.covariance, expected)
    K = np.asarray(event["gain"])
    bound = conditional_linear_error_bound(P, np.asarray(event["H"]), best_weight, K, q + R)
    assert np.allclose(bound, estimator.covariance)


def test_w1_no_information_gain_is_a_counted_physical_attempt() -> None:
    estimator = _filter()
    state_before = estimator.state.copy()
    covariance_before = estimator.covariance.copy()
    assert estimator.update_relative_range(
        2.0,
        np.array([2.0, 0.0, 0.0]),
        100.0 * np.eye(3),
        measurement_variance=1.0,
    ) is False
    assert estimator.last_relative_range_update["rejection_reason"] == "no_information_gain"
    assert np.allclose(estimator.state, state_before)
    assert np.allclose(estimator.covariance, covariance_before)
    assert estimator.diagnostics_snapshot()["uwb_update_attempted"] == 1
    assert estimator.diagnostics_snapshot()["uwb_update_rejected"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"neighbor_covariance": np.full((3, 3), np.nan), "measurement_variance": 0.1},
        {"neighbor_covariance": np.eye(3), "measurement_variance": 0.0},
        {"neighbor_covariance": -np.eye(3), "measurement_variance": 0.1},
    ],
)
def test_invalid_input_has_no_mutation(kwargs: dict[str, object]) -> None:
    estimator = _filter()
    state_before = estimator.state.copy()
    covariance_before = estimator.covariance.copy()
    last_nis_before = estimator.last_nis
    with pytest.raises((ValueError, FloatingPointError)):
        estimator.update_relative_range(2.0, np.array([2.0, 0.0, 0.0]), **kwargs)
    assert np.allclose(estimator.state, state_before)
    assert np.allclose(estimator.covariance, covariance_before)
    assert estimator.last_nis == last_nis_before
    assert estimator.diagnostics_snapshot()["uwb_update_attempted"] == 0


def test_degenerate_distance_is_rejected_and_counted() -> None:
    estimator = _filter()
    assert estimator.update_relative_range(0.0, np.zeros(3), np.eye(3), measurement_variance=0.1) is False
    assert estimator.last_relative_range_update["rejection_reason"] == "degenerate_predicted_range"
    assert estimator.diagnostics_snapshot()["uwb_update_attempted"] == 1
    assert estimator.diagnostics_snapshot()["uwb_update_rejected"] == 1
    assert estimator.rejected_observation_count == 1


def test_partial_gnss_allows_uav0_uav1_and_denies_uav2() -> None:
    packets = _packets()
    packets["uav_00"]["gnss"] = {"valid": True, "position": np.array([0.1, 0.0, 0.0]), "std": 0.5}
    packets["uav_01"]["gnss"] = {"valid": True, "position": np.array([0.0, 0.1, 0.0]), "std": np.array([0.5, 0.5, 0.5])}
    estimator = PartialGNSSJointEKF({uid: np.zeros(15) for uid in UIDS}, EKFConfig(), initial_cross_position=0.0)
    diagnostics = estimator.predict_and_local_updates(packets, 0.2)
    assert diagnostics["gnss_update_attempted"] == 2
    assert [event["uid"] for event in estimator.gnss_update_events] == ["uav_00", "uav_01"]

    denied = _packets()
    denied["uav_02"]["gnss"] = {"valid": True, "position": np.zeros(3), "std": 0.5}
    state_before = estimator.state.copy()
    covariance_before = estimator.covariance.copy()
    with pytest.raises(ValueError, match="uav_02"):
        estimator.predict_and_local_updates(denied, 0.2)
    assert np.allclose(estimator.state, state_before)
    assert np.allclose(estimator.covariance, covariance_before)


def test_partial_joint_no_gnss_matches_frozen_joint_prediction() -> None:
    packets = _packets()
    states = {uid: np.arange(15, dtype=float) + i for i, uid in enumerate(UIDS)}
    partial = PartialGNSSJointEKF(states, EKFConfig(), initial_cross_position=0.0)
    frozen = JointEKF(states, EKFConfig(), initial_cross_position=0.0)
    partial.predict_and_local_updates(copy.deepcopy(packets), 0.2)
    frozen.predict_and_local_updates(copy.deepcopy(packets), 0.2)
    assert np.allclose(partial.state, frozen.state)
    assert np.allclose(partial.covariance, frozen.covariance)
    assert partial.gnss_update_events == []
