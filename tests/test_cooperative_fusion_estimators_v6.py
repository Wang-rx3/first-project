"""Pure contracts for the v6 cooperative estimators.

The tests use the estimator classes directly.  They do not construct an
environment, read truth states, train a policy, or write an output artifact.
"""

import numpy as np
import pytest

from src.analysis.cooperative_fusion_estimators_v6 import JointEKF, UIDS, YoungBoundEKF
from src.simulation.ekf import LocalEKF


def _packets(vo: dict | None = None, gnss: dict | None = None) -> dict:
    result = {
        uid: {
            "imu": {"accel": np.zeros(3), "gyro": np.zeros(3)},
            "vo": {"valid": False},
            "gnss": {"valid": False},
        }
        for uid in UIDS
    }
    if vo is not None:
        result["uav_00"]["vo"] = vo
    if gnss is not None:
        result["uav_00"]["gnss"] = gnss
    return result


def _joint(cross: float = 0.0) -> JointEKF:
    return JointEKF({uid: np.zeros(15) for uid in UIDS}, initial_cross_position=cross)


def test_joint_initializes_full_cross_covariance_and_storage() -> None:
    estimator = _joint(cross=2.0)
    assert estimator.covariance.shape == (45, 45)
    assert np.allclose(estimator.position_covariance_9()[:3, 3:6], 2.0 * np.eye(3))
    assert np.min(np.linalg.eigvalsh(estimator.covariance)) >= -1e-8
    assert estimator.matrix_storage_bytes == 45 * 45 * 8


def test_temporal_vo_matches_native_local_update_without_cross_context() -> None:
    vo = {"valid": True, "delta_position": np.array([0.1, 0.0, 0.0]), "covariance": 0.01 * np.eye(3)}
    packet = _packets(vo=vo)
    local = LocalEKF(np.zeros(15))
    local.predict(packet["uav_00"]["imu"], 0.2)
    context = local.last_prediction_context
    assert context is not None
    local.update_visual_delta(
        vo["delta_position"],
        vo["covariance"],
        previous_state=context["previous_state"],
        previous_covariance=context["previous_covariance"],
        transition_matrix=context["transition_matrix"],
        dt_s=0.2,
    )
    joint = _joint(cross=0.0)
    joint.predict_and_local_updates(packet, 0.2)
    assert joint.vo_update_events[0]["accepted"] is True
    assert np.allclose(joint.local_state("uav_00"), local.state)
    assert np.allclose(joint.local_covariance("uav_00"), local.covariance)


def test_translation_of_all_states_does_not_change_vo_update_or_covariance() -> None:
    vo = {"valid": True, "delta_position": np.array([0.1, 0.2, -0.1]), "covariance": 0.01 * np.eye(3)}
    plain = _joint(cross=0.0)
    shifted = JointEKF(
        {uid: np.r_[np.array([5.0, -3.0, 2.0]), np.zeros(12)] for uid in UIDS},
        initial_cross_position=0.0,
    )
    plain.predict_and_local_updates(_packets(vo=vo), 0.2)
    shifted.predict_and_local_updates(_packets(vo=vo), 0.2)
    translation = np.tile(np.r_[np.array([5.0, -3.0, 2.0]), np.zeros(12)], 3)
    assert np.allclose(plain.covariance, shifted.covariance)
    assert np.allclose(shifted.state - plain.state, translation)


def test_range_ids_are_consumed_after_rejection_and_duplicate_is_error() -> None:
    estimator = _joint(cross=0.0)
    estimator.state[0:3] = np.array([1.0, 0.0, 0.0])
    estimator.state[15:18] = np.zeros(3)
    assert estimator.update_range("uav_00", "uav_01", 20.0, 0.01, measurement_id="r1") is False
    with pytest.raises(ValueError, match="duplicate"):
        estimator.update_range("uav_00", "uav_01", 1.0, 0.01, measurement_id="r1")


def test_measurement_information_is_cross_aware_and_rank_zero_safe() -> None:
    estimator = _joint(cross=2.0)
    estimator.state[0] = 1.0
    report = estimator.measurement_information("uav_00", "uav_01", 0.1)
    assert report["scalar_predicted_S"] > 0.1
    assert 0.0 <= report["relative_vector3d_trace_gain_fraction"] <= 1.0
    estimator.state[15:18] = estimator.state[:3]
    zero = estimator.measurement_information("uav_00", "uav_01", 0.1)
    assert zero["scalar_jacobian_rank"] == 0
    assert zero["own_position_trace_gain_fraction"] == 0.0


def test_young_bound_preserves_native_mean_and_has_psd_covariance() -> None:
    base = LocalEKF(np.zeros(15))
    conservative = YoungBoundEKF(np.zeros(15))
    neighbor_covariance = 4.0 * np.eye(3)
    accepted_base = base.update_relative_range(1.2, np.array([1.0, 0.0, 0.0]), neighbor_covariance, measurement_variance=0.01)
    accepted_conservative = conservative.update_relative_range(1.2, np.array([1.0, 0.0, 0.0]), neighbor_covariance, measurement_variance=0.01)
    assert accepted_base is accepted_conservative is True
    assert np.allclose(base.state, conservative.state)
    assert np.linalg.norm(base.state) > 0
    assert np.min(np.linalg.eigvalsh(conservative.covariance)) >= -1e-8
    assert conservative.last_relative_range_update["covariance_restored"] is False
    assert "young_weight" in conservative.last_relative_range_update


@pytest.mark.parametrize("bad", [np.array([np.nan, 0.0, 0.0]), np.array([0.0, np.inf, 0.0])])
def test_nonfinite_observations_are_rejected(bad: np.ndarray) -> None:
    estimator = _joint()
    with pytest.raises(ValueError):
        estimator.predict_and_local_updates(
            _packets(vo={"valid": True, "delta_position": bad}), 0.2
        )


def test_scalar_range_information_preserves_transverse_uncertainty_and_matches_update():
    estimator = _joint(cross=2.0)
    estimator.state[0] = 1.0
    info = estimator.measurement_information("uav_00", "uav_01", .0064)
    posterior = np.asarray(info["relative_posterior_covariance"])
    assert np.allclose(np.linalg.eigvalsh(posterior)[1:], [4., 4.])
    assert info["relative_jacobian_rank"] == 1
    assert np.isclose(np.trace(posterior), 8.006389776357828)
    estimator.update_range("uav_00", "uav_01", 1.0, .0064, measurement_id="actual")
    assert np.allclose(estimator.relative_covariance("uav_00", "uav_01"), posterior)


def test_second_direction_information_exceeds_repeated_direction():
    estimator = _joint(cross=2.0)
    estimator.state[15] = 1.
    estimator.state[31] = 1.
    estimator.update_range("uav_00", "uav_01", 1., .0064, measurement_id="first")
    repeat = estimator.measurement_information("uav_00", "uav_01", .0064)
    other = estimator.measurement_information("uav_00", "uav_02", .0064)
    assert other["own_position_trace_gain_fraction"] > repeat["own_position_trace_gain_fraction"] * 100
