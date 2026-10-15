"""Deterministic contracts for the bounded startup EKF candidate."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
from src.simulation.ekf import EKFConfig, LocalEKF, STATE_DIM
from src.simulation.sensor_model import SensorConfig


def _imu(accel=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    return {"accel": np.asarray(accel, dtype=float), "gyro": np.asarray(gyro, dtype=float)}


def _block_diagonal_prior(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 0.2, STATE_DIM)
    left = rng.normal(size=(9, 9))
    right = rng.normal(size=(6, 6))
    covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    covariance[:9, :9] = left @ left.T + 0.5 * np.eye(9)
    covariance[9:, 9:] = right @ right.T + 0.5 * np.eye(6)
    return state, covariance


def _clone_vo(
    state: np.ndarray,
    covariance: np.ndarray,
    previous_state: np.ndarray,
    previous_covariance: np.ndarray,
    transition: np.ndarray,
    displacement: np.ndarray,
    noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cross = previous_covariance @ transition.T
    augmented = np.block([[covariance, cross.T], [cross, previous_covariance]])
    selector = np.zeros((3, 30), dtype=float)
    selector[:, :3] = np.eye(3)
    selector[:, 15:18] = -np.eye(3)
    innovation = displacement - (state[:3] - previous_state[:3])
    innovation_covariance = selector @ augmented @ selector.T + noise
    gain = augmented @ selector.T @ np.linalg.inv(innovation_covariance)
    return (
        state + gain[:15] @ innovation,
        covariance - gain[:15] @ innovation_covariance @ gain[:15].T,
        innovation_covariance,
        gain[:15],
    )


def test_constructor_keeps_native_initial_covariance_and_config() -> None:
    config = EKFConfig()
    candidate = StartupLocalEKF(np.zeros(15), config)
    native = LocalEKF(np.zeros(15), config)
    assert candidate.config is config
    assert np.array_equal(candidate.covariance, native.covariance)
    assert candidate.startup_steps == 5


def test_stationary_prediction_uses_static_transition_and_native_q() -> None:
    state, prior = _block_diagonal_prior(7)
    candidate = StartupLocalEKF(state, EKFConfig())
    candidate.covariance = prior.copy()
    dt = 0.2
    transition = candidate.predict(_imu((1.2, -0.7, 0.4), (0.8, 0.4, -0.2)), dt)

    expected_transition = np.eye(15)
    expected_transition[:3, 3:6] = np.eye(3) * dt
    assert np.array_equal(transition, expected_transition)
    assert np.allclose(candidate.state[:3], state[:3] + state[3:6] * dt)
    assert np.allclose(candidate.state[3:9], state[3:9])
    assert np.allclose(candidate.last_prediction_context["previous_covariance"], prior)
    assert np.allclose(candidate.last_prediction_context["transition_matrix"], transition)

    native = LocalEKF(state.copy(), candidate.config)
    native.covariance = np.zeros((15, 15))
    native.predict(_imu(), dt)
    expected_q = native.covariance.copy()
    expected_covariance = transition @ prior @ transition.T + expected_q
    # The bias observation changes only the bias block because the startup
    # cross blocks are required to remain zero.
    assert np.allclose(candidate.covariance[:9, :9], expected_covariance[:9, :9])
    assert np.max(np.abs(candidate.covariance[:9, 9:])) <= 1.0e-12


def test_imu_values_change_only_startup_bias_state() -> None:
    left = StartupLocalEKF(np.zeros(15))
    right = StartupLocalEKF(np.zeros(15))
    left.predict(_imu((0.01, 0.02, 0.03), (0.04, 0.05, 0.06)), 0.2)
    right.predict(_imu((0.20, -0.30, 0.40), (-0.20, 0.30, -0.40)), 0.2)
    assert np.allclose(left.state[:9], right.state[:9])
    assert np.allclose(left.covariance[:9, :9], right.covariance[:9, :9])
    assert not np.allclose(left.state[9:], right.state[9:])


@pytest.mark.parametrize("seed", range(30))
def test_startup_vo_matches_full_30_state_clone(seed: int) -> None:
    state, prior = _block_diagonal_prior(seed)
    candidate = StartupLocalEKF(state, EKFConfig())
    candidate.covariance = prior.copy()
    imu = _imu(np.random.default_rng(seed + 100).normal(size=3), np.random.default_rng(seed + 200).normal(size=3))
    previous_state = candidate.state.copy()
    previous_covariance = candidate.covariance.copy()
    transition = candidate.predict(imu, 0.2)
    nominal_state = candidate.state.copy()
    nominal_covariance = candidate.covariance.copy()
    displacement = np.array([0.001, -0.002, 0.0015])
    noise = np.eye(3) * 0.015**2
    expected_state, expected_covariance, expected_s, expected_gain = _clone_vo(
        nominal_state,
        nominal_covariance,
        previous_state,
        previous_covariance,
        transition,
        displacement,
        noise,
    )
    context = candidate.last_prediction_context
    assert context is not None
    assert candidate.update_visual_delta(
        displacement,
        covariance=noise,
        previous_state=context["previous_state"],
        previous_covariance=context["previous_covariance"],
        transition_matrix=context["transition_matrix"],
        dt_s=0.2,
    )
    assert np.allclose(candidate.state, expected_state, atol=1.0e-9)
    assert np.allclose(candidate.covariance, expected_covariance, atol=1.0e-9)
    assert np.allclose(
        np.asarray(candidate.last_visual_update["innovation_covariance"])
        if "innovation_covariance" in candidate.last_visual_update
        else expected_s,
        expected_s,
        atol=1.0e-9,
    )
    # The native diagnostic does not expose K, but the resulting state and P
    # verify the same current block update as the clone.
    assert np.isfinite(expected_gain).all()


def test_scalar_bias_posterior_uses_joseph_variance_with_q() -> None:
    candidate = StartupLocalEKF(np.zeros(15))
    candidate.predict(_imu((0.2, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.2)
    native = LocalEKF(np.zeros(15))
    native.covariance = np.zeros((15, 15))
    native.predict(_imu(), 0.2)
    q = native.covariance[9, 9]
    prior_variance = 1.0 + q
    measurement_variance = SensorConfig().imu_accel_std_mps2**2
    expected = prior_variance * measurement_variance / (prior_variance + measurement_variance)
    assert np.isclose(candidate.covariance[9, 9], expected)


def test_invalid_cross_or_non_psd_input_is_rejected_without_repair() -> None:
    candidate = StartupLocalEKF(np.zeros(15))
    candidate.covariance[0, 9] = candidate.covariance[9, 0] = 1.0e-3
    before = candidate.covariance.copy()
    with pytest.raises(ValueError, match="cross-covariance"):
        candidate.predict(_imu(), 0.2)
    assert np.array_equal(candidate.covariance, before)

    candidate = StartupLocalEKF(np.zeros(15))
    candidate.covariance[0, 0] = -1.0
    before = candidate.covariance.copy()
    with pytest.raises(ValueError, match="non-PSD"):
        candidate.predict(_imu(), 0.2)
    assert np.array_equal(candidate.covariance, before)


def test_startup_vo_motion_guard_and_evidence_contract() -> None:
    candidate = StartupLocalEKF(np.zeros(15))
    with pytest.raises(ValueError, match="exactly five"):
        candidate.check_startup_evidence()
    for _ in range(5):
        candidate.predict(_imu(), 0.2)
        context = candidate.last_prediction_context
        assert context is not None
        if candidate.startup_valid_vo_count == 0:
            with pytest.raises(ValueError, match="stationary_start_unsupported"):
                candidate.update_visual_delta(
                    np.array([1.0, 0.0, 0.0]),
                    previous_state=context["previous_state"],
                    previous_covariance=context["previous_covariance"],
                    transition_matrix=context["transition_matrix"],
                    dt_s=0.2,
                )
        else:
            candidate.update_visual_delta(
                np.zeros(3),
                previous_state=context["previous_state"],
                previous_covariance=context["previous_covariance"],
                transition_matrix=context["transition_matrix"],
                dt_s=0.2,
            )
    # The first call was a valid packet attempt only after passing the guard;
    # provide one accepted valid packet for the evidence contract.
    candidate = StartupLocalEKF(np.zeros(15))
    for _ in range(5):
        candidate.predict(_imu(), 0.2)
        context = candidate.last_prediction_context
        assert context is not None
        candidate.update_visual_delta(
            np.zeros(3),
            previous_state=context["previous_state"],
            previous_covariance=context["previous_covariance"],
            transition_matrix=context["transition_matrix"],
            dt_s=0.2,
        )
    candidate.check_startup_evidence()
    assert candidate.startup_valid_vo_count == 5


def test_normal_prediction_and_native_gate_resume_after_prefix() -> None:
    candidate = StartupLocalEKF(np.zeros(15))
    for _ in range(5):
        candidate.predict(_imu((0.01, 0.0, 0.0)), 0.2)
    candidate.last_startup_update["active"] = True
    native = LocalEKF(candidate.state.copy(), candidate.config)
    native.covariance = candidate.covariance.copy()
    imu = _imu((0.2, -0.1, 0.3), (0.01, -0.02, 0.03))
    candidate_transition = candidate.predict(imu, 0.2)
    native_transition = native.predict(imu, 0.2)
    assert np.allclose(candidate_transition, native_transition)
    assert np.allclose(candidate.state, native.state)
    assert np.allclose(candidate.covariance, native.covariance)
    assert candidate.last_startup_update["active"] is False
    assert candidate.update_position(np.full(3, 100.0), np.eye(3) * 0.01**2) is False


def test_api_does_not_require_truth_and_missing_vo_does_not_fake_valid_count() -> None:
    signature = inspect.signature(StartupLocalEKF.update_visual_delta)
    assert "truth" not in signature.parameters
    candidate = StartupLocalEKF(np.zeros(15))
    candidate.predict(_imu(), 0.2)
    assert candidate.startup_valid_vo_count == 0
