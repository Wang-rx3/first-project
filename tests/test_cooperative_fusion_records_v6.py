"""Pure record-contract checks; never construct or step an environment."""
from types import SimpleNamespace

import numpy as np

from src.experiments import three_uav_fusion_validation_v6 as v6
from src.simulation.ekf import EKFConfig
from src.simulation.environment import MultiUAVJointEnv


def test_sensor_whitelist_excludes_ground_truth_bias_fields():
    packets = {u: {"imu": {"accel": [0., 0., 0.], "gyro": [0., 0., 0.], "bias_accel": [12345., 0., 0.]},
                  "vo": {"valid": False, "delta_position": None},
                  "gnss": {"valid": False, "position": None, "std": 20.}} for u in v6.UIDS}
    clean = v6.sanitize_sensor_packets(packets, .015)
    assert all(set(clean[u]["imu"]) == {"accel", "gyro"} for u in v6.UIDS)
    assert "12345" not in v6.old.encode(clean)


def test_packet_covariance_replay_matches_native_arrival_and_increment_sequence():
    cfg = EKFConfig()
    proxy = SimpleNamespace(config=SimpleNamespace(ekf=cfg))
    state = np.arange(15, dtype=float)
    covariance = np.diag([4.] * 3 + [1.] * 12)
    expected_state, expected_cov = MultiUAVJointEnv._propagate_state_covariance(proxy, state, covariance, .4)
    expected_state, expected_cov = MultiUAVJointEnv._propagate_state_covariance(proxy, expected_state, expected_cov, .2)
    envelope = {"send_time_s": 0., "arrival_time_s": .4}
    actual_state, actual_cov = v6.propagate_packet(state, covariance, envelope, .6, cfg)
    assert np.allclose(actual_state, expected_state)
    assert np.allclose(actual_cov, expected_cov)
