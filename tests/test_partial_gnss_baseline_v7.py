from unittest.mock import patch

import numpy as np
import pytest

from src.experiments import partial_gnss_baseline_v7 as v7
from src.simulation.motion_model import UAVKinematicState


def test_gnss_patch_is_identity_based_noisy_and_restored():
    env = v7.PartialGNSSEnv.__new__(v7.PartialGNSSEnv)
    env.states = {u: UAVKinematicState(position_m=np.array([20., 30., 25.]), velocity_mps=np.zeros(3),
        attitude_rad=np.zeros(3), acceleration_mps2=np.zeros(3), angular_velocity_rps=np.zeros(3)) for u in v7.UIDS}
    original = v7.environment_module.sample_gnss
    from src.simulation.sensor_model import SensorConfig
    def consume(self, actions):
        return {u: v7.environment_module.sample_gnss(state, "denied", np.random.default_rng(45+i), SensorConfig())
                for i, (u,state) in enumerate(self.states.items())}
    with patch.object(v7.old.DiagnosticEnv, "step", consume):
        result = env.step({})
    assert v7.environment_module.sample_gnss is original
    assert [result[u]["valid"] for u in v7.UIDS] == [1,1,0]
    for u in v7.PEERS:
        assert result[u]["std"] == .5
        assert not np.array_equal(result[u]["position"], env.states[u].position_m)
    with patch.object(v7.old.DiagnosticEnv, "step", side_effect=RuntimeError("forced")):
        with pytest.raises(RuntimeError):
            env.step({})
    assert v7.environment_module.sample_gnss is original


def test_random_peer_uses_legal_sources_and_independent_stream():
    assert v7.choose("fixed0", ["uav_01"], 20267101, 2) == "none"
    assert v7.choose("random_peer", ["uav_01"], 20267101, 2) == "uav_01"
    assert v7.choose("random_peer", [], 20267101, 2) == "none"
    selections = [v7.choose("random_peer", v7.PEERS, 20267101, s) for s in range(2,119,4)]
    assert set(selections) == set(v7.PEERS)
    assert selections == [v7.choose("random_peer", v7.PEERS, 20267101, s) for s in range(2,119,4)]
    assert len(selections) == 30


def test_frozen_full_state_initial_law_and_geometry():
    spec = v7.old.read_json(v7.CONFIG)
    for seed, per_uav in spec["initial_errors_by_seed"].items():
        for uid, error in per_uav.items():
            expected = v7.indexed_rng(int(seed),0,"v7_initial_error",uid).normal(size=15)*np.sqrt([4]*3+[1]*12)
            np.testing.assert_array_equal(error, expected)
            # Native LocalEKF is additive, not an SO(3) error-state filter.
            initial = np.zeros(15)
            ekf = v7.LocalEKF(initial.copy())
            ekf.state += error
            np.testing.assert_array_equal(ekf.state-initial,error)
            np.testing.assert_array_equal(np.diag(ekf.covariance),spec["initial_covariance_diagonal"])
    assert len(spec["cases"]) == 4
    assert len({c["seed"] for c in spec["cases"]}) == 2
    assert all(c["scenario"]["obstacle_specs"] == [] for c in spec["cases"])
