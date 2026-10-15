"""Pure V13 units: no physical rollout and no policy update."""
import copy
import math
import numpy as np
import pytest
from src.analysis import neighbor_selection_policy_v13 as p
from src.experiments.neighbor_selection_validation_v13 import norm_labels, validate_batch


def data():
    context = {"step": 6, "seed": 1, "time_s": 1., "own_covariance": np.eye(15), "own_state": np.zeros(15),
               "peers": {u: {"prelink_delay_s": .2, "gnss_std_m": .5, "gnss_report_time_s": .8} for u in p.PEERS},
               "measurement_budget_remaining": 30, "legal_sources": list(p.PEERS)}
    baseline = {"features": {}, "candidates": {u: {"valid": True, "predicted_position_trace_m2": 2., "reason": "none"} for u in p.PEERS}}
    return context, baseline


def test_cost_unit_and_none():
    c, b = data()
    result = p.extend(c, b)
    assert result["candidates"][p.PEERS[0]]["score"] == pytest.approx(math.sqrt(3) - math.sqrt(2) - .02)
    assert result["none_score"] == 0
    assert p.choose_greedy(result) == p.PEERS[0]
    for x in result["candidates"].values(): x["score"] = -1.
    assert p.choose_greedy(result) == "none"


@pytest.mark.parametrize("key", ["truth", "quality_offset", "phase", "future_error"])
def test_no_unregistered_input(key):
    c, b = data(); c[key] = 1
    with pytest.raises(ValueError): p.extend(c, b)


def test_quality_not_separate_bonus_and_immutability():
    c, b = data(); before = copy.deepcopy(b)
    score = p.extend(c, b)["candidates"][p.PEERS[0]]["score"]
    c["peers"][p.PEERS[0]]["gnss_std_m"] = 3.
    assert p.extend(c, b)["candidates"][p.PEERS[0]]["score"] == score
    assert b == before


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf")])
def test_invalid_delay(value):
    assert p.request_cost(value) is None


def test_norm_labels_units_legal_and_ties():
    branches = {"none": {"sse_m2": [4.] * 4}, p.PEERS[0]: {"sse_m2": [1.] * 4}, p.PEERS[1]: {"sse_m2": [4.] * 4}}
    result = {"branches": branches, "dual_eligible": True, "score_difference": 0.}
    decision = {"context": {"peers": {u: {"prelink_delay_s": .2} for u in p.PEERS}}, "v13_scoring": {"score_difference": 1.}}
    labels = norm_labels(result, decision)
    assert labels["norm_labels"][p.PEERS[0]]["norm_utility_m"] == pytest.approx(.98)
    assert labels["v13_ranking_correct"] == 1
    assert labels["legacy_norm_ranking_correct"] == .5
    result["dual_eligible"] = False
    assert norm_labels(result, decision)["v13_ranking_correct"] is None


def test_batch_cannot_escape():
    with pytest.raises(ValueError): validate_batch("../old")


@pytest.mark.parametrize("x,std", [(40., .5), (45., 1.5), (50., 3.)])
def test_dynamic_gnss_uses_same_standard_normal_and_declared_variance(x, std):
    from types import SimpleNamespace
    from unittest.mock import patch
    from src.experiments import neighbor_selection_environment_v13 as e
    from src.simulation.sensor_model import SensorConfig
    env = e.DynamicEnv.__new__(e.DynamicEnv)
    state = SimpleNamespace(position_m=np.array([x, 125., 28.]))
    env.states = {"uav_00": state}
    env.config = SimpleNamespace(motion=SimpleNamespace(dt_s=.2))
    env.timestep = 1
    env.v13_dynamic_gnss, env.v13_dynamic_radio = True, True
    env.v13_quality_offset = 0
    env._v13_sensor_packets, env._v13_last_gnss, env.v13_quality_log = {}, {}, []
    rng, expected_rng = np.random.default_rng(5), np.random.default_rng(5)
    config = SensorConfig()
    expected = state.position_m + expected_rng.normal(0, std, 3)
    def probe(self, actions):
        return e.environment_module.sample_gnss(state, "nominal", rng, config)
    next_class = e.ORIGINALCLASS.__mro__[1]
    with patch.object(next_class, "step", probe):
        packet = e._dynamic_gnss_step(env, {})
    assert packet["std"] == std
    assert packet["valid"] == 1
    np.testing.assert_array_equal(packet["position"], expected)
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert config.gnss_nominal_std_m == .5


def test_dynamic_gnss_denied_target_consumes_no_gnss_noise():
    from types import SimpleNamespace
    from unittest.mock import patch
    from src.experiments import neighbor_selection_environment_v13 as e
    from src.simulation.sensor_model import SensorConfig
    env = e.DynamicEnv.__new__(e.DynamicEnv)
    state = SimpleNamespace(position_m=np.array([40., 125., 28.]))
    env.states = {"uav_02": state}
    env.config = SimpleNamespace(motion=SimpleNamespace(dt_s=.2))
    env.timestep, env.v13_quality_offset = 1, 0
    env.v13_dynamic_gnss, env.v13_dynamic_radio = True, True
    env._v13_sensor_packets, env._v13_last_gnss, env.v13_quality_log = {}, {}, []
    rng = np.random.default_rng(5)
    before = copy.deepcopy(rng.bit_generator.state)
    next_class = e.ORIGINALCLASS.__mro__[1]
    with patch.object(next_class, "step", lambda self, actions: e.environment_module.sample_gnss(state, "nominal", rng, SensorConfig())):
        packet = e._dynamic_gnss_step(env, {})
    assert packet["valid"] == 0 and packet["position"] is None
    assert rng.bit_generator.state == before


def test_total_delay_overflow_is_not_a_legal_channel_candidate():
    import random
    from src.experiments.neighbor_selection_environment_v13 import channel_metrics
    from src.synthetic.channel_model import ChannelConfig
    m = channel_metrics(dict(distance_m=20., los=True, bandwidth_ratio=.5, queue_length_packets=0.,
                             previous_aoi_s=0., dt_s=.2, rng=random.Random(8), config=ChannelConfig()), 0., 2.)
    assert not m.candidate_edge and not m.effective_edge
