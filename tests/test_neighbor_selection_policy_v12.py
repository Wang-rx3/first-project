"""Synthetic policy and local-information boundary tests; no physical episode."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis import neighbor_selection_policy_v12 as policy
from src.experiments import neighbor_selection_execution_v12 as execution
from src.simulation.environment import SimulationConfig
from tests.test_neighbor_selection_policy_v11 import _context
from tests.test_neighbor_selection_execution_v11 import LocalOnly


def test_proxy_has_no_independent_geometry_or_quality_bonus():
    baseline, new = policy.score_context(_context(), SimulationConfig())
    for uid in policy.PEERS:
        assert new["candidates"][uid]["score"] == pytest.approx(baseline["candidates"][uid]["trace_gain_m2"]-.01)
    assert new["none_score"] == 0.


def test_no_update_and_positive_threshold_include_none():
    score = {"seed": 2, "step": 6, "candidates": {u: {"valid": True, "score": -.01} for u in policy.PEERS}}
    assert policy.choose_greedy(score) == "none"
    score["candidates"][policy.PEERS[0]]["score"] = 0.
    assert policy.choose_greedy(score) == "none"
    score["candidates"][policy.PEERS[1]]["score"] = .1
    assert policy.choose_greedy(score) == policy.PEERS[1]


def test_positive_ties_are_reproducible_and_invalid_peer_is_masked():
    score = {"seed": 2, "step": 6, "candidates": {u: {"valid": True, "score": .2} for u in policy.PEERS}}
    assert policy.choose_greedy(score) == policy.choose_greedy(copy.deepcopy(score))
    score["candidates"][policy.PEERS[0]]["valid"] = False
    assert policy.choose_greedy(score) == policy.PEERS[1]


def test_truth_and_future_inputs_are_rejected():
    context = _context()
    context["evaluation_truth"] = [99, 99, 99]
    with pytest.raises(ValueError, match="non-observable"):
        policy.score_context(context, SimulationConfig())


def test_metadata_collection_does_not_read_true_states_or_private_peer_ekfs():
    env = LocalOnly()
    env.arrived_messages[("uav_00", "uav_02")].delay_s = .012
    context = execution.visible_context(env, ["uav_00"], 11, 6)
    assert context["peers"]["uav_00"]["packet_delay_s"] == .012
    assert context["peers"]["uav_00"]["prelink_delay_s"] is None
    assert context["time_s"] == 1.


def test_missing_and_zero_distance_directional_slots_are_none():
    context = _context()
    context["peers"]["uav_00"]["state"] = np.zeros(15)
    del context["peers"]["uav_01"]
    _, new = policy.score_context(context, SimulationConfig())
    assert policy.choose_greedy(new) == "none"
    assert new["directional_encoding"]["uav_00_direction"] is None
    assert new["directional_encoding"]["uav_01_direction"] is None


def test_adapter_symbols_restore_on_execution_error(monkeypatch):
    original_env, original_policies, original_choose = execution.native.AuditEnv, execution.native.POLICIES, execution.v7.choose
    def fail(*args):
        assert "minimal_greedy" in execution.native.POLICIES
        raise RuntimeError("synthetic error")
    monkeypatch.setattr(execution.native, "physical", fail)
    with pytest.raises(RuntimeError, match="synthetic error"):
        execution.physical({}, {"proxy": {"request_penalty_m2": .01}}, "minimal_greedy")
    assert execution.native.AuditEnv is original_env
    assert execution.native.POLICIES is original_policies
    assert execution.v7.choose is original_choose
