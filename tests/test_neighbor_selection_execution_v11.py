from types import SimpleNamespace
import numpy as np
import pytest
from src.experiments import neighbor_selection_execution_v11 as execution


class LocalOnly:
    timestep=5
    config=SimpleNamespace(motion=SimpleNamespace(dt_s=.2))
    measurement_budget_remaining={"uav_02":30}
    neighbor_beliefs={}
    neighbor_belief_timestamps={}
    neighbor_belief_message_times={}
    uwb_update_times={}
    def __init__(self):
        self.arrived_messages={("uav_00","uav_02"):SimpleNamespace(
            state=np.r_[10.,0.,0.,np.zeros(12)],covariance=np.eye(15),
            send_time_s=.8,last_propagated_time_s=1.)}
    @property
    def states(self):raise AssertionError("truth access forbidden")
    @property
    def ekfs(self):raise AssertionError("private peer filters forbidden")
    def _observations(self):
        return {"uav_02":{"estimate":np.zeros(15),"covariance":np.eye(15),
                          "candidate_incoming_links":{"uav_00":{"rssi_dbm":-60.,"aoi_s":.2}}}}


def test_context_reads_local_observation_and_arrived_packets_only():
    env=LocalOnly()
    result=execution.visible_context(env,["uav_00"],11,6)
    assert result["time_s"]==1.
    assert result["peers"]["uav_00"]["rssi_dbm"]==-60.
    assert "uav_01" not in result["peers"]
    result["peers"]["uav_00"]["state"][0]=99
    assert env.arrived_messages[("uav_00","uav_02")].state[0]==10


def test_release_hook_calls_original_once_without_offline_score_access(monkeypatch):
    calls=[]
    monkeypatch.setattr(execution._OriginalAuditEnv,"_release_messages",lambda self,step:calls.append(step))
    env=execution.NeighborAuditEnv.__new__(execution.NeighborAuditEnv)
    env.v11_active_policy="none"
    env._release_messages(6)
    assert calls==[6]


def test_aborted_selector_stops_before_a_native_transition():
    env=execution.NeighborAuditEnv.__new__(execution.NeighborAuditEnv)
    env.v11_selection_failure="material numerical error"
    with pytest.raises(RuntimeError,match="selection integrity"):
        env.step({})


def test_scoped_adapters_restore_all_symbols_on_error(monkeypatch):
    original_env=execution.native.AuditEnv
    original_policy=execution.native.POLICIES
    original_choose=execution.v7.choose
    def fail(*args):
        assert execution.native.AuditEnv is execution.NeighborAuditEnv
        assert "greedy" in execution.native.POLICIES
        assert execution.v7.choose is not original_choose
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(execution.native,"physical",fail)
    with pytest.raises(RuntimeError,match="synthetic failure"):
        execution.physical({}, {}, "greedy")
    assert execution.native.AuditEnv is original_env
    assert execution.native.POLICIES is original_policy
    assert execution.v7.choose is original_choose
