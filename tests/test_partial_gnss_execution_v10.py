"""Pure contracts for the v10 execution adapters.

These tests monkeypatch the v9 runner; they intentionally do not step the
physical environment or execute a replay tape.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis import partial_gnss_replay_v9 as replay_v9
from src.experiments import partial_gnss_baseline_v7 as v7
from src.experiments import partial_gnss_execution_v10 as v10


def test_replay_forwards_selected_method_class_and_restores_after_error(monkeypatch):
    original = replay_v9.StartupMeasurementCI
    observed = {}

    def fail(record, method, policy, spec):
        observed["class"] = replay_v9.StartupMeasurementCI
        observed["args"] = (method, policy, spec["dt_s"])
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(replay_v9, "replay", fail)
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        v10.replay({"status": "complete"}, "refined_ci", "none", {})
    assert observed["class"] is v10._method_classes()["refined_ci"]
    assert observed["args"] == ("measurement_ci", "none", 0.2)
    assert replay_v9.StartupMeasurementCI is original


def test_replay_keeps_rows_and_events_and_adds_method_timing(monkeypatch):
    original = replay_v9.StartupMeasurementCI

    def fake(record, method, policy, spec):
        assert method == "measurement_ci"
        assert policy == "fixed0"
        assert spec["message_ttl_s"] == 0.6
        event = {"attempted": True, "event": {"compute_time_ms": 1.25, "conditional_bound_residual_norm": 0.0, "conditional_bound_matches_posterior": True}}
        return [{"step": 1}], [event], {"status": "complete", "source_reconstruction_max": 0.0, "packet_reconstruction_max": 0.0}

    monkeypatch.setattr(replay_v9, "replay", fake)
    rows, events, audit = v10.replay({}, "original_ci", "fixed0", {})
    assert rows == [{"step": 1}]
    assert events[0]["method"] == "original_ci"
    assert audit["timing"]["compute_time_ms_total"] == pytest.approx(1.25)
    assert audit["conditional_bounds"]["conditional_bound_match_count"] == 1
    assert replay_v9.StartupMeasurementCI is original


def test_safety_watch_rejects_projection_and_large_or_nonfinite_estimates():
    env = SimpleNamespace(
        ekfs={"uav_00": SimpleNamespace(state=np.zeros(15), covariance=np.eye(15))}
    )
    info = {"boundary_projection_count": 1, "safety_projection_count": 0}
    original = lambda *args: "native"
    with pytest.raises(RuntimeError, match="boundary projection"):
        v10._safety_watch(env, info, {}, {}, original)

    info = {"boundary_projection_count": 0, "safety_projection_count": 0}
    env.ekfs["uav_00"].state[0] = 1.0e3
    with pytest.raises(RuntimeError, match="state magnitude"):
        v10._safety_watch(env, info, {}, {}, original)

    env.ekfs["uav_00"].state[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite state"):
        v10._safety_watch(env, info, {}, {}, original)

    env.ekfs["uav_00"].state[0] = 0.0
    env.ekfs["uav_00"].covariance[0, 0] = 1.0e4
    with pytest.raises(RuntimeError, match="covariance trace limit"):
        v10._safety_watch(env, info, {}, {}, original)

    env.ekfs["uav_00"].covariance = np.eye(15)
    env.ekfs["uav_00"].covariance[0, 0] = -2.0e-8
    with pytest.raises(RuntimeError, match="non-PSD covariance"):
        v10._safety_watch(env, info, {}, {}, original)


def test_navigation_check_uses_only_estimate_and_goal_and_zeroes_startup():
    estimates = {uid: np.zeros(15) for uid in v10.UIDS}
    goals = {uid: np.array([10.0, 0.0, 0.0]) for uid in v10.UIDS}
    actions = {uid: {"delta_v": np.zeros(3)} for uid in v10.UIDS}
    startup = v10._navigation_check(1, estimates, goals, actions)
    assert startup["first_five_zero"] is True
    assert startup["used_truth"] is False
    expected = np.asarray(v10.fixed_navigation_command(np.zeros(15)[:6], goals[v10.UIDS[0]])["command_delta_v_mps"])
    actions[v10.UIDS[0]]["delta_v"] = expected
    check = v10._navigation_check(6, estimates, goals, actions)
    assert check["per_uav"][v10.UIDS[0]]["matches"] is True
    assert check["per_uav"][v10.UIDS[1]]["matches"] is False


def test_feedback_audit_joins_adjacent_live_preestimate_without_truth():
    target = v10.TARGET
    post = np.zeros(15)
    post[0] = 9.0
    before_ci = np.zeros(15)
    goal = np.array([10.0, 0.0, 0.0])
    actual = np.asarray(v10.fixed_navigation_command(post[:6], goal)["command_delta_v_mps"])
    record = {
        "frames": [
            {
                "step": 6,
                "audit": {
                    "states": {target: post},
                    "fusion_event": {"accepted": True, "state_before": before_ci},
                },
            },
            {
                "step": 7,
                "audit": {
                    "pre_estimates": {target: np.full(15, 999.0)},  # must be ignored
                    "actions": {target: {"delta_v": actual}},
                    "live_queue": {
                        "pre_own_estimates": {target: post},
                        "goals": {target: goal},
                    },
                },
            },
        ]
    }
    audit = v10._feedback_audit(record)
    assert audit["feedback_update_pairs"] == 1
    assert audit["feedback_command_change_pairs"] == 1
    assert audit["max_controller_input_error"] == pytest.approx(0.0)
    assert audit["counterfactual_command_delta_max"] > 0.0
    assert audit["pairs"][0]["truth_used"] is False


def test_physical_scoped_patches_restore_when_episode_raises(monkeypatch):
    original_env = v7.PartialGNSSEnv
    original_check = v7.check_safety
    original_ci = replay_v9.StartupMeasurementCI
    observed = {}

    def fail(case, spec, policy):
        observed["env"] = v7.PartialGNSSEnv
        observed["check"] = v7.check_safety
        observed["ci"] = replay_v9.StartupMeasurementCI
        raise RuntimeError("synthetic physical failure")

    monkeypatch.setattr(v10.v9, "episode", fail)
    with pytest.raises(RuntimeError, match="synthetic physical failure"):
        v10.physical({}, {}, "original_ci", "none")
    assert callable(observed["env"])
    assert observed["env"] is not original_env
    assert observed["check"] is not original_check
    assert observed["ci"] is v10._method_classes()["original_ci"]
    assert v7.PartialGNSSEnv is original_env
    assert v7.check_safety is original_check
    assert replay_v9.StartupMeasurementCI is original_ci


def test_expired_wall_budget_stops_before_native_step():
    env=v10.AuditEnv.__new__(v10.AuditEnv)
    env.v10_phase_deadline=0.
    with pytest.raises(RuntimeError, match="frozen phase wall limit"):
        env.step({})
