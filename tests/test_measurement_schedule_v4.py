"""Pure v4 scheduler checks; these tests never construct an environment."""
import copy

import numpy as np
import pytest

from src.experiments import three_uav_measurement_selection_v4 as v4


def _local(uid="uav_00", mask=(True, True, True, True)):
    peers = [x for x in v4.UIDS if x != uid]
    return {"uav_id": uid, "action_mask": np.asarray(mask, dtype=bool),
            "base_action_mask": np.asarray(mask, dtype=bool),
            "candidate_measurement_sources": peers, "effective_sources": peers,
            "uwb_available": {peer: True for peer in peers},
            "measurement_update_eligibility": np.asarray([x != uid for x in v4.UIDS], dtype=bool),
            "measurement_budget_remaining": 30,
            "neighbor_messages": {peer: {"estimate": np.zeros(15), "covariance": np.eye(15)} for peer in peers}}


def test_window_counts_and_schedule():
    scheduler = v4.WindowRoundRobin()
    assert scheduler.decision_steps == tuple(range(2, 119, 4))
    assert sum(scheduler.in_window(step) for step in range(1, 121)) == 30


def test_round_robin_persistent_and_blocked_does_not_advance():
    scheduler = v4.WindowRoundRobin()
    local = _local()
    assert scheduler.select(local, 1) == "none"
    before = scheduler.state_dict()
    assert scheduler.select(local, 2) == "uav_01"
    assert scheduler.select(local, 3) == "none"
    assert scheduler.state_dict()["cursors"]["uav_00"] == before["cursors"]["uav_00"] + 1
    assert scheduler.select(local, 6) == "uav_02"
    blocked = _local(mask=(False, False, False, True))
    state = scheduler.state_dict()
    assert scheduler.select(blocked, 10) == "none"
    assert scheduler.state_dict() == state


def test_mask_window_is_monotone_and_does_not_mutate_input():
    observations = {uid: _local(uid) for uid in v4.UIDS}
    original = copy.deepcopy(observations)
    off = v4.mask_observations(observations, 3)
    on = v4.mask_observations(observations, 6)
    for uid in v4.UIDS:
        assert np.array_equal(observations[uid]["action_mask"], original[uid]["action_mask"])
    for uid in v4.UIDS:
        assert off[uid]["action_mask"].tolist() == [False, False, False, True]
        assert v4._BASE_CONTRACT.admissible_mask(off[uid], uid=uid).tolist() == [False, False, False, True]
        assert bool(on[uid]["action_mask"][3])
        assert sum(bool(value) for value in on[uid]["action_mask"][:3]) == 2
        assert v4._BASE_CONTRACT.admissible_mask(on[uid], uid=uid).tolist() == on[uid]["action_mask"].tolist()
        assert np.all(on[uid]["action_mask"][:3] <= on[uid]["native_action_mask"][:3])
        assert off[uid]["native_action_mask"].tolist() == [x != uid for x in v4.UIDS] + [True]


def test_scheduler_state_round_trip_and_reset():
    scheduler = v4.WindowRoundRobin()
    scheduler.select(_local(), 2)
    restored = v4.WindowRoundRobin.from_state_dict(scheduler.state_dict())
    assert restored.state_dict() == scheduler.state_dict()
    restored.reset()
    assert restored.state_dict()["cursors"] == {uid: 0 for uid in v4.UIDS}


def test_audit_rows_use_predecision_frame_after_next_observation(monkeypatch):
    captured = []
    class Output:
        def __init__(self):
            self.tables = {}
        def row(self, table, values):
            self.tables.setdefault(table, []).append(values)
    output = Output()

    def fake_episode(output, case, sim, cfg, spec, counters, method, bank=None):
        helper = v4.v3.contract()
        native = {uid: {"uav_id": uid} for uid in v4.UIDS}
        helper.admissible_observations(native)  # frame used for step 1
        helper.admissible_observations(native)  # next frame is now current
        output.row("actions", {"step": 1, "uav_id": "uav_00"})
        captured.extend(output.tables["actions"])
        return {"decision_rows": [], "positive_credits": []}

    monkeypatch.setattr(v4.v3, "episode", fake_episode)
    v4._episode(output, {}, None, None, {}, {}, v4.WindowRoundRobin(), [])
    assert captured[0]["decision_step"] == 1
    assert captured[0]["decision_window"] is False


def test_forged_mask_cannot_override_physical_or_budget_gate():
    scheduler = v4.WindowRoundRobin()
    local = _local()
    local['measurement_budget_remaining'] = 0
    assert scheduler.select(local, 2) == 'none'
    local['measurement_budget_remaining'] = 30
    local['uwb_available'] = {peer: False for peer in v4.UIDS}
    assert scheduler.select(local, 6) == 'none'
    assert scheduler.state_dict()['cursors']['uav_00'] == 0


def test_legacy_bindings_restored_on_failure():
    names = ('STEM', 'SCHEMA', 'SOURCE_FILES', 'CONFIG', 'contract')
    before = {name: getattr(v4.v3, name) for name in names}
    with pytest.raises(RuntimeError, match='deliberate'):
        with v4.scheduler_context(v4.WindowRoundRobin()):
            assert v4.v3.STEM == v4.STEM
            raise RuntimeError('deliberate')
    assert {name: getattr(v4.v3, name) for name in names} == before


def test_config_preserves_old_gate_and_rejects_physical_tuning():
    spec = v4.read_json(v4.CONFIG)
    v4.validate_frozen_config(spec)
    assert spec['calibration_gates']['min_separated_decisions'] == 30
    assert spec['learnability_status']['training_ready'] is False
    modified = copy.deepcopy(spec)
    modified['simulation']['sensor']['uwb_update_period_s'] = .4
    with pytest.raises(ValueError, match='Frozen v3 configuration mismatch'):
        v4.validate_frozen_config(modified)
    modified = copy.deepcopy(spec)
    modified['calibration_gates']['min_separated_decisions'] = 4
    with pytest.raises(ValueError, match='Frozen v3 configuration mismatch'):
        v4.validate_frozen_config(modified)


def test_v4_has_no_training_entry():
    with pytest.raises(ValueError, match='calibration only'):
        v4.main('train')

