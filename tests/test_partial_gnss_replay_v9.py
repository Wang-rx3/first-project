"""Unit contracts for the isolated v9 replay kernel."""

from __future__ import annotations

import copy

import numpy as np

from src.analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
from src.analysis.partial_gnss_fusion_v7 import MeasurementCIEKF
from src.analysis.partial_gnss_replay_v9 import StartupMeasurementCI, replay
from src.experiments import partial_gnss_baseline_v7 as v7
from src.experiments import three_uav_fusion_validation_v6 as prior


def _packets(vo_std: float) -> dict[str, dict[str, object]]:
    return {
        uid: {
            "imu": {"accel": np.zeros(3), "gyro": np.zeros(3)},
            "vo": {"valid": True, "delta_position": np.zeros(3), "covariance": np.eye(3) * vo_std**2},
            "gnss": {"valid": False},
        }
        for uid in v7.UIDS
    }


def _record(steps: int = 6) -> dict[str, object]:
    sim = v7.simulation()
    initial = {uid: {"state": np.zeros(15), "covariance": np.eye(15)} for uid in v7.UIDS}
    filters = {
        uid: StartupLocalEKF(np.zeros(15), sim.ekf, sensor_config=sim.sensor, startup_steps=5, vo_motion_limit=16.27)
        for uid in v7.UIDS
    }
    for estimator in filters.values():
        estimator.covariance = np.eye(15)
    frames = []
    spec = {"dt_s": 0.2, "decision_steps": [6], "startup_steps": 5, "measurement_budget": 30}
    for step in range(1, steps + 1):
        packets = _packets(sim.sensor.vo_position_std_m)
        prior.local_sensor_update(filters, packets, None, step)
        states = {uid: filters[uid].state.copy() for uid in v7.UIDS}
        covariances = {uid: filters[uid].covariance.copy() for uid in v7.UIDS}
        frames.append({
            "step": step,
            "observable": {"sensor_packets": packets, "range_bank": {}, "pre_legal_sources": []},
            "evaluation_truth": {"position": {uid: np.zeros(3) for uid in v7.UIDS}, "velocity": {uid: np.zeros(3) for uid in v7.UIDS}},
            "audit": {"states": states, "covariances": covariances},
        })
    return {"status": "complete", "case": {"scene_id": "unit", "seed": 7}, "initial": initial, "frames": frames}


def _record_with_ranges(steps: int = 14) -> dict[str, object]:
    """Small frozen None tape with three fresh successful packets."""
    sim = v7.simulation()
    initial_state = {uid: np.zeros(15) for uid in v7.UIDS}
    initial_state["uav_00"][:3] = [0.0, 0.0, 0.0]
    initial_state["uav_00"][3:6] = [0.0, 1.0, 0.0]
    initial_state["uav_01"][:3] = [10.0, 0.0, 0.0]
    initial_state["uav_02"][:3] = [5.0, 0.0, 0.0]
    initial_covariance = {uid: np.eye(15) * (1.0e-5 if uid == "uav_00" else 0.1) for uid in v7.UIDS}
    initial = {uid: {"state": initial_state[uid].copy(), "covariance": initial_covariance[uid].copy()} for uid in v7.UIDS}
    filters = {
        uid: StartupLocalEKF(initial_state[uid].copy(), sim.ekf, sensor_config=sim.sensor, startup_steps=5, vo_motion_limit=16.27)
        for uid in v7.UIDS
    }
    for uid in v7.UIDS:
        filters[uid].covariance = initial_covariance[uid].copy()
    frames = []
    selected_steps = {6, 10, 14}
    histories = {}
    for step in range(1, steps + 1):
        packets = _packets(sim.sensor.vo_position_std_m)
        histories[step] = {uid: (filters[uid].state.copy(), filters[uid].covariance.copy()) for uid in v7.UIDS}
        prior.local_sensor_update(filters, packets, None, step)
        states = {uid: filters[uid].state.copy() for uid in v7.UIDS}
        covariances = {uid: filters[uid].covariance.copy() for uid in v7.UIDS}
        observable = {"sensor_packets": packets, "range_bank": {}, "pre_legal_sources": ["uav_00"] if step in selected_steps else []}
        if step in selected_steps:
            send_step = step - 2
            send_time = send_step * 0.2
            envelope = {"source": "uav_00", "target": "uav_02", "send_time_s": send_time, "arrival_time_s": send_time, "last_propagated_time_s": step * 0.2, "delay_s": 0.0}
            packet_state, packet_covariance = prior.propagate_packet(*histories[send_step + 1]["uav_00"], envelope, step * 0.2, sim.ekf)
            measurement = float(np.linalg.norm(states["uav_02"][:3] - packet_state[:3]))
            observable["range_bank"]["uav_00"] = {
                "source": "uav_00", "measurement": measurement, "variance": 1.0e-8,
                "nlos_bias": 0.0, "message_envelope": envelope,
                "packet_state": packet_state, "packet_covariance": packet_covariance,
            }
            observable["post_conditions"] = {"uav_00": {"uwb_valid": True, "effective_edge": True, "message_arrived": True, "potential_edge": True}}
        frames.append({
            "step": step, "observable": observable,
            "evaluation_truth": {"position": {uid: initial_state[uid][:3].copy() for uid in v7.UIDS}, "velocity": {uid: np.zeros(3) for uid in v7.UIDS}},
            "audit": {"states": states, "covariances": covariances},
        })
    return {"status": "complete", "case": {"scene_id": "unit_ranges", "seed": 7}, "initial": initial, "frames": frames}


def test_startup_measurement_ci_preserves_the_v8_v7_mro() -> None:
    assert StartupMeasurementCI.mro()[:4] == [StartupMeasurementCI, StartupLocalEKF, MeasurementCIEKF, StartupLocalEKF.__mro__[-2]]
    assert StartupMeasurementCI.predict is StartupLocalEKF.predict
    assert StartupMeasurementCI.update_relative_range is MeasurementCIEKF.update_relative_range


def test_replay_checks_five_step_startup_and_keeps_startup_without_requests() -> None:
    record = _record()
    rows, events, audit = replay(record, "restore_prior", "none", {"dt_s": 0.2, "decision_steps": [6], "startup_steps": 5})
    assert len(rows) == len(record["frames"]) == 6
    assert all(event["selected"] is False for event in events)
    assert all(event["attempted"] is False for event in events)
    assert audit["startup_evidence_checked"] is True
    assert audit["none_driver_reconstruction_max"] <= 1e-8
    assert rows[4]["gnss_accepted_0_1_2"] == [0, 0, 0]


def test_joint_reference_starts_after_step_five_with_zero_cross_blocks() -> None:
    record = _record()
    rows, events, audit = replay(record, "joint_reference", "none", {"dt_s": 0.2, "decision_steps": [6], "startup_steps": 5})
    assert len(rows) == 6
    assert rows[4]["joint_position_covariance_9"] is None
    assert rows[5]["joint_position_covariance_9"] is not None
    assert np.isfinite(audit["joint_none_reconstruction_max"])


def test_selected_trajectory_logs_multiple_fresh_accepted_ranges() -> None:
    record = _record_with_ranges()
    spec = {"dt_s": 0.2, "decision_steps": [6, 10, 14], "startup_steps": 5, "measurement_budget": 30, "message_ttl_s": 0.6, "pair_period_s": 0.8}
    rows, events, audit = replay(record, "restore_prior", "fixed0", spec)
    assert len(events) == 3
    assert all(event["selected"] for event in events)
    assert all(event["attempted"] for event in events)
    assert sum(event["accepted"] for event in events) >= 2
    assert audit["source_reconstruction_max"] <= 1e-8
    assert audit["packet_reconstruction_max"] <= 1e-8
    assert all(row["gnss_accepted_0_1_2"] == [0, 0, 0] for row in rows)


def test_budget_consumed_by_requests_including_expired_packet():
    record = _record_with_ranges()
    record["frames"][5]["observable"]["range_bank"]["uav_00"]["message_envelope"]["send_time_s"] = 0.
    spec = {"decision_steps": [6, 10, 14], "measurement_budget": 2}
    rows, events, _ = replay(record, "restore_prior", "fixed0", spec)
    assert len(events) == 2
    assert events[0]["rejection_reason"] == "packet_expired"
    assert not events[0]["attempted"]
    assert rows[-1]["budget_used"] == 2
    assert rows[-1]["source"] == "none"


def test_outside_window_mask_does_not_create_makeup_request():
    record = _record_with_ranges()
    record["frames"][6]["observable"]["pre_legal_sources"] = ["uav_00"]
    rows, events, _ = replay(record, "restore_prior", "fixed0", {"decision_steps": [6, 10, 14]})
    assert rows[6]["source"] == "none"
    assert len(events) == 3


def test_evaluation_truth_does_not_affect_estimates_or_selections():
    original = _record_with_ranges()
    changed = copy.deepcopy(original)
    for frame in changed["frames"]:
        for uid in v7.UIDS:
            frame["evaluation_truth"]["position"][uid] = np.array([900., -400., 100.])
    spec = {"decision_steps": [6, 10, 14]}
    a, _, _ = replay(original, "measurement_ci", "fixed0", spec)
    b, _, _ = replay(changed, "measurement_ci", "fixed0", spec)
    for left, right in zip(a, b):
        assert left["source"] == right["source"]
        for uid in v7.UIDS:
            np.testing.assert_array_equal(left["states"][uid], right["states"][uid])
            np.testing.assert_array_equal(left["covariances"][uid], right["covariances"][uid])

