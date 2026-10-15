"""Bounded record production and independent shadow-fusion validation.

Only the record producer constructs a native environment. Shadow estimators
receive sanitized observations; evaluation truth never enters their API.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ..control.fixed_navigation_execution import compose_fixed_navigation_actions
from ..simulation.ekf import LocalEKF
from ..simulation.random_streams import indexed_rng
from ..simulation.sensor_model import sample_gnss
from . import three_uav_cooperative_ekf_reliability_v5 as old

ROOT = old.ROOT
STEM = "three_uav_fusion_validation_v6_seed2026_20261012"
CONFIG = ROOT / "data/configs" / (STEM + ".json")
BASE = ROOT / "data/synthetic" / STEM
UIDS = old.UIDS
METHODS = ("restore_prior", "young_bound", "joint_reference")
CONDITIONS = ("no_anchor", "nominal_gnss_uav00")
SOURCE_FILES = tuple(dict.fromkeys((*old.SOURCE_FILES, "src/simulation/random_streams.py",
    "src/experiments/three_uav_fusion_validation_v6.py", "src/experiments/stage3_validate_fusion_v6.py",
    "src/analysis/cooperative_fusion_estimators_v6.py")))


def pair_sse(errors):
    return float(np.mean([np.sum((errors[left] - errors[right]) ** 2) for i, left in enumerate(UIDS) for right in UIDS[i + 1:]]))


class Output:
    def __init__(self, run_id):
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            raise ValueError("unsafe run ID")
        self.data = BASE / run_id
        self.logs = ROOT / "logs/preliminary" / STEM / run_id
        self.plots = ROOT / "plots/preliminary" / STEM / run_id
        if any(p.exists() for p in (self.data, self.logs, self.plots)):
            raise FileExistsError("output run already exists")
        for p in (self.data, self.logs, self.plots):
            p.mkdir(parents=True, exist_ok=False)
        self.tables = {}

    def row(self, table, row):
        self.tables.setdefault(table, []).append(old.normalize(copy.deepcopy(row)))

    def flush(self):
        for name, rows in self.tables.items():
            if not rows:
                continue
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with (self.logs / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: old.encode(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def source_hashes():
    return {p: old.file_hash(ROOT / p) for p in SOURCE_FILES}


def verify_design(spec):
    if old.file_hash(CONFIG) != old.file_hash(BASE / "design.lock.json"):
        raise ValueError("design changed")
    if old.read_json(BASE / "execution_sources.lock.json")["sources"] != source_hashes():
        raise ValueError("source changed")
    if spec["physical_step_cap"] != 480 or spec["horizon"] != 120:
        raise ValueError("registered cap changed")
    protected = old.read_json(BASE / "protected_inputs_before.json")
    old.verify_protected_inputs(protected)
    return protected


def sanitize_sensor_packets(packets, vo_std):
    """A strict whitelist excludes true IMU biases and UWB true ranges."""
    result = {}
    for uid in UIDS:
        packet = packets[uid]
        result[uid] = {
            "imu": {key: copy.deepcopy(packet["imu"][key]) for key in ("accel", "gyro")},
            "vo": {"valid": bool(packet["vo"]["valid"]), "delta_position": copy.deepcopy(packet["vo"].get("delta_position")),
                   "covariance": np.eye(3) * vo_std ** 2},
            "gnss": {key: copy.deepcopy(packet["gnss"][key]) for key in ("valid", "position", "std")},
        }
    return old.normalize(result)


def envelope(message):
    return {k: getattr(message, k) for k in ("source", "target", "send_time_s", "arrival_time_s", "last_propagated_time_s", "delay_s")}


def produce_record(case, spec, counters):
    original = old.read_json(old.CONFIG)
    sim = replace(old.effective_simulation(original), cooperative_ekf_enabled=False)
    env, observations, _ = old.prepare_environment(case, sim, original)
    scheduler = old.v4.WindowRoundRobin()
    initial = {u: {"state": env.ekfs[u].state.copy(), "covariance": env.ekfs[u].covariance.copy()} for u in UIDS}
    frames = []
    for step in range(1, 121):
        if counters["physical_steps"] >= spec["physical_step_cap"]:
            raise RuntimeError("physical step cap before step")
        masked = old.v4.mask_observations(observations, step)
        choices = {u: scheduler.select(masked[u], step) for u in UIDS}
        comm = old.v3.contract().communication_actions(masked, choices)
        actions, navigation = compose_fixed_navigation_actions(masked, comm)
        pre_positions = {u: env.states[u].position_m.copy() for u in UIDS}
        pre_velocities = {u: env.states[u].velocity_mps.copy() for u in UIDS}
        observations, _, terminated, truncated, info = env.step(actions)
        counters["physical_steps"] += 1
        ready = [g for g in info["measurement_gate_events"] if g["failure_stage"] == "cooperative_ekf_disabled"]
        ranges = []
        for gate in ready:
            source, target = gate["source_uav_id"], gate["target_uav_id"]
            raw = env.snapshot.uwb_measurements[(source, target)]
            ranges.append({"source": source, "target": target, "measurement": raw["measured_range"],
                           "variance": float(raw["measurement_std"]) ** 2 * (1 if raw["los"] else sim.sensor.uwb_nlos_variance_multiplier),
                           "nlos_bias": 0 if raw["los"] else sim.sensor.uwb_nlos_bias_m,
                           "measurement_id": f"{case['scene_id']}_{case['seed']}:{step}:{source}:{target}",
                           "message_envelope": envelope(env.arrived_messages[(source, target)])})
        anchor = sample_gnss(env.states["uav_00"], "nominal", indexed_rng(case["seed"], step, "diag_v6_anchor", "uav_00"), sim.sensor)
        anchor = {"uav_id": "uav_00", **{k: anchor[k] for k in ("valid", "position", "std")}}
        position = {u: env.states[u].position_m.copy() for u in UIDS}
        velocity = {u: env.states[u].velocity_mps.copy() for u in UIDS}
        minimum = min(r["distance_m"] for r in old.joint_closest_approach(pre_positions, position))
        acceleration = max(float(np.linalg.norm((velocity[u] - pre_velocities[u]) / .2)) for u in UIDS)
        if info["collision"] or minimum < 2 - 1e-8 or acceleration > 3 + 1e-7 or any(
            not np.isfinite(position[u]).all() or np.any(position[u] < [0, 0, 2]) or np.any(position[u] > [200, 200, 60]) or np.linalg.norm(velocity[u]) > 12 + 1e-8 for u in UIDS
        ):
            raise RuntimeError("native producer safety failure; no replacement run")
        if terminated or (truncated and step < 120):
            raise RuntimeError("early producer termination")
        frames.append(old.normalize({"step": step, "observable": {
            "sensor_packets": sanitize_sensor_packets(info["sensor_packets"], sim.sensor.vo_position_std_m),
            "anchor_probe": anchor, "range_attempts": sorted(ranges, key=lambda r: (r["source"], r["target"])),
            "message_envelopes": [envelope(m) for m in env.arrived_messages.values()]},
            "evaluation_truth": {"position": position, "velocity": velocity},
            "producer_audit": {"choices": choices, "actions": actions, "navigation": navigation,
                "pre_driver_estimates": {u: np.asarray(masked[u]["estimate"]).copy() for u in UIDS},
                "driver_estimates": {u: env.ekfs[u].state.copy() for u in UIDS},
                "driver_covariances": {u: env.ekfs[u].covariance.copy() for u in UIDS},
                "sent_payloads": env.transmitted_payload_history[-1], "sent_edges": env.transmitted_edge_history[-1],
                "message_events": info["message_buffer"]["events"], "budget": info["measurement_budget_events"],
                "overhead": info["measurement_overhead_events"], "gate_events": info["measurement_gate_events"],
                "minimum_pair_distance_m": minimum, "max_acceleration_mps2": acceleration,
                "boundary_count": info["boundary_projection_count"], "safety_count": info["safety_projection_count"]}}))
    return {"case": case, "initial": old.normalize(initial), "frames": frames,
            "driver": "native own-EKF fixed navigation; cooperative fusion disabled only in independent producer",
            "anchor_probe": "conditional native nominal GNSS for uav_00; estimator-only, not control", "dt_s": .2}


def propagate_packet(source_state, source_covariance, message, current_time, config):
    """Replicate native arrival propagation plus subsequent cached increments."""
    state = np.array(source_state, dtype=float, copy=True)
    covariance = np.array(source_covariance, dtype=float, copy=True)
    send, arrival = float(message["send_time_s"]), float(message["arrival_time_s"])
    if arrival < send - 1e-9 or current_time < arrival - 1e-9 or current_time - send > config.stale_message_limit_s + 1e-9:
        raise ValueError("invalid/stale envelope")
    intervals = [arrival - send]
    time = arrival
    while time < current_time - 1e-9:
        duration = min(.2, current_time - time)
        intervals.append(duration)
        time += duration
    for age in intervals:
        transition = np.eye(15)
        transition[:3, 3:6] = np.eye(3) * age
        state[:3] += state[3:6] * age
        process = np.zeros((15, 15))
        process[:3, :3] = np.eye(3) * config.process_position_std_m ** 2 * max(1.0, age)
        process[3:6, 3:6] = np.eye(3) * config.process_velocity_std_mps ** 2 * max(1.0, age)
        covariance = transition @ covariance @ transition.T + process
    return state, (covariance + covariance.T) / 2


def local_sensor_update(filters, packets, anchor, step):
    for uid, ekf in filters.items():
        ekf.begin_step()
        pre_state, pre_covariance = ekf.state.copy(), ekf.covariance.copy()
        transition = ekf.predict(packets[uid]["imu"], .2)
        vo = packets[uid]["vo"]
        if vo["valid"]:
            ekf.update_visual_delta(vo["delta_position"], np.array(vo["covariance"]), previous_state=pre_state,
                previous_covariance=pre_covariance, transition_matrix=transition, dt_s=.2,
                previous_time_s=(step - 1) * .2, current_time_s=step * .2)
        gnss = packets[uid]["gnss"]
        if gnss["valid"]:
            ekf.update_position(gnss["position"], np.eye(3) * gnss["std"] ** 2)
        if anchor is not None and uid == anchor["uav_id"] and anchor["valid"]:
            ekf.update_position(anchor["position"], np.eye(3) * anchor["std"] ** 2)


def replay(record, method, condition, spec, output=None):
    from ..analysis.cooperative_fusion_estimators_v6 import JointEKF, YoungBoundEKF
    case = record["case"]
    block = f"{case['scene_id']}_s{case['seed']}"
    sim = old.effective_simulation(old.read_json(old.CONFIG))
    states = {u: np.asarray(record["initial"][u]["state"]) for u in UIDS}
    joint = JointEKF(states, sim.ekf, initial_cross_position=2.0) if method == "joint_reference" else None
    klass = YoungBoundEKF if method == "young_bound" else LocalEKF
    filters = {u: klass(states[u].copy(), sim.ekf) for u in UIDS} if joint is None else None
    history, rows, event_rows = [], [], []
    pair_times = {}
    measurements_seen = set()
    started = perf_counter()
    status, failure = "complete", None
    for frame in record["frames"]:
        step, observable = frame["step"], frame["observable"]
        anchor = observable["anchor_probe"] if condition == "nominal_gnss_uav00" else None
        try:
            start_step = perf_counter()
            if filters is not None:
                history.append({u: (filters[u].state.copy(), filters[u].covariance.copy()) for u in UIDS})
                local_sensor_update(filters, observable["sensor_packets"], anchor, step)
            else:
                joint.predict_and_local_updates(observable["sensor_packets"], .2, anchor=anchor)
            for measurement in observable["range_attempts"]:
                identity = measurement["measurement_id"]
                if identity in measurements_seen:
                    raise ValueError("duplicate physical range")
                measurements_seen.add(identity)
                source, target = measurement["source"], measurement["target"]
                before_states = {u: joint.local_state(u) if joint is not None else filters[u].state.copy() for u in UIDS}
                if joint is not None:
                    metrics = joint.measurement_information(target, source, measurement["variance"])
                    accepted = joint.update_range(target, source, measurement["measurement"], measurement["variance"],
                        nlos_bias=measurement["nlos_bias"], measurement_id=identity)
                    event = copy.deepcopy(joint.events[-1])
                else:
                    message = measurement["message_envelope"]
                    previous = pair_times.get((source, target))
                    state_time = float(message["last_propagated_time_s"])
                    if previous is not None and state_time - previous <= sim.ekf.stale_message_limit_s:
                        raise ValueError("unexpected fresh CI cache: this independent replay cannot omit CI")
                    pair_times[(source, target)] = state_time
                    send_step = int(round(message["send_time_s"] / .2))
                    if not 0 <= send_step < len(history):
                        raise ValueError("packet points to future/missing source history")
                    state, covariance = propagate_packet(*history[send_step][source], message, step * .2, sim.ekf)
                    ekf = filters[target]
                    own_before = ekf.covariance.copy()
                    local = {"uav_id": target, "estimate": ekf.state.copy(), "covariance": own_before,
                             "neighbor_messages": {source: {"estimate": state, "covariance": covariance}}}
                    proxy = old.v3.contract().expected_relative_credit(local, source, measurement["variance"])
                    accepted = ekf.update_relative_range(measurement["measurement"], state[:3], covariance,
                        measurement_variance=measurement["variance"], nlos_bias_m=measurement["nlos_bias"], timestamp=step * .2)
                    event = copy.deepcopy(ekf.last_relative_range_update)
                    metrics = {"old_linear_credit": proxy, "reported_own_position_trace_change_m2": float(np.trace(own_before[:3, :3]) - np.trace(ekf.covariance[:3, :3]))}
                after_states = {u: joint.local_state(u) if joint is not None else filters[u].state.copy() for u in UIDS}
                event_truth = frame["evaluation_truth"]["position"]
                before_error = {u: before_states[u][:3] - np.asarray(event_truth[u]) for u in UIDS}
                after_error = {u: after_states[u][:3] - np.asarray(event_truth[u]) for u in UIDS}
                event_rows.append(old.normalize({"block_id": block, "scene_id": case["scene_id"], "seed": case["seed"],
                    "method": method, "condition": condition, "step": step, "source": source, "target": target,
                    "measurement_id": identity, "accepted": bool(accepted), "event": event, "information": metrics,
                    "packet_send_time_s": measurement["message_envelope"]["send_time_s"],
                    "packet_arrival_time_s": measurement["message_envelope"]["arrival_time_s"], "measurement_time_s": step * .2,
                    "evaluation_target_distance_improvement_m": float(np.linalg.norm(before_error[target]) - np.linalg.norm(after_error[target])),
                    "evaluation_relative_sse_improvement_m2": pair_sse(before_error) - pair_sse(after_error),
                    "evaluation_only": True}))
            local_states = {u: joint.local_state(u) if joint is not None else filters[u].state for u in UIDS}
            local_covs = {u: joint.local_covariance(u) if joint is not None else filters[u].covariance for u in UIDS}
            check_cov = joint.covariance if joint is not None else None
            for uid in UIDS:
                covariance = local_covs[uid]
                if not np.isfinite(local_states[uid]).all() or np.max(np.abs(local_states[uid])) > spec["state_magnitude_limit"] or not np.isfinite(covariance).all() or np.min(np.linalg.eigvalsh(covariance)) < -spec["psd_tolerance"] or np.trace(covariance) > spec["covariance_trace_limit"]:
                    raise FloatingPointError("registered covariance/state stop")
            if check_cov is not None and np.min(np.linalg.eigvalsh(check_cov)) < -spec["psd_tolerance"]:
                raise FloatingPointError("joint cross covariance PSD failure")
            truth = frame["evaluation_truth"]  # evaluation only, after all estimator operations
            errors = {u: local_states[u][:3] - np.asarray(truth["position"][u]) for u in UIDS}
            velocity_errors = {u: local_states[u][3:6] - np.asarray(truth["velocity"][u]) for u in UIDS}
            relative_errors, relative_nees = [], []
            for i, left in enumerate(UIDS):
                for right in UIDS[i + 1:]:
                    err = errors[left] - errors[right]
                    relative_errors.append(float(err @ err))
                    rel_cov = joint.relative_covariance(left, right) if joint is not None else 2 * (local_covs[left][:3, :3] + local_covs[right][:3, :3])
                    relative_nees.append(float(err @ np.linalg.pinv(rel_cov) @ err))
            row = {"block_id": block, "scene_id": case["scene_id"], "seed": case["seed"], "method": method,
                "condition": condition, "step": step, "position_errors": errors, "states": local_states,
                "position_squared_error_mean_m2": float(np.mean([e @ e for e in errors.values()])),
                "velocity_squared_error_mean_m2_s2": float(np.mean([e @ e for e in velocity_errors.values()])),
                "position_velocity_nees_mean": float(np.mean([np.r_[errors[u], velocity_errors[u]] @ np.linalg.pinv(local_covs[u][:6, :6]) @ np.r_[errors[u], velocity_errors[u]] for u in UIDS])),
                "distance_error_mean_m": float(np.mean([np.linalg.norm(e) for e in errors.values()])),
                "relative_squared_error_mean_m2": float(np.mean(relative_errors)),
                "position_nees_mean": float(np.mean([errors[u] @ np.linalg.pinv(local_covs[u][:3, :3]) @ errors[u] for u in UIDS])),
                "relative_nees_mean": float(np.mean(relative_nees)), "relative_nees_kind": "maintained_cross" if joint is not None else "unknown_cross_upper_bound",
                "position_covariance_trace_mean_m2": float(np.mean([np.trace(local_covs[u][:3, :3]) for u in UIDS])),
                "local_covariances": local_covs, "step_runtime_ms": (perf_counter() - start_step) * 1000,
                "local_diagnostics": joint.diagnostics_snapshot() if joint is not None else {u: filters[u].diagnostics_snapshot() for u in UIDS},
                "joint_position_covariance_9": joint.position_covariance_9() if joint is not None else None,
                "minimum_covariance_eigenvalue": min(float(np.linalg.eigvalsh(local_covs[u])[0]) for u in UIDS),
                "mean_position_error_vector_m": np.mean(list(errors.values()), axis=0),
                "joint_reference_information_advantage": joint is not None}
            driver_errors = {u: np.asarray(frame["producer_audit"]["driver_estimates"][u])[:3] - np.asarray(truth["position"][u]) for u in UIDS}
            row["source_only_driver_position_squared_error_mean_m2"] = float(np.mean([e @ e for e in driver_errors.values()]))
            row["source_only_driver_relative_squared_error_mean_m2"] = pair_sse(driver_errors)
            rows.append(old.normalize(row))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            status, failure = "censored", f"{type(exc).__name__}: {exc}"
            break
    elapsed = (perf_counter() - started) * 1000
    if output is not None:
        for row in rows:
            output.row("localization", row)
        for row in event_rows:
            output.row("fusion_events", row)
        output.row("method_status", {"block_id": block, "method": method, "condition": condition, "status": status,
            "failure": failure, "observed_steps": len(rows), "range_attempt_count": len(event_rows),
            "accepted_count": sum(r["accepted"] for r in event_rows), "total_runtime_ms": elapsed,
            "covariance_current_bytes": 45 * 45 * 8 if joint is not None else 3 * 15 * 15 * 8,
            "peak_augmented_covariance_bytes": 54 * 54 * 8 if joint is not None else 3 * 15 * 15 * 8})
    return {"rows": rows, "events": event_rows, "status": status, "failure": failure, "elapsed_ms": elapsed}


def run(run_id):
    spec = old.read_json(CONFIG)
    protected = verify_design(spec)
    output = Output(run_id)
    counters = {"physical_steps": 0, "optimizer_steps": 0, "normalization_fits": 0, "checkpoint_loads": 0}
    old.write_json(output.data / "configuration.json", spec)
    results = []
    with old._forbid_training_runtime():
        for case in spec["cases"]:
            record = produce_record(case, spec, counters)
            block = f"{case['scene_id']}_s{case['seed']}"
            old.write_json(output.data / (block + "_record.json"), record)
            record_hash = old.content_hash(record)
            output.row("records", {"block_id": block, "record_hash": record_hash, "steps": len(record["frames"]),
                "range_opportunities": sum(len(f["observable"]["range_attempts"]) for f in record["frames"]),
                "budget_consumed": sum(e["budget_consumed"] for f in record["frames"] for e in f["producer_audit"]["budget"]),
                "minimum_separation_m": min(f["producer_audit"]["minimum_pair_distance_m"] for f in record["frames"]),
                "boundary_count": sum(f["producer_audit"]["boundary_count"] for f in record["frames"]),
                "safety_count": sum(f["producer_audit"]["safety_count"] for f in record["frames"])})
            for condition in CONDITIONS:
                for method in METHODS:
                    result = replay(record, method, condition, spec, output)
                    repeat = replay(record, method, condition, spec)
                    # Timings are measured quantities, not deterministic state fields.
                    scientific = lambda r: {"rows": [{k: v for k, v in row.items() if k != "step_runtime_ms"} for row in r["rows"]],
                                            "events": r["events"], "status": r["status"], "failure": r["failure"]}
                    if old.content_hash(scientific(result)) != old.content_hash(scientific(repeat)) or old.content_hash(record) != record_hash:
                        raise RuntimeError("immutable replay mismatch")
                    results.append({"block_id": block, "method": method, "condition": condition, "status": result["status"], "steps": len(result["rows"])})
                    print(block, condition, method, result["status"], len(result["rows"]), flush=True)
            output.flush()
    old.verify_protected_inputs(protected)
    report = {"counters": counters, "results": results, "scientific_training_ready": False, "deployed": False,
              "protected_input_count": len(protected), "deterministic_replay": True,
              "joint_reference_information_advantage": "central raw local sensors/current joint states; distributed delivered stale beliefs"}
    old.write_json(output.data / "report.json", report)
    output.flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(old.encode(run(args.run_id)))


if __name__ == "__main__":
    main()
