"""Isolated partial-GNSS baseline: no policy, optimizer, or core edits."""
from __future__ import annotations

import argparse
import copy
import itertools
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from . import three_uav_cooperative_ekf_reliability_v5 as old
from . import three_uav_fusion_validation_v6 as prior
from ..control.fixed_navigation_execution import compose_fixed_navigation_actions
from ..simulation import environment as environment_module
from ..simulation.ekf import LocalEKF
from ..simulation.random_streams import indexed_rng
from ..simulation.sensor_model import sample_gnss

ROOT = old.ROOT
STEM = "three_uav_partial_gnss_baseline_v7_20261012"
BASE = ROOT / "data/synthetic" / STEM
CONFIG = ROOT / "data/configs" / (STEM + ".json")
UIDS = old.UIDS
TARGET = UIDS[2]
PEERS = UIDS[:2]
POLICIES = ("none", "fixed0", "fixed1", "random_peer")
METHODS = ("restore_prior", "measurement_ci", "joint_reference")


class PartialGNSSEnv(old.DiagnosticEnv):
    """Single-threaded sensor adapter; patch never survives step/exception."""
    def step(self, actions):
        def sensor(state, mode, rng, config):
            uid = next(u for u, value in self.states.items() if value is state)
            return sample_gnss(state, "nominal" if uid in PEERS else "denied", rng, config)
        with patch.object(environment_module, "sample_gnss", sensor):
            return super().step(actions)


def simulation(cooperative=False):
    return replace(old.effective_simulation(old.read_json(old.CONFIG)), cooperative_ekf_enabled=cooperative)


def prepare(case, spec, candidate=False):
    env = PartialGNSSEnv(simulation(candidate))
    env.reset(seed=case["seed"], scenario_config=case["scenario"])
    for uid in UIDS:
        raw = env.ekfs[uid]
        # Native state is additive world-frame p,v,Euler attitude,b_acc,b_gyro.
        raw.state += np.asarray(spec["initial_errors_by_seed"][str(case["seed"])][uid])
        if candidate and uid == TARGET:
            from ..analysis.partial_gnss_fusion_v7 import MeasurementCIEKF
            replacement = MeasurementCIEKF(raw.state.copy(), raw.config)
            replacement.covariance = raw.covariance.copy()
            env.ekfs[uid] = replacement
    env.graph_batch = env._build_graph_batch()
    return env, env._observations()


def choose(policy, legal, seed, step):
    legal = tuple(u for u in PEERS if u in legal)
    if policy == "none" or not legal:
        return "none"
    if policy.startswith("fixed"):
        wanted = PEERS[int(policy[-1])]
        return wanted if wanted in legal else "none"
    if policy == "random_peer":
        return str(indexed_rng(seed, step, "v7_random_peer", TARGET).choice(legal))
    raise ValueError(policy)


def check_safety(env, info, previous_positions, previous_velocities):
    positions = {u: env.states[u].position_m.copy() for u in UIDS}
    velocities = {u: env.states[u].velocity_mps.copy() for u in UIDS}
    closest = min(r["distance_m"] for r in old.joint_closest_approach(previous_positions, positions))
    acceleration = max(np.linalg.norm((velocities[u] - previous_velocities[u]) / .2) for u in UIDS)
    if info["collision"] or closest < 2 - 1e-8 or acceleration > 3 + 1e-7:
        raise RuntimeError("registered physical safety stop")
    for u in UIDS:
        if not np.isfinite(positions[u]).all() or np.any(positions[u] < [0, 0, 2]) or np.any(positions[u] > [200, 200, 60]) or np.linalg.norm(velocities[u]) > 12 + 1e-8:
            raise RuntimeError("registered physical bounds stop")
        p = env.ekfs[u].covariance
        if not np.isfinite(p).all() or np.linalg.eigvalsh(p)[0] < -1e-8 or np.trace(p) > 1e4:
            raise RuntimeError("registered estimator stop")
    return positions, velocities, float(closest), float(acceleration)


def run_physical(case, spec, policy="none", candidate=False):
    env, observations = prepare(case, spec, candidate)
    initial = {u: {"state": env.ekfs[u].state.copy(), "covariance": env.ekfs[u].covariance.copy()} for u in UIDS}
    frames = []
    failure = None
    tracker = old.StrictParkingTracker()
    for step in range(1, 121):
        masked = old.v4.mask_observations(observations, step)
        legal = old.legal_sources(masked[TARGET])
        source = choose(policy, legal, case["seed"], step)
        choices = {u: source if u == TARGET else "none" for u in UIDS}
        comm = old.v3.contract().communication_actions(masked, choices)
        actions, navigation = compose_fixed_navigation_actions(masked, comm)
        previous_positions = {u: env.states[u].position_m.copy() for u in UIDS}
        previous_velocities = {u: env.states[u].velocity_mps.copy() for u in UIDS}
        observations, _, terminated, truncated, info = env.step(actions)
        try:
            positions, velocities, closest, acceleration = check_safety(env, info, previous_positions, previous_velocities)
        except RuntimeError as exc:
            positions = {u: env.states[u].position_m.copy() for u in UIDS}
            velocities = {u: env.states[u].velocity_mps.copy() for u in UIDS}
            closest = min(r["distance_m"] for r in old.joint_closest_approach(previous_positions, positions))
            acceleration = max(float(np.linalg.norm((velocities[u]-previous_velocities[u])/.2)) for u in UIDS)
            failure = {"reason": str(exc), "step": step, "collision": info["collision"], "minimum_separation_m": closest,
                       "max_acceleration_mps2": acceleration, "boundary_count": info["boundary_projection_count"]}
        if (terminated or (truncated and step < 120)) and failure is None:
            raise RuntimeError("unexpected early termination")
        bank = {}
        for peer in PEERS:
            key = (peer, TARGET)
            raw = env.snapshot.uwb_measurements.get(key)
            msg = env.arrived_messages.get(key)
            if raw is not None and msg is not None and key in env.snapshot.effective_edges and raw.get("valid", False):
                bank[peer] = {"source": peer, "measurement": raw["measured_range"],
                    "variance": float(raw["measurement_std"]) ** 2 * (1 if raw["los"] else env.config.sensor.uwb_nlos_variance_multiplier),
                    "nlos_bias": 0 if raw["los"] else env.config.sensor.uwb_nlos_bias_m,
                    "message_envelope": prior.envelope(msg), "packet_state": msg.state.copy(), "packet_covariance": msg.covariance.copy()}
        distances = {u: float(np.linalg.norm(positions[u] - env.goals[u])) for u in UIDS}
        speeds = {u: float(np.linalg.norm(velocities[u])) for u in UIDS}
        parking = tracker.update(distances, speeds, bool(info["collision"]))
        packets = prior.sanitize_sensor_packets(info["sensor_packets"], env.config.sensor.vo_position_std_m)
        if packets[TARGET]["gnss"]["valid"] or not all(packets[p]["gnss"]["valid"] for p in PEERS):
            raise AssertionError("partial GNSS observation contract")
        frames.append(old.normalize({"step": step, "observable": {"sensor_packets": packets, "range_bank": bank,
            "pre_legal_sources": list(legal)}, "evaluation_truth": {"position": positions, "velocity": velocities},
            "audit": {"selection": source, "actions": actions, "navigation": navigation,
                "pre_estimates": {u: masked[u]["estimate"] for u in UIDS},
                "states": {u: env.ekfs[u].state.copy() for u in UIDS}, "covariances": {u: env.ekfs[u].covariance.copy() for u in UIDS},
                "diagnostics": {u: env.ekfs[u].diagnostics_snapshot() for u in UIDS},
                "fusion_event": env.ekfs[TARGET].last_relative_range_update if candidate and source != "none" else {},
                "budget": info["measurement_budget_events"], "overhead": info["measurement_overhead_events"],
                "gate_events": info["measurement_gate_events"], "sent_edges": env.transmitted_edge_history[-1],
                "minimum_separation_m": closest, "max_acceleration_mps2": acceleration,
                "boundary_count": info["boundary_projection_count"], "safety_count": info["safety_projection_count"],
                "distances": distances, "speeds": speeds, "parking": parking}}))
        if failure is not None:
            break
    return {"case": case, "initial": old.normalize(initial), "frames": frames,
            "policy": policy, "kind": "candidate_closed_loop" if candidate else "none_producer", "dt_s": .2,
            "status": "censored" if failure else "complete", "failure": failure}


def replay(record, method, policy):
    from ..analysis.partial_gnss_fusion_v7 import MeasurementCIEKF, PartialGNSSJointEKF
    sim = simulation()
    filters = {u: (MeasurementCIEKF if u == TARGET and method == "measurement_ci" else LocalEKF)(np.array(record["initial"][u]["state"]), sim.ekf) for u in UIDS}
    joint = PartialGNSSJointEKF({u: filters[u].state.copy() for u in UIDS}, sim.ekf, initial_cross_position=0) if method == "joint_reference" else None
    history, rows, events = [], [], []
    used = 0
    pair_time, seen_packet = {}, {}
    source_match, driver_match, packet_match = 0., 0., 0.
    for frame in record["frames"]:
        step = frame["step"]
        history.append({u: (filters[u].state.copy(), filters[u].covariance.copy()) for u in UIDS})
        packets = frame["observable"]["sensor_packets"]
        prior.local_sensor_update(filters, packets, None, step)
        if joint is not None:
            joint.predict_and_local_updates(packets, .2)
        for uid in UIDS:
            source_match = max(source_match, float(np.max(np.abs(filters[uid].state - np.asarray(frame["audit"]["states"][uid]))))) if uid in PEERS else source_match
        if policy == "none":
            driver_match = max(driver_match, max(float(np.max(np.abs(filters[u].state - np.array(frame["audit"]["states"][u])))) for u in UIDS))
        legal = [u for u in frame["observable"]["pre_legal_sources"] if u in PEERS and used < 30
                 and (u not in pair_time or step * .2 - pair_time[u] >= .8 - 1e-9)]
        source = choose(policy, legal, record["case"]["seed"], step)
        selected = source != "none"
        if selected:
            used += 1
        bank = frame["observable"]["range_bank"]
        if selected and source in bank:
            measurement = bank[source]
            envelope = measurement["message_envelope"]
            age = step * .2 - envelope["send_time_s"]
            fresh = seen_packet.get(source) != envelope["send_time_s"]
            if fresh and age <= .6 + 1e-9 and (source not in pair_time or step * .2 - pair_time[source] >= .8 - 1e-9):
                seen_packet[source] = envelope["send_time_s"]
                pair_time[source] = step * .2
                send_step = int(round(envelope["send_time_s"] / .2))
                state, covariance = prior.propagate_packet(*history[send_step][source], envelope, step * .2, sim.ekf)
                packet_match = max(packet_match, float(np.max(np.abs(state - np.array(measurement["packet_state"])))), float(np.max(np.abs(covariance - np.array(measurement["packet_covariance"])))))
                before = joint.local_state(TARGET) if joint is not None else filters[TARGET].state.copy()
                p_before = joint.local_covariance(TARGET) if joint is not None else filters[TARGET].covariance.copy()
                if joint is None:
                    accepted = filters[TARGET].update_relative_range(measurement["measurement"], state[:3], covariance,
                        measurement_variance=measurement["variance"], nlos_bias_m=measurement["nlos_bias"], timestamp=step * .2)
                    event = copy.deepcopy(filters[TARGET].last_relative_range_update)
                else:
                    accepted = joint.update_range(TARGET, source, measurement["measurement"], measurement["variance"],
                        nlos_bias=measurement["nlos_bias"], measurement_id=f"{step}:{source}")
                    event = copy.deepcopy(joint.events[-1])
                after = joint.local_state(TARGET) if joint is not None else filters[TARGET].state.copy()
                p_after = joint.local_covariance(TARGET) if joint is not None else filters[TARGET].covariance.copy()
                truth = np.array(frame["evaluation_truth"]["position"][TARGET])
                events.append({"step": step, "source": source, "accepted": bool(accepted), "packet_age_s": age, "event": event,
                    "evaluation_sse_gain_m2": float(np.sum((before[:3]-truth)**2)-np.sum((after[:3]-truth)**2)),
                    "execution_position_trace_gain_m2": float(np.trace(p_before[:3,:3])-np.trace(p_after[:3,:3]))})
        states = {u: joint.local_state(u) if joint is not None else filters[u].state.copy() for u in UIDS}
        covs = {u: joint.local_covariance(u) if joint is not None else filters[u].covariance.copy() for u in UIDS}
        row = metrics(states, covs, frame, joint)
        row.update({"step": step, "source": source, "budget_used": used})
        row["gnss_accepted_0_1_2"] = [sum(e["uid"] == u and e["accepted"] for e in joint.gnss_update_events) for u in UIDS] if joint is not None else [filters[u].diagnostics_snapshot()["gnss_update_accepted"] for u in UIDS]
        rows.append(row)
    if source_match > 1e-8 or packet_match > 1e-8 or driver_match > 1e-8:
        raise AssertionError(f"record reconstruction mismatch {source_match, packet_match, driver_match}")
    return rows, events, {"source_reconstruction_max": source_match, "packet_reconstruction_max": packet_match, "none_driver_reconstruction_max": driver_match}


def metrics(states, covariances, frame, joint=None):
    truth = frame["evaluation_truth"]["position"]
    errors = {u: np.asarray(states[u])[:3] - truth[u] for u in UIDS}
    covariance = np.asarray(covariances[TARGET])
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance)[0] < -1e-8 or np.trace(covariance) > 1e4:
        raise FloatingPointError("registered covariance stop")
    pair_errors = [errors[a] - errors[b] for a, b in itertools.combinations(UIDS, 2)]
    return old.normalize({"target_sse_m2": float(errors[TARGET] @ errors[TARGET]),
        "relative_sse_mean_m2": float(np.mean([e @ e for e in pair_errors])),
        "target_position_nees": old.nees(errors[TARGET], covariance[:3, :3]),
        "target_covariance_trace_m2": float(np.trace(covariance[:3, :3])),
        "target_position_eigenvalues_m2": np.linalg.eigvalsh(covariance[:3, :3]),
        "target_position_error_m": errors[TARGET], "states": states, "covariances": covariances})


def summary(record, rows, events, method, policy, phase, elapsed):
    accepted = sum(e["accepted"] for e in events)
    nis = [e["event"].get("nis") for e in events if e["event"].get("nis") is not None]
    frames = record["frames"]
    requests = rows[-1]["budget_used"] if phase == "replay" else sum(f["audit"]["selection"] != "none" for f in frames)
    return {"scene_id": record["case"]["scene_id"], "seed": record["case"]["seed"], "phase": phase,
        "method": method, "policy": policy, "steps": len(rows),
        "uav2_absolute_rmse_m": float(np.sqrt(np.mean([r["target_sse_m2"] for r in rows]))),
        "cluster_relative_rmse_m": float(np.sqrt(np.mean([r["relative_sse_mean_m2"] for r in rows]))),
        "uav2_position_nees_mean": float(np.mean([r["target_position_nees"] for r in rows])),
        "uwb_nis_mean": float(np.mean(nis)) if nis else None,
        "final_uav2_covariance_trace_m2": rows[-1]["target_covariance_trace_m2"],
        "max_uav2_covariance_trace_m2": max(r["target_covariance_trace_m2"] for r in rows),
        "uwb_requested": requests, "uwb_attempted": len(events), "uwb_accepted": accepted,
        "measurement_bytes_equivalent": requests * 512,
        "broadcast_edge_transmissions": sum(len(f["audit"]["sent_edges"]) for f in frames),
        "gnss_valid_0_1_2": [sum(bool(f["observable"]["sensor_packets"][u]["gnss"]["valid"]) for f in frames) for u in UIDS],
        "gnss_accepted_0_1_2": [sum(r["gnss_accepted_0_1_2"][i] for r in rows) for i in range(3)] if "gnss_accepted_0_1_2" in rows[0] else [sum(f["audit"]["diagnostics"][u]["gnss_update_accepted"] for f in frames) for u in UIDS],
        "minimum_separation_m": min(f["audit"]["minimum_separation_m"] for f in frames),
        "boundary_projections": sum(f["audit"]["boundary_count"] for f in frames),
        "safety_projections": sum(f["audit"]["safety_count"] for f in frames), "collision_count": 0,
        "uav2_final_goal_distance_m": frames[-1]["audit"]["distances"][TARGET] if phase == "closed_loop" else None,
        "uav2_final_speed_mps": frames[-1]["audit"]["speeds"][TARGET] if phase == "closed_loop" else None,
        "uav2_strict_parking": frames[-1]["audit"]["parking"]["per_uav"][TARGET]["ever_success"] if phase == "closed_loop" else None,
        "elapsed_ms": elapsed,
        "joint_privilege": "all current private IMU/VO/GNSS and cross covariance; same selected UWB" if method == "joint_reference" else "none",
        "communication_model": "None-producer broadcast/queue frozen; request cost accounting only" if phase == "replay" else "native live queue"}


def freeze():
    if CONFIG.exists() or BASE.exists():
        raise FileExistsError("v7 design already exists")
    scenes = {
        "train_noncollinear": [[40, 40, 22], [40, 90, 42], [65, 65, 32]],
        "train_collinear": [[40, 125, 28], [50, 125, 28], [80, 125, 28]],
    }
    cases = []
    for name, starts in scenes.items():
        scenario = {"initial_positions": dict(zip(UIDS, starts)), "initial_velocities": {u: [0, 0, 0] for u in UIDS},
            "initial_attitudes": {u: [0, 0, 0] for u in UIDS},
            "goals": {u: (np.array(p) + [35, 0, 0]).tolist() for u, p in zip(UIDS, starts)}, "obstacles": [], "obstacle_specs": []}
        for seed in (20267101, 20267102):
            cases.append({"scene_id": name, "seed": seed, "split": "train_diagnostic", "scenario": scenario})
    errors = {str(s): {u: (indexed_rng(s, 0, "v7_initial_error", u).normal(size=15) * np.sqrt([4]*3+[1]*12)).tolist() for u in UIDS} for s in (20267101, 20267102)}
    spec = {"schema": "partial_gnss_baseline_v7", "cases": cases, "initial_errors_by_seed": errors,
        "initial_covariance_diagonal": [4]*3+[1]*12, "initial_cross_covariance": 0,
        "physical_step_cap": 2400, "replay_filter_step_cap": 5760,
        "gnss_modes": {"uav_00": "nominal", "uav_01": "nominal", "uav_02": "denied"},
        "gnss_std_m": .5, "decision_steps": list(range(2, 119, 4)), "measurement_budget": 30,
        "closed_loop_gate": "random CI improves None in both geometry pooled MSE and >=3/4 blocks; PSD finite; maxfulltrace<1e4; finalmeanPpos<12",
        "closed_loop_policies": list(POLICIES), "covariance_limit": 1e4, "training_allowed": False}
    old.write_json(CONFIG, spec)
    old.write_json(BASE / "design.lock.json", spec)
    protected = old.read_json(prior.BASE / "protected_inputs_before.json")
    protected.update({str(ROOT / p): h for p, h in old.read_json(prior.BASE / "final_artifact_manifest.json")["artifact_sha256"].items() if p != "README.md"})
    for folder in ("data", "logs", "plots"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and prior.STEM in str(p):
                protected[str(p)] = old.file_hash(p)
    if any(not Path(p).is_file() or old.file_hash(p) != h for p, h in protected.items()):
        raise AssertionError("history has changed before v7")
    old.write_json(BASE / "protected_inputs_before.json", protected)
    (BASE / "README_before.md.txt").write_bytes((ROOT / "README.md").read_bytes())
    old.write_json(BASE / "design.sha256.json", {"config_sha256": old.file_hash(CONFIG)})
    print(f"Frozen {len(cases)} cases; {len(protected)} protected files")


def run(run_id, resume_from=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("invalid run-id")
    spec = old.read_json(CONFIG)
    if old.file_hash(CONFIG) != old.read_json(BASE / "design.sha256.json")["config_sha256"]:
        raise RuntimeError("design changed")
    output = prior.Output.__new__(prior.Output)
    output.data = BASE / run_id
    output.logs = ROOT / "logs/preliminary" / STEM / run_id
    output.plots = ROOT / "plots/preliminary" / STEM / run_id
    for p in (output.data, output.logs, output.plots):
        if p.exists():
            raise FileExistsError(p)
        p.mkdir(parents=True)
    output.tables = {}
    old.write_json(output.data / "execution_sources.json", {name: old.file_hash(ROOT/name) for name in (*prior.SOURCE_FILES,
        "src/experiments/partial_gnss_baseline_v7.py", "src/analysis/partial_gnss_fusion_v7.py")})
    records, summaries, failures = [], [], []
    physical_steps = replay_steps = 0
    completed = {}
    if resume_from is not None:
        original = BASE / resume_from
        for path in original.glob("record_*.json"):
            record = old.read_json(path)
            completed[(record["case"]["scene_id"],record["case"]["seed"])] = record
        # The original failed trajectory has a separately recovered prefix;
        # retain its failure and do not execute this condition again.
        recovered = old.read_json(BASE/"safety_prefix_s20267102/failure.json")
        failures.append({"scene_id":recovered["case"]["scene_id"],"seed":recovered["case"]["seed"],
            "steps":recovered["steps"],"reason":recovered["reason"],"diagnostic_prefix":str(BASE/"safety_prefix_s20267102/failure.json")})
        physical_steps += recovered["steps"]*2 # failed original plus diagnostic reproduction
    with old._forbid_training_runtime():
        for case in spec["cases"]:
            key=(case["scene_id"],case["seed"])
            if any((f["scene_id"],f["seed"])==key for f in failures):
                continue
            record = completed.get(key)
            if record is None:
                record = run_physical(case, spec)
            physical_steps += len(record["frames"])
            old.write_json(output.data / f"record_{case['scene_id']}_s{case['seed']}.json", record)
            if record.get("status","complete") != "complete":
                failures.append({"scene_id":case["scene_id"],"seed":case["seed"],"steps":len(record["frames"]),**record["failure"]})
                print(f"Censored {case['scene_id']} seed {case['seed']}: {record['failure']}",flush=True)
                continue
            records.append(record)
            for method, policy in itertools.product(METHODS, POLICIES):
                started = perf_counter()
                rows, events, audit = replay(record, method, policy)
                replay_steps += len(rows)
                item = summary(record, rows, events, method, policy, "replay", (perf_counter()-started)*1000)
                item.update(audit)
                summaries.append(item)
                for row in rows:
                    output.row("localization", {**{k: item[k] for k in ("scene_id", "seed", "method", "policy", "phase")}, **row})
                for event in events:
                    output.row("fusion_events", {**{k: item[k] for k in ("scene_id", "seed", "method", "policy", "phase")}, **event})
                output.row("summary", item)
            output.flush()
            print(f"Replay complete {case['scene_id']} seed {case['seed']}", flush=True)
        controls = {(r["scene_id"], r["seed"]): r for r in summaries if r["method"] == "measurement_ci" and r["policy"] == "none"}
        selected = [r for r in summaries if r["method"] == "measurement_ci" and r["policy"] == "random_peer"]
        improvements = [r["uav2_absolute_rmse_m"] < controls[(r["scene_id"], r["seed"])]["uav2_absolute_rmse_m"] for r in selected]
        gate = not failures and sum(improvements) >= 3 and np.mean([r["final_uav2_covariance_trace_m2"] for r in selected]) < 12
        for scene in set(r["scene_id"] for r in selected):
            subset = [r for r in selected if r["scene_id"] == scene]
            gate &= sum(r["uav2_absolute_rmse_m"]**2 for r in subset) < sum(controls[(scene,r["seed"])]["uav2_absolute_rmse_m"]**2 for r in subset)
        if gate:
            for case, policy in itertools.product(spec["cases"], POLICIES):
                started = perf_counter()
                record = run_physical(case, spec, policy, True)
                physical_steps += len(record["frames"])
                old.write_json(output.data / f"closed_{case['scene_id']}_s{case['seed']}_{policy}.json", record)
                if record["status"] != "complete":
                    failures.append({"scene_id":case["scene_id"],"seed":case["seed"],"policy":policy,"phase":"closed_loop",**record["failure"]})
                    break
                rows, events = [], []
                for frame in record["frames"]:
                    row = metrics(frame["audit"]["states"], frame["audit"]["covariances"], frame)
                    row.update({"step": frame["step"], "budget_used": sum(f["audit"]["selection"] != "none" for f in record["frames"][:frame["step"]])})
                    rows.append(row)
                    gates = frame["audit"]["gate_events"]
                    event = frame["audit"]["fusion_event"]
                    if any(g["ekf_update_attempted"] for g in gates):
                        events.append({"step": frame["step"], "source": frame["audit"]["selection"], "accepted": bool(event.get("accepted")), "event": event})
                item = summary(record, rows, events, "measurement_ci", policy, "closed_loop", (perf_counter()-started)*1000)
                summaries.append(item)
                output.row("summary", item)
                for row in rows:
                    output.row("localization", {**{k: item[k] for k in ("scene_id", "seed", "method", "policy", "phase")}, **row})
                for event in events:
                    output.row("fusion_events", {**{k: item[k] for k in ("scene_id", "seed", "method", "policy", "phase")}, **event})
                output.flush()
                print(f"Closed loop complete {case['scene_id']} {case['seed']} {policy}", flush=True)
        if physical_steps > spec["physical_step_cap"] or replay_steps > spec["replay_filter_step_cap"]:
            raise AssertionError("execution cap")
    protected = old.read_json(BASE / "protected_inputs_before.json")
    changed = [p for p,h in protected.items() if old.file_hash(p) != h]
    if changed:
        raise AssertionError(changed)
    old.write_json(output.data / "manifest.json", {"summaries": summaries, "closed_loop_gate_passed": bool(gate),
        "physical_steps": physical_steps, "replay_filter_steps": replay_steps, "protected_count": len(protected),
        "protected_changed": changed, "optimizer_updates": 0, "scientific": {"training_ready": False},
        "config_sha256": old.file_hash(CONFIG), "failures": failures, "resume_from": resume_from,
        "physical_count_scope": "includes original completed producers and failed/recovered prefix, excludes replay"})
    print(f"Finished: physical={physical_steps}; replay={replay_steps}; closed-loop gate={gate}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run-id", default="bounded_s20267101_102")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    freeze() if args.freeze else run(args.run_id,args.resume_from)


if __name__ == "__main__":
    main()
