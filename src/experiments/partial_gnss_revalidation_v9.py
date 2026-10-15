"""Isolated v9 paired replay and gated live-queue revalidation; never trains."""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from . import partial_gnss_baseline_v7 as v7
from ..analysis import local_ekf_startup_diagnostic_v8 as v8
from ..analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
from ..analysis.partial_gnss_report_v7 import write_csv

ROOT = v7.ROOT
STEM = "three_uav_partial_gnss_revalidation_v9_20261013"
BASE = ROOT / "data/synthetic" / STEM
LOGS = ROOT / "logs/preliminary" / STEM
CONFIG = ROOT / "data/configs" / (STEM + ".json")
SOURCES = ("src/experiments/partial_gnss_revalidation_v9.py",
           "src/analysis/partial_gnss_replay_v9.py", "src/analysis/partial_gnss_gate_v9.py",
           "src/analysis/partial_gnss_ci_diagnostic_v9.py", "src/analysis/partial_gnss_report_v9.py")


def freeze():
    """Only copies existing cases and priors; no simulator or random draw."""
    if CONFIG.exists() or BASE.exists():
        raise FileExistsError("v9 design exists")
    original = v7.old.read_json(v8.CONFIG)
    extra = v7.old.read_json(v8.BASE / "extra_design.lock.json")
    spec = {"schema": "partial_gnss_revalidation_v9", "cases": original["original_cases"] + extra["cases"],
            "initial_errors_by_seed": {**original["initial_errors_by_seed"], **extra["initial_errors_by_seed"]},
            "startup_steps": original["startup_steps"], "motion_vo_chi_square_limit": original["motion_vo_chi_square_limit"],
            "dt_s": .2, "horizon_steps": 120, "decision_steps": list(range(6, 119, 4)),
            "measurement_budget": 30, "message_ttl_s": .6, "pair_period_s": .8,
            "physical_step_cap": 1920, "none_physical_step_cap": 960, "replay_filter_step_cap": 11520,
            "methods": list(v7.METHODS), "policies": list(v7.POLICIES), "closed_loop_policy": "random_peer",
            "gate": {"positive_seeds_per_geometry": 3, "pooled_rmse_improvement_fraction": .05,
                     "maximum_case_degradation_fraction": .05, "reference_rmse_limit_m": 1.,
                     "reference_gnss_acceptance_min": .9, "covariance_full_trace_limit": 1e4,
                     "target_nees_mean_limit": 7.815, "effective_fusion_rate_min": .2},
            "gate_scope": "engineering admission, not chi-square significance or general consistency proof",
            "frozen_state_steps": [6, 34, 58], "frozen_state_update_cap": 96,
            "training_allowed": False, "optimizer_updates": 0,
            "replay_limit": "None physical broadcast and queue held fixed; selected request overhead accounted only",
            "joint_advantage": "current private IMU/VO/GNSS and maintained cross covariance after equal startup",
            "independent_replicates": 4, "geometry_is_within_seed_factor": True}
    protected = v7.old.read_json(v8.BASE / "protected_inputs_before.json")
    protected.update({str(ROOT / p): h for p, h in v7.old.read_json(v8.BASE / "final_artifact_manifest.json")["artifact_sha256"].items() if p != "README.md"})
    protected[str(v8.BASE / "final_artifact_manifest.json")] = v7.old.file_hash(v8.BASE / "final_artifact_manifest.json")
    for path, digest in protected.items():
        if v7.old.file_hash(path) != digest:
            raise RuntimeError(f"history differs before v9: {path}")
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE / "README_before.md.txt").write_bytes((ROOT / "README.md").read_bytes())
    v7.old.write_json(CONFIG, spec)
    v7.old.write_json(BASE / "design.lock.json", spec)
    v7.old.write_json(BASE / "protected_inputs_before.json", protected)
    v7.old.write_json(BASE / "design.sha256.json", {"config_sha256": v7.old.file_hash(CONFIG),
        "source_sha256": {p: v7.old.file_hash(ROOT / p) for p in SOURCES}})
    print(f"Frozen 8 cases, 960 None physical steps + conditional 960 live steps, 11520 replay steps; {len(protected)} protected files", flush=True)


def verify_design():
    spec = v7.old.read_json(CONFIG)
    lock = v7.old.read_json(BASE / "design.sha256.json")
    if v7.old.file_hash(CONFIG) != lock["config_sha256"]:
        raise RuntimeError("v9 configuration changed")
    for p, h in lock["source_sha256"].items():
        if v7.old.file_hash(ROOT / p) != h:
            raise RuntimeError(f"v9 execution source changed: {p}")
    for p, h in v7.old.read_json(BASE / "protected_inputs_before.json").items():
        if v7.old.file_hash(p) != h:
            raise RuntimeError(f"historical input changed: {p}")
    return spec


def episode(case, spec, policy="none"):
    from ..analysis.partial_gnss_replay_v9 import StartupMeasurementCI
    env, observations = v7.prepare(case, spec)
    if policy != "none":
        from dataclasses import replace
        env.config = replace(env.config, cooperative_ekf_enabled=True)
    for uid, raw in list(env.ekfs.items()):
        klass = StartupMeasurementCI if uid == v7.TARGET else StartupLocalEKF
        candidate = klass(raw.state.copy(), raw.config, sensor_config=env.config.sensor,
                          startup_steps=spec["startup_steps"], vo_motion_limit=spec["motion_vo_chi_square_limit"])
        candidate.covariance = raw.covariance.copy()
        env.ekfs[uid] = candidate
    env.graph_batch = env._build_graph_batch()
    observations = env._observations()
    initial = {u: {"state": env.ekfs[u].state.copy(), "covariance": env.ekfs[u].covariance.copy()} for u in v7.UIDS}
    frames, failure = [], None
    parking = v7.old.StrictParkingTracker()
    for step in range(1, 121):
        masked = v7.old.v4.mask_observations(observations, step)
        legal = v7.old.legal_sources(masked[v7.TARGET]) if step in spec["decision_steps"] else []
        source = v7.choose(policy, legal, case["seed"], step)
        communication = v7.old.v3.contract().communication_actions(masked, {u: source if u == v7.TARGET else "none" for u in v7.UIDS})
        actions, navigation = v7.compose_fixed_navigation_actions(masked, communication)
        requested = copy.deepcopy(actions)
        if step <= 5:
            for command in actions.values():
                command["delta_v"] = np.zeros(3)
            assert source == "none"
        pre_p = {u: env.states[u].position_m.copy() for u in v7.UIDS}
        pre_v = {u: env.states[u].velocity_mps.copy() for u in v7.UIDS}
        try:
            observations, _, terminated, truncated, info = env.step(actions)
            if step == 5:
                for ekf in env.ekfs.values():
                    ekf.check_startup_evidence()
            try:
                positions, velocities, separation, acceleration = v7.check_safety(env, info, pre_p, pre_v)
            except RuntimeError as exc:
                failure = {"step": step, "reason": str(exc)}
                positions = {u: env.states[u].position_m.copy() for u in v7.UIDS}
                velocities = {u: env.states[u].velocity_mps.copy() for u in v7.UIDS}
                separation = min(r["distance_m"] for r in v7.old.joint_closest_approach(pre_p, positions))
                acceleration = max(float(np.linalg.norm((velocities[u]-pre_v[u])/.2)) for u in v7.UIDS)
            if (terminated or (truncated and step < 120)) and failure is None:
                failure = {"step": step, "reason": "unexpected early termination"}
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            failure = {"step": step, "reason": str(exc), "incomplete_transition": True}
            frames.append({"step": step, "failure": failure})
            break
        bank, conditions = {}, {}
        for peer in v7.PEERS:
            key = (peer, v7.TARGET)
            raw = env.snapshot.uwb_measurements.get(key)
            msg = env.arrived_messages.get(key)
            conditions[peer] = {"uwb_valid": bool(raw and raw.get("valid", False)),
                                "effective_edge": key in env.snapshot.effective_edges,
                                "message_arrived": msg is not None, "potential_edge": key in env.snapshot.potential_edges}
            if raw is not None and msg is not None and key in env.snapshot.effective_edges and raw.get("valid", False):
                bank[peer] = {"source": peer, "measurement": raw["measured_range"],
                             "variance": float(raw["measurement_std"])**2 * (1 if raw["los"] else env.config.sensor.uwb_nlos_variance_multiplier),
                             "nlos_bias": 0 if raw["los"] else env.config.sensor.uwb_nlos_bias_m,
                             "message_envelope": v7.prior.envelope(msg), "packet_state": msg.state.copy(), "packet_covariance": msg.covariance.copy()}
        truth = {"position": positions, "velocity": velocities,
                 "accel_bias": {u: env.accel_bias[u].copy() for u in v7.UIDS}, "gyro_bias": {u: env.gyro_bias[u].copy() for u in v7.UIDS}}
        distances = {u: float(np.linalg.norm(positions[u]-env.goals[u])) for u in v7.UIDS}
        speeds = {u: float(np.linalg.norm(velocities[u])) for u in v7.UIDS}
        parked = parking.update(distances, speeds, bool(info["collision"]))
        packets = v7.prior.sanitize_sensor_packets(info["sensor_packets"], env.config.sensor.vo_position_std_m)
        if packets[v7.TARGET]["gnss"]["valid"] or not all(packets[p]["gnss"]["valid"] for p in v7.PEERS):
            raise AssertionError("partial GNSS violated")
        attempted = any(g["ekf_update_attempted"] for g in info["measurement_gate_events"])
        frames.append(v7.old.normalize({"step": step,
            "observable": {"sensor_packets": packets, "range_bank": bank, "pre_legal_sources": list(legal), "post_conditions": conditions},
            "evaluation_truth": truth, "audit": {"selection": source, "actions": actions, "requested_actions": requested,
                "navigation": navigation, "states": {u: env.ekfs[u].state.copy() for u in v7.UIDS},
                "covariances": {u: env.ekfs[u].covariance.copy() for u in v7.UIDS},
                "diagnostics": {u: env.ekfs[u].diagnostics_snapshot() for u in v7.UIDS},
                "fusion_event": copy.deepcopy(env.ekfs[v7.TARGET].last_relative_range_update) if attempted else {},
                "cache_ci_weight": env.ci_weights.get((v7.TARGET, source)),
                "gate_events": info["measurement_gate_events"], "budget": info["measurement_budget_events"],
                "budget_remaining": info["measurement_budget_remaining"], "overhead": info["measurement_overhead_events"],
                "sent_edges": env.transmitted_edge_history[-1], "minimum_separation_m": separation,
                "max_acceleration_mps2": acceleration, "collision": info["collision"],
                "boundary_count": info["boundary_projection_count"], "safety_count": info["safety_projection_count"],
                "parking": parked, "distances": distances, "speeds": speeds}}))
        if failure:
            break
    return {"case": case, "initial": v7.old.normalize(initial), "frames": frames,
            "policy": policy, "kind": "none_producer" if policy == "none" else "ci_live_queue_closed_loop",
            "dt_s": .2, "status": "censored" if failure else "complete", "failure": failure}


def outputs(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("unsafe run id")
    data, logs = BASE / run_id, LOGS / run_id
    for p in (data, logs):
        p.mkdir(parents=True, exist_ok=False)
    return data, logs


def run(run_id):
    from ..analysis.partial_gnss_replay_v9 import replay
    from ..analysis.partial_gnss_gate_v9 import summarize, admission
    spec = verify_design()
    reserve_physical(run_id, spec["none_physical_step_cap"], spec)
    data, logs = outputs(run_id)
    summaries, physical_count, replay_count = [], 0, 0
    with v7.old._forbid_training_runtime():
        for case in spec["cases"]:
            if physical_count + 120 > spec["none_physical_step_cap"]:
                raise RuntimeError("physical step cap")
            record = episode(case, spec)
            physical_count += len(record["frames"])
            block = f"{case['scene_id']}_s{case['seed']}"
            folder = data / block
            folder.mkdir(exist_ok=False)
            v7.old.write_json(folder / "none_record.json", record)
            print(block, record["status"], "physical", len(record["frames"]), flush=True)
            if record["status"] != "complete":
                summaries.append({"scene_id": case["scene_id"], "seed": case["seed"], "method": "producer_failure", "policy": "none", "failure": record["failure"]})
                continue
            for method in spec["methods"]:
                for policy in spec["policies"]:
                    if replay_count + 120 > spec["replay_filter_step_cap"]:
                        raise RuntimeError("replay step cap")
                    started = perf_counter()
                    rows, events, audit = replay(record, method, policy, spec)
                    replay_count += len(rows)
                    label = f"{method}_{policy}"
                    v7.old.write_json(folder / (label + ".json"), {"rows": rows, "events": events, "audit": audit})
                    result = summarize(record, rows, events, audit, method, policy)
                    result["elapsed_ms"] = 1000 * (perf_counter()-started)
                    summaries.append(result)
                    write_csv(logs / f"{block}_{label}_steps.csv", rows)
                    write_csv(logs / f"{block}_{label}_events.csv", events)
            print(block, "12 paired replays complete", flush=True)
    gate = admission(summaries, spec)
    write_csv(logs / "summary.csv", summaries)
    v7.old.write_json(data / "gate.lock.json", gate)
    verify_design()
    v7.old.write_json(data / "manifest.json", {"summaries": summaries, "gate": gate,
        "phase": "paired", "physical_steps": physical_count, "replay_filter_steps": replay_count, "optimizer_updates": 0,
        "artifact_sha256": {str(p.relative_to(ROOT)): v7.old.file_hash(p) for base in (data, logs) for p in base.rglob("*") if p.is_file()},
        "config_sha256": v7.old.file_hash(CONFIG), "scientific_training_ready": False})
    print(v7.old.encode(gate), flush=True)


def closed(run_id, replay_run):
    from ..analysis.partial_gnss_gate_v9 import summarize
    from ..analysis.partial_gnss_replay_v9 import physical_rows
    spec = verify_design()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", replay_run):
        raise ValueError("unsafe replay run id")
    gate = v7.old.read_json(BASE / replay_run / "gate.lock.json")
    paired = v7.old.read_json(BASE / replay_run / "manifest.json")
    from ..analysis.partial_gnss_gate_v9 import admission
    if admission(paired["summaries"], spec) != gate:
        raise RuntimeError("admission differs from frozen paired evidence")
    for p, h in paired["artifact_sha256"].items():
        if v7.old.file_hash(ROOT / p) != h:
            raise RuntimeError("paired evidence changed")
    if not gate["passed"]:
        raise RuntimeError("frozen replay admission failed: candidate closed-loop count must remain zero")
    reserve_physical(run_id, spec["none_physical_step_cap"], spec)
    data, logs = outputs(run_id)
    results, used = [], 0
    with v7.old._forbid_training_runtime():
        for case in spec["cases"]:
            if used + 120 + spec["none_physical_step_cap"] > spec["physical_step_cap"]:
                raise RuntimeError("live physical step cap")
            record = episode(case, spec, spec["closed_loop_policy"])
            used += len(record["frames"])
            block = f"{case['scene_id']}_s{case['seed']}"
            v7.old.write_json(data / (block + "_record.json"), record)
            rows, events, audit = physical_rows(record, spec)
            results.append(summarize(record, rows, events, audit, "measurement_ci", spec["closed_loop_policy"], phase="closed_loop"))
    write_csv(logs / "summary.csv", results)
    v7.old.write_json(data / "manifest.json", {"phase": "closed_loop", "summaries": results, "physical_steps": used, "optimizer_updates": 0, "replay_run": replay_run})


def reserve_physical(run_id, planned, spec):
    """Reservations count interrupted runs too; no silent replenishment."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("unsafe run id")
    previous = sum(v7.old.read_json(p)["reserved_physical_steps"] for p in BASE.glob("execution_reservation_*.json"))
    if previous + planned > spec["physical_step_cap"]:
        raise RuntimeError("frozen total physical reservation cap exhausted")
    path = BASE / ("execution_reservation_" + run_id + ".json")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(v7.old.encode({"run_id": run_id, "reserved_physical_steps": planned}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("freeze", "paired", "closed"))
    parser.add_argument("--run-id", default="paired_001")
    parser.add_argument("--replay-run", default="paired_001")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "paired":
        run(args.run_id)
    else:
        closed(args.run_id, args.replay_run)


if __name__ == "__main__":
    main()
