"""Frozen CI-search comparison and bounded live cooperative closed loops."""
from __future__ import annotations
import argparse
import copy
import re
from pathlib import Path
from time import perf_counter
import numpy as np
from . import partial_gnss_revalidation_v9 as v9
from . import partial_gnss_execution_v10 as execution
from ..analysis import partial_gnss_validation_gate_v10 as gates
from ..analysis.partial_gnss_replay_v9 import physical_rows
from ..analysis.partial_gnss_report_v7 import write_csv

ROOT = v9.ROOT
STEM = "three_uav_ci_closed_loop_v10_20261013"
BASE = ROOT / "data/synthetic" / STEM
LOGS = ROOT / "logs/preliminary" / STEM
CONFIG = ROOT / "data/configs" / (STEM+".json")
SOURCES = ("src/analysis/partial_gnss_ci_search_v10.py", "src/experiments/partial_gnss_execution_v10.py",
           "src/analysis/partial_gnss_validation_gate_v10.py", "src/experiments/partial_gnss_ci_validation_v10.py",
           "src/analysis/partial_gnss_closed_report_v10.py")


def freeze():
    if CONFIG.exists() or BASE.exists():
        raise FileExistsError("v10 frozen design already exists")
    old = v9.verify_design()
    seeds = [2026101301, 2026101302, 2026101303, 2026101304]
    geometries = sorted({c["scene_id"] for c in old["cases"]})
    new = []
    for scene in geometries:
        template = next(c for c in old["cases"] if c["scene_id"] == scene)
        for seed in seeds:
            new.append({**copy.deepcopy(template), "seed": seed, "split": "independent_confirmation"})
    spec = {"schema": "ci_closed_loop_v10", "development_cases": old["cases"], "confirmation_cases": new,
        "development_seeds": [20267101, 20267102, 20268111, 20268112], "confirmation_seeds": seeds,
        "geometry_ids": geometries, "initial_errors_by_seed": old["initial_errors_by_seed"],
        "initial_error_law": "indexed v7_initial_error, independent Gaussian diag[4]*3+[1]*12; new arrays materialized only after candidate lock",
        **{k: old[k] for k in ("startup_steps", "motion_vo_chi_square_limit", "dt_s", "horizon_steps", "decision_steps", "measurement_budget", "message_ttl_s", "pair_period_s")},
        "methods": list(gates.METHODS), "development_policies": list(v9.v7.POLICIES),
        "confirmation_replay_policies": ["none", "random_peer"], "physical_policies": ["none", "random_peer"],
        "coarse_grid_step": .05, "refined_grid_step": .005, "weight_interval": [.05, 1.],
        "physical_step_cap": 3840, "replay_group_step_cap": 11520,
        "phase_caps": {"develop": {"physical": 0, "replay": 7680}, "explore": {"physical": 1920, "replay": 0},
                       "confirm": {"physical": 1920, "replay": 0}, "confirmation_replay": {"physical": 0, "replay": 3840}},
        "phase_wall_cap_s": 900,
        "safety": {"kernel_p95_limit_ms": 200., "minimum_separation_m": 2., "maximum_acceleration_mps2": 3.,
                   "maximum_speed_mps": 12., "bounds_low_m": [0,0,2], "bounds_high_m": [200,200,60],
                   "minimum_covariance_eigenvalue": -1e-8, "maximum_full_covariance_trace": 1e4,
                   "maximum_state_magnitude": 1e3, "stop_on_projection": True,
                   "accuracy_or_fusion_rate_execution_prerequisite": False},
        "selection": {"minimum_fractional_improvement": .01, "absolute_rmse_min_gain_m": .01,
                      "positive_independent_seeds": 3, "maximum_case_degradation_fraction": .05},
        "performance": {"pooled_rmse_improvement_fraction": .05, "positive_seeds_per_geometry": 3,
                        "maximum_case_degradation_fraction": .05, "reference_rmse_limit_m": 1.,
                        "reference_gnss_acceptance_min": .9, "position_nees_descriptive_limit": 7.815},
        "metric_definition": "steps1..120 3D error norm RMSE; relative vector SSE average across3pairs then time; equal-block pooled RMSE=sqrt(mean(blockRMSE**2)); per-seed pool2geometries likewise; NEES3D position marginal pinvP eachstep then time mean",
        "strict_parking": "actual distance<=1m and actual speed<=.5m/s for5steps, separate navigation outcome",
        "training_allowed": False, "v9_twenty_percent_gate_preserved": True,
        "confirmation_data_may_select_candidate": False,
        "development_record_paths": [str((v9.BASE/"paired_001"/f"{c['scene_id']}_s{c['seed']}"/"none_record.json").relative_to(ROOT)) for c in old["cases"]]}
    protected = v9.v7.old.read_json(v9.BASE / "protected_inputs_before.json")
    protected.update({str(ROOT/p): h for p,h in v9.v7.old.read_json(v9.BASE/"final_artifact_manifest.json")["artifact_sha256"].items() if p != "README.md"})
    for name in ("final_artifact_manifest.json", "final_verification.lock.json"):
        protected[str(v9.BASE/name)] = v9.v7.old.file_hash(v9.BASE/name)
    for p,h in protected.items():
        if v9.v7.old.file_hash(p) != h:
            raise RuntimeError("historical input changed before v10: "+p)
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE/"README_before.md.txt").write_bytes((ROOT/"README.md").read_bytes())
    v9.v7.old.write_json(CONFIG, spec)
    v9.v7.old.write_json(BASE/"design.lock.json", spec)
    v9.v7.old.write_json(BASE/"protected_inputs_before.json", protected)
    v9.v7.old.write_json(BASE/"design.sha256.json", {"config_sha256": v9.v7.old.file_hash(CONFIG),
        "source_sha256": {p: v9.v7.old.file_hash(ROOT/p) for p in SOURCES}})
    print(f"v10 frozen: {len(protected)} protected files; 3840 physical, 11520 replay steps per batch; four untouched confirmation seeds", flush=True)


def verify():
    lock = v9.v7.old.read_json(BASE/"design.sha256.json")
    if v9.v7.old.file_hash(CONFIG) != lock["config_sha256"]:
        raise RuntimeError("v10 configuration changed")
    for p,h in lock["source_sha256"].items():
        if v9.v7.old.file_hash(ROOT/p) != h:
            raise RuntimeError("v10 execution source changed: "+p)
    for p,h in v9.v7.old.read_json(BASE/"protected_inputs_before.json").items():
        if v9.v7.old.file_hash(p) != h:
            raise RuntimeError("protected historical file changed: "+p)
    release = v9.v7.old.read_json(BASE/"unit_release.lock.json")
    if not release["passed"] or release["source_sha256"] != lock["source_sha256"]:
        raise RuntimeError("pure-unit/interface release not current")
    return v9.v7.old.read_json(CONFIG)


def batch_path(batch):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", batch):
        raise ValueError("unsafe batch id")
    return BASE / "batches" / batch


def phase_start(batch, phase, spec):
    base = batch_path(batch)
    base.mkdir(parents=True, exist_ok=True)
    reservations = base/"reservations"
    reservations.mkdir(exist_ok=True)
    cap = spec["phase_caps"][phase]
    existing = [v9.v7.old.read_json(p) for p in reservations.glob("*.json")]
    if sum(r["physical"] for r in existing)+cap["physical"] > spec["physical_step_cap"] or sum(r["replay"] for r in existing)+cap["replay"] > spec["replay_group_step_cap"]:
        raise RuntimeError("batch reservation limit")
    path = reservations/(phase+".json")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(v9.v7.old.encode({"phase": phase, **cap})+"\n")
    data, logs = base/phase, LOGS/batch/phase
    for p in (data,logs):
        p.mkdir(parents=True, exist_ok=False)
    return data, logs, perf_counter()+spec["phase_wall_cap_s"]


def write_manifest(data, logs, content):
    content["artifact_sha256"] = {str(p.relative_to(ROOT)): v9.v7.old.file_hash(p) for folder in (data,logs) for p in folder.rglob("*") if p.is_file()}
    v9.v7.old.write_json(data/"manifest.json", content)


def checked_manifest(path):
    result = v9.v7.old.read_json(path)
    for p,h in result["artifact_sha256"].items():
        if v9.v7.old.file_hash(ROOT/p) != h:
            raise RuntimeError("phase evidence changed: "+p)
    return result


def write_result(data, logs, label, record, rows, events, audit, method, policy, phase):
    if record is not None and "replay" not in phase:
        path=data/(label+"_record.json")
        if not path.exists():
            v9.v7.old.write_json(path,record)
    v9.v7.old.write_json(data/(label+"_estimator.json"), {"rows":rows,"events":events,"audit":audit})
    write_csv(logs/(label+"_steps.csv"),rows)
    write_csv(logs/(label+"_events.csv"),events)
    return gates.summarize(record,rows,events,audit,method,policy,phase)


def develop(batch):
    spec=verify()
    data,logs,deadline=phase_start(batch,"develop",spec)
    summary=[]; failure=[]; count=0
    with v9.v7.old._forbid_training_runtime():
        for path in spec["development_record_paths"]:
            record=v9.v7.old.read_json(ROOT/path)
            for method in spec["methods"]:
                for policy in spec["development_policies"]:
                    label=f"{record['case']['scene_id']}_s{record['case']['seed']}_{method}_{policy}"
                    try:
                        if perf_counter()>=deadline:
                            raise RuntimeError("frozen phase wall limit")
                        rows,events,audit=execution.replay(record,method,policy,{**spec,"phase_deadline":deadline})
                        count+=len(rows)
                        summary.append(write_result(data,logs,label,record,rows,events,audit,method,policy,"development_replay"))
                    except (ValueError,RuntimeError,FloatingPointError,AssertionError,np.linalg.LinAlgError) as exc:
                        error={"label":label,"method":method,"policy":policy,"failure":str(exc),"status":"failed"}
                        failure.append(error)
                        v9.v7.old.write_json(data/(label+"_failure.json"),error)
            print(record["case"]["scene_id"],record["case"]["seed"],"development replay done",flush=True)
    safety=gates.safety_admission(summary,spec,tests_passed=True)
    selection=gates.select_candidate(summary,spec,safety)
    write_csv(logs/"summary.csv",summary)
    v9.v7.old.write_json(data/"safety.lock.json",safety)
    v9.v7.old.write_json(data/"selection_decision.json",selection)
    write_manifest(data,logs,{"phase":"development_replay","summaries":summary,"failures":failure,
        "replay_group_steps":count,"physical_steps":0,"optimizer_updates":0,"selection":selection})
    # The decision is sealed before drawing NEW priors or seeing NEW sensors.
    v9.v7.old.write_json(batch_path(batch)/"candidate_selection.lock.json", {"decision":selection,
        "development_manifest_sha256":v9.v7.old.file_hash(data/"manifest.json"),
        "config_sha256":v9.v7.old.file_hash(CONFIG),"source_sha256":v9.v7.old.read_json(BASE/"design.sha256.json")["source_sha256"]})
    if selection["closed_loop_execution_allowed"]:
        errors={str(s):{u:(v9.v7.indexed_rng(s,0,"v7_initial_error",u).normal(size=15)*np.sqrt([4]*3+[1]*12)).tolist() for u in v9.v7.UIDS} for s in spec["confirmation_seeds"]}
        v9.v7.old.write_json(batch_path(batch)/"confirmation_initialization.lock.json", {"initial_errors_by_seed":errors,
            "candidate_selection_sha256":v9.v7.old.file_hash(batch_path(batch)/"candidate_selection.lock.json"),
            "scope":"frozen-law draws after method lock, before any NEW observations"})
    verify()
    print(v9.v7.old.encode(selection),flush=True)


def candidate(batch,spec):
    lock=v9.v7.old.read_json(batch_path(batch)/"candidate_selection.lock.json")
    manifest=checked_manifest(batch_path(batch)/"develop/manifest.json")
    if lock["development_manifest_sha256"]!=v9.v7.old.file_hash(batch_path(batch)/"develop/manifest.json") or lock["config_sha256"]!=v9.v7.old.file_hash(CONFIG):
        raise RuntimeError("candidate evidence/configuration changed")
    again=gates.select_candidate(manifest["summaries"],spec,gates.safety_admission(manifest["summaries"],spec,True))
    if again!=lock["decision"]:
        raise RuntimeError("candidate selection differs from frozen development decision")
    if not again["closed_loop_execution_allowed"]:
        raise RuntimeError("neither candidate passed numerical/interface execution admission")
    return again["selected_method"]


def run_physical(batch,phase):
    spec=verify(); method=candidate(batch,spec)
    if phase=="confirm":
        previous=checked_manifest(batch_path(batch)/"explore/manifest.json")
        if not previous["safety"]["passed"]:
            raise RuntimeError("exploratory physical safety fault blocks new confirmation; accuracy/fusion rate do not gate")
        initialization=v9.v7.old.read_json(batch_path(batch)/"confirmation_initialization.lock.json")
        if initialization["candidate_selection_sha256"]!=v9.v7.old.file_hash(batch_path(batch)/"candidate_selection.lock.json"):
            raise RuntimeError("confirmation priors and candidate lock disagree")
        spec={**spec,"initial_errors_by_seed":{**spec["initial_errors_by_seed"],**initialization["initial_errors_by_seed"]}}
    cases=spec["development_cases"] if phase=="explore" else spec["confirmation_cases"]
    data,logs,deadline=phase_start(batch,phase,spec)
    summary=[];count=0;failures=[]
    with v9.v7.old._forbid_training_runtime():
        for case in cases:
            for policy in spec["physical_policies"]:
                label=f"{case['scene_id']}_s{case['seed']}_{policy}"
                if perf_counter()>=deadline:
                    failure={"label":label,"reason":"frozen phase wall cap, not executed"}
                    failures.append(failure);v9.v7.old.write_json(data/(label+"_not_executed.json"),failure)
                    continue
                started=perf_counter()
                record=execution.physical(case,{**spec,"phase_deadline":deadline},method,policy)
                count+=len(record["frames"])
                # Flush the raw stopped record before any diagnostic parser.
                v9.v7.old.write_json(data/(label+"_record.json"),record)
                try:
                    rows,events,audit=physical_rows(record,spec)
                    result=write_result(data,logs,label,record,rows,events,audit,method,policy,"exploration" if phase=="explore" else "confirmation")
                except (ValueError,RuntimeError,FloatingPointError,AssertionError,np.linalg.LinAlgError) as exc:
                    result={"scene_id":case["scene_id"],"seed":case["seed"],"method":method,"policy":policy,
                        "phase":phase,"status":"censored","full_horizon_eligible":False,"steps":0,
                        "uav2_absolute_rmse_m":None,"cluster_relative_rmse_m":None,"covariance_finite":False,
                        "failure":{"record_failure":record["failure"],"diagnostic_parser_failure":str(exc)}}
                    failures.append({"label":label,**result["failure"]})
                    v9.v7.old.write_json(data/(label+"_diagnostic_failure.json"),result)
                result["episode_wall_ms"]=1000*(perf_counter()-started)
                summary.append(result)
                print(phase,case["scene_id"],case["seed"],policy,record["status"],len(record["frames"]),record["failure"],flush=True)
    safety=gates.physical_safety(summary)
    performance=gates.confirmation_performance(summary,spec)
    write_csv(logs/"summary.csv",summary)
    v9.v7.old.write_json(data/"safety.lock.json",safety)
    v9.v7.old.write_json(data/"performance.lock.json",performance)
    write_manifest(data,logs,{"phase":phase,"method":method,"summaries":summary,"safety":safety,"performance":performance,
        "physical_steps":count,"replay_group_steps":0,"optimizer_updates":0,"failures":failures})
    verify()
    print(v9.v7.old.encode({"phase":phase,"safety":safety,"performance":performance}),flush=True)


def confirmation_replay(batch):
    spec=verify();candidate(batch,spec)
    checked_manifest(batch_path(batch)/"confirm/manifest.json")
    data,logs,deadline=phase_start(batch,"confirmation_replay",spec)
    summary=[];count=0;failures=[]
    with v9.v7.old._forbid_training_runtime():
        for case in spec["confirmation_cases"]:
            label=f"{case['scene_id']}_s{case['seed']}"
            record=v9.v7.old.read_json(batch_path(batch)/"confirm"/(label+"_none_record.json"))
            if record["status"]!="complete":
                failures.append({"label":label,"reason":"censored None producer; no full-horizon replay"})
                continue
            for method in spec["methods"]:
                for policy in spec["confirmation_replay_policies"]:
                    if perf_counter()>=deadline:
                        failures.append({"label":label,"reason":"phase wall cap"});continue
                    rows,events,audit=execution.replay(record,method,policy,{**spec,"phase_deadline":deadline})
                    count+=len(rows)
                    summary.append(write_result(data,logs,label+"_"+method+"_"+policy,record,rows,events,audit,method,policy,"confirmation_replay"))
            print(label,"locked-method-independent diagnostic replay complete",flush=True)
    write_csv(logs/"summary.csv",summary)
    write_manifest(data,logs,{"phase":"confirmation_replay","summaries":summary,"physical_steps":0,
        "replay_group_steps":count,"failures":failures,"optimizer_updates":0,"used_for_selection":False})
    verify()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,choices=("freeze","develop","explore","confirm","confirmation_replay","report"))
    parser.add_argument("--batch-id",default="research_001")
    args=parser.parse_args()
    if args.phase=="freeze":freeze()
    elif args.phase=="develop":develop(args.batch_id)
    elif args.phase in ("explore","confirm"):run_physical(args.batch_id,args.phase)
    elif args.phase=="confirmation_replay":confirmation_replay(args.batch_id)
    else:
        from ..analysis.partial_gnss_closed_report_v10 import report
        report(args.batch_id)


if __name__=="__main__":main()
