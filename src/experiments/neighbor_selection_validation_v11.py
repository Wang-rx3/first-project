"""Frozen v11 physical policy comparison and bounded offline value diagnosis."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
from time import perf_counter
import numpy as np
from . import partial_gnss_ci_validation_v10 as previous
from . import neighbor_selection_execution_v11 as execution
from ..analysis import neighbor_selection_counterfactual_v11 as cf
from ..analysis.partial_gnss_replay_v9 import physical_rows
from ..analysis.partial_gnss_validation_gate_v10 import summarize, physical_safety
from ..analysis.partial_gnss_report_v7 import write_csv

ROOT = previous.ROOT
STEM = "three_uav_neighbor_value_v11_20261013"
BASE = ROOT / "data/synthetic" / STEM
LOGS = ROOT / "logs/preliminary" / STEM
CONFIG = ROOT / "data/configs" / (STEM+".json")
SOURCES = ("src/analysis/neighbor_selection_policy_v11.py",
           "src/analysis/neighbor_selection_counterfactual_v11.py",
           "src/analysis/neighbor_selection_gate_v11.py",
           "src/analysis/neighbor_selection_report_v11.py",
           "src/experiments/neighbor_selection_execution_v11.py",
           "src/experiments/neighbor_selection_validation_v11.py")
old = previous.v9.v7.old


def freeze():
    if BASE.exists() or CONFIG.exists():
        raise FileExistsError("v11 design already frozen")
    prior = previous.verify()
    v10_manifest = old.read_json(previous.BASE/"final_artifact_manifest.json")
    for p,h in v10_manifest["artifact_sha256"].items():
        if old.file_hash(ROOT/p)!=h:
            raise RuntimeError("v10 evidence changed: "+p)
    development = prior["confirmation_cases"]
    dev_errors = old.read_json(previous.batch_path("research_001")/"confirmation_initialization.lock.json")["initial_errors_by_seed"]
    new_seeds = [2026101311,2026101312,2026101313,2026101314]
    cases = [{**copy.deepcopy(next(c for c in development if c["scene_id"]==scene)),
              "seed":seed,"split":"independent_v11_confirmation"}
             for scene in prior["geometry_ids"] for seed in new_seeds]
    new_errors = {str(s):{u:(previous.v9.v7.indexed_rng(s,0,"v7_initial_error",u).normal(size=15)*np.sqrt([4]*3+[1]*12)).tolist() for u in previous.v9.v7.UIDS} for s in new_seeds}
    spec = {k:prior[k] for k in ("startup_steps","motion_vo_chi_square_limit","dt_s","horizon_steps","decision_steps","measurement_budget","message_ttl_s","pair_period_s","geometry_ids","safety")}
    spec.update(schema="neighbor_value_v11",development_cases=development,confirmation_cases=cases,
        development_seeds=prior["confirmation_seeds"],confirmation_seeds=new_seeds,
        initial_errors_by_seed={**dev_errors,**new_errors},policies=list(execution.POLICIES),
        cf_steps=list(execution.CF_STEPS),physical_step_cap=9600,cf_local_state_cap=1152,
        cf_range_call_cap=192,online_clone_cap=4640,phase_wall_cap_s=900,
        phase_caps={"develop":{"physical":4800},"confirm":{"physical":4800},"counterfactual":{"physical":0}},
        score={"weights":[.25,.25,.25,.25],"nominal_variance_m2":.08**2,"planning_nlos_bias_m":0,
               "quality_scale_m2":.75,"rssi_floor_dbm":-100,"rssi_ceiling_dbm":10,
               "description":"current original-CI zero-innovation position trace fraction + directional variance + GNSS-scaled reliability - RSSI/packet-age cost; no fit"},
        gates={"trajectory_fraction":.05,"positive_seeds_per_geometry":3,"maximum_case_harm":.05,
               "dual_overall":48,"dual_confirmation":24,"meaningful_overall":36,"meaningful_confirmation":18,
               "utility_resolution_m2":.01,"ranking_accuracy":.65,"spearman":.2,"positive_rank_seeds":3,
               "score_p95_ms":200,"posterior_identity_tolerance":1e-8,
               "exact_request_window_matching_required":True},
        legacy_proxy_gate={"minimum_distinguishable_dual":30,"credit_gap":1e-6,"independent_of_v11_scientific_gate":True},
        training_allowed=False,training_ready=False,
        independence="four confirmation seeds; geometry/window repeated blocks",
        counterfactual_scope="frozen Random queue/trajectory; omit/update locally, not full policy intervention")
    protected = old.read_json(previous.BASE/"protected_inputs_before.json")
    protected.update({str(ROOT/p):h for p,h in v10_manifest["artifact_sha256"].items() if p!="README.md"})
    for name in ("final_artifact_manifest.json","final_verification.lock.json"):
        protected[str(previous.BASE/name)] = old.file_hash(previous.BASE/name)
    BASE.mkdir(parents=True,exist_ok=False)
    (BASE/"README_before.md.txt").write_bytes((ROOT/"README.md").read_bytes())
    old.write_json(CONFIG,spec)
    old.write_json(BASE/"design.lock.json",spec)
    old.write_json(BASE/"protected_inputs_before.json",protected)
    old.write_json(BASE/"design.sha256.json",{"config_sha256":old.file_hash(CONFIG),"source_sha256":{p:old.file_hash(ROOT/p) for p in SOURCES}})
    print("v11 frozen:",len(protected),"historical files; 9600 physical steps, 96 CF points",flush=True)


def verify():
    lock=old.read_json(BASE/"design.sha256.json")
    if old.file_hash(CONFIG)!=lock["config_sha256"]:
        raise RuntimeError("v11 configuration changed")
    for p,h in lock["source_sha256"].items():
        if old.file_hash(ROOT/p)!=h:
            raise RuntimeError("v11 frozen source changed: "+p)
    for p,h in old.read_json(BASE/"protected_inputs_before.json").items():
        if old.file_hash(p)!=h:
            raise RuntimeError("historical input changed: "+p)
    release=old.read_json(BASE/"unit_release.lock.json")
    if not release["passed"] or release["source_sha256"]!=lock["source_sha256"]:
        raise RuntimeError("v11 unit release not current")
    return old.read_json(CONFIG)


def batch_path(batch):
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",batch):
        raise ValueError("unsafe batch id")
    return BASE/"batches"/batch


def phase_start(batch,phase,spec):
    base=batch_path(batch);base.mkdir(parents=True,exist_ok=True)
    reservations=base/"reservations";reservations.mkdir(exist_ok=True)
    cap=spec["phase_caps"][phase]
    existing=[old.read_json(p) for p in reservations.glob("*.json")]
    if sum(r["physical"] for r in existing)+cap["physical"]>spec["physical_step_cap"]:
        raise RuntimeError("physical step reservation limit")
    with (reservations/(phase+".json")).open("x",encoding="utf-8") as f:f.write(old.encode({"phase":phase,**cap})+"\n")
    data,logs=base/phase,LOGS/batch/phase
    data.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    return data,logs,perf_counter()+spec["phase_wall_cap_s"]


def write_manifest(data,logs,content):
    content["artifact_sha256"]={str(p.relative_to(ROOT)):old.file_hash(p) for folder in (data,logs) for p in folder.rglob("*") if p.is_file()}
    old.write_json(data/"manifest.json",content)


def checked_manifest(path):
    data=old.read_json(path)
    for p,h in data["artifact_sha256"].items():
        if old.file_hash(ROOT/p)!=h:raise RuntimeError("phase evidence changed: "+p)
    return data


def run_physical(batch,phase):
    spec=verify()
    if phase=="confirm" and not checked_manifest(batch_path(batch)/"develop/manifest.json")["safety"]["passed"]:
        raise RuntimeError("development safety blocks confirmation; accuracy is not an execution gate")
    data,logs,deadline=phase_start(batch,phase,spec)
    summaries=[];failures=[];physical_steps=0;clone_calls=0;timings=[]
    cases=spec["development_cases"] if phase=="develop" else spec["confirmation_cases"]
    with old._forbid_training_runtime():
        for case in cases:
            for policy in spec["policies"]:
                label=f"{case['scene_id']}_s{case['seed']}_{policy}"
                if perf_counter()>=deadline:
                    failures.append({"label":label,"reason":"frozen wall cap; not executed"});continue
                record=execution.physical(case,{**spec,"phase_deadline":deadline},policy)
                old.write_json(data/(label+"_record.json"),record)
                physical_steps+=sum("states" in f.get("audit",{}) for f in record["frames"])
                try:
                    rows,events,audit=physical_rows(record,spec)
                    result=summarize(record,rows,events,audit,"original_ci",policy,"v11_physical")
                    result["study_phase"]=phase
                    result["request_windows"]=[f["step"] for f in record["frames"] if f.get("audit",{}).get("selection","none")!="none"]
                    result["selection_counts"]=dict(__import__("collections").Counter(d["selected"] for d in record["v11_decisions"]))
                    result["uav2_final_localization_error_m"]=float(np.linalg.norm(np.asarray(rows[-1]["target_position_error_m"]))) if rows else None
                    result["decision_points_logged"]=len(record["v11_decisions"])
                    scores=[d["scoring"] for d in record["v11_decisions"]]
                    result["score_clone_calls"]=sum(s["clone_calls"] for s in scores)
                    result["score_timing_samples_ms"]=[s["score_compute_ms"] for s in scores]
                    clone_calls+=result["score_clone_calls"];timings.extend(result["score_timing_samples_ms"])
                    summaries.append(result)
                    old.write_json(data/(label+"_estimator.json"),{"rows":rows,"events":events,"audit":audit})
                    write_csv(logs/(label+"_steps.csv"),rows);write_csv(logs/(label+"_events.csv"),events)
                    write_csv(logs/(label+"_decisions.csv"),record["v11_decisions"])
                except (ValueError,RuntimeError,KeyError,FloatingPointError,AssertionError,np.linalg.LinAlgError) as exc:
                    failures.append({"label":label,"reason":str(exc),"raw_record_retained":True})
                print(phase,case["scene_id"],case["seed"],policy,record["status"],len(record["frames"]),record["failure"],flush=True)
    safety=physical_safety(summaries,expected=40)
    if failures or clone_calls>2320 or any(r["decision_points_logged"]!=29 for r in summaries):
        safety["passed"]=False;safety["failures"].append("v11 completeness/clone cap")
    p95=float(np.percentile(timings,95)) if timings else None
    if p95 is None or p95>=200:
        safety["passed"]=False;safety["failures"].append("score computation period budget")
    write_csv(logs/"summary.csv",summaries)
    write_manifest(data,logs,{"study_phase":phase,"summaries":summaries,"failures":failures,"safety":safety,
        "score_p95_ms":p95,"score_clone_calls":clone_calls,"physical_steps":physical_steps,"optimizer_updates":0})
    verify();print(old.encode({"phase":phase,"safety":safety,"physical_steps":physical_steps}),flush=True)


def run_counterfactual(batch):
    spec=verify();data,logs,deadline=phase_start(batch,"counterfactual",spec)
    results=[];failures=[];states=0;calls=0
    with old._forbid_training_runtime():
        for phase in ("develop","confirm"):
            manifest_path=batch_path(batch)/phase/"manifest.json"
            if not manifest_path.exists():
                failures.append({"phase":phase,"reason":"physical phase missing"});continue
            manifest=checked_manifest(manifest_path)
            cases=spec["development_cases"] if phase=="develop" else spec["confirmation_cases"]
            for case in cases:
                label=f"{case['scene_id']}_s{case['seed']}_random_peer"
                path=batch_path(batch)/phase/(label+"_record.json")
                if not path.exists():
                    failures.append({"phase":phase,"label":label,"reason":"record missing"});continue
                record=old.read_json(path)
                for step in spec["cf_steps"]:
                    point={"study_phase":phase,"scene_id":case["scene_id"],"seed":case["seed"],"step":step}
                    snapshot=next((s for s in record.get("v11_snapshots",[]) if s["step"]==step),None)
                    decision=next((d for d in record.get("v11_decisions",[]) if d["step"]==step),None)
                    if snapshot is None or decision is None:
                        results.append({**point,"dual_eligible":False,"reason":"missing planned checkpoint"});continue
                    try:
                        if perf_counter()>=deadline:raise RuntimeError("counterfactual wall cap")
                        result=cf.evaluate_point(record,snapshot,decision["scoring"],spec)
                        states+=result["local_states"];calls+=result["relative_ci_calls"]
                        if states>1152 or calls>192:raise RuntimeError("counterfactual cap violation")
                        old.write_json(data/(phase+"_"+label+f"_step{step}.json"),{**point,**result})
                        results.append({**point,**{k:v for k,v in result.items() if k!="branches"}})
                    except (ValueError,RuntimeError,KeyError,FloatingPointError,AssertionError,np.linalg.LinAlgError) as exc:
                        failure={**point,"reason":str(exc),"dual_eligible":False}
                        results.append(failure);failures.append(failure)
                print(phase,label,"counterfactual completed",flush=True)
    write_csv(logs/"points.csv",results)
    write_manifest(data,logs,{"results":results,"failures":failures,"local_states":states,"relative_ci_calls":calls,
                            "physical_steps":0,"optimizer_updates":0,"planned_points":96})
    verify()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,choices=("freeze","develop","confirm","counterfactual","report"))
    parser.add_argument("--batch-id",default="research_001")
    args=parser.parse_args()
    if args.phase=="freeze":freeze()
    elif args.phase in ("develop","confirm"):run_physical(args.batch_id,args.phase)
    elif args.phase=="counterfactual":run_counterfactual(args.batch_id)
    else:
        from ..analysis.neighbor_selection_report_v11 import report
        report(args.batch_id)


if __name__=="__main__":main()
