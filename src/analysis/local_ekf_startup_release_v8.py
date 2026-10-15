"""Evaluate frozen gates; freeze confirmation seeds without changing old design."""
from __future__ import annotations

import argparse
import copy

import numpy as np

from . import local_ekf_startup_diagnostic_v8 as audit
from ..experiments import partial_gnss_baseline_v7 as v7


def evaluate(run_id,extra=False):
    data=audit.BASE/run_id
    manifest=v7.old.read_json(data/"manifest.json")
    spec=v7.old.read_json(audit.CONFIG)
    original=v7.old.read_json(audit.BASE/"native_original_001/manifest.json")["summary"]
    checks=[]
    for path in sorted(data.glob("record_*.json")):
        record=v7.old.read_json(path)
        case=record["case"];key=(case["scene_id"],case["seed"])
        rows=[r for r in manifest["summary"] if (r["scene_id"],r["seed"])==key]
        item={"scene_id":key[0],"seed":key[1],"complete":record["status"]=="complete" and len(record["frames"])==120,
              "safe":all(r["collision_count"]==0 and r["boundary_count"]==0 and r["safety_count"]==0 and r["minimum_separation_m"]>=2-1e-8 and r["maximum_acceleration_mps2"]<=3+1e-7 for r in rows)}
        reference_checks=[]
        for row in rows:
            uid=row["uid"]
            if uid not in v7.PEERS:continue
            reference={"uid":uid,"position_rmse_m":row["position_rmse_m"],
                "rmse_pass":row["position_rmse_m"]<=spec["candidate_reference_rmse_limit_m"],
                "gnss_acceptance":row["gnss_accepted"]/max(1,row["gnss_attempted"]),
                "vo_acceptance":row["vo_accepted"]/max(1,row["vo_attempted"])}
            reference["acceptance_pass"]=reference["gnss_acceptance"]>=.9 and reference["vo_acceptance"]>=.9
            if not extra and case["seed"]==20267101:
                old=next(r for r in original if (r["scene_id"],r["seed"],r["uid"])==(*key,uid))
                reference["rmse_degradation_m"]=row["position_rmse_m"]-old["position_rmse_m"]
                reference["nondegradation_pass"]=reference["rmse_degradation_m"]<=spec["reference_original_success_rmse_degradation_limit_m"]
            reference_checks.append(reference)
        item["reference_checks"]=reference_checks
        item["bias_step5_pass"]=False;item["covariance_pass"]=False
        if len(record["frames"])>=5 and "accel_bias" in record["frames"][4]["evaluation_truth"]:
            frame=record["frames"][4]
            errors={u:float(np.linalg.norm(np.array(frame["audit"]["states"][u])[9:12]-frame["evaluation_truth"]["accel_bias"][u])) for u in v7.UIDS}
            item["bias_error_norms_step5"]=errors;item["bias_step5_pass"]=all(e<=.1 for e in errors.values())
            matrices=[np.array(P) for f in record["frames"] for P in f["audit"].get("covariances",{}).values()]
            item["maximum_full_covariance_trace"]=max(float(np.trace(P)) for P in matrices)
            item["minimum_eigenvalue"]=min(float(np.linalg.eigvalsh(P)[0]) for P in matrices)
            item["covariance_pass"]=all(np.isfinite(P).all() for P in matrices) and item["minimum_eigenvalue"]>=-1e-8 and item["maximum_full_covariance_trace"]<spec["covariance_full_trace_limit"]
        item["passed"]=all(item[k] for k in ("complete","safe","bias_step5_pass","covariance_pass")) and len(reference_checks)==2 and all(r["rmse_pass"] and r["acceptance_pass"] and r.get("nondegradation_pass",True) for r in reference_checks)
        checks.append(item)
    result={"passed":len(checks)==4 and all(r["passed"] for r in checks),"checks":checks,"run_id":run_id,
            "scope":"conditional stationary fixture local acquisition; not full cooperative/IMU consistency proof"}
    v7.old.write_json(audit.BASE/("extra_candidate_gate.lock.json" if extra else "original_candidate_gate.lock.json"),result)
    print(v7.old.encode(result))
    return result


def freeze_extra():
    if not v7.old.read_json(audit.BASE/"original_candidate_gate.lock.json")["passed"]:raise RuntimeError("original gate failed")
    spec=v7.old.read_json(audit.CONFIG)
    seeds=(20268111,20268112)
    geometries=[c for c in spec["original_cases"] if c["seed"]==20267101]
    cases=[]
    for geometry in geometries:
        for seed in seeds:
            case=copy.deepcopy(geometry);case["seed"]=seed;case["split"]="prospective_confirmation";cases.append(case)
    errors={str(s):{u:(v7.indexed_rng(s,0,"v7_initial_error",u).normal(size=15)*np.sqrt([4]*3+[1]*12)).tolist() for u in v7.UIDS} for s in seeds}
    lock={"cases":cases,"initial_errors_by_seed":errors,"candidate_kernel_sha256":v7.old.file_hash(v7.ROOT/"src/analysis/local_ekf_startup_filter_v8.py"),
          "before_any_confirmation_outcomes":True,"maximum_physical_steps":480}
    v7.old.write_json(audit.BASE/"extra_design.lock.json",lock)
    print("Frozen confirmation seeds",seeds)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-id");parser.add_argument("--extra",action="store_true");parser.add_argument("--freeze-extra",action="store_true")
    args=parser.parse_args();freeze_extra() if args.freeze_extra else evaluate(args.run_id,args.extra)


if __name__=="__main__":main()
