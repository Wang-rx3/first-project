"""Bounded isolated local-EKF validation; cooperative measurements disabled."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from . import partial_gnss_baseline_v7 as v7
from ..analysis import local_ekf_startup_diagnostic_v8 as audit


def episode(case,spec,arm):
    env,obs=v7.prepare(case,spec)
    if arm=="stationary_bias_acquisition":
        from ..analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
        if any(np.any(np.asarray(v)!=0) for v in case["scenario"]["initial_velocities"].values()):
            raise ValueError("stationary startup fixture is not declared")
        for uid,original in list(env.ekfs.items()):
            candidate=StartupLocalEKF(original.state.copy(),original.config,sensor_config=env.config.sensor,
                                     startup_steps=spec["startup_steps"],vo_motion_limit=spec["motion_vo_chi_square_limit"])
            candidate.covariance=original.covariance.copy();env.ekfs[uid]=candidate
        env.graph_batch=env._build_graph_batch();obs=env._observations()
    initial={u:{"state":env.ekfs[u].state.copy(),"covariance":env.ekfs[u].covariance.copy()} for u in v7.UIDS}
    frames=[];rows=[];failure=None;parking=v7.old.StrictParkingTracker()
    for step in range(1,121):
        masked=v7.old.v4.mask_observations(obs,step)
        communication=v7.old.v3.contract().communication_actions(masked,{u:"none" for u in v7.UIDS})
        actions,navigation=v7.compose_fixed_navigation_actions(masked,communication)
        requested=copy.deepcopy(actions)
        if arm=="stationary_bias_acquisition" and step<=spec["startup_steps"]:
            for command in actions.values():command["delta_v"]=np.zeros(3)
        pre_p={u:env.states[u].position_m.copy() for u in v7.UIDS};pre_v={u:env.states[u].velocity_mps.copy() for u in v7.UIDS}
        try:
            obs,_,terminated,truncated,info=env.step(actions)
            if arm=="stationary_bias_acquisition" and step==spec["startup_steps"]:
                for ekf in env.ekfs.values():ekf.check_startup_evidence()
            try:
                positions,velocities,separation,acceleration=v7.check_safety(env,info,pre_p,pre_v)
            except RuntimeError as exc:
                failure={"step":step,"reason":str(exc)}
                positions={u:env.states[u].position_m.copy() for u in v7.UIDS};velocities={u:env.states[u].velocity_mps.copy() for u in v7.UIDS}
                separation=min(r["distance_m"] for r in v7.old.joint_closest_approach(pre_p,positions))
                acceleration=max(float(np.linalg.norm((velocities[u]-pre_v[u])/.2)) for u in v7.UIDS)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            failure={"step":step,"reason":f"{type(exc).__name__}: {exc}","incomplete_native_transition":True}
            frames.append(v7.old.normalize({"step":step,"failure":failure,"evaluation_truth":{"position":{u:env.states[u].position_m for u in v7.UIDS}},
                                          "audit":{"actions":actions,"requested_actions":requested}}))
            break
        truth={"position":positions,"velocity":velocities,"attitude":{u:env.states[u].attitude_rad.copy() for u in v7.UIDS},
               "accel_bias":{u:env.accel_bias[u].copy() for u in v7.UIDS},"gyro_bias":{u:env.gyro_bias[u].copy() for u in v7.UIDS}}
        bias={u:{"accel_bias":truth["accel_bias"][u],"gyro_bias":truth["gyro_bias"][u]} for u in v7.UIDS}
        distances={u:float(np.linalg.norm(positions[u]-env.goals[u])) for u in v7.UIDS};speeds={u:float(np.linalg.norm(velocities[u])) for u in v7.UIDS}
        parked=parking.update(distances,speeds,bool(info["collision"]))
        for uid,ekf in env.ekfs.items():
            row={"scene_id":case["scene_id"],"seed":case["seed"],"arm":arm,"uid":uid,"step":step,
                 **audit.state_metric(uid,ekf.state,ekf.covariance,truth,bias),**ekf.diagnostics_snapshot(),
                 "calibration":copy.deepcopy(getattr(ekf,"last_startup_update",{}))}
            rows.append(row)
        frames.append(v7.old.normalize({"step":step,"observable":{"sensor_packets":v7.prior.sanitize_sensor_packets(info["sensor_packets"],env.config.sensor.vo_position_std_m)},
            "evaluation_truth":truth,"audit":{"states":{u:env.ekfs[u].state.copy() for u in v7.UIDS},"covariances":{u:env.ekfs[u].covariance.copy() for u in v7.UIDS},
                "diagnostics":{u:env.ekfs[u].diagnostics_snapshot() for u in v7.UIDS},"actions":actions,"requested_actions":requested,"navigation":navigation,
                "minimum_separation_m":separation,"max_acceleration_mps2":acceleration,"collision":info["collision"],
                "boundary_count":info["boundary_projection_count"],"safety_count":info["safety_projection_count"],
                "parking":parked,"distances":distances,"speeds":speeds,
                "calibration":{u:copy.deepcopy(getattr(env.ekfs[u],"last_startup_update",{})) for u in v7.UIDS}}}))
        if failure:break
    return {"case":case,"initial":v7.old.normalize(initial),"arm":arm,"frames":frames,"status":"censored" if failure else "complete","failure":failure},rows


def verify_baseline(record):
    key=(record["case"]["scene_id"],record["case"]["seed"])
    path=v7.BASE/"bounded_resume_s20267101_102"/f"record_{key[0]}_s{key[1]}.json"
    residual=0.
    if path.exists():
        previous=v7.old.read_json(path)
        if len(previous["frames"])!=len(record["frames"]):raise RuntimeError("baseline censor length differs")
        for old,new in zip(previous["frames"],record["frames"]):
            for uid in v7.UIDS:
                for field in ("position","velocity"):
                    residual=max(residual,float(np.max(np.abs(np.array(old["evaluation_truth"][field][uid])-new["evaluation_truth"][field][uid]))))
            if v7.old.encode(old["observable"]["sensor_packets"])!=v7.old.encode(new["observable"]["sensor_packets"]):
                raise RuntimeError("native raw sensor stream differs")
    else:
        previous=v7.old.read_json(v7.BASE/"safety_prefix_s20267102/failure.json")
        if previous["steps"]!=len(record["frames"]):raise RuntimeError("baseline missing-prefix censor differs")
        for old,new in zip(previous["rows"],record["frames"]):
            for uid in v7.UIDS:
                residual=max(residual,float(np.max(np.abs(np.array(old["states"][uid])-new["evaluation_truth"]["position"][uid]))))
    if residual>1e-8:raise RuntimeError("baseline reproduced trajectory differs")
    return residual


def run(run_id,arm,extra=False,batch_id="v8_research_001"):
    from ..analysis.partial_gnss_report_v7 import write_csv
    spec=v7.old.read_json(audit.CONFIG)
    if v7.old.file_hash(audit.CONFIG)!=v7.old.read_json(audit.BASE/"design.sha256.json")["sha256"]:raise RuntimeError("v8 frozen config changed")
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",run_id):raise ValueError("unsafe run-id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",batch_id):raise ValueError("unsafe batch-id")
    if arm=="stationary_bias_acquisition":
        evidence=v7.old.read_json(audit.BASE/"evidence_release.lock.json")
        units=v7.old.read_json(audit.BASE/"unit_release.lock.json")
        augmented=v7.old.read_json(audit.BASE/"augmented_unit_release.lock.json")
        if not evidence["offline_and_native_gate"] or not units["passed"] or not augmented["passed"] or units["kernel_sha256"]!=v7.old.file_hash(v7.ROOT/"src/analysis/local_ekf_startup_filter_v8.py"):
            raise RuntimeError("candidate release checks not satisfied")
    data=audit.BASE/run_id;logs=audit.LOGS/run_id
    cases=spec["original_cases"]
    if extra:
        if not v7.old.read_json(audit.BASE/"original_candidate_gate.lock.json")["passed"]:raise RuntimeError("original cases not passed")
        lock=v7.old.read_json(audit.BASE/"extra_design.lock.json")
        if lock["candidate_kernel_sha256"]!=v7.old.file_hash(v7.ROOT/"src/analysis/local_ekf_startup_filter_v8.py"):
            raise RuntimeError("confirmation candidate differs from frozen kernel")
        cases=lock["cases"];spec={**spec,"initial_errors_by_seed":lock["initial_errors_by_seed"]}
    histories=[];rows=[];audits=[]
    previous=[v7.old.read_json(p) for p in audit.BASE.glob("*/manifest.json")]
    previously_executed=sum(m.get("physical_steps",0) for m in previous if m.get("batch_id","v8_research_001")==batch_id)
    if previously_executed+len(cases)*120>spec["physical_step_cap"]:
        raise RuntimeError("whole invocation would exceed frozen batch cap; select a fresh authorized reproduction batch")
    data.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    v7.old.write_json(data/"execution_sources.json",{name:v7.old.file_hash(v7.ROOT/name) for name in (
        "src/experiments/local_ekf_startup_validation_v8.py","src/analysis/local_ekf_startup_filter_v8.py",
        "src/simulation/environment.py","src/simulation/ekf.py","src/control/fixed_navigation.py","src/analysis/partial_gnss_fusion_v7.py")}) if arm!="native" else None
    with v7.old._forbid_training_runtime():
        for case in cases:
            if previously_executed+sum(len(r["frames"]) for r in histories)+120>spec["physical_step_cap"]:
                raise RuntimeError("pre-execution physical step cap")
            record,metrics=episode(case,spec,arm)
            if arm=="native":audits.append({"scene_id":case["scene_id"],"seed":case["seed"],"reproduction_residual":verify_baseline(record)})
            v7.old.write_json(data/f"record_{case['scene_id']}_s{case['seed']}.json",record)
            histories.append(record);rows.extend(metrics)
            print(case["scene_id"],case["seed"],record["status"],len(record["frames"]),record["failure"],flush=True)
    summary=audit.summarize(rows,("scene_id","seed","arm","uid"))
    post_warmup=audit.summarize([r for r in rows if r["step"]>spec["startup_steps"]],("scene_id","seed","arm","uid"))
    for item in summary:
        record=next(r for r in histories if (r["case"]["scene_id"],r["case"]["seed"])==(item["scene_id"],item["seed"]))
        item.update({"status":record["status"],"failure":record["failure"],
                     "strict_parking":record["frames"][-1].get("audit",{}).get("parking",{}).get("per_uav",{}).get(item["uid"],{}).get("ever_success"),
                     "collision_count":sum(f["audit"].get("collision",0) for f in record["frames"]),
                     "boundary_count":sum(f["audit"].get("boundary_count",0) for f in record["frames"]),
                     "safety_count":sum(f["audit"].get("safety_count",0) for f in record["frames"]),
                     "minimum_separation_m":min(f["audit"].get("minimum_separation_m",float("inf")) for f in record["frames"]),
                     "maximum_acceleration_mps2":max(f["audit"].get("max_acceleration_mps2",0) for f in record["frames"])})
    write_csv(logs/"per_uav_steps.csv",rows);write_csv(logs/"summary.csv",summary);write_csv(logs/"post_warmup_summary.csv",post_warmup)
    v7.old.write_json(data/"manifest.json",{"physical_steps":sum(len(r["frames"]) for r in histories),"optimizer_updates":0,"summary":summary,"audits":audits,
        "config_sha256":v7.old.file_hash(audit.CONFIG),"phase":"closed_loop","arm":arm,"extra":extra,"batch_id":batch_id})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-id",required=True);parser.add_argument("--arm",choices=("native","stationary_bias_acquisition"),required=True)
    parser.add_argument("--extra",action="store_true");parser.add_argument("--batch-id",default="v8_research_001")
    args=parser.parse_args();run(args.run_id,args.arm,args.extra,args.batch_id)


if __name__=="__main__":main()
