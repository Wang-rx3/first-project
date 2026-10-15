"""Read-only observation replay and math audit; no optimizers or environments."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace

import numpy as np

from ..experiments import partial_gnss_baseline_v7 as v7
from ..experiments import three_uav_fusion_validation_v6 as v6

ROOT=v7.ROOT
STEM="three_uav_local_ekf_startup_v8_20261013"
BASE=ROOT/"data/synthetic"/STEM
LOGS=ROOT/"logs/preliminary"/STEM
CONFIG=ROOT/"data/configs"/(STEM+".json")
ARMS=("native","diagnostic_vo_unrestricted_first5","diagnostic_gnss_unrestricted_first5","diagnostic_vo_unrestricted_prefix")


def bias_tape(seed,steps,config):
    """Private simulation truth reconstruction ONLY for offline evaluation."""
    accel={u:np.zeros(3) for u in v7.UIDS}; gyro={u:np.zeros(3) for u in v7.UIDS}
    result=[]
    for step in range(steps):
        frame={}
        for uid in v7.UIDS:
            rng=v7.indexed_rng(seed,step,"imu",uid)
            accel[uid]+=rng.normal(0,config.accel_bias_rw_std,3)
            gyro[uid]+=rng.normal(0,config.gyro_bias_rw_std,3)
            frame[uid]={"accel_bias":accel[uid].copy(),"gyro_bias":gyro[uid].copy(),
                "accel_noise":rng.normal(0,config.imu_accel_std_mps2,3),"gyro_noise":rng.normal(0,config.imu_gyro_std_rps,3)}
        result.append(frame)
    return result


def state_metric(uid,state,covariance,truth,bias):
    position=np.array(state[:3])-truth["position"][uid]
    velocity=np.array(state[3:6])-truth["velocity"][uid]
    accel=np.array(state[9:12])-bias[uid]["accel_bias"]
    gyro=np.array(state[12:15])-bias[uid]["gyro_bias"]
    return {"position_sse_m2":float(position@position),"velocity_sse":float(velocity@velocity),
        "position_nees":v7.old.nees(position,covariance[:3,:3]),
        "position_velocity_nees":v7.old.nees(np.r_[position,velocity],covariance[:6,:6]),
        "accel_bias_sse":float(accel@accel),"gyro_bias_sse":float(gyro@gyro),
        "bias_z_error":float(accel[2]),"position_trace":float(np.trace(covariance[:3,:3])),
        "accel_bias_trace":float(np.trace(covariance[9:12,9:12])),"full_trace":float(np.trace(covariance)),
        "state":np.asarray(state).tolist(),"position_error":position.tolist()}


def clone_check(ekf,vo,previous,F):
    """30-state temporal clone, including process noise in the current block."""
    previous_state, previous_cov=previous
    current=ekf.state.copy(); P=ekf.covariance.copy()
    C=previous_cov@F.T
    augmented=np.block([[P,C.T],[C,previous_cov]])
    H=np.zeros((3,30));H[:,:3]=np.eye(3);H[:,15:18]=-np.eye(3)
    R=np.array(vo["covariance"])
    innovation=np.array(vo["delta_position"])-(current[:3]-previous_state[:3])
    S=H@augmented@H.T+R
    K=augmented@H.T@np.linalg.inv(S)
    expected_state=current+K[:15]@innovation
    expected_cov=P-K[:15]@S@K[:15].T
    return S,K[:15],expected_state,expected_cov


def replay(record,arm,sim):
    filters={u:v7.LocalEKF(record["initial"][u]["state"],sim.ekf) for u in v7.UIDS}
    tape=bias_tape(record["case"]["seed"],len(record["frames"]),sim.sensor)
    rows=[];clone_residual=0.;noise_residual=0.;native_match_pre_projection=0.
    previous_v={u:np.array(record["case"]["scenario"]["initial_velocities"][u]) for u in v7.UIDS}
    for frame,bias in zip(record["frames"],tape):
        step=frame["step"]
        packets=frame["observable"]["sensor_packets"]
        for uid,ekf in filters.items():
            ekf.begin_step();pre=(ekf.state.copy(),ekf.covariance.copy())
            F=ekf.predict(packets[uid]["imu"],.2)
            vo=packets[uid]["vo"]
            if vo["valid"]:
                S,K,x_expected,P_expected=clone_check(ekf,vo,pre,F)
                gate_free=(arm=="diagnostic_vo_unrestricted_prefix" or (arm=="diagnostic_vo_unrestricted_first5" and step<=5))
                original=ekf.config
                if gate_free:ekf.config=replace(original,innovation_gate_threshold=float("inf"))
                accepted=ekf.update_visual_delta(np.array(vo["delta_position"]),np.array(vo["covariance"]),
                    previous_state=pre[0],previous_covariance=pre[1],transition_matrix=F,dt_s=.2)
                ekf.config=original
                if accepted:
                    clone_residual=max(clone_residual,float(np.max(np.abs(ekf.state-x_expected))),float(np.max(np.abs(ekf.covariance-P_expected))))
            gnss=packets[uid]["gnss"]
            if gnss["valid"]:
                original=ekf.config
                if arm=="diagnostic_gnss_unrestricted_first5" and step<=5:
                    ekf.config=replace(original,innovation_gate_threshold=float("inf"))
                ekf.update_position(np.array(gnss["position"]),np.eye(3)*gnss["std"]**2)
                ekf.config=original
            acceleration=(np.array(frame["evaluation_truth"]["velocity"][uid])-previous_v[uid])/.2
            previous_v[uid]=np.array(frame["evaluation_truth"]["velocity"][uid])
            if not frame["audit"]["boundary_count"] and not frame["audit"]["safety_count"]:
                residual=np.array(packets[uid]["imu"]["accel"])-acceleration-bias[uid]["accel_bias"]-bias[uid]["accel_noise"]
                noise_residual=max(noise_residual,float(np.max(np.abs(residual))))
            # Preserve the native estimate-bound operation. Physical-projection
            # uncertainty is excluded from frozen failed final frames because
            # v7 only retained aggregate projection counts, not per-UAV flags.
            ekf.project_position_bounds(np.array(sim.motion.bounds_min_m),np.array(sim.motion.bounds_max_m))
            row={"scene_id":record["case"]["scene_id"],"seed":record["case"]["seed"],"arm":arm,"uid":uid,"step":step,
                 **state_metric(uid,ekf.state,ekf.covariance,frame["evaluation_truth"],bias),**ekf.diagnostics_snapshot()}
            rows.append(row)
            if arm=="native" and not frame["audit"]["boundary_count"] and not frame["audit"]["safety_count"]:
                native_match_pre_projection=max(native_match_pre_projection,float(np.max(np.abs(ekf.state-np.array(frame["audit"]["states"][uid])))))
    return rows,{"clone_max_residual":clone_residual,"imu_noise_ledger_max_residual":noise_residual,
                 "native_state_match_before_physical_projection":native_match_pre_projection}


def summarize(rows,keys):
    groups={tuple(r[k] for k in keys) for r in rows}
    result=[]
    for group in sorted(groups):
        subset=[r for r in rows if tuple(r[k] for k in keys)==group]
        result.append({**dict(zip(keys,group)),"steps":len(subset),
            "position_rmse_m":float(np.sqrt(np.mean([r["position_sse_m2"] for r in subset]))),
            "accel_bias_rmse":float(np.sqrt(np.mean([r["accel_bias_sse"] for r in subset]))),
            "gyro_bias_rmse":float(np.sqrt(np.mean([r["gyro_bias_sse"] for r in subset]))),
            "position_nees_mean":float(np.mean([r["position_nees"] for r in subset])),
            "vo_accepted":sum(r["vo_update_accepted"] for r in subset),"vo_attempted":sum(r["vo_update_attempted"] for r in subset),
            "gnss_accepted":sum(r["gnss_update_accepted"] for r in subset),"gnss_attempted":sum(r["gnss_update_attempted"] for r in subset),
            "final_bias_z_error":subset[-1]["bias_z_error"],"final_position_trace":subset[-1]["position_trace"],
            "final_accel_bias_trace":subset[-1]["accel_bias_trace"]})
    return result


def freeze():
    spec={"schema":"local_ekf_startup_v8","original_cases":v7.old.read_json(v7.CONFIG)["cases"],
        "initial_errors_by_seed":v7.old.read_json(v7.CONFIG)["initial_errors_by_seed"],"startup_steps":5,
        "physical_step_cap":1440,"offline_group_step_cap":1076,"extra_cases_allowed_only_after_original_pass":True,
        "motion_vo_chi_square_limit":16.27,"vo_min_valid_startup":1,"training_allowed":False,
        "reference_original_success_rmse_degradation_limit_m":.1,
        "candidate_reference_rmse_limit_m":1.,"covariance_full_trace_limit":1e4}
    v7.old.write_json(CONFIG,spec);v7.old.write_json(BASE/"design.lock.json",spec)
    history=v7.old.read_json(v7.BASE/"protected_inputs_before.json")
    final=v7.old.read_json(v7.BASE/"final_artifact_manifest.json")
    history.update({str(ROOT/p):h for p,h in final["artifact_sha256"].items() if p!="README.md"})
    history[str(v7.BASE/"final_artifact_manifest.json")]=v7.old.file_hash(v7.BASE/"final_artifact_manifest.json")
    for p,h in history.items():
        if v7.old.file_hash(p)!=h:raise RuntimeError(f"history changed: {p}")
    v7.old.write_json(BASE/"protected_inputs_before.json",history)
    (BASE/"README_before.md.txt").write_bytes((ROOT/"README.md").read_bytes())
    v7.old.write_json(BASE/"design.sha256.json",{"sha256":v7.old.file_hash(CONFIG)})
    print("Frozen v8 original cases and caps; protected",len(history))


def run(run_id):
    from .partial_gnss_report_v7 import write_csv
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",run_id):raise ValueError("unsafe run-id")
    data=BASE/run_id;logs=LOGS/run_id
    data.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    rows=[];audits=[];count=0
    sim=v7.simulation()
    input_dir=v7.BASE/"bounded_resume_s20267101_102"
    with v7.old._forbid_training_runtime():
        for path in sorted(input_dir.glob("record_*.json")):
            record=v7.old.read_json(path)
            for arm in ARMS:
                result,audit=replay(record,arm,sim);count+=len(record["frames"])
                rows.extend(result);audits.append({"record":path.name,"arm":arm,**audit})
    if count>1076:raise RuntimeError("offline cap")
    write_csv(logs/"per_uav_steps.csv",rows)
    summary=summarize(rows,("scene_id","seed","arm","uid"))
    write_csv(logs/"summary.csv",summary)
    v7.old.write_json(data/"manifest.json",{"offline_group_steps":count,"physical_steps":0,"optimizer_updates":0,"audits":audits,"summary":summary})
    print(v7.old.encode({"audits":audits,"failed_source1": [r for r in summary if r["seed"]==20267102 and r["uid"]=="uav_01"]}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--freeze",action="store_true");parser.add_argument("--run-id",default="frozen_observations_001")
    args=parser.parse_args();freeze() if args.freeze else run(args.run_id)


if __name__=="__main__":main()
