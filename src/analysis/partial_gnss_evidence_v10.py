"""Post-run, read-only evidence extraction; never advances an estimator."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from ..experiments import partial_gnss_ci_validation_v10 as run
from .partial_gnss_closed_report_v10 import _write_csv


def build(batch="research_001"):
    run.verify()
    base = run.batch_path(batch)
    out = base / "evidence_complete"
    logs = run.LOGS / batch / "evidence_complete"
    out.mkdir(exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    manifests = {p: run.checked_manifest(base/p/"manifest.json")
                 for p in ("develop", "explore", "confirm", "confirmation_replay")}
    summaries = []
    comparisons = []
    kernel = []
    for phase, manifest in manifests.items():
        for method in run.gates.METHODS:
            for policy in ("none", "random_peer"):
                rr = [r for r in manifest["summaries"] if r["method"] == method and r["policy"] == policy]
                if not rr:
                    continue
                timing = [t for r in rr for t in r["kernel_timing_samples_ms"]]
                weight = Counter()
                rejection = Counter()
                for r in rr:
                    weight.update(r["weight_distribution"])
                    rejection.update(r["rejection_counts"])
                summaries.append({"phase": phase, "method": method, "policy": policy,
                    "blocks": len(rr), "absolute_rmse_m": run.gates.pooled(rr, run.gates.METRICS[0]),
                    "relative_rmse_m": run.gates.pooled(rr, run.gates.METRICS[1]),
                    "requests": sum(r["uwb_requested"] for r in rr),
                    "available_selected_broadcasts": sum(r["uwb_message_arrived"] for r in rr),
                    "attempts": sum(r["uwb_attempted"] for r in rr),
                    "absorbed": sum(r["uwb_accepted"] for r in rr),
                    "weight_one": sum(r["ci_weight_one"] for r in rr),
                    "equivalent_request_bytes": sum(r["measurement_bytes_equivalent"] for r in rr),
                    "position_nees_mean": float(np.mean([r["uav2_position_nees_mean"] for r in rr])),
                    "uwb_nis_mean": float(np.mean([r["uwb_nis_mean"] for r in rr if r["uwb_nis_mean"] is not None])) if timing else None,
                    "final_position_trace_mean_m2": float(np.mean([r["target_position_trace_final_m2"] for r in rr])),
                    "full_covariance_min_eig": min(r["covariance_min_eigenvalue"] for r in rr),
                    "full_covariance_trace_max": max(r["full_covariance_trace_max"] for r in rr),
                    "kernel_mean_ms": float(np.mean(timing)) if timing else None,
                    "kernel_p95_ms": float(np.percentile(timing,95)) if timing else None,
                    "weight_distribution": dict(weight), "rejections": dict(rejection)})
        if phase not in ("explore", "confirm"):
            continue
        for case in (run.verify()["development_cases"] if phase == "explore" else run.verify()["confirmation_cases"]):
            scene, seed = case["scene_id"], case["seed"]
            records = {policy: json.loads((base/phase/f"{scene}_s{seed}_{policy}_record.json").read_text(encoding="utf-8")) for policy in ("none", "random_peer")}
            rows = {policy: next(r for r in manifest["summaries"] if r["scene_id"]==scene and r["seed"]==seed and r["policy"]==policy) for policy in records}
            a, b = records["none"], records["random_peer"]
            trajectory = [np.linalg.norm(np.asarray(f["evaluation_truth"]["position"]["uav_02"])-np.asarray(g["evaluation_truth"]["position"]["uav_02"])) for f,g in zip(a["frames"],b["frames"])]
            commands = [np.linalg.norm(np.asarray(f["audit"]["actions"]["uav_02"]["delta_v"])-np.asarray(g["audit"]["actions"]["uav_02"]["delta_v"])) for f,g in zip(a["frames"],b["frames"])]
            item={"phase":phase,"scene_id":scene,"seed":seed,
                  "trajectory_difference_max_m":float(max(trajectory)), "trajectory_final_difference_m":float(trajectory[-1]),
                  "actual_command_difference_max_mps":float(max(commands))}
            for policy, r in rows.items():
                for key in ("uav2_absolute_rmse_m","cluster_relative_rmse_m","uav2_position_nees_mean","uwb_nis_mean","target_position_trace_final_m2","uwb_requested","uwb_attempted","uwb_accepted","ci_weight_one","legal_neighbor_opportunities","two_legal_windows","measurement_bytes_equivalent","minimum_separation_m","maximum_acceleration_mps2","uav2_final_goal_distance_m","uav2_final_speed_mps","uav2_strict_parking"):
                    item[policy+"_"+key]=r[key]
                for key in ("actual_message_arrival_count","request_overhead_packet_load","feedback_update_pairs","feedback_command_change_pairs","controller_command_error_max","max_controller_input_error","sender_payload_state_error_max","sender_payload_covariance_error_max"):
                    item[policy+"_"+key]=r["execution_audit"][key]
                for uid in range(3):
                    item[f"{policy}_uav{uid}_strict_parking"]=r[f"uav{uid}_strict_parking"]
            comparisons.append(item)
        for p in (base/phase).glob("*_estimator.json"):
            d=json.loads(p.read_text(encoding="utf-8"))
            for e in d["events"]:
                event=e.get("event") or {}
                kernel.append({"phase":phase,"record":p.name,"step":e["step"],"attempted":e.get("attempted"),"accepted":e.get("accepted"),
                    "reason":e.get("rejection_reason"),"weight":event.get("weight"),"nis":event.get("nis"),
                    "innovation_m":event.get("innovation_m"),"packet_age_s":e.get("packet_age_s"),
                    "conditional_bound_residual":event.get("conditional_bound_residual_norm")})
    _write_csv(logs/"method_pools.csv",summaries)
    _write_csv(logs/"physical_pairs.csv",comparisons)
    _write_csv(logs/"physical_kernel_events.csv",kernel)
    result={"method_pools":summaries,"physical_pairs":comparisons,
            "optimizer_updates":0,"new_estimator_updates":0,"new_physical_steps":0,
            "replay_audit_scope_note":"Inherited execution_audit in confirmation replay identifies its None record producer, not live shadow-filter communication. Only explore/confirm audit counts here are physical.",
            "input_manifest_sha256":{p:run.v9.v7.old.file_hash(base/p/"manifest.json") for p in manifests}}
    run.v9.v7.old.write_json(out/"metrics.json",result)
    run.write_manifest(out,logs,{"phase":"post_run_read_only_evidence","new_physical_steps":0,"new_estimator_updates":0})
    return result


if __name__ == "__main__":
    build()
