"""Pure descriptive summaries and frozen v9 engineering admission rules."""
from __future__ import annotations

from collections import Counter
import itertools
import numpy as np

from ..experiments import partial_gnss_baseline_v7 as v7


def _mean_valid(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v) and v >= 0]
    return float(np.mean(values)) if values else None


def summarize(record, rows, events, audit, method, policy, phase="replay"):
    frames = [f for f in record["frames"] if "audit" in f]
    attempts = [e for e in events if e.get("attempted", False)]
    accepted = [e for e in attempts if e.get("accepted", False)]
    result = {"scene_id": record["case"]["scene_id"], "seed": record["case"]["seed"],
              "phase": phase, "method": method, "policy": policy, "steps": len(rows),
              "status": record["status"], "failure": record["failure"],
              "uav2_absolute_rmse_m": float(np.sqrt(np.mean([r["target_sse_m2"] for r in rows]))) if rows else None,
              "cluster_relative_rmse_m": float(np.sqrt(np.mean([r["relative_sse_mean_m2"] for r in rows]))) if rows else None,
              "uav2_position_nees_mean": _mean_valid([r["target_position_nees"] for r in rows]),
              "uav2_cumulative_position_sse_m2": float(sum(r["target_sse_m2"] for r in rows)),
              "cluster_cumulative_relative_sse_m2": float(sum(r["relative_sse_mean_m2"] for r in rows)),
              "uwb_requested": len(events), "uwb_message_arrived": sum(bool(e.get("message_arrived")) for e in events),
              "uwb_attempted": len(attempts), "uwb_accepted": len(accepted),
              "effective_fusion_rate": len(accepted)/len(attempts) if attempts else 0.,
              "ci_weight_one": sum(abs(float(e.get("event", {}).get("weight", 0))-1)<1e-12 for e in attempts) if method == "measurement_ci" else None,
              "rejection_counts": dict(Counter(e.get("rejection_reason", "unknown") for e in events if not e.get("accepted"))),
              "uwb_nis_mean": _mean_valid([e.get("event", {}).get("nis") for e in attempts]),
              "packet_age_mean_s": _mean_valid([e.get("packet_age_s") for e in events]),
              "packet_age_max_s": max((e.get("packet_age_s", 0) or 0 for e in events), default=None),
              "measurement_bytes_equivalent": 512*len(events),
              "physical_broadcast_edge_transmissions": sum(len(f["audit"]["sent_edges"]) for f in frames),
              "decision_windows": sum(r["step"] in range(6, 119, 4) for r in rows),
              "legal_neighbor_opportunities": sum(r.get("legal_count", 0) for r in rows),
              "one_legal_windows": sum(r.get("legal_count", 0) == 1 for r in rows),
              "two_legal_windows": sum(r.get("legal_count", 0) == 2 for r in rows),
              "target_position_trace_final_m2": rows[-1]["target_covariance_trace_m2"] if rows else None,
              "target_position_trace_max_m2": max((r["target_covariance_trace_m2"] for r in rows), default=None),
              "minimum_separation_m": min((f["audit"]["minimum_separation_m"] for f in frames), default=None),
              "maximum_acceleration_mps2": max((f["audit"]["max_acceleration_mps2"] for f in frames), default=None),
              "boundary_projections": sum(f["audit"]["boundary_count"] for f in frames),
              "safety_projections": sum(f["audit"]["safety_count"] for f in frames),
              "collision_count": sum(bool(f["audit"]["collision"]) for f in frames),
              "budget_used": rows[-1].get("budget_used", len(events)) if rows else 0,
              "budget_remaining": 30-(rows[-1].get("budget_used", len(events)) if rows else 0),
              "joint_extra_information": method == "joint_reference",
              "communication_model": "fixed None record queue; request accounting only" if phase == "replay" else "native live request/queue",
              "message_arrival_count_scope": "selected reference already-arrived broadcast available at update; not request-response packets",
              "nis_scope": "unknown-cross conservative innovation bound" if method == "measurement_ci" else "method-specific model innovation statistic",
              **audit}
    all_covs = [np.asarray(r["covariances"][u]) for r in rows for u in v7.UIDS]
    result["covariance_finite"] = all(np.isfinite(p).all() for p in all_covs)
    result["covariance_min_eigenvalue"] = min((float(np.linalg.eigvalsh(p)[0]) for p in all_covs), default=None)
    result["full_covariance_trace_max"] = max((float(np.trace(p)) for p in all_covs), default=None)
    for i, uid in enumerate(v7.UIDS):
        errors = [np.asarray(r["states"][uid])[:3]-np.asarray(frames[j]["evaluation_truth"]["position"][uid]) for j, r in enumerate(rows)]
        result[f"uav{i}_rmse_m"] = float(np.sqrt(np.mean([e@e for e in errors]))) if errors else None
        result[f"uav{i}_position_nees_mean"] = _mean_valid([v7.old.nees(e, np.asarray(r["covariances"][uid])[:3, :3]) for e, r in zip(errors, rows)])
        result[f"uav{i}_gnss_valid"] = sum(bool(f["observable"]["sensor_packets"][uid]["gnss"]["valid"]) for f in frames)
        result[f"uav{i}_gnss_accepted"] = sum(r["gnss_accepted_0_1_2"][i] for r in rows)
        diagnostics = [r.get("local_diagnostics", {}).get(uid, frames[j]["audit"]["diagnostics"][uid]) for j, r in enumerate(rows)]
        result[f"uav{i}_vo_accepted"] = sum(d.get("vo_update_accepted", 0) for d in diagnostics)
        result[f"uav{i}_vo_valid"] = sum(bool(f["observable"]["sensor_packets"][uid]["vo"]["valid"]) for f in frames)
        result[f"uav{i}_gnss_nis_mean"] = _mean_valid([d.get("nis_gnss") for d in diagnostics])
        result[f"uav{i}_vo_nis_mean"] = _mean_valid([d.get("nis_vo") for d in diagnostics])
        result[f"uav{i}_mean_position_bias_m"] = np.mean(errors, axis=0).tolist() if errors else None
        result[f"uav{i}_final_position_covariance_trace_m2"] = float(np.trace(np.asarray(rows[-1]["covariances"][uid])[:3, :3])) if rows else None
    if phase == "closed_loop" and frames:
        last = frames[-1]["audit"]
        result.update({"uav2_final_goal_distance_m": last["distances"][v7.TARGET], "uav2_final_speed_mps": last["speeds"][v7.TARGET],
                       "uav2_strict_parking": last["parking"]["per_uav"][v7.TARGET]["ever_success"]})
    assert 0 <= len(accepted) <= len(attempts) <= len(events) <= 29
    assert result["budget_used"] == len(events) <= 30
    return result


def admission(summaries, spec):
    """No strategy selection; all rules apply to predesignated random CI."""
    primary = [r for r in summaries if r["method"] == "measurement_ci" and r["policy"] == "random_peer"]
    baseline = {(r["scene_id"], r["seed"]): r for r in summaries if r["method"] == "measurement_ci" and r["policy"] == "none"}
    g = spec["gate"]
    failures, comparisons, pooled = [], [], {}
    if len(primary) != 8 or len(baseline) != 8:
        failures.append("eight complete paired primary cases required")
    for r in primary:
        b = baseline.get((r["scene_id"], r["seed"]))
        if not b:
            failures.append("missing paired None baseline")
            continue
        change = {"scene_id": r["scene_id"], "seed": r["seed"]}
        for metric in ("uav2_absolute_rmse_m", "cluster_relative_rmse_m"):
            denominator = b.get(metric)
            gain = 1-r[metric]/denominator if denominator is not None and denominator > 1e-12 else None
            change[metric + "_improvement_fraction"] = gain
            if gain is None or gain < -g["maximum_case_degradation_fraction"]:
                failures.append(f"case degradation {r['scene_id']}:{r['seed']}:{metric}")
        comparisons.append(change)
        for label, item in (("baseline", b), ("primary", r)):
            if item.get("steps") != 120 or item.get("status") != "complete":
                failures.append(f"incomplete {label}")
            if any(item.get(x, 1) != 0 for x in ("collision_count", "boundary_projections", "safety_projections")):
                failures.append(f"physical safety {label}")
            if not item.get("covariance_finite") or item.get("covariance_min_eigenvalue", -1) < -1e-8 or item.get("full_covariance_trace_max", 1e5) >= g["covariance_full_trace_limit"]:
                failures.append(f"covariance sanity {label}")
            if item.get("uav2_position_nees_mean", 1e5) > g["target_nees_mean_limit"]:
                failures.append(f"NEES engineering guard {label}")
            for i in (0, 1):
                if item.get(f"uav{i}_rmse_m", 1e5) > g["reference_rmse_limit_m"] or item.get(f"uav{i}_gnss_accepted", 0)/max(1, item.get(f"uav{i}_gnss_valid", 0)) < g["reference_gnss_acceptance_min"]:
                    failures.append(f"reference health {label}:{i}")
            if item.get("uwb_requested", 1e5) > 29 or item.get("budget_used", 1e5) > 30:
                failures.append(f"budget {label}")
    counts = {}
    for scene in sorted({c["scene_id"] for c in spec["cases"]}):
        counts[scene] = {}
        for metric in ("uav2_absolute_rmse_m", "cluster_relative_rmse_m"):
            positive = sum(c["scene_id"] == scene and (c[metric+"_improvement_fraction"] or 0) > 0 for c in comparisons)
            counts[scene][metric] = positive
            if positive < g["positive_seeds_per_geometry"]:
                failures.append(f"cross-seed improvement {scene}:{metric} {positive}/4")
    if len(primary) == 8 and len(baseline) == 8:
        for metric in ("uav2_absolute_rmse_m", "cluster_relative_rmse_m"):
            before = float(np.sqrt(np.mean([b[metric]**2 for b in baseline.values()])))
            after = float(np.sqrt(np.mean([r[metric]**2 for r in primary])))
            gain = 1-after/before if before > 1e-12 else None
            pooled[metric] = {"none": before, "random_ci": after, "improvement_fraction": gain}
            if gain is None or gain < g["pooled_rmse_improvement_fraction"]:
                failures.append(f"pooled improvement {metric}")
    attempted = sum(r.get("uwb_attempted", 0) for r in primary)
    absorbed = sum(r.get("uwb_accepted", 0) for r in primary)
    rate = absorbed/attempted if attempted else 0.
    if rate < g["effective_fusion_rate_min"]:
        failures.append("information absorption engineering guard")
    return {"passed": not failures, "admission": "pass" if not failures else "fail",
            "primary_policy": "random_peer", "method": "measurement_ci", "evaluation_steps": "1..120",
            "independent_seeds": 4, "within_seed_geometry_blocks": 2,
            "paired_changes": comparisons, "positive_seed_counts": counts, "pooled": pooled,
            "limits": dict(g),
            "effective_fusion_rate": rate, "failures": sorted(set(failures)),
            "candidate_closed_loop_allowed": not failures, "scientific_training_ready": False,
            "scope": "pre-registered conservative engineering gate; not statistical consistency/generalization proof"}
