"""Pure v10 safety, development selection, and confirmation assessment."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from ..analysis.partial_gnss_gate_v9 import summarize as summarize_v9

METHODS = ("original_ci", "refined_ci")
METRICS = ("uav2_absolute_rmse_m", "cluster_relative_rmse_m")


def pooled(rows, metric):
    values = [r[metric] for r in rows]
    if not values or any(v is None or not np.isfinite(v) for v in values):
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def summarize(record, rows, events, audit, method, policy, phase):
    result = summarize_v9(record, rows, events, audit, "measurement_ci", policy,
                          phase="replay" if "replay" in phase else "closed_loop")
    result.update(method=method, policy=policy, phase=phase)
    timing = [e["event"]["compute_time_ms"] for e in events if e.get("attempted") and "compute_time_ms" in e.get("event", {})]
    result["kernel_time_ms_mean"] = float(np.mean(timing)) if timing else None
    result["kernel_time_ms_p95"] = float(np.percentile(timing, 95)) if timing else None
    result["kernel_time_ms_sum"] = float(sum(timing))
    result["kernel_timing_samples_ms"] = timing
    result["weight_distribution"] = dict(__import__("collections").Counter(str(e["event"].get("weight")) for e in events if e.get("attempted")))
    result["conditional_bound_mismatch_count"] = sum(e.get("accepted", False) and not e["event"].get("conditional_bound_matches_posterior", False) for e in events)
    result["state_abs_max"] = max((float(np.max(np.abs(x))) for row in rows for x in row["states"].values()), default=None)
    result["replay_interface_valid"] = all(e["step"] in range(6, 119, 4) for e in events) and result["budget_used"] <= 30 and len(events) <= 29
    result["poststartup_abs_rmse_m"] = float(np.sqrt(np.mean([r["target_sse_m2"] for r in rows if r["step"] > 5]))) if len(rows) > 5 else None
    result["relative_nees_mean"] = float(np.mean([r["relative_nees_mean"] for r in rows])) if rows else None
    result["relative_nees_kind"] = "unknown_cross_upper_bound"
    result["execution_audit"] = record.get("execution_audit", {})
    # Shadow replay does not change the producer's physical navigation/parking.
    frames = [] if "replay" in phase else [f for f in record["frames"] if "states" in f.get("audit", {})]
    for i, uid in enumerate(("uav_00", "uav_01", "uav_02")):
        if frames:
            final = frames[-1]["audit"]
            result[f"uav{i}_strict_parking"] = final["parking"]["per_uav"][uid]["ever_success"]
            result[f"uav{i}_final_goal_distance_m"] = final["distances"][uid]
            result[f"uav{i}_final_speed_mps"] = final["speeds"][uid]
    if result.get("status") != "complete" or len(rows) != 120:
        for metric in METRICS:
            result["prefix_"+metric] = result.get(metric)
            result[metric] = None
        result["full_horizon_eligible"] = False
    else:
        result["full_horizon_eligible"] = True
    return result


def safety_admission(summaries, spec, tests_passed):
    output = {}
    for method in METHODS:
        cases = [r for r in summaries if r.get("method") == method]
        failures = []
        if not tests_passed:
            failures.append("unit/interface release missing")
        if len(cases) != 32:
            failures.append("32 development replay combinations required")
        timings = []
        for r in cases:
            label = f"{r.get('scene_id')}:{r.get('seed')}:{r.get('policy')}"
            if not r.get("full_horizon_eligible"):
                failures.append("incomplete replay "+label)
                continue
            if not r.get("covariance_finite") or r.get("covariance_min_eigenvalue", -1) < -1e-8 or r.get("full_covariance_trace_max", 1e5) >= 1e4:
                failures.append("covariance numerical bound "+label)
            if r.get("state_abs_max", 1e5) >= 1e3:
                failures.append("state numerical bound "+label)
            if r.get("conditional_bound_mismatch_count", 1):
                failures.append("conditional linear identity "+label)
            if not r.get("replay_interface_valid") or r.get("packet_reconstruction_max", 1) > 1e-8 or r.get("source_reconstruction_max", 1) > 1e-8:
                failures.append("communication/reconstruction interface "+label)
            timings.extend(r.get("kernel_timing_samples_ms", []))
        p95 = float(np.percentile(timings, 95)) if timings else None
        if p95 is None or p95 >= spec["safety"]["kernel_p95_limit_ms"]:
            failures.append("kernel exceeds decision period computation budget")
        output[method] = {"passed": not failures, "failures": failures, "kernel_p95_ms": p95,
                          "scope": "numerical/interface execution admission; no RMSE, NEES or absorption-rate threshold"}
    return output


def select_candidate(summaries, spec, safety):
    by_method = {m: {(r["scene_id"], r["seed"]): r for r in summaries if r.get("method") == m and r.get("policy") == "random_peer"} for m in METHODS}
    original, refined = by_method["original_ci"], by_method["refined_ci"]
    failures, changes, aggregate = [], [], {}
    if len(original) != 8 or len(refined) != 8:
        failures.append("eight complete development primary pairs required")
    for metric in METRICS:
        before, after = pooled(list(original.values()), metric), pooled(list(refined.values()), metric)
        gain = 1-after/before if before is not None and before > 1e-12 and after is not None else None
        aggregate[metric] = {"original": before, "refined": after, "improvement_fraction": gain,
                             "improvement_m": before-after if before is not None and after is not None else None}
        if gain is None or gain < spec["selection"]["minimum_fractional_improvement"]:
            failures.append("pooled fractional improvement "+metric)
        if metric == METRICS[0] and (before is None or after is None or before-after < spec["selection"]["absolute_rmse_min_gain_m"]):
            failures.append("pooled absolute dimensional improvement")
    for key, b in original.items():
        a = refined.get(key)
        if a is None:
            continue
        row = {"scene_id": key[0], "seed": key[1]}
        for metric in METRICS:
            gain = 1-a[metric]/b[metric] if a.get(metric) is not None and b.get(metric) is not None and b[metric] > 1e-12 else None
            row[metric+"_improvement_fraction"] = gain
            if gain is None or gain < -spec["selection"]["maximum_case_degradation_fraction"]:
                failures.append("case harm "+str(key)+":"+metric)
        changes.append(row)
    seed_counts = {}
    for metric in METRICS:
        count = 0
        for seed in spec["development_seeds"]:
            before = pooled([r for r in original.values() if r["seed"] == seed], metric)
            after = pooled([r for r in refined.values() if r["seed"] == seed], metric)
            count += int(before is not None and after is not None and before-after > 1e-12)
        seed_counts[metric] = count
        if count < spec["selection"]["positive_independent_seeds"]:
            failures.append("seed-level benefit "+metric)
    beneficial = not failures
    if safety["refined_ci"]["passed"] and beneficial:
        selected, reason = "refined_ci", "safe refined grid meets all predeclared development benefit rules"
    elif safety["original_ci"]["passed"]:
        selected, reason = "original_ci", "retain safe original; refined benefit/safety criterion not met"
    elif safety["refined_ci"]["passed"]:
        selected, reason = "refined_ci", "only safe candidate; exploratory execution without improvement claim"
    else:
        selected, reason = None, "both candidates failed numerical/interface safety admission"
    return {"selected_method": selected, "reason": reason, "refined_development_benefit": beneficial,
            "benefit_failures": sorted(set(failures)), "safety": safety, "pooled": aggregate,
            "seed_positive_counts": seed_counts, "paired_changes": changes,
            "closed_loop_execution_allowed": selected is not None, "scientific_training_ready": False,
            "selection_data": "development replay only; immutable before confirmation observations"}


def physical_safety(summaries, expected=16):
    failures = []
    if len(summaries) != expected:
        failures.append("all predeclared physical episodes required")
    for r in summaries:
        label = f"{r['scene_id']}:{r['seed']}:{r['policy']}"
        if not r.get("full_horizon_eligible"):
            failures.append("stopped episode "+label)
        if any(r.get(k, 1) != 0 for k in ("collision_count", "boundary_projections", "safety_projections")):
            failures.append("physical safety event "+label)
        if not r.get("covariance_finite") or (r.get("covariance_min_eigenvalue") or 0) < -1e-8 or (r.get("full_covariance_trace_max") or 1e5) >= 1e4:
            failures.append("covariance numerical event "+label)
        if not r.get("replay_interface_valid"):
            failures.append("budget/selection interface "+label)
        audit = r.get("execution_audit", {})
        if (not audit.get("native_live_queue", False) or audit.get("controller_command_error_max", 1) > 1e-10
                or audit.get("sender_payload_state_error_max", 1) > 1e-10
                or audit.get("sender_payload_covariance_error_max", 1) > 1e-10
                or audit.get("max_controller_input_error", 1) > 1e-10
                or not audit.get("source_constraints_uid0_uid1_none", False)
                or not audit.get("selection_window_legal", False)
                or abs(audit.get("request_overhead_packet_load", -1)-r.get("uwb_requested", -2)) > 1e-10):
            failures.append("live queue/navigation payload audit "+label)
    return {"passed": not failures, "failures": failures,
            "scope": "physical/numerical/interface safety only; does not require accuracy improvement or fusion rate"}


def confirmation_performance(summaries, spec):
    safety = physical_safety(summaries)
    failures = list(safety["failures"])
    before = {(r["scene_id"], r["seed"]): r for r in summaries if r["policy"] == "none"}
    primary = [r for r in summaries if r["policy"] == "random_peer"]
    changes, aggregate, counts = [], {}, {}
    if len(primary) != 8 or len(before) != 8:
        failures.append("eight full paired confirmation blocks required")
    for r in summaries:
        if not r.get("full_horizon_eligible"):
            continue
        for uid in (0, 1):
            if r[f"uav{uid}_rmse_m"] > spec["performance"]["reference_rmse_limit_m"] or r[f"uav{uid}_gnss_accepted"]/max(1, r[f"uav{uid}_gnss_valid"]) < spec["performance"]["reference_gnss_acceptance_min"]:
                failures.append(f"reference health {r['scene_id']}:{r['seed']}:{uid}")
        if r["uav2_position_nees_mean"] > spec["performance"]["position_nees_descriptive_limit"]:
            failures.append(f"descriptive NEES guard {r['scene_id']}:{r['seed']}:{r['policy']}")
    for r in primary:
        b = before.get((r["scene_id"], r["seed"]))
        row = {"scene_id": r["scene_id"], "seed": r["seed"]}
        for metric in METRICS:
            gain = 1-r[metric]/b[metric] if b and b.get(metric) is not None and b[metric] > 1e-12 and r.get(metric) is not None else None
            row[metric+"_improvement_fraction"] = gain
            if gain is None or gain < -spec["performance"]["maximum_case_degradation_fraction"]:
                failures.append(f"case error degradation {r['scene_id']}:{r['seed']}:{metric}")
        changes.append(row)
    for metric in METRICS:
        b, a = pooled(list(before.values()), metric), pooled(primary, metric)
        gain = 1-a/b if b is not None and b > 1e-12 and a is not None else None
        aggregate[metric] = {"none": b, "selected_random": a, "improvement_fraction": gain}
        if gain is None or gain < spec["performance"]["pooled_rmse_improvement_fraction"]:
            failures.append("pooled confirmation benefit "+metric)
    for scene in spec["geometry_ids"]:
        counts[scene] = {}
        for metric in METRICS:
            count = sum(r["scene_id"] == scene and (r[metric+"_improvement_fraction"] or 0) > 0 for r in changes)
            counts[scene][metric] = count
            if count < spec["performance"]["positive_seeds_per_geometry"]:
                failures.append("independent seed repeatability "+scene+":"+metric)
    return {"passed": not failures, "failures": sorted(set(failures)), "pooled": aggregate,
            "paired_changes": changes, "positive_seed_counts": counts, "safety": safety,
            "position_nees_scope": "3D marginal, full1..120 mean; descriptive guard not calibrated chi-square test",
            "strict_parking_is_separate_navigation_outcome": True, "scientific_training_ready": False}
