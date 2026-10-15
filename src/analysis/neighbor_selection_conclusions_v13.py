"""Read-only decomposition: localization gain versus shared cost shortcuts."""
from __future__ import annotations
import math
from collections import Counter
import numpy as np
from ..experiments import neighbor_selection_validation_v13 as run
from .neighbor_selection_report_v13 import correlation, _write_csv

PEERS = ("uav_00", "uav_01")


def stratify(points):
    result = []
    for phase in ("develop", "confirm"):
        for seed in ("all", *sorted({p["seed"] for p in points if p["study_phase"] == phase})):
            subset = [p for p in points if p["study_phase"] == phase and p.get("dual_eligible") and (seed == "all" or p["seed"] == seed)]
            for label in ("current_norm_gain_m", "four_norm_gain_m", "norm_utility_m"):
                valid = [p for p in subset if all(p.get("norm_labels", {}).get(u, {}).get(label) is not None for u in PEERS)]
                for stratum in ("all_dual", "raw_difference_at_least_0.01m"):
                    pp = [p for p in valid if stratum == "all_dual" or abs(p["norm_labels"][PEERS[0]][label] - p["norm_labels"][PEERS[1]][label]) >= .01]
                    truth = [p["norm_labels"][PEERS[0]][label] - p["norm_labels"][PEERS[1]][label] for p in pp]
                    score = [p["v13_score_difference"] for p in pp]
                    # Remove the very same known cost used in both score/utility.
                    raw_score = [p["v13_score_difference"] + .01*(p["norm_labels"][PEERS[0]]["cost_unit"] - p["norm_labels"][PEERS[1]]["cost_unit"]) for p in pp]
                    correct = [.5 if abs(s) <= 1e-12 else float(s * g > 0) for s, g in zip(score, truth) if abs(g) >= .01]
                    raw_correct = [.5 if abs(s) <= 1e-12 else float(s * g > 0) for s, g in zip(raw_score, truth) if abs(g) >= .01]
                    result.append({"study_phase": phase, "seed": seed, "label": label, "stratum": stratum, "n": len(pp),
                        "score_pearson": correlation(score, truth), "score_spearman": correlation(score, truth, True),
                        "root_trace_only_pearson": correlation(raw_score, truth), "root_trace_only_spearman": correlation(raw_score, truth, True),
                        "rank_accuracy": float(np.mean(correct)) if correct else None,
                        "root_trace_only_rank_accuracy": float(np.mean(raw_correct)) if raw_correct else None,
                        "zero_gain_both_neighbors": sum(abs(p["norm_labels"][u]["four_norm_gain_m"]) < 1e-12 for p in pp for u in PEERS) // 2
                        if all(all(abs(p["norm_labels"][u]["four_norm_gain_m"]) < 1e-12 for u in PEERS) or all(abs(p["norm_labels"][u]["four_norm_gain_m"]) >= 1e-12 for u in PEERS) for p in pp) else
                        sum(all(abs(p["norm_labels"][u]["four_norm_gain_m"]) < 1e-12 for u in PEERS) for p in pp)})
    return result


def analyze():
    spec = run.verify()
    base = run.validate_batch("research_001")
    aggregate = run.old.read_json(base / "report/aggregate.json")
    points = run.checked_manifest(base / "counterfactual/manifest.json")["results"]
    rows = sum((run.checked_manifest(base / phase / "manifest.json")["summaries"] for phase in ("develop", "confirm")), [])
    analysis = stratify(points)
    confirm_dual = [p for p in points if p["study_phase"] == "confirm" and p.get("dual_eligible")]
    utility_only = [p for p in confirm_dual if p["meaningful_norm"] and
                    abs(p["norm_labels"][PEERS[0]]["four_norm_gain_m"] - p["norm_labels"][PEERS[1]]["four_norm_gain_m"]) < .01]
    raw_meaningful = [p for p in confirm_dual if abs(p["norm_labels"][PEERS[0]]["four_norm_gain_m"] - p["norm_labels"][PEERS[1]]["four_norm_gain_m"]) >= .01]
    bykey = {(r["study_phase"], r["scene_id"], r["seed"], r["policy"]): r for r in rows}
    matched = []
    for phase in ("develop", "confirm"):
        for case in spec["development_cases"] if phase == "develop" else spec["confirmation_cases"]:
            key = (phase, case["scene_id"], case["seed"])
            random, greedy = (bykey[key + (p,)] for p in ("random_peer", "v13_greedy"))
            common = sorted(set(random["request_windows"]) & set(greedy["request_windows"]))
            matched.append({"study_phase": phase, "scene_id": case["scene_id"], "seed": case["seed"],
                "random_requests": random["uwb_requested"], "v13_requests": greedy["uwb_requested"],
                "request_windows_identical": random["request_windows"] == greedy["request_windows"],
                "common_request_windows": common, "common_count": len(common),
                "scope": "different feedback histories even at common windows; not equal-budget peer-only causal comparison"})
    # Observable-feature diagnostics against raw gain, rather than only net utility.
    feature_rows = []
    features = sorted({k for p in points for k in p.get("v13_features", {})})
    for phase in ("develop", "confirm"):
        for seed in ("all", *(spec["development_seeds"] if phase == "develop" else spec["confirmation_seeds"])):
            for feature in features:
                pairs = []
                for p in points:
                    if p["study_phase"] != phase or not p.get("dual_eligible") or (seed != "all" and p["seed"] != seed):
                        continue
                    v = p.get("v13_features", {}).get(feature)
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        y = p["norm_labels"][PEERS[0]]["four_norm_gain_m"] - p["norm_labels"][PEERS[1]]["four_norm_gain_m"]
                        pairs.append((v, y))
                feature_rows.append({"study_phase": phase, "seed": seed, "feature": feature, "n": len(pairs),
                    "pearson": correlation([x for x,y in pairs], [y for x,y in pairs]),
                    "spearman": correlation([x for x,y in pairs], [y for x,y in pairs], True),
                    "scope": "descriptive multiple-feature inspection, no fitted model or independent variable selection"})
    qa = {phase: run.old.read_json(base / (phase + "_interface_audit_v2.json")) for phase in ("develop", "confirm")}
    report = {"raw_vs_cost_strata": analysis, "confirm_dual": len(confirm_dual), "confirm_raw_meaningful": len(raw_meaningful),
        "confirm_net_meaningful": aggregate["assessment"]["confirmation"]["informative_points"],
        "confirm_net_meaningful_without_0_01m_localization_difference": len(utility_only),
        "confirm_both_zero_localization_gain": sum(all(abs(p["norm_labels"][u]["four_norm_gain_m"]) < 1e-12 for u in PEERS) for p in confirm_dual),
        "raw_meaningful_by_step": dict(Counter(p["step"] for p in raw_meaningful)),
        "matched_request_windows": matched, "feature_raw_gain_correlations": feature_rows,
        "interface_qa_passed": all(v["passed"] for v in qa.values()),
        "minimum_separation_m": min(r["minimum_separation_m"] for r in rows),
        "maximum_acceleration_mps2": max(r["maximum_acceleration_mps2"] for r in rows),
        "collision_count": sum(r["collision_count"] for r in rows), "boundary_projections": sum(r["boundary_projections"] for r in rows),
        "safety_projections": sum(r["safety_projections"] for r in rows),
        "uav0_max_rmse_m": max(r["uav0_rmse_m"] for r in rows), "uav1_max_rmse_m": max(r["uav1_rmse_m"] for r in rows),
        "uav0_min_gnss_acceptance": min(r["uav0_gnss_accepted"] / r["uav0_gnss_valid"] for r in rows),
        "uav1_min_gnss_acceptance": min(r["uav1_gnss_accepted"] / r["uav1_gnss_valid"] for r in rows),
        "ref_anisotropy_ratio_max": max(p["anisotropy_ratio"] for v in qa.values() for q in v["records"] for p in q["reference_position_covariances"]),
        "gnss_role_flips_all_records": all(q["declared_quality_roles"].get("peer0_better",0)>0 and q["declared_quality_roles"].get("peer1_better",0)>0 for v in qa.values() for q in v["records"]),
        "final_training_ready": aggregate["training_ready"], "optimizer_updates": 0,
        "interpretation": "raw gain reveals a smaller genuine localization-choice sample than net utility; cost-known ranking is not localization-value proof"}
    path, logs = base / "interpretation", run.LOGS / "research_001/interpretation"
    path.mkdir(exist_ok=False); logs.mkdir(exist_ok=False)
    run.old.write_json(path / "evidence.json", report)
    _write_csv(logs / "raw_vs_net_ranking.csv", analysis)
    _write_csv(logs / "raw_gain_feature_correlations.csv", feature_rows)
    _write_csv(logs / "matched_request_windows.csv", matched)
    _write_csv(logs / "per_seed_localization_communication.csv", rows)
    run.write_manifest(path, logs, {"scope": "read-only interpretation addendum", "training_ready": report["final_training_ready"]})
    print(run.old.encode({k:report[k] for k in ("confirm_raw_meaningful", "confirm_net_meaningful", "confirm_both_zero_localization_gain", "raw_meaningful_by_step", "gnss_role_flips_all_records", "final_training_ready")}), flush=True)


if __name__ == "__main__":
    analyze()
