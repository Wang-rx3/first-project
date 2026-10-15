"""Post-analysis completeness and legal-action audit; no new environment work.

Preserve frozen diagnostics verbatim.  This authoritative supplement separates
planned failures from successful points and excludes unavailable actions from
offline oracle ceilings.  Original scientific thresholds remain unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import numpy as np

from ..experiments import neighbor_selection_validation_v12 as run
from .neighbor_selection_diagnostic_v12 import collect_points, summarize_points
from .partial_gnss_report_v7 import write_csv


def point_id(point):
    return point["study_phase"], point["scene_id"], int(point["seed"]), int(point["step"])


def planned_ids(spec):
    return {(phase, c["scene_id"], c["seed"], step)
            for phase, key in (("develop", "development_cases"), ("confirm", "confirmation_cases"))
            for c in spec[key] for step in spec["cf_steps"]}


def classify_points(points, expected):
    counts = Counter(point_id(p) for p in points)
    duplicate = [key for key, n in counts.items() if n != 1]
    missing, extra = expected-set(counts), set(counts)-expected
    valid = [p for p in points if p.get("label_complete") and p.get("identity_matches") is True
             and all(len(p.get("branches", {}).get(a, {}).get("sse_m2", [])) == 4
                     for a in ("none", "uav_00", "uav_01"))]
    return {"planned": len(expected), "entries": len(points), "valid": len(valid),
            "unavailable": len(points)-len(valid), "duplicate": duplicate, "missing": sorted(missing),
            "extra": sorted(extra), "exhaustive": not duplicate and not missing and not extra}, valid


def legal_regret(row):
    allowed = ["none", *[u for u in ("uav_00", "uav_01") if u in row.get("pre_context", {}).get("legal_sources", [])]]
    if not row.get("label_complete"):
        return None
    current = {a: float(row["offline_reward_current_"+a]) for a in allowed}
    four = {a: float(row["offline_reward_fourstate_"+a]) for a in allowed}
    old, minimal = row.get("old_choice"), row.get("minimal_choice")
    best_current = max(allowed, key=lambda a: current[a])  # Offline label only.
    ceiling = max(four.values())
    return {"study_phase": row["study_phase"], "scene_id": row["scene_id"], "seed": row["seed"], "step": row["step"],
            "allowed_actions": allowed, "dual_eligible": row.get("dual_eligible"),
            "oracle_four_net_ceiling_m2": ceiling, "oracle_current_action": best_current,
            "oracle_current_vs_four_correct": float(abs(four[best_current]-ceiling) <= 1e-12),
            "old_choice": old, "minimal_choice": minimal,
            "old_four_net_regret_m2": ceiling-four[old] if old in four else None,
            "minimal_four_net_regret_m2": ceiling-four[minimal] if minimal in four else None,
            "oracle_deployable": False}


def exhaustive_membership(base, logs):
    manifest = run.checked_manifest(base / "manifest.json")
    actual = {str(p.relative_to(run.ROOT)): run.old.file_hash(p) for folder in (base, logs)
              for p in folder.rglob("*") if p.is_file() and p != base / "manifest.json"}
    if actual != manifest["artifact_sha256"]:
        raise RuntimeError("phase artifact membership/hash mismatch: " + str(base))
    return {"manifest_sha256": run.old.file_hash(base / "manifest.json"), "artifact_count": len(actual), "exhaustive": True}


def audit(batch):
    spec = run.verify()
    base = run.validate_batch(batch)
    phase_audits = {p: exhaustive_membership(base / p, run.LOGS / batch / p)
                    for p in ("develop", "confirm", "counterfactual", "report")}
    points = collect_points(base / "counterfactual", base)
    count, valid = classify_points(points, planned_ids(spec))
    if not count["exhaustive"]:
        raise RuntimeError("predeclared checkpoint identity/membership failure")
    eligible = [p for p in valid if p.get("dual_eligible")]
    diagnosis = summarize_points(eligible)
    regrets = [legal_regret(p) for p in valid]
    groups = []
    for phase in ("develop", "confirm"):
        for scene in ["all", *spec["geometry_ids"]]:
            rr = [r for r in regrets if r["study_phase"] == phase and (scene == "all" or r["scene_id"] == scene)]
            groups.append({"study_phase": phase, "scene_id": scene, "valid_points": len(rr),
                           "old_mean_four_net_regret_m2": float(np.mean([r["old_four_net_regret_m2"] for r in rr if r["old_four_net_regret_m2"] is not None])) if rr else None,
                           "minimal_mean_four_net_regret_m2": float(np.mean([r["minimal_four_net_regret_m2"] for r in rr if r["minimal_four_net_regret_m2"] is not None])) if rr else None,
                           "oracle_current_vs_four_accuracy": float(np.mean([r["oracle_current_vs_four_correct"] for r in rr])) if rr else None,
                           "scope": "legal actions only; oracle consistency is offline, not an executable reward validation"})
    manifests = {p: run.checked_manifest(base / p / "manifest.json") for p in ("develop", "confirm", "counterfactual")}
    physical = manifests["develop"]["summaries"]+manifests["confirm"]["summaries"]
    attempted = sum(r["failure"]["step"] if not r["full_horizon_eligible"] and r.get("failure") else r["steps"] for r in physical)
    complete_steps = sum(r["steps"] for r in physical)
    failed = [r for r in physical if not r["full_horizon_eligible"]]
    safety = {"collision_count": sum(r["collision_count"] for r in physical),
              "boundary_projections": sum(r["boundary_projections"] for r in physical),
              "safety_projections": sum(r["safety_projections"] for r in physical),
              "maximum_observed_acceleration_mps2": max(r["maximum_acceleration_mps2"] for r in physical),
              "minimum_observed_separation_m": min(r["minimum_separation_m"] for r in physical),
              "scope": "complete audited steps and retained prefixes; eight incomplete startup transitions cannot be promoted to successful runs"}
    data, logs = base / "evidence_audit", run.LOGS / batch / "evidence_audit"
    data.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    governance = {str(p.relative_to(run.ROOT)): run.old.file_hash(p) for p in
                  [run.CONFIG, run.BASE/"design.lock.json", run.BASE/"design.sha256.json", run.BASE/"unit_release.lock.json",
                   run.BASE/"reporting_addendum.lock.json", run.ROOT/"knowledge/neighbor_selection_v12_protocol.md",
                   Path(__file__).resolve(), run.ROOT/"tests/test_neighbor_selection_evidence_audit_v12.py"]}
    run.old.write_json(data / "source_inputs_before.json", governance)
    write_csv(logs / "legal_action_regrets.csv", regrets)
    write_csv(logs / "legal_action_reward_comparison.csv", groups)
    write_csv(logs / "eligible_feature_correlations.csv", diagnosis["correlations"])
    write_csv(logs / "eligible_rankings.csv", diagnosis["groups"])
    output = {"checkpoint_counts": count, "eligible_dual_points": len(eligible),
              "eligible_meaningful_points": sum(p["meaningful"] for p in eligible),
              "eligible_diagnostic": diagnosis, "legal_regret_groups": groups,
              "physical": {"planned_episodes": 64, "recorded_episodes": len(physical),
                           "full_120step_episodes": len(physical)-len(failed), "failed_episodes": len(failed),
                           "attempted_joint_transitions": attempted, "completed_audited_joint_steps": complete_steps,
                           "incomplete_transitions": attempted-complete_steps, "startup_failure_seed": 2026101324,
                           "failure_before_first_cooperative_window": all(r["failure"]["step"] < 6 and r["uwb_requested"] == 0 for r in failed)},
              "safety": safety, "phase_membership": phase_audits, "governance_sha256": governance,
              "post_analysis_supplement": True, "science_thresholds_changed": False, "optimizer_updates": 0,
              "training_ready": False,
              "interpretation": "supersedes raw diagnostic three-action ceilings and ambiguous point counts; does not change recorded data or frozen scientific assessment"}
    run.old.write_json(data / "audited_metrics.json", output)
    run.write_manifest(data, logs, {"checkpoint_counts": count, "optimizer_updates": 0})
    for name, digest in governance.items():
        if run.old.file_hash(run.ROOT/name) != digest:
            raise RuntimeError("source/input changed during evidence audit")
    run.verify()
    print(run.old.encode({"checkpoint_counts": count, "physical": output["physical"], "safety": safety}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="research_001")
    audit(parser.parse_args().batch_id)
