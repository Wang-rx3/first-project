"""Recompute v5 log invariants offline, without constructing an environment."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .cooperative_ekf_reliability_report_v5 import ROOT, STEM, UIDS, read_rows, value
from ..experiments import three_uav_cooperative_ekf_reliability_v5 as v5


def verify(run_id):
    base = ROOT / "logs/preliminary" / STEM / run_id
    loc = read_rows(base / "localization.csv")
    steps = read_rows(base / "transitions.csv")
    updates = read_rows(base / "relative_update.csv")
    branches = [{k: value(r, k) for k in r} for r in read_rows(base / "branches.csv")]
    errors = []

    def check(condition, label):
        if not condition:
            errors.append(label)

    update_groups = defaultdict(list)
    for row in updates:
        event = value(row, "event")
        update_groups[(row["episode_id"], row["step"])].append(event)
        check(abs(float(row["raw_final_trace"]) - event["position_covariance_trace_after_final_m2"]) < 1e-10, "final_trace_field")
        check(abs(event["nis"] - event["innovation_m"] ** 2 / event["innovation_variance_m2"]) < 1e-10, "range_nis")
    budgets = defaultdict(float)
    attempts = accepted = 0
    for row in steps:
        budget = value(row, "measurement_budget_events", [])
        overhead = value(row, "measurement_overhead_events", [])
        events = update_groups[(row["episode_id"], row["step"])]
        _, recomputed = v5.v3.contract().reward_components({"relative_range_update_events": events, "measurement_budget_events": budget, "measurement_overhead_events": overhead}, float(v5.read_json(v5.CONFIG)["credit_scale"]))
        logged = value(row, "reward_components")
        check(all(abs(recomputed[u][k] - logged[u][k]) < 1e-12 for u in UIDS for k in recomputed[u]), "reward_interface")
        for item in budget:
            budgets[(row["episode_id"], item["uav_id"])] += item["budget_consumed"]
        attempts += len(events)
        accepted += sum(bool(e["accepted"]) for e in events)
        check(all(bool(value(row, key)) for key in ("bounds_ok", "speed_limit_ok", "acceleration_limit_ok", "pair_clearance_ok")), "physical_limits")
    check(max(budgets.values()) <= 30, "budget_cap")
    own_rows = [r for r in loc if r["uav_id"] in UIDS]
    check(sum(int(r["uwb_nis_attempted"]) for r in own_rows) == attempts, "nis_attempt_count")
    check(sum(int(r["uwb_nis_accepted"]) for r in own_rows) == accepted, "nis_accept_count")
    for row in own_rows:
        eig = np.asarray(value(row, "position_covariance_eigenvalues"))
        check(np.allclose(eig, eig[0], atol=1e-12), "isotropy")
        check(abs(float(row["nees_position_3d"]) - float(row["position_error_m"]) ** 2 / eig[0]) < 1e-9, "position_nees")
        check(not bool(value(row, "collision")) and not bool(value(row, "boundary_projected")) and not bool(value(row, "safety_projected")), "projection_or_collision")

    groups = defaultdict(list)
    for row in branches:
        groups[(row["scene_id"], row["seed"], row["snapshot_step"])].append(row)
    for key, rows in groups.items():
        originals = [r for r in rows if not r["branch_id"].endswith("_replay")]
        check(len(originals) == 3, "branch_trio")
        none = next(r for r in originals if r["target"] == "none")
        for field in ("pre_hash", "pre_motion_hash", "pre_bandwidth_hash", "exogenous_sensor_uwb_hash", "sender_payload_hash"):
            check(len({r[field] for r in originals}) == 1, "branch_" + field)
        for row in originals:
            for uid in UIDS:
                check(np.allclose(row["post_beliefs"][uid]["covariance"], none["post_beliefs"][uid]["covariance"], atol=1e-12), "branch_covariance_restoration")
            if row["target"] != "none":
                correction = np.asarray(row["relative_update_events"][0]["state_correction"])
                difference = np.asarray(row["post_beliefs"]["uav_00"]["state"]) - np.asarray(none["post_beliefs"]["uav_00"]["state"])
                check(np.allclose(difference, correction, atol=1e-10), "branch_mean_correction")
                check(abs(row["improvement_vs_none"] - (none["target_post_position_error_m"] - row["target_post_position_error_m"])) < 1e-12, "paired_post_contrast")
        replay = next((r for r in rows if r["branch_id"].endswith("_replay")), None)
        if replay:
            check(replay["post_hash"] == next(r for r in originals if r["target"] == "uav_01")["post_hash"], "replay")
    v5.validate_execution_sources_lock()
    protected = v5.protected_input_hashes()
    result = {"ok": not errors, "errors": errors, "trajectory_steps": len(steps), "localization_uid_steps": len(own_rows), "update_attempts": attempts, "accepted_updates": accepted, "branch_steps": len(branches), "protected_file_entries": len(protected), "maximum_budget_consumed_per_uid_episode": max(budgets.values()), "training_updates": 0, "normalization_fits": 0}
    path = base / "independent_verification.json"
    v5.write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe run id")
    result = verify(args.run_id)
    print(json.dumps(result))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
