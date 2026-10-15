"""Independent checks of saved v6 records, packets, and covariance metrics."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .cooperative_ekf_reliability_report_v5 import read_rows, value
from ..control.fixed_navigation import fixed_navigation_command
from ..experiments import three_uav_fusion_validation_v6 as v6


def verify(run_id):
    root = v6.BASE / run_id
    logs = v6.ROOT / "logs/preliminary" / v6.STEM / run_id
    rows = [{k: value(r, k) for k in r} for r in read_rows(logs / "localization.csv")]
    events = [{k: value(r, k) for k in r} for r in read_rows(logs / "fusion_events.csv")]
    manifests = [{k: value(r, k) for k in r} for r in read_rows(logs / "records.csv")]
    records = {r["block_id"]: v6.old.read_json(root / (r["block_id"] + "_record.json")) for r in manifests}
    failures = []
    check = lambda condition, label: failures.append(label) if not condition else None
    max_nav_diff = max_packet_diff = 0.
    for manifest in manifests:
        record = records[manifest["block_id"]]
        check(v6.old.content_hash(record) == manifest["record_hash"], "record_checksum")
        check(len(record["frames"]) == 120, "horizon")
        budget = defaultdict(float)
        for frame in record["frames"]:
            audit = frame["producer_audit"]
            for uid in v6.UIDS:
                command = fixed_navigation_command(np.asarray(audit["pre_driver_estimates"][uid])[:6], record["case"]["scenario"]["goals"][uid])["command_delta_v_mps"]
                max_nav_diff = max(max_nav_diff, float(np.max(np.abs(command - np.asarray(audit["actions"][uid]["delta_v"])))))
                check(set(frame["observable"]["sensor_packets"][uid]["imu"]) == {"accel", "gyro"}, "sensor_truth_whitelist")
            for item in audit["budget"]:
                budget[item["uav_id"]] += item["budget_consumed"]
            check(audit["minimum_pair_distance_m"] >= 2 and audit["boundary_count"] == 0 and audit["safety_count"] == 0, "native_safety")
            anchor = frame["observable"]["anchor_probe"]
            check(anchor["std"] == .5 and anchor["valid"] and anchor["uav_id"] == "uav_00", "native_anchor_contract")
            check(not np.array_equal(anchor["position"], frame["evaluation_truth"]["position"]["uav_00"]), "anchor_not_truth")
        check(all(b <= 30 for b in budget.values()), "budget")
    by_step = {(r["block_id"], r["method"], r["condition"], r["step"]): r for r in rows}
    cfg = v6.old.effective_simulation(v6.old.read_json(v6.old.CONFIG)).ekf
    for event in events:
        if event["method"] == "joint_reference":
            continue
        native = event["event"]
        record = records[event["block_id"]]
        measurement = next(m for m in record["frames"][event["step"] - 1]["observable"]["range_attempts"] if m["measurement_id"] == event["measurement_id"])
        send_step = int(round(event["packet_send_time_s"] / .2))
        if send_step == 0:
            source = record["initial"][event["source"]]
            state, covariance = source["state"], source["covariance"]
        else:
            source = by_step[(event["block_id"], event["method"], event["condition"], send_step)]
            state, covariance = source["states"][event["source"]], source["local_covariances"][event["source"]]
        propagated_state, propagated_cov = v6.propagate_packet(state, covariance, measurement["message_envelope"], event["step"] * .2, cfg)
        direction = np.asarray([native["direction_x"], native["direction_y"], native["direction_z"]])
        q = direction @ propagated_cov[:3, :3] @ direction
        distance = np.linalg.norm(np.asarray(native["state_before"])[:3] - propagated_state[:3])
        max_packet_diff = max(max_packet_diff, abs(q - native["neighbor_direction_variance_m2"]), abs(distance - native["predicted_range_m"]))
    joint_tail = defaultdict(list)
    for row in rows:
        if row["method"] != "joint_reference":
            continue
        covariance = np.asarray(row["joint_position_covariance_9"])
        errors = row["position_errors"]
        pair_values = []
        for i, left in enumerate(v6.UIDS):
            for j in range(i + 1, 3):
                right = v6.UIDS[j]
                selector = np.zeros((3, 9))
                selector[:, 3 * i:3 * i + 3] = np.eye(3)
                selector[:, 3 * j:3 * j + 3] = -np.eye(3)
                err = np.asarray(errors[left]) - np.asarray(errors[right])
                pair_values.append(float(err @ np.linalg.pinv(selector @ covariance @ selector.T) @ err))
        check(abs(np.mean(pair_values) - row["relative_nees_mean"]) < 1e-8, "joint_relative_nees_recompute")
        joint_tail[row["condition"]].extend(pair_values)
    check(max_nav_diff < 1e-12, "own_ekf_navigation")
    check(max_packet_diff < 1e-9, "delayed_filter_owned_packet_history")
    attempt_ids = defaultdict(set)
    for event in events:
        attempt_ids[(event["block_id"], event["method"], event["condition"])].add(event["measurement_id"])
    for block in records:
        identities = {m["measurement_id"] for f in records[block]["frames"] for m in f["observable"]["range_attempts"]}
        check(all(attempt_ids[(block, m, c)] == identities for m in v6.METHODS for c in v6.CONDITIONS), "same_physical_range_attempts")
    protected = v6.verify_design(v6.old.read_json(v6.CONFIG))
    result = {"ok": not failures, "failures": failures, "record_count": len(records), "physical_steps": sum(len(r["frames"]) for r in records.values()),
              "primary_shadow_steps": len(rows), "max_raw_own_ekf_navigation_difference": max_nav_diff,
              "max_delayed_packet_field_difference": max_packet_diff, "protected_entries": len(protected),
              "joint_pair_nees_descriptive": {c: {"mean": float(np.mean(vals)), "above_chi3_95_fraction": float(np.mean(np.array(vals) > 7.8147)), "samples": len(vals), "independent_samples": False} for c, vals in joint_tail.items()},
              "ppo_updates": 0, "full_statistical_consistency_claim": False}
    v6.old.write_json(logs / "independent_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe run ID")
    result = verify(args.run_id)
    print(v6.old.encode(result))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
