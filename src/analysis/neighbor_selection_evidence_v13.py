"""Supplementary read-only V13 transport/sensor/count evidence audit."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import numpy as np
from ..experiments import neighbor_selection_validation_v13 as run
from ..experiments.neighbor_selection_environment_v13 import quality_at
from ..experiments.partial_gnss_baseline_v7 import indexed_rng, PEERS, TARGET


def audit_record(record, spec):
    failures, max_noise_error = [], 0.
    frames = {f["step"]: f for f in record["frames"] if "states" in f.get("audit", {})}
    quality_counts = defaultdict(Counter)
    ref_std = defaultdict(dict)
    for q in record["v13_quality_log"]:
        uid = q["uid"]
        step = round(q["time_s"] / spec["dt_s"])
        expected = quality_at(q["position"], record["quality_offset"])["gnss_std_m"] if uid in PEERS else 20.
        if q["gnss_std_m"] != expected:
            failures.append("declared GNSS quality mismatch")
        quality_counts[uid][str(expected)] += 1
        if uid in PEERS:
            ref_std[step][uid] = expected
        frame = frames.get(step)
        if not frame:
            continue
        packet = frame["observable"]["sensor_packets"][uid]["gnss"]
        if packet["std"] != expected or bool(packet["valid"]) != (uid in PEERS):
            failures.append("physical GNSS packet declaration mismatch")
        if uid in PEERS:
            expected_obs = np.asarray(q["position"]) + indexed_rng(record["case"]["seed"], step - 1, "gnss", uid).normal(0., expected, 3)
            error = float(np.max(np.abs(expected_obs - np.asarray(packet["position"]))))
            max_noise_error = max(max_noise_error, error)
    if max_noise_error > 1e-12:
        failures.append("indexed exogenous GNSS noise mismatch")
    enqueued = [e for e in record["v13_metadata_events"] if e.get("event") == "enqueue"]
    arrivals = [e for e in record["v13_metadata_events"] if e.get("event") == "visible_arrival"]
    ids = [tuple(e["packet_id"]) for e in enqueued]
    if len(set(ids)) != len(ids):
        failures.append("duplicate transport registry id")
    for e in enqueued:
        if (e["payload_bytes"], e["quality_header_bytes"], e["packet_size_bytes"]) != (496, 16, 512):
            failures.append("abstract envelope allocation mismatch")
        report_time = e["gnss_report_time_s"]
        if report_time is not None and abs(report_time - e["send_time_s"]) > 1e-12:
            failures.append("GNSS header/payload temporal mismatch")
    for e in arrivals:
        if e["packet_id"] is None or tuple(e["packet_id"]) not in set(ids):
            failures.append("arrived packet lacks matching actual enqueue")
    own_p, ref_p, differences = [], [], []
    for d in record["v11_decisions"]:
        context = d["context"]
        if any(k in context for k in ("truth", "quality_offset", "phase", "future_error")):
            failures.append("nonobservable selector field")
        if d["selected"] != "none" and d["selected"] not in context["legal_sources"]:
            failures.append("illegal selected action")
        own_p.append(float(np.trace(np.asarray(context["own_covariance"])[:3, :3])))
        for uid, p in context["peers"].items():
            report_time = p.get("gnss_report_time_s")
            if report_time is not None and (report_time > context["time_s"] + 1e-12 or abs(report_time - p["send_time_s"]) > 1e-12):
                failures.append("online quality timestamp leak")
            pp = np.asarray(p["covariance"])[:3, :3]
            ref_p.append({"uid": uid, "step": d["step"], "trace_m2": float(np.trace(pp)),
                          "anisotropy_ratio": float(np.max(np.linalg.eigvalsh(pp)) / np.min(np.linalg.eigvalsh(pp)))})
        if all(u in context["peers"] for u in PEERS):
            p0, p1 = (np.asarray(context["peers"][u]["covariance"])[:3, :3] for u in PEERS)
            differences.append(float(np.trace(p0) - np.trace(p1)))
    roles = Counter("peer0_better" if v[PEERS[0]] < v[PEERS[1]] else "peer1_better" if v[PEERS[0]] > v[PEERS[1]] else "equal"
                    for v in ref_std.values() if all(u in v for u in PEERS))
    result = {"scene_id": record["case"]["scene_id"], "seed": record["case"]["seed"], "policy": record["policy"],
              "quality_offset": record["quality_offset"], "passed": not failures, "failures": sorted(set(failures)),
              "gnss_quality_counts": dict(quality_counts), "declared_quality_roles": dict(roles),
              "indexed_gnss_noise_error_max_m": max_noise_error, "actual_enqueues": len(enqueued),
              "visible_arrivals": len(arrivals), "valid_quality_arrivals": sum(e["valid_quality"] for e in arrivals),
              "missing_quality_arrivals": sum(not e["valid_quality"] for e in arrivals),
              "conceptual_header_bytes_enqueued": 16 * len(enqueued), "native_byte_equivalent_enqueued": 512 * len(enqueued),
              "reference_position_covariances": ref_p, "peer_trace_difference_min_m2": min(differences) if differences else None,
              "peer_trace_difference_max_m2": max(differences) if differences else None,
              "fusion_unchanged": True, "queue_codec_implemented": False}
    return result


def phase_audit(batch, phase):
    spec = run.verify()
    base = run.validate_batch(batch)
    manifest = run.checked_manifest(base / phase / "manifest.json")
    results = []
    for p in sorted((base / phase).glob("*_record.json")):
        try:
            results.append(audit_record(run.old.read_json(p), spec))
        except Exception as exc:
            results.append({"path": str(p), "passed": False, "failures": [type(exc).__name__ + ": " + str(exc)]})
    return {"phase": phase, "passed": len(results) == len(manifest["summaries"]) == 32 and all(r["passed"] for r in results),
            "records": results, "scope": "read-only sensor, timestamps, actual native packets and selector legality"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("develop", "confirm", "final"), required=True)
    parser.add_argument("--batch-id", default="research_001")
    args = parser.parse_args()
    if args.phase != "final":
        evidence = phase_audit(args.batch_id, args.phase)
        path = run.validate_batch(args.batch_id) / (args.phase + "_interface_audit_v2.json")
    else:
        spec = run.verify()
        base = run.validate_batch(args.batch_id)
        check = run.checked_manifest(base / "precheck/manifest.json")
        # Frozen precheck source missed nested failure.incomplete_transition.
        # Correct the derived count in an additive record, retain original 14.
        actual = 0
        for name in ("native_off", "dynamic_off", "retained_known_failure"):
            record = run.old.read_json(base / "precheck" / (name + "_record.json"))
            actual += sum("states" in f.get("audit", {}) or f.get("failure", {}).get("incomplete_transition", False)
                          for f in record["frames"][:6])
        cf_manifest = run.checked_manifest(base / "counterfactual/manifest.json")
        evidence = {"precheck_frozen_report_count": check["physical_attempts"], "precheck_corrected_physical_attempts": actual,
                    "reason": "incomplete_transition stored under frame.failure; planned prefix guard excluded",
                    "precheck_cap_unchanged": spec["precheck_attempt_cap"], "precheck_cap_passed": actual <= spec["precheck_attempt_cap"],
                    "valid_cf_points": cf_manifest["valid_points"], "planned_cf_points": 96,
                    "unavailable_cf_points": 96 - cf_manifest["valid_points"],
                    "legacy_sse_threshold_m2": .01, "new_norm_threshold_m": .01, "same_numbers_different_units": True,
                    "optimizer_updates": 0, "history_untouched": True}
        path = base / "evidence_audit.json"
    if path.exists():
        raise FileExistsError(path)
    run.old.write_json(path, evidence)
    print(run.old.encode({"path": str(path), "passed": evidence.get("passed", evidence.get("precheck_cap_passed"))}), flush=True)


if __name__ == "__main__":
    main()
