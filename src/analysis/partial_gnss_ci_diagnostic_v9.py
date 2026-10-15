"""Read-only CI objective audit and predesignated instantaneous counterfactuals."""
from __future__ import annotations

import argparse
from collections import Counter
import numpy as np

from ..experiments import partial_gnss_revalidation_v9 as v9
from .partial_gnss_fusion_v7 import MeasurementCIEKF
from .partial_gnss_report_v7 import write_csv


def endpoint_derivative(position_covariance, direction, pseudo_variance):
    """Derivative of existing position-trace CI objective at weight one.

    Negative derivative means weight one minimizes the convex continuous
    objective. Positive does not guarantee the existing discrete grid absorbs.
    This is an objective diagnostic, not a new estimator or truth gain metric.
    """
    p = np.asarray(position_covariance, dtype=float)
    h = np.asarray(direction, dtype=float)
    return float(np.dot(p @ h, p @ h)/float(pseudo_variance)-np.trace(p))


def diagnose(run_id):
    spec = v9.verify_design()
    source = v9.BASE / run_id
    data, logs = v9.BASE / ("ci_diagnostic_"+run_id), v9.LOGS / ("ci_diagnostic_"+run_id)
    for p in (data, logs):
        p.mkdir(parents=True, exist_ok=False)
    comparisons, objective_rows = [], []
    updates = 0
    for case in spec["cases"]:
        block = f"{case['scene_id']}_s{case['seed']}"
        record = v9.v7.old.read_json(source / block / "none_record.json")
        if record["status"] != "complete":
            comparisons.append({"scene_id": case["scene_id"], "seed": case["seed"], "status": "producer_censored"})
            continue
        for step in spec["frozen_state_steps"]:
            frame = record["frames"][step-1]
            bank = frame["observable"]["range_bank"]
            legal = frame["observable"]["pre_legal_sources"]
            base = {"scene_id": case["scene_id"], "seed": case["seed"], "step": step,
                    "both_prelegal_and_observed": all(u in legal and u in bank for u in v9.v7.PEERS),
                    "kind": "offline instantaneous identical-state and recorded-sample; not cumulative or closed-loop"}
            if not base["both_prelegal_and_observed"]:
                comparisons.append(base)
                continue
            x = np.array(frame["audit"]["states"][v9.v7.TARGET])
            p = np.array(frame["audit"]["covariances"][v9.v7.TARGET])
            truth = np.array(frame["evaluation_truth"]["position"][v9.v7.TARGET])
            before = float(np.sum((x[:3]-truth)**2))
            directions = [x[:3]-np.array(bank[u]["packet_state"])[:3] for u in v9.v7.PEERS]
            directions = [d/np.linalg.norm(d) for d in directions]
            base["estimated_axis_angle_deg"] = float(np.degrees(np.arccos(np.clip(abs(directions[0]@directions[1]), 0, 1))))
            true_vectors = [truth-np.array(frame["evaluation_truth"]["position"][u]) for u in v9.v7.PEERS]
            true_vectors = [d/np.linalg.norm(d) for d in true_vectors]
            base["evaluation_true_axis_angle_deg"] = float(np.degrees(np.arccos(np.clip(abs(true_vectors[0]@true_vectors[1]), 0, 1))))
            comparisons.append({**base, "method": "none", "source": "none", "accepted": False,
                                "evaluation_sse_gain_m2": 0., "cost_bytes_equivalent": 0})
            for method, klass in (("measurement_ci", MeasurementCIEKF), ("restore_prior", v9.v7.LocalEKF)):
                for uid in v9.v7.PEERS:
                    if updates >= spec["frozen_state_update_cap"]:
                        raise RuntimeError("frozen counterfactual update cap")
                    m = bank[uid]
                    ekf = klass(x.copy(), v9.v7.simulation().ekf)
                    ekf.covariance = p.copy()
                    accepted = ekf.update_relative_range(m["measurement"], np.array(m["packet_state"])[:3],
                        np.array(m["packet_covariance"])[:3, :3], measurement_variance=m["variance"], nlos_bias_m=m["nlos_bias"], timestamp=step*.2)
                    updates += 1
                    event = ekf.last_relative_range_update
                    row = {**base, "method": method, "source": uid, "accepted": bool(accepted),
                           "rejection_reason": event.get("rejection_reason"), "weight": event.get("weight"),
                           "pre_sse_m2": before, "evaluation_sse_gain_m2": before-float(np.sum((ekf.state[:3]-truth)**2)),
                           "position_trace_gain_m2": float(np.trace(p[:3, :3])-np.trace(ekf.covariance[:3, :3])),
                           "position_change_norm_m": float(np.linalg.norm(ekf.state[:3]-x[:3])),
                           "direction": event.get("direction"), "q_m2": event.get("q"), "R_m2": m["variance"],
                           "packet_age_s": step*.2-m["message_envelope"]["send_time_s"], "cost_bytes_equivalent": 512}
                    if method == "measurement_ci":
                        row["objective_derivative_at_weight_one"] = endpoint_derivative(p[:3, :3], event["direction"], event["q"]+event["R"])
                    comparisons.append(row)
        for policy in spec["policies"]:
            replay = v9.v7.old.read_json(source / block / ("measurement_ci_"+policy+".json"))
            for e in replay["events"]:
                if not e.get("attempted"):
                    continue
                kernel = e["event"]
                w = kernel["weight"]
                H = np.array(kernel["H"])
                final = np.array(kernel["state_covariance_after"])
                t = kernel["q"]+kernel["R"]
                prior = np.linalg.inv((np.linalg.inv(final)-(1-w)*H.T@H/t)/w)
                derivative = endpoint_derivative(prior[:3, :3], kernel["direction"], t)
                if abs(np.trace(prior[:3, :3])-kernel["prior_position_covariance_trace"]) > 1e-7:
                    raise AssertionError("CI prior objective reconstruction")
                objective_rows.append({"scene_id": case["scene_id"], "seed": case["seed"], "policy": policy,
                    "step": e["step"], "source": e["source"], "accepted": e["accepted"],
                    "reason": e.get("rejection_reason"), "weight": w,
                    "objective_derivative_at_weight_one": derivative,
                    "continuous_objective_prefers_no_update": derivative <= 0,
                    "prior_position_eigenvalues_m2": np.linalg.eigvalsh(prior[:3, :3]).tolist(),
                    "direction_variance_m2": kernel["p"], "reference_variance_m2": kernel["q"],
                    "uwb_variance_m2": kernel["R"], "packet_age_s": e.get("packet_age_s"),
                    "evaluation_sse_gain_m2": e.get("evaluation_sse_gain_m2")})
    write_csv(logs / "frozen_state_comparisons.csv", comparisons)
    write_csv(logs / "ci_objective_audit.csv", objective_rows)
    rejection = [r for r in objective_rows if r["reason"] == "no_information_gain"]
    summary = {"fixed_state_updates": updates, "cap": spec["frozen_state_update_cap"],
               "all_ci_attempts": len(objective_rows), "no_information_gain": len(rejection),
               "no_information_gain_with_endpoint_optimum": sum(r["continuous_objective_prefers_no_update"] for r in rejection),
               "rejection_reasons": dict(Counter(r["reason"] for r in objective_rows)),
               "scope": "existing objective audit and instantaneous historical-model comparison, no new fusion algorithm or reward"}
    v9.v7.old.write_json(data / "manifest.json", summary)
    print(v9.v7.old.encode(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="paired_001")
    diagnose(parser.parse_args().run_id)
