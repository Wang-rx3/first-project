"""Post-hoc, read-only objective resolution audit; never changes CI weights.

Dense objective samples are diagnostics only: no dense-grid posterior is
applied. Eight predesignated step-58 historical restore-prior probes at most
complete the already frozen 96-update counterfactual ceiling.
"""
from __future__ import annotations
import argparse
import numpy as np
from ..experiments import partial_gnss_revalidation_v9 as v9
from .partial_gnss_report_v7 import write_csv


def objective_values(position_covariance, direction, pseudo_variance, weights):
    p = np.asarray(position_covariance)
    h = np.asarray(direction)
    w = np.asarray(weights)
    radial = float(h @ p @ h)
    numerator = float(np.dot(p @ h, p @ h))
    alpha = (1-w)/(w*float(pseudo_variance))
    return (np.trace(p)-alpha*numerator/(1+alpha*radial))/w


def audit(run_id):
    spec = v9.verify_design()
    previous = v9.v7.old.read_json(v9.BASE / ("ci_diagnostic_"+run_id) / "manifest.json")
    data, logs = v9.BASE / ("resolution_"+run_id), v9.LOGS / ("resolution_"+run_id)
    for path in (data, logs):
        path.mkdir(parents=True, exist_ok=False)
    curves, probes = [], []
    consumed = previous["fixed_state_updates"]
    for case in spec["cases"]:
        block = f"{case['scene_id']}_s{case['seed']}"
        record = v9.v7.old.read_json(v9.BASE / run_id / block / "none_record.json")
        for policy in ("fixed0", "fixed1", "random_peer"):
            tape = v9.v7.old.read_json(v9.BASE / run_id / block / ("measurement_ci_"+policy+".json"))
            for event in tape["events"]:
                if event.get("rejection_reason") != "no_information_gain":
                    continue
                kernel = event["event"]
                p = np.array(kernel["state_covariance_after"])
                h = np.array(kernel["direction"])
                variance = kernel["q"]+kernel["R"]
                original_weights = np.arange(1, 21)/20
                original_scores = objective_values(p[:3, :3], h, variance, original_weights)
                if abs(float(np.min(original_scores))-np.trace(p[:3, :3])) > 1e-7:
                    raise AssertionError("existing grid objective reconstruction")
                diagnostic_weights = np.linspace(.95, 1., 501)
                diagnostic_scores = objective_values(p[:3, :3], h, variance, diagnostic_weights)
                idx = int(np.argmin(diagnostic_scores))
                gain = float(np.trace(p[:3, :3])-diagnostic_scores[idx])
                curves.append({"scene_id": case["scene_id"], "seed": case["seed"], "policy": policy,
                               "step": event["step"], "source": event["source"], "original_weight": 1.,
                               "diagnostic_objective_best_sample_weight": float(diagnostic_weights[idx]),
                               "diagnostic_trace_gain_m2": gain, "mean_or_covariance_update_applied": False})
            if policy != "random_peer":
                continue
            selected = next((e for e in tape["events"] if e["step"] == 58), None)
            base = {"scene_id": case["scene_id"], "seed": case["seed"], "step": 58,
                    "method": "existing_restore_prior_offline_probe_on_actual_CI_prior", "available": bool(selected and selected.get("attempted")),
                    "scope": "instantaneous only; historical covariance has no general consistency guarantee"}
            if not base["available"]:
                probes.append(base)
                continue
            if consumed >= spec["frozen_state_update_cap"]:
                raise RuntimeError("shared frozen 96-update counterfactual cap")
            kernel = selected["event"]
            w = kernel["weight"]
            h = np.array(kernel["H"])
            post = np.array(kernel["state_covariance_after"])
            p = np.linalg.inv((np.linalg.inv(post)-(1-w)*h.T@h/(kernel["q"]+kernel["R"]))/w)
            x = np.array(kernel["state_before"])
            frame = record["frames"][57]
            source = selected["source"]
            m = frame["observable"]["range_bank"][source]
            estimator = v9.v7.LocalEKF(x.copy(), v9.v7.simulation().ekf)
            estimator.covariance = (p+p.T)/2
            accepted = estimator.update_relative_range(m["measurement"], np.array(m["packet_state"])[:3],
                np.array(m["packet_covariance"])[:3, :3], measurement_variance=m["variance"], nlos_bias_m=m["nlos_bias"], timestamp=58*.2)
            consumed += 1
            truth = np.array(frame["evaluation_truth"]["position"][v9.v7.TARGET])
            probes.append({**base, "source": source, "ci_weight": w, "ci_accepted": selected["accepted"],
                           "ci_instant_sse_gain_m2": selected["evaluation_sse_gain_m2"], "restore_accepted": bool(accepted),
                           "restore_instant_sse_gain_m2": float(np.sum((x[:3]-truth)**2)-np.sum((estimator.state[:3]-truth)**2))})
    write_csv(logs / "existing_objective_resolution.csv", curves)
    write_csv(logs / "step58_actual_ci_prior_probes.csv", probes)
    gains = [r["diagnostic_trace_gain_m2"] for r in curves if r["diagnostic_trace_gain_m2"] > 1e-9]
    random = [r for r in curves if r["policy"] == "random_peer"]
    summary = {"post_hoc_objective_audit": True, "existing_no_information_gain_count": len(curves),
               "between_original_grid_points_objective_improvement_count": len(gains),
               "diagnostic_positive_trace_gain_m2_min_median_max": [min(gains), float(np.median(gains)), max(gains)] if gains else None,
               "random_no_information_gain_count": len(random),
               "random_between_grid_points_positive_objective_count": sum(r["diagnostic_trace_gain_m2"] > 1e-9 for r in random),
               "total_frozen_state_updates": consumed, "total_frozen_state_update_cap": spec["frozen_state_update_cap"],
               "step58_probes": probes, "weight_rules_changed": False, "new_fusion_algorithms": 0,
               "dense_objective_scope": "read-only scalar position-trace samples; never covariance/mean update or true benefit claim"}
    v9.v7.old.write_json(data / "manifest.json", summary)
    print(v9.v7.old.encode(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="paired_001")
    audit(parser.parse_args().run_id)
