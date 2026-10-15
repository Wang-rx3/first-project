"""Offline radial/transverse and reduced-Jacobian diagnostics, not a filter."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .cooperative_ekf_reliability_report_v5 import read_rows, value
from ..experiments import three_uav_fusion_validation_v6 as v6


def run(run_id):
    logs = v6.ROOT / "logs/preliminary" / v6.STEM / run_id
    rows = [{k: value(r, k) for k in r} for r in read_rows(logs / "localization.csv")]
    events = [{k: value(r, k) for k in r} for r in read_rows(logs / "fusion_events.csv")]
    blocks = {r["block_id"] for r in rows}
    records = {b: v6.old.read_json(v6.BASE / run_id / (b + "_record.json")) for b in blocks}
    centering = np.eye(9) - np.kron(np.ones((3, 3)) / 3, np.eye(3))
    eigenvalues, vectors = np.linalg.eigh(centering)
    basis = vectors[:, eigenvalues > .5]
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == "joint_reference":
            groups[(row["block_id"], row["condition"])].append(row)
    summaries = []
    for (block, condition), group in groups.items():
        radial_error, transverse_error, radial_nees, transverse_nees, angles = [], [], [], [], []
        for row in group:
            covariance = np.asarray(row["joint_position_covariance_9"])
            truth = records[block]["frames"][row["step"] - 1]["evaluation_truth"]["position"]
            for i, left in enumerate(v6.UIDS):
                for j in range(i + 1, 3):
                    right = v6.UIDS[j]
                    difference = np.asarray(row["states"][left])[:3] - np.asarray(row["states"][right])[:3]
                    unit = difference / np.linalg.norm(difference)
                    true_difference = np.asarray(truth[left]) - np.asarray(truth[right])
                    true_unit = true_difference / np.linalg.norm(true_difference)
                    angles.append(float(np.degrees(np.arccos(np.clip(unit @ true_unit, -1, 1)))))
                    error = np.asarray(row["position_errors"][left]) - np.asarray(row["position_errors"][right])
                    selector = np.zeros((3, 9))
                    selector[:, 3 * i:3 * i + 3] = np.eye(3)
                    selector[:, 3 * j:3 * j + 3] = -np.eye(3)
                    relative_covariance = selector @ covariance @ selector.T
                    projector = np.eye(3) - np.outer(unit, unit)
                    transverse = projector @ error
                    radial_error.append(float((unit @ error) ** 2))
                    transverse_error.append(float(transverse @ transverse))
                    radial_nees.append(float((unit @ error) ** 2 / (unit @ relative_covariance @ unit)))
                    transverse_nees.append(float(transverse @ np.linalg.pinv(projector @ relative_covariance @ projector) @ transverse))
        estimated_gram, true_gram = np.zeros((6, 6)), np.zeros((6, 6))
        for event in events:
            if event["method"] != "joint_reference" or event["block_id"] != block or event["condition"] != condition:
                continue
            step = event["step"]
            raw = next(m for m in records[block]["frames"][step - 1]["observable"]["range_attempts"] if m["measurement_id"] == event["measurement_id"])
            truth = records[block]["frames"][step - 1]["evaluation_truth"]["position"]
            delta = np.asarray(truth[event["target"]]) - np.asarray(truth[event["source"]])
            for unit, gram in ((np.asarray(event["event"]["direction"]), estimated_gram), (delta / np.linalg.norm(delta), true_gram)):
                row = np.zeros(9)
                target = v6.UIDS.index(event["target"])
                source = v6.UIDS.index(event["source"])
                row[3 * target:3 * target + 3] = unit
                row[3 * source:3 * source + 3] = -unit
                reduced = row @ basis
                gram += np.outer(reduced, reduced) / raw["variance"]
        summaries.append({"block_id": block, "condition": condition,
            "radial_error_rmse_m": float(np.sqrt(np.mean(radial_error))),
            "transverse_error_rmse_m": float(np.sqrt(np.mean(transverse_error))),
            "radial_component_nees_mean": float(np.mean(radial_nees)),
            "transverse_component_nees_mean": float(np.mean(transverse_nees)),
            "direction_error_angle_mean_deg": float(np.mean(angles)),
            "estimated_reduced_static_offset_gram_eigenvalues": np.linalg.eigvalsh(estimated_gram).tolist(),
            "truth_reduced_static_offset_gram_eigenvalues": np.linalg.eigvalsh(true_gram).tolist(),
            "claim_limit": "Diagnostic only: reduced static-offset approximation ignores VO noise, velocity/bias states and full dynamic observability; component NEES do not sum when correlated.",
            "estimator_truth_input": False})
    v6.old.write_json(logs / "relative_geometry_diagnostic.json", {"summaries": summaries, "environment_steps": 0})
    return summaries


def curvature(run_id):
    logs = v6.ROOT / "logs/preliminary" / v6.STEM / run_id
    rows = [{k: value(r, k) for k in r} for r in read_rows(logs / "localization.csv")]
    blocks = {r["block_id"] for r in rows}
    records = {b: v6.old.read_json(v6.BASE / run_id / (b + "_record.json")) for b in blocks}
    values = defaultdict(list)
    maximum_identity_error = 0.
    for row in rows:
        if row["method"] != "joint_reference":
            continue
        truth = records[row["block_id"]]["frames"][row["step"] - 1]["evaluation_truth"]["position"]
        for i, left in enumerate(v6.UIDS):
            for right in v6.UIDS[i + 1:]:
                estimated = np.asarray(row["states"][left])[:3] - np.asarray(row["states"][right])[:3]
                actual = np.asarray(truth[left]) - np.asarray(truth[right])
                direction = estimated / np.linalg.norm(estimated)
                radial_error = float(direction @ (estimated - actual))
                distance_error = float(np.linalg.norm(estimated) - np.linalg.norm(actual))
                omitted_term = float(np.linalg.norm(actual) - direction @ actual)
                maximum_identity_error = max(maximum_identity_error, abs(radial_error - distance_error - omitted_term))
                values[(row["block_id"], row["condition"])].append((radial_error, distance_error, omitted_term))
    result = {"maximum_exact_identity_error": maximum_identity_error,
        "identity": "projected vector error = range norm error + true norm*(1-cos direction error)",
        "diagnostic_truth_only": True, "environment_steps": 0,
        "summaries": [{"block_id": b, "condition": c, "radial_vector_error_rmse_m": float(np.sqrt(np.mean(np.asarray(v)[:, 0] ** 2))),
                       "range_norm_error_rmse_m": float(np.sqrt(np.mean(np.asarray(v)[:, 1] ** 2))),
                       "curvature_term_rmse_m": float(np.sqrt(np.mean(np.asarray(v)[:, 2] ** 2)))} for (b, c), v in values.items()]}
    v6.old.write_json(logs / "curvature_diagnostic.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--curvature-only", action="store_true")
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe run ID")
    print(v6.old.encode(curvature(args.run_id) if args.curvature_only else run(args.run_id)))


if __name__ == "__main__":
    main()
