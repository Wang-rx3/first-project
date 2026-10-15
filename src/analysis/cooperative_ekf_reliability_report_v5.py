"""Offline, descriptive aggregation of the locked v5 diagnostic CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = "three_uav_cooperative_ekf_reliability_diag_v5_seed2026_20261012"
UIDS = ("uav_00", "uav_01", "uav_02")


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def value(row, key, default=None):
    raw = row.get(key, "")
    if raw == "":
        return default
    if raw in ("True", "False"):
        return raw == "True"
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def mean(rows, key, *, nonnegative=False):
    numbers = [float(value(row, key)) for row in rows if value(row, key) is not None]
    numbers = [x for x in numbers if np.isfinite(x) and (not nonnegative or x >= 0)]
    return float(np.mean(numbers)) if numbers else None


def aggregate(run_id):
    log_root = ROOT / "logs/preliminary" / STEM / run_id
    localization = read_rows(log_root / "localization.csv")
    transitions = read_rows(log_root / "transitions.csv")
    episodes = read_rows(log_root / "episode_metrics.csv")
    branches = read_rows(log_root / "branches.csv")
    updates = read_rows(log_root / "relative_update.csv")
    groups = {}
    for row in localization:
        if row["uav_id"] in UIDS:
            groups.setdefault(row["episode_id"], []).append(row)
    result_rows = []
    for episode in episodes:
        rows = groups[episode["episode_id"]]
        team_rows = [r for r in localization if r["episode_id"] == episode["episode_id"] and r["uav_id"] == "team"]
        # Persisted episode IDs can differ in separator style; scene/seed/arm
        # are the authoritative pairing fields when joining native step logs.
        steps = [row for row in transitions if row.get("episode_id") == episode["episode_id"]]
        if not steps:
            steps = [row for row in transitions if row.get("arm") == episode["arm"] and
                     row.get("scene_id") == episode["scene_id"] and row.get("seed") == episode["seed"]]
        errors = np.asarray([float(row["position_error_m"]) for row in rows])
        bias = np.mean([value(row, "position_error_vector_m") for row in rows], axis=0)
        accepted = 0
        for row in updates:
            if row.get("episode_id") == episode["episode_id"]:
                accepted += bool(value(row, "event", {}).get("accepted", False))
        budget = overhead = ordinary = 0.0
        for row in steps:
            budget += sum(float(event.get("budget_consumed", 0)) for event in value(row, "measurement_budget_events", []))
            overhead += sum(float(event.get("load_packets_applied", 0)) for event in value(row, "measurement_overhead_events", []))
            ordinary += float(value(row, "ordinary_state_tx_count", 0))
        result_rows.append({
            "scene_id": episode["scene_id"], "seed": int(episode["seed"]), "arm": episode["arm"],
            "episode_id": episode["episode_id"], "observed_steps": int(episode["observed_steps"]),
            "censored": str(episode["censored"]).lower() == "true", "position_sample_denominator": len(rows),
            "position_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "cumulative_mean_position_error_m_s": float(np.sum(errors) / 3 * 0.2),
            "integrated_mean_squared_position_error_m2_s": float(np.sum(errors ** 2) / 3 * 0.2),
            "relative_pair_vector_rmse_m": float(np.sqrt(np.mean([float(r["relative_pair_squared_error_m2"]) for r in team_rows]))),
            "position_bias_xyz_m": bias.tolist(), "nees_position_mean": mean(rows, "nees_position_3d"),
            "nees_position_velocity_mean": mean(rows, "nees_position_velocity_6d"),
            "nees_position_above_chi3_95_fraction_descriptive": float(np.mean([float(r["nees_position_3d"]) > 7.8147 for r in rows])),
            "vo_nis_mean": mean(rows, "vo_nis", nonnegative=True),
            "uwb_nis_mean": mean(rows, "uwb_nis", nonnegative=True),
            "accepted_measurements": accepted, "consumed_budget_units": budget,
            "measurement_overhead_packets_equivalent": overhead,
            "measurement_overhead_bytes_equivalent": overhead * 512,
            "ordinary_successful_state_tx_count": ordinary,
            "minimum_pair_distance_m": float(episode["minimum_pair_distance_m"]),
            "boundary_projected_uid_steps": sum(bool(value(row, "boundary_projected", False)) for row in rows),
            "safety_projected_uid_steps": sum(bool(value(row, "safety_projected", False)) for row in rows),
            "max_actual_speed_mps": max(float(row["actual_speed_mps"]) for row in rows),
            "max_actual_acceleration_mps2": max(float(row["actual_acceleration_mps2"]) for row in rows),
            "covariance_position_max_eigenratio": max(float(max(value(r, "position_covariance_eigenvalues")) / min(value(r, "position_covariance_eigenvalues"))) for r in rows),
            "strict_parking_ever_uid_count": sum(any(bool(value(r, "strict_parking_ever_success", False)) for r in rows if r["uav_id"] == uid) for uid in UIDS),
            "final_mean_goal_distance_m": mean([r for r in rows if int(r["step"]) == int(episode["observed_steps"])], "goal_distance_m"),
        })
    pair_rows = []
    for row in result_rows:
        baseline = next(item for item in result_rows if item["scene_id"] == row["scene_id"] and
                        item["seed"] == row["seed"] and item["arm"] == "none")
        base_paths = {(r["uav_id"], r["step"]): np.asarray(value(r, "true_position_m")) for r in localization if r["episode_id"] == baseline["episode_id"] and r["uav_id"] in UIDS}
        path_differences = [float(np.linalg.norm(np.asarray(value(r, "true_position_m")) - base_paths[(r["uav_id"], r["step"])])) for r in localization if r["episode_id"] == row["episode_id"] and r["uav_id"] in UIDS and (r["uav_id"], r["step"]) in base_paths]
        pair_rows.append({**row, "rmse_change_vs_none_m": row["position_rmse_m"] - baseline["position_rmse_m"],
                          "cum_error_change_vs_none_m_s": row["cumulative_mean_position_error_m_s"] - baseline["cumulative_mean_position_error_m_s"],
                          "max_true_path_displacement_vs_none_m": max(path_differences),
                          "paired_h120_eligible": not row["censored"] and not baseline["censored"]})
    accepted_events = [value(r, "event", {}) for r in updates if bool(value(r, "event", {}).get("accepted", False))]
    proxy = np.asarray([(e["position_covariance_trace_before_m2"] - e["position_covariance_trace_after_linear_m2"]) / e["position_covariance_trace_before_m2"] for e in accepted_events])
    improvement = np.asarray([e["true_position_error_before_m"] - e["true_position_error_after_m"] for e in accepted_events])
    alignment = {"accepted_events": len(accepted_events), "improving_events": int(np.sum(improvement > 0)),
                 "worsening_events": int(np.sum(improvement < 0)), "mean_immediate_distance_improvement_m": float(np.mean(improvement)),
                 "proxy_vs_immediate_error_improvement_pearson_descriptive": float(np.corrcoef(proxy, improvement)[0, 1]) if len(proxy) > 1 and np.std(proxy) > 0 and np.std(improvement) > 0 else None,
                 "final_covariance_gain_max_abs_m2": max(abs(e["position_covariance_trace_before_m2"] - e["position_covariance_trace_after_final_m2"]) for e in accepted_events)}
    report = {"schema": "v5_offline_descriptive_report", "run_id": run_id, "episodes": pair_rows,
              "reward_alignment": alignment,
              "branch_rows": [{k: value(row, k) for k in row} for row in branches],
              "claim_scope": "Four paired TRAIN blocks, two shared seed families; correlated steps are not independent replicates.",
              "scientific_training_ready": False}
    out = ROOT / "data/synthetic" / STEM / run_id / "descriptive_report.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = log_root / "strategy_comparison.csv"
    if csv_path.exists():
        raise FileExistsError(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({key: json.dumps(val) if isinstance(val, list) else val for key, val in row.items()})
    return report, localization


def plot(run_id, report, localization):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blocks = sorted({(row["scene_id"], row["seed"]) for row in report["episodes"]})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"none": "#333333", "fixed": "#377EB8", "random_peer": "#984EA3", "random-peer": "#984EA3", "greedy": "#D95F02"}
    for ax, (scene, seed) in zip(axes.flat, blocks):
        for episode in [r for r in report["episodes"] if r["scene_id"] == scene and r["seed"] == seed]:
            rows = [r for r in localization if r["episode_id"] == episode["episode_id"] and r["uav_id"] in UIDS]
            steps = sorted({int(r["step"]) for r in rows})
            rms = [np.sqrt(np.mean([float(r["position_error_m"]) ** 2 for r in rows if int(r["step"]) == step])) for step in steps]
            ax.plot(np.asarray(steps) * .2, rms, label=episode["arm"], color=colors.get(episode["arm"]))
        ax.set_title(f"{scene}; seed {seed}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Team position RMS error (m)")
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False, ncol=2)
    fig.suptitle("Bounded TRAIN diagnostic: four paired blocks, two seed families")
    out = ROOT / "plots/preliminary" / STEM / run_id / "trajectory_position_error.png"
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe run id")
    report, rows = aggregate(args.run_id)
    plot(args.run_id, report, rows)
    print(json.dumps({"episodes": len(report["episodes"]), "scientific_training_ready": False}))


if __name__ == "__main__":
    main()
