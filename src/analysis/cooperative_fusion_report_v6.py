"""Descriptive paired summaries from immutable v6 shadow-filter logs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cooperative_ekf_reliability_report_v5 import read_rows, value
from ..experiments import three_uav_fusion_validation_v6 as v6


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 1e-15 and np.std(y) > 1e-15 else None


def aggregate(run_id):
    log = v6.ROOT / "logs/preliminary" / v6.STEM / run_id
    rows = [{k: value(r, k) for k in r} for r in read_rows(log / "localization.csv")]
    events = [{k: value(r, k) for k in r} for r in read_rows(log / "fusion_events.csv")]
    status = [{k: value(r, k) for k in r} for r in read_rows(log / "method_status.csv")]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["block_id"], row["method"], row["condition"])].append(row)
    summary = []
    alignment = []
    for key, samples in groups.items():
        block, method, condition = key
        group_events = [e for e in events if (e["block_id"], e["method"], e["condition"]) == key]
        metadata = next(s for s in status if (s["block_id"], s["method"], s["condition"]) == key)
        vo_nis, anchor_nis = [], []
        for row in samples:
            diag = row["local_diagnostics"]
            if method == "joint_reference":
                vo_nis.extend(e["nis"] for e in diag["vo_events"])
                anchor_nis.extend(e["nis"] for e in diag["gnss_events"])
            else:
                for uid in v6.UIDS:
                    if diag[uid]["nis_vo"] >= 0:
                        vo_nis.append(diag[uid]["nis_vo"])
                    if diag[uid]["nis_gnss"] >= 0:
                        anchor_nis.append(diag[uid]["nis_gnss"])
        gauge_slack = [float(np.linalg.eigvalsh(np.asarray(r["joint_position_covariance_9"]) - np.kron(np.ones((3, 3)), 2 * np.eye(3)))[0]) for r in samples] if method == "joint_reference" and condition == "no_anchor" else []
        summary.append({"block_id": block, "scene_id": samples[0]["scene_id"], "seed": samples[0]["seed"],
            "method": method, "condition": condition, "status": metadata["status"], "observed_steps": len(samples),
            "absolute_position_rmse_m": float(np.sqrt(np.mean([r["position_squared_error_mean_m2"] for r in samples]))),
            "relative_position_rmse_m": float(np.sqrt(np.mean([r["relative_squared_error_mean_m2"] for r in samples]))),
            "velocity_rmse_mps": float(np.sqrt(np.mean([r["velocity_squared_error_mean_m2_s2"] for r in samples]))),
            "cumulative_distance_error_m_s": float(.2 * sum(r["distance_error_mean_m"] for r in samples)),
            "cumulative_relative_squared_error_m2_s": float(.2 * sum(r["relative_squared_error_mean_m2"] for r in samples)),
            "position_bias_xyz_m": np.mean([r["mean_position_error_vector_m"] for r in samples], axis=0).tolist(),
            "position_nees_mean": float(np.mean([r["position_nees_mean"] for r in samples])),
            "position_velocity_nees_mean": float(np.mean([r["position_velocity_nees_mean"] for r in samples])),
            "relative_nees_mean": float(np.mean([r["relative_nees_mean"] for r in samples])),
            "relative_nees_kind": samples[0]["relative_nees_kind"],
            "vo_nis_mean": float(np.mean(vo_nis)) if vo_nis else None,
            "absolute_probe_nis_mean": float(np.mean(anchor_nis)) if anchor_nis else None,
            "range_nis_mean": float(np.mean([e["event"]["nis"] for e in group_events])) if group_events else None,
            "range_attempts": len(group_events), "accepted_ranges": sum(e["accepted"] for e in group_events),
            "final_position_covariance_trace_mean_m2": samples[-1]["position_covariance_trace_mean_m2"],
            "max_position_covariance_trace_mean_m2": max(r["position_covariance_trace_mean_m2"] for r in samples),
            "minimum_covariance_eigenvalue": min(r["minimum_covariance_eigenvalue"] for r in samples),
            "shared_initial_translation_component_psd_min_slack": min(gauge_slack) if gauge_slack else None,
            "mean_step_pipeline_ms": float(np.mean([r["step_runtime_ms"] for r in samples])),
            "total_replay_pipeline_ms": metadata["total_runtime_ms"],
            "current_covariance_bytes": metadata["covariance_current_bytes"],
            "augmented_covariance_bytes": metadata["peak_augmented_covariance_bytes"],
            "source_only_no_anchor_driver_rmse_m": float(np.sqrt(np.mean([r["source_only_driver_position_squared_error_mean_m2"] for r in samples]))),
            "central_current_information_advantage": method == "joint_reference"})
        for metric in (["relative_vector3d_trace_gain_fraction", "relative_vector3d_logdet_gain", "own_position_trace_gain_fraction"] if method == "joint_reference" else ["old_linear_credit", "reported_own_position_trace_change_m2"]):
            metric_events = [e for e in group_events if e["information"].get(metric) is not None]
            x = [e["information"][metric] for e in metric_events]
            immediate = [e["evaluation_relative_sse_improvement_m2"] for e in metric_events]
            # Temporal associations are descriptive, not causal forecasts.
            remaining = [.2 * sum(r["relative_squared_error_mean_m2"] for r in samples if r["step"] >= e["step"]) for e in metric_events]
            alignment.append({"block_id": block, "method": method, "condition": condition, "metric": metric,
                              "events": len(x), "std": float(np.std(x)) if x else None,
                              "pearson_vs_immediate_relative_sse_improvement": correlation(x, immediate),
                              "pearson_vs_remaining_cumulative_relative_sse": correlation(x, remaining),
                              "temporal_confounding": True, "execution_available_current_distributed": method != "joint_reference",
                              "validated_reward": False})
    pooled = []
    for condition in v6.CONDITIONS:
        for method in v6.METHODS:
            cells = [r for r in summary if r["condition"] == condition and r["method"] == method]
            pooled.append({"condition": condition, "method": method, "blocks": len(cells),
                "absolute_position_rmse_m": float(np.sqrt(np.mean([r["absolute_position_rmse_m"] ** 2 for r in cells]))),
                "relative_position_rmse_m": float(np.sqrt(np.mean([r["relative_position_rmse_m"] ** 2 for r in cells]))),
                "position_nees_mean": float(np.mean([r["position_nees_mean"] for r in cells])),
                "position_velocity_nees_mean": float(np.mean([r["position_velocity_nees_mean"] for r in cells])),
                "range_nis_mean": float(np.mean([r["range_nis_mean"] for r in cells])),
                "mean_step_pipeline_ms": float(np.mean([r["mean_step_pipeline_ms"] for r in cells])),
                "final_position_covariance_trace_mean_m2": float(np.mean([r["final_position_covariance_trace_mean_m2"] for r in cells])),
                "accepted_ranges": sum(r["accepted_ranges"] for r in cells)})
    report = {"summary": summary, "pooled": pooled, "alignment": alignment,
              "scope": "Four TRAIN blocks, two shared seed families; shadow replay, not candidate closed-loop navigation.",
              "reward_designed": False, "scientific_training_ready": False}
    v6.old.write_json(v6.BASE / run_id / "descriptive_report.json", report)
    for name, items in (("fusion_comparison", summary), ("information_alignment", alignment), ("pooled_comparison", pooled)):
        path = log / (name + ".csv")
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            for row in items:
                writer.writerow({k: v6.old.encode(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return report, rows


def plot(run_id, report, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = v6.ROOT / "plots/preliminary" / v6.STEM / run_id
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"restore_prior": "#377EB8", "young_bound": "#D95F02", "joint_reference": "#4DAF4A"}
    for condition, axisrow in zip(v6.CONDITIONS, axes):
        for method in v6.METHODS:
            cells = [r for r in report["summary"] if r["condition"] == condition and r["method"] == method]
            i = v6.METHODS.index(method)
            for ax, metric in zip(axisrow, ("absolute_position_rmse_m", "relative_position_rmse_m")):
                y = [r[metric] for r in cells]
                ax.scatter(np.full(len(y), i) + np.linspace(-.1, .1, len(y)), y, color=colors[method], zorder=3)
                ax.plot([i - .2, i + .2], [np.sqrt(np.mean(np.array(y) ** 2))] * 2, color=colors[method])
        for ax, ylabel in zip(axisrow, ("Absolute position RMSE (m)", "Relative vector RMSE (m)")):
            ax.set_xticks(range(3), ["Restore prior", "Young bound", "Joint reference*"])
            ax.set_ylabel(ylabel)
            ax.set_title(condition)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Four paired TRAIN blocks; *central current-information reference")
    fig.savefig(out / "paired_fusion_rmse.png", dpi=180)
    fig.savefig(out / "paired_fusion_rmse.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for condition, ax in zip(v6.CONDITIONS, axes):
        for method in v6.METHODS:
            curve = [np.mean([r["position_covariance_trace_mean_m2"] for r in rows if r["condition"] == condition and r["method"] == method and r["step"] == step]) for step in range(1, 121)]
            ax.semilogy(np.arange(1, 121) * .2, curve, color=colors[method], label=method)
        ax.set_title(condition)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean position covariance trace (m²)")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Covariance size is not an accuracy score")
    fig.savefig(out / "covariance_trajectory.png", dpi=180)
    fig.savefig(out / "covariance_trajectory.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe run ID")
    report, rows = aggregate(args.run_id)
    plot(args.run_id, report, rows)
    print(v6.old.encode(report["pooled"]))


if __name__ == "__main__":
    main()
