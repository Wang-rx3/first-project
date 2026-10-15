"""Export v9 scientific plots from existing records only; zero simulation."""
from __future__ import annotations
import argparse
import numpy as np
from ..experiments import partial_gnss_revalidation_v9 as v9
from .partial_gnss_report_v7 import write_csv


def draw(run_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = v9.BASE / run_id
    summaries = v9.v7.old.read_json(source / "manifest.json")["summaries"]
    selected = [r for r in summaries if r["method"] == "measurement_ci"]
    scenes = sorted({r["scene_id"] for r in selected})
    seeds = sorted({r["seed"] for r in selected})
    policies = ("none", "fixed0", "fixed1", "random_peer")
    colors = ("#555555", "#0072B2", "#D55E00", "#009E73")
    plots = v9.ROOT / "plots/preliminary" / v9.STEM / ("visual_"+run_id)
    logs = v9.LOGS / ("visual_"+run_id)
    for p in (plots, logs):
        p.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for j, scene in enumerate(scenes):
        for k, policy in enumerate(policies):
            subset = sorted((r for r in selected if r["scene_id"] == scene and r["policy"] == policy), key=lambda r: r["seed"])
            axes[0, j].bar(np.arange(4)+(k-1.5)*.19, [r["uav2_absolute_rmse_m"] for r in subset], .19, color=colors[k], label=policy)
        axes[0, j].set(title=scene.replace("train_", ""), ylabel="UAV 2 absolute RMSE (m)", xticks=np.arange(4), xticklabels=[str(s) for s in seeds])
        axes[0, j].tick_params(axis="x", labelrotation=15)
        cooperative = [p for p in policies if p != "none"]
        rates, labels = [], []
        for policy in cooperative:
            subset = [r for r in selected if r["scene_id"] == scene and r["policy"] == policy]
            absorbed = sum(r["uwb_accepted"] for r in subset)
            attempts = sum(r["uwb_attempted"] for r in subset)
            rates.append(absorbed/attempts)
            labels.append(f"{absorbed}/{attempts}")
        bars = axes[1, j].bar(cooperative, rates, color=colors[1:])
        axes[1, j].bar_label(bars, labels=labels, padding=4)
        axes[1, j].set(ylabel="CI absorbed / attempted", ylim=(0, max(.5, max(rates)+.1)))
        axes[1, j].axhline(.2, color="#777777", linestyle=":", label="Frozen engineering guard")
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=9)
    fig.suptitle("Paired record replay; four independent seeds, two within-seed geometries")
    for suffix in ("png", "svg"):
        fig.savefig(plots / ("paired_ci_results."+suffix), dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    curves = []
    for j, scene in enumerate(scenes):
        for k, policy in enumerate(policies):
            tapes = [v9.v7.old.read_json(source / f"{scene}_s{s}" / ("measurement_ci_"+policy+".json"))["rows"] for s in seeds]
            steps = np.arange(1, 121)
            error = np.sqrt(np.mean([[r["target_sse_m2"] for r in rows] for rows in tapes], axis=0))
            trace = np.mean([[r["target_covariance_trace_m2"] for r in rows] for rows in tapes], axis=0)
            count = np.mean([[r["cumulative_accepted"] for r in rows] for rows in tapes], axis=0)
            for step, e, t, n in zip(steps, error, trace, count):
                curves.append({"scene_id": scene, "policy": policy, "step": int(step), "uav2_error_rms_m": float(e), "position_trace_mean_m2": float(t), "cumulative_accepted_mean": float(n)})
            for col, values in enumerate((error, trace, count)):
                axes[j, col].plot(steps, values, color=colors[k], label=policy)
                axes[j, col].set_xlabel("Decision step")
        for col, label in enumerate(("UAV 2 error RMS (m)", "Mean position covariance trace (m²)", "Mean cumulative CI absorption")):
            axes[j, col].axvspan(1, 5, color="#eeeeee", zorder=0)
            axes[j, col].set_ylabel(label)
            axes[j, col].set_title(scene.replace("train_", ""))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("None-driven trajectories; covariance trace is not a true accuracy reward")
    for suffix in ("png", "svg"):
        fig.savefig(plots / ("paired_ci_trajectories."+suffix), dpi=240)
    plt.close(fig)
    write_csv(logs / "curves.csv", curves)
    print(str(plots), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="paired_001")
    draw(parser.parse_args().run_id)
