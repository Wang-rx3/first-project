"""Reproducible five-seed figures from frozen train/validation files only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import final_mainline_audit as base

PARTS = ("reward_loc", "reward_com", "reward_collision", "reward_energy")
STYLES = ("-", "--", "-.")
MARKERS = ("o", "s", "^")
TOL = 1e-6
HORIZONS = (1, 2, 3, 5)


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("required numeric value is not finite")
    return value


def optional(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def write_table(path, rows):
    """Union columns; NA is explicit rather than silently coerced to zero."""
    if not rows:
        raise ValueError(f"empty evidence table: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("NA" if isinstance(v, (float, np.floating)) and not math.isfinite(v) else v)
                             for k, v in row.items()})


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SourceGuard:
    """Audit actual file opens; forbid test reads and source writes while active."""
    def __init__(self, roots, outputs):
        self.roots = [p.resolve() for p in roots]
        self.outputs = [p.resolve() for p in outputs]
        self.hashes = {}
        self.reads = 0
        self.test_attempts = 0
        self.active = False
        self.busy = False

    def __enter__(self):
        self.active = True
        sys.addaudithook(self.hook)
        return self

    def __exit__(self, *args):
        self.active = False

    def hook(self, event, args):
        if not self.active or self.busy or event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(p) for p in self.outputs) or not any(path.is_relative_to(p) for p in self.roots):
            return
        mode, flags = args[1], args[2]
        if (mode and any(c in mode for c in "wa+")) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            raise PermissionError(f"source write blocked: {path}")
        if "test" in path.parts or path.name.startswith("test_"):
            self.test_attempts += 1
            raise PermissionError(f"sealed test read blocked: {path}")
        if path.parent.name == "diagnostics" and path.name in {"cooperative_updates.jsonl", "episode_metrics.csv", "measurement_quality.jsonl"}:
            raise PermissionError(f"mixed-split diagnostic read blocked: {path}")
        self.reads += 1
        if path not in self.hashes and path.is_file():
            self.busy = True
            try:
                self.hashes[path] = digest(path)
            finally:
                self.busy = False

    def verify(self):
        self.busy = True
        try:
            return [{"path": str(p), "sha256_before": h, "sha256_after": digest(p),
                     "unchanged": digest(p) == h} for p, h in self.hashes.items()]
        finally:
            self.busy = False


def reward_windows(rows, scenario):
    """Reconstruct signed contributions using the saved weighted normalization."""
    config = scenario["reward_config"]
    weights = config["weights"]
    grouped = defaultdict(list)
    for row in rows:
        if row.get("split") != "train" or row.get("episode_id") != scenario["episode_id"]:
            raise ValueError("reward row does not belong to the bound train episode")
        grouped[number(row["timestamp"])].append(row)
    if not grouped:
        raise ValueError("empty training reward file")
    output = []
    for timestamp, group in sorted(grouped.items()):
        if len(group) != int(scenario["uav_count"]) or len({r["uav_id"] for r in group}) != len(group):
            raise ValueError("incomplete or duplicated UAV window")
        mean = lambda key: float(np.mean([number(r[key]) for r in group]))
        loc = number(weights["uncertainty"]) * mean("normalized_uncertainty")
        com = sum(number(weights[k]) * mean("normalized_" + col)
                  for k, col in (("loss", "packet_loss"), ("aoi", "aoi"), ("delay", "delay")))
        com -= number(weights["connectivity"]) * mean("normalized_connectivity")
        result = {"timestamp": timestamp, "mean_reward": mean("reward_total"),
                  "reward_loc": -loc, "reward_com": -com,
                  "reward_collision": mean("reward_collision"), "reward_energy": mean("reward_energy")}
        result["closure_error"] = result["mean_reward"] - sum(result[k] for k in PARTS)
        result["objective_closure_error"] = loc + com - mean("joint_objective")
        result["collision_closure_error"] = result["reward_collision"] + number(config["lambda_coll"]) * mean("collision")
        if max(abs(result[k]) for k in ("closure_error", "objective_closure_error", "collision_closure_error")) > TOL:
            raise ValueError(f"reward closure failed at {timestamp}: {result}")
        output.append(result)
    return output


def training_evidence(config):
    output, window_rows, contributions = [], [], []
    for seed in config.seeds:
        for branch in config.branches:
            scale = base._scale_dir(config.reference_root, seed, branch)
            logs = [r for r in base._read_jsonl(scale / "diagnostics/training_metrics.jsonl") if r.get("phase") == "train"]
            expected = set(range(1, config.expected_training_updates + 1))
            if len(logs) != len(expected) or {int(r["update"]) for r in logs} != expected:
                raise ValueError(f"missing/duplicate updates: {seed}/{branch}")
            if len({r["episode_id"] for r in logs}) != len(logs):
                raise ValueError("profile requires one unique episode per update")
            previous = None
            cell = []
            for log in sorted(logs, key=lambda r: int(r["update"])):
                episode_id = log["episode_id"]
                if Path(episode_id).name != episode_id or not episode_id.startswith("episode_"):
                    raise ValueError("invalid episode binding")
                episode = scale / "train" / episode_id
                scenario = base._read_json(episode / "scenario.json")
                if scenario["episode_id"] != episode_id or scenario["split"] != "train":
                    raise ValueError("scenario binding mismatch")
                windows = reward_windows(base._read_csv(episode / "rewards.csv"), scenario)
                identifiers = {"base_seed": seed, "branch": branch, "update": int(log["update"]), "episode_id": episode_id}
                row = {**identifiers, "window_count": len(windows)}
                for metric in base.TRAINING_METRICS:
                    row[metric] = number(log[metric])
                row["collision_occurred"] = int(base._bool(log.get("collision_occurred")))
                for metric in PARTS:
                    row[metric] = float(np.mean([w[metric] for w in windows]))
                    row[metric + "_mean_absolute"] = float(np.mean([abs(w[metric]) for w in windows]))
                    row[metric + "_delta_update"] = row[metric] - previous[metric] if previous else float("nan")
                row["mean_reward_delta_update"] = row["mean_reward"] - previous["mean_reward"] if previous else float("nan")
                row["closure_error"] = row["mean_reward"] - sum(row[k] for k in PARTS)
                row["logged_reward_error"] = row["mean_reward"] - float(np.mean([w["mean_reward"] for w in windows]))
                if max(abs(row["closure_error"]), abs(row["logged_reward_error"])) > TOL:
                    raise ValueError(f"training mean does not match bound episode: {identifiers}")
                output.append(row)
                cell.append(row)
                window_rows.extend({**identifiers, **w} for w in windows)
                previous = row
            reward = np.array([r["mean_reward"] for r in cell])
            variance = float(np.var(reward, ddof=1)) if len(cell) > 1 else 0.
            for metric in PARTS:
                values = np.array([r[metric] for r in cell])
                cov = float(np.cov(values, reward, ddof=1)[0, 1]) if len(cell) > 1 else 0.
                contributions.append({"base_seed": seed, "branch": branch, "component": metric,
                                      "signed_mean": float(values.mean()), "mean_absolute_update_contribution": float(np.abs(values).mean()),
                                      "sum_absolute_update_contribution": float(np.abs(values).sum()),
                                      "covariance_with_total": cov, "total_reward_variance": variance,
                                      "covariance_fraction": cov / variance if variance > 0 else float("nan"),
                                      "interpretation": "descriptive; not causal"})
            base._progress(config, f"v2 reward binding/closure checked: seed={seed} {branch}")
    return output, window_rows, contributions


def bootstrap_indices(config):
    return np.random.default_rng(config.bootstrap_seed).integers(0, len(config.seeds), (config.bootstrap_samples, len(config.seeds)))


def curve_stats(rows, branch, metric, config, indices):
    index = {(int(r["base_seed"]), int(r["update"])): r for r in rows if r["branch"] == branch}
    matrix = np.array([[number(index[(s, u)][metric]) for u in range(1, config.expected_training_updates + 1)] for s in config.seeds])
    # One seed draw for the whole curve, reused for every branch and metric.
    boot = matrix[indices].mean(axis=1)
    return matrix.mean(axis=0), np.quantile(boot, .025, axis=0), np.quantile(boot, .975, axis=0), matrix.std(axis=0, ddof=1)


def quality_episode(row, evidence, window_count):
    result = dict(row)
    accepted = int(evidence["accepted_update_count"])
    if accepted != int(float(row["accepted_update_count"])):
        raise ValueError("accepted update counts disagree")
    if sum(int(evidence[k + "_update_count"]) for k in ("beneficial", "neutral", "harmful")) != accepted:
        raise ValueError("update classification counts do not close")
    for kind in ("beneficial", "neutral", "harmful"):
        count = int(evidence[kind + "_update_count"])
        result[kind + "_fraction"] = count / accepted if accepted else float("nan")
        result[kind + "_per_window"] = count / window_count
    for h in HORIZONS:
        for suffix in ("benefit_retained_fraction", "beneficial_observed_count", "censored_event_count"):
            key = f"h{h}_{suffix}"
            result[key] = optional(evidence.get(key))
    result["decision_windows"] = window_count
    result["measurement_load_bytes"] = number(row["measurement_overhead_packets"]) * 512
    return result


def validation_evidence(config, derived):
    episodes = base._read_csv(derived / "episode_metrics.csv")
    persist = base._read_csv(derived / "ekf_update_persistence_audit/episode_persistence_summary.csv")
    pindex = {(int(r["base_seed"]), r["branch"], r["episode_id"]): r for r in persist}
    fixed, missing = [], []
    for row in episodes:
        seed, branch = int(row["base_seed"]), row["branch"]
        key = seed, branch, row["episode_id"]
        rewards = base._read_csv(base._scale_dir(config.reference_root, seed, branch) / "validation" / row["episode_id"] / "rewards.csv")
        windows = len({number(r["timestamp"]) for r in rewards})
        if key not in pindex or not windows:
            raise ValueError(f"missing persistence evidence: {key}")
        fixed.append(quality_episode(row, pindex[key], windows))
    costs = base._read_csv(derived / "communication_cost_by_seed.csv")
    seeds = base._seed_rows(fixed, costs, config)
    extra = ["neutral_per_window", "measurement_load_bytes"] + [f"h{h}_{suffix}" for h in HORIZONS for suffix in ("benefit_retained_fraction", "beneficial_observed_count", "censored_event_count")]
    for row in seeds:
        group = [r for r in fixed if int(r["base_seed"]) == row["base_seed"] and r["branch"] == row["branch"]]
        for metric in extra:
            values = [optional(r.get(metric)) for r in group]
            valid = [v for v in values if math.isfinite(v)]
            row[metric] = float(np.mean(valid)) if valid else float("nan")
            row[metric + "_valid_episode_count"] = len(valid)
        for h in HORIZONS:
            if not math.isfinite(row[f"h{h}_benefit_retained_fraction"]):
                missing.append({"base_seed": row["base_seed"], "branch": row["branch"], "horizon": h, "reason": "no observable immediately beneficial update"})
    return fixed, seeds, missing


def panel_style(ax, title, ylabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return float("nan"), float("nan"), float("nan")
    boot = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def paired_statistics(rows, config, indices):
    """Use exactly the same paired draws as the displayed confidence intervals."""
    output = base._paired_contrasts(rows, config)
    lookup = {(int(r["base_seed"]), r["branch"]): r for r in rows}
    for row in output:
        diffs = [optional(lookup[s, row["treatment"]].get(row["metric"])) -
                 optional(lookup[s, row["control"]].get(row["metric"])) for s in config.seeds]
        mean, lo, hi = interval(diffs, indices)
        row.update(mean_difference=mean, ci95_lower=lo, ci95_upper=hi)
    return output


def zoom_limits(arrays):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if not values.size or np.all(values == 0):
        return None
    # Common 10--90% central region highlights near-zero variation; full spikes remain in parent.
    lo, hi = np.quantile(values, [.1, .9])
    lo, hi = min(lo, 0.), max(hi, 0.)
    if hi - lo < 1e-12:
        lo, hi = min(values.min(), 0.), max(values.max(), 0.)
    margin = max((hi - lo) * .1, 1e-12)
    return float(lo - margin), float(hi + margin)


def make_figures(config, plots, training, seeds, indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    paths, aggregate_rows, zoom_rows = {}, [], []
    cache = {}
    for branch in config.branches:
        for metric in (*base.TRAINING_METRICS, *PARTS):
            stats = curve_stats(training, branch, metric, config, indices)
            cache[branch, metric] = stats
            for u, (mean, lo, hi, std) in enumerate(zip(*stats), 1):
                aggregate_rows.append({"branch": branch, "update": u, "metric": metric, "seed_count": len(config.seeds), "mean": mean, "std": std, "ci95_lower": lo, "ci95_upper": hi})
    handles = [Line2D([0], [0], color=base.BRANCH_COLORS[b], linestyle=STYLES[i], label=base.BRANCH_LABELS[b]) for i, b in enumerate(config.branches)]
    x = np.arange(1, config.expected_training_updates + 1)

    def save(fig, name, split, legend=True):
        if legend:
            fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1), frameon=False, fontsize=9)
        fig.suptitle(f"3 UAV | {split} | {len(config.seeds)} training seeds", y=.947, fontsize=12)
        fig.subplots_adjust(top=.86, bottom=.1, hspace=.48, wspace=.34)
        fig.savefig(plots / (name + ".png"), dpi=300, facecolor="white")
        fig.savefig(plots / (name + ".pdf"), facecolor="white")
        plt.close(fig)
        paths[name] = str((plots / (name + ".png")).resolve())

    def curves(ax, metric, title, ylabel="Reward contribution"):
        for i, branch in enumerate(config.branches):
            mean, lo, hi, _ = cache[branch, metric]
            ax.plot(x, mean, color=base.BRANCH_COLORS[branch], linestyle=STYLES[i], linewidth=1.35)
            ax.fill_between(x, lo, hi, color=base.BRANCH_COLORS[branch], alpha=.10)
        panel_style(ax, title, ylabel)
        ax.set_xlabel("Update")
        ax.set_xlim(1, config.expected_training_updates)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    curves(axes[0], "mean_reward", "Mean team reward", "Reward / decision window")
    curves(axes[1], "loss", "MAPPO total loss", "Loss")
    save(fig, "training_reward_and_mappo_loss", "train; pointwise bootstrap 95% CI")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, metric, title in zip(axes.flat, base.TRAINING_METRICS[2:], ("Actor loss", "Critic loss", "Entropy", "Clip fraction")):
        curves(ax, metric, title, "Fraction" if metric == "clip_fraction" else title)
    save(fig, "training_loss_components", "train; pointwise bootstrap 95% CI")

    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.2, 1, 1))
    total = fig.add_subplot(grid[:, 0])
    curves(total, "mean_reward", "Mean team reward", "Reward / decision window")
    total.axhline(0, color="black", lw=.7)
    titles = (r"Localization contribution: $-J_{\mathrm{loc}}$", r"Communication contribution: $-J_{\mathrm{com}}$", "Collision contribution", "Energy contribution")
    for k, (metric, title) in enumerate(zip(PARTS, titles)):
        ax = fig.add_subplot(grid[k // 2, 1 + k % 2])
        curves(ax, metric, title)
        ax.axhline(0, color="black", lw=.7)
        if k >= 2:
            limits = zoom_limits([a for b in config.branches for a in cache[b, metric][:3]])
            if limits is None:
                ax.text(.5, .5, "All observed values are zero", ha="center", transform=ax.transAxes)
            else:
                # Reserve a positive, data-free region for the inset: no parent
                # spike or uncertainty band is hidden by its opaque background.
                low = min(float(a.min()) for b in config.branches for a in cache[b, metric][:3])
                span = max(abs(low), 1e-9)
                ax.set_ylim(low - span * .05, span * 1.15)
                inset = ax.inset_axes([.13, .60, .72, .32])
                curves(inset, metric, "")
                inset.set(xlabel="", ylabel="", ylim=limits)
                inset.tick_params(labelsize=7)
                inset.set_facecolor("white")
                inset.set_title("Near-zero detail (same data)", fontsize=7, pad=2)
                inset.axhline(0, color="black", lw=.5)
                ax.indicate_inset_zoom(inset, edgecolor="gray", alpha=.6)
                zoom_rows.append({"component": metric, "x_min": 1, "x_max": config.expected_training_updates, "y_min": limits[0], "y_max": limits[1], "rule": "joint branches/CI central 10-90% and zero; 10% padding; full parent retained"})
    save(fig, "training_reward_decomposition", "train; signed contributions; pointwise 95% CI")

    seed_index = {(int(r["base_seed"]), r["branch"]): r for r in seeds}

    def values(metric, branch):
        return [optional(seed_index[s, branch].get(metric)) for s in config.seeds]

    def metric_panel(ax, metric, title, unit):
        for i, branch in enumerate(config.branches):
            vals = np.array(values(metric, branch))
            offset = np.linspace(-.12, .12, len(vals))
            ax.scatter(i + offset, vals, color=base.BRANCH_COLORS[branch], marker=MARKERS[i], s=28, alpha=.8)
            mean, lo, hi = interval(vals, indices)
            if math.isfinite(mean):
                ax.plot([i, i], [lo, hi], color="black", lw=1.4)
                ax.scatter([i], [mean], color="black", marker="_", s=140, zorder=5)
            else:
                ax.text(i, .04, f"NA ({np.isfinite(vals).sum()}/{len(vals)} seeds)", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        short_labels = {base.BASELINE_BRANCH: "Baseline", base.PROXY_BRANCH: "Ablation", base.PROPOSED_BRANCH: "Proposed"}
        ax.set_xticks(range(len(config.branches)), [short_labels[b] for b in config.branches])
        panel_style(ax, title, unit)
        if metric == "update_coverage_any":
            ax.set_ylim(-.03, 1.05)
            ax.set_yticks([0, .25, .5, .75, 1.])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, spec in zip(axes, (("position_rmse_norm_m", "Euclidean position RMSE", "m"), ("position_p95_norm_m", "Euclidean error P95", "m"), ("update_coverage_any", "Episodes with accepted update", "Episode fraction"))):
        metric_panel(ax, *spec)
    save(fig, "localization_performance", "validation; seed points + mean / 95% CI")
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    specs = (("measurement_count", "Selected measurements", "Count / episode"), ("measurement_load_bytes", "Applied measurement payload", "Bytes / episode"), ("expected_communication_cost", "Expected proxy cost", "Normalized cost / UAV-window"), ("actual_communication_cost", "Selected-action proxy cost", "Normalized cost / UAV-window"), ("j_com", r"Communication objective $J_{\mathrm{com}}$", "Objective value"), ("packet_loss", "Packet loss", "Fraction"), ("delay_s", "Delay", "s"), ("aoi_s", "Age of information", "s"))
    for ax, spec in zip(axes.flat, specs):
        metric_panel(ax, *spec)
    save(fig, "communication_cost", "validation; seed points + mean / 95% CI")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    specs = [(k + "_fraction", k.title() + " accepted updates", "Fraction among accepted") for k in ("beneficial", "neutral", "harmful")]
    specs += [(k + "_per_window", k.title() + " updates / window", "Updates / team window") for k in ("beneficial", "neutral", "harmful")]
    specs += [("accepted_updates_per_window", "All accepted updates / window", "Updates / team window"), ("j_loc", r"Covariance objective $J_{\mathrm{loc}}$", "Objective value")]
    for ax, spec in zip(axes.flat, specs):
        metric_panel(ax, *spec)
    ax = axes.flat[8]
    for i, branch in enumerate(config.branches):
        stats = [interval(values(f"h{h}_benefit_retained_fraction", branch), indices) for h in HORIZONS]
        mean, lo, hi = np.array(stats).T
        ax.plot(HORIZONS, mean, marker=MARKERS[i], linestyle=STYLES[i], color=base.BRANCH_COLORS[branch])
        ax.fill_between(HORIZONS, lo, hi, color=base.BRANCH_COLORS[branch], alpha=.12)
        if not np.isfinite(mean).all():
            ax.text(.02, .03 + i * .07, f"{base.BRANCH_LABELS[branch]}: missing horizons", transform=ax.transAxes, fontsize=7)
    panel_style(ax, "Retained benefit after update", "Fraction of observed beneficial events")
    ax.set_xlabel("Future decision windows")
    ax.set_xticks(HORIZONS)
    save(fig, "ekf_update_quality", "validation; episode then seed averages; NA excluded within seed")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for s in config.seeds:
        b, p = seed_index[s, base.BASELINE_BRANCH], seed_index[s, base.PROPOSED_BRANCH]
        ax.annotate("", xy=(number(p["actual_communication_cost"]), number(p["position_rmse_norm_m"])), xytext=(number(b["actual_communication_cost"]), number(b["position_rmse_norm_m"])), arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.1})
        for branch, row in ((base.BASELINE_BRANCH, b), (base.PROPOSED_BRANCH, p)):
            ax.scatter(number(row["actual_communication_cost"]), number(row["position_rmse_norm_m"]), color=base.BRANCH_COLORS[branch], s=45)
            ax.annotate(str(s), (number(row["actual_communication_cost"]), number(row["position_rmse_norm_m"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    panel_style(ax, "Paired baseline to proposed changes", "Position RMSE (m)")
    ax.set_xlabel("Actual selected-measurement proxy cost / UAV-window (not bytes)")
    ax.legend(handles=[handles[0], handles[-1]], fontsize=8)
    save(fig, "communication_localization_tradeoff", "validation; one pair per training seed", legend=False)

    paired = []
    specs = (("position_rmse_norm_m", "Position RMSE difference", "m"), ("position_p95_norm_m", "P95 difference", "m"), ("actual_communication_cost", "Actual proxy cost difference", "Normalized cost"), ("j_com", r"$J_{\mathrm{com}}$ difference", "Objective value"), ("harmful_fraction", "Harmful fraction difference", "Fraction"), ("mean_reward", "Team reward difference", "Reward"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (metric, title, unit) in zip(axes.flat, specs):
        diffs = np.array(values(metric, base.PROPOSED_BRANCH)) - np.array(values(metric, base.BASELINE_BRANCH))
        mean, lo, hi = interval(diffs, indices)
        ax.scatter(diffs, np.arange(len(config.seeds)), color=base.BRANCH_COLORS[base.PROPOSED_BRANCH], s=32)
        ax.plot([lo, hi], [len(config.seeds)] * 2, color="black", lw=2)
        ax.scatter([mean], [len(config.seeds)], color="black", marker="D", s=35)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks(range(len(config.seeds) + 1), [str(s) for s in config.seeds] + ["Mean / CI"])
        ax.set_xlabel("Proposed minus baseline (" + unit + ")")
        panel_style(ax, title, "Training seed")
        for s, diff in zip(config.seeds, diffs):
            paired.append({"metric": metric, "base_seed": s, "difference": diff, "mean_difference": mean, "ci95_lower": lo, "ci95_upper": hi, "paired_seed_count": int(np.isfinite(diffs).sum())})
    save(fig, "seed_level_paired_contrasts", "validation; paired seed bootstrap 95% CI", legend=False)
    return paths, aggregate_rows, zoom_rows, paired


def run_mainline_figures_v2(config):
    reference = Path(config.reference_root).resolve()
    name = reference.name
    # Resolve old CLI defaults to a fresh experiment-specific destination.
    old = base.FinalMainlineAuditConfig(reference_root=reference)
    logs = Path(config.output_dir)
    if logs == old.output_dir:
        logs = Path("logs/preliminary") / name / "mainline_figures_v2"
    plots = Path(config.plots_dir) if config.plots_dir else Path("plots/preliminary") / name / "mainline_figures_v2"
    if plots.name == "final_mainline_figures":
        plots = Path("plots/preliminary") / name / "mainline_figures_v2"
    logs, plots = logs.resolve(), plots.resolve()
    source_logs = (Path("logs/preliminary") / name).resolve()
    for path in (logs, plots):
        if path.is_relative_to(reference) or path == source_logs or path in reference.parents or path in source_logs.parents:
            raise ValueError("output overlaps protected source/root")
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"v2 preserves existing results even with --force; select a fresh directory: {path}")
    if logs == plots or logs.is_relative_to(plots) or plots.is_relative_to(logs):
        raise ValueError("logs and plots must be separate directories")
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    config = replace(config, reference_root=reference, output_dir=logs, plots_dir=plots, force=False)
    with SourceGuard((reference, source_logs), (logs, plots)) as guard:
        # Fingerprint checkpoints without reading any episode split or test diagnostic.
        checkpoints = []
        for root, dirs, files in os.walk(reference):
            dirs[:] = [d for d in dirs if d not in {"train", "validation", "test", "warmup", "warm-up", "warm_up", "diagnostics"}]
            for filename in files:
                if filename.endswith((".pt", ".pth", ".ckpt")):
                    p = Path(root) / filename
                    digest(p)
                    checkpoints.append(str(p))
        training, windows, contributions = training_evidence(config)
        write_table(logs / "training_reward_decomposition.csv", training)
        write_table(logs / "reward_window_closure.csv", windows)
        write_table(logs / "reward_contribution_summary.csv", contributions)
        derived = logs / "source_audit"
        legacy = base.run_final_mainline_audit(replace(config, figure_profile="legacy", output_dir=derived, plots_dir=plots / "diagnostics_legacy"))
        episodes, seeds, missing = validation_evidence(config, derived)
        write_table(logs / "validation_episode_metrics.csv", episodes)
        write_table(logs / "seed_level_metrics.csv", seeds)
        indices = bootstrap_indices(config)
        write_table(logs / "paired_contrasts.csv", paired_statistics(seeds, config, indices))
        paths, aggregates, zooms, paired = make_figures(config, plots, training, seeds, indices)
        write_table(logs / "training_aggregates.csv", aggregates)
        write_table(logs / "seed_paired_differences.csv", paired)
        if zooms:
            write_table(logs / "zoom_ranges.csv", zooms)
        if missing:
            write_table(logs / "missing_persistence.csv", missing)
        hash_rows = guard.verify()
        write_table(logs / "source_hashes.csv", hash_rows)
        unchanged = all(row["unchanged"] for row in hash_rows)
        checks = {"source_hashes_unchanged": unchanged, "test_read_attempts": guard.test_attempts,
                  "training_reward_closure": all(abs(r["closure_error"]) <= TOL for r in windows + training),
                  "training_seed_branch_cells": len(config.seeds) * len(config.branches),
                  "training_updates": len(training), "validation_episodes": len(episodes),
                  "persistence_all_seed_branch_horizons_available": not missing,
                  "existing_structural_audit_ok": bool(legacy["audit_ok"]),
                  "metric_consistency_ok": bool(legacy["checks"]["metric_consistency_audit_ok"]),
                  "persistence_audit_ok": bool(legacy["checks"]["ekf_persistence_audit_ok"])}
        ok = unchanged and not guard.test_attempts and not missing and all(checks[k] for k in ("training_reward_closure", "existing_structural_audit_ok", "metric_consistency_ok", "persistence_audit_ok"))
        report = {"figure_profile": "mainline-v2", "audit_ok": bool(ok), "reference_root": str(reference),
                  "seed_set": list(config.seeds), "branches": list(config.branches), "checks": checks,
                  "source_read_count": guard.reads, "fingerprinted_files": len(hash_rows), "checkpoint_paths": checkpoints,
                  "test_data_read": False, "source_script": str(Path(__file__).resolve()),
                  "bootstrap_samples": config.bootstrap_samples, "bootstrap_seed": config.bootstrap_seed,
                  "bootstrap_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                  "statistics": "equal seed weights; episode then seed reduction; complete-seed curve resampling; pointwise intervals",
                  "missing_persistence": missing, "existing_decision": legacy["decision"],
                  "existing_primary_comparison": legacy["primary_comparison"],
                  "decision_supported": bool(ok and legacy["decision"]["decision_supported"]),
                  "thresholds_changed": False, "figures": paths,
                  "source_audit": str(derived / "audit_report.json"),
                  "limitations": ["five development seeds, not universal or independent final-test evidence", "confidence interval crossing zero does not establish equivalence", "reward covariance attribution is descriptive, not causal", "J_loc is covariance uncertainty, not RMSE", "future observed error retention is observational, not a no-update counterfactual"],
                  "regeneration": f'python main.py run-final-mainline-audit --reference "{reference}" --output "{logs}" --plots-output "{plots}" --figure-profile mainline-v2 --bootstrap-samples {config.bootstrap_samples} --bootstrap-seed {config.bootstrap_seed}'}
        (logs / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        captions = "# 3-UAV 五种子主线图注与口径\n\n来源：" + str(reference) + "\n\n"
        captions += f"训练种子：{list(config.seeds)}。三个分支使用同一冻结实验。训练图横轴 Update，按日志 episode_id 对齐；每窗口先对无人机取平均，再对窗口取平均。区间为完整 seed 曲线重采样所得逐点 95% Bootstrap 区间（{config.bootstrap_samples} 次，随机种子 {config.bootstrap_seed}），不是同时置信带。未平滑。validation 先按 episode 汇总，再按 seed 等权汇总；test 不读取。\n\n"
        descriptions = [
            "训练奖励和 MAPPO total loss。奖励为每决策窗口团队平均，loss 沿用实际训练日志；总损失及其跨分支高低仅作训练诊断。",
            "Actor/Critic loss、Entropy 和 Clip fraction。阴影口径与奖励图一致，各项采用独立纵轴。",
            "带符号的四项贡献和团队奖励。定位项由冻结权重乘归一化不确定度计算；通信项使用冻结的丢包、AoI、时延与连通性权重。主图保留完整区间；碰撞/能耗插图由全部分支及置信区间的共同 10%-90% 中央范围与零线确定，加 10% 边距。尖峰仍在主图中。闭合误差容差 1e-6。量级与协方差贡献表仅作描述，不能推断因果。",
            "validation 三维位置误差向量范数 RMSE、P95 和有更新 episode 比例。每个彩色点为一个 seed，黑色横线为均值，竖线为 seed Bootstrap 区间。覆盖率不等同更新质量。",
            "测量次数及实际负载以 episode 为分母；负载字节数为已应用等效包数乘 512。期望和实际代理代价为每 UAV 决策窗口平均，丢包、时延与 AoI 沿用冻结 validation 指标的候选链路统计口径。代理代价不等于真实传输字节或 J_com。",
            "更新分类沿用固定 0.001 m 误差变化阈值。比例的分母为已接受更新，无更新 episode 比例记 NA；每窗口数量的分母为团队窗口，无更新仍为零。收益保持率以可观察到对应未来窗口的即时有益更新为分母，先 episode 再 seed 平均；删失与有效样本数见 CSV。它是沿实际轨迹的误差保持诊断，不是因果反事实收益。J_loc 为协方差目标。",
            "每个箭头连接相同训练 seed 的基线和主要方法。横轴为实际选中测量的归一化代理代价/无人机窗口，纵轴为位置 RMSE。不能由五个点拟合可靠 Pareto 前沿。",
            "主要方法减基线。五个差值和其均值 Bootstrap 区间同时展示；负值对误差和代价有利，正值对奖励有利。均值区间跨零不能证明相等或非劣效；既有门槛见 source_audit 报告。"]
        for (name, path), description in zip(paths.items(), descriptions):
            captions += f"## {name}\n\n{description}\n\n图片：{path}\n\n"
        captions += "统计和图源：training_aggregates.csv、training_reward_decomposition.csv、reward_window_closure.csv、reward_contribution_summary.csv、validation_episode_metrics.csv、seed_level_metrics.csv、seed_paired_differences.csv、source_hashes.csv。源代码采用本地 standalone figure contract；无独立 manuscript include/claim gate 文件。诊断旧样式产物在 diagnostics_legacy 子目录，八张顶层图片为本次主图候选；是否写入论文由证据判定决定。\n"
        (logs / "figure_captions_zh.md").write_text(captions, encoding="utf-8")
        write_figure_index(logs, paths)
        base._progress(config, f"v2 complete: eight figures; audit_ok={ok}; decision_supported={report['decision_supported']}")
        return report


def write_figure_index(logs, paths):
    titles = ("训练奖励与 MAPPO 总损失", "MAPPO 损失分量", "平均团队奖励与四项奖励贡献",
              "定位性能", "通信代价", "EKF 更新质量", "通信—定位权衡", "逐种子配对差异")
    lines = ["# 3-UAV 五种子绘图索引", "", "最新 VO 校准组；主线三分支。红色为主要方法，蓝色为基线，绿色为消融。", ""]
    for title, (name, path) in zip(titles, paths.items()):
        png = Path(path).resolve().as_posix()
        pdf = Path(path).with_suffix(".pdf").resolve().as_posix()
        lines += [f"## {title}", "", f"[PNG](<{png}>) · [矢量 PDF](<{pdf}>)", "", f"![{title}](<{png}>)", ""]
    for filename, label in (("figure_captions_zh.md", "中文图注与统计口径"), ("audit_report.json", "绘图审计报告")):
        lines += [f"[{label}](<{(logs / filename).resolve().as_posix()}>)", ""]
    (logs / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")
