"""Five descriptive seed42 figures; train and validation are separate surfaces."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .mainline_baseline_comparison import (
    ACTIVE_LEARNED_METHODS,
    ACTIVE_MAINLINE_METHODS,
    COLORS,
    FIXED_CONTROL_METHODS,
    PAPER_BRANCH_LABELS,
    PROPOSED_METHOD,
    _write_table,
)
from .baseline import _read_csv, _read_json
from .mainline_figures_v2 import SourceGuard, reward_windows
from .termination import classify_episode_outcome


def generate_safety_collision_attribution(
    paths,
    validation_rows,
    logs,
    plots,
    *,
    arm="reference",
    figure_profile="compact-proposed-v4",
    overwrite=False,
):
    """Create a split-scoped safety attribution diagnostic.

    This is an internal diagnostic, not a fifth performance claim.  It keeps
    communication-only occluders separate from physical collision sources and
    reads validation action traces only; sealed test traces are rejected by
    ``SourceGuard``.  Missing action traces remain ``NA`` in the summary.
    """

    import json as _json
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    logs, plots = Path(logs), Path(plots)
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    output_png = plots / "safety_collision_attribution.png"
    output_pdf = plots / "safety_collision_attribution.pdf"
    if (output_png.exists() or output_pdf.exists()) and not overwrite:
        raise FileExistsError("safety attribution output already exists")

    methods = tuple(
        method
        for method in (*FIXED_CONTROL_METHODS, PROPOSED_METHOD)
        if method in paths
    )
    if not methods:
        raise ValueError("no methods available for safety attribution")

    def number(value, default=float("nan")):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if np.isfinite(result) else float(default)

    def outcome(row):
        value = str(row.get("outcome_class", "")).strip()
        if value in {"collision", "early_termination", "time_limit", "unknown_short", "normal"}:
            return value
        return classify_episode_outcome(
            row,
            max_steps=int(number(row.get("max_steps", 50), 50)),
        )

    # Validation-level counts and clearances are kept at episode granularity.
    validation = [row for row in validation_rows if row.get("method") in methods]
    categories = ("collision", "early_termination", "time_limit", "unknown_short", "normal")
    category_colors = {
        "collision": "#b91c1c",
        "early_termination": "#d97706",
        "time_limit": "#9ca3af",
        "unknown_short": "#7c3aed",
        "normal": "#2563eb",
    }
    summary = []
    counts = {method: {category: 0 for category in categories} for method in methods}
    clearances = {method: [] for method in methods}
    for row in validation:
        method = str(row["method"])
        classified = outcome(row)
        counts[method][classified] = counts[method].get(classified, 0) + 1
        clearance = number(row.get("minimum_clearance"))
        if np.isfinite(clearance):
            clearances[method].append(clearance)

    # Requested/executed changes are calculated from the action trace, rather
    # than inferred from a short episode or from safety_projection_count.
    action_stats = {
        method: {
            "action_rows": 0,
            "changed_rows": 0,
            "projected_rows": 0,
            "episodes_with_action_trace": 0,
            "missing_action_trace_episodes": 0,
        }
        for method in methods
    }
    source_roots = [Path(paths[method]) for method in methods]
    with SourceGuard(source_roots, [logs, plots]) as guard:
        for row in validation:
            method = str(row["method"])
            source_scale = Path(str(row.get("source_scale", "")))
            split = str(row.get("split", "validation"))
            if split != "validation" or "test" in source_scale.parts:
                raise ValueError("safety attribution may only read validation actions")
            action_path = source_scale / split / str(row["episode_id"]) / "actions.csv"
            if not action_path.is_file():
                action_stats[method]["missing_action_trace_episodes"] += 1
                continue
            action_stats[method]["episodes_with_action_trace"] += 1
            action_rows = _read_csv(action_path)
            for action in action_rows:
                stats = action_stats[method]
                stats["action_rows"] += 1
                projected = str(action.get("safety_projected", "0")).strip().lower() in {
                    "1", "true", "yes"
                }
                clipped = str(action.get("action_clipped", "0")).strip().lower() in {
                    "1", "true", "yes"
                }
                boundary = str(action.get("boundary_projected", "0")).strip().lower() in {
                    "1", "true", "yes"
                }
                stats["projected_rows"] += int(projected)
                raw = str(action.get("action_raw", "")).strip()
                applied = str(action.get("action_applied", "")).strip()
                changed = projected or clipped or boundary
                if raw and applied:
                    try:
                        raw_action = _json.loads(raw)
                        applied_action = _json.loads(applied)
                        if raw_action.get("measurement_uav_id") != applied_action.get(
                            "measurement_uav_id"
                        ):
                            changed = True
                        raw_delta = np.asarray(raw_action.get("delta_v", []), dtype=float)
                        applied_delta = np.asarray(applied_action.get("delta_v", []), dtype=float)
                        if raw_delta.shape != applied_delta.shape or not np.allclose(
                            raw_delta, applied_delta, atol=1e-6, rtol=1e-6
                        ):
                            changed = True
                        raw_bandwidth = {
                            str(key): float(value)
                            for key, value in dict(raw_action.get("bandwidth", {})).items()
                        }
                        applied_bandwidth = {
                            str(key): float(value)
                            for key, value in dict(applied_action.get("bandwidth", {})).items()
                        }
                        if set(raw_bandwidth) != set(applied_bandwidth) or any(
                            not np.isclose(
                                raw_bandwidth[key], applied_bandwidth[key], atol=1e-6, rtol=1e-6
                            )
                            for key in raw_bandwidth
                        ):
                            changed = True
                    except (TypeError, ValueError, _json.JSONDecodeError):
                        # The explicit projection flags remain the reliable
                        # fallback when a legacy action trace is not JSON.
                        pass
                stats["changed_rows"] += int(changed)
        unchanged = guard.verify()
    if guard.test_attempts or not all(item["unchanged"] for item in unchanged):
        raise ValueError("safety attribution source access audit failed")

    for method in methods:
        stats = action_stats[method]
        action_rows = stats["action_rows"]
        summary.append(
            {
                "method": method,
                "validation_episode_count": sum(counts[method].values()),
                **{f"{category}_episode_count": counts[method][category] for category in categories},
                "minimum_clearance_mean_m": (
                    float(np.mean(clearances[method])) if clearances[method] else "NA"
                ),
                "minimum_clearance_min_m": (
                    float(np.min(clearances[method])) if clearances[method] else "NA"
                ),
                "action_rows": action_rows,
                "requested_executed_changed_rows": stats["changed_rows"] if action_rows else "NA",
                "requested_executed_change_rate": (
                    float(stats["changed_rows"] / action_rows) if action_rows else "NA"
                ),
                "safety_projected_rows": stats["projected_rows"] if action_rows else "NA",
                "safety_projection_rate": (
                    float(stats["projected_rows"] / action_rows) if action_rows else "NA"
                ),
                "episodes_with_action_trace": stats["episodes_with_action_trace"],
                "missing_action_trace_episodes": stats["missing_action_trace_episodes"],
                "source_split": "validation",
                "communication_occluder_is_physical": False,
            }
        )
    _write_table(logs / "safety_collision_attribution.csv", summary)

    short_labels = {
        "local_only_fixed": "Local",
        "random_cooperative": "Random",
        "budgeted_greedy_ig": "Greedy IG",
        PROPOSED_METHOD: "Proposed",
    }
    labels = [short_labels.get(method, method) for method in methods]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = np.arange(len(methods), dtype=float)

    bottom = np.zeros(len(methods), dtype=float)
    for category in categories:
        values = np.asarray([counts[method][category] for method in methods], dtype=float)
        axes[0, 0].bar(
            x,
            values,
            bottom=bottom,
            color=category_colors[category],
            alpha=0.82,
            label=category.replace("_", " "),
        )
        bottom += values
    axes[0, 0].set_title("Validation termination attribution")
    axes[0, 0].set_ylabel("Episode count")
    axes[0, 0].set_xticks(x, labels, rotation=20)
    axes[0, 0].legend(fontsize=7, frameon=False)

    for index, method in enumerate(methods):
        values = np.asarray(clearances[method], dtype=float)
        if values.size:
            jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
            axes[0, 1].scatter(
                np.full(values.size, x[index]) + jitter,
                values,
                color=COLORS[method],
                alpha=0.65,
                s=16,
            )
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            axes[0, 1].errorbar(
                x[index], mean, yerr=std, fmt="_", color="black", capsize=4, markersize=14
            )
    axes[0, 1].set_title("Validation minimum clearance")
    axes[0, 1].set_ylabel("Minimum clearance (m)")
    axes[0, 1].set_xticks(x, labels, rotation=20)

    projection = np.asarray(
        [number(row["safety_projection_rate"]) for row in summary], dtype=float
    )
    changed = np.asarray(
        [number(row["requested_executed_change_rate"]) for row in summary], dtype=float
    )
    width = 0.36
    axes[1, 0].bar(x - width / 2, projection, width, label="Safety projection rate", color="#7c3aed")
    axes[1, 0].bar(x + width / 2, changed, width, label="Requested/executed change rate", color="#0891b2")
    axes[1, 0].set_title("Execution intervention diagnostic")
    axes[1, 0].set_ylabel("Fraction of validation action rows")
    axes[1, 0].set_xticks(x, labels, rotation=20)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].legend(fontsize=7, frameon=False)

    training_path = logs / "training_termination_attribution.csv"
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    training_rows = _read_csv(training_path)
    training_methods = [method for method in methods if any(row.get("method") == method for row in training_rows)]
    marker_handles = []
    for method in training_methods:
        rows = sorted(
            [row for row in training_rows if row.get("method") == method],
            key=lambda row: number(row.get("training_episode"), 0),
        )
        tx = np.asarray([number(row.get("training_episode"), np.nan) for row in rows])
        ty = np.asarray([number(row.get("window_count"), np.nan) for row in rows])
        valid = np.isfinite(tx) & np.isfinite(ty)
        axes[1, 1].plot(
            tx[valid], ty[valid], color=COLORS[method], label=short_labels.get(method, method), linewidth=1.0
        )
        for row, episode, length in zip(rows, tx, ty):
            if not (np.isfinite(episode) and np.isfinite(length)):
                continue
            if outcome(row) == "collision":
                axes[1, 1].scatter(episode, length, marker="X", facecolors="none", edgecolors="#b91c1c", s=46)
            elif outcome(row) == "early_termination":
                axes[1, 1].scatter(episode, length, marker="^", facecolors="none", edgecolors="#d97706", s=46)
    axes[1, 1].set_title("Training episode length and termination")
    axes[1, 1].set_xlabel("Training episode")
    axes[1, 1].set_ylabel("Window count")
    axes[1, 1].legend(fontsize=7, frameon=False)
    marker_handles = [
        Line2D([], [], linestyle="None", marker="X", markeredgecolor="#b91c1c", markerfacecolor="none", label="Collision"),
        Line2D([], [], linestyle="None", marker="^", markeredgecolor="#d97706", markerfacecolor="none", label="Non-collision early termination"),
    ]
    axes[1, 1].legend(handles=axes[1, 1].get_legend_handles_labels()[0] + marker_handles, frameon=False, fontsize=7)
    for ax in axes.reshape(-1):
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"3 UAV | seed=42 | {arm} | safety collision attribution", fontsize=11)
    for extension, destination in (("png", output_png), ("pdf", output_pdf)):
        fig.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    (logs / "safety_collision_attribution_caption.md").write_text(
        "安全归因诊断：validation按episode统计终止类别；最小安全距离为每回合记录值；"
        "请求/执行动作变化率和安全投影率以validation action row为分母。通信遮挡区域"
        "保持LOS/NLOS影响但不进入实体碰撞集合；训练面板保留原始回合长度和原因标记。"
        "缺失action trace记为NA，不补零；该图不用于单独证明算法性能。\n",
        encoding="utf-8",
    )
    return [str(output_png.resolve()), str(output_pdf.resolve())]


def generate_compact_figures(
    paths,
    validation_rows,
    logs,
    plots,
    *,
    arm,
    figure_profile="compact-mainline",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    logs, plots = Path(logs), Path(plots)
    for destination in (logs, plots):
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"figure output must be empty: {destination}")
    logs.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    # Only the current paper scope is plotted.  ``paths`` may still contain
    # deprecated historical branches when plot-only is run on an old archive;
    # filtering here keeps the archive readable without presenting that branch
    # as a current contribution.
    if figure_profile == "legacy":
        figure_profile = "compact-mainline"
    if figure_profile not in {
        "compact-mainline",
        "compact-proposed-v3",
        "compact-proposed-v4",
    }:
        raise ValueError(f"unknown compact figure profile: {figure_profile}")
    proposed_only = figure_profile in {"compact-proposed-v3", "compact-proposed-v4"}
    if proposed_only:
        # The v3 paper surface deliberately separates training diagnostics
        # from baseline validation comparisons.  Historical MLP artifacts are
        # not loaded even if a caller accidentally supplies their path.
        figure_methods = tuple(
            method
            for method in (*FIXED_CONTROL_METHODS, PROPOSED_METHOD)
            if method in paths
        )
        figure_learners = (PROPOSED_METHOD,) if PROPOSED_METHOD in paths else ()
    else:
        figure_methods = tuple(method for method in ACTIVE_MAINLINE_METHODS if method in paths)
        figure_learners = tuple(method for method in ACTIVE_LEARNED_METHODS if method in paths)
    if not figure_methods:
        raise ValueError("no active mainline methods are available for plotting")
    if not figure_learners:
        raise ValueError("no active learned methods are available for plotting")
    training, losses, monitor, parts = [], [], [], []
    finite_horizon = any(_read_json(Path(paths[m])/"manifest.json")["config"].get("episode_objective") == "finite_horizon" for m in figure_learners)
    if finite_horizon:
        reward_mode = _read_json(Path(paths[figure_learners[0]])/"manifest.json")["config"]["learning_reward_mode"]
        arm = "finite_horizon_report_direct" if reward_mode == "report_direct" else "finite_horizon_existing_reward"
    with SourceGuard([Path(p) for p in paths.values()], [logs, plots]) as guard:
        for method in figure_learners:
            scale = Path(paths[method])
            for file, target in (("training_episodes.jsonl", training), ("training_metrics.jsonl", losses), ("development_monitor.jsonl", monitor)):
                file_path = scale/"diagnostics"/file
                if not file_path.is_file():
                    raise FileNotFoundError(file_path)
                target.extend({"method": method, **json.loads(line)} for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())
            for row in [r for r in training if r["method"] == method]:
                episode = scale/"train"/row["episode_id"]
                windows = reward_windows(_read_csv(episode/"rewards.csv"), _read_json(episode/"scenario.json"))
                if len(windows) != int(row["window_count"]):
                    raise ValueError("training episode/update binding failed")
                if abs(sum(w["mean_reward"] for w in windows)-float(row["cumulative_return"])) > 1e-6:
                    raise ValueError("training return closure failed")
                parts.extend({"method":method, "episode_id":row["episode_id"], **w} for w in windows)
        unchanged = guard.verify()
    if guard.test_attempts or not all(r["unchanged"] for r in unchanged):
        raise ValueError("plot source access audit failed")
    active_validation_rows = [row for row in validation_rows if row.get("method") in figure_methods]
    if not active_validation_rows:
        raise ValueError("active validation rows are missing")
    for file, values in (("training_episodes",training), ("training_updates",losses), ("development_monitor",monitor),
                         ("reward_decomposition",parts), ("validation_episodes",active_validation_rows)):
        _write_table(logs/f"{file}.csv",values)
    termination_rows = []
    for row in training:
        method = str(row["method"])
        manifest = _read_json(Path(paths[method]) / "manifest.json")
        max_steps = int(manifest.get("config", {}).get("max_steps", 50))
        classified = classify_episode_outcome(row, max_steps=max_steps)
        termination_rows.append(
            {
                "method": method,
                "episode_id": row.get("episode_id", ""),
                "training_episode": row.get("training_episode", ""),
                "window_count": row.get("window_count", ""),
                "max_steps": max_steps,
                "termination_reason": row.get("termination_reason", ""),
                "terminated": row.get("terminated", ""),
                "truncated": row.get("truncated", ""),
                "collision_occurred": row.get("collision_occurred", ""),
                "outcome_class": classified,
                "minimum_clearance": row.get("minimum_clearance", ""),
                "pre_projection_minimum_clearance": row.get(
                    "pre_projection_minimum_clearance", ""
                ),
                "collision_types": row.get("collision_types", ""),
                "collision_objects": row.get("collision_objects", ""),
                "safety_projection_count": row.get("safety_projection_count", ""),
            }
        )
    _write_table(logs / "training_termination_attribution.csv", termination_rows)
    figures = []
    styles = {
        method: style
        for method, style in zip(figure_learners, ("-", "--", "-."))
    }

    def display_label(method):
        return PAPER_BRANCH_LABELS.get(method, method)

    def curves(ax, data, x, y):
        for method in figure_learners:
            rows = sorted([r for r in data if r["method"]==method], key=lambda r:float(r[x]))
            ax.plot([r[x] for r in rows], [r[y] for r in rows], color=COLORS[method], linestyle=styles[method], marker="o" if len(rows)==1 else None, label=display_label(method), linewidth=1.1)
            if finite_horizon and x == "training_episode" and y != "window_count":
                rolling = [np.mean([float(r[y]) for r in rows[max(0,i-4):i+1]]) for i in range(len(rows))]
                ax.plot([r[x] for r in rows], rolling, color="black", linewidth=1.3, label="5-episode rolling mean (visual aid)")
        ax.grid(alpha=.2); ax.set_xlabel(x.replace("_"," ").title()); ax.set_ylabel(y.replace("_"," "))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    def legend(fig, ax, extras=None):
        handles, labels=ax.get_legend_handles_labels()
        if extras:
            handles.extend(extras)
            labels.extend([handle.get_label() for handle in extras])
        fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.08),ncol=3,fontsize=8,frameon=False)

    marker_handles = [
        Line2D(
            [], [], linestyle="None", marker="X", markersize=6,
            markeredgecolor="#b91c1c", markerfacecolor="none",
            label="X Collision termination",
        ),
        Line2D(
            [], [], linestyle="None", marker="^", markersize=6,
            markeredgecolor="#d97706", markerfacecolor="none",
            label="△ Non-collision early termination",
        ),
    ]
    def save(fig, name, caption):
        fig.suptitle(f"3 UAV | seed=42 | {arm} | descriptive development experiment", fontsize=11, y=1.04)
        fig.canvas.draw()
        for extension in ("png","pdf"):
            path=plots/f"{name}.{extension}"
            fig.savefig(path,dpi=300,bbox_inches="tight",pad_inches=.18)
            figures.append(str(path.resolve()))
        plt.close(fig)
        (logs/f"{name}_caption.md").write_text(caption+"\n\n共同口径：seed=42单次训练；无跨种子置信区间；原始曲线保留；有限时域组叠加最近最多5回合滚动均值作为视觉辅助；不读test；规则方法无训练损失，不以零线替代。\n",encoding="utf-8")
    has_learning_channel = bool(training) and all(
        "learning_cumulative_return" in row and "mean_learning_reward" in row
        for row in training
    )
    fig, axes=plt.subplots(
        3 if has_learning_channel else 2,
        1,
        figsize=(10, 9 if has_learning_channel else 7),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    curves(axes[0],training,"training_episode","cumulative_return")
    if has_learning_channel:
        curves(axes[1],training,"training_episode","learning_cumulative_return")
        axes[1].set_title("Learning return used by PPO (training-only feedback)")
        if finite_horizon and all(abs(r["learning_cumulative_return"]-r["cumulative_return"]) < 1e-6 for r in training):
            axes[1].set_title("Learning return = reporting return (no clipping)")
        curves(axes[2],training,"training_episode","window_count")
    else:
        curves(axes[1],training,"training_episode","window_count")
    for row in termination_rows:
        if row["outcome_class"] == "collision":
            marker, color = "X", "#b91c1c"
        elif row["outcome_class"] == "early_termination":
            marker, color = "^", "#d97706"
        else:
            continue
        training_episode = row["training_episode"]
        source = next(
            item for item in training
            if item["method"] == row["method"]
            and item.get("episode_id") == row["episode_id"]
        )
        axes[0].scatter(
            [training_episode], [source["cumulative_return"]],
            marker=marker, facecolors="none", edgecolors=color, s=42,
            linewidths=1.2, zorder=5,
        )
        if has_learning_channel:
            axes[1].scatter(
                [training_episode], [source["learning_cumulative_return"]],
                marker=marker, facecolors="none", edgecolors=color, s=42,
                linewidths=1.2, zorder=5,
            )
    legend(fig,axes[0], marker_handles)
    save(fig,"01_episode_return",(
        "训练回合累计团队回报：每窗口先对3架UAV取均值，再在本回合实际窗口上求和。"
        "X表示碰撞终止，△表示非碰撞提前终止；达到最大步数的正常time_limit回合不标记。"
        "缺少可靠终止原因的短回合不用于原因归因。下方面板为回合长度；负奖励下提前终止可能造成表面累计回报升高，不能只看上图。"
        + ("本版前三张训练图仅展示 Local-EKF-GAT-MAPPO (Proposed Method)，不读取MAPPO-MLP。" if proposed_only else "当前论文主线训练图同时保留历史MAPPO-MLP架构对照。")
    ))
    fig, axes=plt.subplots(4 if finite_horizon else 2,1,figsize=(10,11 if finite_horizon else 7),layout="constrained")
    curves(axes[0],training,"training_episode","mean_reward"); axes[0].set_title("Train: stochastic rollouts")
    means=[]
    for method in figure_learners:
        for episode in sorted({r["completed_training_episodes"] for r in monitor if r["method"]==method}):
            group=[r for r in monitor if r["method"]==method and r["completed_training_episodes"]==episode]
            means.append({"method":method,"completed_training_episodes":episode,"mean_reward":np.mean([r["mean_reward"] for r in group]), "position_rmse_norm_m":np.mean([r["position_rmse_norm_m"] for r in group])})
    curves(axes[1],means,"completed_training_episodes","mean_reward"); axes[1].set_title("Fixed development monitor: deterministic policy, 5 scenes")
    axes[1].set_xlim(0,max(float(r["training_episode"]) for r in training))
    if finite_horizon:
        curves(axes[2],training,"training_episode","mean_learning_reward")
        axes[2].set_title("Train: learning reward")
        curves(axes[3],means,"completed_training_episodes","position_rmse_norm_m")
        axes[3].set_title("Fixed development monitor: position RMSE (m)")
    legend(fig,axes[0])
    save(fig,"02_mean_team_reward","平均团队奖励为累计团队回报除以实际窗口数。上图为随机策略train；下图为固定5个开发监测场景的确定性评估，二者不能逐点当作同一数据集。监测不用于中间checkpoint选择；Smoke可能仅含训练前监测点。训练图仅展示 Proposed Method。" if proposed_only else "平均团队奖励为累计团队回报除以实际窗口数。上图为随机策略train；下图为固定5个开发监测场景的确定性评估，二者不能逐点当作同一数据集。监测不用于中间checkpoint选择；Smoke可能仅含训练前监测点。")
    loss_fields = ("loss", "actor_loss", "critic_loss")
    if proposed_only:
        loss_fields = ("loss", "actor_loss", "critic_loss", "entropy", "clip_fraction", "approximate_kl")
    fig, axes=plt.subplots(2,3,figsize=(14,7),layout="constrained") if proposed_only else plt.subplots(1,3,figsize=(14,4),layout="constrained")
    axes = np.asarray(axes).reshape(-1)
    for ax,field in zip(axes,loss_fields):
        curves(ax,losses,"update",field)
        ax.set_title(field.replace("_"," ").title())
        if finite_horizon and field == "approximate_kl":
            for method in figure_learners:
                rows = sorted([r for r in losses if r["method"] == method], key=lambda r:r["update"])
                ax.plot([r["update"] for r in rows], [r["approximate_kl_max"] for r in rows], "k--", label="Minibatch maximum")
            ax.legend(fontsize=7)
    for ax in axes[len(loss_fields):]:
        ax.set_visible(False)
    legend(fig,axes[0])
    save(fig,"03_training_loss","Total、Actor、Critic损失按真实Update记录； Proposed-only版本同时展示Entropy、Clip fraction和Approximate KL。rollout4中100个episode对应25次Update，其他配置为100次；对应episode列表、环境窗口数、优化步数及乘子更新次数见training_updates.csv。loss不要求单调下降，不能用其降低证明定位改善。" if proposed_only else "Total、Actor、Critic损失按真实Update记录。rollout4中100个episode对应25次Update，其他配置为100次；对应episode列表、环境窗口数、优化步数及乘子更新次数见training_updates.csv。loss不要求单调下降，不能用其降低证明定位改善。")
    short={
        "local_only_fixed":"Local",
        "random_cooperative":"Random",
        "budgeted_greedy_ig":"Greedy IG",
        "mappo_mlp":"MLP",
        PROPOSED_METHOD:"Proposed",
    }
    def comparison(ax, field, unit):
        for index,method in enumerate(figure_methods):
            values=np.asarray([float(r[field]) for r in active_validation_rows if r["method"]==method])
            if values.size==0 or not np.isfinite(values).all(): raise ValueError(f"missing/nonfinite {method}/{field}")
            offsets=np.linspace(-.16,.16,len(values))
            ax.scatter(index+offsets,values,s=13,color=COLORS[method],alpha=.65)
            ax.errorbar(index,float(values.mean()),yerr=float(values.std(ddof=1)) if len(values)>1 else 0,fmt="_",color="black",capsize=4,markersize=15)
        ax.set_xticks(range(len(figure_methods)),[short.get(method, method) for method in figure_methods],rotation=25); ax.set_ylabel(unit); ax.grid(axis="y",alpha=.2)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    comparison(axes[0],"position_rmse_norm_m","Position-vector RMSE (m)")
    comparison(axes[1],"position_p95_norm_m","Position-vector P95 (m)")
    save(fig,"04_localization_error","validation定位向量范数RMSE、P95。彩色点为共同场景的逐回合结果（从左到右偏移顺序为episode编号）；黑标为均值±场景间标准差，不是跨初始化95%区间。包含提前终止回合，另报告共同时间前缀敏感性分析。")
    fig,axes=plt.subplots(2 if finite_horizon else 1,3,figsize=(14,8 if finite_horizon else 4),layout="constrained")
    fields = ("actual_measurement_bytes","j_com","packet_loss")
    units = ("Measurement payload (bytes/episode)","J_com (normalized objective)","Candidate-link packet loss")
    if finite_horizon:
        for row in active_validation_rows:
            row["measurement_count_from_bytes"] = float(row["actual_measurement_bytes"])/512
        fields += ("measurement_count_from_bytes", "delay_s", "aoi_s")
        units += ("Measurements / episode", "Delay (s)", "AoI (s)")
    for ax,field,unit in zip(np.asarray(axes).reshape(-1),fields,units):
        comparison(ax,field,unit)
    save(fig,"05_communication_performance","validation实际测量负载、J_com及候选链路丢包率。负载=实际有效测量数×512字节；不含背景通信。独立基线没有协同测量负载，但并非零背景通信。J_com可能因连通性收益为负，不等同于字节数或训练期代理代价。时延/AoI见validation_episodes.csv。")
    figures.extend(
        generate_safety_collision_attribution(
            paths,
            active_validation_rows,
            logs,
            plots,
            arm=arm,
            figure_profile=figure_profile,
        )
    )
    (logs/"figure_index.md").write_text(f"# seed42 {arm} 五张主图\n\n绘图配置：{figure_profile}\n\n"+"\n".join(f"- {p}" for p in figures)+"\n\n源代码：src/experiments/mainline_compact_figures.py；相应CSV在本目录。训练图为诊断，不单独支持算法有效。\n",encoding="utf-8")
    return figures
