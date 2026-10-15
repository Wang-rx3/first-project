"""Read-only v12 report: oracle labels never become execution rewards."""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from ..experiments import neighbor_selection_validation_v12 as run
from ..experiments import neighbor_selection_validation_v11 as legacy
from .neighbor_selection_joint_audit_v11 import joint_assessment
from .neighbor_selection_gate_v11 import correlation
from .partial_gnss_validation_gate_v10 import METRICS, pooled
from .partial_gnss_report_v7 import write_csv


def groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["study_phase"], row["scene_id"], row["policy"])].append(row)
    output = []
    for (phase, scene, policy), rr in grouped.items():
        complete = all(r.get("full_horizon_eligible") for r in rr)
        output.append({"study_phase": phase, "scene_id": scene, "policy": policy, "blocks": len(rr),
                       **{k: pooled(rr, k) for k in METRICS},
                       "position_nees_mean": float(np.mean([r["uav2_position_nees_mean"] for r in rr])),
                       "uwb_nis_mean": float(np.mean([r["uwb_nis_mean"] for r in rr if r["uwb_nis_mean"] is not None])) if any(r["uwb_nis_mean"] is not None for r in rr) else None,
                       "requests": sum(r["uwb_requested"] for r in rr),
                       "attempts": sum(r["uwb_attempted"] for r in rr),
                       "fused": sum(r["uwb_accepted"] for r in rr),
                       "request_bytes_equivalent": sum(r["measurement_bytes_equivalent"] for r in rr),
                       "strict_parking": sum(r.get("uav2_strict_parking", False) for r in rr),
                       "final_localization_error_mean_m": float(np.mean([r["uav2_final_localization_error_m"] for r in rr if r["uav2_final_localization_error_m"] is not None])) if complete else None,
                       "full_horizon_eligible": complete})
    return output


def report(batch):
    from .neighbor_selection_diagnostic_v12 import collect_points, summarize_points
    spec = run.verify()
    base = run.validate_batch(batch)
    manifests = {p: run.checked_manifest(base / p / "manifest.json") for p in ("develop", "confirm", "counterfactual")}
    rows = manifests["develop"]["summaries"] + manifests["confirm"]["summaries"]
    points = collect_points(base / "counterfactual", base)
    diagnostic = summarize_points(points)
    integrity = (all(manifests[p]["safety"]["passed"] for p in ("develop", "confirm"))
                 and not manifests["counterfactual"]["failures"] and len(points) == 96
                 and all(p.get("identity_matches") for p in points))
    # Apply the existing joint scientific predicate to the new candidate.
    # Unequal request windows are reported, never relabelled as peer-only gain.
    candidate_rows = [{**r, "policy": "greedy"} if r["policy"] == "minimal_greedy" else r
                      for r in rows if r["policy"] != "greedy"]
    candidate_points = [{**p, "score_difference": p["minimal_score_difference"],
                         "ranking_correct": p["minimal_ranking_correct"]} for p in points]
    assessment = joint_assessment(candidate_rows, candidate_points, spec, integrity)
    assessment["candidate"] = "minimal_greedy"
    assessment["request_volume_scope"] = "whole action none/peer comparison; unmatched windows cannot establish a peer-only causal improvement"
    assessment["training_ready"] = False
    assessment["formal_reward_modified"] = False
    per_seed = []
    bykey = {(r["study_phase"], r["scene_id"], r["seed"], r["policy"]): r for r in rows}
    for phase in ("develop", "confirm"):
        seeds = spec["development_seeds"] if phase == "develop" else spec["confirmation_seeds"]
        for scene in spec["geometry_ids"]:
            for seed in seeds:
                item = {"study_phase": phase, "scene_id": scene, "seed": seed}
                for policy in spec["policies"]:
                    r = bykey.get((phase, scene, seed, policy))
                    if r:
                        for field in (*METRICS, "uav2_position_nees_mean", "uwb_nis_mean", "uwb_requested", "uwb_attempted", "uwb_accepted", "uav2_strict_parking", "uav2_final_localization_error_m"):
                            item[policy + "_" + field] = r.get(field)
                r = bykey.get((phase, scene, seed, "random_peer"))
                m = bykey.get((phase, scene, seed, "minimal_greedy"))
                item["request_windows_match"] = bool(r and m and r["request_windows"] == m["request_windows"])
                if r and m:
                    for metric in METRICS:
                        item["minimal_vs_random_" + metric + "_gain"] = 1-m[metric]/r[metric] if m[metric] is not None and r[metric] else None
                per_seed.append(item)
    reproduction = []
    prior = legacy.checked_manifest(legacy.batch_path("research_001") / "develop/manifest.json")
    for r in rows:
        if r["study_phase"] != "develop" or r["policy"] == "minimal_greedy":
            continue
        oldrow = next(x for x in prior["summaries"] if all(x[k] == r[k] for k in ("scene_id", "seed", "policy")))
        error = max(abs(r[k]-oldrow[k]) for k in METRICS) if r.get("full_horizon_eligible") else None
        reproduction.append({"scene_id": r["scene_id"], "seed": r["seed"], "policy": r["policy"], "rmse_difference_max_m": error,
                             "requests_equal": oldrow["uwb_requested"] == r["uwb_requested"], "fusions_equal": oldrow["uwb_accepted"] == r["uwb_accepted"]})
    conditional_totals = defaultdict(lambda: {"conditional_selected_four_gain_sum_m2": 0., "conditional_selected_current_gain_sum_m2": 0., "checkpoints": 0})
    for p in points:
        key = (p["study_phase"], p["scene_id"], p["seed"])
        conditional_totals[key]["checkpoints"] += 1
        # These six conditional labels are not a trajectory attribution.
        selected = p.get("actual_selection", "none")
        conditional_totals[key]["conditional_selected_four_gain_sum_m2"] += float(p.get("four_state_sse_gain_"+selected, 0) or 0)
        conditional_totals[key]["conditional_selected_current_gain_sum_m2"] += float(p.get("current_sse_gain_"+selected, 0) or 0)
    cumulative_rows = []
    for (phase, scene, seed), item in conditional_totals.items():
        none, random = bykey.get((phase, scene, seed, "none")), bykey.get((phase, scene, seed, "random_peer"))
        cumulative_rows.append({"study_phase": phase, "scene_id": scene, "seed": seed, **item,
                               "physical_full_horizon_mean_SSE_gain_m2": none[METRICS[0]]**2-random[METRICS[0]]**2 if none and random and none[METRICS[0]] is not None and random[METRICS[0]] is not None else None})
    cumulative_correlations = []
    for phase in ("develop", "confirm"):
        rr = [r for r in cumulative_rows if r["study_phase"] == phase and r["physical_full_horizon_mean_SSE_gain_m2"] is not None]
        xx = [r["conditional_selected_four_gain_sum_m2"] for r in rr]
        yy = [r["physical_full_horizon_mean_SSE_gain_m2"] for r in rr]
        cumulative_correlations.append({"study_phase": phase, "geometry_blocks": len(rr), "independent_seeds": len({r["seed"] for r in rr}),
                                        "pearson": correlation(xx, yy), "spearman": correlation(xx, yy, True),
                                        "scope": "descriptive conditional labels versus whole-policy endpoints; feedback and prior updates differ; no causal attribution"})
    data, logs = base / "report", run.LOGS / batch / "report"
    plots = run.ROOT / "plots/preliminary" / run.STEM / batch
    for folder in (data, logs, plots):
        folder.mkdir(parents=True, exist_ok=False)
    gg = groups(rows)
    write_csv(logs / "full_summary.csv", rows)
    write_csv(logs / "grouped.csv", gg)
    write_csv(logs / "per_seed.csv", per_seed)
    write_csv(logs / "conditional_points.csv", points)
    write_csv(logs / "baseline_reproduction.csv", reproduction)
    write_csv(logs / "conditional_vs_physical_cumulative.csv", cumulative_rows)
    write_csv(logs / "conditional_vs_physical_correlations.csv", cumulative_correlations)
    for name in ("correlations", "rankings"):
        if isinstance(diagnostic.get(name), list):
            write_csv(logs / (name + ".csv"), diagnostic[name])
    stage1 = run.old.read_json(run.BASE / "stage1/diagnosis.json")
    evidence = {"assessment": assessment, "diagnostic": diagnostic, "groups": gg, "per_seed": per_seed,
                "stage1": stage1, "integrity": integrity,
                "conditional_vs_physical_cumulative": cumulative_correlations,
                "baseline_reproduction": reproduction, "physical_steps": sum(manifests[p]["physical_steps"] for p in ("develop", "confirm")),
                "cf_local_states": manifests["counterfactual"]["local_states"], "cf_range_calls": manifests["counterfactual"]["relative_ci_calls"],
                "cf_points": len(points), "cf_identity_max_error": max(p.get("identity_error_max", 0) or 0 for p in points),
                "safety": {p: manifests[p]["safety"] for p in ("develop", "confirm")},
                "optimizer_updates": 0, "training_ready": False, "formal_reward_changed": False,
                "interpretation": "oracle true-gain net labels only offline; CI trace proxy not proven expected MSE; four seeds are independent, geometry/windows repeated blocks"}
    run.old.write_json(data / "aggregate.json", evidence)
    lines = ["# V12 邻居选择可学习性诊断", "", "真实误差只作为离线标签；未训练、未改正式奖励或原滤波/导航。", "",
             "## 独立确认真实闭环", "", "|几何|策略|绝对 RMSE m|相对 RMSE m|请求|融合|严格停车/4|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in gg:
        if r["study_phase"] == "confirm":
            lines.append(f"|{r['scene_id']}|{r['policy']}|{r['uav2_absolute_rmse_m']:.6f}|{r['cluster_relative_rmse_m']:.6f}|{r['requests']}|{r['fused']}|{r['strict_parking']}|")
    lines += ["", "## 冻结科学判据", "", "```json", json.dumps(assessment, ensure_ascii=False, indent=2), "```", "",
              "## 方法与限制", "", "在线最小代理使用决策前可观测估计和原 CI 零创新克隆的三维位置协方差迹下降，减去每次请求 0.01 m² 的冻结诊断成本。none 得分为零。独立几何/可靠性正加分被移除。", "",
              "新增方向投影状态仅为原可观测输入的编码，不创造新信息。无模型拟合、无真值决策。真实净收益标签不能部署；同一真实收益参与定义与评价产生的相关性不能证明代理有效。", "",
              "反事实固定 Random 物理轨迹和消息，在本地更新后、协同更新前分叉，仅重放三个已记录本地传感器包。与真实 none 请求策略的通信/运动闭环不同。120 步闭环 RMSE 才是策略终点。", "",
              "请求次数不同的结果属于测量/none 与成本联合决策，不能归因为单独的邻居身份选择。逐种子结果和所有伤害样本见 per_seed.csv；NEES/NIS 是描述统计，不构成严格一般一致性保证。", "",
              "v11 的原 30 个代理差异门槛及失败结论完整保留；原冻结科学门槛和联合双指标种子判据未降低。完成后停止，等待确认。"]
    (data / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    positions = np.arange(len(spec["policies"]))
    for index, scene in enumerate(spec["geometry_ids"]):
        rr = {r["policy"]: r for r in gg if r["study_phase"] == "confirm" and r["scene_id"] == scene}
        axes[0].bar(positions + (index-.5)*.35, [rr[p][METRICS[0]] for p in spec["policies"]], width=.35, label=scene.replace("train_", ""))
        axes[1].bar(positions + (index-.5)*.35, [rr[p]["requests"] for p in spec["policies"]], width=.35, label=scene.replace("train_", ""))
    axes[0].set_ylabel("UAV2 absolute RMSE (m), pooled four seeds")
    axes[1].set_ylabel("Total requests, four seeds")
    for ax in axes:
        ax.set_xticks(positions, spec["policies"], rotation=15)
        ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(plots / ("confirmation_rmse_and_cost." + ext), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, score in zip(axes, ("score_difference", "minimal_score_difference")):
        for scene in spec["geometry_ids"]:
            pp = [p for p in points if p["study_phase"] == "confirm" and p["scene_id"] == scene and p[score] is not None]
            ax.scatter([p[score] for p in pp], [p["utility_difference"] for p in pp], label=scene.replace("train_", ""), alpha=.7)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.axvline(0, color="gray", linewidth=.7)
        ax.set_xlabel("Old score difference" if score == "score_difference" else "Minimal CI proxy difference (m²)")
        ax.set_ylabel("Conditional four-state SSE gain difference (m²)")
        ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(plots / ("confirmation_proxy_and_true_labels." + ext), dpi=160)
    plt.close(fig)
    run.write_manifest(data, logs, {"assessment": assessment, "artifact_count": len(list(data.iterdir()))})
    run.verify()
    print(run.old.encode({"assessment": assessment, "physical_steps": evidence["physical_steps"], "cf_points": len(points)}), flush=True)
