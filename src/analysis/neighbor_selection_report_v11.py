"""Read-only v11 evidence, seed-level comparisons and observable-value checks."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from ..experiments import neighbor_selection_validation_v11 as run
from .neighbor_selection_gate_v11 import assess, correlation
from .partial_gnss_validation_gate_v10 import METRICS, pooled
from .partial_gnss_closed_report_v10 import _write_csv
from ..experiments.three_uav_measurement_selection_contract import expected_relative_credit


def report(batch):
    spec=run.verify();base=run.batch_path(batch)
    manifests={p:run.checked_manifest(base/p/"manifest.json") for p in ("develop","confirm","counterfactual") if (base/p/"manifest.json").exists()}
    rows=[r for p in ("develop","confirm") if p in manifests for r in manifests[p]["summaries"]]
    points=manifests.get("counterfactual",{}).get("results",[])
    integrity=all(p in manifests for p in ("develop","confirm","counterfactual")) and all(manifests[p]["safety"]["passed"] for p in ("develop","confirm") if p in manifests) and not manifests.get("counterfactual",{}).get("failures")
    assessment=assess(rows,points,spec,integrity)
    data=base/"report";logs=run.LOGS/batch/"report";plots=run.ROOT/"plots/preliminary"/run.STEM/batch
    data.mkdir(exist_ok=False);logs.mkdir(parents=True,exist_ok=False);plots.mkdir(parents=True,exist_ok=False)
    groups=[]
    for phase in ("develop","confirm"):
        for scene in spec["geometry_ids"]:
            for policy in spec["policies"]:
                rr=[r for r in rows if r["study_phase"]==phase and r["scene_id"]==scene and r["policy"]==policy]
                if not rr:continue
                n=sum(r["uwb_attempted"] for r in rr);accepted=sum(r["uwb_accepted"] for r in rr)
                groups.append({"study_phase":phase,"scene_id":scene,"policy":policy,"blocks":len(rr),
                    **{k:pooled(rr,k) for k in METRICS},
                    "position_nees_mean":float(np.mean([r["uav2_position_nees_mean"] for r in rr])),
                    "uwb_nis_mean":float(np.mean([r["uwb_nis_mean"] for r in rr if r["uwb_nis_mean"] is not None])) if n else None,
                    "final_localization_error_mean_m":float(np.mean([r["uav2_final_localization_error_m"] for r in rr])),
                    "requests":sum(r["uwb_requested"] for r in rr),"attempts":n,"fused":accepted,
                    "fusion_rate":accepted/n if n else None,"request_bytes_equivalent":sum(r["measurement_bytes_equivalent"] for r in rr),
                    "strict_parking":sum(r["uav2_strict_parking"] for r in rr),
                    "final_position_trace_mean_m2":float(np.mean([r["target_position_trace_final_m2"] for r in rr])),
                    "none_choices":sum(r["selection_counts"].get("none",0) for r in rr),
                    "peer0_choices":sum(r["selection_counts"].get("uav_00",0) for r in rr),
                    "peer1_choices":sum(r["selection_counts"].get("uav_01",0) for r in rr)})
    pairs=[]
    for phase in ("develop","confirm"):
        seeds=spec["development_seeds"] if phase=="develop" else spec["confirmation_seeds"]
        for scene in spec["geometry_ids"]:
            for seed in seeds:
                rr=[r for r in rows if r["study_phase"]==phase and r["scene_id"]==scene and r["seed"]==seed]
                item={"study_phase":phase,"scene_id":scene,"seed":seed}
                for r in rr:
                    for metric in (*METRICS,"uwb_requested","uwb_accepted","uav2_position_nees_mean","uav2_final_localization_error_m","uav2_strict_parking"):
                        item[r["policy"]+"_"+metric]=r[metric]
                pairs.append(item)
    decisions={};feature_rows=[];legacy=[]
    for phase in ("develop","confirm"):
        if phase not in manifests:continue
        for r in manifests[phase]["summaries"]:
            record=run.old.read_json(base/phase/f"{r['scene_id']}_s{r['seed']}_{r['policy']}_record.json")
            dual=distinguishable=0
            for d in record["v11_decisions"]:
                key=(phase,r["scene_id"],r["seed"],d["step"])
                if r["policy"]=="random_peer":decisions[key]=d
                c=d["context"];legal=set(c["legal_sources"])
                if all(u in legal and u in c["peers"] for u in run.previous.v9.v7.PEERS):
                    dual+=1
                    local={"uav_id":"uav_02","estimate":c["own_state"],"covariance":c["own_covariance"],
                        "neighbor_messages":{u:{"estimate":v["state"],"covariance":v["covariance"]} for u,v in c["peers"].items()}}
                    credits=[expected_relative_credit(local,u,.08**2) for u in run.previous.v9.v7.PEERS]
                    distinguishable+=int(abs(credits[0]-credits[1])>1e-6)
            legacy.append({"study_phase":phase,"scene_id":r["scene_id"],"seed":r["seed"],"policy":r["policy"],"dual_windows":dual,"old_linear_credit_gap_over_1e6":distinguishable})
    feature_points=[]
    for p in points:
        if not p.get("dual_eligible"):continue
        d=decisions.get((p["study_phase"],p["scene_id"],p["seed"],p["step"]))
        if d is None:continue
        f=d["scoring"]["features"];c=d["scoring"]["candidates"]
        x={**p,"self_trace":f["self_position_trace_m2"],"shared_angle":f["angle01_rad"]}
        for name,keys in (("peer_trace",("peer0_position_trace_m2","peer1_position_trace_m2")),("distance",("distance20_m","distance21_m")),("rssi",("rssi0_dbm","rssi1_dbm")),("packet_age",("aoi0_s","aoi1_s"))):
            x[name+"_difference"]=f[keys[0]]-f[keys[1]] if f[keys[0]] is not None and f[keys[1]] is not None else None
        for name in ("information_fraction","geometry","reliability","communication_cost"):
            x[name+"_difference"]=c["uav_00"][name]-c["uav_01"][name]
        feature_points.append(x)
    for phase in ("develop","confirm"):
        for scene in ["all",*spec["geometry_ids"]]:
            for seed in ["all",*(spec["development_seeds"] if phase=="develop" else spec["confirmation_seeds"])]:
                rr=[r for r in feature_points if r["study_phase"]==phase and (scene=="all" or r["scene_id"]==scene) and (seed=="all" or r["seed"]==seed)]
                for name in ("peer_trace_difference","distance_difference","rssi_difference","packet_age_difference","information_fraction_difference","geometry_difference","reliability_difference","communication_cost_difference","self_trace","shared_angle","score_difference"):
                    usable=[r for r in rr if r.get(name) is not None]
                    x=[r[name] for r in usable];y=[abs(r["utility_difference"]) if name in ("self_trace","shared_angle") else r["utility_difference"] for r in usable]
                    feature_rows.append({"study_phase":phase,"scene_id":scene,"seed":seed,"feature":name,"n":len(usable),
                        "feature_min":min(x) if x else None,"feature_max":max(x) if x else None,
                        "pearson":correlation(x,y),"spearman":correlation(x,y,True),
                        "outcome":"absolute_peer_utility_gap" if name in ("self_trace","shared_angle") else "signed_peer_utility_difference"})
    _write_csv(logs/"full_summary.csv",rows);_write_csv(logs/"grouped.csv",groups);_write_csv(logs/"per_seed.csv",pairs)
    _write_csv(logs/"counterfactual_points.csv",points);_write_csv(logs/"feature_relationships.csv",feature_rows);_write_csv(logs/"legacy_proxy_counts.csv",legacy)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,spec["geometry_ids"]):
        rr=[g for g in groups if g["study_phase"]=="confirm" and g["scene_id"]==scene]
        ax.bar([r["policy"] for r in rr],[r[METRICS[0]] for r in rr])
        ax.set(title=scene,ylabel="UAV2 absolute RMSE (m)");ax.tick_params(axis="x",rotation=30);ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(plots/("confirmation_policy_rmse."+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,spec["geometry_ids"]):
        for phase,marker in (("develop","o"),("confirm","x")):
            rr=[r for r in points if r.get("dual_eligible") and r["scene_id"]==scene and r["study_phase"]==phase]
            ax.scatter([r["score_difference"] for r in rr],[r["utility_difference"] for r in rr],marker=marker,label=phase,alpha=.7)
        ax.set(title=scene,xlabel="predecision score0 - score1",ylabel="four-state utility0 - utility1 (m²)")
        ax.axhline(0,color="k",lw=.6);ax.axvline(0,color="k",lw=.6);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(plots/("score_vs_conditional_peer_value."+ext),dpi=160)
    plt.close(fig)
    aggregate={"assessment":assessment,"grouped":groups,"legacy_proxy_counts":legacy,"feature_relationships":feature_rows,
        "physical_steps":sum(manifests[p]["physical_steps"] for p in ("develop","confirm") if p in manifests),
        "cf_local_states":manifests.get("counterfactual",{}).get("local_states"),"optimizer_updates":0,"training_ready":False}
    run.old.write_json(data/"aggregate.json",aggregate)
    lines=["# v11结果（冻结设计）","",f"科学准入：{assessment['status']}；training_ready=false。","",
        "|阶段|几何|策略|绝对RMSE(m)|相对RMSE(m)|请求/尝试/融合|NEES|停车|","|---|---|---|---:|---:|---|---:|---:|"]
    for r in groups:lines.append(f"|{r['study_phase']}|{r['scene_id']}|{r['policy']}|{r[METRICS[0]]:.6f}|{r[METRICS[1]]:.6f}|{r['requests']}/{r['attempts']}/{r['fused']}|{r['position_nees_mean']:.4f}|{r['strict_parking']}|")
    lines += ["","## 冻结门禁实际计算","", "```json",json.dumps(assessment,ensure_ascii=False,indent=2),"```","",
        "四个独立确认种子，几何与窗口是重复块；相关性是描述性，不是逐窗口独立显著性检验。",
        "counterfactual固定Random消息队列和物理轨迹，None只是省略一次本地更新；不能冒充无请求完整闭环。",
        "十标量不包含方向协方差；Greedy额外使用本机完整P与收到的估计方向、缓存。信息项为原CI协方差克隆代理，不保证真实误差改善。",
        "原30个可区分双候选代理门槛单独报告，不等同本轮科学准入；没有降门槛、训练或改变奖励。"]
    (data/"results.md").write_text("\n".join(lines),encoding="utf-8")
    run.write_manifest(data,logs,{"phase":"read_only_report","assessment":assessment,"optimizer_updates":0})
    run.verify();print(run.old.encode(assessment),flush=True)
    return aggregate
