"""Offline v7 evidence tables, paired frozen-state checks and plots."""
from __future__ import annotations

import argparse
import copy
import csv
from collections import defaultdict

import numpy as np

from ..experiments import partial_gnss_baseline_v7 as v7
from .partial_gnss_fusion_v7 import MeasurementCIEKF


def grouped(summaries):
    groups = defaultdict(list)
    for row in summaries:
        groups[(row["phase"], row["method"], row["policy"])].append(row)
    result = []
    for key, rows in groups.items():
        result.append({"phase": key[0], "method": key[1], "policy": key[2], "blocks": len(rows),
            "absolute_rmse_m": float(np.sqrt(np.mean([r["uav2_absolute_rmse_m"]**2 for r in rows]))),
            "relative_rmse_m": float(np.sqrt(np.mean([r["cluster_relative_rmse_m"]**2 for r in rows]))),
            "position_nees": float(np.mean([r["uav2_position_nees_mean"] for r in rows])),
            "uwb_nis": float(np.mean([r["uwb_nis_mean"] for r in rows if r["uwb_nis_mean"] is not None])) if any(r["uwb_nis_mean"] is not None for r in rows) else None,
            "final_position_trace": float(np.mean([r["final_uav2_covariance_trace_m2"] for r in rows])),
            "accepted": sum(r["uwb_accepted"] for r in rows), "requested": sum(r["uwb_requested"] for r in rows),
            "measurement_byte_equivalent": sum(r["measurement_bytes_equivalent"] for r in rows),
            "strict_parking": sum(bool(r["uav2_strict_parking"]) for r in rows) if key[0] == "closed_loop" else None,
            "minimum_separation_m": min(r["minimum_separation_m"] for r in rows),
            "boundary": sum(r["boundary_projections"] for r in rows), "safety": sum(r["safety_projections"] for r in rows),
            "collision": sum(r["collision_count"] for r in rows)})
    return result


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v7.old.encode(v) if isinstance(v, (dict,list)) else v for k,v in row.items()})


def frozen_checks(data):
    """12 predetermined target states; two real bank observations, never resampled."""
    rows = []
    for path in sorted(data.glob("record_*.json")):
        record = v7.old.read_json(path)
        if record.get("status", "complete") != "complete":
            continue
        for step in (6, 34, 58):
            frame = record["frames"][step-1]
            initial = np.array(frame["audit"]["states"][v7.TARGET])
            P = np.array(frame["audit"]["covariances"][v7.TARGET])
            truth = np.array(frame["evaluation_truth"]["position"][v7.TARGET])
            before_sse = float(np.sum((initial[:3]-truth)**2))
            bank = frame["observable"]["range_bank"]
            for source in ("none", *v7.PEERS):
                row = {"scene_id": record["case"]["scene_id"], "seed": record["case"]["seed"], "step": step,
                    "source": source, "available": source == "none" or source in bank,
                    "pre_sse_m2": before_sse, "evaluation_only": True}
                if source == "none":
                    row.update({"accepted": False,"post_sse_m2":before_sse,"true_sse_gain_m2":0.,"trace_gain_m2":0.,"weight":1.,"cost_byte_equivalent":0})
                elif source in bank:
                    measurement = bank[source]
                    candidate = MeasurementCIEKF(initial.copy(), v7.simulation().ekf)
                    candidate.covariance = P.copy()
                    accepted = candidate.update_relative_range(measurement["measurement"], np.array(measurement["packet_state"])[:3],
                        np.array(measurement["packet_covariance"]), measurement_variance=measurement["variance"], nlos_bias_m=measurement["nlos_bias"],timestamp=step*.2)
                    event = candidate.last_relative_range_update
                    after = float(np.sum((candidate.position-truth)**2))
                    row.update({"accepted": accepted,"post_sse_m2":after,"true_sse_gain_m2":before_sse-after,
                        "trace_gain_m2":float(np.trace(P[:3,:3])-np.trace(candidate.covariance[:3,:3])),
                        "weight":event["weight"],"cost_byte_equivalent":512,"nis":event["nis"],
                        "direction":event["direction"],"reference_direction_variance_m2":event["q"],
                        "packet_age_s":step*.2-measurement["message_envelope"]["send_time_s"],
                        "final_position_covariance":candidate.covariance[:3,:3].tolist(), "estimated_position_change_m":(candidate.position-initial[:3]).tolist()})
                rows.append(row)
    return rows


def report(run_id):
    data = v7.BASE/run_id
    logs = v7.ROOT/"logs/preliminary"/v7.STEM/run_id
    plots = v7.ROOT/"plots/preliminary"/v7.STEM/run_id
    manifest = v7.old.read_json(data/"manifest.json")
    aggregates = grouped(manifest["summaries"])
    v7.old.write_json(data/"aggregate.json",aggregates)
    write_csv(logs/"aggregate.csv",aggregates)
    checks = frozen_checks(data)
    write_csv(logs/"frozen_counterfactual.csv", checks)
    gains = [r for r in checks if r["source"] != "none" and r["available"]]
    x=np.array([r["trace_gain_m2"] for r in gains]); y=np.array([r["true_sse_gain_m2"] for r in gains])
    corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
    pairs=[]
    for key in {(r["scene_id"],r["seed"],r["step"]) for r in checks}:
        rows=[r for r in gains if (r["scene_id"],r["seed"],r["step"]) == key]
        if len(rows)==2:
            pairs.append({"scene_id":key[0],"seed":key[1],"step":key[2],
                "absolute_direction_cosine":float(abs(np.array(rows[0]["direction"])@np.array(rows[1]["direction"]))),
                "peer_sse_gain_difference_m2":float(rows[0]["true_sse_gain_m2"]-rows[1]["true_sse_gain_m2"]),
                "proxy_order_matches_true_order":bool((rows[0]["trace_gain_m2"]-rows[1]["trace_gain_m2"])*(rows[0]["true_sse_gain_m2"]-rows[1]["true_sse_gain_m2"])>0)})
    diagnostic={"frozen_state_count":len(checks)//3,"available_peer_comparisons":len(pairs),"proxy_instant_sse_correlation":corr,
        "proxy_order_matches_count":sum(r["proxy_order_matches_true_order"] for r in pairs),"pairs":pairs,
        "scope":"12 preselected None-producer states, instantaneous gains; not cumulative or learned reward validation"}
    v7.old.write_json(data/"frozen_diagnostic.json",diagnostic)
    text=["# v7 实际结果\n",f"物理步 {manifest['physical_steps']}；回放滤波步 {manifest['replay_filter_steps']}；优化器更新 0；闭环门限 {manifest['closed_loop_gate_passed']}。\n",
        "|阶段|方法|策略|UAV2绝对RMSE(m)|集群相对RMSE(m)|位置NEES|UWB NIS|终态位置迹|接受/请求|严格停车|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregates:
        nis="—" if r["uwb_nis"] is None else f"{r['uwb_nis']:.3f}"
        text.append(f"|{r['phase']}|{r['method']}|{r['policy']}|{r['absolute_rmse_m']:.3f}|{r['relative_rmse_m']:.3f}|{r['position_nees']:.3f}|{nis}|{r['final_position_trace']:.3f}|{r['accepted']}/{r['requested']}|{r['strict_parking']}|")
    text += ["\n所有逐场景/种子指标见 summary.csv；真值只用于离线评估。回放轨迹来自 None，闭环结果单独列出。",
        f"\n预定 6/34/58 步的冻结状态配对：可用双参考 {len(pairs)}，位置迹收益与即时真实平方误差收益相关系数 {corr}，代理与真值排序一致 {diagnostic['proxy_order_matches_count']}/{len(pairs)}。不得解释为累计收益因果验证。",
        "\n两个独立噪声种子，无 formal consistency/generalization 结论。保守创新界 NIS 不按卡方校准解读；联合 EKF 使用当前私有信息，有额外信息优势。",
        "\n独立候选未部署到主系统；scientific.training_ready=false，停止并等待确认。"]
    text += ["\n## 未完成的冻结条件\n",v7.old.encode(manifest.get("failures", [])),
        "\n存在截断条件时，汇总只覆盖完成的完整时域块，不能把部分轨迹混入完整 RMSE，也不能宣称两个种子复现成功。"]
    path=data/"results.md"
    if path.exists(): raise FileExistsError(path)
    path.write_text("\n".join(text)+"\n",encoding="utf-8")
    draw(manifest,plots)
    print(v7.old.encode({"aggregate":aggregates,"frozen_diagnostic":diagnostic}))


def draw(manifest,plots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries=manifest["summaries"]
    fig, axes=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    keys=sorted({(c["scene_id"],c["seed"]) for c in v7.old.read_json(v7.CONFIG)["cases"]})
    for ax,key in zip(axes.flat,keys):
        selected=[r for r in summaries if (r["scene_id"],r["seed"])==key and r["phase"]=="replay"]
        if not selected:
            failure=next(f for f in manifest["failures"] if (f["scene_id"],f["seed"])==key)
            ax.text(.5,.5,f"{key[0]} / {key[1]}\nCENSORED: boundary safety stop at step {failure.get('step',failure.get('steps'))}\nNo full-horizon comparison",ha="center",va="center",transform=ax.transAxes)
            ax.axis("off")
            continue
        for method in v7.METHODS:
            values=[next(r["uav2_absolute_rmse_m"] for r in selected if r["method"]==method and r["policy"]==p) for p in v7.POLICIES]
            ax.plot(v7.POLICIES,values,marker="o",label=method)
        ax.set_title(f"{key[0]} / {key[1]}")
        ax.set_ylabel("UAV 2 absolute position RMSE (m)")
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    for extension in ("png","svg"):
        path=plots/f"paired_replay_rmse_with_censoring.{extension}"
        if path.exists(): raise FileExistsError(path)
        fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="bounded_s20267101_102")
    report(parser.parse_args().run_id)


if __name__=="__main__": main()
