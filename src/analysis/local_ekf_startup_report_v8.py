"""Offline three-UAV result tables and verified plots for the v8 batch."""
from __future__ import annotations

import csv
import numpy as np

from . import local_ekf_startup_diagnostic_v8 as audit
from .partial_gnss_report_v7 import write_csv
from ..experiments import partial_gnss_baseline_v7 as v7


RUNS=("native_original_001","startup_original_001","startup_confirmation_001")


def report():
    data=audit.BASE/"final_report_001";logs=audit.LOGS/"final_report_001"
    plots=v7.ROOT/"plots/preliminary"/audit.STEM/"final_report_001"
    for path in (data,logs,plots):path.mkdir(parents=True,exist_ok=False)
    manifests={r:v7.old.read_json(audit.BASE/r/"manifest.json") for r in RUNS}
    all_summary=[{**row,"run_id":run} for run,m in manifests.items() for row in m["summary"]]
    all_steps=[]
    for run in RUNS:
        with (audit.LOGS/run/"per_uav_steps.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["run_id"]=run;all_steps.append(row)
    comparison=[]
    for item in all_summary:
        subset=[r for r in all_steps if r["run_id"]==item["run_id"] and r["scene_id"]==item["scene_id"] and int(r["seed"])==item["seed"] and r["uid"]==item["uid"]]
        item["vo_nis_mean_valid"]=float(np.mean([float(r["nis_vo"]) for r in subset if float(r["nis_vo"])>=0]))
        gnss=[float(r["nis_gnss"]) for r in subset if float(r["nis_gnss"])>=0]
        item["gnss_nis_mean_valid"]=float(np.mean(gnss)) if gnss else None
        item["position_velocity_nees_mean"]=float(np.mean([float(r["position_velocity_nees"]) for r in subset]))
        comparison.append(item)
    write_csv(logs/"three_uav_comparison.csv",comparison)
    matched=[]
    baseline=manifests[RUNS[0]]["summary"]
    for old in baseline:
        prefix=[r for r in all_steps if r["run_id"]==RUNS[1] and r["scene_id"]==old["scene_id"] and int(r["seed"])==old["seed"] and r["uid"]==old["uid"] and int(r["step"])<=old["steps"]]
        matched.append({"scene_id":old["scene_id"],"seed":old["seed"],"uid":old["uid"],"steps":old["steps"],
            "native_rmse_m":old["position_rmse_m"],"startup_same_prefix_rmse_m":float(np.sqrt(np.mean([float(r["position_sse_m2"]) for r in prefix]))),
            "scope":"same keyed noise and prefix length, different physical trajectories"})
    write_csv(logs/"matched_prefix_comparison.csv",matched)
    text=["# v8 三无人机本地 EKF 启动结果\n",
        "结论：**有条件通过**。四个原条件和四个前瞻确认条件均通过冻结准入；只适用于声明的静止启动及原世界系无重力传感器模型。未验证协同 CI 接入后的闭环，不具备直接 MAPPO 训练结论。\n",
        "|方案|几何|种子|机|步数/状态|位置RMSE(m)|位置NEES|VO接受/有效|GNSS接受/有效|加速度偏置RMSE(m/s²)|陀螺偏置RMSE(rad/s)|终态位置迹|",
        "|---|---|---:|---|---|---:|---:|---|---|---:|---:|---:|"]
    for r in comparison:
        text.append(f"|{r['arm']}|{r['scene_id']}|{r['seed']}|{r['uid']}|{r['steps']}/{r['status']}|{r['position_rmse_m']:.4f}|{r['position_nees_mean']:.3f}|{r['vo_accepted']}/{r['vo_attempted']}|{r['gnss_accepted']}/{r['gnss_attempted']}|{r['accel_bias_rmse']:.4f}|{r['gyro_bias_rmse']:.4f}|{r['final_position_trace']:.4f}|")
    total=sum(m["physical_steps"] for m in manifests.values())
    protected=v7.old.read_json(audit.BASE/"protected_inputs_before.json")
    changed=[p for p,h in protected.items() if v7.old.file_hash(p)!=h]
    if changed:raise RuntimeError(changed)
    native_stop=[r for r in baseline if r["uid"]=="uav_01" and r["status"]=="censored"]
    text += ["\n原失败保留："+v7.old.encode(native_stop),
        "\n同前缀长度对照见 matched_prefix_comparison.csv；完整失效前缀与完整修复时域不可不加说明地当成相同时长。第 6–120 步表见各运行的 post_warmup_summary.csv。",
        f"\n物理步 {total}/1440，离线组步 1076/1076；优化器更新 0；历史哈希改变 {len(changed)}/{len(protected)}。",
        "\n两个几何局部运动是平移等价的；没有协同测距或安全投影时同种子各机误差接近一致，不能把几何重复算成额外独立随机样本。新增种子只运行候选，不宣称未运行的新增原生对照表现。",
        "\nGNSS/VO 原门限始终 9，初始误差、P0、传感器及过程噪声不改。前 1 秒新偏置观测及已知静止条件增加了信息，成本包含在原 120 步内。",
        "\n参考机全部实现严格停车，GNSS 受限机全部未实现真值严格停车；后者绝对偏移不可由 IMU/位移 VO 单独消除。",
        "\nNEES/NIS 仅为有限种子诊断，不证明一般一致性。后续需在不改变 v7 CI/调度/预算的隔离版本中复验合作定位；训练、部署继续等待确认。"]
    (data/"results.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    v7.old.write_json(data/"manifest.json",{"physical_steps_this_report":0,"physical_steps_total":total,"offline_group_steps_total":1076,
        "optimizer_updates":0,"protected_count":len(protected),"protected_changed":changed,"admission":"conditional_pass",
        "scientific_training_ready":False,"original_gate":v7.old.read_json(audit.BASE/"original_candidate_gate.lock.json")["passed"],
        "confirmation_gate":v7.old.read_json(audit.BASE/"extra_candidate_gate.lock.json")["passed"]})
    draw(plots)
    print("Report generated",data,"physical",total,"protected",len(protected))


def draw(plots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    old=v7.old.read_json(audit.BASE/RUNS[0]/"record_train_collinear_s20267102.json")
    new=v7.old.read_json(audit.BASE/RUNS[1]/"record_train_collinear_s20267102.json")
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for record,label in ((old,"native / stopped at 29"),(new,"startup / full 120")):
        steps=[f["step"] for f in record["frames"]]
        error=[np.linalg.norm(np.array(f["audit"]["states"]["uav_01"][:3])-f["evaluation_truth"]["position"]["uav_01"]) for f in record["frames"]]
        bias=[f["audit"]["states"]["uav_01"][11]-f["evaluation_truth"]["accel_bias"]["uav_01"][2] for f in record["frames"]]
        nis=[f["audit"]["diagnostics"]["uav_01"]["nis_vo"] for f in record["frames"]]
        nis=[value if value>=0 else np.nan for value in nis]
        altitude=[f["evaluation_truth"]["position"]["uav_01"][2] for f in record["frames"]]
        for ax,values in zip(axes.flat,(error,bias,nis,altitude)):ax.plot(steps,values,label=label)
    for ax,ylabel in zip(axes.flat,("Position error (m)","Accelerometer z bias error (m/s²)","VO NIS","Actual altitude (m)")):
        ax.set_xlabel("Decision step (total horizon 120)");ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.axvspan(1,5,color="gray",alpha=.12)
    axes[1,0].axhline(9,color="black",ls="--",lw=1,label="unchanged VO gate")
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle("UAV 1 / seed 20267102 / collinear: physical closed-loop comparison")
    for extension in ("png","svg"):
        path=plots/f"failure_vs_startup_valid_nis.{extension}"
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=160)
    plt.close(fig)


if __name__=="__main__":report()
