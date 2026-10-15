"""Consolidate existing v9 evidence and archive final provenance; no simulation."""
from __future__ import annotations
import argparse
import csv
import subprocess
import numpy as np
from ..experiments import partial_gnss_revalidation_v9 as v9
from .partial_gnss_report_v7 import write_csv


def finish(run_id):
    spec = v9.verify_design()
    manifest = v9.v7.old.read_json(v9.BASE / run_id / "manifest.json")
    summaries, gate = manifest["summaries"], manifest["gate"]
    ci = [r for r in summaries if r["method"] == "measurement_ci"]
    none = [r for r in ci if r["policy"] == "none"]
    random = [r for r in ci if r["policy"] == "random_peer"]
    resolution = v9.v7.old.read_json(v9.BASE / ("resolution_"+run_id) / "manifest.json")
    grouped = v9.v7.old.read_json(v9.BASE / ("report_"+run_id) / "aggregate.json")["grouped"]
    data, logs = v9.BASE / ("evidence_"+run_id), v9.LOGS / ("evidence_"+run_id)
    for p in (data, logs):
        p.mkdir(parents=True, exist_ok=False)
    case_metrics = []
    for r in ci:
        tape = v9.v7.old.read_json(v9.BASE / run_id / f"{r['scene_id']}_s{r['seed']}" / ("measurement_ci_"+r["policy"]+".json"))
        rows = tape["rows"][5:]
        case_metrics.append({**{k: r[k] for k in ("scene_id", "seed", "policy", "uav2_absolute_rmse_m", "cluster_relative_rmse_m", "uav2_position_nees_mean", "uwb_requested", "uwb_message_arrived", "uwb_attempted", "uwb_accepted", "ci_weight_one", "measurement_bytes_equivalent", "target_position_trace_final_m2", "uwb_nis_mean", "two_legal_windows")},
            "poststartup_abs_rmse_m": float(np.sqrt(np.mean([x["target_sse_m2"] for x in rows]))),
            "poststartup_relative_rmse_m": float(np.sqrt(np.mean([x["relative_sse_mean_m2"] for x in rows]))),
            "relative_nees_mean": float(np.mean([x["relative_nees_mean"] for x in tape["rows"]])),
            "relative_nees_kind": "unknown_cross_upper_bound"})
    write_csv(logs / "ci_case_metrics.csv", case_metrics)
    with (v9.LOGS / ("ci_diagnostic_"+run_id) / "frozen_state_comparisons.csv").open(encoding="utf-8") as stream:
        paired = list(csv.DictReader(stream))
    geometry_angles = {}
    for scene in sorted({r["scene_id"] for r in none}):
        values = [float(r["evaluation_true_axis_angle_deg"]) for r in paired if r["scene_id"] == scene and r.get("evaluation_true_axis_angle_deg")]
        geometry_angles[scene] = {"min_deg": min(values), "max_deg": max(values), "scope": "predesignated available two-candidate states at steps6/34/58; unsigned range axis angle"}
    metrics = {"physical_steps": manifest["physical_steps"], "replay_filter_steps": manifest["replay_filter_steps"],
        "candidate_cooperative_closed_loop_steps": 0, "pure_counterfactual_updates": resolution["total_frozen_state_updates"],
        "optimizer_updates": 0, "independent_seeds": 4, "admission": "fail", "gate_failures": gate["failures"],
        "random_requests": sum(r["uwb_requested"] for r in random), "random_message_available": sum(r["uwb_message_arrived"] for r in random),
        "random_attempts": sum(r["uwb_attempted"] for r in random), "random_absorbed": sum(r["uwb_accepted"] for r in random),
        "random_weight_one": sum(r["ci_weight_one"] for r in random), "random_request_byte_equivalent": sum(r["measurement_bytes_equivalent"] for r in random),
        "random_packet_age_max_s": max(r["packet_age_max_s"] for r in random),
        "random_position_covariance_final_range_m2": [min(r["target_position_trace_final_m2"] for r in random), max(r["target_position_trace_final_m2"] for r in random)],
        "random_position_nees_mean_range": [min(r["uav2_position_nees_mean"] for r in random), max(r["uav2_position_nees_mean"] for r in random)],
        "all_covariance_min_eigenvalue": min(r["covariance_min_eigenvalue"] for r in summaries),
        "all_local_full_covariance_trace_max": max(r["full_covariance_trace_max"] for r in summaries),
        "reference_rmse_range_m": [min(r[f"uav{i}_rmse_m"] for r in none for i in (0,1)), max(r[f"uav{i}_rmse_m"] for r in none for i in (0,1))],
        "reference_gnss_accepted_range_per120": [min(r[f"uav{i}_gnss_accepted"] for r in none for i in (0,1)), max(r[f"uav{i}_gnss_accepted"] for r in none for i in (0,1))],
        "none_minimum_separation_m": min(r["minimum_separation_m"] for r in none),
        "none_collision_boundary_safety_projection_counts": [sum(r[k] for r in none) for k in ("collision_count", "boundary_projections", "safety_projections")],
        "none_physical_broadcast_edge_transmissions": sum(r["physical_broadcast_edge_transmissions"] for r in none),
        "none_two_legal_windows": sum(r["two_legal_windows"] for r in none),
        "none_one_legal_windows": sum(r["one_legal_windows"] for r in none),
        "source_reconstruction_max": max(r["source_reconstruction_max"] for r in summaries),
        "packet_reconstruction_max": max(r["packet_reconstruction_max"] for r in summaries),
        "joint_none_reconstruction_max": max(r["joint_none_reconstruction_max"] for r in summaries),
        "geometry_axis_angles": geometry_angles, "resolution_diagnostic": resolution,
        "primary_pooled": gate["pooled"], "scientific_training_ready": False}
    v9.v7.old.write_json(data / "metrics.json", metrics)
    lines = ["# v9 部分 GNSS 协同定位基线复验（2026-10-13）", "",
        "本轮总准入结论：**未通过候选协同闭环准入**。本地参考机可靠性通过；协同定位回放收益得到四个既定独立种子的有限实验支持。唯一失败项是预先冻结的20%有效融合率工程门槛，实际26/228=11.40%。这不等于平均定位收益失败，也不是CI普遍不一致的证据。未降低门槛，未执行候选协同闭环，训练准入保持false。", "",
        "冻结种子20267101、20267102、20268111、20268112；共线/非共线为同一种子内的两种几何，不能视为8个独立重复。新的8条None物理记录共960步；96组回放共11520组步；全部完整120步。总时域24秒，启动5步1秒，窗口6/10/…/118共29个。", "",
        "## 参考机与安全", "",
        "|种子|UAV0 RMSE(m)|UAV1 RMSE(m)|GNSS接受0/1（各120有效）|VO接受/有效0|VO接受/有效1|位置NEES0/1|", "|---|---:|---:|---|---|---|---|"]
    for r in sorted((x for x in none if x["scene_id"] == "train_noncollinear"), key=lambda x: x["seed"]):
        lines.append(f"|{r['seed']}|{r['uav0_rmse_m']:.4f}|{r['uav1_rmse_m']:.4f}|{r['uav0_gnss_accepted']}/{r['uav1_gnss_accepted']}|{r['uav0_vo_accepted']}/{r['uav0_vo_valid']}|{r['uav1_vo_accepted']}/{r['uav1_vo_valid']}|{r['uav0_position_nees_mean']:.3f}/{r['uav1_position_nees_mean']:.3f}|")
    lines += ["", "两种几何中的本地误差只差浮点舍入；参考0/1仅本地IMU/VO/GNSS，没有目标UWB反馈。上述结论限于持续GNSS可用、声明的静止启动及原世界系无重力IMU模型，没有验证GNSS间歇失锁。",
        f"新的None物理基准8/8安全完成，碰撞/边界/安全投影均0，最小机间距{metrics['none_minimum_separation_m']:.6f}米。协同回放不改变这条物理轨迹，不能借此宣称协同导航闭环安全或停车成功。", "",
        "## 测量CI对照（每几何四种子汇总）", "",
        "|几何|策略|绝对RMSE(m)|相对RMSE(m)|位置NEES|UWB创新界NIS|请求/可用消息/尝试/吸收|权重1|等价字节|", "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for g in grouped:
        if g["method"] != "measurement_ci":
            continue
        nis = "无测量" if g["NIS"] is None else f"{g['NIS']:.3f}"
        lines.append(f"|{g['geometry']}|{g['policy']}|{g['pooled_sqrt_mean_RMSE2_m']:.4f}|{g['relative_pooled_sqrt_mean_RMSE2_m']:.4f}|{g['NEES']:.3f}|{nis}|{g['requests']}/{g['arrivals']}/{g['attempted']}/{g['accepted']}|{g['w1']}|{g['cost']}|")
    lines += ["", "集群相对RMSE为三对无人机相对位置向量误差的均方根，不只是距离残差。512字节/请求是原模型的等价队列负载，不是完整消息序列化的线缆字节。可用消息指所选参考已到达的广播消息，不是测量请求应答包。回放的队列和广播固定为None记录，成本仅记账。", "",
        "## 逐种子主比较", "",
        "|几何|种子|None绝对RMSE|Random绝对RMSE|绝对改善|相对改善|尝试/吸收|位置NEES|终态位置迹(m²)|", "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    baselines = {(r["scene_id"], r["seed"]): r for r in none}
    changes = {(r["scene_id"], r["seed"]): r for r in gate["paired_changes"]}
    for r in sorted(random, key=lambda r: (r["scene_id"], r["seed"])):
        key = (r["scene_id"], r["seed"])
        delta = changes[key]
        lines.append(f"|{r['scene_id']}|{r['seed']}|{baselines[key]['uav2_absolute_rmse_m']:.4f}|{r['uav2_absolute_rmse_m']:.4f}|{100*delta['uav2_absolute_rmse_m_improvement_fraction']:.2f}%|{100*delta['cluster_relative_rmse_m_improvement_fraction']:.2f}%|{r['uwb_attempted']}/{r['uwb_accepted']}|{r['uav2_position_nees_mean']:.3f}|{r['target_position_trace_final_m2']:.3f}|")
    lines += ["", "Random的绝对RMSE汇总2.3795→1.8986米，改善20.21%；相对RMSE1.9623→1.5623米，改善20.39%。每种几何4/4种子均改善，两项逐块均无退化。原成功/失败种子未筛除。该规模不构成泛化、严格统计一致性或可学习奖励的证明。", "",
        "## CI为何不融合", "",
        "Random共232请求、232所选消息可用、228尝试、26吸收；4次UWB无效、202次权重1/no_information_gain。没有TTL、复用消息、测量周期或创新门控拒绝。窗口内通常双合法，消息年龄最大0.4秒。预算每条消耗29、剩1；None和参考机不消耗测量预算。",
        "已验证原CI搜索严格执行三维位置迹最小化：单条测距只约束一个方向，而保守权重同时影响未观测方向；重复同方向后很快趋于不吸收。原权重网格步长0.05还遗漏接近1的较小目标改善。三策略合计616次不融合，127次端点导数支持连续目标也选不更新；488次在0.95—1的只读细采样中存在正的迹改善（中位0.01322平方米、最大0.10126平方米）。Random的202次不融合中146次存在该网格间改善。只计算目标曲线，未应用新权重或修改均值/协方差，不能把迹改善直接解释为真实定位改善。",
        "第58步全部8个预定实际CI状态均不吸收；用同一先验和同一测距调用历史restore-prior，仅作瞬时反事实，7个误差改善、1个恶化。共计96次纯更新达到冻结上限，未新增轨迹。说明部分被跳过的测量可能有价值，但更多吸收并不必然改善累计定位；restore-prior本身仍缺乏一般协方差一致性保证。", "",
        "## 几何、相关性与奖励限制", "",
        f"在预定可用双候选状态，共线真实测距轴夹角{geometry_angles['train_collinear']['min_deg']:.3f}—{geometry_angles['train_collinear']['max_deg']:.3f}度，非共线{geometry_angles['train_noncollinear']['min_deg']:.3f}—{geometry_angles['train_noncollinear']['max_deg']:.3f}度（无向测距轴）。前者方向近重复；后者具有互补方向。两个测距方向仍不足以一般地观测三维所有方向，VO与时间运动条件不可省略。",
        "非共线下Random比两个Fixed的汇总定位更好，支持方向互补的有限证据；共线下三策略接近。逐种子的最佳固定邻居可不同，不能从四个噪声种子宣称已具稳定的学习排序信号。参考机GNSS质量相近、时效均在TTL内；不能将主要瓶颈归咎于消息失效或当地EKF发散。CI需要保护未知相关性，但真实跨机相关性的贡献没有由本轮等信息受控实验独立分离，仍待验证。",
        f"数值协方差有限且PSD，Random位置NEES均值范围{metrics['random_position_nees_mean_range'][0]:.3f}—{metrics['random_position_nees_mean_range'][1]:.3f}，末态位置迹{metrics['random_position_covariance_final_range_m2'][0]:.3f}—{metrics['random_position_covariance_final_range_m2'][1]:.3f}平方米；较低NEES可能体现保守性。NEES时序相关，不能按120个独立样本做卡方一致性检验；CI的NIS是未知相关性创新界，也不是校准卡方统计。联合参考维护交叉协方差且获得当前私有传感器，不能据其表现直接推断分布式等信息算法不足。未修改原线性奖励代理，实际收益与奖励的一致性仍需后续设计。", "",
        "## 准入与下一步", "",
        "本地参考基础：通过。协同定位回放收益：四独立种子有限支持。协同闭环及学习型邻居选择准入：本轮未通过、尚未验证。20%信息吸收是人为预定的工程门槛而非理论必要条件；它是唯一阻断项，不能将阻断解释为定位平均精度不改善，也不在看完数据后降低或绕过。",
        "建议先在下一轮获确认的隔离验证中检查原CI目标的离散搜索分辨率/三维迹准则饱和，保持融合公式、相关性保护、观测、场景及预算不变，再比较累计真实误差与成本。若另拟准入应有独立科学依据、事先冻结，并与本轮失败标准分别报告。之后仍需真实请求/队列/导航反馈闭环，才固定为后续邻居选择基线。当前不启动GAT/MAPPO训练、不加入新融合算法。", "",
        "## 文件与PyCharm复现", "",
        "解释器：`D:\\research project\\.venv\\Scripts\\python.exe`（CPython3.13.5）；工作目录：`D:\\research project`。PyCharm选择Module name而非直接运行含相对导入的源文件。", "",
        "本轮实际命令（PowerShell工作目录为项目根）：", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m src.experiments.partial_gnss_revalidation_v9 --phase freeze",
        ".\\.venv\\Scripts\\python.exe -m src.experiments.partial_gnss_revalidation_v9 --phase paired --run-id paired_001",
        ".\\.venv\\Scripts\\python.exe -m src.analysis.partial_gnss_ci_diagnostic_v9 --run-id paired_001",
        ".\\.venv\\Scripts\\python.exe -m src.analysis.partial_gnss_report_v9 --run-id paired_001",
        ".\\.venv\\Scripts\\python.exe -m src.analysis.partial_gnss_visualize_v9 --run-id paired_001",
        ".\\.venv\\Scripts\\python.exe -m src.analysis.partial_gnss_ci_resolution_v9 --run-id paired_001", "```", "",
        "PyCharm参数与上述命令中模块名之后部分相同。当前目录已存在，写入步骤会拒绝重复执行/覆盖；这些命令记录完整执行顺序，不能删除历史再重跑。复跑须先确认独立批次、冻结和运行上限；本轮停止。", "",
        "闭环入口实际存在：`src.experiments.partial_gnss_revalidation_v9 --phase closed --run-id closed_001 --replay-run paired_001`。本轮门控false，此命令会在构建环境前拒绝。不得强改gate.lock或阈值来运行。", "",
        "新增实现为src/experiments/partial_gnss_revalidation_v9.py与src/analysis/partial_gnss_{replay,gate,report,ci_diagnostic,ci_resolution,visualize,evidence}_v9.py；对应纯测试在tests/test_partial_gnss_*_v9.py。仅新增隔离模块；未修改正式在线EKF、原CI、导航、GAT/MAPPO或奖励。", "",
        f"数据目录`data/synthetic/{v9.STEM}`；CSV目录`logs/preliminary/{v9.STEM}`；图片目录`plots/preliminary/{v9.STEM}/visual_{run_id}`。全部历史SHA保护968项；README原文已归档。没有训练、部署、提交或推送。"]
    report_path = v9.ROOT / "sections/3_partial_gnss_revalidation_v9.md"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    with (v9.ROOT / "README.md").open("ab") as stream:
        stream.write(("\n\n### v9 部分 GNSS 协同定位基线复验（2026-10-13）\n\n"
            "已用v8五步静止启动生成8条新的None安全闭环记录，并完成96组配对回放；四个独立种子、两种几何全部保留。"
            "原测量CI Random-peer使绝对/相对RMSE汇总改善20.21%/20.39%，每种几何均4/4种子改善。"
            "实际融合26/228=11.40%，未达到预先冻结20%工程吸收门槛；本轮未通过协同闭环准入，候选闭环0步、训练0次。"
            "只读诊断确认三维迹保守性与0.05权重网格分辨率影响；未改变原CI、传感器、导航或门槛。"
            "详细逐种子、协方差、通信指标、限制和PyCharm命令见[第三阶段v9复验](sections/3_partial_gnss_revalidation_v9.md)。"
            "保持scientific.training_ready=false，停止等待确认。\n").encode("utf-8"))
    v9.verify_design()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=v9.ROOT, text=True).strip()
    artifacts = {str(p.relative_to(v9.ROOT)): v9.v7.old.file_hash(p) for group in (v9.BASE, v9.LOGS, v9.ROOT/"plots/preliminary"/v9.STEM) for p in group.rglob("*") if p.is_file()}
    for pattern in ("src/analysis/partial_gnss_*_v9.py", "src/experiments/partial_gnss_revalidation_v9.py", "tests/test_partial_gnss_*_v9.py", "knowledge/partial_gnss_revalidation_v9_protocol.md", "sections/3_partial_gnss_revalidation_v9.md", "README.md"):
        for p in v9.ROOT.glob(pattern):
            artifacts[str(p.relative_to(v9.ROOT))] = v9.v7.old.file_hash(p)
    artifacts[str(v9.CONFIG.relative_to(v9.ROOT))] = v9.v7.old.file_hash(v9.CONFIG)
    v9.v7.old.write_json(v9.BASE / "final_artifact_manifest.json", {"schema": "v9_final_evidence",
        "git_head": head, "artifact_sha256": artifacts, "protected_count": 968, "protected_changed": [],
        "physical_steps": 960, "replay_group_steps": 11520, "pure_counterfactual_updates": 96,
        "candidate_cooperative_closed_loop_steps": 0, "optimizer_updates": 0,
        "admission": "fail", "scientific_training_ready": False, "formal_core_or_ci_changed": False})
    print(v9.v7.old.encode({k: metrics[k] for k in metrics if k != "resolution_diagnostic"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="paired_001")
    finish(parser.parse_args().run_id)
