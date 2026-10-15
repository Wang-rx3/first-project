"""Final additive V13 figure, README update and exclusive artifact seal."""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
from ..experiments import neighbor_selection_validation_v13 as run


def quality_plot(base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    record = run.old.read_json(base / "confirm/train_noncollinear_s2026101344_random_peer_record.json")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for uid, color in (("uav_00", "#0072B2"), ("uav_01", "#D55E00")):
        q = [x for x in record["v13_quality_log"] if x["uid"] == uid]
        axes[0].step([x["time_s"] for x in q], [x["gnss_std_m"] for x in q], where="post", label=uid, color=color)
        f = [x for x in record["frames"] if "states" in x.get("audit", {})]
        axes[1].plot([x["step"] * .2 for x in f], [np.trace(np.asarray(x["audit"]["covariances"][uid])[:3, :3]) for x in f], color=color, label=uid)
    axes[0].set_ylabel("Generated GNSS std (m)")
    axes[0].legend(ncol=2); axes[0].set_title("Confirmation 2026101344, non-collinear, Random-peer")
    axes[1].set_ylabel("Local position covariance trace (m²)")
    axes[1].set_xlabel("Physical time (s)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    folder = run.ROOT / "plots/preliminary" / run.STEM / "research_001"
    for ext in ("png", "svg"):
        path = folder / ("dynamic_gnss_quality_and_covariance." + ext)
        if path.exists(): raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)


def finalize():
    spec = run.verify()
    base = run.validate_batch("research_001")
    for phase in ("precheck", "develop", "confirm", "counterfactual", "report", "interpretation"):
        manifest = run.checked_manifest(base / phase / "manifest.json")
        actual = {str(p.relative_to(run.ROOT)) for folder in (base / phase, run.LOGS / "research_001" / phase)
                  if folder.exists() for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
        if actual != set(manifest["artifact_sha256"]):
            raise RuntimeError("phase file membership mismatch: " + phase)
    lock = run.old.read_json(run.BASE / "reporting_addendum.lock.json")
    for path, digest in lock["source_sha256"].items():
        if run.old.file_hash(run.ROOT / path) != digest:
            raise RuntimeError("read-only audit source changed")
    for phase in ("develop", "confirm"):
        if not run.old.read_json(base / (phase + "_interface_audit_v2.json"))["passed"]:
            raise RuntimeError("dynamic interface audit failed")
    evidence = run.old.read_json(base / "interpretation/evidence.json")
    aggregate = run.old.read_json(base / "report/aggregate.json")
    if aggregate["training_ready"] or not evidence["interface_qa_passed"]:
        raise RuntimeError("unexpected scientific verdict/interface failure")
    if sum(run.old.read_json(base / p / "manifest.json")["physical_steps"] for p in ("develop", "confirm")) != 7680:
        raise RuntimeError("main physical count mismatch")
    if not run.old.read_json(base / "evidence_audit.json")["precheck_cap_passed"]:
        raise RuntimeError("precheck cap exceeded")
    quality_plot(base)
    readme = run.ROOT / "README.md"
    before = (run.BASE / "README_before.bin").read_bytes()
    if readme.read_bytes() != before:
        raise RuntimeError("README changed before intended append")
    content = """

### 2026-10-13：V13 动态异构参考邻居价值验证完成

新增隔离空间 GNSS/RF 质量模型，保留原 EKF、CI、UWB、导航、预算及全部历史成果。开发/确认各四种子、两几何、四策略，64/64条120步真实闭环安全完成；碰撞、边界和安全投影均0。GNSS声明标准差与逐包噪声匹配，质量头只随实际到达消息可见。

确认V13 Greedy相比Random请求230→30（−86.96%），定位RMSE变化不足1%；未达到原定双几何双指标5%性能门槛。净效用排序24/25正确，但含时延成本优势；去掉成本，真正定位价值可分点仅15个，14/15排序正确，且14个集中在第6/10步。已有有限局部状态-价值规律，仍缺持续闭环定位优化证据。

结论：可学习性有条件通过，正式MAPPO训练准入未通过（training_ready=false）。不训练、不部署、不提交、不推送。下一轮建议先确认等请求窗口的邻居身份收益与持续优化空间，等待用户确认。

详见 [V13研究报告](sections/3_dynamic_neighbor_value_v13.md)、[执行前协议](knowledge/neighbor_selection_v13_protocol.md)。独立结果目录为 `data/synthetic/three_uav_dynamic_neighbor_value_v13_20261013/batches/research_001`，CSV和图分别在对应 `logs/preliminary`、`plots/preliminary` 下。
"""
    with readme.open("ab") as stream:
        stream.write(content.encode("utf-8"))
    audit = subprocess.run([str(run.ROOT / ".venv/Scripts/python.exe"), "main.py", "audit-symbols"], cwd=run.ROOT, capture_output=True, text=True, encoding="utf-8")
    if audit.returncode != 0:
        raise RuntimeError(audit.stdout + audit.stderr)
    run.old.write_json(run.BASE / "symbol_audit.json", {"returncode": audit.returncode, "output": audit.stdout})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip()
    if head != run.old.read_json(run.BASE / "design.sha256.json")["git_head"]:
        raise RuntimeError("Git HEAD changed")
    run.verify()
    if not readme.read_bytes().startswith(before):
        raise RuntimeError("README append-only failure")
    final = run.BASE / "final_artifact_manifest.json"
    verification = run.BASE / "final_verification.lock.json"
    if final.exists() or verification.exists(): raise FileExistsError("V13 already sealed")
    paths = set()
    for folder in (run.BASE, run.LOGS, run.ROOT / "plots/preliminary" / run.STEM):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    for folder, pattern in (("src/analysis", "neighbor_selection_*_v13.py"), ("src/experiments", "neighbor_selection_*_v13.py"), ("tests", "test_neighbor_selection_*_v13.py")):
        paths.update((run.ROOT / folder).glob(pattern))
    paths.update((run.CONFIG, readme, run.ROOT / "sections/3_dynamic_neighbor_value_v13.md", run.ROOT / "knowledge/neighbor_selection_v13_protocol.md"))
    hashes = {str(p.relative_to(run.ROOT)): run.old.file_hash(p) for p in sorted(paths)}
    run.old.write_json(final, {"schema": "V13_final_artifact_seal", "artifact_sha256": hashes, "optimizer_updates": 0,
                             "training_ready": False, "physical_main_steps": 7680, "precheck_attempts": 15})
    for path, digest in hashes.items():
        if run.old.file_hash(run.ROOT / path) != digest: raise RuntimeError("final artifact changed: " + path)
    result = {"file_integrity": "passed", "protected_history_files": len(run.old.read_json(run.BASE / "protected_inputs_before.json")),
              "sealed_artifacts": len(hashes), "core_frozen_sources_unchanged": True, "README_append_only": True,
              "git_head": head, "main_physical_steps": 7680, "precheck_physical_attempts": 15,
              "planned_cf_points": 96, "valid_cf_points": 96, "cf_states": 1152, "cf_relative_CI_calls": 170,
              "research_status": "conditional_pass", "training_ready": False, "optimizer_updates": 0,
              "deployed": False, "committed": False, "pushed": False, "final_seal_sha256": run.old.file_hash(final)}
    run.old.write_json(verification, result)
    print(run.old.encode(result), flush=True)


if __name__ == "__main__":
    finalize()
