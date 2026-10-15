"""Offline report for the v9 paired partial-GNSS revalidation.

The report is deliberately read-only with respect to the execution run.  It
reads the paired manifest and gate lock, then writes a new, exclusive
``report_<run_id>`` directory under the v9 data and log roots.  No replay,
simulation, training, or resampling is performed here.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..experiments import partial_gnss_baseline_v7 as v7


ROOT = v7.ROOT
STEM = "three_uav_partial_gnss_revalidation_v9_20261013"
BASE = ROOT / "data/synthetic" / STEM
LOGS = ROOT / "logs/preliminary" / STEM
PLOTS = ROOT / "plots/preliminary" / STEM
EXPECTED_SUMMARIES = 96
EXPECTED_BLOCKS = 8
EXPECTED_INDEPENDENT_SEEDS = 4
EXPECTED_GEOMETRIES = 2
POLICIES = ("none", "fixed0", "fixed1", "random_peer")
METHODS = ("restore_prior", "measurement_ci", "joint_reference")


def _finite(value: Any) -> float | None:
    """Return a finite float, treating missing/non-numeric values as absent."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [number for row in rows if (number := _finite(row.get(key))) is not None]
    return float(np.mean(values)) if values else None


def _pooled_rmse(rows: Iterable[dict[str, Any]], key: str = "uav2_absolute_rmse_m") -> float | None:
    values = [number for row in rows if (number := _finite(row.get(key))) is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _delta(after: Any, before: Any) -> float | None:
    a, b = _finite(after), _finite(before)
    return None if a is None or b is None else float(a - b)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    n, d = _finite(numerator), _finite(denominator)
    return None if n is None or d is None or abs(d) < 1e-15 else float(n / d)


def _geometry(scene_id: Any) -> str:
    """Use stable human labels while retaining the exact scene_id in tables."""
    text = str(scene_id)
    if "noncollinear" in text:
        return "noncollinear"
    if "collinear" in text:
        return "collinear"
    return text


def _json_value(value: Any) -> Any:
    """CSV-safe representation for nested summary values and numpy scalars."""
    if isinstance(value, (dict, list, tuple)):
        return v7.old.encode(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a new CSV and fail loudly rather than overwriting an artifact."""
    if path.exists():
        raise FileExistsError(str(path))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key)) for key in fieldnames})


def _rejections(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = row.get("rejection_counts", {})
        if isinstance(raw, dict):
            for reason, count in raw.items():
                try:
                    counts[str(reason)] += int(count)
                except (TypeError, ValueError):
                    counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _summary_groups(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool all complete paired blocks by method, policy, and geometry."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        groups[(str(row.get("method", "")), str(row.get("policy", "")), _geometry(row.get("scene_id")))].append(row)

    result: list[dict[str, Any]] = []
    for (method, policy, geometry), rows in sorted(groups.items()):
        requests = sum(int(row.get("uwb_requested", 0) or 0) for row in rows)
        arrivals = sum(int(row.get("uwb_message_arrived", 0) or 0) for row in rows)
        attempted = sum(int(row.get("uwb_attempted", 0) or 0) for row in rows)
        accepted = sum(int(row.get("uwb_accepted", 0) or 0) for row in rows)
        w1 = sum(int(row.get("ci_weight_one", 0) or 0) for row in rows)
        rejections = _rejections(rows)
        result.append({
            "method": method,
            "policy": policy,
            "geometry": geometry,
            "blocks": len(rows),
            "pooled_sqrt_mean_RMSE2_m": _pooled_rmse(rows),
            "relative_pooled_sqrt_mean_RMSE2_m": _pooled_rmse(rows, "cluster_relative_rmse_m"),
            "NEES": _mean(rows, "uav2_position_nees_mean"),
            "NIS": _mean(rows, "uwb_nis_mean"),
            "requests": requests,
            "arrivals": arrivals,
            "attempted": attempted,
            "accepted": accepted,
            "effective": _ratio(accepted, attempted) if attempted else 0.0,
            "w1": w1,
            "w1_rate": _ratio(w1, attempted) if attempted else 0.0,
            "rejections": rejections,
            "cost": sum(int(row.get("measurement_bytes_equivalent", 0) or 0) for row in rows),
            "scene_ids": sorted({str(row.get("scene_id")) for row in rows}),
        })
    return result


def _per_seed_changes(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute every CI policy minus the measurement-CI None baseline."""
    keyed = {(str(row.get("scene_id")), row.get("seed"), str(row.get("method")), str(row.get("policy"))): row for row in summaries}
    keys = sorted({(str(row.get("scene_id")), row.get("seed")) for row in summaries})
    rows: list[dict[str, Any]] = []
    numeric = (
        ("uav2_absolute_rmse_m", "rmse_m"),
        ("cluster_relative_rmse_m", "relative_rmse_m"),
        ("uav2_position_nees_mean", "NEES"),
        ("uwb_nis_mean", "NIS"),
        ("uwb_requested", "requests"),
        ("uwb_message_arrived", "arrivals"),
        ("uwb_attempted", "attempted"),
        ("uwb_accepted", "accepted"),
        ("effective_fusion_rate", "effective"),
        ("ci_weight_one", "w1"),
        ("measurement_bytes_equivalent", "cost"),
    )
    for scene_id, seed in keys:
        baseline = keyed.get((scene_id, seed, "measurement_ci", "none"))
        if baseline is None:
            # Keep a visible row for a missing comparison instead of silently
            # dropping a geometry/seed block from the evidence table.
            rows.append({"geometry": _geometry(scene_id), "scene_id": scene_id, "seed": seed,
                         "method": "measurement_ci", "policy": "missing_baseline",
                         "baseline_status": "missing"})
            continue
        for policy in POLICIES:
            candidate = keyed.get((scene_id, seed, "measurement_ci", policy))
            row: dict[str, Any] = {"geometry": _geometry(scene_id), "scene_id": scene_id, "seed": seed,
                                   "method": "measurement_ci", "policy": policy,
                                   "baseline_status": "present" if candidate else "missing"}
            for source_key, output_key in numeric:
                base_value = baseline.get(source_key)
                candidate_value = candidate.get(source_key) if candidate else None
                row[f"ci_none_{output_key}"] = base_value
                row[f"ci_policy_{output_key}"] = candidate_value
                row[f"delta_{output_key}"] = _delta(candidate_value, base_value)
            row["ci_none_rejections"] = baseline.get("rejection_counts", {})
            row["ci_policy_rejections"] = candidate.get("rejection_counts", {}) if candidate else {}
            rows.append(row)
    return rows


def _reference_quality(summaries: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract reference UAV health from measurement-CI None only."""
    rows: list[dict[str, Any]] = []
    rmse_limit = 1.0
    acceptance_min = 0.9
    # The gate lock is authoritative for this report.  Older/newer locks may
    # omit the thresholds, so the frozen v9 values remain the fallback.
    limits = gate.get("limits", {}) if isinstance(gate, dict) else {}
    if isinstance(limits, dict):
        rmse_limit = float(limits.get("reference_rmse_limit_m", rmse_limit))
        acceptance_min = float(limits.get("reference_gnss_acceptance_min", acceptance_min))
    for item in sorted((row for row in summaries if row.get("method") == "measurement_ci" and row.get("policy") == "none"),
                       key=lambda row: (str(row.get("scene_id")), row.get("seed"))):
        def acceptance(index: int) -> float | None:
            return _ratio(item.get(f"uav{index}_gnss_accepted"), item.get(f"uav{index}_gnss_valid"))

        rate0, rate1 = acceptance(0), acceptance(1)
        covariance_eigenvalue = _finite(item.get("covariance_min_eigenvalue"))
        covariance_trace = _finite(item.get("full_covariance_trace_max"))
        checks = {
            "complete_120": item.get("status") == "complete" and item.get("steps") == 120,
            "uav0_rmse_ok": (_finite(item.get("uav0_rmse_m")) is not None and float(item["uav0_rmse_m"]) <= rmse_limit),
            "uav1_rmse_ok": (_finite(item.get("uav1_rmse_m")) is not None and float(item["uav1_rmse_m"]) <= rmse_limit),
            "uav0_gnss_ok": rate0 is not None and rate0 >= acceptance_min,
            "uav1_gnss_ok": rate1 is not None and rate1 >= acceptance_min,
            "safety_ok": all(int(item.get(key, 1) or 0) == 0 for key in ("collision_count", "boundary_projections", "safety_projections")),
            "covariance_ok": bool(item.get("covariance_finite")) and covariance_eigenvalue is not None and covariance_eigenvalue >= -1e-8 and covariance_trace is not None and covariance_trace < 1e4,
        }
        rows.append({
            "geometry": _geometry(item.get("scene_id")),
            "scene_id": item.get("scene_id"),
            "seed": item.get("seed"),
            "status": item.get("status"),
            "steps": item.get("steps"),
            "uav0_rmse_m": item.get("uav0_rmse_m"),
            "uav1_rmse_m": item.get("uav1_rmse_m"),
            "uav0_gnss_valid": item.get("uav0_gnss_valid"),
            "uav0_gnss_accepted": item.get("uav0_gnss_accepted"),
            "uav0_gnss_acceptance_rate": rate0,
            "uav1_gnss_valid": item.get("uav1_gnss_valid"),
            "uav1_gnss_accepted": item.get("uav1_gnss_accepted"),
            "uav1_gnss_acceptance_rate": rate1,
            "uav0_vo_valid": item.get("uav0_vo_valid"),
            "uav0_vo_accepted": item.get("uav0_vo_accepted"),
            "uav1_vo_valid": item.get("uav1_vo_valid"),
            "uav1_vo_accepted": item.get("uav1_vo_accepted"),
            "target_NEES": item.get("uav2_position_nees_mean"),
            "covariance_finite": item.get("covariance_finite"),
            "covariance_min_eigenvalue": item.get("covariance_min_eigenvalue"),
            "full_covariance_trace_max": item.get("full_covariance_trace_max"),
            "collision_count": item.get("collision_count"),
            "boundary_projections": item.get("boundary_projections"),
            "safety_projections": item.get("safety_projections"),
            "reference_rmse_limit_m": rmse_limit,
            "reference_gnss_acceptance_min": acceptance_min,
            "reference_health_pass": all(checks.values()),
            "reference_health_checks": checks,
        })
    return rows


def _gate_improvements(grouped: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    by_key = {(row["geometry"], row["method"], row["policy"]): row for row in grouped}
    result: dict[str, dict[str, float | None]] = {}
    for geometry in sorted({row["geometry"] for row in grouped}):
        base = by_key.get((geometry, "measurement_ci", "none"), {})
        candidate = by_key.get((geometry, "measurement_ci", "random_peer"), {})
        base_rmse = base.get("pooled_sqrt_mean_RMSE2_m")
        candidate_rmse = candidate.get("pooled_sqrt_mean_RMSE2_m")
        result[geometry] = {"none_rmse_m": base_rmse, "random_ci_rmse_m": candidate_rmse,
                            "improvement_fraction": _delta(base_rmse, candidate_rmse)}
        if base_rmse is not None and candidate_rmse is not None and base_rmse:
            result[geometry]["improvement_fraction"] = float(1.0 - candidate_rmse / base_rmse)
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def _gate_text(gate: dict[str, Any]) -> str:
    passed = bool(gate.get("passed", False))
    label = "PASS" if passed else "FAIL"
    failures = gate.get("failures", []) or []
    lines = [f"**工程门禁：{label}**（gate.lock 的 `passed={str(passed).lower()}`）。"]
    lines.append(f"门禁记录的独立 seed 数为 {gate.get('independent_seeds', EXPECTED_INDEPENDENT_SEEDS)}，每个 seed 包含 {gate.get('within_seed_geometry_blocks', 2)} 个几何 block；这不是 8 个独立 seed。")
    if failures:
        lines.append("失败项：" + "；".join(str(item) for item in failures) + "。")
    else:
        lines.append("失败项：无。")
    return "\n".join(lines)


def _results_markdown(run_id: str, manifest: dict[str, Any], gate: dict[str, Any], summaries: list[dict[str, Any]],
                      grouped: list[dict[str, Any]], changes: list[dict[str, Any]], references: list[dict[str, Any]]) -> str:
    block_keys = {(str(row.get("scene_id")), row.get("seed")) for row in summaries}
    full_keys = {(str(row.get("scene_id")), row.get("seed")) for row in summaries
                 if row.get("status") == "complete" and row.get("steps") == 120}
    seeds = sorted({key[1] for key in block_keys})
    geometries = sorted({_geometry(key[0]) for key in block_keys})
    full_design = len(block_keys) == EXPECTED_BLOCKS and len(full_keys) == len(block_keys) and len(seeds) == EXPECTED_INDEPENDENT_SEEDS
    improvements = _gate_improvements(grouped)
    random_rows = [row for row in grouped if row.get("method") == "measurement_ci" and row.get("policy") == "random_peer"]
    max_requests = max((int(row.get("requests", 0) or 0) for row in random_rows), default=0)
    total_cost = sum(int(row.get("cost", 0) or 0) for row in random_rows)
    healthy = sum(bool(row.get("reference_health_pass")) for row in references)
    ref_rmse = [_finite(row.get(key)) for row in references for key in ("uav0_rmse_m", "uav1_rmse_m")]
    ref_rates = [_finite(row.get(key)) for row in references for key in ("uav0_gnss_acceptance_rate", "uav1_gnss_acceptance_rate")]
    gate_counts = gate.get("positive_seed_counts", {}) if isinstance(gate, dict) else {}

    lines = [
        "# partial GNSS v9 paired replay 报告",
        "",
        f"运行：`{run_id}`；manifest phase：`{manifest.get('phase', 'unknown')}`；本报告只读取已有执行产物。",
        "",
        "## 数据范围和可复核性",
        "",
        f"manifest summary 共 {len(manifest.get('summaries', []))} 条（预期 96 条）；本报告使用 replay 条目 {len(summaries)} 条，覆盖 {len(block_keys)} 个 geometry×seed block、{len(geometries)} 个 geometry、{len(seeds)} 个独立 seed（{', '.join(map(str, seeds))}）。",
        f"完整性检查：{len(full_keys)}/{len(block_keys)} 个 block 为完整 120 步；{('满足' if full_design else '不满足')}“四个独立 seed、两个几何、每块完整 120 步”的设计条件。所有 RMSE/NEES 轨迹均包含第 1–5 步启动阶段；启动后解读从第 6 步开始，未把启动步从全时域指标中删除。",
        "固定导航仅使用本机EKF估计和航点；预注册的 `none/fixed0/fixed1/random_peer` 只使用合法候选，不使用真实定位误差。真值只在事后评价中计算误差。没有学习型Actor、训练或optimizer update。",
        "",
        "## 工程门禁",
        "",
        _gate_text(gate),
        "",
        "## 六个问题的实际回答",
        "",
        "### 1. CI 是否相对 CI None 降低目标定位误差？",
        "",
    ]
    for geometry in sorted(improvements):
        item = improvements[geometry]
        lines.append(f"- {geometry}：CI None pooled RMSE={_fmt(item['none_rmse_m'])} m，random CI pooled RMSE={_fmt(item['random_ci_rmse_m'])} m，改善={_fmt(item['improvement_fraction'] * 100 if item['improvement_fraction'] is not None else None, 2)}%。")
    if not improvements:
        lines.append("- 没有足够的 measurement-CI paired 条目可计算改善。")
    lines += [
        "",
        "### 2. 改善是否跨 seed 稳定？",
        "",
        "门禁使用每个几何 4 个 seed block，而不是把 8 个 geometry×seed block 当作 8 个独立重复。positive-seed 计数如下：",
    ]
    for geometry, metrics in sorted(gate_counts.items()):
        lines.append(f"- {geometry}：" + "，".join(f"{metric}={count}/4" for metric, count in sorted(metrics.items())) + "。")
    if not gate_counts:
        lines.append("- gate.lock 未提供 positive-seed 计数；请以 per_seed_changes.csv 逐行复核。")
    lines += [
        "",
        "### 3. 参考 UAV 的 GNSS/VO 健康和滤波器数值是否保持？",
        "",
        f"CI None 参考健康表覆盖 {len(references)} 个 block，其中 {healthy} 个通过本报告的参考 RMSE、GNSS 接受率、完整性、安全和协方差检查。参考 UAV 的最大 RMSE={_fmt(max(ref_rmse) if ref_rmse else None)} m，最小 GNSS 接受率={_fmt(min(ref_rates) if ref_rates else None, 3)}。详细实际值见 reference_quality.csv。",
        "",
        "### 4. 请求、到达、吸收和成本是多少？",
        "",
        f"random CI 的每几何四个案例合计请求最多 {max_requests} 次；每案例最多29个窗口、预算30次。两几何合计字节等价成本为 {total_cost}（每请求512字节等价负载）。requests、arrivals、attempted、accepted、effective、w1、rejections、cost 全部列在grouped.csv；None广播和队列在回放中保持不变，选择只计请求开销。",
        "",
        "### 5. NIS 能否作为统计一致性或卡方显著性证据？",
        "",
        "不能。CI 的 NIS 是带未知交叉协方差时的保守创新上界，不是 χ² 校准统计量；本实验的 gate 是工程 admission guard，不是 formal consistency、显著性检验或 generalization proof。若 gate 为 FAIL，不能声称 CI 可靠；若 gate 为 PASS，也只能声称满足预注册工程门限。",
        "",
        "### 6. 能否据此宣称分布式闭环、泛化或 RL 成功？",
        "",
        "不能。`joint_reference`使用当前私有IMU/VO/GNSS和交叉协方差，是具有额外信息的中央参考。此表为固定None广播/队列的配对回放，没有协同闭环验证；gate通过后仍须单独检查真实队列闭环。固定导航和非学习邻居选择均在使用，但没有RL或optimizer update，不能给出训练或泛化结论。",
        "",
        "## 分组结果（pooled RMSE 为 sqrt(mean(block RMSE²))）",
        "",
        "`grouped.csv` 保留所有方法和策略；下面列出同样的实际数值。NIS 的空值表示该组没有有效融合事件。",
        "",
        "| method | policy | geometry | pooled RMSE (m) | NEES | NIS | requests | arrivals | attempted | accepted | effective | w1 | cost |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append("| {method} | {policy} | {geometry} | {rmse} | {nees} | {nis} | {requests} | {arrivals} | {attempted} | {accepted} | {effective} | {w1} | {cost} |".format(
            method=row["method"], policy=row["policy"], geometry=row["geometry"],
            rmse=_fmt(row.get("pooled_sqrt_mean_RMSE2_m")), nees=_fmt(row.get("NEES")), nis=_fmt(row.get("NIS")),
            requests=row.get("requests", 0), arrivals=row.get("arrivals", 0), attempted=row.get("attempted", 0),
            accepted=row.get("accepted", 0), effective=_fmt(row.get("effective"), 3), w1=row.get("w1", 0), cost=row.get("cost", 0)))
    lines += [
        "",
        "## 解释边界",
        "",
        "本文件的结果只描述这 2 个几何、4 个独立 seed、120 步 paired replay；它不把 geometry×seed block 数误报成独立样本，不把中心化 joint reference 当作分布式候选，也不把 CI NIS 上界当作 χ² 统计量。所有失败门限保留在 gate.lock，逐 seed 差值保留在 per_seed_changes.csv，参考健康保留在 reference_quality.csv。",
    ]
    return "\n".join(lines) + "\n"


def report(run_id: str) -> dict[str, Any]:
    """Create a read-only-derived report for one completed paired run."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("unsafe run id")
    data = BASE / run_id
    manifest_path = data / "manifest.json"
    gate_path = data / "gate.lock.json"
    if not data.is_dir():
        raise FileNotFoundError(str(data))
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    if not gate_path.is_file():
        raise FileNotFoundError(str(gate_path))
    manifest = v7.old.read_json(manifest_path)
    gate = v7.old.read_json(gate_path)
    raw_summaries = manifest.get("summaries", [])
    if not isinstance(raw_summaries, list):
        raise ValueError("manifest summaries must be a list")
    # Paired v9 summaries are replay phase entries.  Keep a phase-less row for
    # compatibility with early frozen manifests, but never mix closed-loop
    # results into this report.
    summaries = [row for row in raw_summaries if row.get("phase", "replay") == "replay"]
    grouped = _summary_groups(summaries)
    changes = _per_seed_changes(summaries)
    references = _reference_quality(summaries, gate)

    # These directories are intentionally created only after every source file
    # has been read and validated.  Both are exclusive to prevent reruns from
    # overwriting a prior report; execution artifacts are never touched.
    data_report = BASE / f"report_{run_id}"
    logs_report = LOGS / f"report_{run_id}"
    data_report.mkdir(exist_ok=False)
    logs_report.mkdir(exist_ok=False)

    grouped_path = logs_report / "grouped.csv"
    changes_path = logs_report / "per_seed_changes.csv"
    references_path = logs_report / "reference_quality.csv"
    aggregate_path = data_report / "aggregate.json"
    results_path = data_report / "results.md"
    _write_csv(grouped_path, grouped)
    _write_csv(changes_path, changes)
    _write_csv(references_path, references)
    v7.old.write_json(aggregate_path, {
        "run_id": run_id,
        "summary_count_manifest": len(raw_summaries),
        "summary_count_replay": len(summaries),
        "expected_summary_count": EXPECTED_SUMMARIES,
        "grouped": grouped,
        "gate": gate,
    })
    results_path.write_text(_results_markdown(run_id, manifest, gate, summaries, grouped, changes, references), encoding="utf-8")

    result = {
        "aggregate": grouped,
        "gate": {"passed": bool(gate.get("passed", False)), "failures": gate.get("failures", [])},
        "paths": {
            "data_report": str(data_report),
            "logs_report": str(logs_report),
            "aggregate": str(aggregate_path),
            "grouped": str(grouped_path),
            "per_seed_changes": str(changes_path),
            "reference_quality": str(references_path),
            "results": str(results_path),
            "plots": None,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="paired_001")
    args = parser.parse_args()
    print(v7.old.encode(report(args.run_id)))


if __name__ == "__main__":
    main()
