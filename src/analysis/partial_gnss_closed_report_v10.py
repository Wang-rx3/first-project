"""Read-only v10 closed-loop report.

The v10 runner owns execution and candidate locking.  This module only checks
the frozen inputs and phase manifests, then derives tables and plots from the
already written summaries.  In particular, confirmation replay is descriptive
evidence and can never invoke candidate selection again.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..experiments import partial_gnss_ci_validation_v10 as v10


ROOT = v10.ROOT
STEM = v10.STEM
BASE = v10.BASE
LOGS = v10.LOGS
PLOTS = ROOT / "plots/preliminary" / STEM

PHASES = ("develop", "explore", "confirm", "confirmation_replay")
PHASE_LABELS = {
    "develop": "development_replay",
    "explore": "exploration",
    "confirm": "confirmation",
    "confirmation_replay": "confirmation_replay",
}
METRICS = ("uav2_absolute_rmse_m", "cluster_relative_rmse_m")
AUDIT_FIELDS = (
    "native_live_queue",
    "actual_message_arrival_count",
    "request_overhead_packet_load",
    "request_overhead_bytes_equivalent512",
    "controller_command_error_max",
    "sender_payload_state_error_max",
    "sender_payload_covariance_error_max",
    "feedback_update_pairs",
    "feedback_command_change_pairs",
    "navigation_truth_used",
    "no_fixed_queue",
    "native_arrival_statuses_only",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [x for value in values if (x := _finite(value)) is not None]
    return float(np.mean(numbers)) if numbers else None


def _p95(values: Iterable[Any]) -> float | None:
    numbers = [x for value in values if (x := _finite(value)) is not None]
    return float(np.percentile(numbers, 95)) if numbers else None


def _pooled(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [x for row in rows if (x := _finite(row.get(key))) is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    n, d = _finite(numerator), _finite(denominator)
    return None if n is None or d is None or abs(d) < 1e-15 else float(n / d)


def _geometry(scene_id: Any) -> str:
    text = str(scene_id)
    if "noncollinear" in text:
        return "noncollinear"
    if "collinear" in text:
        return "collinear"
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_jsonable)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(str(path))
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})


def _phase_manifest(batch_dir: Path, phase: str) -> dict[str, Any] | None:
    path = batch_dir / phase / "manifest.json"
    if not path.is_file():
        return None
    # The runner's checker is the canonical artifact proof.  It verifies every
    # data/log artifact recorded by the phase before this report consumes it.
    return v10.checked_manifest(path)


def _failed_row(phase: str, failure: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "status": "failed",
        "failed_prefix": True,
        "failed_prefix_marker": "FAILED_PREFIX",
        "full_horizon_eligible": False,
        "pool_eligible": False,
        "failure": dict(failure),
    }
    for key in METRICS:
        row[key] = None
        row["prefix_" + key] = None
    for key in AUDIT_FIELDS:
        row[key] = None
    return row


def _normal_summary(phase: str, item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item)
    row["phase"] = phase
    row.setdefault("geometry", _geometry(row.get("scene_id")))
    row.setdefault("failed_prefix", False)
    complete = row.get("status") == "complete" and _int(row.get("steps")) == 120
    eligible = bool(row.get("full_horizon_eligible", complete)) and complete
    row["pool_eligible"] = bool(eligible)
    row["failed_prefix"] = not eligible
    row["failed_prefix_marker"] = "FAILED_PREFIX" if not eligible else ""
    for metric in METRICS:
        row.setdefault("prefix_" + metric, row.get(metric) if not eligible else None)
        if not eligible:
            # A censored prefix is visible in full_summary, but remains absent
            # from every 120-step pooled calculation.
            row[metric] = None
    audit = row.get("execution_audit")
    if not isinstance(audit, Mapping):
        audit = {}
    for key in AUDIT_FIELDS:
        row[key] = audit.get(key)
    row["actual_native_arrivals"] = audit.get("actual_message_arrival_count")
    row["actual_message_arrival_count"] = audit.get("actual_message_arrival_count")
    row["sender_payload_cov_error_max"] = audit.get("sender_payload_covariance_error_max")
    row["request_overhead_packet_load"] = audit.get("request_overhead_packet_load")
    row["request_overhead_bytes_equivalent512"] = audit.get("request_overhead_bytes_equivalent512")
    return row


def _phase_rows(phase: str, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries = manifest.get("summaries", [])
    if isinstance(summaries, list):
        rows.extend(_normal_summary(phase, item) for item in summaries if isinstance(item, Mapping))
    failures = manifest.get("failures", [])
    if isinstance(failures, list):
        rows.extend(_failed_row(phase, item) for item in failures if isinstance(item, Mapping))
    return rows


def _timings(row: Mapping[str, Any]) -> list[float]:
    samples = row.get("kernel_timing_samples_ms")
    if isinstance(samples, list):
        values = [x for item in samples if (x := _finite(item)) is not None]
        if values:
            return values
    value = _finite(row.get("kernel_time_ms_mean"))
    return [value] if value is not None else []


def _audit_sum(rows: list[Mapping[str, Any]], key: str) -> float | int | None:
    values = [_finite(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values))


def _grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("phase", "")), str(row.get("method", "")),
                str(row.get("policy", "")), str(row.get("geometry", _geometry(row.get("scene_id")))))] .append(row)
    result: list[dict[str, Any]] = []
    for (phase, method, policy, geometry), all_items in sorted(groups.items()):
        full = [row for row in all_items if row.get("pool_eligible")]
        timings = [value for row in full for value in _timings(row)]
        requests = sum(_int(row.get("uwb_requested")) for row in all_items)
        arrivals = sum(_int(row.get("uwb_message_arrived")) for row in all_items)
        attempts = sum(_int(row.get("uwb_attempted")) for row in all_items)
        absorbed = sum(_int(row.get("uwb_accepted")) for row in all_items)
        costs = sum(_int(row.get("measurement_bytes_equivalent")) for row in all_items)
        parking_values = [bool(row.get("uav2_strict_parking")) for row in full if row.get("uav2_strict_parking") is not None]
        finals = [row.get("uav2_final_position_covariance_trace_m2", row.get("target_position_trace_final_m2")) for row in full]
        result.append({
            "phase": phase,
            "method": method,
            "policy": policy,
            "geometry": geometry,
            "blocks": len(all_items),
            "full_horizon_blocks": len(full),
            "failed_prefix_blocks": len(all_items) - len(full),
            "pooled_absolute_rmse_m": _pooled(full, "uav2_absolute_rmse_m"),
            "pooled_relative_rmse_m": _pooled(full, "cluster_relative_rmse_m"),
            "pooled_abs_rmse_m": _pooled(full, "uav2_absolute_rmse_m"),
            "pooled_rel_rmse_m": _pooled(full, "cluster_relative_rmse_m"),
            "pooled_sqrt_mean_RMSE2_m": _pooled(full, "uav2_absolute_rmse_m"),
            "relative_pooled_sqrt_mean_RMSE2_m": _pooled(full, "cluster_relative_rmse_m"),
            "NEES": _mean(row.get("uav2_position_nees_mean") for row in full),
            "NIS": _mean(row.get("uwb_nis_mean") for row in full),
            "requests": requests,
            "arrivals": arrivals,
            "attempts": attempts,
            "absorbed": absorbed,
            "accepted": absorbed,
            "effective": _ratio(absorbed, attempts) if attempts else 0.0,
            "effective_fusion_rate": _ratio(absorbed, attempts) if attempts else 0.0,
            "cost_byte_equivalent": costs,
            "cost_bytes_equivalent": costs,
            "measurement_bytes_equivalent": costs,
            "request_overhead_packet_load": _audit_sum(full, "request_overhead_packet_load"),
            "request_overhead_bytes_equivalent512": _audit_sum(full, "request_overhead_bytes_equivalent512"),
            "actual_native_arrivals": _audit_sum(full, "actual_message_arrival_count"),
            "actual_message_arrival_count": _audit_sum(full, "actual_message_arrival_count"),
            "kernel_time_ms_mean": _mean(timings),
            "kernel_time_ms_p95": _p95(timings),
            "kernel_mean_ms": _mean(timings),
            "kernel_p95_ms": _p95(timings),
            "kernel_timing_samples": len(timings),
            "final_P_mean": _mean(finals),
            "finalPmean": _mean(finals),
            "final_position_covariance_trace_mean_m2": _mean(finals),
            "collision_count": sum(_int(row.get("collision_count")) for row in all_items),
            "boundary_projections": sum(_int(row.get("boundary_projections")) for row in all_items),
            "safety_projections": sum(_int(row.get("safety_projections")) for row in all_items),
            "strict_parking_count": sum(parking_values) if parking_values else None,
            "parking_success_count": sum(parking_values) if parking_values else None,
            "strict_parking_observed": len(parking_values),
            "final_goal_distance_mean_m": _mean(row.get("uav2_final_goal_distance_m") for row in full),
            "final_speed_mean_mps": _mean(row.get("uav2_final_speed_mps") for row in full),
            "native_live_queue_count": sum(bool(row.get("native_live_queue")) for row in all_items),
            "feedback_update_pairs": _audit_sum(full, "feedback_update_pairs"),
            "feedback_command_change_pairs": _audit_sum(full, "feedback_command_change_pairs"),
            "feedback_update_pairs_sum": _audit_sum(full, "feedback_update_pairs"),
            "feedback_command_change_pairs_sum": _audit_sum(full, "feedback_command_change_pairs"),
            "controller_command_error_max": max((_finite(row.get("controller_command_error_max")) or 0.0 for row in full), default=None),
            "sender_payload_state_error_max": max((_finite(row.get("sender_payload_state_error_max")) or 0.0 for row in full), default=None),
            "sender_payload_covariance_error_max": max((_finite(row.get("sender_payload_covariance_error_max")) or 0.0 for row in full), default=None),
            "scene_ids": sorted({str(row.get("scene_id")) for row in all_items if row.get("scene_id") is not None}),
        })
    return result


def _improvement(after: Any, before: Any) -> float | None:
    a, b = _finite(after), _finite(before)
    return None if a is None or b is None or b <= 1e-15 else float(1.0 - a / b)


def _change_row(phase: str, comparison: str, scene: Any, seed: Any,
                old: Mapping[str, Any] | None, new: Mapping[str, Any] | None,
                old_label: str, new_label: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "comparison": comparison,
        "geometry": _geometry(scene),
        "scene_id": scene,
        "seed": seed,
        "old_label": old_label,
        "new_label": new_label,
        "old_status": old.get("status") if old else None,
        "new_status": new.get("status") if new else None,
        "old_full_horizon_eligible": old.get("pool_eligible") if old else False,
        "new_full_horizon_eligible": new.get("pool_eligible") if new else False,
    }
    for metric, short in ((METRICS[0], "absolute_rmse_m"), (METRICS[1], "relative_rmse_m")):
        before = old.get(metric) if old else None
        after = new.get(metric) if new else None
        row["old_" + short] = before
        row["new_" + short] = after
        row["delta_" + short] = None if before is None or after is None else float(after) - float(before)
        row["improvement_fraction_" + short] = _improvement(after, before)
    return row


def _paired_changes(rows: list[dict[str, Any]], selection: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_key = {(str(row.get("phase")), str(row.get("scene_id")), row.get("seed"), str(row.get("method")), str(row.get("policy"))): row for row in rows}
    # Physical live runs: selected method random_peer against the same-method
    # None episode, separately for every geometry and seed.
    selected = str(selection.get("decision", {}).get("selected_method")) if selection else ""
    for phase in ("explore", "confirm"):
        keys = sorted({(str(row.get("scene_id")), row.get("seed")) for row in rows if row.get("phase") == phase})
        for scene, seed in keys:
            old = by_key.get((phase, scene, seed, selected, "none"))
            new = by_key.get((phase, scene, seed, selected, "random_peer"))
            result.append(_change_row(phase, "random_selected_vs_none", scene, seed, old, new, "none", "random_peer_selected"))
    # Development and confirmation replay are paired method comparisons.  The
    # latter is reported only; this function intentionally never selects a
    # candidate from it.
    for phase in ("develop", "confirmation_replay"):
        keys = sorted({(str(row.get("scene_id")), row.get("seed")) for row in rows if row.get("phase") == phase})
        for scene, seed in keys:
            old = by_key.get((phase, scene, seed, "original_ci", "random_peer"))
            new = by_key.get((phase, scene, seed, "refined_ci", "random_peer"))
            result.append(_change_row(phase, "refined_vs_original_random", scene, seed, old, new, "original_ci", "refined_ci"))
    return result


def _selection_lock(batch_dir: Path, develop: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    path = batch_dir / "candidate_selection.lock.json"
    problems: list[str] = []
    if not path.is_file():
        return None, ["candidate_selection.lock.json missing"]
    lock = _read_json(path)
    if develop is not None:
        manifest_path = batch_dir / "develop" / "manifest.json"
        expected = lock.get("development_manifest_sha256")
        if expected and expected != _sha256(manifest_path):
            problems.append("candidate lock development manifest hash mismatch")
    if not lock.get("decision"):
        problems.append("candidate lock has no decision")
    return lock, problems


def _ci_assessment(rows: list[dict[str, Any]], selection: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> dict[str, Any]:
    replay = [row for row in rows if row.get("phase") == "confirmation_replay" and row.get("pool_eligible") and row.get("policy") == "random_peer"]
    development_beneficial = bool(selection and selection.get("decision", {}).get("refined_development_benefit"))
    by_key = {(str(row.get("scene_id")), row.get("seed"), str(row.get("method"))): row for row in replay}
    pooled: dict[str, Any] = {}
    for metric in METRICS:
        old = [row for (scene, seed, method), row in by_key.items() if method == "original_ci"]
        new = [row for (scene, seed, method), row in by_key.items() if method == "refined_ci"]
        pooled[metric] = {"original": _pooled(old, metric), "refined": _pooled(new, metric),
                          "improvement_fraction": _improvement(_pooled(new, metric), _pooled(old, metric))}
    seed_counts = {}
    all_pairs = [(by_key.get((scene, seed, "original_ci")), by_key.get((scene, seed, "refined_ci")))
                 for scene in spec["geometry_ids"] for seed in spec["confirmation_seeds"]]
    complete = len(all_pairs) == 8 and all(old and new for old, new in all_pairs)
    no_harm = complete and all(_improvement(new[metric], old[metric]) is not None
        and _improvement(new[metric], old[metric]) >= -spec["selection"]["maximum_case_degradation_fraction"]
        for old, new in all_pairs for metric in METRICS)
    for metric in METRICS:
        count = 0
        for seed in spec["confirmation_seeds"]:
            old = [row for (scene, s, method), row in by_key.items() if s == seed and method == "original_ci"]
            new = [row for (scene, s, method), row in by_key.items() if s == seed and method == "refined_ci"]
            gain = _improvement(_pooled(new, metric), _pooled(old, metric))
            count += int(len(old) == 2 and len(new) == 2 and gain is not None and gain > 0)
        seed_counts[metric] = count
    fractional = all(item["improvement_fraction"] is not None and item["improvement_fraction"] >= spec["selection"]["minimum_fractional_improvement"] for item in pooled.values())
    absolute = pooled[METRICS[0]]["original"] is not None and pooled[METRICS[0]]["refined"] is not None and pooled[METRICS[0]]["original"]-pooled[METRICS[0]]["refined"] >= spec["selection"]["absolute_rmse_min_gain_m"]
    repeatable = all(count >= spec["selection"]["positive_independent_seeds"] for count in seed_counts.values())
    passed = development_beneficial and complete and no_harm and fractional and absolute and repeatable
    return {"status": "pass_limited_scope" if passed else "not_passed", "passed": passed,
            "reason": "frozen development and independent replay benefit rules passed" if passed else "refined improvement did not meet all frozen benefit rules",
            "development_refined_beneficial": development_beneficial,
            "confirmation_replay_pooled": pooled, "confirmation_replay_seed_pools": seed_counts,
            "complete_eight_pairs": complete, "every_case_no_harm": no_harm,
            "criteria": {**spec["selection"], "independent_unit": "four seeds, each pooled over two geometries"},
            "selection_source": "development candidate_selection.lock only; confirmation replay never reselects"}



def _closed_loop_assessment(manifests: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    manifest = manifests.get("confirm")
    if manifest is None:
        return {"status": "blocked", "passed": False, "reason": "confirm/manifest.json missing"}
    performance = manifest.get("performance", {})
    passed = bool(performance.get("passed"))
    failures = performance.get("failures", []) if isinstance(performance, Mapping) else []
    executed = bool(manifest.get("summaries")) or _int(manifest.get("physical_steps")) > 0
    if passed:
        status = "PASS"
        reason = "confirm manifest performance.passed=true"
    elif executed:
        status = "not_passed"
        reason = "live confirmation executed but performance gate failed: " + "; ".join(map(str, failures))
    else:
        status = "blocked"
        reason = "confirmation was not executed"
    return {"status": status, "passed": passed, "reason": reason, "failures": failures,
            "performance": performance, "scope": "bounded native closed-loop confirmation; no general consistency or generalization claim"}


def _learning_assessment(closed: Mapping[str, Any]) -> dict[str, Any]:
    if closed.get("passed"):
        return {"status": "conditional_pass", "training_ready": False,
                "reason": "closed-loop confirmation passed; reward/observation/action training validation remains required"}
    return {"status": "not_passed", "training_ready": False,
            "reason": "confirmation performance did not pass; learning prevalidation cannot pass"}


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _results_markdown(batch: str, manifests: Mapping[str, Mapping[str, Any] | None], rows: list[dict[str, Any]], grouped: list[dict[str, Any]], changes: list[dict[str, Any]], selection: Mapping[str, Any] | None, selection_problems: list[str], ci: Mapping[str, Any], closed: Mapping[str, Any], learning: Mapping[str, Any], blocked: list[str]) -> str:
    full = [row for row in rows if row.get("pool_eligible")]
    geometries = sorted({_geometry(row.get("scene_id")) for row in rows if row.get("scene_id") is not None})
    seeds = sorted({row.get("seed") for row in rows if row.get("seed") is not None})
    random_groups = [row for row in grouped if row.get("policy") == "random_peer"]
    total_actual_arrivals = sum(_int(row.get("actual_message_arrival_count")) for row in full if row.get("phase") in ("explore", "confirm"))
    total_requests = sum(_int(row.get("uwb_requested")) for row in full if row.get("phase") in ("explore", "confirm"))
    total_absorbed = sum(_int(row.get("uwb_accepted")) for row in full if row.get("phase") in ("explore", "confirm"))
    total_attempts = sum(_int(row.get("uwb_attempted")) for row in full if row.get("phase") in ("explore", "confirm"))
    lines = [
        "# partial GNSS v10 闭环报告", "",
        f"批次：`{batch}`。本报告只读取已完成阶段的 manifest 和摘要；四个阶段覆盖 {len(geometries)} 个几何、{len(seeds)} 个独立 seed（每个几何内四个 seed，不把重复几何 block 计作八个独立 seed）。",
        f"全时域可汇总条目：{len(full)}；被截断/失败前缀条目：{sum(not row.get('pool_eligible') for row in rows)}。前缀值只在 `full_summary.csv` 标记，未与 120 步条目混合计算。",
        "", "## 阶段和证据完整性", "",
        "|阶段|manifest|摘要条数|门禁/状态|", "|---|---|---:|---|",
    ]
    for phase in PHASES:
        manifest = manifests.get(phase)
        if manifest is None:
            lines.append(f"|{PHASE_LABELS[phase]}|缺失|0|BLOCKED：缺少阶段产物|")
        else:
            status = manifest.get("performance", manifest.get("safety", manifest.get("selection", "已读取")))
            if isinstance(status, Mapping):
                status = "passed=" + str(status.get("passed", status.get("closed_loop_execution_allowed", "记录")))
            lines.append(f"|{PHASE_LABELS[phase]}|已通过 SHA 校验|{len(manifest.get('summaries', []))}|{status}|")
    if blocked or selection_problems:
        lines += ["", "阻塞项：" + "；".join(blocked + selection_problems) + "。"]
    lines += [
        "", "## 七个问题的实际回答", "",
        "### 1. 冻结设计、阶段完整性和汇总范围如何？", "",
        f"冻结配置、源文件哈希和 protected history 由 v10.verify() 校验；本批次可见几何为 {', '.join(geometries) or '—'}，seed 为 {', '.join(map(str, seeds)) or '—'}。缺失阶段保持 BLOCKED，报告没有补造数据。",
        "", "### 2. refined CI 的改善是否通过预注册标准？", "",
        f"发展阶段候选锁的 refined benefit={ci.get('development_refined_beneficial')}；确认回放状态为 **{ci.get('status')}**。确认回放 pooled 改善如下：" ,
        "|指标|original CI|refined CI|改善|", "|---|---:|---:|---:|",
    ]
    for metric, label in ((METRICS[0], "绝对 RMSE"), (METRICS[1], "相对 RMSE")):
        item = ci.get("confirmation_replay_pooled", {}).get(metric, {})
        lines.append(f"|{label}|{_fmt(item.get('original'))}|{_fmt(item.get('refined'))}|{_fmt(100*item.get('improvement_fraction'), 2)}%|")
    lines += [
        f"只有发展阶段 refined 有利、确认回放两项 pooled 至少 1%、绝对改善至少 0.01 m、至少 3/4 独立种子在两几何汇总后为正、每个场景均不恶化超过 5% 时，才给出 limited-scope 通过；当前结论：{ci.get('reason')}。",
        "", "### 3. 原生分布式闭环是否通过？", "",
        f"结论：**{closed.get('status')}**。{closed.get('reason')}。这只覆盖 bounded native closed-loop confirmation，不推出一般非线性时延下的协方差一致性或泛化。",
        "", "### 4. 请求、到达、吸收和通信成本怎样解释？", "",
        f"全时域条目合计 request={total_requests}、absorbed={total_absorbed}、effective={_fmt(_ratio(total_absorbed, total_attempts), 3)}、native actual arrivals={total_actual_arrivals}。native communication 是 live periodic broadcast 加 measurement request 的 512-byte-equivalent load，并从已到达的 passive UWB snapshot 中选择；缓存可用次数不等于实际 arrival，也不等于硬件 request-response serialized byte cost。",
        "", "### 5. 数值、安全和停车结果如何？", "",
        "分组表同时报告协方差、边界/安全/碰撞事件和严格停车计数。严格停车定义为实际距离 ≤1 m、实际速度 ≤0.5 m/s 并保持 5 步；120 步完成不等于停车。最终 goal distance 与 final velocity 分开列出。",
        "", "### 6. NIS 和 CI 线性界能否支持统计一致性结论？", "",
        "不能。NIS 在未知交叉协方差条件下是保守创新上界；固定 H 线性条件下的 Young 型数值界只支持本实验工程 admission 检查，不是任意非线性时延、χ² 校准或 formal consistency 证明。",
        "", "### 7. 是否已经证明 RL 学习或泛化？", "",
        f"没有。learning prevalidation={learning.get('status')}；`training_ready=false`。固定导航只使用各机自身估计、速度和目标航点，真值仅离线评价；没有 Actor、RL optimizer 或训练。后续仍需预注册 reward/observation/action 的独立训练与验证。",
        "", "## 分组事实表", "",
        "|phase|method|policy|geometry|blocks/full|abs RMSE|rel RMSE|NEES|NIS|requests/arrivals/attempts/absorbed|effective|cost byte-eq|kernel mean/p95 ms|final P mean|native arrivals|parking|feedback pairs|",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in grouped:
        lines.append("|{phase}|{method}|{policy}|{geometry}|{blocks}/{full_horizon_blocks}|{a}|{r}|{nees}|{nis}|{requests}/{arrivals}/{attempts}/{absorbed}|{eff}|{cost}|{km}/{kp}|{fp}|{na}|{park}|{fb}|".format(
            phase=item.get("phase"), method=item.get("method"), policy=item.get("policy"), geometry=item.get("geometry"),
            blocks=item.get("blocks"), full_horizon_blocks=item.get("full_horizon_blocks"), a=_fmt(item.get("pooled_absolute_rmse_m")),
            r=_fmt(item.get("pooled_relative_rmse_m")), nees=_fmt(item.get("NEES")), nis=_fmt(item.get("NIS")),
            requests=item.get("requests"), arrivals=item.get("arrivals"), attempts=item.get("attempts"), absorbed=item.get("absorbed"),
            eff=_fmt(item.get("effective"), 3), cost=item.get("cost_byte_equivalent"), km=_fmt(item.get("kernel_time_ms_mean")), kp=_fmt(item.get("kernel_time_ms_p95")),
            fp=_fmt(item.get("final_P_mean")), na=_fmt(item.get("actual_native_arrivals"), 1), park=item.get("strict_parking_count"),
            fb=_fmt(item.get("feedback_update_pairs"), 1)))
    lines += ["", "## 三个结论", "", f"1. CI 改善：{ci.get('status')}（范围限于冻结发展 paired replay 与独立确认 replay）。", f"2. 原生闭环：{closed.get('status')}（门禁依据 confirm manifest.performance.passed，失败原因已保留）。", f"3. 学习预验证：{learning.get('status')}；不能作为训练成功或泛化结论。", ""]
    return "\n".join(lines)


def _draw_plots(plot_dir: Path, rows: list[dict[str, Any]], changes: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    geometries = sorted({_geometry(row.get("scene_id")) for row in rows if row.get("scene_id") is not None}) or ["no-data"]
    seeds = sorted({row.get("seed") for row in rows if row.get("seed") is not None})
    fig, axes = plt.subplots(1, len(geometries), figsize=(6 * len(geometries), 4), squeeze=False)
    for axis, geometry in zip(axes.flat, geometries):
        subset = [row for row in rows if _geometry(row.get("scene_id")) == geometry and row.get("phase") in ("explore", "confirm") and row.get("pool_eligible")]
        for phase, marker in (("explore", "o"), ("confirm", "s")):
            for policy, linestyle in (("none", "--"), ("random_peer", "-")):
                vals = []
                for seed in seeds:
                    candidates = [row for row in subset if row.get("phase") == phase and row.get("seed") == seed and row.get("policy") == policy]
                    vals.append(_finite(candidates[0].get(METRICS[0])) if candidates else np.nan)
                if any(np.isfinite(value) for value in vals):
                    axis.plot(range(len(seeds)), vals, marker=marker, linestyle=linestyle, label=f"{phase}/{policy}")
        axis.set_title(geometry); axis.set_xlabel("seed index"); axis.set_ylabel("absolute RMSE (m)"); axis.grid(alpha=.25)
    if any(axis.lines for axis in axes.flat):
        axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = plot_dir / f"explore_vs_confirmation_absolute_rmse.{ext}"
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    plt.close(fig)

    fig, axes = plt.subplots(1, len(geometries), figsize=(6 * len(geometries), 4), squeeze=False)
    for axis, geometry in zip(axes.flat, geometries):
        subset = [row for row in changes if row.get("comparison") == "refined_vs_original_random" and row.get("geometry") == geometry]
        for phase, marker in (("develop", "o"), ("confirmation_replay", "s")):
            values = [_finite(row.get("improvement_fraction_absolute_rmse_m")) for row in subset if row.get("phase") == phase]
            values = [value for value in values if value is not None]
            if values:
                axis.plot(range(len(values)), values, marker=marker, label=phase)
        axis.axhline(0.01, color="k", linestyle=":", linewidth=1)
        axis.set_title(geometry); axis.set_xlabel("paired seed index"); axis.set_ylabel("refined improvement"); axis.grid(alpha=.25)
    if any(axis.lines for axis in axes.flat):
        axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = plot_dir / f"development_vs_confirmation_replay_ci.{ext}"
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    plt.close(fig)
    return paths


def report(batch_id: str) -> dict[str, Any]:
    """Build one exclusive report from the frozen v10 batch evidence."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", batch_id):
        raise ValueError("unsafe batch id")
    # This checks frozen design, protected history/source hashes, and the unit
    # release before any phase data is interpreted.
    spec = v10.verify()
    batch_dir = BASE / "batches" / batch_id
    if not batch_dir.is_dir():
        raise FileNotFoundError(str(batch_dir))
    manifests = {phase: _phase_manifest(batch_dir, phase) for phase in PHASES}
    missing = [phase for phase, manifest in manifests.items() if manifest is None]
    develop = manifests.get("develop")
    selection, selection_problems = _selection_lock(batch_dir, develop)
    rows = [row for phase in PHASES if manifests.get(phase) is not None for row in _phase_rows(phase, manifests[phase])]  # type: ignore[arg-type]
    grouped = _grouped(rows)
    changes = _paired_changes(rows, selection)
    ci = _ci_assessment(rows, selection, spec)
    closed = _closed_loop_assessment(manifests)
    learning = _learning_assessment(closed)
    blocked = [f"{phase} manifest missing" for phase in missing]
    if selection_problems:
        blocked.extend(selection_problems)

    data_report = batch_dir / "report"
    logs_report = LOGS / batch_id / "report"
    plot_report = PLOTS / batch_id
    data_report.mkdir(exist_ok=False)
    logs_report.mkdir(parents=True, exist_ok=False)
    plot_report.mkdir(parents=True, exist_ok=False)

    full_path = logs_report / "full_summary.csv"
    grouped_path = logs_report / "grouped.csv"
    changes_path = logs_report / "paired_changes.csv"
    _write_csv(full_path, rows)
    _write_csv(grouped_path, grouped)
    _write_csv(changes_path, changes)
    plot_paths = _draw_plots(plot_report, rows, changes)
    results_path = data_report / "results.md"
    results_path.write_text(_results_markdown(batch_id, manifests, rows, grouped, changes, selection, selection_problems, ci, closed, learning, blocked), encoding="utf-8")
    aggregate = {
        "batch_id": batch_id,
        "stem": STEM,
        "phase_manifests": {phase: manifest is not None for phase, manifest in manifests.items()},
        "missing_phases": missing,
        "blocked": blocked,
        "candidate_selection_lock": selection,
        "candidate_selection_problems": selection_problems,
        "ci_improvement": ci,
        "closed_loop": closed,
        "learning_prevalidation": learning,
        "training_ready": False,
        "summary_count": len(rows),
        "full_horizon_summary_count": sum(bool(row.get("pool_eligible")) for row in rows),
        "grouped": grouped,
        "paths": {"full_summary": str(full_path), "grouped": str(grouped_path), "paired_changes": str(changes_path), "results": str(results_path), "plots": plot_paths},
        "scope": "old explore and new confirmation each use four seeds; geometry blocks are not independent seed replications",
    }
    aggregate_path = data_report / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    return {"status": "blocked" if blocked else "complete", "aggregate": grouped,
            "ci_improvement": ci, "closed_loop": closed, "learning_prevalidation": learning,
            "training_ready": False, "blocked": blocked,
            "paths": {"data_report": str(data_report), "logs_report": str(logs_report), "aggregate": str(aggregate_path),
                      "full_summary": str(full_path), "grouped": str(grouped_path), "paired_changes": str(changes_path),
                      "results": str(results_path), "plots": plot_paths}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="research_001")
    args = parser.parse_args()
    print(json.dumps(report(args.batch_id), ensure_ascii=False, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
