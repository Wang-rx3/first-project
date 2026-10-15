"""Read-only, failure-aware report for the V13 joint GNSS/RF study.

The module consumes manifests and JSON points already produced by
``neighbor_selection_validation_v13``.  It never imports a simulator, filter,
navigation controller, optimiser, or training entry point.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .neighbor_selection_gate_v11 import correlation as _v11_correlation

METRICS = (
    "uav2_absolute_rmse_m", "cluster_relative_rmse_m",
    "uav2_position_nees_mean", "uwb_nis_mean", "uwb_requested",
    "uwb_attempted", "uwb_accepted", "measurement_bytes_equivalent",
    "uav2_strict_parking", "uav2_final_localization_error_m",
)
RMSE_METRICS = ("uav2_absolute_rmse_m", "cluster_relative_rmse_m")
POLICIES = ("none", "random_peer", "greedy", "v13_greedy")
PEER_LABELS = ("uav_00", "uav_01")
MEANINGFUL_NORM = 0.01


def correlation(x: Sequence[Any], y: Sequence[Any], rank: bool = False) -> float | None:
    """Use the frozen V11 Pearson/Spearman implementation with null filtering."""
    if len(x) != len(y) or len(x) < 3:
        return None
    xx = [_number(value) for value in x]
    yy = [_number(value) for value in y]
    if any(value is None for value in (*xx, *yy)):
        return None
    return _v11_correlation(xx, yy, rank)


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _complete(row: Mapping[str, Any]) -> bool:
    return bool(row.get("full_horizon_eligible")) and all(
        _number(row.get(name)) is not None for name in RMSE_METRICS
    )


def _pool(values: Sequence[Any]) -> float | None:
    numbers = [_number(value) for value in values]
    if not numbers or any(value is None for value in numbers):
        return None
    return float(math.sqrt(sum(value * value for value in numbers) / len(numbers)))


def _mean(values: Sequence[Any], *, allow_missing: bool = False) -> float | None:
    numbers = [_number(value) for value in values]
    if allow_missing:
        numbers = [value for value in numbers if value is not None]
    elif any(value is None for value in numbers):
        return None
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _sum(values: Sequence[Any]) -> float | int | None:
    numbers = [_number(value) for value in values]
    if not numbers:
        return None
    if any(value is None for value in numbers):
        return None
    result = sum(numbers)
    return int(result) if result.is_integer() else float(result)


def _aggregate(rows: Sequence[Mapping[str, Any]], name: str) -> Any:
    if name in RMSE_METRICS:
        return _pool([row.get(name) for row in rows])
    if name in ("uwb_requested", "uwb_attempted", "uwb_accepted", "measurement_bytes_equivalent"):
        return _sum([row.get(name) for row in rows])
    if name == "uav2_strict_parking":
        return sum(bool(row.get(name)) for row in rows) if rows else None
    if name == "uwb_nis_mean":
        return _mean([row.get(name) for row in rows], allow_missing=True)
    return _mean([row.get(name) for row in rows])


def groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group physical summaries with explicit censored and complete subsets.

    Full-cohort values are null whenever one block is incomplete.  The
    ``complete_case_*`` aliases are retained for convenient CSV consumers.
    """
    grouped: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("study_phase"), row.get("scene_id"), row.get("policy"))].append(row)
    output: list[dict[str, Any]] = []
    for (phase, scene, policy), block in grouped.items():
        complete = [row for row in block if _complete(row)]
        all_complete = bool(block) and len(complete) == len(block)
        item: dict[str, Any] = {
            "study_phase": phase, "scene_id": scene, "policy": policy,
            "blocks": len(block), "complete_blocks": len(complete),
            "censored_blocks": len(block) - len(complete),
            "full_horizon_eligible": all_complete,
        }
        for name in METRICS:
            value = _aggregate(block, name) if all_complete else None
            subset = _aggregate(complete, name)
            item[name] = value
            item["complete_case_" + name] = subset
            item[name + "_completed_subset"] = subset
            item["completed_subset_" + name] = subset
        aliases = {
            "position_nees_mean": "uav2_position_nees_mean",
            "final_localization_error_mean_m": "uav2_final_localization_error_m",
            "requests": "uwb_requested", "attempts": "uwb_attempted",
            "fused": "uwb_accepted", "request_bytes_equivalent": "measurement_bytes_equivalent",
            "strict_parking": "uav2_strict_parking",
        }
        for alias, name in aliases.items():
            item[alias] = item[name]
            item["complete_case_" + alias] = item["complete_case_" + name]
            item[alias + "_completed_subset"] = item["complete_case_" + name]
            item["completed_subset_" + alias] = item["complete_case_" + name]
        output.append(item)
    return output


def _seeds(spec: Mapping[str, Any], phase: str, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    key = "development_seeds" if phase == "develop" else "confirmation_seeds"
    values = spec.get(key)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return sorted({row.get("seed") for row in rows if row.get("study_phase") == phase}, key=str)


def _geometries(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    values = spec.get("geometry_ids")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return sorted({row.get("scene_id") for row in rows}, key=str)


def _norm_difference(point: Mapping[str, Any]) -> float | None:
    for key in ("norm_utility_difference", "utility_difference_norm", "norm_peer_difference"):
        value = _number(point.get(key))
        if value is not None:
            return value
    return None


def _meaningful(point: Mapping[str, Any], threshold: float = MEANINGFUL_NORM) -> bool:
    value = point.get("meaningful_norm")
    if value is not None:
        return bool(value)
    norm = _norm_difference(point)
    return norm is not None and abs(norm) >= threshold


def _preferred(point: Mapping[str, Any], threshold: float = MEANINGFUL_NORM) -> str | None:
    norm = _norm_difference(point)
    # ``preferred_peer`` in historical records is the old SSE preference.
    # V13 coverage labels must come from the signed norm utility label first.
    if norm is not None:
        return None if abs(norm) < threshold else PEER_LABELS[0] if norm > 0 else PEER_LABELS[1]
    for key in ("norm_preferred_peer", "preferred_peer_norm", "true_preferred_peer"):
        value = point.get(key)
        if value in PEER_LABELS:
            return str(value)
    return None


def _informative(points: Sequence[Mapping[str, Any]], threshold: float = MEANINGFUL_NORM) -> list[Mapping[str, Any]]:
    return [point for point in points if point.get("dual_eligible") and _meaningful(point, threshold)]


def _ranking(points: Sequence[Mapping[str, Any]], threshold: float = MEANINGFUL_NORM) -> dict[str, Any]:
    usable = []
    for point in _informative(points, threshold):
        norm = _norm_difference(point)
        score = _number(point.get("v13_score_difference"))
        correct = point.get("v13_ranking_correct")
        if norm is None or score is None:
            continue
        if correct is None:
            correct = float((score * norm) > 0.0) if abs(score) > 1e-12 else 0.5
        usable.append((point, float(correct), score, norm))
    return {
        "dual_points": sum(bool(p.get("dual_eligible")) for p in points),
        "informative_points": len(usable),
        "ranking_accuracy": _mean([item[1] for item in usable]),
        "pearson": correlation([item[2] for item in usable], [item[3] for item in usable]),
        "spearman": correlation([item[2] for item in usable], [item[3] for item in usable], True),
        "preferred_peers": sorted({label for p in (item[0] for item in usable) if (label := _preferred(p, threshold))}),
        "geometries": sorted({item[0].get("scene_id") for item in usable}, key=str),
        "seeds": sorted({item[0].get("seed") for item in usable}, key=str),
    }


def _old_sse_difference(point: Mapping[str, Any]) -> float | None:
    branches = point.get("branches")
    if not isinstance(branches, Mapping):
        return _number(point.get("utility_difference_sse"))
    values = []
    for peer in PEER_LABELS:
        branch = branches.get(peer)
        base = branches.get("none")
        if not isinstance(branch, Mapping) or not isinstance(base, Mapping):
            return None
        peer_sse, base_sse = branch.get("sse_m2", branch.get("sse")), base.get("sse_m2", base.get("sse"))
        try:
            peer_values, base_values = [float(v) for v in peer_sse], [float(v) for v in base_sse]
        except (TypeError, ValueError):
            return None
        if len(peer_values) != len(base_values) or not peer_values:
            return None
        values.append(sum(a - b for a, b in zip(base_values, peer_values)) / len(peer_values))
    return values[0] - values[1]


def _legacy_reference(points: Sequence[Mapping[str, Any]], threshold: float = MEANINGFUL_NORM) -> dict[str, Any]:
    # The new V13 norm label cannot be used to re-evaluate the historical V12
    # source-30 gate.  Keep it as an unevaluated historical reference.
    old_norm = [p for p in points if p.get("dual_eligible") and p.get("legacy_norm_ranking_correct") is not None]
    old_sse = []
    for point in points:
        if not point.get("dual_eligible"):
            continue
        score = _number(point.get("score_difference"))
        truth = _old_sse_difference(point)
        if score is not None and truth is not None and abs(truth) >= threshold:
            correct = .5 if abs(score) <= 1e-12 else float(score * truth > 0)
            old_sse.append((score, truth, correct))
    return {
        "historical_source30": {"source": "V12 legacy source-30 gate", "minimum_distinguishable_dual": 30,
                                "count": None, "passed": None, "evaluated_this_round": False,
                                "reason": "V13 norm labels are a new utility and cannot recreate the historical gate"},
        "old_norm": {"count": len(old_norm), "passed": None, "evaluated_this_round": False,
                     "reason": "legacy_norm_ranking_correct is against the new V13 norm label"},
        "old_sse": {"informative_points": len(old_sse),
                    "ranking_accuracy": _mean([item[2] for item in old_sse]),
                    "pearson": correlation([item[0] for item in old_sse], [item[1] for item in old_sse]),
                    "spearman": correlation([item[0] for item in old_sse], [item[1] for item in old_sse], True),
                    "utility_resolution_m": threshold, "evaluated_this_round": True,
                    "scope": "old SSE score versus old branch-SSE label; descriptive reference only"},
        "separate_reference": True, "does_not_lower_v13_gates": True,
    }


def _physical_gate(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> dict[str, Any]:
    confirm = [row for row in rows if row.get("study_phase") == "confirm"]
    scenes = _geometries(spec, rows)
    seeds = _seeds(spec, "confirm", rows)
    bykey = {(row.get("scene_id"), row.get("seed"), row.get("policy")): row for row in confirm}
    failures: list[str] = []
    missing: list[str] = []
    required_policies = tuple(spec.get("policies", ("random_peer", "v13_greedy")))
    gates = spec.get("gates", {}) if isinstance(spec.get("gates", {}), Mapping) else {}
    trajectory_fraction = float(gates.get("trajectory_fraction", .05))
    maximum_case_harm = float(gates.get("maximum_case_harm", .05))
    positive_joint_seeds = int(gates.get("positive_joint_seeds", 3))
    per_geometry: dict[str, Any] = {}
    paired: list[dict[str, Any]] = []
    all_safety_full = all(bool(row.get("full_horizon_eligible")) for row in rows)
    for scene in scenes:
        per_geometry[str(scene)] = {}
        for seed in seeds:
            for policy in required_policies:
                if bykey.get((scene, seed, policy)) is None:
                    missing.append(f"{scene}:{seed}:{policy}")
        random_rows = [bykey.get((scene, seed, "random_peer")) for seed in seeds]
        candidate_rows = [bykey.get((scene, seed, "v13_greedy")) for seed in seeds]
        for seed, random, candidate in zip(seeds, random_rows, candidate_rows):
            if random is None or candidate is None:
                missing.append(f"{scene}:{seed}:random_peer/v13_greedy")
                continue
            for metric in RMSE_METRICS:
                baseline, value = _number(random.get(metric)), _number(candidate.get(metric))
                gain = None if baseline is None or baseline <= 1e-12 or value is None else 1.0 - value / baseline
                paired.append({"scene_id": scene, "seed": seed, "metric": metric,
                               "improvement_fraction": gain})
                if gain is None:
                    missing.append(f"{scene}:{seed}:{metric}")
                elif gain < -maximum_case_harm:
                    failures.append(f"case harm >5% {scene}:{seed}:{metric}")
        for metric in RMSE_METRICS:
            random_complete = bool(random_rows) and len(random_rows) == len(seeds) and all(
                r is not None and _complete(r) for r in random_rows)
            candidate_complete = bool(candidate_rows) and len(candidate_rows) == len(seeds) and all(
                r is not None and _complete(r) for r in candidate_rows)
            baseline = _pool([r.get(metric) for r in random_rows]) if random_complete else None
            value = _pool([r.get(metric) for r in candidate_rows]) if candidate_complete else None
            gain = None if baseline is None or baseline <= 1e-12 or value is None else 1.0 - value / baseline
            per_geometry[str(scene)][metric] = {"random_peer": baseline, "v13_greedy": value,
                                                "improvement_fraction": gain,
                                                "passed": gain is not None and gain >= trajectory_fraction}
            if gain is None or gain < trajectory_fraction:
                failures.append(f"pooled confirmation benefit {scene}:{metric}")
    positive_seed_count = 0
    seed_metric_averages = {}
    for seed in seeds:
        averages = {}
        for metric in RMSE_METRICS:
            gains = [item["improvement_fraction"] for item in paired
                     if item["seed"] == seed and item["metric"] == metric]
            averages[metric] = sum(gains) / len(gains) if len(gains) == len(scenes) and gains and all(v is not None for v in gains) else None
        seed_metric_averages[str(seed)] = averages
        if all(averages.get(metric) is not None and averages[metric] > 0 for metric in RMSE_METRICS):
            positive_seed_count += 1
    if positive_seed_count < min(positive_joint_seeds, len(seeds)):
        failures.append("joint positive confirmation seeds")
    if missing:
        failures.append("missing confirmation case")
    if not all_safety_full:
        failures.append("incomplete physical horizon")
    return {"passed": not failures and not missing and all_safety_full,
            "failures": sorted(set(failures)), "missing_cases": sorted(set(missing)),
            "all_safety_full": all_safety_full, "per_geometry": per_geometry,
            "paired_changes": paired, "positive_seed_rule": "same 3/4 seeds jointly positive averaged over two geometries and both RMSE metrics",
            "positive_seed_count": positive_seed_count, "seed_metric_averages": seed_metric_averages,
            "positive_joint_seed_threshold": positive_joint_seeds,
            "trajectory_fraction_threshold": trajectory_fraction,
            "maximum_case_harm_threshold": maximum_case_harm,
            "policy_comparison": "v13_greedy versus random_peer; full physical 120-step endpoints only"}


def assess(rows: Sequence[Mapping[str, Any]], points: Sequence[Mapping[str, Any]],
           spec: Mapping[str, Any], integrity: bool) -> dict[str, Any]:
    """Apply frozen V13 gates to already-produced rows and points."""
    gates = spec.get("gates", {}) if isinstance(spec.get("gates", {}), Mapping) else {}
    utility_resolution = float(gates.get("utility_resolution_m", MEANINGFUL_NORM))
    overall_points = _informative(points, utility_resolution)
    confirm_points = [point for point in points if point.get("study_phase") == "confirm"]
    overall = _ranking(points, utility_resolution)
    confirmation = _ranking(confirm_points, utility_resolution)
    seeds = _seeds(spec, "confirm", rows)
    seed_pools = {}
    for seed in seeds:
        seed_pools[str(seed)] = _ranking([p for p in confirm_points if p.get("seed") == seed], utility_resolution)
    positive_rank_seeds = sum(
        pool["ranking_accuracy"] is not None and pool["ranking_accuracy"] > 0.5
        and pool["spearman"] is not None and pool["spearman"] > 0
        for pool in seed_pools.values()
    )
    geometries = _geometries(spec, rows)
    geometry_coverage = {
        phase: sorted({p.get("scene_id") for p in points if p.get("study_phase") == phase}, key=str)
        for phase in ("develop", "confirm")
    }
    labels = {}
    label_seed_coverage = {}
    phase_seed_sets = {phase: set(_seeds(spec, phase, rows)) for phase in ("develop", "confirm")}
    for phase in ("develop", "confirm"):
        phase_points = [p for p in points if p.get("study_phase") == phase and p.get("dual_eligible") and _meaningful(p, utility_resolution)]
        labels[phase] = sorted({_preferred(p, utility_resolution) for p in phase_points if _preferred(p, utility_resolution)}, key=str)
        label_seed_coverage[phase] = {
            label: sorted({p.get("seed") for p in phase_points if _preferred(p, utility_resolution) == label}, key=str)
            for label in PEER_LABELS
        }
    coverage_details = {}
    for phase in ("develop", "confirm"):
        phase_points = [p for p in points if p.get("study_phase") == phase and p.get("dual_eligible") and _meaningful(p, utility_resolution)]
        coverage_details[phase] = {"informative": len(phase_points),
                                   "geometries": sorted({p.get("scene_id") for p in phase_points}, key=str),
                                   "labels": sorted({_preferred(p, utility_resolution) for p in phase_points if _preferred(p, utility_resolution)}, key=str),
                                   "seeds": sorted({p.get("seed") for p in phase_points}, key=str),
                                   "expected_seeds": sorted(phase_seed_sets[phase], key=str)}
    coverage_passed = (
        all(set(coverage_details[phase]["geometries"]) >= set(geometries) and
            set(coverage_details[phase]["labels"]) >= set(PEER_LABELS) and
            set(coverage_details[phase]["seeds"]) >= phase_seed_sets[phase]
            for phase in ("develop", "confirm"))
    )
    ranking_accuracy_gate = float(gates.get("ranking_accuracy", .65))
    spearman_gate = float(gates.get("spearman", .20))
    positive_rank_gate = int(gates.get("positive_rank_seeds", 3))
    informative_overall_gate = int(gates.get("meaningful_overall", 36))
    informative_confirmation_gate = int(gates.get("meaningful_confirmation", 18))
    ranking_passed = (
        confirmation["informative_points"] >= informative_confirmation_gate
        and confirmation["ranking_accuracy"] is not None and confirmation["ranking_accuracy"] >= ranking_accuracy_gate
        and confirmation["spearman"] is not None and confirmation["spearman"] >= spearman_gate
        and positive_rank_seeds >= positive_rank_gate
    )
    informative_passed = len(overall_points) >= informative_overall_gate and len(_informative(confirm_points, utility_resolution)) >= informative_confirmation_gate
    physical = _physical_gate(rows, spec)
    legacy = _legacy_reference(points, utility_resolution)
    task_none = not overall_points or not rows
    unsafe = not integrity or not physical["all_safety_full"] or bool(physical["missing_cases"])
    all_joint = bool(integrity and informative_passed and coverage_passed and ranking_passed and physical["passed"])
    differences_exist = bool(overall_points or any(
        value.get("improvement_fraction") is not None
        for scene in physical["per_geometry"].values() for value in scene.values()
    ))
    if all_joint:
        research_status = "通过"
    elif unsafe or task_none:
        research_status = "未通过"
    elif differences_exist:
        research_status = "有条件通过"
    else:
        research_status = "未通过"
    return {
        "research_status": research_status,
        "status": "pass_limited_scope" if all_joint else ("conditional_pass" if research_status == "有条件通过" else "not_passed"),
        "training_ready": all_joint,
        "integrity_passed": bool(integrity), "unsafe": unsafe, "task_none": task_none,
        "gates": {"informative_points": informative_passed, "coverage": coverage_passed,
                  "v13_ranking": ranking_passed, "physical_v13_vs_random": physical["passed"],
                  "all_joint_limited": all_joint},
        # Compatibility aliases make the gate auditable beside V11/V12
        # reports while keeping the V13 predicate independent.
        "coverage_passed": coverage_passed, "meaningful_difference_passed": informative_passed,
        "ranking_passed": ranking_passed, "trajectory_passed": physical["passed"],
        "physical_passed": physical["passed"], "legacy_gate": legacy,
        "old_greedy_norm_reference": legacy["old_norm"],
        "old_greedy_sse_reference": legacy["old_sse"],
        "overall": overall, "confirmation": confirmation, "seed_pools": seed_pools,
        "positive_rank_seed_pools": positive_rank_seeds,
        "informative_overall": len(overall_points), "informative_confirmation": len(_informative(confirm_points, utility_resolution)),
        "geometry_coverage": geometry_coverage, "coverage_details": coverage_details,
        "preferred_labels": labels,
        "preferred_label_seed_coverage": label_seed_coverage,
        "physical": physical, "legacy_reference": legacy,
        "scope": "four independent confirmation seeds; geometry and decision windows are repeated blocks; no optimizer or training",
    }


def _feature_correlations(points: Sequence[Mapping[str, Any]], threshold: float = MEANINGFUL_NORM) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    usable = [p for p in points if p.get("dual_eligible") and _norm_difference(p) is not None]
    keys = sorted({key for point in usable if isinstance(point.get("v13_features"), Mapping)
                   for key, value in point["v13_features"].items() if _number(value) is not None}, key=str)
    for stratum in ("all_dual", "informative"):
        stratum_points = [p for p in usable if stratum == "all_dual" or _meaningful(p, threshold)]
        for phase in ("develop", "confirm"):
            for scene in ("all", *sorted({p.get("scene_id") for p in stratum_points if p.get("study_phase") == phase}, key=str)):
                for seed in ("all", *sorted({p.get("seed") for p in stratum_points if p.get("study_phase") == phase}, key=str)):
                    subset = [p for p in stratum_points if p.get("study_phase") == phase and
                              (scene == "all" or p.get("scene_id") == scene) and
                              (seed == "all" or p.get("seed") == seed)]
                    target = [_norm_difference(p) for p in subset]
                    for feature in keys:
                        values = [_number(p.get("v13_features", {}).get(feature)) for p in subset]
                        pairs = [(x, y) for x, y in zip(values, target) if x is not None and y is not None]
                        x, y = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                        rows.append({"study_phase": phase, "scene_id": scene, "seed": seed,
                                     "stratum": stratum, "feature": feature, "n": len(pairs),
                                     "pearson": correlation(x, y), "spearman": correlation(x, y, True),
                                     "outcome": "signed_peer_norm_utility_difference",
                                     "scope": "evaluation-only descriptive; decision windows are repeated blocks, not independent samples"})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _load_points(cf_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = [dict(point) for point in manifest.get("results", []) if isinstance(point, Mapping)]
    bykey = {(p.get("study_phase"), p.get("scene_id"), p.get("seed"), p.get("step")): p for p in base}
    for path in sorted(cf_dir.glob("*.json")):
        if path.name == "manifest.json" or path.name.endswith("_failure.json"):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping) or "branches" not in value:
            continue
        key = (value.get("study_phase"), value.get("scene_id"), value.get("seed"), value.get("step"))
        if any(item is None for item in key):
            continue
        if key in bykey:
            bykey[key].update(value)
        else:
            bykey[key] = dict(value)
    return list(bykey.values())


def report(batch: str) -> dict[str, Any]:
    """Create a read-only V13 report from checked phase manifests."""
    from ..experiments import neighbor_selection_validation_v13 as run

    spec = run.verify()
    base = run.validate_batch(batch)
    manifests = {phase: run.checked_manifest(base / phase / "manifest.json")
                 for phase in ("develop", "confirm", "counterfactual")}
    rows = list(manifests["develop"].get("summaries", [])) + list(manifests["confirm"].get("summaries", []))
    points = _load_points(base / "counterfactual", manifests["counterfactual"])
    report_gates = spec.get("gates", {}) if isinstance(spec.get("gates", {}), Mapping) else {}
    utility_resolution = float(report_gates.get("utility_resolution_m", MEANINGFUL_NORM))
    integrity_failures = []
    for phase in ("develop", "confirm"):
        if not manifests[phase].get("safety", {}).get("passed", False):
            integrity_failures.append(phase + " safety failed")
        if manifests[phase].get("failures"):
            integrity_failures.append(phase + " has failures")
        expected = len(_geometries(spec, rows)) * len(_seeds(spec, phase, rows)) * len(spec.get("policies", POLICIES))
        if len(manifests[phase].get("summaries", [])) != expected:
            integrity_failures.append(phase + " physical case count mismatch")
    if manifests["counterfactual"].get("failures"):
        integrity_failures.append("counterfactual has failures")
    if len(points) != int(manifests["counterfactual"].get("planned_points", 96)):
        integrity_failures.append("counterfactual point count mismatch")
    if any(point.get("identity_matches") is not True for point in points):
        integrity_failures.append("counterfactual identity mismatch")
    integrity = not integrity_failures
    assessment = assess(rows, points, spec, integrity)
    gg = groups(rows)
    per_seed: list[dict[str, Any]] = []
    all_metric_names = METRICS + ("full_horizon_eligible",)
    for phase in ("develop", "confirm"):
        seeds = _seeds(spec, phase, rows)
        for scene in _geometries(spec, rows):
            for seed in seeds:
                item: dict[str, Any] = {"study_phase": phase, "scene_id": scene, "seed": seed}
                selected = {(r.get("policy")): r for r in rows if r.get("study_phase") == phase and
                            r.get("scene_id") == scene and r.get("seed") == seed}
                for policy in POLICIES:
                    row = selected.get(policy)
                    for metric in all_metric_names:
                        item[f"{policy}_{metric}"] = row.get(metric) if row else None
                random = selected.get("random_peer")
                for policy in POLICIES:
                    row = selected.get(policy)
                    for metric in RMSE_METRICS:
                        base_value = _number(random.get(metric)) if random else None
                        value = _number(row.get(metric)) if row else None
                        gain = None if base_value is None or base_value <= 1e-12 or value is None else 1.0 - value / base_value
                        item[f"{policy}_vs_random_{metric}_gain"] = gain
                        item[f"{policy}_gain_vs_random_{metric}"] = gain
                per_seed.append(item)
    feature_rows = _feature_correlations(points, utility_resolution)
    data, logs = base / "report", run.LOGS / batch / "report"
    plots = run.ROOT / "plots/preliminary" / run.STEM / batch
    for folder in (data, logs):
        folder.mkdir(parents=False, exist_ok=False)
    plots.mkdir(parents=True, exist_ok=False)
    _write_csv(logs / "full_summary.csv", rows)
    _write_csv(logs / "grouped.csv", gg)
    _write_csv(logs / "per_seed.csv", per_seed)
    _write_csv(logs / "counterfactual_points.csv", points)
    _write_csv(logs / "feature_relationships.csv", feature_rows)
    aggregate = {
        "assessment": assessment, "groups": gg, "per_seed": per_seed,
        "feature_relationships": feature_rows, "integrity": integrity,
        "integrity_failures": integrity_failures,
        "physical_steps": sum(int(manifests[p].get("physical_steps", 0) or 0) for p in ("develop", "confirm")),
        "attempted_steps": sum(int(manifests[p].get("attempted_steps", 0) or 0) for p in ("develop", "confirm")),
        "optimizer_updates": 0, "formal_reward_changed": False, "training_ready": assessment["training_ready"],
        "joint_gnss_rf": True, "separate_gnss_rf_causal_claim": False,
        "virtual_field_calibrated_city": False, "counterfactual_points": len(points),
        "counterfactual_scope": "post-local current plus three frozen local packets; no queue/navigation counterfactual",
        "report_paths": {"develop_manifest": str(base / "develop" / "manifest.json"),
                         "confirm_manifest": str(base / "confirm" / "manifest.json"),
                         "counterfactual_manifest": str(base / "counterfactual" / "manifest.json")},
    }
    _json_write(data / "aggregate.json", aggregate)
    _json_write(logs / "aggregate.json", aggregate)
    lines = ["# V13 联合 GNSS/RF 邻居选择报告", "",
             f"研究状态：{assessment['research_status']}；training_ready={assessment['training_ready']}。",
             "本报告只读取已核验 manifest 和实际 JSON；optimizer_updates=0，正式 reward 未改变。", "",
             "## 完整时域与失败感知汇总", "",
             "|阶段|几何|策略|块数|完整块|截断块|绝对 RMSE(m)|相对 RMSE(m)|请求|融合|严格停车|",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gg:
        def fmt(value: Any) -> str:
            return "—" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["study_phase"], row["scene_id"], row["policy"], row["blocks"], row["complete_blocks"],
            row["censored_blocks"], fmt(row["uav2_absolute_rmse_m"]), fmt(row["cluster_relative_rmse_m"]),
            fmt(row["uwb_requested"]), fmt(row["uwb_accepted"]), fmt(row["uav2_strict_parking"])))
    lines += ["", "## 冻结门控", "", "```json", json.dumps(assessment, ensure_ascii=False, indent=2), "```", "",
              "V13 的 raw norm gain 使用米为单位，未归一化、未拟合；所有真值标签只用于离线评价。", "",
              "反事实在本地更新后、协同更新前使用当前状态和三个已冻结本地传感器包，不重放队列或导航。请求数量不同对应 none/peer 联合行为，不能归因为 peer identity 的因果效果。", "",
              "GNSS 与 RF 在联合动态固定场中共同变化，报告不作分离因果结论；virtual field 未按城市校准。512-byte 等价负载仅为 496+16 header 的抽象，不保证真实 codec 或完整协方差序列化。", "",
              "V12 legacy source-30 gate 作为独立 reference 保留，不能降低 V13 门槛。相关系数按 phase/geometry/seed 作描述性汇总，决策窗口是重复块而非独立样本。"]
    (data / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        confirm_groups = [row for row in gg if row["study_phase"] == "confirm"]
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [f"{row['scene_id']}\n{row['policy']}" for row in confirm_groups]
        values = [row["uav2_absolute_rmse_m"] for row in confirm_groups]
        ax.bar(labels, [float(value) if value is not None else float("nan") for value in values])
        ax.set_ylabel("UAV2 absolute RMSE (m), complete cohorts only"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(plots / f"confirmation_rmse_comparison.{ext}", dpi=160)
        plt.close(fig)
        scatter = [p for p in points if p.get("dual_eligible") and _norm_difference(p) is not None and _number(p.get("v13_score_difference")) is not None]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([_number(p.get("v13_score_difference")) for p in scatter], [_norm_difference(p) for p in scatter], alpha=.7)
        ax.axhline(0, color="gray", linewidth=.7); ax.axvline(0, color="gray", linewidth=.7)
        ax.set_xlabel("V13 score difference"); ax.set_ylabel("Signed peer norm utility difference (m)")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(plots / f"v13_score_vs_true_norm.{ext}", dpi=160)
        plt.close(fig)
    except Exception as exc:
        aggregate["plot_error"] = type(exc).__name__ + ": " + str(exc)
        _json_write(data / "aggregate.json", aggregate)
        _json_write(logs / "aggregate.json", aggregate)
    run.write_manifest(data, logs, {"study_phase": "report", "assessment": assessment,
                                    "artifact_count": len(list(data.iterdir())) + len(list(logs.iterdir()))})
    return aggregate


__all__ = ["groups", "assess", "report", "correlation"]
