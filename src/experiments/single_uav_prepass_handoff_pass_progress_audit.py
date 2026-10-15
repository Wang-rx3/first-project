"""Pure, read-only post-handoff audit of frozen single-UAV trajectories.

Only the Python standard library is imported. The module reads CSV/JSON inputs,
validates their frozen identity, computes descriptive trajectory diagnostics,
and atomically publishes eight new artifacts. It never imports a runtime,
controller, evaluator, environment, simulator, or training module.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


RUN_REL = Path("logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/run")
PREP_REL = Path("logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/preparation")
PRIOR_REL = Path("logs/preliminary/single_uav_prepass_reorientation_reference_audit_seed2026/prepass_reorientation_summary.json")
OUTPUT_REL = Path("logs/preliminary/single_uav_prepass_handoff_pass_progress_audit_seed2026")
PASS_CLEARANCE_M = 2.5
LATERAL_THRESHOLD_MPS = 0.5
NEAR_PASS_BAND_M = (-1.0, 0.0)
MAIN_IDS = tuple(f"test_r99_episode_{i:04d}" for i in (2, 4, 5, 6, 11, 20))
RESCUE_ID = "test_r99_episode_0012"
SAFETY_ID = "test_r99_episode_0013"
BOUNDARY_ID = "test_r99_episode_0014"
EXPECTED_IDS = tuple(f"test_r99_episode_{i:04d}" for i in range(1, 21))
EXPECTED_PASSES = {"off": 11, "on": 12}
OUTPUT_FILES = (
    "prepass_handoff_summary.json",
    "prepass_handoff_episode_diagnostics.csv",
    "prepass_handoff_timing.csv",
    "prepass_handoff_pass_progress_windows.csv",
    "prepass_handoff_assist_event_response.csv",
    "prepass_handoff_goal_vs_pass_progress.csv",
    "prepass_handoff_special_cases.csv",
    "prepass_handoff_report.md",
)
ASSIST_FIELDS = (
    "passed_margin_m", "relevant_obstacle_clearance_m", "goal_distance_m",
    "estimated_radial_velocity_mps", "actual_radial_velocity_mps",
    "estimated_lateral_velocity_mps", "actual_lateral_velocity_mps",
)


class AuditError(RuntimeError):
    """Input, identity, or output contract failed closed."""


def number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    result = number(value)
    return int(result) if result is not None and result.is_integer() else None


def flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON input must be an object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise AuditError(f"CSV has no header: {path}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise AuditError(f"cannot read CSV input {path}: {exc}") from exc


def _vec(value: Any) -> tuple[float, ...] | None:
    if value is None or str(value).strip() in {"", "[]"}:
        return None
    try:
        parsed = json.loads(str(value)) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        return None
    result = tuple(number(item) for item in parsed)
    if any(item is None for item in result):
        return None
    return tuple(float(item) for item in result)


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _ols(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    points = [(integer(row.get("step")), number(row.get(key))) for row in rows]
    pairs = [(float(x), y) for x, y in points if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else None


def _margins(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, float]]:
    result = []
    for row in rows:
        step, margin = integer(row.get("step")), number(row.get("passed_margin_m"))
        if step is not None and margin is not None:
            result.append((step, margin))
    return result


def _clearance(row: Mapping[str, Any]) -> float | None:
    return number(row.get("relevant_obstacle_clearance_m"))


def _pass_gate(row: Mapping[str, Any]) -> bool:
    margin, clearance = number(row.get("passed_margin_m")), _clearance(row)
    return flag(row.get("passed_obstacle")) and margin is not None and clearance is not None and margin > 0.0 and clearance >= PASS_CLEARANCE_M


def _angle(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    na, nb = math.hypot(a[0], a[1]), math.hypot(b[0], b[1])
    if na == 0.0 or nb == 0.0:
        return None
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosine))


def _angle_from_row(row: Mapping[str, Any], *, use_assist_axis: bool) -> float | None:
    center = _vec(row.get("obstacle_center_m"))
    rel = _vec(row.get("obstacle_relative_vector_m"))
    encounter = _vec(row.get("signed_encounter_estimated_position_m"))
    goal_ray = _vec(row.get("lateral_exit_assist_horizontal_goal_ray_m"))
    current_radial = _vec(row.get("radial_intent_assist_radial_unit")) if use_assist_axis else None
    if current_radial is None:
        current_radial = _vec(row.get("signed_axis_radial_unit"))
    if not (center and rel and encounter and goal_ray and current_radial):
        return None
    current_position = (center[0] - rel[0], center[1] - rel[1])
    goal = (encounter[0] + goal_ray[0], encounter[1] + goal_ray[1])
    direction = (goal[0] - current_position[0], goal[1] - current_position[1])
    return _angle(current_radial, direction)


def _longest_run(values: Sequence[bool]) -> int:
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _negative_adjacent(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[int, float]] = []
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) if integer(row.get("step")) is not None else -1)
    for left, right in zip(ordered, ordered[1:]):
        s0, s1 = integer(left.get("step")), integer(right.get("step"))
        m0, m1 = number(left.get("passed_margin_m")), number(right.get("passed_margin_m"))
        if s0 is not None and s1 == s0 + 1 and m0 is not None and m1 is not None:
            pairs.append((s0, m1 - m0))
    negative = [delta < 0.0 for _, delta in pairs]
    return {
        "adjacent_pair_count": len(pairs),
        "negative_adjacent_margin_count": sum(negative),
        "negative_adjacent_margin_longest_run": _longest_run(negative),
        "negative_adjacent_margin_fraction": sum(negative) / len(pairs) if pairs else None,
        "negative_adjacent_steps": [step + 1 for (step, delta) in pairs if delta < 0.0],
    }


def _slope_delta(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "ols_margin_slope_m_per_step": _ols(rows, "passed_margin_m"),
        **_negative_adjacent(rows),
    }


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _expected_prep_hash(prep: Mapping[str, Any], lock: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    digest = canonical_hash(prep)
    _check(digest == lock.get("payload_sha256") == lock.get("active_radial_intent_preregistration_hash"), "active preregistration payload hash mismatch")
    _check(digest == runtime.get("preregistration_payload_sha256") == runtime.get("new_radial_intent_preregistration_hash"), "runtime lock preregistration hash mismatch")
    _check(runtime.get("parent_lateral_exit_preregistration_hash") == lock.get("parent_lateral_exit_preregistration_hash"), "parent preregistration lock mismatch")
    return digest


def _runtime_lock_checks(root: Path, runtime: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, str]:
    expected = runtime.get("source_hashes")
    _check(isinstance(expected, dict) and expected, "runtime source lock lacks source_hashes")
    checkpoint = (((evaluation.get("results") or {}).get("on") or {}).get("provenance") or {}).get("checkpoint") or {}
    observed: dict[str, str] = {}
    for label, digest in expected.items():
        if label.startswith("checkpoint/"):
            path = Path(str(checkpoint.get("resolved_path", "")))
        else:
            path = root / Path(label)
        _check(path.is_file(), f"runtime lock source is missing: {label} -> {path}")
        observed[label] = sha256_file(path)
        _check(observed[label] == digest, f"runtime lock source hash mismatch: {label}")
    return observed


def _validate_summary_identity(evaluation: Mapping[str, Any], prep_hash: str, prep: Mapping[str, Any]) -> dict[str, Any]:
    checks = evaluation.get("radial_intent_identity_checks") or {}
    required_true = (
        "off_baseline_fingerprint_match", "same_actor_critic_gat_normalizer_hashes",
        "same_ordered_episode_seed_checkpoint_runtime_identity", "both_arms_goal_recovery_enabled",
        "both_arms_lateral_exit_assistance_enabled", "only_radial_intent_treatment_differs",
    )
    for name in required_true:
        _check(flag(checks.get(name)), f"evaluation identity check failed: {name}")
    _check(evaluation.get("condition") == "paired", "evaluation summary is not paired")
    _check(flag(evaluation.get("radial_intent_diagnostic_prepared")) and flag(evaluation.get("radial_intent_diagnostic_executed")), "radial intent paired diagnostic is not marked prepared and executed")
    _check(evaluation.get("radial_intent_new_preregistration_hash") == prep_hash, "evaluation active preregistration hash mismatch")
    expected_fingerprint = ((prep.get("off_baseline") or {}).get("fingerprint"))
    _check(evaluation.get("radial_intent_off_baseline_fingerprint") == expected_fingerprint, "OFF baseline fingerprint mismatch")
    forbidden = ("training_enabled", "training_performed", "optimizer_created", "optimizer_steps", "backward_calls", "actor_updates", "buffer_updates", "checkpoint_writes", "formal_training_experiment_executed")
    for name in forbidden:
        value = evaluation.get(name)
        _check(value in (False, 0, None), f"evaluation summary indicates prohibited work: {name}={value}")
    return {name: True for name in required_true} | {"paired_condition": True, "active_preregistration_hash": True, "off_baseline_fingerprint": True, "training_flags_clear": True}


def _load_trajectories(root: Path, evaluation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    run = root / RUN_REL
    per_condition: dict[str, Any] = {}
    episodes: dict[str, list[dict[str, str]]] = {}
    steps: dict[str, list[dict[str, str]]] = {}
    for condition in ("off", "on"):
        folder = run / condition
        per_condition[condition] = _json(folder / "summary.json")
        episodes[condition] = _csv(folder / "episodes.csv")
        steps[condition] = _csv(folder / "steps.csv")
    eval_summary = _json(run / "evaluation_summary.json")
    comparison = _csv(run / "comparison_episodes.csv")
    # Keep this argument in the signature to make call sites explicit about the source summary.
    _ = evaluation
    return per_condition, eval_summary, episodes, steps, {"comparison": comparison}


def _validate_trajectories(root: Path, evaluation: Mapping[str, Any], per_condition: Mapping[str, Any], episodes: Mapping[str, Sequence[Mapping[str, str]]], steps: Mapping[str, Sequence[Mapping[str, str]]], comparison: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    prior_manifest = evaluation.get("scenario_manifest_hash")
    ordered_pool = evaluation.get("ordered_pool_identity") or []
    _check(len(ordered_pool) == 20, "paired evaluation ordered pool must contain 20 episodes")
    _check([entry[1] for entry in ordered_pool] == list(EXPECTED_IDS), "evaluation episode order differs from frozen 20-episode order")
    _check([row.get("episode_id") for row in comparison] == list(EXPECTED_IDS), "comparison episode order differs from frozen 20-episode order")
    seed_by_id = {entry[1]: integer(entry[2]) for entry in ordered_pool}
    episode_rows: dict[str, dict[str, dict[str, str]]] = {"off": {}, "on": {}}
    step_groups: dict[str, dict[str, list[dict[str, str]]]] = {"off": {}, "on": {}}
    checks: dict[str, Any] = {"episode_order": True, "paired_seeds": True, "paired_scenario_hashes": True, "step_contiguity": True, "run_summary_identity": True, "pass_counts": {}}
    for condition in ("off", "on"):
        rows = list(episodes[condition])
        _check([row.get("episode_id") for row in rows] == list(EXPECTED_IDS), f"{condition} episodes.csv order mismatch")
        _check(all(row.get("condition") == condition for row in rows), f"{condition} episodes.csv condition mismatch")
        summary = per_condition[condition]
        _check(summary.get("condition") == condition, f"{condition} run summary condition mismatch")
        provenance = summary.get("provenance") or {}
        _check(provenance.get("scenario_manifest_hash") == prior_manifest, f"{condition} scenario manifest mismatch")
        _check(provenance.get("ordered_pool_identity") == ordered_pool, f"{condition} run summary episode identity mismatch")
        _check(summary.get("stage") == "single_uav_goal_recovery_correction_evaluation", f"unexpected {condition} run summary stage")
        _check(flag(summary.get("training_enabled")) is False and flag(summary.get("training_performed")) is False, f"{condition} run summary training flag is set")
        for row in rows:
            episode_id = row["episode_id"]
            episode_rows[condition][episode_id] = row
            _check(integer(row.get("noise_seed")) == seed_by_id[episode_id], f"{condition} noise seed mismatch for {episode_id}")
        grouped: dict[str, list[dict[str, str]]] = {episode_id: [] for episode_id in EXPECTED_IDS}
        for row in steps[condition]:
            episode_id = row.get("episode_id", "")
            _check(episode_id in grouped, f"unexpected {condition} step episode {episode_id}")
            _check(row.get("condition") == condition, f"{condition} step row condition mismatch")
            grouped[episode_id].append(dict(row))
        for episode_id, group in grouped.items():
            group.sort(key=lambda item: integer(item.get("step")) if integer(item.get("step")) is not None else -1)
            ep = episode_rows[condition][episode_id]
            expected_count = integer(ep.get("steps"))
            _check(expected_count == len(group), f"{condition} episode step count mismatch for {episode_id}")
            _check([integer(row.get("step")) for row in group] == list(range(len(group))), f"{condition} non-contiguous steps for {episode_id}")
            for row in group:
                _check(integer(row.get("noise_seed")) == integer(ep.get("noise_seed")), f"{condition} step noise seed mismatch for {episode_id}")
                _check(integer(row.get("scenario_seed")) == integer(ep.get("scenario_seed")), f"{condition} step scenario seed mismatch for {episode_id}")
            step_groups[condition][episode_id] = group
        passed = sum(flag(row.get("passed_obstacle")) for row in rows)
        checks["pass_counts"][condition] = passed
        _check(passed == EXPECTED_PASSES[condition], f"unexpected {condition} passed_obstacle count: {passed}")
        aggregate = summary.get("aggregate") or {}
        _check(integer(aggregate.get("episodes")) == 20, f"{condition} aggregate episode count mismatch")
        _check(integer(aggregate.get("passed_obstacle")) == passed, f"{condition} aggregate passed count mismatch")
    for episode_id in EXPECTED_IDS:
        off, on = episode_rows["off"][episode_id], episode_rows["on"][episode_id]
        _check(integer(off.get("noise_seed")) == integer(on.get("noise_seed")), f"paired noise seed mismatch: {episode_id}")
        _check(integer(off.get("scenario_seed")) == integer(on.get("scenario_seed")), f"paired scenario seed mismatch: {episode_id}")
        _check(off.get("scenario_hash") == on.get("scenario_hash"), f"paired scenario hash mismatch: {episode_id}")
        cmp = comparison[EXPECTED_IDS.index(episode_id)]
        for metric in ("passed_obstacle", "entered_8m", "post_bypass_recovery"):
            for condition in ("OFF", "ON"):
                _check(integer(cmp.get(f"{metric}_{condition}")) == integer(episode_rows[condition.lower()][episode_id].get(metric)), f"comparison {metric}_{condition} mismatch for {episode_id}")
        delta = number(cmp.get("passed_obstacle_delta"))
        _check(delta == float(flag(on.get("passed_obstacle"))) - float(flag(off.get("passed_obstacle"))), f"comparison pass delta mismatch for {episode_id}")
    source_condition_mismatches = []
    for condition in ("off", "on"):
        for episode_id, group in step_groups[condition].items():
            prior_passed = False
            for row in group:
                margin, clearance = number(row.get("passed_margin_m")), _clearance(row)
                source = row.get("passed_condition_now")
                if margin is not None and clearance is not None and source not in (None, ""):
                    expected = margin > 0.0 and clearance >= PASS_CLEARANCE_M
                    if flag(source) != expected:
                        source_condition_mismatches.append({"condition": condition, "episode_id": episode_id, "step": integer(row.get("step")), "source": flag(source), "expected": expected})
                latch = flag(row.get("passed_obstacle"))
                _check(not (prior_passed and not latch), f"passed_obstacle latch regressed in {condition}/{episode_id}")
                prior_passed = latch
            _check(not any(item["condition"] == condition and item["episode_id"] == episode_id for item in source_condition_mismatches), f"passed_condition_now inconsistent with margin/clearance in {condition}/{episode_id}")
    # OFF is the paired baseline; its frozen identity was checked in the summary against the preregistration fingerprint.
    _check(all(not flag(row.get("radial_intent_assist_active")) for group in step_groups["off"].values() for row in group), "OFF arm contains active radial-intent assistance")
    return {"checks": checks, "episode_rows": episode_rows, "step_groups": step_groups, "source_pass_condition_mismatches": source_condition_mismatches}


def _episode_metrics(condition: str, episode_id: str, rows: Sequence[Mapping[str, Any]], ep: Mapping[str, Any], cohort: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Mapping[str, Any]]]]:
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) if integer(row.get("step")) is not None else -1)
    by_step = {integer(row.get("step")): row for row in ordered if integer(row.get("step")) is not None}
    encounter = next((integer(row.get("signed_encounter_step")) for row in ordered if integer(row.get("signed_encounter_step")) is not None), None)
    if encounter is None:
        encounter = next((integer(row.get("lateral_exit_assist_encounter_step")) for row in ordered if integer(row.get("lateral_exit_assist_encounter_step")) is not None), None)
    ready = next((integer(row.get("step")) for row in ordered if flag(row.get("estimated_lateral_exit_ready"))), None)
    eligible = next((integer(row.get("step")) for row in ordered if flag(row.get("radial_intent_assist_eligible"))), None)
    active_steps = [integer(row.get("step")) for row in ordered if flag(row.get("radial_intent_assist_active"))]
    active_steps = [step for step in active_steps if step is not None]
    first_active = active_steps[0] if active_steps else None
    last_active = active_steps[-1] if active_steps else None
    margins = _margins(ordered)
    best_margin = max((value for _, value in margins), default=None)
    best_step = min((step for step, value in margins if value == best_margin), default=None) if best_margin is not None else None
    best_row = by_step.get(best_step) if best_step is not None else None
    first_pass_row = next((row for row in ordered if _pass_gate(row)), None)
    first_pass = integer(first_pass_row.get("step")) if first_pass_row else None
    first_recovery = next((integer(row.get("step")) for row in ordered if flag(row.get("post_bypass_recovery"))), None)
    end = integer(ordered[-1].get("step")) if ordered else None
    margin_start = margins[0][1] if margins else None
    margin_final = margins[-1][1] if margins else None
    pre = [row for row in ordered if ready is not None and best_step is not None and ready <= integer(row.get("step"),) <= best_step]
    post = [row for row in ordered if best_step is not None and integer(row.get("step")) is not None and best_step < integer(row.get("step")) <= (end if end is not None else -1)]
    # Explicit pre-pass interval retained for the Goal Recovery activity record.
    whole_prepass_end = first_pass if first_pass is not None else end
    whole_prepass = [row for row in ordered if ready is not None and integer(row.get("step")) is not None and ready <= integer(row.get("step")) <= (whole_prepass_end if whole_prepass_end is not None else -1)]
    best_clearance = _clearance(best_row) if best_row else None
    if best_margin is None or best_clearance is None:
        pass_limiter = "UNAVAILABLE"
    elif best_margin <= 0 and best_clearance < PASS_CLEARANCE_M:
        pass_limiter = "BOTH_LIMITED"
    elif best_margin <= 0:
        pass_limiter = "MARGIN_LIMITED"
    elif best_clearance < PASS_CLEARANCE_M:
        pass_limiter = "CLEARANCE_LIMITED"
    else:
        pass_limiter = "PASS_GATE_SATISFIED_AT_BEST"
    first_ready_step = ready
    if first_active is None:
        assist_timing = "ASSIST_NEVER_ACTIVE"
    elif best_step is None:
        assist_timing = "ASSIST_BEFORE_BEST"
    elif first_active < best_step:
        assist_timing = "ASSIST_BEFORE_BEST"
    elif first_active == best_step:
        assist_timing = "ASSIST_AT_BEST"
    else:
        assist_timing = "ASSIST_AFTER_BEST"
    pre_slope = _ols(pre, "passed_margin_m")
    post_slope = _ols(post, "passed_margin_m")
    pre_adj, post_adj = _negative_adjacent(pre), _negative_adjacent([by_step[s] for s in sorted(by_step) if best_step is not None and s >= best_step])
    whole_adj = _negative_adjacent(ordered)
    first_regression_step = next((step for step in post_adj["negative_adjacent_steps"]), None)
    regression_interval = [row for row in ordered if best_step is not None and first_regression_step is not None and best_step <= integer(row.get("step")) <= first_regression_step]
    recovery_active = lambda row: flag(row.get("radial_intent_goal_recovery_active")) or flag(row.get("goal_recovery_correction_active"))
    gr_pre_count = sum(recovery_active(row) for row in pre)
    gr_whole_count = sum(recovery_active(row) for row in whole_prepass)
    gr_evidence_inactive = bool(regression_interval) and not any(recovery_active(row) for row in regression_interval)
    best_safety = flag(best_row.get("safety_projection_applied")) if best_row else None
    collision = any(flag(row.get("collision")) for row in ordered)
    post_best_margin_regression = best_margin is not None and margin_final is not None and best_margin - margin_final > 0.0
    pass_commit = {
        "C1_exit_ready_occurred": ready is not None,
        "C2_post_window_regresses": bool(post) and margin_final is not None and best_margin is not None and best_margin - margin_final > 0.0 and post_adj["negative_adjacent_margin_count"] > 0,
        "C3_goal_recovery_inactive_best_to_first_regression": gr_evidence_inactive,
        "C3_prepass_goal_recovery_active_count": gr_pre_count,
        "C3_whole_prepass_goal_recovery_active_count": gr_whole_count,
        "C3_whole_prepass_row_count": len(whole_prepass),
        "C3_regression_evidence_step": first_regression_step,
        "C4_best_clearance_ge_2_5": best_clearance is not None and best_clearance >= PASS_CLEARANCE_M,
        "C4_no_collision": not collision,
        "C4_best_not_safety_projected": best_safety is False,
        "C5_assist_active_at_least_once": bool(active_steps),
        "C5_post_best_regression_persists_while_goal_recovery_inactive": bool(first_regression_step is not None and post_adj["negative_adjacent_margin_count"] > 0 and gr_evidence_inactive),
    }
    pass_commit["supported"] = all(bool(pass_commit.get(key)) for key in ("C1_exit_ready_occurred", "C2_post_window_regresses", "C3_goal_recovery_inactive_best_to_first_regression", "C4_best_clearance_ge_2_5", "C4_no_collision", "C4_best_not_safety_projected", "C5_assist_active_at_least_once", "C5_post_best_regression_persists_while_goal_recovery_inactive"))
    actual_post = [abs(x) for row in post if (x := number(row.get("actual_lateral_velocity_mps"))) is not None]
    estimated_post = [abs(x) for row in post if (x := number(row.get("estimated_lateral_velocity_mps"))) is not None]
    def lateral_diagnostics(key: str) -> dict[str, Any]:
        transitions = []
        crossings = []
        for left, right in zip(post, post[1:]):
            s0, s1 = integer(left.get("step")), integer(right.get("step"))
            v0, v1 = number(left.get(key)), number(right.get(key))
            if s0 is None or s1 != s0 + 1 or v0 is None or v1 is None:
                continue
            transitions.append((s1, abs(v1) - abs(v0)))
            if (abs(v0) > LATERAL_THRESHOLD_MPS) != (abs(v1) > LATERAL_THRESHOLD_MPS):
                crossings.append(s1)
        increases = [delta > 0.0 for _, delta in transitions]
        return {
            "reacceleration_step_count": sum(increases),
            "reacceleration_steps": [step for (step, delta) in transitions if delta > 0.0],
            "reacceleration_longest_run": _longest_run(increases),
            "threshold_crossing_steps_at_0_5": crossings,
            "threshold_crossing_count_at_0_5": len(crossings),
            "post_best_peak_abs_mps": max((abs(number(row.get(key))) for row in post if number(row.get(key)) is not None), default=None),
        }
    actor_signs = [integer(row.get("actor_lateral_sign")) for row in post]
    actor_signs = [value for value in actor_signs if value not in (None, 0)]
    actor_sign_changes = sum(left != right for left, right in zip(actor_signs, actor_signs[1:]))
    angles_pre = [_angle_from_row(row, use_assist_axis=True) for row in pre]
    angles_post = [_angle_from_row(row, use_assist_axis=True) for row in post]
    angles_active = [_angle_from_row(row, use_assist_axis=True) for row in ordered if flag(row.get("radial_intent_assist_active"))]
    def angle_stats(values: Sequence[float | None]) -> dict[str, Any]:
        finite = [value for value in values if value is not None]
        return {"count": len(finite), "mean_deg": _mean(finite), "max_deg": max(finite) if finite else None}
    along_at_best = number(best_row.get("along_obstacle_m")) if best_row else None
    along_final = number(ordered[-1].get("along_obstacle_m")) if ordered else None
    final_margin_regression = best_margin is not None and margin_final is not None and margin_final < best_margin
    along_regression = along_at_best is not None and along_final is not None and along_final > along_at_best
    geometric_flag = "OBSTACLE_RELATIVE_REGRESSION" if final_margin_regression and along_regression else "NO_OBSTACLE_RELATIVE_REGRESSION_FLAG"
    diagnostics = {
        "condition": condition, "episode_id": episode_id, "cohort": cohort,
        "episode_passed_obstacle": flag(ep.get("passed_obstacle")),
        "first_ready_step": ready, "best_margin_step": best_step, "first_pass_step": first_pass,
        "first_post_bypass_recovery_step": first_recovery, "end_step": end,
        "margin_start_m": margin_start, "margin_best_m": best_margin, "margin_final_m": margin_final,
        "margin_best_minus_start_m": best_margin - margin_start if best_margin is not None and margin_start is not None else None,
        "margin_best_minus_final_m": best_margin - margin_final if best_margin is not None and margin_final is not None else None,
        "best_clearance_m": best_clearance, "best_pass_limiter": pass_limiter,
        "near_pass_but_regressed": bool(best_margin is not None and NEAR_PASS_BAND_M[0] <= best_margin <= NEAR_PASS_BAND_M[1] and best_clearance is not None and best_clearance >= PASS_CLEARANCE_M and post_best_margin_regression),
        "best_source_passed_condition_now": flag(best_row.get("passed_condition_now")) if best_row else None,
        "best_exact_pass_gate": bool(best_row and best_margin is not None and best_margin > 0 and best_clearance is not None and best_clearance >= PASS_CLEARANCE_M),
        "pass_source_condition_consistent_at_best": bool(best_row and flag(best_row.get("passed_condition_now")) == (best_margin is not None and best_margin > 0 and best_clearance is not None and best_clearance >= PASS_CLEARANCE_M)),
        "first_assist_eligible_step": eligible, "first_assist_active_step": first_active, "last_assist_active_step": last_active,
        "assist_timing_vs_best": assist_timing, "assist_active_step_count": len(active_steps),
        "pre_best_row_count": len(pre), "post_best_row_count": len(post),
        "pre_best_ols_margin_slope_m_per_step": pre_slope, "post_best_ols_margin_slope_m_per_step": post_slope,
        "pre_best_negative_adjacent_margin_count": pre_adj["negative_adjacent_margin_count"],
        "pre_best_negative_adjacent_margin_longest_run": pre_adj["negative_adjacent_margin_longest_run"],
        "pre_best_negative_adjacent_margin_fraction": pre_adj["negative_adjacent_margin_fraction"],
        "post_best_negative_adjacent_margin_count": post_adj["negative_adjacent_margin_count"],
        "post_best_negative_adjacent_margin_longest_run": post_adj["negative_adjacent_margin_longest_run"],
        "post_best_negative_adjacent_margin_fraction": post_adj["negative_adjacent_margin_fraction"],
        "whole_negative_adjacent_margin_count": whole_adj["negative_adjacent_margin_count"],
        "best_to_final_regression": post_best_margin_regression,
        "first_post_best_regression_evidence_step": first_regression_step,
        "pre_best_estimated_lateral_peak_abs_mps": max((abs(number(row.get("estimated_lateral_velocity_mps"))) for row in pre if number(row.get("estimated_lateral_velocity_mps")) is not None), default=None),
        "pre_best_actual_lateral_peak_abs_mps": max((abs(number(row.get("actual_lateral_velocity_mps"))) for row in pre if number(row.get("actual_lateral_velocity_mps")) is not None), default=None),
        "post_best_estimated_lateral_peak_abs_mps": max(estimated_post, default=None),
        "post_best_actual_lateral_peak_abs_mps": max(actual_post, default=None),
        "post_best_estimated_lateral_diagnostics": lateral_diagnostics("estimated_lateral_velocity_mps"),
        "post_best_actual_lateral_diagnostics": lateral_diagnostics("actual_lateral_velocity_mps"),
        "post_best_actor_lateral_sign_change_count": actor_sign_changes,
        "post_best_actor_logged_lateral_sign_change_count": integer(ordered[-1].get("actor_lateral_sign_changes")) if ordered else None,
        "frame_angle_pre_best": angle_stats(angles_pre), "frame_angle_post_best": angle_stats(angles_post),
        "frame_angle_radial_assist_active": angle_stats(angles_active),
        "angle_unavailable_reason": "STRUCTURAL_NA_DEGENERATE_OR_MISSING_FRAME" if not any(angles_pre + angles_post + angles_active) else None,
        "along_obstacle_start_m": number(ordered[0].get("along_obstacle_m")) if ordered else None,
        "along_obstacle_best_m": along_at_best, "along_obstacle_final_m": along_final,
        "post_best_along_delta_m": along_final - along_at_best if along_final is not None and along_at_best is not None else None,
        "post_best_clearance_delta_m": _clearance(ordered[-1]) - best_clearance if ordered and _clearance(ordered[-1]) is not None and best_clearance is not None else None,
        "obstacle_geometry_flag": geometric_flag,
        "best_safety_projection_applied": best_safety, "collision_any": collision,
        "pass_commit": pass_commit,
        "whole_prepass_goal_recovery_active_count": gr_whole_count,
        "whole_prepass_goal_recovery_rows": len(whole_prepass),
        "near_pass_descriptive_band": list(NEAR_PASS_BAND_M),
        "near_pass_rows_in_frozen_band": sum(NEAR_PASS_BAND_M[0] <= number(row.get("passed_margin_m")) <= NEAR_PASS_BAND_M[1] for row in ordered if number(row.get("passed_margin_m")) is not None),
    }
    goal_pass = _goal_pass_rows(condition, episode_id, cohort, ordered, ready, best_step)
    for window_name, selected in (
        ("pre_best", [row for row in goal_pass if row["window"] == "PRE_BEST"]),
        ("post_best", [row for row in goal_pass if row["window"] == "POST_BEST"]),
        ("radial_assist_active", [row for row in goal_pass if flag(row.get("radial_assist_active_at_step"))]),
    ):
        valid = [row for row in selected if row["classification"] != "UNAVAILABLE"]
        goal_improves = [row for row in valid if row["goal_improves"] is True]
        mismatch = [row for row in valid if row["classification"] == "GOAL_IMPROVES_PASS_WORSENS"]
        diagnostics[f"{window_name}_goal_improves_pass_worsens_count"] = len(mismatch)
        diagnostics[f"{window_name}_goal_improves_pass_worsens_fraction_all_valid"] = len(mismatch) / len(valid) if valid else None
        diagnostics[f"{window_name}_goal_improves_pass_worsens_fraction_given_goal_improves"] = len(mismatch) / len(goal_improves) if goal_improves else None
    primary_diagnoses: list[str] = []
    if assist_timing == "ASSIST_AFTER_BEST":
        primary_diagnoses.append("ASSISTANCE_TOO_LATE_FOR_EXISTING_PASS_WINDOW")
    if post_best_margin_regression:
        primary_diagnoses.append("PASS_MARGIN_REGRESSION_AFTER_LOCAL_IMPROVEMENT")
    if diagnostics["post_best_goal_improves_pass_worsens_count"] > 0:
        primary_diagnoses.append("GOAL_PROGRESS_PASS_GEOMETRY_MISALIGNMENT")
    if diagnostics["post_best_estimated_lateral_diagnostics"]["reacceleration_step_count"] > 0 or diagnostics["post_best_actual_lateral_diagnostics"]["reacceleration_step_count"] > 0:
        primary_diagnoses.append("LATERAL_REACTIVATION")
    if geometric_flag == "OBSTACLE_RELATIVE_REGRESSION":
        primary_diagnoses.append("OBSTACLE_RELATIVE_GEOMETRY_REGRESSION")
    if pass_limiter in {"MARGIN_LIMITED", "CLEARANCE_LIMITED", "BOTH_LIMITED"}:
        primary_diagnoses.append(pass_limiter)
    diagnostics["radial_assistance_activated_after_best_pass_window"] = assist_timing == "ASSIST_AFTER_BEST"
    diagnostics["primary_diagnoses"] = primary_diagnoses or ["NOT_ASSESSABLE"]
    timing = {
        "condition": condition, "episode_id": episode_id, "cohort": cohort,
        "encounter_step": encounter, "first_estimated_lateral_exit_ready_step": ready,
        "first_assist_eligible_step": eligible, "first_assist_active_step": first_active,
        "last_assist_active_step": last_active, "best_pass_margin_m": best_margin,
        "best_pass_margin_earliest_step": best_step, "final_pass_margin_m": margin_final,
        "first_exact_pass_step": first_pass, "first_post_bypass_recovery_step": first_recovery,
        "end_step": end, "assist_timing_vs_best": assist_timing,
        "pre_best_start_step": ready, "pre_best_end_step_inclusive": best_step,
        "post_best_start_step_exclusive": best_step, "post_best_end_step_inclusive": end,
        "assist_start_minus_best_margin_step": first_active - best_step if first_active is not None and best_step is not None else None,
        "lateral_exit_minus_best_margin_step": first_ready_step - best_step if first_ready_step is not None and best_step is not None else None,
        "best_margin_to_end_steps": end - best_step if end is not None and best_step is not None else None,
        "assist_start_to_end_steps": end - first_active if end is not None and first_active is not None else None,
        "best_margin_to_pass_steps": first_pass - best_step if first_pass is not None and best_step is not None else None,
    }
    windows = {"PRE_BEST": pre, "POST_BEST": post, "WHOLE_PREPASS": whole_prepass, "ALL": ordered}
    return diagnostics, timing, windows


def _window_rows(diagnostics: Mapping[str, Any], windows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for label in ("PRE_BEST", "POST_BEST"):
        selected = windows[label]
        slopes = _slope_delta(selected)
        margins = [number(row.get("passed_margin_m")) for row in selected if number(row.get("passed_margin_m")) is not None]
        steps = [integer(row.get("step")) for row in selected if integer(row.get("step")) is not None]
        rows.append({
            "condition": diagnostics["condition"], "episode_id": diagnostics["episode_id"], "cohort": diagnostics["cohort"],
            "window": label, "inclusive_start_step": min(steps) if steps else None, "inclusive_end_step": max(steps) if steps else None,
            "row_count": len(selected), "nonempty": bool(selected), "margin_first_m": margins[0] if margins else None,
            "margin_last_m": margins[-1] if margins else None,
            "margin_delta_m": margins[-1] - margins[0] if len(margins) >= 2 else None,
            **{key: value for key, value in slopes.items() if key != "negative_adjacent_steps"},
            "negative_adjacent_steps": json.dumps(slopes["negative_adjacent_steps"], separators=(",", ":")),
        })
    return rows


def _assist_response(episode_id: str, cohort: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) if integer(row.get("step")) is not None else -1)
    by_step = {integer(row.get("step")): row for row in ordered}
    last = integer(ordered[-1].get("step")) if ordered else None
    output = []
    for row in ordered:
        if not flag(row.get("radial_intent_assist_active")):
            continue
        step = integer(row.get("step"))
        for horizon in (1, 3, 5):
            target_step = step + horizon if step is not None else None
            target = by_step.get(target_step)
            record: dict[str, Any] = {"condition": "on", "episode_id": episode_id, "cohort": cohort, "assist_step": step, "horizon_steps": horizon, "target_step": target_step}
            if target is None and target_step is not None and last is not None and target_step > last:
                record["response_status"] = "STRUCTURAL_NA_BEYOND_EPISODE_END"
            elif target is None:
                raise AuditError(f"missing exact assist-response step within episode: {episode_id} step {target_step}")
            else:
                record["response_status"] = "OBSERVED_EXACT_STEP"
            for field in ASSIST_FIELDS:
                before = number(row.get(field))
                after = number(target.get(field)) if target else None
                if field in {"estimated_lateral_velocity_mps", "actual_lateral_velocity_mps"}:
                    before = abs(before) if before is not None else None
                    after = abs(after) if after is not None else None
                record[f"{field}_at_assist"] = before
                record[f"{field}_at_horizon"] = after
                record[f"delta_{field}"] = after - before if after is not None and before is not None else None
            dm = record.get("delta_passed_margin_m")
            record["pass_margin_response"] = "PASS_MARGIN_IMPROVES" if dm is not None and dm > 0 else "PASS_MARGIN_WORSENS" if dm is not None and dm < 0 else "PASS_MARGIN_FLAT" if dm == 0 else "STRUCTURAL_NA"
            output.append(record)
    return output


def _classify_goal_pass(goal_delta: float | None, pass_delta: float | None) -> str:
    if goal_delta is None or pass_delta is None:
        return "UNAVAILABLE"
    goal_axis = "GOAL_IMPROVES" if goal_delta < 0 else "GOAL_WORSENS" if goal_delta > 0 else "GOAL_ZERO"
    pass_axis = "PASS_IMPROVES" if pass_delta > 0 else "PASS_WORSENS" if pass_delta < 0 else "PASS_ZERO"
    return f"{goal_axis}_{pass_axis}"


def _goal_pass_rows(condition: str, episode_id: str, cohort: str, rows: Sequence[Mapping[str, Any]], ready: int | None, best: int | None) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) if integer(row.get("step")) is not None else -1)
    output = []
    for left, right in zip(ordered, ordered[1:]):
        step, next_step = integer(left.get("step")), integer(right.get("step"))
        if step is None or next_step != step + 1:
            continue
        gd = number(right.get("goal_distance_m"))
        gc = number(left.get("goal_distance_m"))
        md = number(right.get("passed_margin_m"))
        mc = number(left.get("passed_margin_m"))
        goal_delta = gd - gc if gd is not None and gc is not None else None
        pass_delta = md - mc if md is not None and mc is not None else None
        phase = "PRE_BEST" if ready is not None and best is not None and step >= ready and next_step <= best else "POST_BEST" if best is not None and step >= best else "OTHER"
        active = flag(left.get("radial_intent_assist_active"))
        output.append({
            "condition": condition, "episode_id": episode_id, "cohort": cohort,
            "step": step, "next_step": next_step, "goal_distance_delta_m": goal_delta,
            "pass_margin_delta_m": pass_delta, "goal_improves": goal_delta < 0 if goal_delta is not None else None,
            "pass_improves": pass_delta > 0 if pass_delta is not None else None,
            "classification": _classify_goal_pass(goal_delta, pass_delta), "window": phase,
            "radial_assist_active_at_step": active,
        })
    return output


def _goal_pass_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((str(row["condition"]), str(row["window"])), []).append(row)
        if flag(row.get("radial_assist_active_at_step")):
            buckets.setdefault((str(row["condition"]), "RADIAL_ACTIVE"), []).append(row)
        if row.get("cohort") == "MAIN_FAILURE":
            buckets.setdefault((str(row["condition"]), f"MAIN_{row['window']}"), []).append(row)
    output = []
    for (condition, window), group in sorted(buckets.items()):
        valid = [row for row in group if row.get("classification") != "UNAVAILABLE"]
        counts: dict[str, int] = {}
        for row in valid:
            key = str(row["classification"])
            counts[key] = counts.get(key, 0) + 1
        gi_pw = counts.get("GOAL_IMPROVES_PASS_WORSENS", 0)
        goal_improves = sum(1 for row in valid if row.get("goal_improves") is True)
        output.append({
            "condition": condition, "window": window, "transition_count": len(group),
            "valid_transition_count": len(valid), "classification_counts": counts,
            "goal_improves_pass_worsens_count": gi_pw,
            "goal_improves_pass_worsens_fraction_all_valid": gi_pw / len(valid) if valid else None,
            "goal_improves_pass_worsens_fraction_given_goal_improves": gi_pw / goal_improves if goal_improves else None,
        })
    return output


def _special_rows(analysis: Mapping[str, Any], episode_rows: Mapping[str, Mapping[str, Mapping[str, str]]], step_groups: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]]) -> list[dict[str, Any]]:
    output = []
    kinds = {RESCUE_ID: "RESCUE", SAFETY_ID: "SAFETY_SPECIAL", BOUNDARY_ID: "BOUNDARY_SPECIAL"}
    for episode_id, kind in kinds.items():
        for condition in ("off", "on"):
            rows = step_groups[condition][episode_id]
            metric = analysis[(condition, episode_id)]
            active = [row for row in rows if flag(row.get("radial_intent_assist_active"))]
            safety_rows = [row for row in rows if flag(row.get("safety_projection_applied"))]
            boundary_rows = [row for row in rows if flag(row.get("boundary_projection_applied"))]
            if kind == "SAFETY_SPECIAL":
                overlaps = [row for row in rows if flag(row.get("radial_intent_assist_active")) and flag(row.get("safety_projection_applied"))]
                special_evidence = [{"step": integer(row.get("step")), "clearance_m": _clearance(row), "assist_mps": number(row.get("applied_radial_assist_mps")), "margin_m": number(row.get("passed_margin_m"))} for row in overlaps]
                count_name = "assist_safety_overlap_count"
            elif kind == "BOUNDARY_SPECIAL":
                overlaps = [row for row in rows if flag(row.get("boundary_projection_applied"))]
                special_evidence = [{"step": integer(row.get("step")), "clearance_m": _clearance(row), "assist_mps": number(row.get("applied_radial_assist_mps")), "margin_m": number(row.get("passed_margin_m")), "boundary_delta_v_mps": row.get("motion_boundary_delta_v_mps")} for row in overlaps]
                count_name = "boundary_projection_count"
            else:
                overlaps = []
                special_evidence = []
                count_name = "not_applicable"
            output.append({
                "episode_id": episode_id, "condition": condition, "case_type": kind,
                "passed_obstacle": flag(episode_rows[condition][episode_id].get("passed_obstacle")),
                "first_pass_step": metric["first_pass_step"], "best_margin_m": metric["margin_best_m"],
                "final_margin_m": metric["margin_final_m"], "first_assist_step": metric["first_assist_active_step"],
                "assist_active_rows": len(active), count_name: len(overlaps),
                "safety_projection_step_count": len(safety_rows),
                "boundary_projection_step_count": len(boundary_rows),
                "first_special_step": integer(overlaps[0].get("step")) if overlaps else None,
                "first_special_clearance_m": _clearance(overlaps[0]) if overlaps else None,
                "first_special_assist_mps": number(overlaps[0].get("applied_radial_assist_mps")) if overlaps else None,
                "first_special_margin_m": number(overlaps[0].get("passed_margin_m")) if overlaps else None,
                "special_evidence_rows": json.dumps(special_evidence, separators=(",", ":")),
                "case_role": "descriptive only; excluded from main six PASS_COMMIT vote" if episode_id != RESCUE_ID else "separate rescue subgroup; excluded from main six PASS_COMMIT vote",
            })
    return output


def _main_comparisons(analysis: Mapping[tuple[str, str], Mapping[str, Any]], episode_rows: Mapping[str, Mapping[str, Mapping[str, str]]]) -> list[dict[str, Any]]:
    output = []
    for episode_id in MAIN_IDS:
        off, on = analysis[("off", episode_id)], analysis[("on", episode_id)]
        off_ep, on_ep = episode_rows["off"][episode_id], episode_rows["on"][episode_id]
        best_delta = number(on["margin_best_m"])
        off_best = number(off["margin_best_m"])
        best_margin_delta = best_delta - off_best if best_delta is not None and off_best is not None else None
        clearance_delta = number(on["best_clearance_m"])
        off_clearance = number(off["best_clearance_m"])
        best_clearance_delta = clearance_delta - off_clearance if clearance_delta is not None and off_clearance is not None else None
        off_min, on_min = number(off_ep.get("minimum_goal_distance_m")), number(on_ep.get("minimum_goal_distance_m"))
        goal_distance_improvement = off_min - on_min if off_min is not None and on_min is not None else None
        improved_dimensions = []
        if best_margin_delta is not None and best_margin_delta > 0.0:
            improved_dimensions.append("BEST_PASS_MARGIN")
        if best_clearance_delta is not None and best_clearance_delta > 0.0:
            improved_dimensions.append("BEST_CLEARANCE")
        if goal_distance_improvement is not None and goal_distance_improvement > 0.0:
            improved_dimensions.append("MINIMUM_GOAL_DISTANCE")
        off_pass, on_pass = flag(off_ep.get("passed_obstacle")), flag(on_ep.get("passed_obstacle"))
        label = "PASS_GAIN" if on_pass and not off_pass else "LOCAL_MECHANISM_IMPROVED_BUT_NO_PASS" if improved_dimensions and not on_pass else "NO_MEANINGFUL_GEOMETRIC_IMPROVEMENT"
        output.append({
            "episode_id": episode_id, "off_best_margin_m": off["margin_best_m"], "off_best_step": off["best_margin_step"],
            "off_final_margin_m": off["margin_final_m"], "off_best_to_final_regression": off["best_to_final_regression"],
            "on_best_margin_m": on["margin_best_m"], "on_best_step": on["best_margin_step"],
            "on_final_margin_m": on["margin_final_m"], "on_best_to_final_regression": on["best_to_final_regression"],
            "best_margin_delta_on_minus_off_m": best_margin_delta,
            "off_first_assist_active_step": off["first_assist_active_step"], "on_first_assist_active_step": on["first_assist_active_step"],
            "off_minimum_goal_distance_m": off_min, "on_minimum_goal_distance_m": on_min,
            "minimum_goal_distance_improvement_off_minus_on_m": goal_distance_improvement,
            "best_clearance_delta_on_minus_off_m": best_clearance_delta,
            "off_entered_8m": flag(off_ep.get("entered_8m")), "on_entered_8m": flag(on_ep.get("entered_8m")),
            "off_passed": off_pass, "on_passed": on_pass,
            "exact_positive_improvement_dimensions": json.dumps(improved_dimensions, separators=(",", ":")),
            "label": label,
        })
    return output


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(",", ":"), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    return stream.getvalue()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _report(summary: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]], special: Sequence[Mapping[str, Any]]) -> str:
    passes = summary["validation_checks"]["pass_counts"]
    supported = summary["pass_commit"]["supported_episode_count"]
    lines = [
        "# Single-UAV Pre-pass Handoff / Pass-progress Audit",
        "",
        "Descriptive, frozen-trajectory audit only. The paired OFF/ON contrasts are not presented as uniquely causal proof or as a controller recommendation.",
        "",
        "## A. Inputs and identity",
        "",
        f"Active preregistration SHA-256: `{summary['active_preregistration_sha256']}`. Runtime source-lock checks passed for {len(summary['runtime_source_hashes_checked'])} frozen source files. Input bytes matched before/after analysis: `{summary['input_hashes_unchanged']}`.",
        "",
        "## B. Cohorts and exact pass gate",
        "",
        f"Main failure cohort: {', '.join(summary['cohorts']['main_failure_ids'])}. Rescue: `{RESCUE_ID}`. Prior-summary success reference contains {len(summary['cohorts']['success_reference_ids'])} exact IDs. Specials `{SAFETY_ID}` and `{BOUNDARY_ID}` are descriptive only.",
        f"Computed passed episodes: OFF {passes['off']}/20, ON {passes['on']}/20. A pass row requires the frozen `passed_obstacle` latch, `passed_margin_m > 0`, and same-row relevant obstacle clearance `>= {PASS_CLEARANCE_M} m`; `passed_condition_now` was checked against margin and clearance on every finite row.",
        "",
        "## C. Literal handoff timing",
        "",
        "The timing table records encounter, first estimated lateral-exit readiness, first assist eligibility/activity, last active step, earliest step attaining best margin, first exact pass, first post-bypass recovery, and end. `PRE_BEST` is [first ready, best] inclusive; `POST_BEST` is (best, end] and contains no repeated best row. Assist timing labels are relative to the earliest best step.",
        "",
        "## D. Pass-progress windows",
        "",
        "Start/best/final margins, best-minus-start and best-minus-final, OLS slopes, and exact negative adjacent-margin transition counts/runs/fractions are in the episode and window tables. The descriptive near-pass band is the preregistered-safe frozen interval [-1, 0] m and is not used as a gate or recommendation.",
        "",
        "## E. Assist event response",
        "",
        "Every ON radial-assist-active row has exact +1/+3/+5-step responses for margin, obstacle clearance, goal distance, radial velocities, and absolute lateral velocities. Horizons beyond an episode end are `STRUCTURAL_NA`; no interpolation is used.",
        "",
        "## F. Goal distance versus pass progress",
        "",
        "Each contiguous t→t+1 pair is classified from goal-distance delta (<0 improves) and pass-margin delta (>0 improves). The detailed table and summary aggregates retain all four nonzero quadrants and explicit zero-axis combinations. `GOAL_IMPROVES_PASS_WORSENS` is reported both among all valid pairs and conditional on goal-distance improvement.",
        "",
        "## G. Lateral motion after best margin",
        "",
        "Post-best peak absolute estimated/actual lateral speed, exact increases in absolute speed, longest reacceleration runs, actor lateral sign changes, and exact crossings of |lateral speed|=0.5 m/s are recorded in episode diagnostics.",
        "",
        "## H. Frozen radial frame angle",
        "",
        "For pre-best, post-best, and radial-assist-active rows, current estimated XY position is reconstructed as obstacle center minus obstacle-relative vector; the goal is encounter estimated position plus frozen horizontal goal ray. Angles compare current XY goal direction with the frozen radial unit. Degenerate/missing vectors remain structural unavailable values.",
        "",
        "## I. Obstacle geometry and special cases",
        "",
        "The episode table records along-obstacle, clearance, and margin at start/best/end. `OBSTACLE_RELATIVE_REGRESSION` means post-best along-obstacle increases while pass margin falls. The rescue, safety-overlap, and boundary-projection episodes are retained in the special-case table and excluded from the six-episode vote.",
        "",
        "## J. Paired interpretation and PASS_COMMIT",
        "",
        f"The six main-episode comparisons use exact positive improvement (>0) in best margin, best clearance, or minimum goal distance; no tuned epsilon is applied. PASS_COMMIT supports {supported}/6 main episodes; the frozen recommendation gate (at least 4/6) is `{summary['pass_commit']['recommendation_gate_met']}`.",
        f"The only next recommendation is `{summary['pass_commit']['recommendation']}`.",
        "",
        "Main paired labels:",
        "",
    ]
    lines.extend(f"- `{row['episode_id']}`: `{row['label']}`; best-margin ON−OFF={row['best_margin_delta_on_minus_off_m']}; ON pass={row['on_passed']}." for row in comparisons)
    lines.extend(["", "PASS_COMMIT requires all five preregistered checks per main ON episode: exit readiness; nonempty post-best regression evidence; Goal Recovery inactive across best to first regression; best clearance ≥2.5 m with no collision and no safety projection at best; and active radial assistance with regression persisting while Goal Recovery is inactive. Support remains a descriptive mechanism signal and does not establish unique cause.", ""])
    return "\n".join(lines)


def _cohort(episode_id: str, success_ids: Sequence[str]) -> str:
    if episode_id in MAIN_IDS:
        return "MAIN_FAILURE"
    if episode_id == RESCUE_ID:
        return "RESCUE"
    if episode_id in {SAFETY_ID, BOUNDARY_ID}:
        return "SPECIAL"
    if episode_id in success_ids:
        return "SUCCESS_REFERENCE"
    return "OTHER"


def _input_paths(root: Path, explicit_lateral: Path | None) -> tuple[dict[str, Path], Path | None]:
    paths = {
        "off_steps": root / RUN_REL / "off" / "steps.csv",
        "off_episodes": root / RUN_REL / "off" / "episodes.csv",
        "off_summary": root / RUN_REL / "off" / "summary.json",
        "on_steps": root / RUN_REL / "on" / "steps.csv",
        "on_episodes": root / RUN_REL / "on" / "episodes.csv",
        "on_summary": root / RUN_REL / "on" / "summary.json",
        "evaluation_summary": root / RUN_REL / "evaluation_summary.json",
        "comparison_episodes": root / RUN_REL / "comparison_episodes.csv",
        "active_preregistration": root / PREP_REL / "preregistered_radial_intent_persistence_endpoints.json",
        "runtime_source_lock": root / PREP_REL / "radial_intent_runtime_source_lock.json",
        "preregistration_lock": root / PREP_REL / "radial_intent_preregistration_lock.json",
        "prior_prepass_summary": root / PRIOR_REL,
    }
    lateral = explicit_lateral
    if lateral is not None and not lateral.is_absolute():
        lateral = root / lateral
    if lateral is None:
        candidates = (
            root / "logs/preliminary/single_uav_lateral_exit_authority_audit_seed2026/lateral_exit_authority_audit_summary.json",
            root / "logs/preliminary/single_uav_lateral_exit_authority_audit_seed2026/lateral_exit_authority_summary.json",
            root / "logs/preliminary/single_uav_post_lateral_exit_reorientation_audit_seed2026/post_lateral_exit_reorientation_summary.json",
        )
        lateral = next((path for path in candidates if path.is_file()), None)
    return paths, lateral


def run_audit(project_root: Path | str | None = None, *, output_root: Path | str | None = None, prior_lateral_summary: Path | str | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    output = Path(output_root).resolve() if output_root is not None else root / OUTPUT_REL
    output_path = output
    input_paths, lateral_path = _input_paths(root, Path(prior_lateral_summary) if prior_lateral_summary else None)
    if lateral_path is not None:
        input_paths["optional_prior_lateral_summary"] = lateral_path
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    _check(not missing, "required input files missing: " + ", ".join(missing))
    _check(not output.exists(), f"refusing to overwrite existing output root: {output}")
    hashes_before = {name: sha256_file(path) for name, path in input_paths.items()}
    prep = _json(input_paths["active_preregistration"])
    runtime_lock = _json(input_paths["runtime_source_lock"])
    prereg_lock = _json(input_paths["preregistration_lock"])
    prior = _json(input_paths["prior_prepass_summary"])
    evaluation = _json(input_paths["evaluation_summary"])
    prep_hash = _expected_prep_hash(prep, prereg_lock, runtime_lock)
    runtime_hashes = _runtime_lock_checks(root, runtime_lock, evaluation)
    summary_checks = _validate_summary_identity(evaluation, prep_hash, prep)
    success_ids = (prior.get("cohorts") or {}).get("success_reference_episode_ids")
    _check(isinstance(success_ids, list) and len(success_ids) == 11 and len(set(success_ids)) == 11, "prior summary success reference must contain exactly 11 unique IDs")
    _check(set(success_ids).isdisjoint(set(MAIN_IDS) | {RESCUE_ID, SAFETY_ID, BOUNDARY_ID}), "prior success reference overlaps fixed non-success cohorts")
    _check(set(prior.get("cohorts", {}).get("main_failure_episode_ids", [])) == set(MAIN_IDS) | {RESCUE_ID}, "prior main-failure cohort differs from fixed failure plus rescue cohort")
    _check(set(prior.get("cohorts", {}).get("special_excluded_episode_ids", [])) == {SAFETY_ID, BOUNDARY_ID}, "prior special cohort mismatch")
    per_condition, _, episode_csv, step_csv, other = _load_trajectories(root, evaluation)
    comparison = other["comparison"]
    trajectory_state = _validate_trajectories(root, evaluation, per_condition, episode_csv, step_csv, comparison)
    episode_rows = trajectory_state["episode_rows"]
    step_groups = trajectory_state["step_groups"]
    analysis: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    assist_rows: list[dict[str, Any]] = []
    goal_rows: list[dict[str, Any]] = []
    for condition in ("off", "on"):
        for episode_id in EXPECTED_IDS:
            cohort = _cohort(episode_id, success_ids)
            diagnostic, timing, windows = _episode_metrics(condition, episode_id, step_groups[condition][episode_id], episode_rows[condition][episode_id], cohort)
            analysis[(condition, episode_id)] = diagnostic
            diagnostics_rows.append(diagnostic)
            timing_rows.append(timing)
            window_rows.extend(_window_rows(diagnostic, windows))
            if condition == "on":
                assist_rows.extend(_assist_response(episode_id, cohort, step_groups[condition][episode_id]))
            goal_rows.extend(_goal_pass_rows(condition, episode_id, cohort, step_groups[condition][episode_id], diagnostic["first_ready_step"], diagnostic["best_margin_step"]))
    main_comparisons = _main_comparisons(analysis, episode_rows)
    special_rows = _special_rows(analysis, episode_rows, step_groups)
    pass_commit_rows = [{"episode_id": episode_id, **analysis[("on", episode_id)]["pass_commit"]} for episode_id in MAIN_IDS]
    supported = sum(bool(row["supported"]) for row in pass_commit_rows)
    main_diagnoses = [
        {
            "episode_id": episode_id,
            "primary_diagnoses": analysis[("on", episode_id)]["primary_diagnoses"],
            "assist_timing_vs_best": analysis[("on", episode_id)]["assist_timing_vs_best"],
            "assist_start_minus_best_margin_step": next(row["assist_start_minus_best_margin_step"] for row in timing_rows if row["condition"] == "on" and row["episode_id"] == episode_id),
            "best_pass_limiter": analysis[("on", episode_id)]["best_pass_limiter"],
            "near_pass_but_regressed": analysis[("on", episode_id)]["near_pass_but_regressed"],
            "margin_best_m": analysis[("on", episode_id)]["margin_best_m"],
            "margin_final_m": analysis[("on", episode_id)]["margin_final_m"],
            "post_best_goal_improves_pass_worsens_count": analysis[("on", episode_id)]["post_best_goal_improves_pass_worsens_count"],
            "radial_assist_active_goal_improves_pass_worsens_count": analysis[("on", episode_id)]["radial_assist_active_goal_improves_pass_worsens_count"],
            "post_best_actual_lateral_reacceleration_step_count": analysis[("on", episode_id)]["post_best_actual_lateral_diagnostics"]["reacceleration_step_count"],
            "post_best_estimated_lateral_reacceleration_step_count": analysis[("on", episode_id)]["post_best_estimated_lateral_diagnostics"]["reacceleration_step_count"],
            "frame_angle_post_best": analysis[("on", episode_id)]["frame_angle_post_best"],
            "frame_angle_radial_assist_active": analysis[("on", episode_id)]["frame_angle_radial_assist_active"],
            "obstacle_geometry_flag": analysis[("on", episode_id)]["obstacle_geometry_flag"],
        }
        for episode_id in MAIN_IDS
    ]
    rescue_on = analysis[("on", RESCUE_ID)]
    rescue_off = analysis[("off", RESCUE_ID)]
    safety_special = {row["condition"]: row for row in special_rows if row["episode_id"] == SAFETY_ID}
    boundary_special = {row["condition"]: row for row in special_rows if row["episode_id"] == BOUNDARY_ID}
    output = {
        "schema_version": "single_uav_prepass_handoff_pass_progress_audit_v1",
        "status": "PASS",
        "interpretation": "Frozen-trajectory descriptive mechanism audit; paired contrasts are not unique-cause proof.",
        "active_preregistration_sha256": prep_hash,
        "runtime_source_hashes_checked": runtime_hashes,
        "input_hashes_before": hashes_before,
        "input_hashes_after": {},
        "input_hashes_unchanged": None,
        "optional_prior_lateral_summary": str(lateral_path) if lateral_path else None,
        "cohorts": {"main_failure_ids": list(MAIN_IDS), "rescue_id": RESCUE_ID, "success_reference_ids": success_ids, "safety_special_id": SAFETY_ID, "boundary_special_id": BOUNDARY_ID, "pass_commit_vote_ids": list(MAIN_IDS)},
        "validation_checks": {**summary_checks, **trajectory_state["checks"], "source_pass_condition_mismatch_count": len(trajectory_state["source_pass_condition_mismatches"]), "source_pass_condition_consistency": True, "runtime_source_lock": True, "active_preregistration_lock": True, "prior_success_cohort_exact": True},
        "pass_gate": {"exact_rule": "passed_obstacle latch AND passed_margin_m > 0 AND same-row relevant_obstacle_clearance_m >= 2.5", "clearance_threshold_m": PASS_CLEARANCE_M, "limiter_classes": ["MARGIN_LIMITED", "CLEARANCE_LIMITED", "BOTH_LIMITED"], "source_passed_condition_consistency": "passed_condition_now equals (passed_margin_m > 0 AND relevant_obstacle_clearance_m >= 2.5) whenever both inputs are finite"},
        "pass_counts": trajectory_state["checks"]["pass_counts"],
        "timing_windows": {"PRE_BEST": "[first estimated_lateral_exit_ready, earliest best-margin step] inclusive", "POST_BEST": "(earliest best-margin step, final logged step]", "no_overlap": True},
        "near_pass_descriptive_band_m": list(NEAR_PASS_BAND_M),
        "assist_response_horizons_steps": [1, 3, 5],
        "goal_vs_pass_progress_aggregates": _goal_pass_aggregates(goal_rows),
        "main_off_on_comparison": main_comparisons,
        "main_failure_diagnoses": main_diagnoses,
        "rescue_control": {
            "episode_id": RESCUE_ID,
            "off_passed": rescue_off["episode_passed_obstacle"], "on_passed": rescue_on["episode_passed_obstacle"],
            "off_best_margin_m": rescue_off["margin_best_m"], "on_best_margin_m": rescue_on["margin_best_m"],
            "on_first_assist_active_step": rescue_on["first_assist_active_step"], "on_best_margin_step": rescue_on["best_margin_step"],
            "on_first_pass_step": rescue_on["first_pass_step"], "on_assist_timing_vs_best": rescue_on["assist_timing_vs_best"],
            "on_best_pass_limiter": rescue_on["best_pass_limiter"],
            "on_post_best_goal_improves_pass_worsens_count": rescue_on["post_best_goal_improves_pass_worsens_count"],
            "on_post_best_actual_lateral_diagnostics": rescue_on["post_best_actual_lateral_diagnostics"],
            "on_frame_angle_post_best": rescue_on["frame_angle_post_best"],
        },
        "special_cases": {
            "0013_safety": {
                "role": "descriptive only; excluded from main diagnosis and vote",
                "off_safety_projection_step_count": safety_special["off"]["safety_projection_step_count"],
                "on_safety_projection_step_count": safety_special["on"]["safety_projection_step_count"],
                "on_assist_safety_overlap_count": safety_special["on"].get("assist_safety_overlap_count", 0),
                "first_on_overlap_step": safety_special["on"]["first_special_step"],
                "first_on_overlap_clearance_m": safety_special["on"]["first_special_clearance_m"],
                "first_on_overlap_assist_mps": safety_special["on"]["first_special_assist_mps"],
                "first_on_overlap_margin_m": safety_special["on"]["first_special_margin_m"],
                "interpretation": "Temporal overlap is descriptive association only and does not establish that radial assistance caused the additional safety projections.",
            },
            "0014_boundary": {
                "role": "descriptive only; excluded from main diagnosis and vote",
                "off_boundary_projection_step_count": boundary_special["off"]["boundary_projection_step_count"],
                "on_boundary_projection_step_count": boundary_special["on"]["boundary_projection_step_count"],
                "note": "Boundary projection remains present in both frozen arms; this case is boundary-dominated and is not evidence for the main handoff mechanism.",
            },
        },
        "pass_commit": {"criteria": {"C1": "estimated lateral-exit readiness observed", "C2": "post-best window nonempty, best-final > 0, at least one negative adjacent pass-margin transition", "C3": "Goal Recovery inactive from best through first regression evidence; pre-best and whole-prepass activity counts recorded", "C4": "best clearance >=2.5 m, no collision, best row not safety projected", "C5": "radial assistance active and post-best regression persists over Goal-Recovery-inactive evidence interval"}, "episodes": pass_commit_rows, "supported_episode_count": supported, "eligible_main_episode_count": len(MAIN_IDS), "recommendation_threshold": 4, "recommendation_gate_met": supported >= 4, "recommendation": "PREPARE_MINIMAL_PASS_COMMIT_CAUSAL_DIAGNOSTIC" if supported >= 4 else "CONTINUE_OFFLINE_MECHANISM_AUDIT"},
        "special_case_counts": {"rescue_rows": sum(row["episode_id"] == RESCUE_ID for row in special_rows), "safety_overlap_rows": sum(integer(row.get("assist_safety_overlap_count")) or 0 for row in special_rows), "boundary_projection_rows": sum(integer(row.get("boundary_projection_count")) or 0 for row in special_rows)},
        "output_files": list(OUTPUT_FILES),
    }
    hashes_after = {name: sha256_file(path) for name, path in input_paths.items()}
    _check(hashes_before == hashes_after, "one or more frozen input bytes changed during analysis")
    output["input_hashes_after"] = hashes_after
    output["input_hashes_unchanged"] = True
    report = _report(output, main_comparisons, special_rows)
    artifacts = {
        OUTPUT_FILES[0]: json.dumps(_json_safe(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        OUTPUT_FILES[1]: _csv_text(diagnostics_rows),
        OUTPUT_FILES[2]: _csv_text(timing_rows),
        OUTPUT_FILES[3]: _csv_text(window_rows),
        OUTPUT_FILES[4]: _csv_text(assist_rows),
        OUTPUT_FILES[5]: _csv_text(goal_rows + [{"record_type": "AGGREGATE", **row} for row in output["goal_vs_pass_progress_aggregates"]]),
        OUTPUT_FILES[6]: _csv_text(special_rows),
        OUTPUT_FILES[7]: report,
    }
    _check(tuple(artifacts) == OUTPUT_FILES and len(artifacts) == 8, "internal output contract mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.stage-", dir=output_path.parent))
    try:
        for name, content in artifacts.items():
            target = stage / name
            with target.open("w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        _check({path.name for path in stage.iterdir()} == set(OUTPUT_FILES), "staging output set mismatch")
        _check(not output_path.exists(), f"refusing to overwrite existing output root: {output_path}")
        os.replace(stage, output_path)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    _check({path.name for path in output_path.iterdir()} == set(OUTPUT_FILES), "published output set mismatch")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--prior-lateral-summary", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_audit(args.project_root, output_root=args.output_root, prior_lateral_summary=args.prior_lateral_summary)
    except (AuditError, OSError, ValueError) as exc:
        print(f"PREPASS_HANDOFF_AUDIT_FAILED: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"PREPASS_HANDOFF_AUDIT_PASS: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
