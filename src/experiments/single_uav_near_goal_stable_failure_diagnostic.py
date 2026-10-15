"""Offline diagnosis of near-goal stable failures in correction-ON logs.

This module only reads existing CSV/JSON artifacts.  It never imports or runs
the simulator, a policy, a controller, a Teacher, or a training routine.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
import statistics
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE = "single_uav_near_goal_stable_failure_diagnostic"
DEFAULT_EVALUATION_ROOT = Path(
    "logs/preliminary/single_uav_goal_recovery_correction_eval_formal20_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/diagnostics/single_uav_near_goal_stable_failure_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/diagnostics/single_uav_near_goal_stable_failure_seed2026"
)
STABLE_SPEED_THRESHOLD_MPS = 0.5
MODE_ORDER = (
    "HIGH_ENTRY_SPEED",
    "CORRECTION_OVERSHOOT",
    "TANGENTIAL_RESIDUAL",
    "ACTOR_CORRECTION_CONFLICT",
    "INSUFFICIENT_BRAKING",
    "GOAL_REGION_OSCILLATION",
    "STABLE_HOLD_INTERRUPTED",
)
PRIMARY_PRIORITY = (
    "CORRECTION_OVERSHOOT",
    "GOAL_REGION_OSCILLATION",
    "INSUFFICIENT_BRAKING",
    "ACTOR_CORRECTION_CONFLICT",
    "TANGENTIAL_RESIDUAL",
    "HIGH_ENTRY_SPEED",
    "STABLE_HOLD_INTERRUPTED",
)
RULES: dict[str, Any] = {
    "version": "near_goal_failure_rules_v1",
    "magnitude_epsilon": 1.0e-6,
    "distance_rebound_tolerance_m": 0.05,
    "noticeable_radial_correction_mps": 0.05,
    "conflict_fraction_threshold": 0.5,
    "weak_braking_delta_threshold_mps": -0.05,
    "insufficient_braking_fraction_threshold": 0.5,
    "tangential_fraction_threshold": 0.5,
    "oscillation_min_entries": 2,
    "oscillation_min_exits": 1,
    "primary_priority": list(PRIMARY_PRIORITY),
}
DIAGNOSTIC_ONLY_FLAGS: dict[str, Any] = {
    "training_performed": False,
    "rollout_executed": False,
    "ppo_updates": 0,
    "mappo_updates": 0,
    "bc_retraining_executed": False,
    "dagger_collection_executed": False,
    "teacher_rollout_executed": False,
    "control_algorithm_modified": False,
}

EPISODE_REQUIRED = {
    "condition", "episode_id", "entered_1m", "stable_goal",
    "first_entered_1m_step", "goal_threshold_m", "stable_hold_steps",
}
STEP_REQUIRED = {
    "condition", "episode_id", "step", "goal_distance_m",
    "goal_distance_after_m", "current_radial_velocity_mps",
    "current_tangential_velocity_mps", "desired_radial_velocity_mps",
    "actor_raw_delta_v_mps", "goal_recovery_correction_active",
    "goal_recovery_correction_alpha", "goal_recovery_correction_delta_v_mps",
    "goal_recovery_correction_norm_mps", "goal_recovery_radial_delta_v_mps",
    "goal_recovery_tangential_delta_v_mps", "corrected_delta_v_pre_transform_mps",
    "corrected_delta_v_pre_safety_mps", "applied_delta_v_mps",
    "safety_projection_applied", "boundary_projection_applied",
    "passed_obstacle", "post_bypass_recovery", "stable_goal",
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int:
    number = _finite(value)
    if number is None or int(number) != number:
        raise ValueError(f"expected integer value, got {value!r}")
    return int(number)


def _vector(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                raw = ast.literal_eval(str(value))
            except (ValueError, SyntaxError):
                return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    parsed = [_finite(item) for item in raw]
    return None if any(item is None for item in parsed) else [float(item) for item in parsed]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _goal_direction(row: Mapping[str, Any]) -> list[float] | None:
    """Recover the logged goal unit direction without simulator state."""
    radial = _vector(row.get("goal_recovery_radial_delta_v_mps"))
    current = _finite(row.get("current_radial_velocity_mps"))
    desired = _finite(row.get("desired_radial_velocity_mps"))
    if radial is None or current is None or desired is None or _norm(radial) <= RULES["magnitude_epsilon"]:
        return None
    sign = 1.0 if desired - current > 0.0 else -1.0
    return [sign * value / _norm(radial) for value in radial]


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        return [dict(row) for row in reader], fields


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def discover_formal_on_logs(root: Path) -> tuple[Path, Path, Path]:
    """Find one unambiguous ON episode/step pair by schema signatures."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates: list[tuple[Path, Path]] = []
    csv_files = list(root.rglob("*.csv"))
    signatures: dict[Path, set[str]] = {}
    for path in csv_files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            signatures[path] = set(next(csv.reader(stream), []))
    episode_files = [path for path, fields in signatures.items() if EPISODE_REQUIRED <= fields]
    step_files = [path for path, fields in signatures.items() if STEP_REQUIRED <= fields]
    for episodes in episode_files:
        for steps in step_files:
            if episodes.parent == steps.parent:
                candidates.append((episodes, steps))
    valid: list[tuple[Path, Path]] = []
    for episodes, steps in candidates:
        ep_rows, _ = _read_csv(episodes)
        st_rows, _ = _read_csv(steps)
        if ep_rows and st_rows and all(row.get("condition", "").lower() == "on" for row in ep_rows + st_rows):
            valid.append((episodes, steps))
    if len(valid) != 1:
        rendered = [f"{a} | {b}" for a, b in valid]
        raise ValueError(f"expected one unambiguous ON episode/step pair, found {len(valid)}: {rendered}")
    summary = root / "evaluation_summary.json"
    if not summary.is_file():
        matches = [path for path in root.glob("*.json") if isinstance(_read_json(path), dict)]
        formal = [path for path in matches if _read_json(path).get("formal") is True]
        if len(formal) != 1:
            raise ValueError("formal paired evaluation summary is missing or ambiguous")
        summary = formal[0]
    payload = _read_json(summary)
    if payload.get("stage") != "single_uav_goal_recovery_correction_evaluation" or payload.get("formal") is not True:
        raise ValueError(f"not a formal Goal Recovery Correction evaluation: {summary}")
    return valid[0][0], valid[0][1], summary


def load_inputs(root: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    episodes_path, steps_path, summary_path = discover_formal_on_logs(root)
    episodes, _ = _read_csv(episodes_path)
    steps, _ = _read_csv(steps_path)
    episode_ids = {row["episode_id"] for row in episodes}
    grouped: dict[str, list[dict[str, str]]] = {episode_id: [] for episode_id in episode_ids}
    seen: set[tuple[str, int]] = set()
    for row in steps:
        episode_id = row["episode_id"]
        step = _int(row["step"])
        key = (episode_id, step)
        if episode_id not in grouped:
            raise ValueError(f"step row has no episode summary: {episode_id}")
        if key in seen:
            raise ValueError(f"duplicate step key: {key}")
        seen.add(key)
        grouped[episode_id].append(row)
    for episode_id, rows in grouped.items():
        rows.sort(key=lambda row: _int(row["step"]))
        values = [_int(row["step"]) for row in rows]
        if values != sorted(values):
            raise ValueError(f"non-monotonic steps: {episode_id}")
    metadata = {
        "evaluation_root": str(root.resolve()),
        "episode_summary_file": str(episodes_path),
        "step_log_file": str(steps_path),
        "evaluation_summary_file": str(summary_path),
        "evaluation_summary": _read_json(summary_path),
    }
    return episodes, grouped, metadata


def select_failure_episodes(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if _int(row["entered_1m"]) == 1 and _int(row["stable_goal"]) == 0]


def _aligned_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        step = _int(row["step"])
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        contiguous = next_row is not None and _int(next_row["step"]) == step + 1
        radial = _finite(next_row.get("current_radial_velocity_mps")) if contiguous else None
        tangential = _finite(next_row.get("current_tangential_velocity_mps")) if contiguous else None
        speed = math.hypot(radial, tangential) if radial is not None and tangential is not None else None
        current_radial = _finite(row.get("current_radial_velocity_mps"))
        row.update({
            "post_state_proxy_source_step": _int(next_row["step"]) if contiguous else None,
            "post_radial_velocity_proxy_mps": radial,
            "post_tangential_velocity_proxy_mps": tangential,
            "post_total_speed_proxy_mps": speed,
            "post_total_speed_proxy_derived": speed is not None,
            "radial_velocity_change_proxy_mps": (
                radial - current_radial if radial is not None and current_radial is not None else None
            ),
        })
        aligned.append(row)
    return aligned


def _mode(evaluated: bool, matched: bool, evidence: Mapping[str, Any], steps: Sequence[int],
          required: Sequence[str], missing: Sequence[str] = (), substitutions: Sequence[str] = (),
          confidence: str = "proxy") -> dict[str, Any]:
    return {
        "evaluated": bool(evaluated), "matched": bool(matched),
        "evidence": dict(evidence), "evidence_steps": list(steps),
        "required_fields": list(required), "missing_fields": list(missing),
        "substitutions": list(substitutions), "confidence": confidence,
    }


def _crossings(rows: Sequence[Mapping[str, Any]], threshold: float, tolerance: float) -> tuple[int, int, list[int]]:
    entries = exits = 0
    crossing_steps: list[int] = []
    first = _finite(rows[0].get("goal_distance_m")) if rows else None
    inside = bool(first is not None and first <= threshold)
    for row in rows:
        distance = _finite(row.get("goal_distance_after_m"))
        if distance is None:
            continue
        if not inside and distance <= threshold:
            entries += 1; inside = True; crossing_steps.append(_int(row["step"]))
        elif inside and distance > threshold + tolerance:
            exits += 1; inside = False; crossing_steps.append(_int(row["step"]))
    return entries, exits, crossing_steps


def _max_candidate_streak(rows: Sequence[Mapping[str, Any]], threshold: float, speed_threshold: float) -> int:
    longest = current = 0
    for row in rows:
        distance = _finite(row.get("goal_distance_after_m"))
        speed = _finite(row.get("post_total_speed_proxy_mps"))
        if distance is not None and speed is not None and distance <= threshold and speed <= speed_threshold:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return longest


def diagnose_episode(episode: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]],
                     pre_steps: int = 15, post_steps: int = 20) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold = float(episode["goal_threshold_m"])
    hold_required = _int(episode["stable_hold_steps"])
    anchor = _int(episode["first_entered_1m_step"])
    all_rows = _aligned_rows(raw_rows)
    window = [row for row in all_rows if anchor - pre_steps <= _int(row["step"]) <= anchor + post_steps]
    if not window or not any(_int(row["step"]) == anchor for row in window):
        raise ValueError(f"anchor step {anchor} absent for {episode['episode_id']}")
    for row in window:
        row["relative_step"] = _int(row["step"]) - anchor
    post = [row for row in window if _int(row["step"]) >= anchor]
    entry = next(row for row in window if _int(row["step"]) == anchor)
    eps = float(RULES["magnitude_epsilon"])
    tolerance = float(RULES["distance_rebound_tolerance_m"])
    correction_floor = float(RULES["noticeable_radial_correction_mps"])
    entry_speed = _finite(entry.get("post_total_speed_proxy_mps"))
    entry_radial = _finite(entry.get("post_radial_velocity_proxy_mps"))
    entry_tangential = _finite(entry.get("post_tangential_velocity_proxy_mps"))
    modes: dict[str, dict[str, Any]] = {}
    modes["HIGH_ENTRY_SPEED"] = _mode(
        entry_speed is not None, entry_speed is not None and entry_speed > STABLE_SPEED_THRESHOLD_MPS,
        {"entry_speed_proxy_mps": entry_speed, "stable_speed_threshold_mps": STABLE_SPEED_THRESHOLD_MPS,
         "excess_ratio": entry_speed / STABLE_SPEED_THRESHOLD_MPS - 1.0 if entry_speed is not None else None},
        [anchor] if entry_speed is not None else [], ["aligned post-state speed"],
        [] if entry_speed is not None else ["post-step actual speed"],
        ["next contiguous row estimated radial/tangential velocity norm"],
    )

    inside = [row for row in post if _finite(row.get("goal_distance_after_m")) is not None
              and float(row["goal_distance_after_m"]) <= threshold]
    inside_t = [abs(value) for row in inside if (value := _finite(row.get("post_tangential_velocity_proxy_mps"))) is not None]
    inside_r = [abs(value) for row in inside if (value := _finite(row.get("post_radial_velocity_proxy_mps"))) is not None]
    tangential_fraction = (sum(value > STABLE_SPEED_THRESHOLD_MPS for value in inside_t) / len(inside_t)) if inside_t else None
    tangential_match = bool(inside_t and ((
        (_mean(inside_t) or 0.0) > STABLE_SPEED_THRESHOLD_MPS and inside_r and (_mean(inside_r) or math.inf) <= STABLE_SPEED_THRESHOLD_MPS
    ) or tangential_fraction >= float(RULES["tangential_fraction_threshold"])))
    modes["TANGENTIAL_RESIDUAL"] = _mode(
        bool(inside_t), tangential_match,
        {"mean_tangential_speed_inside_mps": _mean(inside_t), "max_tangential_speed_inside_mps": max(inside_t) if inside_t else None,
         "mean_abs_radial_speed_inside_mps": _mean(inside_r), "fraction_tangential_above_threshold": tangential_fraction},
        [_int(row["step"]) for row in inside if _finite(row.get("post_tangential_velocity_proxy_mps")) is not None],
        ["aligned tangential/radial velocity inside goal region"], [] if inside_t else ["aligned speed proxy"],
        ["next contiguous row estimated velocity components"],
    )

    conflict_flags: list[tuple[int, bool, float, bool]] = []
    for row in post:
        if _int(row.get("goal_recovery_correction_active", 0)) != 1:
            continue
        actor = _vector(row.get("actor_raw_delta_v_mps")); correction = _vector(row.get("goal_recovery_correction_delta_v_mps"))
        goal_direction = _goal_direction(row)
        if actor is None or correction is None or goal_direction is None or _norm(actor) <= eps or _norm(correction) < correction_floor:
            continue
        cosine = _dot(actor, correction) / (_norm(actor) * _norm(correction))
        radial_conflict = _dot(actor, goal_direction) * _dot(correction, goal_direction) < 0.0
        conflict_flags.append((_int(row["step"]), radial_conflict, cosine, cosine < 0.0))
    conflict_fraction = (sum(flag for _, flag, _, _ in conflict_flags) / len(conflict_flags)) if conflict_flags else None
    negative_cosine_fraction = (sum(flag for _, _, _, flag in conflict_flags) / len(conflict_flags)) if conflict_flags else None
    modes["ACTOR_CORRECTION_CONFLICT"] = _mode(
        bool(conflict_flags), conflict_fraction is not None and conflict_fraction >= float(RULES["conflict_fraction_threshold"]),
        {"evaluable_steps": len(conflict_flags), "radial_sign_conflict_fraction": conflict_fraction,
         "negative_cosine_fraction": negative_cosine_fraction,
         "mean_cosine_similarity": _mean(item[2] for item in conflict_flags)},
        [step for step, flag, _, _ in conflict_flags if flag], ["actor and correction radial projections"],
        [] if conflict_flags else ["nontrivial active actor/correction vector pairs"], confidence="logged",
    )

    braking: list[tuple[int, bool, float, str]] = []
    for row in post:
        current = _finite(row.get("current_radial_velocity_mps")); desired = _finite(row.get("desired_radial_velocity_mps"))
        goal_direction = _goal_direction(row)
        applied = _vector(row.get("applied_delta_v_mps"))
        applied_radial = _dot(applied, goal_direction) if applied is not None and goal_direction is not None else None
        source = "logged applied_delta_v projected on recovered goal direction"
        if applied_radial is None:
            applied_radial = _finite(row.get("radial_velocity_change_proxy_mps"))
            source = "next-row estimated radial velocity finite difference"
        if current is not None and desired is not None and applied_radial is not None and current > desired + eps:
            braking.append((_int(row["step"]), applied_radial < float(RULES["weak_braking_delta_threshold_mps"]), applied_radial, source))
    effective = sum(flag for _, flag, _, _ in braking)
    insufficient = len(braking) - effective
    effective_fraction = effective / len(braking) if braking else None
    insufficient_fraction = insufficient / len(braking) if braking else None
    modes["INSUFFICIENT_BRAKING"] = _mode(
        bool(braking), insufficient_fraction is not None and insufficient_fraction >= float(RULES["insufficient_braking_fraction_threshold"]),
        {"braking_required_steps": len(braking), "effective_braking_steps": effective,
         "insufficient_braking_steps": insufficient, "effective_braking_fraction": effective_fraction,
         "insufficient_braking_fraction": insufficient_fraction},
        [step for step, flag, _, _ in braking if not flag], ["current/desired radial velocity", "applied radial delta-v"],
        [] if braking else ["evaluable braking-required transitions"],
        sorted({source for _, _, _, source in braking if "finite difference" in source}),
    )

    distances = [(_int(row["step"]), _finite(row.get("goal_distance_after_m"))) for row in post]
    finite_distances = [(step, value) for step, value in distances if value is not None]
    minimum_step = min(finite_distances, key=lambda item: item[1])[0] if finite_distances else None
    minimum_distance = min((value for _, value in finite_distances), default=None)
    exits_after_min = [(step, value) for step, value in finite_distances if minimum_step is not None and step > minimum_step and value > threshold]
    first_exit = exits_after_min[0][0] if exits_after_min else None
    active_inward = []
    for row in post:
        step = _int(row["step"])
        if first_exit is not None and step > first_exit:
            continue
        current = _finite(row.get("current_radial_velocity_mps")); desired = _finite(row.get("desired_radial_velocity_mps"))
        radial_vector = _vector(row.get("goal_recovery_radial_delta_v_mps"))
        if (_int(row.get("goal_recovery_correction_active", 0)) == 1 and current is not None and desired is not None
                and desired - current > eps and radial_vector is not None and _norm(radial_vector) >= correction_floor):
            active_inward.append(step)
    growth = (max(value for step, value in finite_distances if minimum_step is not None and step >= minimum_step) - minimum_distance
              if minimum_distance is not None else None)
    overshoot = bool(first_exit is not None and growth is not None and growth >= tolerance and active_inward)
    modes["CORRECTION_OVERSHOOT"] = _mode(
        bool(finite_distances), overshoot,
        {"minimum_distance_after_entry_m": minimum_distance, "minimum_distance_step": minimum_step,
         "first_exit_1m_step": first_exit, "overshoot_distance_growth_m": growth,
         "active_inward_correction_steps": active_inward,
         "causality_note": "association only; distance rebound does not prove correction causation"},
        active_inward, ["post distance", "active inward radial correction"],
        [] if finite_distances else ["post distance"], confidence="association",
    )

    entries, exits, crossing_steps = _crossings(post, threshold, tolerance)
    modes["GOAL_REGION_OSCILLATION"] = _mode(
        bool(post), entries >= int(RULES["oscillation_min_entries"]) and exits >= int(RULES["oscillation_min_exits"]),
        {"number_of_1m_entries": entries, "number_of_1m_exits": exits,
         "distance_crossings_1m": entries + exits,
         "post_entry_distance_std": statistics.pstdev([v for _, v in finite_distances]) if len(finite_distances) > 1 else 0.0,
         "post_entry_speed_std": statistics.pstdev([v for row in post if (v := _finite(row.get("post_total_speed_proxy_mps"))) is not None]) if sum(_finite(row.get("post_total_speed_proxy_mps")) is not None for row in post) > 1 else 0.0},
        crossing_steps, ["post distance sequence"], confidence="logged",
    )

    max_streak = _max_candidate_streak(post, threshold, STABLE_SPEED_THRESHOLD_MPS)
    modes["STABLE_HOLD_INTERRUPTED"] = _mode(
        any(_finite(row.get("post_total_speed_proxy_mps")) is not None for row in post), 0 < max_streak < hold_required,
        {"max_consecutive_stable_candidate_steps": max_streak, "stable_hold_steps_required": hold_required},
        [], ["post distance and aligned speed proxy"],
        [] if post else ["aligned speed proxy"], ["estimated speed proxy replaces unavailable post-step actual speed"],
    )
    matched = [mode for mode in MODE_ORDER if modes[mode]["matched"]]
    primary = next((mode for mode in PRIMARY_PRIORITY if mode in matched), "UNRESOLVED")
    secondary = [mode for mode in matched if mode != primary]
    inside_speeds = [_finite(row.get("post_total_speed_proxy_mps")) for row in inside]
    active_inside = [row for row in inside if _int(row.get("goal_recovery_correction_active", 0)) == 1]
    correction_inside = [_finite(row.get("goal_recovery_correction_norm_mps")) for row in inside]
    first_logged_entry = next((_int(row["step"]) for row in all_rows
                               if _finite(row.get("goal_distance_after_m")) is not None
                               and float(row["goal_distance_after_m"]) <= threshold), None)
    diagnostic = {
        "episode_id": episode["episode_id"], "first_entered_1m_step": anchor,
        "first_logged_threshold_entry_step": first_logged_entry,
        "anchor_matches_log": first_logged_entry == anchor,
        "window_start": _int(window[0]["step"]), "window_end": _int(window[-1]["step"]),
        "entry_goal_distance_m": _finite(entry.get("goal_distance_after_m")),
        "entry_total_speed_mps": entry_speed, "entry_total_speed_derived": True,
        "entry_radial_velocity_mps": entry_radial, "entry_tangential_velocity_mps": entry_tangential,
        "stable_goal_distance_threshold_m": threshold,
        "stable_speed_threshold_mps": STABLE_SPEED_THRESHOLD_MPS,
        "stable_hold_steps_required": hold_required,
        "minimum_goal_distance_after_entry_m": minimum_distance, "first_exit_1m_step": first_exit,
        "mean_speed_inside_1m_mps": _mean(inside_speeds),
        "mean_radial_speed_inside_1m_mps": _mean(inside_r),
        "mean_tangential_speed_inside_1m_mps": _mean(inside_t),
        "max_tangential_speed_inside_1m_mps": max(inside_t) if inside_t else None,
        "correction_active_fraction_inside_1m": len(active_inside) / len(inside) if inside else None,
        "mean_correction_norm_inside_1m": _mean(correction_inside),
        "actor_correction_radial_conflict_fraction": conflict_fraction,
        "braking_required_steps": len(braking), "effective_braking_steps": effective,
        "insufficient_braking_steps": insufficient, "effective_braking_fraction": effective_fraction,
        "number_of_1m_entries": entries, "number_of_1m_exits": exits,
        "distance_crossings_1m": entries + exits,
        "max_consecutive_stable_candidate_steps": max_streak,
        "primary_failure_mode": primary, "secondary_failure_modes": secondary,
        **{mode.lower(): bool(modes[mode]["matched"]) for mode in MODE_ORDER},
        "mode_diagnostics": modes, "rules": RULES,
        "stable_criterion": {
            "definition": "estimated goal distance <= threshold AND simulator actual speed <= threshold for consecutive hold steps",
            "source": "src/simulation/environment.py:3509-3556",
            "speed_threshold_source": "FormalExperimentConfig.single_uav_goal_speed_threshold_mps default",
            "exact_recomputation_possible": False,
            "missing_from_logs": ["post-step simulator actual speed", "single_uav_goal_speed_ok", "single_uav_stable_steps"],
            "proxy": "goal_distance_after_m[t] aligned with estimated velocity decomposition at contiguous step t+1",
        },
        **DIAGNOSTIC_ONLY_FLAGS,
    }
    return diagnostic, window


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _plot_episode(path: Path, diagnostic: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    steps = [_int(row["step"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, [_finite(row.get("goal_distance_after_m")) for row in rows], label="goal distance")
    axes[0].axhline(diagnostic["stable_goal_distance_threshold_m"], color="black", linestyle="--")
    axes[1].plot(steps, [_finite(row.get("post_total_speed_proxy_mps")) for row in rows], label="total speed proxy")
    axes[1].plot(steps, [_finite(row.get("post_radial_velocity_proxy_mps")) for row in rows], label="radial proxy")
    axes[1].plot(steps, [_finite(row.get("post_tangential_velocity_proxy_mps")) for row in rows], label="tangential proxy")
    axes[1].plot(steps, [_finite(row.get("desired_radial_velocity_mps")) for row in rows], label="desired radial")
    axes[1].axhline(diagnostic["stable_speed_threshold_mps"], color="black", linestyle="--")
    axes[2].plot(steps, [_finite(row.get("goal_recovery_correction_norm_mps")) for row in rows], label="correction norm")
    for axis in axes:
        axis.axvline(diagnostic["first_entered_1m_step"], color="red", linestyle=":")
        axis.grid(alpha=0.25); axis.legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{diagnostic['episode_id']} — {diagnostic['primary_failure_mode']}")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _summary(diagnostics: Sequence[Mapping[str, Any]], all_episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {mode: sum(bool(row.get(mode.lower())) for row in diagnostics) for mode in MODE_ORDER}
    primary_counts = Counter(str(row["primary_failure_mode"]) for row in diagnostics)
    count = len(diagnostics)
    return {
        "stage": STAGE, "analyzed_episodes": count,
        "entered_1m_episodes": sum(_int(row["entered_1m"]) for row in all_episodes),
        "stable_goal_episodes": sum(_int(row["stable_goal"]) for row in all_episodes),
        "failure_mode_counts": {**counts, "UNRESOLVED": primary_counts.get("UNRESOLVED", 0)},
        "failure_mode_rates": {**{mode: counts[mode] / count if count else 0.0 for mode in MODE_ORDER},
                               "UNRESOLVED": primary_counts.get("UNRESOLVED", 0) / count if count else 0.0},
        "primary_failure_mode_counts": dict(primary_counts),
        "mean_entry_speed": _mean(row.get("entry_total_speed_mps") for row in diagnostics),
        "median_entry_speed": _median(row.get("entry_total_speed_mps") for row in diagnostics),
        "mean_entry_radial_velocity": _mean(row.get("entry_radial_velocity_mps") for row in diagnostics),
        "mean_entry_tangential_velocity": _mean(row.get("entry_tangential_velocity_mps") for row in diagnostics),
        "mean_speed_inside_1m": _mean(row.get("mean_speed_inside_1m_mps") for row in diagnostics),
        "mean_tangential_speed_inside_1m": _mean(row.get("mean_tangential_speed_inside_1m_mps") for row in diagnostics),
        "mean_correction_active_fraction_inside_1m": _mean(row.get("correction_active_fraction_inside_1m") for row in diagnostics),
        "mean_correction_norm_inside_1m": _mean(row.get("mean_correction_norm_inside_1m") for row in diagnostics),
        "mean_actor_correction_conflict_fraction": _mean(row.get("actor_correction_radial_conflict_fraction") for row in diagnostics),
        "mean_max_consecutive_stable_candidate_steps": _mean(row.get("max_consecutive_stable_candidate_steps") for row in diagnostics),
        "rules": RULES,
        "proxy_limitations": "Actual post-step speed and stable-hold counter were not logged; speed/hold diagnoses use an aligned estimated-state proxy.",
        **DIAGNOSTIC_ONLY_FLAGS,
    }


def run_diagnostic(*, evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
                   logs_root: Path = DEFAULT_LOG_ROOT, plots_root: Path = DEFAULT_PLOT_ROOT,
                   pre_steps: int = 15, post_steps: int = 20, plots: bool = True,
                   quiet: bool = False) -> dict[str, Any]:
    if pre_steps < 0 or post_steps < 0:
        raise ValueError("pre/post steps must be non-negative")
    evaluation_root = evaluation_root.resolve(); logs_root = logs_root.resolve(); plots_root = plots_root.resolve()
    if logs_root.exists() or (plots and plots_root.exists()):
        raise FileExistsError("diagnostic output roots must not already exist")
    if (logs_root == evaluation_root or logs_root.is_relative_to(evaluation_root)
            or (plots and (plots_root == evaluation_root or plots_root.is_relative_to(evaluation_root)))):
        raise ValueError("diagnostic output must not be the evaluation input root")
    episodes, grouped, provenance = load_inputs(evaluation_root)
    selected = select_failure_episodes(episodes)
    logs_stage = logs_root.with_name(f".{logs_root.name}.staging-{uuid.uuid4().hex}")
    plots_stage = plots_root.with_name(f".{plots_root.name}.staging-{uuid.uuid4().hex}")
    created: list[Path] = []
    committed: list[Path] = []
    try:
        logs_stage.mkdir(parents=True); created.append(logs_stage)
        if plots:
            plots_stage.mkdir(parents=True); created.append(plots_stage)
        diagnostics: list[dict[str, Any]] = []
        for episode in selected:
            diagnostic, window = diagnose_episode(episode, grouped[episode["episode_id"]], pre_steps, post_steps)
            diagnostics.append(diagnostic)
            safe_id = _safe_id(str(episode["episode_id"]))
            _write_csv(logs_stage / f"near_goal_episode_{safe_id}.csv", window)
            _write_json(logs_stage / f"near_goal_episode_{safe_id}_diagnostic.json", diagnostic)
            if plots:
                _plot_episode(plots_stage / f"near_goal_episode_{safe_id}.png", diagnostic, window)
            if not quiet:
                print(f"[{STAGE}] {episode['episode_id']}: {diagnostic['primary_failure_mode']}", flush=True)
        summary = _summary(diagnostics, episodes)
        summary["input_provenance"] = provenance
        summary["stable_goal_criterion"] = diagnostics[0]["stable_criterion"] if diagnostics else {}
        summary["episode_diagnostics"] = diagnostics
        _write_json(logs_stage / "near_goal_stable_failure_summary.json", summary)
        flat_rows = []
        for item in diagnostics:
            flat_rows.append({key: value for key, value in item.items() if not isinstance(value, (dict, list))} | {
                "secondary_failure_modes": item["secondary_failure_modes"],
            })
        _write_csv(logs_stage / "near_goal_stable_failure_episodes.csv", flat_rows)
        summary_rows: list[dict[str, Any]] = []
        for key, value in summary.items():
            if not isinstance(value, (dict, list)):
                summary_rows.append({"metric": key, "value": value})
        for mode, value in summary["failure_mode_counts"].items():
            summary_rows.append({"metric": f"{mode}_count", "value": value})
        for mode, value in summary["failure_mode_rates"].items():
            summary_rows.append({"metric": f"{mode}_rate", "value": value})
        _write_csv(logs_stage / "near_goal_stable_failure_summary.csv", summary_rows)
        _write_json(logs_stage / "diagnostic_manifest.json", {
            "stage": STAGE, "input_provenance": provenance, "rules": RULES,
            "pre_steps": pre_steps, "post_steps": post_steps,
            "selected_episode_ids": [row["episode_id"] for row in selected],
            "stable_speed_threshold_mps": STABLE_SPEED_THRESHOLD_MPS,
            **DIAGNOSTIC_ONLY_FLAGS,
        })
        for path in logs_stage.glob("*.json"):
            _read_json(path)
        logs_stage.rename(logs_root); created.remove(logs_stage); committed.append(logs_root)
        if plots:
            plots_stage.rename(plots_root); created.remove(plots_stage); committed.append(plots_root)
        return summary
    except Exception:
        for path in reversed(created):
            if path.exists() and path in {logs_stage, plots_stage}:
                shutil.rmtree(path)
        for path in reversed(committed):
            if path.exists():
                shutil.rmtree(path)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--pre-steps", type=int, default=15)
    parser.add_argument("--post-steps", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_diagnostic(evaluation_root=args.evaluation_root, logs_root=args.logs_root,
                   plots_root=args.plots_root, pre_steps=args.pre_steps,
                   post_steps=args.post_steps, plots=not args.no_plots, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
