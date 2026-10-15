"""Offline-only audit of v2.1 goal-boundary chattering and stable hold.

This module reads frozen CSV/JSON artifacts.  It deliberately imports no
controller, policy, simulator, environment, or training module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SOURCE = Path("logs/preliminary/single_uav_terminal_capture_v21_validation_seed2026")
DEFAULT_OUTPUT = Path("logs/preliminary/single_uav_terminal_v21_boundary_chattering_audit_seed2026")
SOURCE_FILES = (
    "steps.csv",
    "terminal_capture_v21_validation_summary.json",
    "terminal_capture_v21_validation_episodes.csv",
    "episodes.csv",
)
OUTPUT_FILES = (
    "boundary_chattering_summary.json",
    "boundary_chattering_episodes.csv",
    "boundary_crossings.csv",
    "boundary_excursions.csv",
    "hold_radius_screen.csv",
    "boundary_chattering_report.md",
)
ENTER_RADII = (1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85)
RELEASE_RADII = (1.00, 1.01, 1.02, 1.03, 1.05, 1.08, 1.10)
S_IDS = {"test_r99_episode_0007", "test_r99_episode_0009", "test_r99_episode_0010", "test_r99_episode_0018"}
F_IDS = {"test_r99_episode_0003", "test_r99_episode_0015", "test_r99_episode_0016", "test_r99_episode_0019"}
N_IDS = {"test_r99_episode_0001", "test_r99_episode_0008", "test_r99_episode_0017"}
REQUIRED_STEPS = {
    "episode_id", "step", "goal_distance_for_stable_m", "goal_speed_for_stable_mps",
    "stable_goal_distance_threshold_m", "stable_speed_threshold_mps", "stable_hold_steps_required",
    "goal_distance_ok", "goal_speed_ok", "stable_candidate_step", "stable_hold_counter",
    "stable_goal_now", "post_step_actual_radial_velocity_mps",
    "post_step_actual_tangential_velocity_mps", "post_step_actual_speed_mps",
    "terminal_v2_inside_goal_hold", "terminal_v21_actor_radial_delta_v_mps",
    "terminal_v21_conflict_compensation_radial_mps", "terminal_v21_net_radial_post_transform_mps",
}
STATUS_FLAGS = {
    "control_algorithm_modified": False,
    "goal_recovery_controller_modified": False,
    "terminal_v2_modified": False,
    "terminal_v21_modified": False,
    "actor_modified": False,
    "gat_modified": False,
    "critic_modified": False,
    "reward_modified": False,
    "dynamics_modified": False,
    "stable_goal_criterion_modified": False,
    "action_transform_modified": False,
    "max_correction_modified": False,
    "safety_boundary_obstacle_logic_modified": False,
    "training_performed": False,
    "formal_evaluation_executed_by_codex": False,
    "smoke_evaluation_executed_by_codex": False,
    "rollout_executed": False,
    "offline_boundary_audit_executed": True,
}


def _float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field}: expected finite number, got {value!r}")
    return result


def _int(value: Any, field: str) -> int:
    number = _float(value, field)
    if number != int(number):
        raise ValueError(f"{field}: expected integer, got {value!r}")
    return int(number)


def _bool(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError(f"{field}: expected binary value, got {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        raise ValueError("steps.csv is empty")
    missing = REQUIRED_STEPS - set(rows[0])
    if missing:
        raise ValueError(f"steps.csv missing fields: {sorted(missing)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    last: dict[str, int] = {}
    counter: dict[str, int] = defaultdict(int)
    for raw in rows:
        episode = str(raw["episode_id"])
        if not episode:
            raise ValueError("empty episode_id")
        step = _int(raw["step"], "step")
        key = (episode, step)
        if key in seen:
            raise ValueError(f"duplicate episode-step: {key}")
        if episode in last and step != last[episode] + 1:
            raise ValueError(f"non-contiguous/non-monotonic steps for {episode}: {last[episode]} -> {step}")
        seen.add(key)
        last[episode] = step
        distance = _float(raw["goal_distance_for_stable_m"], "goal_distance_for_stable_m")
        speed = _float(raw["goal_speed_for_stable_mps"], "goal_speed_for_stable_mps")
        dlim = _float(raw["stable_goal_distance_threshold_m"], "stable_goal_distance_threshold_m")
        slim = _float(raw["stable_speed_threshold_mps"], "stable_speed_threshold_mps")
        hold = _int(raw["stable_hold_steps_required"], "stable_hold_steps_required")
        if dlim != 1.0 or slim != 0.5 or hold != 5:
            raise ValueError(f"canonical stable thresholds mismatch at {key}: {dlim}, {slim}, {hold}")
        distance_ok = distance <= dlim
        speed_ok = speed <= slim
        candidate = distance_ok and speed_ok
        counter[episode] = counter[episode] + 1 if candidate else 0
        stable = candidate and counter[episode] >= hold
        checks = {
            "goal_distance_ok": distance_ok,
            "goal_speed_ok": speed_ok,
            "stable_candidate_step": candidate,
            "stable_goal_now": stable,
        }
        for field, expected in checks.items():
            if _bool(raw[field], field) != expected:
                raise ValueError(f"canonical {field} mismatch at {key}")
        if _int(raw["stable_hold_counter"], "stable_hold_counter") != counter[episode]:
            raise ValueError(f"stable counter mismatch at {key}")
        item = dict(raw)
        item.update({
            "_step": step, "_distance": distance, "_speed": speed,
            "_distance_ok": distance_ok, "_speed_ok": speed_ok,
            "_candidate": candidate, "_counter": counter[episode], "_stable": stable,
            "_radial": _float(raw["post_step_actual_radial_velocity_mps"], "radial velocity"),
            "_tangential": _float(raw["post_step_actual_tangential_velocity_mps"], "tangential velocity"),
            "_hold": _bool(raw["terminal_v2_inside_goal_hold"], "terminal_v2_inside_goal_hold"),
        })
        grouped[episode].append(item)
    return dict(grouped)


def classify_resets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    resets: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        if int(previous["_counter"]) > 0 and int(current["_counter"]) == 0:
            d, s = bool(current["_distance_ok"]), bool(current["_speed_ok"])
            kind = "both_reset" if not d and not s else "distance_only_reset" if not d else "speed_only_reset" if not s else "unexplained"
            resets.append({"step_before": previous["_step"], "step_after": current["_step"], "type": kind})
    return resets


def detect_crossings(rows: Sequence[Mapping[str, Any]], radius: float = 1.0) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    first_entry = next((i for i, row in enumerate(rows) if float(row["_distance"]) <= radius), None)
    if first_entry is None:
        return output
    # The initial outside->inside entry is not a post-entry recrossing.
    for previous, current in zip(rows[first_entry:], rows[first_entry + 1:]):
        before, after = float(previous["_distance"]), float(current["_distance"])
        direction = "outside_to_inside" if before > radius and after <= radius else "inside_to_outside" if before <= radius and after > radius else None
        if direction:
            output.append({
                "direction": direction, "step_before": previous["_step"], "step_after": current["_step"],
                "distance_before": before, "distance_after": after,
                "speed_before": previous["_speed"], "speed_after": current["_speed"],
                "radial_velocity_before": previous["_radial"], "radial_velocity_after": current["_radial"],
                "tangential_velocity_before": previous["_tangential"], "tangential_velocity_after": current["_tangential"],
                "stable_counter_before": previous["_counter"], "stable_counter_after": current["_counter"],
                "terminal_hold_before": previous.get("terminal_v2_inside_goal_hold", ""),
                "terminal_hold_after": current.get("terminal_v2_inside_goal_hold", ""),
                "desired_radial_before": previous.get("terminal_desired_radial_velocity_mps", ""),
                "desired_radial_after": current.get("terminal_desired_radial_velocity_mps", ""),
                "actor_radial_before": previous.get("terminal_v21_actor_radial_delta_v_mps", ""),
                "actor_radial_after": current.get("terminal_v21_actor_radial_delta_v_mps", ""),
                "conflict_compensation_radial_before": previous.get("terminal_v21_conflict_compensation_radial_mps", ""),
                "conflict_compensation_radial_after": current.get("terminal_v21_conflict_compensation_radial_mps", ""),
                "net_radial_pre_transform_before": previous.get("terminal_v21_net_radial_pre_transform_mps", ""),
                "net_radial_pre_transform_after": current.get("terminal_v21_net_radial_pre_transform_mps", ""),
                "net_radial_post_transform_before": previous.get("terminal_v21_net_radial_post_transform_mps", ""),
                "net_radial_post_transform_after": current.get("terminal_v21_net_radial_post_transform_mps", ""),
            })
    return output


def detect_excursions(rows: Sequence[Mapping[str, Any]], radius: float = 1.0) -> list[dict[str, Any]]:
    first = next((i for i, row in enumerate(rows) if float(row["_distance"]) <= radius), None)
    if first is None:
        return []
    excursions: list[dict[str, Any]] = []
    index = first + 1
    while index < len(rows):
        if float(rows[index]["_distance"]) <= radius:
            index += 1
            continue
        start = index
        while index + 1 < len(rows) and float(rows[index + 1]["_distance"]) > radius:
            index += 1
        end = index
        peak_index = max(range(start, end + 1), key=lambda i: float(rows[i]["_distance"]))
        peak = rows[peak_index]
        returned = end + 1 < len(rows) and float(rows[end + 1]["_distance"]) <= radius
        excursions.append({
            "start_step": rows[start]["_step"], "end_step": rows[end]["_step"],
            "duration_steps": end - start + 1,
            "entry_distance_before_excursion": rows[start - 1]["_distance"],
            "maximum_distance_m": peak["_distance"],
            "peak_excess_over_1m": max(float(peak["_distance"]) - radius, 0.0),
            "speed_at_peak": peak["_speed"], "radial_velocity_at_peak": peak["_radial"],
            "tangential_velocity_at_peak": peak["_tangential"],
            "return_to_1m": returned,
            "steps_until_return": (int(rows[end + 1]["_step"]) - int(rows[start]["_step"])) if returned else None,
            "censored": not returned,
        })
        index += 1
    return excursions


def penetration_before_first_reset(rows: Sequence[Mapping[str, Any]], resets: Sequence[Mapping[str, Any]]) -> float | None:
    first = next((i for i, row in enumerate(rows) if float(row["_distance"]) <= 1.0), None)
    if first is None:
        return None
    reset_step = int(resets[0]["step_after"]) if resets else None
    eligible = [row for row in rows[first:] if reset_step is None or int(row["_step"]) < reset_step]
    return max(0.0, 1.0 - min(float(row["_distance"]) for row in eligible))


def first_candidate_window(rows: Sequence[Mapping[str, Any]], start: int, width: int = 5) -> int | None:
    for index in range(start, len(rows) - width + 1):
        if all(bool(row["_candidate"]) for row in rows[index:index + width]):
            return int(rows[index]["_step"])
    return None


def screen_enter(rows: Sequence[Mapping[str, Any]], radius: float) -> dict[str, Any]:
    index = next((i for i, row in enumerate(rows) if float(row["_distance"]) <= radius), None)
    if index is None:
        return {"reached": False, "first_reach_step": None, "observed_5step_window": False, "window_start_step": None}
    window = first_candidate_window(rows, index)
    return {"reached": True, "first_reach_step": int(rows[index]["_step"]), "observed_5step_window": window is not None, "window_start_step": window}


def screen_release(excursions: Sequence[Mapping[str, Any]], radius: float) -> int:
    return sum(float(item["maximum_distance_m"]) > radius for item in excursions)


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values: Iterable[float]) -> dict[str, float | None]:
    data = list(values)
    if not data:
        return {key: None for key in ("mean", "median", "p25", "p75", "p90", "p95", "min", "max")}
    return {
        "mean": statistics.fmean(data), "median": statistics.median(data),
        "p25": percentile(data, .25), "p75": percentile(data, .75),
        "p90": percentile(data, .90), "p95": percentile(data, .95),
        "min": min(data), "max": max(data),
    }


def classify_failure(rows: Sequence[Mapping[str, Any]], resets: Sequence[Mapping[str, Any]], excursions: Sequence[Mapping[str, Any]], crossings: Sequence[Mapping[str, Any]], first_entry: int | None) -> tuple[str, dict[str, bool | int]]:
    kinds = Counter(item["type"] for item in resets)
    outward_drift_crossings = 0
    for crossing in crossings:
        if crossing["direction"] != "inside_to_outside":
            continue
        current_index = next(i for i, row in enumerate(rows) if int(row["_step"]) == int(crossing["step_after"]))
        prior = rows[max(0, current_index - 2):current_index]
        if len(prior) == 2 and all(float(row["_distance"]) < 1.0 and bool(row["_hold"]) and float(row["_radial"]) < 0 for row in prior):
            outward_drift_crossings += 1
    evidence = {
        "distance_chatter": bool(kinds["distance_only_reset"] or kinds["both_reset"] or excursions),
        "speed_reset": bool(kinds["speed_only_reset"] or kinds["both_reset"]),
        "late_entry": first_entry is not None and first_entry >= 95,
        "hold_outward_drift": outward_drift_crossings > 0,
        "hold_outward_drift_crossings": outward_drift_crossings,
    }
    labels = [name.upper() for name in ("distance_chatter", "speed_reset", "late_entry", "hold_outward_drift") if evidence[name]]
    return ("UNKNOWN" if not labels else labels[0] if len(labels) == 1 else "MIXED"), evidence


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(grouped: Mapping[str, Sequence[Mapping[str, Any]]], episode_rows: Sequence[Mapping[str, Any]], strict: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = {str(row["episode_id"]): row for row in episode_rows}
    if strict:
        if len(grouped) != 20 or sum(map(len, grouped.values())) != 1968:
            raise ValueError("canonical input must contain exactly 20 episodes and 1968 steps")
        if len(metadata) != 20:
            raise ValueError("episode summary must contain exactly 20 episodes")
    actual_s = {eid for eid, row in metadata.items() if _bool(row["entered_1m"], "entered_1m") and _bool(row["stable_goal"], "stable_goal")}
    actual_f = {eid for eid, row in metadata.items() if _bool(row["entered_1m"], "entered_1m") and not _bool(row["stable_goal"], "stable_goal")}
    actual_n = {eid for eid, row in metadata.items() if _bool(row["post_bypass_recovery"], "post_bypass_recovery") and not _bool(row["entered_1m"], "entered_1m")}
    if strict and (actual_s != S_IDS or actual_f != F_IDS or actual_n != N_IDS):
        raise ValueError("canonical S/F/N episode groups do not match the frozen contract")
    episode_out: list[dict[str, Any]] = []
    crossing_out: list[dict[str, Any]] = []
    excursion_out: list[dict[str, Any]] = []
    screen_out: list[dict[str, Any]] = []
    all_resets: list[dict[str, Any]] = []
    penetration: dict[str, list[float]] = defaultdict(list)
    group_excess: dict[str, list[float]] = defaultdict(list)
    for episode_id in sorted(grouped):
        rows = grouped[episode_id]
        group = "S" if episode_id in actual_s else "F" if episode_id in actual_f else "N" if episode_id in actual_n else "OTHER"
        resets = classify_resets(rows)
        crossings = detect_crossings(rows)
        excursions = detect_excursions(rows)
        first_entry = next((int(row["_step"]) for row in rows if float(row["_distance"]) <= 1.0), None)
        pen = penetration_before_first_reset(rows, resets)
        if pen is not None:
            penetration[group].append(pen)
        group_excess[group].extend(float(item["peak_excess_over_1m"]) for item in excursions)
        for index, item in enumerate(crossings, 1):
            crossing_out.append({"episode_id": episode_id, "group": group, "crossing_index": index, **item})
        for index, item in enumerate(excursions, 1):
            excursion_out.append({"episode_id": episode_id, "group": group, "excursion_index": index, **item})
        for item in resets:
            all_resets.append({"episode_id": episode_id, "group": group, **item})
        label, evidence = classify_failure(rows, resets, excursions, crossings, first_entry) if group == "F" else ("", {})
        reset_counts = Counter(item["type"] for item in resets)
        pre_reset_evidence = []
        for reset in resets:
            reset_index = next(i for i, row in enumerate(rows) if int(row["_step"]) == int(reset["step_after"]))
            for relative, evidence_row in enumerate(rows[max(0, reset_index - 3):reset_index], start=-min(3, reset_index)):
                pre_reset_evidence.append({
                    "reset_step": reset["step_after"], "relative_step": relative,
                    "step": evidence_row["_step"], "distance_m": evidence_row["_distance"],
                    "radial_velocity_mps": evidence_row["_radial"], "tangential_velocity_mps": evidence_row["_tangential"],
                    "actor_radial_delta_v_mps": evidence_row.get("terminal_v21_actor_radial_delta_v_mps", ""),
                    "conflict_compensation_radial_mps": evidence_row.get("terminal_v21_conflict_compensation_radial_mps", ""),
                    "net_radial_pre_transform_mps": evidence_row.get("terminal_v21_net_radial_pre_transform_mps", ""),
                    "net_radial_post_transform_mps": evidence_row.get("terminal_v21_net_radial_post_transform_mps", ""),
                })
        episode_out.append({
            "episode_id": episode_id, "group": group, "first_entered_1m_step": first_entry,
            "remaining_steps_after_entry": None if first_entry is None else 100 - first_entry,
            "minimum_goal_distance_m": min(float(row["_distance"]) for row in rows),
            "max_stable_hold_counter": max(int(row["_counter"]) for row in rows),
            "inside_to_outside_crossings": sum(item["direction"] == "inside_to_outside" for item in crossings),
            "outside_to_inside_crossings": sum(item["direction"] == "outside_to_inside" for item in crossings),
            "counter_resets": len(resets), "distance_only_resets": reset_counts["distance_only_reset"],
            "speed_only_resets": reset_counts["speed_only_reset"], "both_resets": reset_counts["both_reset"],
            "unexplained_resets": reset_counts["unexplained"], "interior_penetration_before_first_reset_m": pen,
            "failure_classification": label, "pre_reset_1_to_3_step_evidence_json": json.dumps(pre_reset_evidence, separators=(",", ":")), **evidence,
        })
        if group in {"S", "F"}:
            for radius in ENTER_RADII:
                result = screen_enter(rows, radius)
                screen_out.append({"screen_type": "enter", "radius_m": radius, "episode_id": episode_id, "group": group, **result, "excursions_exceeding_radius": ""})
            for radius in RELEASE_RADII:
                screen_out.append({"screen_type": "release", "radius_m": radius, "episode_id": episode_id, "group": group, "reached": "", "first_reach_step": "", "observed_5step_window": "", "window_start_step": "", "excursions_exceeding_radius": screen_release(excursions, radius)})
    entered_resets = [item for item in all_resets if item["group"] in {"S", "F"}]
    reset_counts = Counter(item["type"] for item in entered_resets)
    reset_total = sum(reset_counts.values())
    reset_proportions = {key: value / reset_total for key, value in reset_counts.items()} if reset_total else {}
    f_rows = [row for row in episode_out if row["group"] == "F"]
    unexplained = reset_counts["unexplained"]
    labels_known = all(row["failure_classification"] != "UNKNOWN" for row in f_rows)
    verdict = "AUDIT_READY_FOR_V22" if strict and unexplained == 0 and labels_known and bool(f_rows) else "NOT_READY_FOR_V22"
    enter_aggregate = []
    for radius in ENTER_RADII:
        rows = [row for row in screen_out if row["screen_type"] == "enter" and row["radius_m"] == radius]
        reached = sum(bool(row["reached"]) for row in rows)
        with_window = sum(bool(row["observed_5step_window"]) for row in rows)
        enter_aggregate.append({"radius_m": radius, "episodes_reaching_radius": reached, "episodes_with_observed_5step_stable_window_after_reach": with_window, "rate": with_window / reached if reached else None})
    release_aggregate = []
    original_excursion_count = sum(1 for row in excursion_out if row["group"] in {"S", "F"})
    for radius in RELEASE_RADII:
        rows = [row for row in screen_out if row["screen_type"] == "release" and row["radius_m"] == radius]
        exceeding = sum(int(row["excursions_exceeding_radius"]) for row in rows)
        f_exceeding = sum(int(row["excursions_exceeding_radius"]) for row in rows if row["group"] == "F")
        release_aggregate.append({"radius_m": radius, "original_excursions": original_excursion_count, "excursions_exceeding": exceeding, "fraction_exceeding": exceeding / original_excursion_count if original_excursion_count else None, "group_f_excursions_exceeding": f_exceeding})
    group_stats = {}
    for group in ("S", "F", "N"):
        e_rows = [row for row in episode_out if row["group"] == group]
        x_rows = [row for row in crossing_out if row["group"] == group]
        q_rows = [row for row in excursion_out if row["group"] == group]
        r_rows = [row for row in all_resets if row["group"] == group]
        group_stats[group] = {
            "episodes": len(e_rows),
            "inside_to_outside_crossings": sum(row["direction"] == "inside_to_outside" for row in x_rows),
            "outside_to_inside_crossings": sum(row["direction"] == "outside_to_inside" for row in x_rows),
            "outward_excursions": len(q_rows),
            "counter_resets": len(r_rows),
            "reset_types": dict(Counter(row["type"] for row in r_rows)),
            "reset_type_proportions": ({key: value / len(r_rows) for key, value in Counter(row["type"] for row in r_rows).items()} if r_rows else {}),
            "peak_excess_over_1m_stats": stats(float(row["peak_excess_over_1m"]) for row in q_rows),
            "penetration_before_first_reset_stats": stats(float(row["interior_penetration_before_first_reset_m"]) for row in e_rows if row["interior_penetration_before_first_reset_m"] is not None),
            "post_entry_radial_velocity_stats": stats(float(row["_radial"]) for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if any(float(prior["_distance"]) <= 1.0 for prior in grouped[eid] if int(prior["_step"]) <= int(row["_step"]))),
            "post_entry_tangential_velocity_stats": stats(float(row["_tangential"]) for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if any(float(prior["_distance"]) <= 1.0 for prior in grouped[eid] if int(prior["_step"]) <= int(row["_step"]))),
            "inside_1m_radial_velocity_stats": stats(float(row["_radial"]) for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if float(row["_distance"]) <= 1.0),
            "strictly_inside_1m_negative_radial_fraction": (
                sum(float(row["_radial"]) < 0 for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if float(row["_distance"]) < 1.0)
                / max(1, sum(1 for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if float(row["_distance"]) < 1.0))
            ),
            "hold_latched_strictly_inside_negative_radial_fraction": (
                sum(float(row["_radial"]) < 0 for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if float(row["_distance"]) < 1.0 and bool(row["_hold"]))
                / max(1, sum(1 for eid in sorted(grouped) if any(item["episode_id"] == eid and item["group"] == group for item in episode_out) for row in grouped[eid] if float(row["_distance"]) < 1.0 and bool(row["_hold"])))
            ),
        }
    supported_enter = [item["radius_m"] for item in enter_aggregate if sum(1 for row in screen_out if row["screen_type"] == "enter" and row["radius_m"] == item["radius_m"] and row["group"] == "S" and bool(row["observed_5step_window"])) == len(actual_s)]
    release_lower = next((item["radius_m"] for item in release_aggregate if item["group_f_excursions_exceeding"] == 0), None)
    release_upper = next((item["radius_m"] for item in release_aggregate if item["excursions_exceeding"] == 0), None)
    primary_mechanism = "DISTANCE_CHATTER" if reset_counts["distance_only_reset"] > reset_counts["speed_only_reset"] else "SPEED_RESET" if reset_counts["speed_only_reset"] else "UNKNOWN"
    any_drift = any(bool(row.get("hold_outward_drift")) for row in f_rows)
    any_late = any(bool(row.get("late_entry")) for row in f_rows)
    secondary_mechanism = "HOLD_OUTWARD_DRIFT" if any_drift else "LATE_ENTRY" if any_late else "NONE_OBSERVED"
    hysteresis_alone_sufficient = not any_drift
    summary = {
        "stage": "single_uav_terminal_v21_boundary_chattering_audit",
        "verdict": verdict,
        "canonical_stable_contract": {"distance_source": "post_step_estimated_state", "distance_threshold_m": 1.0, "speed_source": "simulator_actual_state", "speed_threshold_mps": 0.5, "hold_steps": 5, "reconstruction_match": True},
        "counts": {"episodes": len(grouped), "steps": sum(map(len, grouped.values())), "group_s": len(actual_s), "group_f": len(actual_f), "group_n": len(actual_n)},
        "groups": {"S": sorted(actual_s), "F": sorted(actual_f), "N": sorted(actual_n)},
        "group_statistics": group_stats,
        "q1_counter_reset_cause": dict(reset_counts),
        "q1_counter_reset_proportions": reset_proportions,
        "total_resets": reset_total,
        "reset_scope": "entered episodes only (Group S and Group F)",
        "q2_distance_vs_speed": "DISTANCE_DOMINANT" if reset_counts["distance_only_reset"] > reset_counts["speed_only_reset"] else "NOT_DISTANCE_DOMINANT",
        "q3_crossings": dict(Counter(row["direction"] for row in crossing_out)),
        "q4_excursion_depth": {group: stats(values) for group, values in sorted(group_excess.items())},
        "q5_inward_margin": {group: stats(values) for group, values in sorted(penetration.items())},
        "q6_mechanisms": {row["episode_id"]: row["failure_classification"] for row in f_rows},
        "failure_episode_details": f_rows,
        "enter_radius_screen": enter_aggregate,
        "release_radius_screen": release_aggregate,
        "recommended_hold_enter_radius_range_m": ({"lower_inclusive": min(value for value in supported_enter if value < 1.0), "upper_exclusive": 1.0, "interior_values_not_screened": True} if any(value < 1.0 for value in supported_enter) else None),
        "hold_enter_observational_screening_interval_m": [min(supported_enter), max(supported_enter)] if supported_enter else None,
        "recommended_hold_release_radius_range_m": [release_lower, release_upper] if release_lower is not None and release_upper is not None else None,
        "primary_failure_mechanism": primary_mechanism,
        "secondary_failure_mechanism": secondary_mechanism,
        "late_entry_threshold_step": 95,
        "late_entry_failure_count": sum(bool(row["late_entry"]) for row in f_rows),
        "whether_hysteresis_alone_likely_sufficient": hysteresis_alone_sufficient,
        "whether_restorative_inward_hold_needed": "LIKELY_WORTH_TESTING" if any_drift else "NOT_INDICATED_BY_CURRENT_LOGS",
        "estimated_vs_actual_distance_noise_assessment": "NOT_ASSESSABLE",
        "estimated_vs_actual_distance_noise_reason": "No independent simulator-actual goal distance is logged; canonical distance is post-step estimated distance.",
        "control_switching_boundary_vs_evaluation_boundary": "Any future release radius is controller-mode hysteresis only; canonical stable evaluation remains distance <= 1.0 m.",
        "evidence_strength": "MODERATE_OFFLINE_SINGLE_SEED",
        "limitations": ["single frozen 20-episode seed", "observational screening does not predict changed-controller trajectories", "successful episodes terminate at stable hold, so post-stable robustness is unobserved", "no independent actual goal distance"],
        "entry_window_convention": "first reach row inclusive; five-row window uses original canonical candidate and need not stay inside r_enter",
        "release_window_convention": "fixed original >1.0 m excursions; strict peak > release radius; no per-radius resegmentation",
        "status_flags": dict(STATUS_FLAGS),
    }
    return summary, episode_out, crossing_out, excursion_out, screen_out


def render_report(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    failure_lines = "\n".join(
        f"- `{row['episode_id']}`: first entry={row['first_entered_1m_step']}, max counter={row['max_stable_hold_counter']}, "
        f"resets={row['counter_resets']} (distance={row['distance_only_resets']}, speed={row['speed_only_resets']}, both={row['both_resets']}), "
        f"outward crossings={row['inside_to_outside_crossings']}, classification={row['failure_classification']}."
        for row in summary["failure_episode_details"]
    )
    return f"""# Terminal Capture v2.1 Goal-Boundary Chattering Offline Audit

## Result

**{summary['verdict']}**

The canonical counter was reconstructed exactly for {counts['steps']} rows in {counts['episodes']} episodes.
All conclusions are descriptive observations from the frozen trajectory, not claims about a changed controller.

## Answers to Q1–Q6

- Q1 reset causes: `{json.dumps(summary['q1_counter_reset_cause'], sort_keys=True)}`.
- Q1 reset proportions: `{json.dumps(summary['q1_counter_reset_proportions'], sort_keys=True)}`.
- Q2 distance versus speed: **{summary['q2_distance_vs_speed']}**.
- Q3 boundary crossings: `{json.dumps(summary['q3_crossings'], sort_keys=True)}`.
- Q4 excursion excess distributions: `{json.dumps(summary['q4_excursion_depth'], sort_keys=True)}`.
- Q5 penetration before first reset: `{json.dumps(summary['q5_inward_margin'], sort_keys=True)}`.
- Q6 F-episode mechanisms: `{json.dumps(summary['q6_mechanisms'], sort_keys=True)}`.

## Group statistics

`{json.dumps(summary['group_statistics'], sort_keys=True)}`

## Failed entered episodes

{failure_lines}

## Interpretation

Candidate enter range: `{summary['recommended_hold_enter_radius_range_m']}` m. Candidate release range:
`{summary['recommended_hold_release_radius_range_m']}` m. Hysteresis alone is not judged sufficient from this
offline single-seed evidence; a bounded restorative inward hold is worth testing, not implemented here.
The evaluation stable boundary remains 1.0 m and is distinct from any future controller switching boundary.

The enter range is the fixed-grid interval for which every observed Group S episode still has a later original
five-row canonical window. The release lower bound is the first fixed-grid radius exceeded by no Group F
excursion; the upper bound is the first fixed-grid radius exceeded by no observed Group S/F excursion.
These are observational screens, not predictions of a modified controller.

### Enter-radius screen

`{json.dumps(summary['enter_radius_screen'], sort_keys=True)}`

### Release-radius screen

`{json.dumps(summary['release_radius_screen'], sort_keys=True)}`

Estimated-versus-actual distance-noise attribution: **{summary['estimated_vs_actual_distance_noise_assessment']}**.
Reason: {summary['estimated_vs_actual_distance_noise_reason']}

Late entry uses the fixed rule `first_entry_step >= {summary['late_entry_threshold_step']}`; F episodes meeting it:
{summary['late_entry_failure_count']}.

## Limitations

""" + "\n".join(f"- {item}" for item in summary["limitations"]) + "\n"


def run_audit(source: Path, output: Path, strict: bool = True) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("output must not overlap source")
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    paths = {name: source / name for name in SOURCE_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    before = {name: sha256(path) for name, path in paths.items()}
    steps = read_csv(paths["steps.csv"])
    episodes = read_csv(paths["terminal_capture_v21_validation_episodes.csv"])
    if read_csv(paths["episodes.csv"]) != episodes:
        raise ValueError("episodes.csv copies disagree")
    with paths["terminal_capture_v21_validation_summary.json"].open("r", encoding="utf-8") as handle:
        source_summary = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")))
    expected_source_counts = {"episodes": 20, "passed_obstacle": 11, "post_bypass_recovery": 11, "entered_1m": 8, "stable_goal": 4}
    if strict and any(source_summary.get(key) != value for key, value in expected_source_counts.items()):
        raise ValueError("source summary does not match frozen v2.1 counts")
    grouped = parse_steps(steps)
    summary, episode_out, crossing_out, excursion_out, screen_out = analyze(grouped, episodes, strict=strict)
    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source artifacts changed during read-only audit")
    summary["source_sha256"] = before
    summary["frozen_source_counts"] = expected_source_counts if strict else {key: source_summary.get(key) for key in expected_source_counts}
    output.mkdir(parents=True, exist_ok=False)
    (output / OUTPUT_FILES[0]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / OUTPUT_FILES[1], episode_out, list(episode_out[0]))
    crossing_fields = list(crossing_out[0]) if crossing_out else ["episode_id", "group", "crossing_index"]
    excursion_fields = list(excursion_out[0]) if excursion_out else ["episode_id", "group", "excursion_index"]
    _write_csv(output / OUTPUT_FILES[2], crossing_out, crossing_fields)
    _write_csv(output / OUTPUT_FILES[3], excursion_out, excursion_fields)
    _write_csv(output / OUTPUT_FILES[4], screen_out, list(screen_out[0]))
    (output / OUTPUT_FILES[5]).write_text(render_report(summary), encoding="utf-8")
    if sorted(path.name for path in output.iterdir()) != sorted(OUTPUT_FILES):
        raise RuntimeError("unexpected output artifact set")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_audit(args.source_root, args.output_root)
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
