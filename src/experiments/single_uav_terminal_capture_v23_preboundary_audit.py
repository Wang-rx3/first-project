"""Offline pre-boundary audit for a possible terminal-capture v2.3 target.

The audit is intentionally a pure reader of the frozen v2.2 CSV/JSON artifacts.
It reconstructs the canonical stable counter, identifies distance-only resets,
and screens a small, fixed set of hold-target radii.  It does not import or
invoke a controller, evaluator, simulator, environment, or rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SOURCE = Path("logs/preliminary/single_uav_terminal_capture_v22_validation_seed2026")
DEFAULT_OUTPUT = Path("logs/preliminary/single_uav_terminal_capture_v23_preboundary_audit_seed2026")

SOURCE_FILES = (
    "steps.csv",
    "episodes.csv",
    "terminal_capture_v22_validation_episodes.csv",
    "terminal_capture_v22_validation_summary.json",
)
OUTPUT_FILES = (
    "terminal_capture_v23_preboundary_audit_summary.json",
    "terminal_capture_v23_preboundary_audit_reset_windows.csv",
    "terminal_capture_v23_preboundary_audit_candidate_scan.csv",
    "terminal_capture_v23_preboundary_audit_report.md",
)
CANDIDATE_TARGETS_M = (0.94, 0.95, 0.96, 0.97)
STABLE_DISTANCE_THRESHOLD_M = 1.0
STABLE_SPEED_THRESHOLD_MPS = 0.5
STABLE_HOLD_STEPS = 5
PREDICTED_SPEED_CAP_MPS = 0.25
PREDICTED_SPEED_DENOMINATOR_M = 1.05

REQUIRED_STEP_FIELDS = {
    "episode_id",
    "step",
    "goal_distance_for_stable_m",
    "goal_speed_for_stable_mps",
    "stable_goal_distance_threshold_m",
    "stable_speed_threshold_mps",
    "stable_hold_steps_required",
    "goal_distance_ok",
    "goal_speed_ok",
    "stable_candidate_step",
    "stable_hold_counter",
    "stable_goal_now",
    "post_step_actual_radial_velocity_mps",
    "post_step_actual_tangential_velocity_mps",
    "post_step_actual_speed_mps",
    "goal_distance_m",
    "goal_recovery_radial_delta_v_mps",
    "goal_recovery_correction_norm_mps",
    "terminal_correction_clipped",
    "terminal_v21_actor_radial_delta_v_mps",
    "terminal_v21_conflict_compensation_radial_mps",
    "terminal_v21_net_radial_pre_transform_mps",
    "terminal_v21_net_radial_post_transform_mps",
    "terminal_v22_enabled",
    "terminal_v22_hold_latched",
    "terminal_v22_hold_enter_event",
    "terminal_v22_hold_release_event",
    "terminal_v22_hold_mode",
    "terminal_v22_restorative_active",
    "terminal_v22_distance_outside_stable_m",
    "terminal_v22_restorative_desired_radial_speed_mps",
    "terminal_v22_restorative_radial_error_mps",
    "terminal_v22_restorative_delta_v_mps",
    "terminal_v22_inside_canonical_goal",
    "stable_goal",
    "goal_distance_source",
    "goal_speed_source",
    "stable_criterion_timing",
    "stable_log_schema_version",
}
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "scenario_seed",
    "noise_seed",
    "scenario_hash",
    "steps",
    "stable_goal",
    "entered_1m",
    "goal_threshold_m",
    "stable_speed_threshold_mps",
    "stable_hold_steps",
    "stable_log_schema_version",
    "v22_hold_entered",
}

STATUS_FLAGS = {
    "source_control_algorithm_modified": False,
    "control_algorithm_modified": False,
    "goal_recovery_controller_modified": False,
    "stable_goal_criterion_modified": False,
    "terminal_capture_v23_added": False,
    "terminal_capture_v23_modified": False,
    "terminal_v22_modified": False,
    "terminal_v21_modified": False,
    "terminal_v2_modified": False,
    "terminal_capture_v22_modified": False,
    "terminal_capture_v21_modified": False,
    "terminal_capture_v2_modified": False,
    "actor_modified": False,
    "actor_network_modified": False,
    "actor_parameters_modified": False,
    "critic_modified": False,
    "reward_modified": False,
    "dynamics_modified": False,
    "action_transform_modified": False,
    "max_correction_modified": False,
    "training_performed": False,
    "formal_evaluation_executed_by_codex": False,
    "smoke_evaluation_executed_by_codex": False,
    "rollout_executed": False,
    "offline_preboundary_audit_executed": True,
}


def _float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field}: expected finite number, got {value!r}")
    return number


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


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read one UTF-8 CSV in read-only mode and retain its exact schema."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def _validate_schema(fieldnames: Sequence[str], required: set[str], label: str) -> None:
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")


def parse_steps(rows: Sequence[Mapping[str, Any]], strict: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Parse and independently reconstruct the frozen stable-counter contract."""
    if not rows:
        raise ValueError("steps.csv is empty")
    _validate_schema(list(rows[0]), REQUIRED_STEP_FIELDS, "steps.csv")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    last_step: dict[str, int] = {}
    counters: dict[str, int] = defaultdict(int)
    for raw in rows:
        episode_id = str(raw["episode_id"])
        if not episode_id:
            raise ValueError("empty episode_id")
        step = _int(raw["step"], "step")
        key = (episode_id, step)
        if key in seen:
            raise ValueError(f"duplicate episode-step: {key}")
        if episode_id in last_step and step != last_step[episode_id] + 1:
            raise ValueError(f"non-contiguous/non-monotonic steps for {episode_id}: {last_step[episode_id]} -> {step}")
        seen.add(key)
        last_step[episode_id] = step
        distance = _float(raw["goal_distance_for_stable_m"], "goal_distance_for_stable_m")
        speed = _float(raw["goal_speed_for_stable_mps"], "goal_speed_for_stable_mps")
        distance_threshold = _float(raw["stable_goal_distance_threshold_m"], "stable_goal_distance_threshold_m")
        speed_threshold = _float(raw["stable_speed_threshold_mps"], "stable_speed_threshold_mps")
        hold_steps = _int(raw["stable_hold_steps_required"], "stable_hold_steps_required")
        if strict and (
            distance_threshold != STABLE_DISTANCE_THRESHOLD_M
            or speed_threshold != STABLE_SPEED_THRESHOLD_MPS
            or hold_steps != STABLE_HOLD_STEPS
        ):
            raise ValueError(f"canonical stable thresholds mismatch at {key}: {distance_threshold}, {speed_threshold}, {hold_steps}")
        distance_ok = distance <= distance_threshold
        speed_ok = speed <= speed_threshold
        candidate = distance_ok and speed_ok
        counters[episode_id] = counters[episode_id] + 1 if candidate else 0
        stable_now = candidate and counters[episode_id] >= hold_steps
        for field, expected in {
            "goal_distance_ok": distance_ok,
            "goal_speed_ok": speed_ok,
            "stable_candidate_step": candidate,
            "stable_goal_now": stable_now,
        }.items():
            if _bool(raw[field], field) != expected:
                raise ValueError(f"canonical {field} mismatch at {key}")
        if _int(raw["stable_hold_counter"], "stable_hold_counter") != counters[episode_id]:
            raise ValueError(f"stable counter mismatch at {key}")
        item = dict(raw)
        item.update(
            {
                "_step": step,
                "_distance": distance,
                "_speed": speed,
                "_distance_ok": distance_ok,
                "_speed_ok": speed_ok,
                "_candidate": candidate,
                "_counter": counters[episode_id],
                "_stable": stable_now,
                "_radial": _float(raw["post_step_actual_radial_velocity_mps"], "post_step_actual_radial_velocity_mps"),
                "_actual_speed": _float(raw["post_step_actual_speed_mps"], "post_step_actual_speed_mps"),
                "_hold_latched": _bool(raw["terminal_v22_hold_latched"], "terminal_v22_hold_latched"),
                "_v22_enabled": _bool(raw["terminal_v22_enabled"], "terminal_v22_enabled"),
            }
        )
        grouped[episode_id].append(item)
    return dict(grouped)


def _is_outward_precursor(rows: Sequence[Mapping[str, Any]], index: int) -> bool:
    """Whether row *j* is an outward precursor under the fixed two-step rule."""
    if index < 0 or index >= len(rows):
        return False
    row = rows[index]
    if not (float(row["_distance"]) < STABLE_DISTANCE_THRESHOLD_M and float(row["_radial"]) < 0.0):
        return False
    return any(
        index + offset < len(rows) and float(rows[index + offset]["_distance"]) > STABLE_DISTANCE_THRESHOLD_M
        for offset in (1, 2)
    )


def is_outward_precursor(rows: Sequence[Mapping[str, Any]], index: int) -> bool:
    """Public pure predicate used by tests and downstream offline review."""
    return _is_outward_precursor(rows, index)


def _precursor_horizon(rows: Sequence[Mapping[str, Any]], index: int) -> int | None:
    if not _is_outward_precursor(rows, index):
        return None
    if index + 1 < len(rows) and float(rows[index + 1]["_distance"]) > STABLE_DISTANCE_THRESHOLD_M:
        return 1
    return 2


def detect_distance_only_resets(grouped: Mapping[str, Sequence[Mapping[str, Any]]], strict: bool = True) -> list[dict[str, Any]]:
    """Detect the mandated reset transition and attach its outward evidence."""
    resets: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        rows = grouped[episode_id]
        for index in range(1, len(rows)):
            previous, current = rows[index - 1], rows[index]
            if not (
                int(previous["_counter"]) > 0
                and int(current["_counter"]) == 0
                and not bool(current["_distance_ok"])
                and bool(current["_speed_ok"])
            ):
                continue
            precursor_indices = [
                j for j in (index - 2, index - 1) if j >= 0 and _is_outward_precursor(rows, j)
            ]
            precursor_horizons = [_precursor_horizon(rows, j) for j in precursor_indices]
            resets.append(
                {
                    "episode_id": episode_id,
                    "previous_step": int(previous["_step"]),
                    "reset_step": int(current["_step"]),
                    "previous_counter": int(previous["_counter"]),
                    "current_counter": int(current["_counter"]),
                    "current_distance_m": float(current["_distance"]),
                    "current_speed_mps": float(current["_speed"]),
                    "current_radial_velocity_mps": float(current["_radial"]),
                    "precursor_indices": precursor_indices,
                    "precursor_steps": [int(rows[j]["_step"]) for j in precursor_indices],
                    "precursor_horizons": precursor_horizons,
                    "outward_precursor": bool(precursor_indices),
                }
            )
    if strict and len(resets) != 9:
        raise ValueError(f"expected exactly 9 distance-only resets, found {len(resets)}")
    return resets


# Short aliases make the pure contract easy to discover without changing the
# actual frozen-field semantics.
classify_resets = detect_distance_only_resets


def candidate_predicted_inward_speed(distance_m: float, target_radius_m: float) -> float:
    """Return .25*clip((distance-target)/(1.05-target), 0, 1) in m/s."""
    distance = _float(distance_m, "distance_m")
    target = _float(target_radius_m, "target_radius_m")
    if target >= 0.98:
        raise ValueError("target_radius_m must be below 0.98 m")
    fraction = max(0.0, min(1.0, (distance - target) / (PREDICTED_SPEED_DENOMINATOR_M - target)))
    return PREDICTED_SPEED_CAP_MPS * fraction


def nearest_rank_percentile(values: Iterable[float], probability: float = 0.90) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    rank = max(1, math.ceil(probability * len(data)))
    return data[rank - 1]


def _metric(values: Sequence[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean_mps": sum(data) / len(data) if data else None,
        "p90_nearest_rank_mps": nearest_rank_percentile(data, 0.90),
        "max_mps": max(data) if data else None,
    }


def _hold_entered_ids(episode_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["episode_id"]) for row in episode_rows if _bool(row["v22_hold_entered"], "v22_hold_entered")}


def _stable_ids(episode_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["episode_id"]) for row in episode_rows if _bool(row["stable_goal"], "stable_goal")}


def evaluate_candidates(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_rows: Sequence[Mapping[str, Any]],
    resets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Screen candidate radii using all guard-active, hold-latched source rows."""
    hold_ids = _hold_entered_ids(episode_rows)
    stable_ids = _stable_ids(episode_rows)
    precursor_ids = {
        str(item["episode_id"])
        for item in resets
        if bool(item["outward_precursor"])
    }
    all_hold_rows = [
        row
        for episode_id in sorted(hold_ids)
        for row in grouped.get(episode_id, ())
        if bool(row["_v22_enabled"]) and bool(row["_hold_latched"])
    ]
    stable_hold_rows = [row for row in all_hold_rows if str(row["episode_id"]) in stable_ids]
    scan: list[dict[str, Any]] = []
    for target in CANDIDATE_TARGETS_M:
        covered_resets = 0
        for reset in resets:
            rows = grouped[str(reset["episode_id"])]
            covered = any(float(rows[index]["_distance"]) > target for index in reset["precursor_indices"])
            covered_resets += int(covered)

        subset_metrics: dict[str, dict[str, Any]] = {}
        for subset_name, subset_ids in (
            ("all_pop", hold_ids),
            ("precursor_episodes", precursor_ids),
            ("stable_goal_episodes", stable_ids),
        ):
            source_rows = [row for row in all_hold_rows if str(row["episode_id"]) in subset_ids]
            eligible = [row for row in source_rows if float(row["_distance"]) > target]
            predictions = [candidate_predicted_inward_speed(float(row["_distance"]), target) for row in eligible]
            subset_metrics[subset_name] = {
                **_metric(predictions),
                "eligible_steps": len(eligible),
                "episode_count": len({str(row["episode_id"]) for row in eligible}),
                "stable_episodes_touched": len({str(row["episode_id"]) for row in eligible if str(row["episode_id"]) in stable_ids}),
            }
        stable_denominator = len(stable_hold_rows)
        stable_numerator = sum(float(row["_distance"]) > target for row in stable_hold_rows)
        stable_fraction = stable_numerator / stable_denominator if stable_denominator else None
        all_pop = subset_metrics["all_pop"]
        row = {
            "target_radius_m": target,
            "reset_count": len(resets),
            "outward_precursor_count": sum(bool(item["outward_precursor"]) for item in resets),
            "coverage_count": covered_resets,
            "coverage_fraction": covered_resets / len(resets) if resets else None,
            "successful_activation_numerator": stable_numerator,
            "successful_activation_denominator": stable_denominator,
            "successful_activation_fraction": stable_fraction,
            "stable_episodes_touched": subset_metrics["stable_goal_episodes"]["stable_episodes_touched"],
            "all_pop_guard_active_hold_latched_distance_gt_target_steps": all_pop["eligible_steps"],
            "all_pop_p90_nearest_rank_predicted_inward_speed_mps": all_pop["p90_nearest_rank_mps"],
            "all_pop_max_predicted_inward_speed_mps": all_pop["max_mps"],
            "subset_metrics": subset_metrics,
        }
        row["passes_gate"] = bool(
            row["reset_count"] == 9
            and row["outward_precursor_count"] >= 7
            and row["coverage_count"] >= 7
            and stable_fraction is not None
            and stable_fraction <= 0.50
            and all_pop["p90_nearest_rank_mps"] is not None
            and all_pop["max_mps"] is not None
            and all_pop["p90_nearest_rank_mps"] <= PREDICTED_SPEED_CAP_MPS
            and all_pop["max_mps"] <= PREDICTED_SPEED_CAP_MPS
        )
        scan.append(row)
    selected = next((row for row in reversed(scan) if row["passes_gate"]), None)
    return scan, {
        "hold_entered_episode_count": len(hold_ids),
        "hold_entered_episode_ids": sorted(hold_ids),
        "precursor_episode_count": len(precursor_ids),
        "precursor_episode_ids": sorted(precursor_ids),
        "stable_goal_episode_count": len(stable_ids),
        "stable_goal_episode_ids": sorted(stable_ids),
        "stable_hold_latched_step_denominator": len(stable_hold_rows),
        "selected": selected,
    }


def build_reset_windows(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    resets: Sequence[Mapping[str, Any]],
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Export exactly t-3..t with actual frozen columns for every reset."""
    fields = (
        "episode_id",
        "reset_step",
        "relative_step",
        "step",
        "goal_distance_m",
        "goal_distance_for_stable_m",
        "goal_speed_for_stable_mps",
        "post_step_actual_radial_velocity_mps",
        "post_step_actual_tangential_velocity_mps",
        "post_step_actual_speed_mps",
        "actor_raw_radial_component_mps",
        "v21_conflict_compensation_radial_mps",
        "goal_recovery_radial_component_mps",
        "net_radial_pre_transform_mps",
        "net_radial_post_transform_mps",
        "correction_norm_mps",
        "correction_clipped",
        "terminal_v21_actor_radial_delta_v_mps",
        "terminal_v21_conflict_compensation_radial_mps",
        "goal_recovery_radial_delta_v_mps",
        "terminal_v21_net_radial_pre_transform_mps",
        "terminal_v21_net_radial_post_transform_mps",
        "goal_recovery_correction_norm_mps",
        "terminal_correction_clipped",
        "stable_hold_counter",
        "stable_goal_now",
        "terminal_v22_enabled",
        "terminal_v22_hold_latched",
        "terminal_v22_hold_enter_event",
        "terminal_v22_hold_release_event",
        "terminal_v22_hold_mode",
        "terminal_v22_restorative_active",
        "terminal_v22_distance_outside_stable_m",
        "terminal_v22_restorative_desired_radial_speed_mps",
        "terminal_v22_restorative_radial_error_mps",
        "terminal_v22_restorative_delta_v_mps",
        "terminal_v22_inside_canonical_goal",
        "goal_distance_source",
        "goal_speed_source",
        "stable_criterion_timing",
        "stable_log_schema_version",
        "outward_precursor",
    )
    output: list[dict[str, Any]] = []
    for reset in resets:
        rows = grouped[str(reset["episode_id"])]
        current_index = next(i for i, row in enumerate(rows) if int(row["_step"]) == int(reset["reset_step"]))
        if strict and current_index < 3:
            raise ValueError(f"reset window unavailable for {reset['episode_id']} at step {reset['reset_step']}")
        start = current_index - 3 if current_index >= 3 else 0
        window_indices = list(range(start, current_index + 1))
        if strict and len(window_indices) != 4:
            raise ValueError(f"reset window is not exactly t-3..t for {reset['episode_id']} at step {reset['reset_step']}")
        for index in window_indices:
            row = rows[index]
            mapped = {field: row.get(field, "") for field in fields if field not in {"episode_id", "reset_step", "relative_step", "step", "outward_precursor"}}
            mapped.update(
                {
                    "episode_id": row["episode_id"],
                    "reset_step": reset["reset_step"],
                    "relative_step": int(row["_step"]) - int(reset["reset_step"]),
                    "step": row["step"],
                    "goal_distance_m": row["goal_distance_m"],
                    "actor_raw_radial_component_mps": row["terminal_v21_actor_radial_delta_v_mps"],
                    "v21_conflict_compensation_radial_mps": row["terminal_v21_conflict_compensation_radial_mps"],
                    "goal_recovery_radial_component_mps": row["goal_recovery_radial_delta_v_mps"],
                    "net_radial_pre_transform_mps": row["terminal_v21_net_radial_pre_transform_mps"],
                    "net_radial_post_transform_mps": row["terminal_v21_net_radial_post_transform_mps"],
                    "correction_norm_mps": row["goal_recovery_correction_norm_mps"],
                    "correction_clipped": row["terminal_correction_clipped"],
                    "outward_precursor": int(_is_outward_precursor(rows, index)),
                }
            )
            output.append(mapped)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_constant_error(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_json_constant_error)
    if not isinstance(payload, dict):
        raise ValueError("summary JSON must be an object")
    return payload


def _validate_episode_rows(episode_rows: Sequence[Mapping[str, Any]], strict: bool = True) -> None:
    if not episode_rows:
        raise ValueError("episodes.csv is empty")
    _validate_schema(list(episode_rows[0]), REQUIRED_EPISODE_FIELDS, "episodes.csv")
    ids = [str(row["episode_id"]) for row in episode_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode_id in episodes.csv")
    for row in episode_rows:
        if strict and _int(row["steps"], "steps") not in {89, 90, 93, 96, 100}:
            raise ValueError(f"unexpected frozen episode step count for {row['episode_id']}")
        if strict and (
            _float(row["goal_threshold_m"], "goal_threshold_m") != STABLE_DISTANCE_THRESHOLD_M
            or _float(row["stable_speed_threshold_mps"], "stable_speed_threshold_mps") != STABLE_SPEED_THRESHOLD_MPS
            or _int(row["stable_hold_steps"], "stable_hold_steps") != STABLE_HOLD_STEPS
        ):
            raise ValueError(f"episode stable thresholds mismatch at {row['episode_id']}")


def _validate_episode_step_counts(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Cross-check the per-episode row count against both episode ledgers."""
    metadata = {str(row["episode_id"]): row for row in episode_rows}
    if set(metadata) != set(grouped):
        raise ValueError("steps.csv and episode ledgers have different episode identities")
    for episode_id, rows in grouped.items():
        declared = _int(metadata[episode_id]["steps"], f"steps[{episode_id}]")
        if declared != len(rows):
            raise ValueError(f"episode step-count mismatch for {episode_id}: declared {declared}, observed {len(rows)}")


def _preflight(
    source: Path,
    paths: Mapping[str, Path],
    steps_rows: Sequence[Mapping[str, Any]],
    steps_schema: Sequence[str],
    episode_rows: Sequence[Mapping[str, Any]],
    episode_schema: Sequence[str],
    capture_rows: Sequence[Mapping[str, Any]],
    capture_schema: Sequence[str],
    source_summary: Mapping[str, Any],
    strict: bool,
) -> dict[str, Any]:
    _validate_schema(steps_schema, REQUIRED_STEP_FIELDS, "steps.csv")
    _validate_episode_rows(episode_rows, strict=strict)
    if list(episode_schema) != list(capture_schema) or list(episode_rows) != list(capture_rows):
        raise ValueError("episodes.csv and terminal_capture_v22_validation_episodes.csv disagree")
    if strict:
        if len(episode_rows) != 20 or len({str(row["episode_id"]) for row in episode_rows}) != 20:
            raise ValueError("canonical input must contain exactly 20 unique episodes")
        if len(steps_rows) != 1968:
            raise ValueError("canonical input must contain exactly 1968 steps")
        if source_summary.get("stage") != "single_uav_terminal_capture_v22_validation":
            raise ValueError("source summary stage is not v2.2 validation")
        if (
            _float(source_summary.get("stable_distance_threshold"), "stable_distance_threshold") != STABLE_DISTANCE_THRESHOLD_M
            or _float(source_summary.get("stable_speed_threshold"), "stable_speed_threshold") != STABLE_SPEED_THRESHOLD_MPS
            or _int(source_summary.get("stable_hold_steps"), "stable_hold_steps") != STABLE_HOLD_STEPS
        ):
            raise ValueError("source summary stable thresholds do not match the frozen contract")
        frozen_counts = {
            "episodes": 20,
            "stable_goal": 4,
            "entered_1m": 8,
            "stable_counter_reset_count": 9,
            "distance_only_reset_count": 9,
        }
        for key, expected in frozen_counts.items():
            if source_summary.get(key) != expected:
                raise ValueError(f"source summary frozen count mismatch for {key}")
    manifest = {}
    for name in SOURCE_FILES:
        path = paths[name]
        manifest[name] = {
            "relative_identifier": f"{source.name}/{name}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return {
        "source_identifier": source.name,
        "source_files": manifest,
        "schemas": {
            "steps.csv": list(steps_schema),
            "episodes.csv": list(episode_schema),
            "terminal_capture_v22_validation_episodes.csv": list(capture_schema),
            "terminal_capture_v22_validation_summary.json": sorted(source_summary),
        },
        "row_counts": {
            "steps.csv": len(steps_rows),
            "episodes.csv": len(episode_rows),
            "terminal_capture_v22_validation_episodes.csv": len(capture_rows),
        },
        "frozen_contract": {
            "episodes": 20,
            "steps": 1968,
            "unique_contiguous_episode_step_keys": True,
            "cross_file_episode_identity": True,
            "stable_distance_threshold_m": STABLE_DISTANCE_THRESHOLD_M,
            "stable_speed_threshold_mps": STABLE_SPEED_THRESHOLD_MPS,
            "stable_hold_steps": STABLE_HOLD_STEPS,
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_hold_target_radius_m")
    lines = [
        "# Terminal Capture v2.3 Pre-Boundary Offline Audit",
        "",
        f"## Verdict: **{summary['verdict']}**",
        "",
        "This report is a read-only audit of frozen v2.2 trajectories. It does not change a controller and does not run a rollout.",
        "",
        f"- Frozen data: {summary['preflight']['row_counts']['episodes.csv']} episodes and {summary['preflight']['row_counts']['steps.csv']} steps.",
        f"- Distance-only resets: {summary['reset_count']} (required exactly 9).",
        f"- Outward precursor rows: {summary['evidence']['outward_precursor_row_count']} ({summary['evidence']['one_step_precursor_count']} one-step resets, {summary['evidence']['two_step_precursor_count']} two-step-only resets; row counts {summary['evidence']['one_step_precursor_row_count']}/{summary['evidence']['two_step_precursor_row_count']}).",
        f"- Selected target: {selected if selected is not None else 'none'} m.",
        f"- Stable-episode successful activation fraction: {summary['fraction']}",
        "",
        "## Fixed gate",
        "",
        "The gate is reset count = 9, outward precursor count ≥ 7, candidate coverage ≥ 7, stable-episode activation fraction ≤ 0.50, and all-population p90/max predicted inward speed ≤ 0.25 m/s. Candidates are screened least-intervention-first (.97, .96, .95, .94).",
        "",
        "```json",
        json.dumps(summary["gate"], indent=2, sort_keys=True),
        "```",
        "",
        "## Candidate scan",
        "",
        "```json",
        json.dumps(summary["candidate_scan"], indent=2, sort_keys=True),
        "```",
        "",
        "## Limitations",
        "",
        "- This is observational evidence from one frozen seed and does not predict a changed-controller trajectory.",
        "- The canonical stable boundary remains 1.0 m; candidate radii are controller-screening targets only.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT, strict: bool = True) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("output must not overlap source")
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    paths = {name: source / name for name in SOURCE_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    steps_rows, steps_schema = read_csv(paths["steps.csv"])
    episode_rows, episode_schema = read_csv(paths["episodes.csv"])
    capture_rows, capture_schema = read_csv(paths["terminal_capture_v22_validation_episodes.csv"])
    source_summary = _load_json(paths["terminal_capture_v22_validation_summary.json"])
    preflight = _preflight(source, paths, steps_rows, steps_schema, episode_rows, episode_schema, capture_rows, capture_schema, source_summary, strict)
    grouped = parse_steps(steps_rows, strict=strict)
    if strict and (len(grouped) != 20 or sum(len(rows) for rows in grouped.values()) != 1968):
        raise ValueError("parsed canonical input must contain exactly 20 episodes and 1968 steps")
    _validate_episode_step_counts(grouped, episode_rows)
    resets = detect_distance_only_resets(grouped, strict=strict)
    reset_windows = build_reset_windows(grouped, resets)
    candidate_scan, population = evaluate_candidates(grouped, episode_rows, resets)
    selected = population["selected"]
    if selected is None:
        verdict = "NOT_READY_FOR_V23"
        selected_target = None
        evidence = {}
        coverage = None
        fraction = None
    else:
        verdict = "AUDIT_READY_FOR_V23"
        selected_target = selected["target_radius_m"]
        evidence = {
            "outward_precursor_count": selected["outward_precursor_count"],
            "all_pop_p90_nearest_rank_predicted_inward_speed_mps": selected["all_pop_p90_nearest_rank_predicted_inward_speed_mps"],
            "all_pop_max_predicted_inward_speed_mps": selected["all_pop_max_predicted_inward_speed_mps"],
            "stable_episodes_touched": selected["stable_episodes_touched"],
        }
        coverage = selected["coverage_count"]
        fraction = selected["successful_activation_fraction"]
    precursor_rows = [
        (episode_id, index, _precursor_horizon(rows, index))
        for episode_id, rows in grouped.items()
        for index in range(len(rows))
        if any(int(rows[index]["_step"]) == step for reset in resets for step in reset["precursor_steps"] if reset["episode_id"] == episode_id)
        and _is_outward_precursor(rows, index)
    ]
    one_step_rows = sum(horizon == 1 for _, _, horizon in precursor_rows)
    two_step_rows = sum(horizon == 2 for _, _, horizon in precursor_rows)
    # A reset with both horizons is counted once as one-step evidence. This
    # avoids double-counting the t-2 precursor when t-1 already predicts the
    # immediate outside transition.
    reset_one_step = 0
    reset_two_step = 0
    for reset in resets:
        horizons = [_precursor_horizon(grouped[str(reset["episode_id"])], index) for index in reset["precursor_indices"]]
        if 1 in horizons:
            reset_one_step += 1
        elif 2 in horizons:
            reset_two_step += 1
    after_hashes = {name: sha256(path) for name, path in paths.items()}
    before_hashes = {name: details["sha256"] for name, details in preflight["source_files"].items()}
    after_sizes = {name: path.stat().st_size for name, path in paths.items()}
    before_sizes = {name: details["size_bytes"] for name, details in preflight["source_files"].items()}
    if before_hashes != after_hashes or before_sizes != after_sizes:
        raise RuntimeError("source artifacts changed during read-only audit")
    preflight["rehash_matches_preflight"] = True
    summary = {
        "stage": "single_uav_terminal_capture_v23_preboundary_audit",
        "verdict": verdict,
        "selected_hold_target_radius_m": selected_target,
        "reset_count": len(resets),
        "distance_only_reset_count": len(resets),
        "coverage": coverage,
        "candidate_coverage_count": coverage,
        "fraction": fraction,
        "successful_activation_fraction": fraction,
        "outward_precursor_count": sum(bool(item["outward_precursor"]) for item in resets),
        "stable_episodes_touched": evidence.get("stable_episodes_touched"),
        "evidence": {
            "outward_precursor_row_count": len(precursor_rows),
            "one_step_precursor_count": reset_one_step,
            "two_step_precursor_count": reset_two_step,
            "one_step_precursor_row_count": one_step_rows,
            "two_step_precursor_row_count": two_step_rows,
            "outward_precursor_reset_count": sum(bool(item["outward_precursor"]) for item in resets),
            **evidence,
        },
        "selection_evidence": evidence,
        "reset_coverage": coverage,
        "successful_episode_intervention_fraction": fraction,
        "scanned_scope": {
            "source_root_id": "logs/preliminary/single_uav_terminal_capture_v22_validation_seed2026",
            "input_files": list(SOURCE_FILES),
            "episodes": len(grouped),
            "steps": sum(len(rows) for rows in grouped.values()),
            "hold_entered_episodes": population["hold_entered_episode_count"],
            "guard_definition": "terminal_v22_enabled=1 and terminal_v22_hold_latched=1",
            "distance_field": "steps.csv:goal_distance_for_stable_m",
            "stable_episode_field": "episodes.csv:stable_goal",
        },
        "missing": {
            "source_files": [],
            "required_step_fields": [],
            "required_episode_fields": [],
            "reset_windows": [],
        },
        "duplicate": {
            "episode_step_keys": 0,
            "episode_ids": 0,
            "episode_ledger_rows": 0,
        },
        "order": {
            "episode_step_keys_unique": True,
            "episode_steps_contiguous": True,
            "candidate_scan_ascending": True,
            "selection_order_least_intervention_first": list(reversed(CANDIDATE_TARGETS_M)),
        },
        "data_dictionary": {
            "goal_distance_m": "steps.csv:goal_distance_m",
            "goal_distance_for_stable_m": "steps.csv:goal_distance_for_stable_m",
            "actual_speed_mps": "steps.csv:post_step_actual_speed_mps",
            "actual_radial_velocity_mps": "steps.csv:post_step_actual_radial_velocity_mps",
            "actual_tangential_velocity_mps": "steps.csv:post_step_actual_tangential_velocity_mps",
            "stable_hold_counter": "steps.csv:stable_hold_counter",
            "hold_latched": "steps.csv:terminal_v22_hold_latched",
            "restorative_active": "steps.csv:terminal_v22_restorative_active",
            "restorative_desired_radial_speed_mps": "steps.csv:terminal_v22_restorative_desired_radial_speed_mps",
            "actor_raw_radial_component_mps": "steps.csv:terminal_v21_actor_radial_delta_v_mps",
            "v21_conflict_compensation_radial_mps": "steps.csv:terminal_v21_conflict_compensation_radial_mps",
            "goal_recovery_radial_component_mps": "steps.csv:goal_recovery_radial_delta_v_mps",
            "net_radial_pre_transform_mps": "steps.csv:terminal_v21_net_radial_pre_transform_mps",
            "net_radial_post_transform_mps": "steps.csv:terminal_v21_net_radial_post_transform_mps",
            "correction_norm_mps": "steps.csv:goal_recovery_correction_norm_mps",
            "correction_clipped": "steps.csv:terminal_correction_clipped",
        },
        "population": population,
        "candidate_scan": candidate_scan,
        "gate": {
            "reset_count_exact": 9,
            "outward_precursor_count_min": 7,
            "coverage_count_min": 7,
            "successful_activation_fraction_max": 0.50,
            "all_pop_p90_max_mps": PREDICTED_SPEED_CAP_MPS,
            "all_pop_max_max_mps": PREDICTED_SPEED_CAP_MPS,
            "least_intervention_first_order": list(reversed(CANDIDATE_TARGETS_M)),
            "selected_passes": selected is not None,
        },
        "preflight": preflight,
        "source_sha256": before_hashes,
        "status_flags": dict(STATUS_FLAGS),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / OUTPUT_FILES[0]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reset_fields = list(reset_windows[0]) if reset_windows else ["episode_id", "reset_step", "relative_step", "step"]
    _write_csv(output / OUTPUT_FILES[1], reset_windows, reset_fields)
    candidate_rows: list[dict[str, Any]] = []
    for item in candidate_scan:
        flat = {key: value for key, value in item.items() if key != "subset_metrics"}
        for subset, metrics in item["subset_metrics"].items():
            candidate_rows.append({**flat, "subset": subset, **metrics})
    candidate_fields = list(candidate_rows[0]) if candidate_rows else ["target_radius_m", "subset"]
    _write_csv(output / OUTPUT_FILES[2], candidate_rows, candidate_fields)
    (output / OUTPUT_FILES[3]).write_text(render_report(summary), encoding="utf-8")
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
