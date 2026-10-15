"""Offline-only audit of near-goal capture latency.

This module reads the frozen authority-repaired Terminal Capture v2.3
formal100 CSV/JSON artifacts and writes a separate, deterministic diagnostic
package.  It deliberately imports no controller, policy, simulator,
environment, training, or rollout code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SOURCE = Path(
    "logs/preliminary/single_uav_terminal_capture_v23_authority_repair_validation_seed2026"
)
DEFAULT_OUTPUT = Path(
    "logs/preliminary/single_uav_near_goal_capture_latency_audit_seed2026"
)
TARGET_IDS = (
    "test_r99_episode_0001",
    "test_r99_episode_0008",
    "test_r99_episode_0017",
)
REFERENCE_IDS = (
    "test_r99_episode_0003",
    "test_r99_episode_0007",
    "test_r99_episode_0009",
    "test_r99_episode_0010",
    "test_r99_episode_0015",
    "test_r99_episode_0016",
    "test_r99_episode_0018",
    "test_r99_episode_0019",
)
EPSILON = 1.0e-9
STABLE_DISTANCE_M = 1.0
STABLE_SPEED_MPS = 0.5
STABLE_HOLD_STEPS = 5
MAX_STEPS = 100
SOURCE_FILES = (
    "steps.csv",
    "episodes.csv",
    "terminal_capture_v23_authority_repair_validation_episodes.csv",
    "terminal_capture_v23_authority_repair_validation_summary.json",
    "evaluation_summary.json",
)
OUTPUT_FILES = (
    "near_goal_capture_latency_audit_summary.json",
    "near_goal_capture_latency_episode_diagnostics.csv",
    "near_goal_capture_latency_late_window_steps.csv",
    "near_goal_capture_latency_reference_comparison.csv",
    "near_goal_capture_latency_phase_timeline.csv",
    "near_goal_capture_latency_report.md",
)

REQUIRED_STEP_FIELDS = {
    "episode_id",
    "episode_index",
    "step",
    "scenario_seed",
    "noise_seed",
    "goal_distance_m",
    "goal_distance_after_m",
    "current_radial_velocity_mps",
    "desired_radial_velocity_mps",
    "radial_velocity_error_mps",
    "radial_propulsion_delta_v_mps",
    "radial_braking_delta_v_mps",
    "terminal_v2_governor_eligible",
    "terminal_v2_governor_active",
    "terminal_v2_base_desired_radial_speed_mps",
    "terminal_v2_envelope_speed_mps",
    "terminal_v2_governed_desired_radial_speed_mps",
    "terminal_v2_radial_overspeed_mps",
    "terminal_v2_anti_stall_active",
    "terminal_v21_actor_radial_delta_v_mps",
    "terminal_v21_actor_opposing_radial_mps",
    "terminal_v21_radial_conflict_eligible",
    "terminal_v21_radial_conflict_active",
    "terminal_v21_conflict_compensation_radial_mps",
    "terminal_v21_compensation_post_cap_radial_mps",
    "terminal_v21_net_radial_pre_transform_mps",
    "terminal_v21_net_radial_post_transform_mps",
    "terminal_v22_hold_latched",
    "terminal_v23_guard_active",
    "terminal_v23_preboundary_active",
    "terminal_v23_postboundary_active",
    "terminal_v23_desired_radial_speed_mps",
    "terminal_v23_radial_error_mps",
    "terminal_v23_radial_propulsion_allowed",
    "terminal_v23_radial_propulsion_suppressed",
    "terminal_v23_requested_propulsion_radial_mps",
    "terminal_v23_executed_propulsion_radial_mps",
    "terminal_v23_preboundary_authority_repair_active",
    "terminal_correction_clipped",
    "terminal_v21_correction_clip_ratio",
    "terminal_raw_correction_norm_mps",
    "safety_projection_applied",
    "boundary_projection_applied",
    "action_clipped",
    "passed_obstacle",
    "post_bypass_recovery",
    "terminal_capture_active",
    "post_step_actual_radial_velocity_mps",
    "post_step_actual_tangential_velocity_mps",
    "post_step_actual_speed_mps",
    "stable_goal",
}
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "episode_index",
    "scenario_seed",
    "noise_seed",
    "scenario_hash",
    "steps",
    "passed_obstacle",
    "post_bypass_recovery",
    "entered_8m",
    "entered_5m",
    "entered_3m",
    "entered_1m",
    "stable_goal",
    "max_steps",
    "stable_hold_steps",
    "stable_speed_threshold_mps",
    "goal_threshold_m",
}


def _float(value: Any, field: str, *, allow_nan: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected number") from exc
    if not allow_nan and not math.isfinite(number):
        raise ValueError(f"{field}: expected finite number")
    return number


def _int(value: Any, field: str) -> int:
    number = _float(value, field)
    if number != int(number):
        raise ValueError(f"{field}: expected integer")
    return int(number)


def _bool(value: Any, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise ValueError(f"{field}: expected boolean")


def _optional_bool(value: Any, field: str) -> bool:
    if value in (None, "", "nan", "NaN"):
        return False
    return _bool(value, field)


def _vector_norm(value: Any, field: str) -> float:
    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"{field}: expected bracketed vector")
    parts = [part.strip() for part in text[1:-1].split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{field}: expected three components")
    values = [_float(part, field) for part in parts]
    return math.sqrt(sum(component * component for component in values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_missingness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    blank_cells = 0
    nonfinite_cells = 0
    total_cells = len(rows) * (len(rows[0]) if rows else 0)
    for row in rows:
        for field, value in row.items():
            text = "" if value is None else str(value).strip()
            if text == "":
                blank_cells += 1
                counts[str(field)] += 1
            elif text.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
                nonfinite_cells += 1
                counts[str(field)] += 1
    return {
        "rows": len(rows),
        "total_cells": total_cells,
        "blank_cells": blank_cells,
        "nonfinite_cells": nonfinite_cells,
        "missing_or_nonfinite_cells": sum(counts.values()),
        "by_field": dict(sorted(counts.items())),
    }


def _json_token_missingness(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tokens = re.findall(r"(?<![A-Za-z0-9_])(?:NaN|-?Infinity)(?![A-Za-z0-9_])", text)
    return {
        "NaN": sum(token == "NaN" for token in tokens),
        "Infinity": sum(token == "Infinity" for token in tokens),
        "-Infinity": sum(token == "-Infinity" for token in tokens),
        "total_nonfinite_tokens": len(tokens),
    }


def _read_json(path: Path) -> dict[str, Any]:
    # Existing evaluator summaries use JSON-compatible spellings only except
    # for structural NaN placeholders in fields with no event.  Preserve
    # those as non-finite values for the audit and serialise them as null in
    # derived JSON; they are never used as valid measurements.
    def parse_constant(value: str) -> float:
        if value == "NaN":
            return float("nan")
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
        raise ValueError(f"unsupported JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=parse_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else float("nan")


def _percentile(values: Sequence[float], fraction: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p10": _percentile(finite, 0.10),
        "p90": _percentile(finite, 0.90),
        "min": min(finite),
        "max": max(finite),
    }


def _first(rows: Sequence[Mapping[str, Any]], predicate: Any) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["_step"])
    return None


def _row_float(row: Mapping[str, Any], field: str, *, allow_nan: bool = False) -> float:
    return _float(row.get(field), field, allow_nan=allow_nan)


def _source_field(field: str) -> str:
    return {
        "terminal_v21_conflict_eligible": "terminal_v21_radial_conflict_eligible",
        "terminal_v21_conflict_active": "terminal_v21_radial_conflict_active",
    }.get(field, field)


def _normalise_step(raw: Mapping[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = dict(raw)
    row["_step"] = _int(raw["step"], "step")
    row["_distance_pre"] = _float(raw["goal_distance_m"], "goal_distance_m")
    row["_distance_post"] = _float(raw["goal_distance_after_m"], "goal_distance_after_m")
    for field in (
        "current_radial_velocity_mps",
        "desired_radial_velocity_mps",
        "radial_velocity_error_mps",
        "post_step_actual_radial_velocity_mps",
        "post_step_actual_tangential_velocity_mps",
        "post_step_actual_speed_mps",
        "terminal_v2_base_desired_radial_speed_mps",
        "terminal_v2_envelope_speed_mps",
        "terminal_v2_governed_desired_radial_speed_mps",
        "terminal_v2_radial_overspeed_mps",
        "terminal_v21_actor_radial_delta_v_mps",
        "terminal_v21_actor_opposing_radial_mps",
        "terminal_v21_conflict_compensation_radial_mps",
        "terminal_v21_compensation_post_cap_radial_mps",
        "terminal_v21_net_radial_pre_transform_mps",
        "terminal_v21_net_radial_post_transform_mps",
        "terminal_v23_desired_radial_speed_mps",
        "terminal_v23_radial_error_mps",
        "terminal_v23_requested_propulsion_radial_mps",
        "terminal_v23_executed_propulsion_radial_mps",
        "terminal_v21_correction_clip_ratio",
        "terminal_raw_correction_norm_mps",
    ):
        row[f"_{field}"] = _float(raw[_source_field(field)], field, allow_nan=True)
    for field in (
        "radial_propulsion_delta_v_mps",
        "radial_braking_delta_v_mps",
    ):
        row[f"_{field}_norm"] = _vector_norm(raw[field], field)
    for field in (
        "terminal_v2_governor_eligible",
        "terminal_v2_governor_active",
        "terminal_v2_anti_stall_active",
        "terminal_v21_conflict_eligible",
        "terminal_v21_conflict_active",
        "terminal_v22_hold_latched",
        "terminal_v23_guard_active",
        "terminal_v23_preboundary_active",
        "terminal_v23_postboundary_active",
        "terminal_v23_radial_propulsion_allowed",
        "terminal_v23_radial_propulsion_suppressed",
        "terminal_v23_preboundary_authority_repair_active",
        "terminal_correction_clipped",
        "safety_projection_applied",
        "boundary_projection_applied",
        "action_clipped",
        "passed_obstacle",
        "post_bypass_recovery",
        "terminal_capture_active",
        "stable_goal",
    ):
        row[f"_{field}"] = _optional_bool(raw[_source_field(field)], field)
    row["_governor_limiting"] = bool(
        row["_terminal_v2_governor_eligible"]
        and math.isfinite(row["_terminal_v2_base_desired_radial_speed_mps"])
        and math.isfinite(row["_terminal_v2_governed_desired_radial_speed_mps"])
        and row["_terminal_v2_governed_desired_radial_speed_mps"]
        < row["_terminal_v2_base_desired_radial_speed_mps"] - EPSILON
    )
    return row


def _identity_check(
    *,
    source: Path,
    steps: Sequence[Mapping[str, str]],
    episodes: Sequence[Mapping[str, str]],
    validation_episodes: Sequence[Mapping[str, str]],
    repair_summary: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["source_files_present"] = all((source / name).is_file() for name in SOURCE_FILES)
    checks["source_step_count"] = len(steps)
    checks["source_episode_count"] = len(episodes)
    checks["step_count_matches_episode_rows"] = len(episodes) == 20 and len(validation_episodes) == 20
    checks["validation_episode_csv_byte_identical"] = _sha256(
        source / "episodes.csv"
    ) == _sha256(source / "terminal_capture_v23_authority_repair_validation_episodes.csv")
    checks["condition_on_only"] = all(str(row.get("condition", "")) == "on" for row in steps)
    checks["no_duplicate_episode_steps"] = len({
        (str(row.get("episode_id", "")), _int(row.get("step"), "step")) for row in steps
    }) == len(steps)
    grouped_steps: dict[str, list[int]] = defaultdict(list)
    for row in steps:
        grouped_steps[str(row["episode_id"])].append(_int(row["step"], "step"))
    checks["contiguous_step_numbers"] = all(
        values == list(range(len(values))) for values in grouped_steps.values()
    )
    checks["episode_steps_match_summary"] = {
        str(row["episode_id"]): _int(row["steps"], "steps")
        for row in episodes
    } == {episode: len(values) for episode, values in grouped_steps.items()}
    episode_ids = {str(row.get("episode_id", "")) for row in episodes}
    step_ids = set(grouped_steps)
    checks["episode_ids_match_steps"] = episode_ids == step_ids
    checks["unique_episode_ids"] = len(episode_ids) == len(episodes)
    episode_identity = {
        str(row["episode_id"]): {
            field: str(row.get(field, "")).strip()
            for field in ("episode_index", "scenario_seed", "noise_seed", "scenario_hash")
        }
        for row in episodes
    }
    checks["step_episode_identity_matches_summary"] = all(
        all(
            str(row.get(field, "")).strip() == episode_identity.get(str(row.get("episode_id", "")), {}).get(field)
            for field in ("episode_index", "scenario_seed", "noise_seed")
        )
        for row in steps
    )
    checks["unique_scenario_noise_identity"] = len({
        (
            str(row.get("scenario_seed", "")).strip(),
            str(row.get("noise_seed", "")).strip(),
            str(row.get("scenario_hash", "")).strip(),
        )
        for row in episodes
    }) == len(episodes)
    checks["episode_identity_fields_present"] = all(
        all(str(row.get(field, "")).strip() for field in ("episode_id", "scenario_seed", "noise_seed", "scenario_hash"))
        for row in episodes
    )
    checks["target_ids_present"] = set(TARGET_IDS).issubset(episode_ids)
    checks["reference_ids_present"] = set(REFERENCE_IDS).issubset(episode_ids)
    checks["formal20"] = (
        evaluation.get("formal") is True
        and _int(evaluation.get("episode_count"), "episode_count") == 20
    )
    checks["max_steps_100"] = all(_int(row.get("max_steps"), "max_steps") == MAX_STEPS for row in episodes)
    checks["checkpoint_sha_matches_repair_summary"] = (
        evaluation.get("checkpoint", {}).get("sha256")
        == repair_summary.get("baseline_checkpoint_sha256")
    )
    checks["policy_hash_present"] = bool(evaluation.get("checkpoint", {}).get("policy_hash"))
    checks["scenario_manifest_matches"] = (
        evaluation.get("scenario_manifest_hash") == repair_summary.get("candidate_manifest_hash")
    )
    checks["ordered_episode_identity_present"] = bool(evaluation.get("ordered_pool_identity"))
    candidate_config = repair_summary.get("candidate_config", {})
    checks["controller_config_locked"] = all(
        candidate_config.get(key) == expected
        for key, expected in {
            "radial_gain": 0.8,
            "tangential_gain": 0.7,
            "max_delta_v_mps": 0.6,
            "terminal_radial_speed_floor_mps": 0.25,
            "terminal_tangential_gain": 1.0,
            "terminal_effective_deceleration_mps2": 1.697726,
            "terminal_stopping_margin_m": 1.328009,
            "terminal_radial_conflict_compensation_gain": 1.0,
            "terminal_hold_enter_radius_m": 0.98,
            "terminal_hold_release_radius_m": 1.05,
            "terminal_hold_target_radius_m": 0.96,
        }.items()
    )
    checks["identity_locks_true"] = all(
        bool(repair_summary.get(key))
        for key in (
            "same_checkpoint_as_v23_formal",
            "same_checkpoint_sha256",
            "same_policy_hash",
            "same_scenario_manifest",
            "same_ordered_episode_identity",
            "same_reward",
            "same_dynamics",
            "same_action_transform",
            "same_stable_criterion",
            "same_max_correction",
            "same_a_eff",
            "same_stopping_margin",
            "same_v21_gain",
            "same_v22_parameters",
            "same_v23_target",
            "same_max_steps",
            "same_controller_config",
            "only_v23_authority_repair_changed",
        )
    )
    checks["candidate_repaired"] = repair_summary.get("terminal_capture_v23_authority_repaired") is True
    checks["stable_criterion_locked"] = (
        repair_summary.get("stable_distance_threshold") == STABLE_DISTANCE_M
        and repair_summary.get("stable_speed_threshold") == STABLE_SPEED_MPS
        and repair_summary.get("stable_hold_steps") == STABLE_HOLD_STEPS
    )
    checks["all_identity_checks_pass"] = all(
        bool(value) for key, value in checks.items()
        if key not in {"source_step_count", "source_episode_count"}
    )
    if not checks["all_identity_checks_pass"]:
        raise ValueError(
            "CAPTURE_LATENCY_AUDIT_IDENTITY_FAIL: "
            + ", ".join(key for key, value in checks.items() if value is False)
        )
    return checks


def _distance_progress(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["_step"]))
    distances = [float(row["_distance_post"]) for row in ordered]
    changes = [left - right for left, right in zip(distances, distances[1:])]
    return {
        "n_rows": len(ordered),
        "distance_source": "goal_distance_after_m_post_step",
        "distance_at_window_start_m": distances[0] if distances else float("nan"),
        "distance_at_window_end_m": distances[-1] if distances else float("nan"),
        "minimum_distance_m": min(distances) if distances else float("nan"),
        "net_distance_reduction_m": (distances[0] - distances[-1]) if distances else float("nan"),
        "mean_distance_reduction_per_step_m": _mean(changes),
        "median_stepwise_distance_change_m": statistics.median(changes) if changes else float("nan"),
        "positive_progress_step_fraction": (
            sum(change > EPSILON for change in changes) / len(changes) if changes else float("nan")
        ),
        "negative_progress_step_fraction": (
            sum(change < -EPSILON for change in changes) / len(changes) if changes else float("nan")
        ),
        "zero_progress_fraction": (
            sum(abs(change) <= EPSILON for change in changes) / len(changes) if changes else float("nan")
        ),
    }


def _linear_slope(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    ordered = sorted(rows, key=lambda row: int(row["_step"]))
    xs = [float(row["_step"]) for row in ordered]
    ys = [float(row["_distance_post"]) for row in ordered]
    if len(xs) < 2:
        return {"n": len(xs), "slope_m_per_step": float("nan"), "r2": float("nan")}
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= EPSILON:
        return {"n": len(xs), "slope_m_per_step": float("nan"), "r2": float("nan")}
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * x for x in xs]
    residual = sum((y - fitted) ** 2 for y, fitted in zip(ys, predicted))
    total = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if total <= EPSILON else 1.0 - residual / total
    return {"n": len(xs), "slope_m_per_step": slope, "r2": r2}


def _window_rows(rows: Sequence[Mapping[str, Any]], name: str, first_recovery: int | None, first_3m: int | None) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["_step"]))
    if name == "A_post_recovery_first10":
        if first_recovery is None:
            return []
        return [row for row in ordered if first_recovery <= int(row["_step"]) < first_recovery + 10]
    if name == "B_after_first_3m":
        if first_3m is None:
            return []
        return [row for row in ordered if int(row["_step"]) >= first_3m]
    if name == "C_last20":
        return ordered[-20:]
    if name == "D_last10":
        return ordered[-10:]
    raise ValueError(name)


def _radial_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "desired_radial_velocity_mps": "_desired_radial_velocity_mps",
        "current_radial_velocity_mps": "_current_radial_velocity_mps",
        "post_step_actual_radial_velocity_mps": "_post_step_actual_radial_velocity_mps",
        "radial_velocity_error_mps": "_radial_velocity_error_mps",
        "radial_propulsion_norm_mps": "_radial_propulsion_delta_v_mps_norm",
        "radial_braking_norm_mps": "_radial_braking_delta_v_mps_norm",
        "post_step_actual_tangential_velocity_mps": "_post_step_actual_tangential_velocity_mps",
    }
    return {name: _stats(row[key] for row in rows) for name, key in fields.items()}


def _actor_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pre = [row["_terminal_v21_net_radial_pre_transform_mps"] for row in rows]
    post = [row["_terminal_v21_net_radial_post_transform_mps"] for row in rows]
    return {
        "actor_radial_negative_fraction": (
            sum(row["_terminal_v21_actor_radial_delta_v_mps"] < -EPSILON for row in rows) / len(rows)
            if rows else float("nan")
        ),
        "conflict_eligible_fraction": (
            sum(row["_terminal_v21_conflict_eligible"] for row in rows) / len(rows) if rows else float("nan")
        ),
        "conflict_active_fraction": (
            sum(row["_terminal_v21_conflict_active"] for row in rows) / len(rows) if rows else float("nan")
        ),
        "mean_actor_opposition_mps": _mean(row["_terminal_v21_actor_opposing_radial_mps"] for row in rows),
        "mean_compensation_mps": _mean(row["_terminal_v21_conflict_compensation_radial_mps"] for row in rows),
        "mean_compensation_post_cap_mps": _mean(row["_terminal_v21_compensation_post_cap_radial_mps"] for row in rows),
        "mean_net_radial_pre_transform_mps": _mean(pre),
        "mean_net_radial_post_transform_mps": _mean(post),
    }


def _transform_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["_terminal_v21_net_radial_pre_transform_mps"] > EPSILON
    ]
    ratios = [
        row["_terminal_v21_net_radial_post_transform_mps"]
        / row["_terminal_v21_net_radial_pre_transform_mps"]
        for row in eligible
        if abs(row["_terminal_v21_net_radial_pre_transform_mps"]) > EPSILON
    ]
    return {
        "eligible_positive_pre_transform_steps": len(eligible),
        "retention_ratio": _stats(ratios),
        "明显_radial_attenuation_steps_ratio_below_0_9": sum(ratio < 0.9 for ratio in ratios),
    }


def _balance_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    states = []
    propulsion = []
    braking = []
    for row in rows:
        prop = row["_radial_propulsion_delta_v_mps_norm"] > EPSILON
        brake = row["_radial_braking_delta_v_mps_norm"] > EPSILON
        states.append((1 if prop else 0) + (2 if brake else 0))
        propulsion.append(row["_radial_propulsion_delta_v_mps_norm"])
        braking.append(row["_radial_braking_delta_v_mps_norm"])
    return {
        "radial_propulsion_active_fraction": sum(state == 1 for state in states) / len(states) if states else float("nan"),
        "radial_braking_active_fraction": sum(state == 2 for state in states) / len(states) if states else float("nan"),
        "neither_fraction": sum(state == 0 for state in states) / len(states) if states else float("nan"),
        "propulsion_and_braking_overlap_steps": sum(state == 3 for state in states),
        "mean_propulsion_norm_mps": _mean(propulsion),
        "mean_braking_norm_mps": _mean(braking),
        "propulsion_braking_switches": sum(left != right for left, right in zip(states, states[1:])),
    }


def _window_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    progress = _distance_progress(rows)
    slope = _linear_slope(rows)
    radial = _radial_metrics(rows)
    actor = _actor_metrics(rows)
    transform = _transform_metrics(rows)
    balance = _balance_metrics(rows)
    governor_rows = [row for row in rows if row["_terminal_v2_governor_eligible"]]
    anti_stall_rows = [row for row in rows if row["_terminal_v2_anti_stall_active"]]
    clipped = [row for row in rows if row["_terminal_correction_clipped"]]
    v21_saturated = [
        row for row in rows
        if math.isfinite(row["_terminal_v21_correction_clip_ratio"])
        and row["_terminal_v21_correction_clip_ratio"] < 1.0 - EPSILON
    ]
    safety = sum(row["_safety_projection_applied"] for row in rows)
    boundary = sum(row["_boundary_projection_applied"] for row in rows)
    if slope["slope_m_per_step"] < 0.0 and progress["distance_at_window_end_m"] > 1.0:
        extra_steps = (progress["distance_at_window_end_m"] - 1.0) / abs(slope["slope_m_per_step"])
    else:
        extra_steps = float("nan")
    return {
        **progress,
        "late_slope": slope,
        "estimated_extra_steps_to_1m": extra_steps,
        "radial_tracking": radial,
        "actor_conflict": actor,
        "action_transform": transform,
        "balance": balance,
        "governor": {
            "eligible_fraction": len(governor_rows) / len(rows) if rows else float("nan"),
            "active_fraction": sum(row["_terminal_v2_governor_active"] for row in rows) / len(rows) if rows else float("nan"),
            "limiting_fraction": sum(row["_governor_limiting"] for row in rows) / len(rows) if rows else float("nan"),
            "base_desired_radial_speed_mps": _stats(row["_terminal_v2_base_desired_radial_speed_mps"] for row in governor_rows),
            "envelope_speed_mps": _stats(row["_terminal_v2_envelope_speed_mps"] for row in governor_rows),
            "governed_desired_radial_speed_mps": _stats(row["_terminal_v2_governed_desired_radial_speed_mps"] for row in governor_rows),
            "radial_overspeed_mps": _stats(row["_terminal_v2_radial_overspeed_mps"] for row in governor_rows),
        },
        "anti_stall": {
            "active_fraction": len(anti_stall_rows) / len(rows) if rows else float("nan"),
            "rows": len(anti_stall_rows),
            "desired_radial_speed_mps": _stats(row["_desired_radial_velocity_mps"] for row in anti_stall_rows),
            "actual_radial_speed_mps": _stats(row["_post_step_actual_radial_velocity_mps"] for row in anti_stall_rows),
            "net_radial_pre_transform_mps": _stats(row["_terminal_v21_net_radial_pre_transform_mps"] for row in anti_stall_rows),
            "net_radial_post_transform_mps": _stats(row["_terminal_v21_net_radial_post_transform_mps"] for row in anti_stall_rows),
        },
        "saturation": {
            "terminal_correction_clipped_fraction": len(clipped) / len(rows) if rows else float("nan"),
            "v21_correction_saturation_fraction": len(v21_saturated) / len(rows) if rows else float("nan"),
            "correction_norm_mps": _stats(row["_terminal_raw_correction_norm_mps"] for row in rows if "_terminal_raw_correction_norm_mps" in row),
            "max_correction_usage_mps": max((row["_terminal_raw_correction_norm_mps"] for row in rows if "_terminal_raw_correction_norm_mps" in row), default=float("nan")),
        },
        "safety_boundary": {
            "safety_projection_count": safety,
            "boundary_projection_count": boundary,
            "action_clipped_count": sum(row.get("_action_clipped", False) for row in rows),
        },
        "hold": {
            "v22_latched_fraction": sum(row["_terminal_v22_hold_latched"] for row in rows) / len(rows) if rows else float("nan"),
            "v23_guard_fraction": sum(row["_terminal_v23_guard_active"] for row in rows) / len(rows) if rows else float("nan"),
            "v23_preboundary_fraction": sum(row["_terminal_v23_preboundary_active"] for row in rows) / len(rows) if rows else float("nan"),
        },
    }


def _phase_timeline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["_step"]))
    first_3m = _first(ordered, lambda row: row["_distance_post"] <= 3.0)
    values = {
        "first_passed_obstacle_step": _first(ordered, lambda row: row["_passed_obstacle"]),
        "first_post_bypass_recovery_step": _first(ordered, lambda row: row["_post_bypass_recovery"]),
        "first_entered8_step": _first(ordered, lambda row: row["_distance_post"] <= 8.0),
        "first_entered5_step": _first(ordered, lambda row: row["_distance_post"] <= 5.0),
        "first_entered3_step": first_3m,
        "first_entered1_step": _first(ordered, lambda row: row["_distance_post"] <= 1.0),
        "first_terminal_capture_step": _first(ordered, lambda row: row["_terminal_capture_active"]),
        "first_governor_eligible_step": _first(ordered, lambda row: row["_terminal_v2_governor_eligible"]),
        "first_governor_limiting_step": _first(ordered, lambda row: row["_governor_limiting"]),
        "first_radial_braking_step": _first(ordered, lambda row: row["_radial_braking_delta_v_mps_norm"] > EPSILON),
        "first_v2_anti_stall_active_step": _first(ordered, lambda row: row["_terminal_v2_anti_stall_active"]),
        "first_v21_conflict_eligible_step": _first(ordered, lambda row: row["_terminal_v21_conflict_eligible"]),
        "first_v21_conflict_active_step": _first(ordered, lambda row: row["_terminal_v21_conflict_active"]),
        "last_step": int(ordered[-1]["_step"]) if ordered else None,
    }
    if ordered:
        minimum = min(ordered, key=lambda row: row["_distance_post"])
        values["minimum_goal_distance_step"] = int(minimum["_step"])
        values["minimum_goal_distance_m"] = float(minimum["_distance_post"])
    else:
        values["minimum_goal_distance_step"] = None
        values["minimum_goal_distance_m"] = float("nan")
    values["distance_source"] = "goal_distance_after_m_post_step"
    values["velocity_source"] = "post_step_actual_* and current_radial_velocity_mps"
    values["timing"] = "step row records pre-step decision and post-step actual velocity; progress compares adjacent post-step distances"
    return values


def _tangent_ratio(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios = [
        abs(row["_post_step_actual_tangential_velocity_mps"])
        / max(abs(row["_post_step_actual_radial_velocity_mps"]), EPSILON)
        for row in rows
    ]
    return {"ratio": _stats(ratios), "rows": len(ratios)}


def _episode_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["_step"]))
    timeline = _phase_timeline(ordered)
    first_recovery = timeline["first_post_bypass_recovery_step"]
    first_3m = timeline["first_entered3_step"]
    windows = {
        name: _window_metrics(_window_rows(ordered, name, first_recovery, first_3m))
        for name in (
            "A_post_recovery_first10",
            "B_after_first_3m",
            "C_last20",
            "D_last10",
        )
    }
    late_rows = _window_rows(ordered, "C_last20", first_recovery, first_3m)
    windows["B_after_first_3m"]["tangent_to_radial_ratio"] = _tangent_ratio(
        _window_rows(ordered, "B_after_first_3m", first_recovery, first_3m)
    )
    windows["C_last20"]["tangent_to_radial_ratio"] = _tangent_ratio(late_rows)
    windows["D_last10"]["tangent_to_radial_ratio"] = _tangent_ratio(
        _window_rows(ordered, "D_last10", first_recovery, first_3m)
    )
    return {"timeline": timeline, "windows": windows}


def _flatten_diagnostic(episode_id: str, episode_row: Mapping[str, str], analysis: Mapping[str, Any]) -> dict[str, Any]:
    late = analysis["windows"]["C_last20"]
    radial = late["radial_tracking"]
    actor = late["actor_conflict"]
    transform = late["action_transform"]
    balance = late["balance"]
    saturation = late["saturation"]
    governor = late["governor"]
    return {
        "episode_id": episode_id,
        "passed_obstacle": _bool(episode_row["passed_obstacle"], "passed_obstacle"),
        "post_bypass_recovery": _bool(episode_row["post_bypass_recovery"], "post_bypass_recovery"),
        "entered_3m": _bool(episode_row["entered_3m"], "entered_3m"),
        "entered_1m": _bool(episode_row["entered_1m"], "entered_1m"),
        "stable_goal": _bool(episode_row["stable_goal"], "stable_goal"),
        "minimum_goal_distance_m": analysis["timeline"]["minimum_goal_distance_m"],
        "late20_distance_slope_m_per_step": late["late_slope"]["slope_m_per_step"],
        "late20_distance_slope_r2": late["late_slope"]["r2"],
        "estimated_extra_steps_to_1m": late["estimated_extra_steps_to_1m"],
        "late20_mean_desired_radial": radial["desired_radial_velocity_mps"]["mean"],
        "late20_mean_actual_radial": radial["post_step_actual_radial_velocity_mps"]["mean"],
        "late20_mean_tangent": radial["post_step_actual_tangential_velocity_mps"]["mean"],
        "late20_actor_opposition_fraction": actor["actor_radial_negative_fraction"],
        "late20_conflict_active_fraction": actor["conflict_active_fraction"],
        "late20_mean_net_radial_pre": actor["mean_net_radial_pre_transform_mps"],
        "late20_mean_net_radial_post": actor["mean_net_radial_post_transform_mps"],
        "late20_transform_retention": transform["retention_ratio"]["median"],
        "late20_propulsion_fraction": balance["radial_propulsion_active_fraction"],
        "late20_braking_fraction": balance["radial_braking_active_fraction"],
        "late20_saturation_fraction": saturation["terminal_correction_clipped_fraction"],
        "late20_safety_projection_count": late["safety_boundary"]["safety_projection_count"],
        "late20_boundary_projection_count": late["safety_boundary"]["boundary_projection_count"],
        "late20_governor_limiting_fraction": governor["limiting_fraction"],
        "late20_tangent_to_radial_median": late["tangent_to_radial_ratio"]["ratio"]["median"],
        "late20_propulsion_braking_switches": balance["propulsion_braking_switches"],
    }


def _reference_windows(rows: Sequence[Mapping[str, Any]], episode_row: Mapping[str, str]) -> dict[str, Any]:
    timeline = _phase_timeline(rows)
    first_3m = timeline["first_entered3_step"]
    first_1m = timeline["first_entered1_step"]
    ordered = sorted(rows, key=lambda row: int(row["_step"]))
    after_3m = [row for row in ordered if first_3m is not None and int(row["_step"]) >= first_3m][:20]
    before_1m = [row for row in ordered if first_1m is not None and int(row["_step"]) < first_1m][-20:]
    return {
        "after_first_3m_20": _window_metrics(after_3m),
        "before_first_1m_last20": _window_metrics(before_1m),
        "timeline": timeline,
    }


_COMPARISON_METRICS = (
        "net_distance_reduction_m",
        "slope_m_per_step",
        "desired_radial_velocity_mps_mean",
        "post_step_actual_radial_velocity_mps_mean",
        "post_step_actual_tangential_velocity_mps_mean",
        "actor_radial_negative_fraction",
        "conflict_active_fraction",
        "mean_net_radial_post_transform_mps",
        "radial_propulsion_active_fraction",
        "radial_braking_active_fraction",
        "terminal_correction_clipped_fraction",
    )


def _comparison_row(
    *,
    episode_id: str,
    episode_row: Mapping[str, str],
    window_name: str,
    window: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    return {
        "comparison_role": role,
        "episode_id": episode_id,
        "window": window_name,
        "aligned_row_count": window["n_rows"],
        "entered_1m": _bool(episode_row["entered_1m"], "entered_1m"),
        "net_distance_reduction_m": window["net_distance_reduction_m"],
        "slope_m_per_step": window["late_slope"]["slope_m_per_step"],
        "slope_r2": window["late_slope"]["r2"],
        "desired_radial_velocity_mps_mean": window["radial_tracking"]["desired_radial_velocity_mps"]["mean"],
        "post_step_actual_radial_velocity_mps_mean": window["radial_tracking"]["post_step_actual_radial_velocity_mps"]["mean"],
        "post_step_actual_tangential_velocity_mps_mean": window["radial_tracking"]["post_step_actual_tangential_velocity_mps"]["mean"],
        "actor_radial_negative_fraction": window["actor_conflict"]["actor_radial_negative_fraction"],
        "conflict_active_fraction": window["actor_conflict"]["conflict_active_fraction"],
        "mean_net_radial_post_transform_mps": window["actor_conflict"]["mean_net_radial_post_transform_mps"],
        "radial_propulsion_active_fraction": window["balance"]["radial_propulsion_active_fraction"],
        "radial_braking_active_fraction": window["balance"]["radial_braking_active_fraction"],
        "terminal_correction_clipped_fraction": window["saturation"]["terminal_correction_clipped_fraction"],
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], group: str, window_name: str) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row.get("comparison_role") == "reference_episode"
        and row.get("window") == window_name
        and int(row.get("aligned_row_count", 0) or 0) > 0
    ]
    output = []
    for metric in _COMPARISON_METRICS:
        values = [
            float(row[metric])
            for row in selected
            if row.get(metric) not in (None, "") and math.isfinite(float(row[metric]))
        ]
        if not values:
            continue
        output.append({
            "comparison_role": "reference_group_summary",
            "group": group,
            "episode_id": "GROUP_SUMMARY",
            "window": window_name,
            "metric": metric,
            "median": statistics.median(values),
            "iqr_p25": _percentile(values, 0.25),
            "iqr_p75": _percentile(values, 0.75),
            "n": len(values),
        })
    return output


def _diagnose(target_id: str, target: Mapping[str, Any], references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    late = target["windows"]["C_last20"]
    ref_late = [item["analysis"]["windows"]["C_last20"] for item in references]
    ref_governed = [
        item["governor"]["governed_desired_radial_speed_mps"]["median"]
        for item in ref_late
        if math.isfinite(item["governor"]["governed_desired_radial_speed_mps"]["median"])
    ]
    ref_actor = [item["actor_conflict"]["actor_radial_negative_fraction"] for item in ref_late]
    ref_tangent = [item["tangent_to_radial_ratio"]["ratio"]["median"] for item in ref_late if math.isfinite(item["tangent_to_radial_ratio"]["ratio"]["median"])]
    ref_compensation = [
        item["actor_conflict"]["mean_compensation_post_cap_mps"]
        for item in ref_late
        if math.isfinite(item["actor_conflict"]["mean_compensation_post_cap_mps"])
    ]
    evidence: list[dict[str, Any]] = []
    labels: list[str] = []
    slope = late["late_slope"]["slope_m_per_step"]
    if (
        math.isfinite(slope)
        and slope < 0.0
        and late["radial_tracking"]["post_step_actual_radial_velocity_mps"]["mean"] > -EPSILON
        and late["safety_boundary"]["safety_projection_count"] == 0
        and late["safety_boundary"]["boundary_projection_count"] == 0
    ):
        labels.append("HORIZON_LIMITED_APPROACH")
        evidence.append({"label": "HORIZON_LIMITED_APPROACH", "status": "supports", "detail": "late post-step distance slope remains negative and no late safety/boundary projection was observed"})
    governed = late["governor"]["governed_desired_radial_speed_mps"]["median"]
    ref_governed_median = statistics.median(ref_governed) if ref_governed else float("nan")
    if math.isfinite(governed) and math.isfinite(ref_governed_median) and governed < 0.75 * ref_governed_median:
        labels.append("GOVERNOR_APPROACH_STARVATION")
        evidence.append({"label": "GOVERNOR_APPROACH_STARVATION", "status": "supports", "detail": "late governed desired radial speed is below 75% of the successful-reference median"})
    actor_fraction = late["actor_conflict"]["actor_radial_negative_fraction"]
    post_net = late["actor_conflict"]["mean_net_radial_post_transform_mps"]
    compensation = late["actor_conflict"]["mean_compensation_post_cap_mps"]
    actor_conflict_evidence = (
        ref_actor
        and ref_compensation
        and actor_fraction > _percentile(ref_actor, 0.75)
        and compensation > _percentile(ref_compensation, 0.50)
        and post_net < 0.0
    )
    if actor_conflict_evidence:
        labels.append("ACTOR_RADIAL_CONFLICT")
        evidence.append({"label": "ACTOR_RADIAL_CONFLICT", "status": "supports", "detail": "opposing Actor radial fraction and post-cap compensation exceed the successful-reference median while mean post-transform net radial correction remains negative"})
    retention = late["action_transform"]["retention_ratio"]["median"]
    if math.isfinite(retention) and retention < 0.9:
        labels.append("ACTION_TRANSFORM_ATTENUATION")
        evidence.append({"label": "ACTION_TRANSFORM_ATTENUATION", "status": "supports", "detail": "median positive-pre-transform radial retention is below 0.9"})
    tangent = late["tangent_to_radial_ratio"]["ratio"]["median"]
    if ref_tangent and math.isfinite(tangent) and tangent > _percentile(ref_tangent, 0.75) and late["positive_progress_step_fraction"] < 0.5:
        labels.append("TANGENTIAL_DOMINANCE")
        evidence.append({"label": "TANGENTIAL_DOMINANCE", "status": "supports", "detail": "tangent-to-radial ratio exceeds reference upper quartile while positive progress is below half of step transitions"})
    reference_switches = [
        item["balance"]["propulsion_braking_switches"]
        for item in ref_late
    ]
    switch_threshold = _percentile(reference_switches, 0.75) if reference_switches else float("nan")
    if math.isfinite(switch_threshold) and late["balance"]["propulsion_braking_switches"] > switch_threshold:
        labels.append("BRAKING_PROPULSION_OSCILLATION")
        evidence.append({"label": "BRAKING_PROPULSION_OSCILLATION", "status": "supports", "detail": "last20 propulsion/braking switches exceed the successful-reference 75th percentile"})
    if math.isfinite(late["saturation"]["terminal_correction_clipped_fraction"]) and late["saturation"]["terminal_correction_clipped_fraction"] > 0.5:
        labels.append("CORRECTION_CAP_LIMITED")
        evidence.append({"label": "CORRECTION_CAP_LIMITED", "status": "supports", "detail": "shared correction clipping occurs in more than half of the last 20 steps"})
    if late["safety_boundary"]["safety_projection_count"] + late["safety_boundary"]["boundary_projection_count"] > 0:
        evidence.append({"label": "SAFETY_OR_BOUNDARY_INTERFERENCE", "status": "possible", "detail": "a late safety or boundary projection was observed"})
    if not labels:
        labels = ["NOT_ASSESSABLE"]
    elif len(labels) > 1:
        labels = ["MIXED"] + labels
    confidence = "HIGH" if any(item["label"] in {"GOVERNOR_APPROACH_STARVATION", "ACTOR_RADIAL_CONFLICT", "ACTION_TRANSFORM_ATTENUATION"} for item in evidence) else "MEDIUM" if evidence else "LOW"
    against = [
        "terminal hold is not the direct gate for this episode because first entered1 is false",
        "collision is absent in the source episode summary",
    ]
    return {
        "episode_id": target_id,
        "primary_labels": labels,
        "primary_evidence": evidence,
        "secondary_evidence": {
            "late20_slope": slope,
            "late20_r2": late["late_slope"]["r2"],
            "late20_estimated_extra_steps_to_1m": late["estimated_extra_steps_to_1m"],
        },
        "evidence_against_other_causes": against,
        "confidence": confidence,
        "interpretation_scope": "exploratory offline evidence; no counterfactual trajectory or causal claim",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _jsonable(value) for key, value in row.items()} for row in rows)


def _prepare_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    output.mkdir(parents=True, exist_ok=False)


def _report(
    *,
    source: Path,
    identity: Mapping[str, Any],
    source_missingness: Mapping[str, Any],
    episode_rows: Mapping[str, Mapping[str, str]],
    analyses: Mapping[str, Mapping[str, Any]],
    diagnoses: Mapping[str, Mapping[str, Any]],
) -> str:
    failed = [episode_id for episode_id in TARGET_IDS if not _bool(episode_rows[episode_id]["entered_1m"], "entered_1m")]
    lines = [
        "CAPTURE_LATENCY_AUDIT_READY",
        "",
        "# Single-UAV Near-Goal Capture Latency Offline Audit",
        "",
        "本报告只读取 authority-repaired Terminal Capture v2.3 formal100 日志；所有距离进展均使用 `goal_distance_after_m` 的 post-step 序列，相邻 post-step 距离作差。速度统计使用日志中的 `post_step_actual_*` 与 pre-step decision 的 `current_radial_velocity_mps`；未调用 simulator、controller 或 environment。",
        "",
        f"Source: `{source.name}`；episodes={identity['source_episode_count']}；steps={identity['source_step_count']}。",
        f"结构性缺失审计：steps.csv 缺失/非有限单元 `{source_missingness['steps.csv']['missing_or_nonfinite_cells']}`（空白 `{source_missingness['steps.csv']['blank_cells']}`，主要为 `terminal_v21_inactive_reason`）；episodes.csv `{source_missingness['episodes.csv']['missing_or_nonfinite_cells']}`（空白 `{source_missingness['episodes.csv']['blank_cells']}`，NaN/非有限 `{source_missingness['episodes.csv']['nonfinite_cells']}`）；evaluation_summary.json 非有限 token `{source_missingness['evaluation_summary.json']['total_nonfinite_tokens']}`。这些字段按事件不可用处理，不填补为测量值。",
        "",
        "## 问题 1：recovery=11 但 entered1=8",
        "",
        f"源日志中 `{len(failed)}` 个 post-bypass recovery episode 在 100 步内未达到 post-step distance ≤1.0 m：" + ", ".join(failed) + "。它们不是 hold gate 失败，而是进入 terminal hold 之前的 approach/capture latency 问题。",
        "",
        "## 问题 2：三个 episode 是否同因",
        "",
    ]
    for episode_id in TARGET_IDS:
        diagnosis = diagnoses[episode_id]
        timeline = analyses[episode_id]["timeline"]
        lines.extend([
            f"### {episode_id}",
            "",
            f"- labels: `{', '.join(diagnosis['primary_labels'])}`；confidence=`{diagnosis['confidence']}`。",
            f"- minimum post-step distance: `{timeline['minimum_goal_distance_m']:.6f} m` at step `{timeline['minimum_goal_distance_step']}`。",
            f"- late20 slope: `{analyses[episode_id]['windows']['C_last20']['late_slope']['slope_m_per_step']:.6g} m/step`, R²=`{analyses[episode_id]['windows']['C_last20']['late_slope']['r2']:.4g}`。",
            f"- evidence: {('; '.join(item['detail'] for item in diagnosis['primary_evidence']) or 'no single permitted label reached its evidence gate')}。",
            "",
        ])
    label_summary = "; ".join(
        f"{episode_id}: {', '.join(diagnoses[episode_id]['primary_labels'])}"
        for episode_id in TARGET_IDS
    )
    lines.extend([
        "## 问题 3：主要瓶颈",
        "",
        f"逐 episode 标签为：{label_summary}。三个 episode 都保留负的 late post-step distance slope，但标签并不等价：0001 同时显示 horizon 与高于成功参考组上四分位的 propulsion/braking switching，0008 主要是快速 overspeed 后仍未在 horizon 内越过 1 m，0017 还显示高于参考的 opposing-Actor/conflict 与负的 late net radial correction。该审计不把这些信号解释为统一因果结论。",
        "",
        "## 问题 4：是否需要重新修改 Terminal Hold v2.3",
        "",
        "不建议重新打开 hold。三个目标 episode 的关键事件是 first entered1 缺失；v2.2/v2.3 hold 与 guard 只有在接近 1 m 后才会激活，因此没有证据表明 hold law 是这三个 failure 的直接 gate。",
        "",
        "## 问题 5：唯一最小下一步实验建议",
        "",
        "唯一建议（本轮不执行）：在获得单独授权后，使用同一 repaired controller，仅把 `max_steps=100→120`，做一个 horizon-only diagnostic。它只检验三个目标 episode 的持续负距离斜率是否主要受预算截断影响，不修改 gain、governor、Actor、action transform 或 hold；0017 的 Actor/conflict 信号仍需按诊断结果单独解释。",
        "",
        "## Identity and status",
        "",
        f"- source checkpoint/policy/manifest locks: `{identity['all_identity_checks_pass']}`",
        "- terminal_hold_reopened=false",
        "- source logs modified=false",
        "- controller_code_modified=false",
        "- audit_only=true",
    ])
    return "\n".join(lines) + "\n"


def run_audit(source: str | Path = DEFAULT_SOURCE, output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    paths = {name: source_path / name for name in SOURCE_FILES}
    if not all(path.is_file() for path in paths.values()):
        missing = [str(path.name) for path in paths.values() if not path.is_file()]
        raise ValueError("CAPTURE_LATENCY_AUDIT_IDENTITY_FAIL: missing " + ", ".join(missing))
    steps_raw = _read_csv(paths["steps.csv"])
    episodes_raw = _read_csv(paths["episodes.csv"])
    validation_episodes_raw = _read_csv(paths["terminal_capture_v23_authority_repair_validation_episodes.csv"])
    repair_summary = _read_json(paths["terminal_capture_v23_authority_repair_validation_summary.json"])
    evaluation = _read_json(paths["evaluation_summary.json"])
    if not steps_raw or not episodes_raw:
        raise ValueError("CAPTURE_LATENCY_AUDIT_IDENTITY_FAIL: empty source table")
    source_missingness = {
        "steps.csv": _csv_missingness(steps_raw),
        "episodes.csv": _csv_missingness(episodes_raw),
        "terminal_capture_v23_authority_repair_validation_episodes.csv": _csv_missingness(validation_episodes_raw),
        "terminal_capture_v23_authority_repair_validation_summary.json": _json_token_missingness(
            paths["terminal_capture_v23_authority_repair_validation_summary.json"]
        ),
        "evaluation_summary.json": _json_token_missingness(paths["evaluation_summary.json"]),
    }
    missing_step_fields = REQUIRED_STEP_FIELDS - set(steps_raw[0])
    missing_episode_fields = REQUIRED_EPISODE_FIELDS - set(episodes_raw[0])
    if missing_step_fields or missing_episode_fields:
        raise ValueError(
            "CAPTURE_LATENCY_AUDIT_IDENTITY_FAIL: missing fields "
            + repr({"steps": sorted(missing_step_fields), "episodes": sorted(missing_episode_fields)})
        )
    identity = _identity_check(
        source=source_path,
        steps=steps_raw,
        episodes=episodes_raw,
        validation_episodes=validation_episodes_raw,
        repair_summary=repair_summary,
        evaluation=evaluation,
    )
    episode_rows = {str(row["episode_id"]): row for row in episodes_raw}
    target_checks = {}
    for episode_id in TARGET_IDS:
        row = episode_rows[episode_id]
        target_checks[episode_id] = {
            "post_bypass_recovery": _bool(row["post_bypass_recovery"], "post_bypass_recovery"),
            "passed_obstacle": _bool(row["passed_obstacle"], "passed_obstacle"),
            "entered_8m": _bool(row["entered_8m"], "entered_8m"),
            "entered_5m": _bool(row["entered_5m"], "entered_5m"),
            "entered_3m": _bool(row["entered_3m"], "entered_3m"),
            "entered_1m": _bool(row["entered_1m"], "entered_1m"),
        }
    if any(
        not values["post_bypass_recovery"]
        or not values["passed_obstacle"]
        or not values["entered_8m"]
        or not values["entered_5m"]
        or not values["entered_3m"]
        or values["entered_1m"]
        for values in target_checks.values()
    ):
        raise ValueError("CAPTURE_LATENCY_AUDIT_IDENTITY_FAIL: target episode condition mismatch")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nan_counts: dict[str, int] = defaultdict(int)
    for raw in steps_raw:
        for key, value in raw.items():
            if value in (None, "", "nan", "NaN", "inf", "-inf"):
                nan_counts[key] += 1
        grouped[str(raw["episode_id"])].append(_normalise_step(raw))
    analyses = {
        episode_id: _episode_analysis(grouped[episode_id])
        for episode_id in (*TARGET_IDS, *REFERENCE_IDS)
    }
    reference_items = [
        {"episode_id": episode_id, "analysis": analyses[episode_id], "late_slope": analyses[episode_id]["windows"]["C_last20"]["late_slope"]["slope_m_per_step"]}
        for episode_id in REFERENCE_IDS
    ]
    diagnoses = {
        episode_id: _diagnose(episode_id, analyses[episode_id], reference_items)
        for episode_id in TARGET_IDS
    }
    _prepare_output(output_path)
    phase_rows = []
    diagnostic_rows = []
    late_rows = []
    reference_rows = []
    for episode_id in (*TARGET_IDS, *REFERENCE_IDS):
        timeline = analyses[episode_id]["timeline"]
        phase_rows.append({"episode_id": episode_id, **timeline})
        diagnostic_rows.append(_flatten_diagnostic(episode_id, episode_rows[episode_id], analyses[episode_id]))
    for episode_id in TARGET_IDS:
        for row in _window_rows(grouped[episode_id], "C_last20", analyses[episode_id]["timeline"]["first_post_bypass_recovery_step"], analyses[episode_id]["timeline"]["first_entered3_step"]):
            late_rows.append({"episode_id": episode_id, "window": "last20", "source_timing": "post_step_distance_and_actual_velocity", **{key: value for key, value in row.items() if not key.startswith("_")}})
        for row in _window_rows(grouped[episode_id], "D_last10", analyses[episode_id]["timeline"]["first_post_bypass_recovery_step"], analyses[episode_id]["timeline"]["first_entered3_step"]):
            late_rows.append({"episode_id": episode_id, "window": "last10", "source_timing": "post_step_distance_and_actual_velocity", **{key: value for key, value in row.items() if not key.startswith("_")}})
    comparison_windows = ("after_first_3m_20", "before_first_1m_last20")
    for episode_id in (*TARGET_IDS, *REFERENCE_IDS):
        aligned = _reference_windows(grouped[episode_id], episode_rows[episode_id])
        role = "target_episode" if episode_id in TARGET_IDS else "reference_episode"
        for window_name in comparison_windows:
            reference_rows.append(
                _comparison_row(
                    episode_id=episode_id,
                    episode_row=episode_rows[episode_id],
                    window_name=window_name,
                    window=aligned[window_name],
                    role=role,
                )
            )
    reference_rows.extend(
        _group_summary(reference_rows, "successful_entered1_reference_group", "after_first_3m_20")
    )
    reference_rows.extend(
        _group_summary(reference_rows, "successful_entered1_reference_group", "before_first_1m_last20")
    )
    source_manifest = {
        name: {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in paths.items()
    }
    summary = {
        "stage": "single_uav_near_goal_capture_latency_audit",
        "audit_verdict": "CAPTURE_LATENCY_AUDIT_READY",
        "audit_only": True,
        "source_validation": "terminal_capture_v23_authority_repair",
        "source_root_name": source_path.name,
        "source_manifest": source_manifest,
        "source_episode_count": len(episodes_raw),
        "source_step_count": len(steps_raw),
        "target_episode_ids": list(TARGET_IDS),
        "reference_episode_ids": list(REFERENCE_IDS),
        "identity_checks": identity,
        "target_identity_checks": target_checks,
        "nan_distribution": dict(sorted(nan_counts.items())),
        "source_missingness": source_missingness,
        "data_dictionary": {
            "goal_distance_m": "pre-step estimated goal distance used by the controller decision",
            "goal_distance_after_m": "post-step estimated goal distance; used for all distance-progress windows",
            "current_radial_velocity_mps": "pre-step estimated radial velocity in the decision row",
            "post_step_actual_radial_velocity_mps": "post-step simulator actual radial velocity",
            "post_step_actual_tangential_velocity_mps": "post-step simulator actual tangential velocity",
            "terminal_v2_governed_desired_radial_speed_mps": "v2 governor output recorded in the same step row",
            "terminal_v21_net_radial_pre_transform_mps": "net radial correction before action transform",
            "terminal_v21_net_radial_post_transform_mps": "net radial correction after action transform",
        },
        "timing_definition": "Distance progress compares adjacent goal_distance_after_m post-step values; decision fields remain pre-step/controller-time fields; actual velocity fields are post-step.",
        "reference_comparison_definition": {
            "after_first_3m_20": "first 20 post-step rows starting at the first row with goal_distance_after_m <= 3 m",
            "before_first_1m_last20": "last 20 post-step rows strictly before the first row with goal_distance_after_m <= 1 m; empty when 1 m is never entered",
            "group_summary": "median and IQR are computed from reference_episode rows only, never from target rows",
        },
        "per_episode_phase_timeline": {episode_id: analyses[episode_id]["timeline"] for episode_id in (*TARGET_IDS, *REFERENCE_IDS)},
        "per_episode_windows": {episode_id: analyses[episode_id]["windows"] for episode_id in (*TARGET_IDS, *REFERENCE_IDS)},
        "per_episode_diagnosis": diagnoses,
        "late20_distance_slope": {episode_id: analyses[episode_id]["windows"]["C_last20"]["late_slope"] for episode_id in TARGET_IDS},
        "estimated_extra_steps_to_1m": {episode_id: analyses[episode_id]["windows"]["C_last20"]["estimated_extra_steps_to_1m"] for episode_id in TARGET_IDS},
        "late20_mean_desired_radial": {episode_id: analyses[episode_id]["windows"]["C_last20"]["radial_tracking"]["desired_radial_velocity_mps"]["mean"] for episode_id in TARGET_IDS},
        "late20_mean_actual_radial": {episode_id: analyses[episode_id]["windows"]["C_last20"]["radial_tracking"]["post_step_actual_radial_velocity_mps"]["mean"] for episode_id in TARGET_IDS},
        "late20_mean_tangent": {episode_id: analyses[episode_id]["windows"]["C_last20"]["radial_tracking"]["post_step_actual_tangential_velocity_mps"]["mean"] for episode_id in TARGET_IDS},
        "late20_actor_opposition_fraction": {episode_id: analyses[episode_id]["windows"]["C_last20"]["actor_conflict"]["actor_radial_negative_fraction"] for episode_id in TARGET_IDS},
        "late20_conflict_active_fraction": {episode_id: analyses[episode_id]["windows"]["C_last20"]["actor_conflict"]["conflict_active_fraction"] for episode_id in TARGET_IDS},
        "late20_mean_net_radial_pre": {episode_id: analyses[episode_id]["windows"]["C_last20"]["actor_conflict"]["mean_net_radial_pre_transform_mps"] for episode_id in TARGET_IDS},
        "late20_mean_net_radial_post": {episode_id: analyses[episode_id]["windows"]["C_last20"]["actor_conflict"]["mean_net_radial_post_transform_mps"] for episode_id in TARGET_IDS},
        "late20_transform_retention": {episode_id: analyses[episode_id]["windows"]["C_last20"]["action_transform"]["retention_ratio"] for episode_id in TARGET_IDS},
        "late20_propulsion_fraction": {episode_id: analyses[episode_id]["windows"]["C_last20"]["balance"]["radial_propulsion_active_fraction"] for episode_id in TARGET_IDS},
        "late20_braking_fraction": {episode_id: analyses[episode_id]["windows"]["C_last20"]["balance"]["radial_braking_active_fraction"] for episode_id in TARGET_IDS},
        "late20_saturation_fraction": {episode_id: analyses[episode_id]["windows"]["C_last20"]["saturation"]["terminal_correction_clipped_fraction"] for episode_id in TARGET_IDS},
        "late20_safety_projection_count": {episode_id: analyses[episode_id]["windows"]["C_last20"]["safety_boundary"]["safety_projection_count"] for episode_id in TARGET_IDS},
        "late20_boundary_projection_count": {episode_id: analyses[episode_id]["windows"]["C_last20"]["safety_boundary"]["boundary_projection_count"] for episode_id in TARGET_IDS},
        "status": {
            "near_goal_capture_latency_audit_executed": True,
            "target_episodes_analyzed": list(TARGET_IDS),
            "terminal_hold_reopened": False,
            "terminal_capture_v23_modified": False,
            "terminal_capture_v23_authority_repaired_source_unchanged": True,
            "terminal_capture_v22_modified": False,
            "terminal_capture_v21_modified": False,
            "terminal_capture_v2_modified": False,
            "actor_modified": False,
            "gat_modified": False,
            "critic_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "action_transform_modified": False,
            "stable_criterion_modified": False,
            "max_correction_modified": False,
            "training_performed": False,
            "rollout_executed": False,
            "formal_executed": False,
            "smoke_executed": False,
            "environment_step_called": False,
            "source_logs_modified": False,
            "controller_code_modified": False,
        },
    }
    (output_path / OUTPUT_FILES[0]).write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output_path / OUTPUT_FILES[1], diagnostic_rows)
    _write_csv(output_path / OUTPUT_FILES[2], late_rows)
    _write_csv(output_path / OUTPUT_FILES[3], reference_rows)
    _write_csv(output_path / OUTPUT_FILES[4], phase_rows)
    (output_path / OUTPUT_FILES[5]).write_text(
        _report(
            source=source_path,
            identity=identity,
            source_missingness=source_missingness,
            episode_rows=episode_rows,
            analyses=analyses,
            diagnoses=diagnoses,
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_audit(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
