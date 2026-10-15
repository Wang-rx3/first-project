"""Read-only route/reward audit for the single-UAV PPO-A transfer run.

This module intentionally uses only CSV/JSON parsing and scalar summaries.  It
does not import the simulator, construct an environment, call ``step``, fit a
normalizer, load checkpoints, or train a policy.  Raw rows are never emitted;
derived CSV/JSON summaries are written below the run's reward_audit folder.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_NAME = "single_uav_ppo_a_norm_transfer_v2_seed2026_20261011"
OUTPUT_RUN_NAME = "single_uav_ppo_a_route_feasibility_v3_seed2026_20261011"
ARMS = ("C", "N", "G", "R")
PHASE_ORDER = (
    "online",
    "initial_train",
    "final_train",
    "initial_validation",
    "final_validation",
    "initial_challenge",
    "final_challenge",
)
BANDS = (
    ("d_le_1", "actual_distance_m", None, 1.0),
    ("d_le_2", "actual_distance_m", None, 2.0),
    ("d_le_4", "actual_distance_m", None, 4.0),
    ("d_le_5", "actual_distance_m", None, 5.0),
    ("d_gt_5", "actual_distance_m", 5.0, None),
)
REWARD_FIELDS = (
    "single_uav_goal_progress_reward",
    "single_uav_goal_direction_reward",
    "single_uav_braking_action_reward",
    "single_uav_braking_progress_reward",
    "single_uav_terminal_braking_reward",
    "single_uav_terminal_radial_progress_reward",
    "single_uav_terminal_target_velocity_vector_reward",
    "single_uav_target_velocity_action_reward",
    "single_uav_near_goal_speed_cost",
    "single_uav_speed_deficit_cost",
    "single_uav_goal_bonus",
    "single_uav_goal_hold_reward",
    "single_uav_learning_core",
    "single_uav_learning_reward_preclip",
    "single_uav_learning_reward",
)


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value or str(value).strip() in {"{}", "null", "None"}:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _vector_norm(value: Any) -> float:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return math.nan
    if not isinstance(value, (list, tuple)):
        return math.nan
    numbers = [_number(item) for item in value]
    if not numbers or any(not math.isfinite(item) for item in numbers):
        return math.nan
    return math.sqrt(sum(item * item for item in numbers))


def _mean(values: Iterable[float]) -> float | None:
    usable = [x for x in values if math.isfinite(x)]
    return sum(usable) / len(usable) if usable else None


def _median(values: Iterable[float]) -> float | None:
    usable = sorted(x for x in values if math.isfinite(x))
    if not usable:
        return None
    middle = len(usable) // 2
    if len(usable) % 2:
        return usable[middle]
    return (usable[middle - 1] + usable[middle]) / 2.0


def _quantile(values: Iterable[float], fraction: float) -> float | None:
    usable = sorted(x for x in values if math.isfinite(x))
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    position = (len(usable) - 1) * fraction
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return usable[left]
    weight = position - left
    return usable[left] * (1.0 - weight) + usable[right] * weight


def _metric(values: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values)
    return {
        "n": sum(math.isfinite(value) for value in values),
        "mean": _mean(values),
        "median": _median(values),
        "p90": _quantile(values, 0.90),
    }


def _load_rows(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        path = run_root / arm / "steps.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        prepared: list[dict[str, Any]] = []
        for row in rows:
            action = _json(row.get("action_execution"))
            reward = _json(row.get("reward_components"))
            actual_speed = _number(row.get("speed_mps_actual"))
            pre_speed = _number(action.get("pre_speed_mps"))
            post_speed = _number(action.get("post_speed_mps"))
            applied_longitudinal = _number(
                action.get("applied_longitudinal_delta_v_mps")
            )
            target_speed = _number(reward.get("single_uav_target_speed_mps"))
            pre_excess = _number(reward.get("single_uav_pre_speed_excess_mps"))
            if not math.isfinite(pre_excess) and math.isfinite(pre_speed) and math.isfinite(target_speed):
                pre_excess = max(0.0, pre_speed - target_speed)
            post_excess = _number(reward.get("single_uav_post_speed_excess_mps"))
            if not math.isfinite(post_excess) and math.isfinite(post_speed) and math.isfinite(target_speed):
                post_excess = max(0.0, post_speed - target_speed)
            prepared.append(
                {
                    "raw": row,
                    "arm": arm,
                    "phase": str(row.get("phase", "")),
                    "split": str(row.get("split", "")),
                    "episode_id": str(row.get("episode_id", "")),
                    "episode_step": int(_number(row.get("episode_step"), -1)),
                    "actual_distance_m": _number(row.get("goal_distance_m_actual")),
                    "estimated_distance_m": _number(row.get("goal_distance_m_estimated")),
                    "actual_speed_mps": actual_speed,
                    "estimated_speed_mps": _vector_norm(row.get("estimated_velocity_mps")),
                    "pre_speed_mps": pre_speed,
                    "post_speed_mps": post_speed,
                    "target_speed_mps": target_speed,
                    "pre_excess_mps": pre_excess,
                    "post_excess_mps": post_excess,
                    "applied_longitudinal_delta_v_mps": applied_longitudinal,
                    "speed_change_mps": _number(action.get("speed_change_mps")),
                    "speed_reduction_mps": _number(action.get("speed_reduction_mps")),
                    "braking_action": _flag(action.get("braking_action")),
                    "brake_effective": _flag(action.get("brake_effective")),
                    "actual_acceleration_mps2": _number(row.get("acceleration_mps2_actual")),
                    "nominal_acceleration_mps2": _number(row.get("nominal_acceleration_mps2")),
                    "goal_condition": _flag(reward.get("single_uav_goal_condition")),
                    "goal_stable": _flag(reward.get("single_uav_goal_stable")),
                    "bonus_triggered": _flag(reward.get("single_uav_goal_bonus_triggered")),
                    "stable_steps": int(_number(reward.get("single_uav_stable_steps"), 0)),
                    "strict_success": _flag(row.get("strict_success")) or _flag(reward.get("strict_success")),
                    "rewards": reward,
                    "action": action,
                }
            )
        rows_by_arm[arm] = prepared
    return rows_by_arm


def _in_band(row: dict[str, Any], upper: float | None, lower: float | None) -> bool:
    distance = row["actual_distance_m"]
    if not math.isfinite(distance):
        return False
    return (upper is None or distance <= upper) and (lower is None or distance > lower)


def _rows_for(rows: list[dict[str, Any]], phase: str, band_name: str) -> list[dict[str, Any]]:
    band = next(spec for spec in BANDS if spec[0] == band_name)
    return [
        row
        for row in rows
        if row["phase"] == phase and _in_band(row, band[3], band[2])
    ]


def _share(rows: list[dict[str, Any]], predicate: Any) -> dict[str, float | int]:
    numerator = sum(bool(predicate(row)) for row in rows)
    return {"n": numerator, "denominator": len(rows), "fraction": numerator / len(rows) if rows else None}


def _action_summary(rows: list[dict[str, Any]], action_context: str) -> dict[str, Any]:
    if action_context == "near_goal_d_le_4":
        selected = [row for row in rows if row["actual_distance_m"] <= 4.0]
    elif action_context == "outside_near_goal_d_gt_5":
        selected = [row for row in rows if row["actual_distance_m"] > 5.0]
    else:
        selected = rows
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        overspeed = row["pre_excess_mps"] > 1e-9
        signed = row["applied_longitudinal_delta_v_mps"]
        sign = "accel" if signed > 1e-9 else "brake" if signed < -1e-9 else "neutral"
        categories[f"{'overspeed' if overspeed else 'at_or_below_target'}_{sign}"].append(row)
    result: dict[str, Any] = {"rows": len(selected), "categories": {}}
    for name, category_rows in sorted(categories.items()):
        result["categories"][name] = {
            "n": len(category_rows),
            "actual_speed_mps": _metric(row["actual_speed_mps"] for row in category_rows),
            "target_speed_mps": _metric(row["target_speed_mps"] for row in category_rows),
            "applied_longitudinal_delta_v_mps": _metric(
                row["applied_longitudinal_delta_v_mps"] for row in category_rows
            ),
            "speed_change_mps": _metric(row["speed_change_mps"] for row in category_rows),
            "goal_progress_reward": _metric(
                _number(row["rewards"].get("single_uav_goal_progress_reward"))
                for row in category_rows
            ),
            "goal_direction_reward": _metric(
                _number(row["rewards"].get("single_uav_goal_direction_reward"))
                for row in category_rows
            ),
            "braking_action_reward": _metric(
                _number(row["rewards"].get("single_uav_braking_action_reward"))
                for row in category_rows
            ),
            "learning_core": _metric(
                _number(row["rewards"].get("single_uav_learning_core"))
                for row in category_rows
            ),
        }
    return result


def _longest_qualifying_run(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return longest actual-goal hold run, qualifying rows, and episodes."""
    by_episode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[(row["phase"], row["episode_id"])].append(row)
    longest = 0
    qualifying = 0
    episodes_with_qualifying = 0
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda item: item["episode_step"])
        run = 0
        episode_best = 0
        for row in episode_rows:
            qualifies = row["actual_distance_m"] <= 1.0 and row["actual_speed_mps"] <= 0.5
            if qualifies:
                qualifying += 1
                run += 1
                episode_best = max(episode_best, run)
            else:
                run = 0
        longest = max(longest, episode_best)
        if episode_best:
            episodes_with_qualifying += 1
    return longest, qualifying, episodes_with_qualifying


def _band_summary(rows_by_arm: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ARMS:
        rows = rows_by_arm[arm]
        phases = sorted({row["phase"] for row in rows}, key=lambda item: PHASE_ORDER.index(item) if item in PHASE_ORDER else 99)
        for phase in phases:
            for band_name, _, _, _ in BANDS:
                selected = _rows_for(rows, phase, band_name)
                if not selected:
                    continue
                output.append(
                    {
                        "arm": arm,
                        "phase": phase,
                        "band": band_name,
                        "rows": len(selected),
                        "episodes": len({row["episode_id"] for row in selected}),
                        "actual_speed_mps": _metric(row["actual_speed_mps"] for row in selected),
                        "estimated_speed_mps": _metric(row["estimated_speed_mps"] for row in selected),
                        "pre_speed_mps": _metric(row["pre_speed_mps"] for row in selected),
                        "post_speed_mps": _metric(row["post_speed_mps"] for row in selected),
                        "target_speed_mps": _metric(row["target_speed_mps"] for row in selected),
                        "actual_distance_m": _metric(row["actual_distance_m"] for row in selected),
                        "estimated_distance_m": _metric(row["estimated_distance_m"] for row in selected),
                        "applied_longitudinal_delta_v_mps": _metric(
                            row["applied_longitudinal_delta_v_mps"] for row in selected
                        ),
                        "speed_change_mps": _metric(row["speed_change_mps"] for row in selected),
                        "speed_reduction_mps": _metric(row["speed_reduction_mps"] for row in selected),
                        "actual_acceleration_mps2": _metric(row["actual_acceleration_mps2"] for row in selected),
                        "accel_action": _share(
                            selected,
                            lambda row: row["applied_longitudinal_delta_v_mps"] > 1e-9,
                        ),
                        "speed_increase": _share(
                            selected,
                            lambda row: row["speed_change_mps"] > 1e-9,
                        ),
                        "braking_action": _share(
                            selected,
                            lambda row: row["applied_longitudinal_delta_v_mps"] < -1e-9,
                        ),
                        "effective_braking": _share(
                            selected,
                            lambda row: row["speed_reduction_mps"] > 1e-9 or row["brake_effective"],
                        ),
                        "goal_condition": _share(selected, lambda row: row["goal_condition"]),
                        "goal_stable": _share(selected, lambda row: row["goal_stable"]),
                        "bonus_triggered": _share(selected, lambda row: row["bonus_triggered"]),
                        "rewards": {
                            field: _metric(_number(row["rewards"].get(field)) for row in selected)
                            for field in REWARD_FIELDS
                        },
                        "clipped_learning_reward": _share(
                            selected,
                            lambda row: abs(_number(row["rewards"].get("single_uav_learning_reward"))) >= 1.0 - 1e-9,
                        ),
                    }
                )
    return output


def _phase_summary(rows_by_arm: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ARMS:
        rows = rows_by_arm[arm]
        for phase in PHASE_ORDER:
            selected = [row for row in rows if row["phase"] == phase]
            if not selected:
                continue
            longest, qualifying, episodes_with_qualifying = _longest_qualifying_run(selected)
            near = [row for row in selected if row["actual_distance_m"] <= 4.0]
            output.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "rows": len(selected),
                    "episodes": len({row["episode_id"] for row in selected}),
                    "distance_min_m": min((row["actual_distance_m"] for row in selected if math.isfinite(row["actual_distance_m"])), default=None),
                    "distance_median_m": _median(row["actual_distance_m"] for row in selected),
                    "speed_mps_actual": _metric(row["actual_speed_mps"] for row in selected),
                    "speed_mps_estimated": _metric(row["estimated_speed_mps"] for row in selected),
                    "near_goal_d_le_1_rows": sum(row["actual_distance_m"] <= 1.0 for row in selected),
                    "near_goal_d_le_2_rows": sum(row["actual_distance_m"] <= 2.0 for row in selected),
                    "near_goal_d_le_4_rows": len(near),
                    "near_goal_d_le_5_rows": sum(row["actual_distance_m"] <= 5.0 for row in selected),
                    "near_goal_d_le_4_episodes": len({row["episode_id"] for row in near}),
                    "near_goal_d_le_4_accel_rows": sum(row["applied_longitudinal_delta_v_mps"] > 1e-9 for row in near),
                    "near_goal_d_le_4_speed_increase_rows": sum(row["speed_change_mps"] > 1e-9 for row in near),
                    "near_goal_d_le_4_braking_rows": sum(row["applied_longitudinal_delta_v_mps"] < -1e-9 for row in near),
                    "near_goal_d_le_4_effective_braking_rows": sum(
                        row["speed_reduction_mps"] > 1e-9 or row["brake_effective"] for row in near
                    ),
                    "near_goal_d_le_4_overspeed_rows": sum(row["pre_excess_mps"] > 1e-9 for row in near),
                    "near_goal_d_le_4_overspeed_accel_rows": sum(
                        row["pre_excess_mps"] > 1e-9 and row["applied_longitudinal_delta_v_mps"] > 1e-9
                        for row in near
                    ),
                    "near_goal_d_le_4_overspeed_braking_rows": sum(
                        row["pre_excess_mps"] > 1e-9 and row["applied_longitudinal_delta_v_mps"] < -1e-9
                        for row in near
                    ),
                    "actual_hold_qualifying_rows": qualifying,
                    "episodes_with_any_actual_qualifying_row": episodes_with_qualifying,
                    "longest_actual_qualifying_run_steps": longest,
                    "logged_goal_condition_rows": sum(row["goal_condition"] for row in selected),
                    "logged_goal_stable_rows": sum(row["goal_stable"] for row in selected),
                    "logged_bonus_triggered_rows": sum(row["bonus_triggered"] for row in selected),
                    "logged_max_stable_steps": max((row["stable_steps"] for row in selected), default=0),
                    "logged_strict_success_rows": sum(row["strict_success"] for row in selected),
                    "clipped_learning_reward_rows": sum(
                        abs(_number(row["rewards"].get("single_uav_learning_reward")))
                        >= 1.0 - 1e-9
                        for row in selected
                    ),
                    "clipped_learning_reward_fraction": (
                        sum(
                            abs(_number(row["rewards"].get("single_uav_learning_reward")))
                            >= 1.0 - 1e-9
                            for row in selected
                        )
                        / len(selected)
                        if selected
                        else None
                    ),
                    "outside_near_goal_accel_share": _share(
                        [row for row in selected if row["actual_distance_m"] > 5.0],
                        lambda row: row["applied_longitudinal_delta_v_mps"] > 1e-9,
                    ),
                    "outside_near_goal_braking_share": _share(
                        [row for row in selected if row["actual_distance_m"] > 5.0],
                        lambda row: row["applied_longitudinal_delta_v_mps"] < -1e-9,
                    ),
                    "reward_activity": {
                        field: {
                            "nonzero_rows": sum(abs(_number(row["rewards"].get(field))) > 1e-9 for row in selected),
                            "mean": _mean(_number(row["rewards"].get(field)) for row in selected),
                        }
                        for field in REWARD_FIELDS
                    },
                }
            )
    return output


def _episode_summary(rows_by_arm: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in ARMS:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_arm[arm]:
            groups[(row["phase"], row["episode_id"])].append(row)
        for (phase, episode_id), rows in sorted(groups.items()):
            rows.sort(key=lambda item: item["episode_step"])
            qualifying = [row for row in rows if row["actual_distance_m"] <= 1.0 and row["actual_speed_mps"] <= 0.5]
            near = [row for row in rows if row["actual_distance_m"] <= 4.0]
            run = 0
            longest = 0
            for row in rows:
                if row["actual_distance_m"] <= 1.0 and row["actual_speed_mps"] <= 0.5:
                    run += 1
                    longest = max(longest, run)
                else:
                    run = 0
            output.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "episode_id": episode_id,
                    "rows": len(rows),
                    "min_actual_distance_m": min((row["actual_distance_m"] for row in rows if math.isfinite(row["actual_distance_m"])), default=None),
                    "final_actual_distance_m": rows[-1]["actual_distance_m"],
                    "final_actual_speed_mps": rows[-1]["actual_speed_mps"],
                    "near_goal_d_le_1_rows": sum(row["actual_distance_m"] <= 1.0 for row in rows),
                    "near_goal_d_le_2_rows": sum(row["actual_distance_m"] <= 2.0 for row in rows),
                    "near_goal_d_le_4_rows": len(near),
                    "near_goal_d_le_4_braking_rows": sum(row["applied_longitudinal_delta_v_mps"] < -1e-9 for row in near),
                    "near_goal_d_le_4_effective_braking_rows": sum(
                        row["speed_reduction_mps"] > 1e-9 or row["brake_effective"] for row in near
                    ),
                    "near_goal_d_le_4_accel_rows": sum(row["applied_longitudinal_delta_v_mps"] > 1e-9 for row in near),
                    "actual_qualifying_rows": len(qualifying),
                    "longest_actual_qualifying_run_steps": longest,
                    "logged_max_stable_steps": max((row["stable_steps"] for row in rows), default=0),
                    "logged_bonus_triggered": any(row["bonus_triggered"] for row in rows),
                    "logged_strict_success": any(row["strict_success"] for row in rows),
                }
            )
    return output


def _normalization_summary(run_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        path = run_root.parent.parent.parent / "data" / "synthetic" / RUN_NAME / arm / "effective_config.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            output[arm] = {"readable": False}
            continue
        normalizer = payload.get("normalizer", {})
        output[arm] = {
            "readable": True,
            "source_sha256": normalizer.get("source_sha256"),
            "artifact_basename": Path(str(normalizer.get("artifact_path", ""))).name,
            "fit_split": normalizer.get("fit_split"),
            "statistics_frozen": normalizer.get("statistics_frozen"),
            "refit_performed": normalizer.get("refit_performed"),
            "feature_index_map": normalizer.get("feature_index_map"),
        }
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def run_audit(project_root: Path) -> dict[str, Any]:
    log_root = project_root / "logs" / "preliminary" / RUN_NAME
    output_root = project_root / "logs" / "preliminary" / OUTPUT_RUN_NAME / "reward_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_arm = _load_rows(log_root)
    band_summary = _band_summary(rows_by_arm)
    phase_summary = _phase_summary(rows_by_arm)
    episode_summary = _episode_summary(rows_by_arm)
    action_summary: dict[str, Any] = {}
    for arm in ARMS:
        action_summary[arm] = {}
        for phase in PHASE_ORDER:
            selected = [row for row in rows_by_arm[arm] if row["phase"] == phase]
            if selected:
                action_summary[arm][phase] = {
                    "near_goal_d_le_4": _action_summary(selected, "near_goal_d_le_4"),
                    "outside_near_goal_d_gt_5": _action_summary(selected, "outside_near_goal_d_gt_5"),
                }
    source_contract = {
        "environment_file": "src/simulation/environment.py",
        "reward_lines": [3434, 3509, 3522, 3530, 3541, 3570, 3602, 3638, 3667, 3680, 3724, 3766, 3784, 3841, 4066, 4085, 4097],
        "adapter_file": "src/experiments/single_uav_ppo_a_environment.py",
        "adapter_lines": [43, 49, 52, 53, 55, 70, 73],
        "baseline_file": "src/experiments/single_uav_ppo_baseline.py",
        "baseline_lines": [413, 416, 507, 516, 536, 570, 581, 593, 614],
        "constants_from_runtime_contract": {
            "learning_mode": "loc_goal_v2",
            "speed_control_mode": "braking_distance",
            "goal_threshold_m": 1.0,
            "goal_speed_threshold_mps": 0.5,
            "near_goal_radius_m": 5.0,
            "far_target_speed_mps": 6.0,
            "braking_acceleration_mps2": 2.0,
            "braking_margin_m": 2.0,
            "stable_hold_steps": 5,
            "single_uav_goal_bonus_magnitude": 5.0,
            "goal_hold_reward": 0.0,
            "learning_reward_clip_abs": 1.0,
            "dt_s": 0.2,
            "max_delta_v_mps": 0.6,
        },
        "interpretation": {
            "outside_near_goal": "the legacy stopping-distance envelope applies at all distances: the target grows from 0.5 m/s at distance<=3 m to 6 m/s only at about distance>=11.9375 m; goal progress and direction credit remain active, while braking need is zero when pre-action speed is at or below this target",
            "inside_near_goal": "the target speed falls with the stopping-distance envelope; overspeed braking receives action credit and continued positive longitudinal acceleration is penalized by the braking-action term",
            "goal_arrival": "the strategy environment suppresses the legacy one-step goal bonus and hold reward; the truth-state adapter requires distance at most 1 m and speed at most 0.5 m/s for five consecutive post-states before strict success and its one-shot bonus",
            "terminal_channels": "terminal braking, terminal acceleration penalty, radial progress, target-velocity-vector, and target-velocity-action coefficients are zero in the baseline builder unless a run explicitly overrides them",
        },
    }
    selected_phase_names = {"final_train", "final_validation", "final_challenge"}
    key_route_rows = [
        {
            "arm": row["arm"],
            "phase": row["phase"],
            "rows": row["rows"],
            "episodes": row["episodes"],
            "min_actual_distance_m": row["distance_min_m"],
            "d_le_1_rows": row["near_goal_d_le_1_rows"],
            "d_le_2_rows": row["near_goal_d_le_2_rows"],
            "d_le_4_rows": row["near_goal_d_le_4_rows"],
            "d_le_5_rows": row["near_goal_d_le_5_rows"],
            "d_le_4_accel_rows": row["near_goal_d_le_4_accel_rows"],
            "d_le_4_braking_rows": row["near_goal_d_le_4_braking_rows"],
            "d_le_4_effective_braking_rows": row["near_goal_d_le_4_effective_braking_rows"],
            "d_le_4_overspeed_accel_rows": row["near_goal_d_le_4_overspeed_accel_rows"],
            "d_le_4_overspeed_braking_rows": row["near_goal_d_le_4_overspeed_braking_rows"],
            "longest_actual_qualifying_run_steps": row["longest_actual_qualifying_run_steps"],
            "logged_goal_stable_rows": row["logged_goal_stable_rows"],
            "logged_bonus_triggered_rows": row["logged_bonus_triggered_rows"],
            "logged_strict_success_rows": row["logged_strict_success_rows"],
            "clipped_learning_reward_rows": row["clipped_learning_reward_rows"],
            "clipped_learning_reward_fraction": row["clipped_learning_reward_fraction"],
        }
        for row in phase_summary
        if row["phase"] in selected_phase_names
    ]
    key_findings = {
        "final_route_rows": key_route_rows,
        "initial_r_validation": next(
            (
                {
                    "rows": row["rows"],
                    "episodes": row["episodes"],
                    "d_le_1_rows": row["near_goal_d_le_1_rows"],
                    "d_le_4_rows": row["near_goal_d_le_4_rows"],
                    "d_le_4_accel_rows": row["near_goal_d_le_4_accel_rows"],
                    "d_le_4_braking_rows": row["near_goal_d_le_4_braking_rows"],
                    "actual_hold_qualifying_rows": row["actual_hold_qualifying_rows"],
                    "longest_actual_qualifying_run_steps": row["longest_actual_qualifying_run_steps"],
                    "logged_goal_stable_rows": row["logged_goal_stable_rows"],
                    "logged_bonus_triggered_rows": row["logged_bonus_triggered_rows"],
                    "logged_strict_success_rows": row["logged_strict_success_rows"],
                }
                for row in phase_summary
                if row["arm"] == "R" and row["phase"] == "initial_validation"
            ),
            {},
        ),
        "reward_coefficients_observed": {
            "goal_progress_coef": 1.0,
            "goal_direction_coef": 0.75,
            "braking_action_reward_coef": 1.0,
            "speed_deficit_penalty_weight": 0.05,
            "near_goal_speed_penalty_weight": 0.1,
            "braking_progress_reward_coef": 0.0,
            "terminal_braking_reward_coef": 0.0,
            "terminal_acceleration_penalty_coef": 0.0,
            "terminal_radial_progress_reward_coef": 0.0,
            "terminal_target_velocity_vector_reward_coef": 0.0,
            "target_velocity_action_reward_coef": 0.0,
        },
        "decision": {
            "unequivocal_reward_bug_found": False,
            "minimal_fix": None,
                "evidence": "The logged signs and gates match the source contract: dense distance progress, direction credit for non-braking increments, braking credit/acceleration penalty only when pre-action speed exceeds the distance-aware target, and a truth-state five-step success gate. The logged failure includes insufficient near-goal braking exposure and persistent acceleration in R; neither alone establishes a unique causal explanation or licenses changing the action limits.",
        },
    }
    report = {
        "run_name": RUN_NAME,
        "scope": {
            "arms": list(ARMS),
            "input": "steps.csv only; reward_components and action_execution JSON fields are parsed without simulator execution",
            "rows_by_arm": {arm: len(rows_by_arm[arm]) for arm in ARMS},
            "phases_by_arm": {arm: sorted({row["phase"] for row in rows_by_arm[arm]}) for arm in ARMS},
        },
        "normalization": _normalization_summary(log_root),
        "source_contract": source_contract,
        "key_findings": key_findings,
        "phase_summary": phase_summary,
        "action_summary": action_summary,
        "band_summary": band_summary,
        "episode_summary": episode_summary,
        "sanity": {
            "all_reward_components_present": all(
                bool(row["rewards"]) for rows in rows_by_arm.values() for row in rows
            ),
            "all_action_execution_present": all(
                bool(row["action"]) for rows in rows_by_arm.values() for row in rows
            ),
            "bonus_rows": sum(
                row["bonus_triggered"] for rows in rows_by_arm.values() for row in rows
            ),
            "goal_condition_rows": sum(
                row["goal_condition"] for rows in rows_by_arm.values() for row in rows
            ),
            "strict_success_rows": sum(
                row["strict_success"] for rows in rows_by_arm.values() for row in rows
            ),
            "max_logged_stable_steps": max(
                (row["stable_steps"] for rows in rows_by_arm.values() for row in rows),
                default=0,
            ),
        },
        "limitations": [
            "This is an exploratory audit of logged trajectories, not a causal intervention or a proof that a different reward would train successfully.",
            "Final validation/challenge are finite fixed-seed trajectories; R has 549 initial-validation rows because eight episodes terminated after strict five-step success, so row counts are not treated as equal exposure.",
            "The CSV does not expose policy gradients or counterfactual actions; action dependence is assessed from executed longitudinal increments and the logged reward terms.",
            "Actual speed and distance are post-state telemetry; reward target speed and reward gates use estimated state in the environment, so small disagreements are expected.",
        ],
    }
    _write_csv(output_root / "route_reward_band_summary.csv", band_summary)
    _write_csv(output_root / "route_reward_phase_summary.csv", phase_summary)
    _write_csv(output_root / "route_reward_episode_summary.csv", episode_summary)
    (output_root / "route_reward_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "route_reward_key_findings.json").write_text(
        json.dumps(key_findings, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    result = run_audit(project)
    print(json.dumps({"run_name": result["run_name"], "rows_by_arm": result["scope"]["rows_by_arm"], "output": "logs/preliminary/" + OUTPUT_RUN_NAME + "/reward_audit"}, ensure_ascii=False))
