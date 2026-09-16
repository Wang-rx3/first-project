"""Read-only visualization and consistency audit for a single-UAV run.

The module intentionally has no training or environment control path.  It
reads CSV/JSON/checkpoint metadata, computes diagnostic summaries, and writes
figures plus provenance files to a caller-selected directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


DISTANCE_BIN_ORDER = (">8m", "8–5m", "5–3m", "3–1m", "≤1m")
POLICY_ORDER = (
    "baseline_rl",
    "dagger_round1",
    "navigation_preserving_fine_tuned",
    "bc_actor",
    "oracle_teacher",
)
POLICY_LABELS = {
    "baseline_rl": "Baseline RL",
    "dagger_round1": "DAgger Actor",
    "navigation_preserving_fine_tuned": "Fine-tuned Actor",
    "bc_actor": "BC Actor",
    "oracle_teacher": "Oracle / Teacher",
    "baseline_42_migrated": "Baseline migrated",
}
POLICY_COLORS = {
    "baseline_rl": "#2563eb",
    "dagger_round1": "#7c3aed",
    "navigation_preserving_fine_tuned": "#f59e0b",
    "bc_actor": "#dc2626",
    "oracle_teacher": "#16a34a",
    "baseline_42_migrated": "#6b7280",
}
PHASE_COLORS = {
    "normal": "#dbeafe",
    "decision": "#fef3c7",
    "approach": "#d1fae5",
    "bypass": "#fee2e2",
    "post_bypass": "#ede9fe",
    "recovery": "#cffafe",
    "transition": "#f3f4f6",
    "boundary_onset": "#ffedd5",
    "goal_approach": "#dcfce7",
    "braking": "#e0e7ff",
}

EPISODE_FIELDS = {
    "policy", "scenario", "safety_mode", "episode_index", "steps",
    "entered_8m", "entered_5m", "entered_3m", "entered_1m", "stable_goal",
    "minimum_goal_distance_m", "minimum_true_goal_distance_m", "mean_speed_mps",
    "mean_speed_within_1m_mps", "mean_speed_at_8m_entry_mps",
    "mean_speed_at_5m_entry_mps", "mean_speed_at_3m_entry_mps",
    "mean_speed_at_1m_entry_mps", "maximum_consecutive_goal_hold_steps",
    "hold_steps_ge_1", "hold_steps_ge_3", "hold_steps_ge_5",
    "pre_candidate_static_swept_conflict_steps", "pre_candidate_static_unsafe_steps",
    "safety_projection_steps", "safety_projection_episode",
    "boundary_projection_steps", "boundary_projection_episode",
    "raw_action_changed_steps", "action_clipping_fraction",
    "minimum_pre_clearance_m", "minimum_post_clearance_m", "collision",
    "physical_collision", "clearance_violation", "passed_obstacle",
    "termination_reason", "split", "kind", "episode_id", "scenario_hash",
    "scenario_seed", "noise_seed", "round_index", "pair_id", "collection_mode",
    "student_closed_loop", "teacher_query", "truth_state_read_for_control",
    "writer_summary", "post_bypass_recovery",
    "post_bypass_max_same_sign_lateral_steps",
    "post_bypass_positive_radial_progress_steps",
    "training_performed", "policy_updates_performed", "ppo_updates",
    "evaluation_family",
}

STEP_FIELDS = {
    "policy", "scenario", "safety_mode", "episode_index", "step", "timestamp_s",
    "estimated_goal_distance_m", "true_goal_distance_before_m",
    "true_goal_distance_after_m", "true_delta_goal_distance_m",
    "true_radial_progress_mps", "estimated_radial_velocity_mps",
    "true_radial_velocity_before_mps", "true_radial_velocity_after_mps",
    "tangential_speed_before_mps", "tangential_speed_after_mps",
    "total_speed_before_mps", "total_speed_mps",
    "raw_delta_v_x", "raw_delta_v_y", "raw_delta_v_z", "raw_action_norm_mps",
    "applied_delta_v_x", "applied_delta_v_y", "applied_delta_v_z",
    "applied_action_norm_mps", "raw_applied_difference_norm_mps",
    "raw_action_changed", "action_clipped", "safety_projected",
    "boundary_projected", "pre_candidate_static_swept_minimum_clearance_m",
    "pre_candidate_static_swept_conflict", "pre_candidate_static_unsafe",
    "relevant_obstacle_id", "relevant_obstacle_clearance_m",
    "relevant_obstacle_candidate_clearance_m", "oracle_obstacle_category",
    "oracle_route_phase", "oracle_lateral_action_projection_mps",
    "oracle_desired_speed_mps", "oracle_query_gate", "collision",
    "physical_collision", "clearance_violation", "minimum_clearance_m",
    "termination_reason", "stable_condition", "goal_stable",
    "consecutive_stable_steps", "scenario_hash", "category", "side",
    "obstacle_category", "obstacle_side", "expected_lateral_sign", "phase",
    "base_phase", "passed_obstacle_before", "passed_obstacle_after",
    "first_obstacle_pass_step", "first_boundary_projection_step", "yaw_rad",
    "goal_relative_x_m", "goal_relative_y_m", "goal_relative_z_m",
    "goal_bearing_rad", "heading_to_goal_error_rad", "forward_velocity_mps",
    "lateral_velocity_mps", "obstacle_relative_x_body_m",
    "obstacle_relative_y_body_m", "obstacle_relative_z_body_m",
    "position_before_x_m", "position_before_y_m", "position_before_z_m",
    "position_after_x_m", "position_after_y_m", "position_after_z_m",
    "velocity_before_x_mps", "velocity_before_y_mps", "velocity_before_z_mps",
    "goal_direction_x", "goal_direction_y", "goal_direction_z",
    "raw_actor_delta_v_x_mps", "raw_actor_delta_v_y_mps",
    "raw_actor_delta_v_z_mps", "after_braking_delta_v_x_mps",
    "after_braking_delta_v_y_mps", "after_braking_delta_v_z_mps",
    "after_boundary_delta_v_x_mps", "after_boundary_delta_v_y_mps",
    "after_boundary_delta_v_z_mps", "after_safety_delta_v_x_mps",
    "after_safety_delta_v_y_mps", "after_safety_delta_v_z_mps",
    "raw_vs_boundary_action_difference_norm_mps",
    "raw_vs_safety_action_difference_norm_mps",
    "safety_vs_boundary_action_difference_norm_mps",
    "raw_lateral_action_mps", "final_lateral_action_mps",
    "raw_forward_action_mps", "raw_braking_magnitude_mps",
    "effective_braking_magnitude_mps", "lateral_forward_action_ratio",
    "cumulative_lateral_displacement_after_pass_m", "obstacle_passed_after",
    "nearest_boundary", "distance_to_nearest_boundary_m",
    "distance_to_x_min_boundary_m", "distance_to_x_max_boundary_m",
    "distance_to_y_min_boundary_m", "distance_to_y_max_boundary_m",
    "distance_to_z_min_boundary_m", "distance_to_z_max_boundary_m",
    "execution_order", "after_boundary_is_reconstructed",
    "after_safety_is_final_applied_action", "braking_safety_projected",
    "evaluation_stage", "collection_mode", "student_closed_loop",
    "teacher_query", "episode_id", "scenario_seed", "noise_seed", "round_index",
    "pair_id", "kind", "teacher_delta_v_x_mps", "teacher_delta_v_y_mps",
    "teacher_delta_v_z_mps", "teacher_action_norm_mps",
    "teacher_actor_component_limit_mps", "teacher_actor_constraint_affected",
    "teacher_actor_any_axis_saturated", "teacher_action_componentwise_legal",
    "oracle_truth_state_read", "raw_radial_action_projection_mps",
    "raw_tangential_action_mps", "negative_radial_velocity",
    "truth_state_read_for_control", "evaluation_family", "evaluation_role",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_csv(path: Path, fields: set[str] | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if fields is None:
            return [dict(row) for row in reader]
        return [{field: row.get(field, "") for field in fields} for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fields = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite(values: Iterable[Any]) -> np.ndarray:
    result: list[float] = []
    for value in values:
        number = _number(value)
        if np.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def _mean(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return float(np.mean(numbers)) if numbers.size else float("nan")


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distance_bin(distance_m: Any) -> str | None:
    """Map a distance to the documented half-open diagnostic bins."""

    distance = _number(distance_m)
    if not np.isfinite(distance):
        return None
    if distance > 8.0:
        return ">8m"
    if distance > 5.0:
        return "8–5m"
    if distance > 3.0:
        return "5–3m"
    if distance > 1.0:
        return "3–1m"
    return "≤1m"


def aggregate_episode_returns(returns: Sequence[float]) -> list[float]:
    """Return cumulative means without changing the episode return values."""

    cumulative: list[float] = []
    total = 0.0
    for index, value in enumerate(returns, start=1):
        total += float(value)
        cumulative.append(total / index)
    return cumulative


def parse_mappo_loss_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep trainer-reported loss fields only; never infer BC loss as PPO loss."""

    allowed = (
        "update", "episode", "actor_loss", "critic_loss", "entropy", "approximate_kl",
        "clip_fraction", "loss", "target_kl", "samples", "optimizer_steps",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        update = _int(row.get("update"), -1)
        if update < 0:
            continue
        item = {field: row.get(field, "") for field in allowed}
        item["update"] = update
        output.append(item)
    return output


def same_scenario_pairing(rows: Sequence[Mapping[str, Any]], policies: Sequence[str]) -> dict[str, Any]:
    by_policy = {
        policy: {str(row.get("scenario_hash", "")) for row in rows if str(row.get("policy", "")) == policy}
        for policy in policies
    }
    sets = [value for value in by_policy.values()]
    paired = bool(sets) and all(value == sets[0] for value in sets)
    return {
        "paired": paired,
        "scenario_counts": {policy: len(value) for policy, value in by_policy.items()},
        "intersection_count": len(set.intersection(*sets)) if sets else 0,
        "sets": {policy: sorted(value) for policy, value in by_policy.items()},
    }


def phase_distance_counts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        split = str(row.get("split", ""))
        mode = str(row.get("collection_mode", ""))
        phase = str(row.get("phase", ""))
        distance_bin = _distance_bin(row.get("goal_distance_m"))
        if split and mode and phase and distance_bin:
            counts[(split, mode, phase, distance_bin)] += 1
    denominators: Counter[tuple[str, str]] = Counter()
    for (split, mode, _phase, distance_key), count in counts.items():
        denominators[(split, mode)] += count
    output: list[dict[str, Any]] = []
    for (split, mode, phase, distance_bin), count in sorted(counts.items()):
        denominator = denominators[(split, mode)]
        output.append({
            "split": split,
            "collection_mode": mode,
            "phase": phase,
            "distance_bin": distance_bin,
            "count": count,
            "denominator": denominator,
            "fraction": _rate(count, denominator),
        })
    return output


def vector_from_row(row: Mapping[str, Any], prefix: str) -> np.ndarray:
    values: list[float] = []
    for axis in ("x", "y", "z"):
        value = row.get(f"{prefix}_{axis}")
        if value in (None, ""):
            value = row.get(f"{prefix}_{axis}_mps")
        values.append(_number(value))
    return np.asarray(values, dtype=float)


def _goal_direction(row: Mapping[str, Any]) -> np.ndarray:
    direction = np.asarray([
        _number(row.get("goal_direction_x")),
        _number(row.get("goal_direction_y")),
        _number(row.get("goal_direction_z")),
    ], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.full(3, np.nan)
    return direction / norm


def radial_tangential(action: Sequence[float], direction: Sequence[float]) -> tuple[float, float]:
    """Return signed goal-radial projection and nonnegative tangent magnitude."""

    action_array = np.asarray(action, dtype=float).reshape(3)
    direction_array = np.asarray(direction, dtype=float).reshape(3)
    radial = float(np.dot(action_array, direction_array))
    tangential = float(np.linalg.norm(action_array - radial * direction_array))
    return radial, tangential


def funnel_counts(rows: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    policy_rows = [row for row in rows if str(row.get("policy", "")) == policy]
    stages = (
        ("total_episodes", None),
        ("passed_obstacle", "passed_obstacle"),
        ("post_bypass_recovery", "post_bypass_recovery"),
        ("entered_8m", "entered_8m"),
        ("entered_5m", "entered_5m"),
        ("entered_3m", "entered_3m"),
        ("entered_1m", "entered_1m"),
        ("stable_goal", "stable_goal"),
    )
    active = list(policy_rows)
    output: list[dict[str, Any]] = []
    total = len(policy_rows)
    for stage, field in stages:
        if field is not None:
            active = [row for row in active if _bool(row.get(field))]
        count = len(active)
        output.append({"policy": policy, "stage": stage, "episode_count": count, "rate": _rate(count, total)})
    return output


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _resolve_input_run(input_run: Path) -> tuple[Path, Path]:
    log_root = input_run.resolve()
    dagger = log_root / "dagger_bc"
    if dagger.is_dir():
        return log_root, dagger
    if (log_root / "evaluation_episode_metrics.csv").is_file():
        return log_root.parent, log_root
    raise FileNotFoundError(f"cannot locate dagger_bc logs below {input_run}")


def _infer_data_root(log_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    run_name = log_root.name
    candidate = _repo_root() / "data" / "synthetic" / run_name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"cannot infer data root for {run_name}; pass --data-run")


def _writer_path(row: Mapping[str, Any], data_root: Path) -> Path | None:
    raw = str(row.get("writer_summary", ""))
    if raw:
        try:
            payload = json.loads(raw)
            path_value = payload.get("path")
            if path_value:
                candidate = Path(str(path_value))
                if not candidate.is_absolute():
                    candidate = _repo_root() / candidate
                if candidate.is_dir():
                    return candidate
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    policy = str(row.get("policy", ""))
    kind = str(row.get("kind", ""))
    split = str(row.get("split", ""))
    episode_id = str(row.get("episode_id", ""))
    candidate = data_root / "dagger_bc" / "evaluation" / (
        "static_holdout" if kind == "static_obstacle" else "normal_far_distance"
    ) / policy / "scale_1" / split / episode_id
    return candidate if candidate.is_dir() else None


def _load_scenario(trace_dir: Path | None) -> dict[str, Any]:
    if trace_dir is None:
        return {}
    path = trace_dir / "scenario.json"
    return _read_json(path) if path.is_file() else {}


def _load_trajectory(trace_dir: Path | None) -> list[dict[str, str]]:
    if trace_dir is None:
        return []
    path = trace_dir / "trajectory.csv"
    return _read_csv(path, {"timestamp", "x", "y", "z", "ground_truth_x", "ground_truth_y", "ground_truth_z"}) if path.is_file() else []


def _subprocess(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(args), cwd=_repo_root(), check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def _snapshot_checkpoints(payloads: Sequence[Any], explicit: Sequence[Path]) -> list[dict[str, Any]]:
    candidates: set[Path] = set(path.resolve() for path in explicit if path.is_file())
    for payload in payloads:
        for value in _walk_strings(payload):
            if value.lower().endswith(".pt"):
                path = Path(value)
                if not path.is_absolute():
                    path = _repo_root() / path
                if path.is_file():
                    candidates.add(path.resolve())
    result = []
    for path in sorted(candidates):
        result.append({"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return result


def _scenario_snapshot(scenario: Mapping[str, Any]) -> dict[str, Any]:
    generation = scenario.get("scenario_generation", {})
    contract = scenario.get("motion_action_contract", {})
    reward = scenario.get("reward_config", {})
    return {
        "scenario_id": scenario.get("scenario_id"),
        "phase": scenario.get("phase"),
        "uav_count": scenario.get("uav_count"),
        "duration_s": scenario.get("duration_s"),
        "max_steps": scenario.get("max_steps"),
        "units": scenario.get("units"),
        "motion_action_contract": contract,
        "world_bounds": {
            "min_m": generation.get("bounds_min_m"),
            "max_m": generation.get("bounds_max_m"),
        },
        "initial_positions": generation.get("initial_positions"),
        "goals": generation.get("goals"),
        "obstacle_specs": generation.get("obstacle_specs"),
        "reward_config": reward,
        "termination_reason": scenario.get("termination_reason"),
        "safety_execution_mode": scenario.get("safety_execution_mode"),
    }


def _sorted_rows(rows: Sequence[Mapping[str, Any]], *, policy: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    selected = [
        dict(row) for row in rows
        if (policy is None or str(row.get("policy", "")) == policy)
        and (kind is None or str(row.get("kind", "")) == kind)
    ]
    return sorted(selected, key=lambda row: (_int(row.get("episode_index"), 10**9), _int(row.get("step"), 10**9)))


def _first_step(rows: Sequence[Mapping[str, Any]], predicate: Any) -> int | None:
    for row in rows:
        if predicate(row):
            return _int(row.get("step"), 0)
    return None


def _episode_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("policy", "")), str(row.get("kind", "")), str(row.get("episode_id", "")))


def _group_steps(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_episode_key(row)].append(dict(row))
    for key in grouped:
        grouped[key].sort(key=lambda row: _int(row.get("step"), 10**9))
    return grouped


def _choose_static_pair(
    episode_rows: Sequence[Mapping[str, Any]],
    required_policies: Sequence[str],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    static = [row for row in episode_rows if str(row.get("kind", "")) == "static_obstacle"]
    by_policy_hash: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in static:
        by_policy_hash[str(row.get("policy", ""))][str(row.get("scenario_hash", ""))] = dict(row)
    common = set.intersection(
        *[set(by_policy_hash.get(policy, {})) for policy in required_policies]
    ) if required_policies else set()
    candidates = sorted(common)
    for scenario_hash in candidates:
        bc = by_policy_hash.get("bc_actor", {}).get(scenario_hash)
        if bc and _bool(bc.get("passed_obstacle")) and _bool(bc.get("post_bypass_recovery")) and not _bool(bc.get("entered_1m")):
            return scenario_hash, {policy: by_policy_hash[policy][scenario_hash] for policy in required_policies}, {
                "selection": "bc_passed_recovered_but_not_entered_1m",
                "bc_episode_id": bc.get("episode_id"),
            }
    if candidates:
        scenario_hash = candidates[0]
        return scenario_hash, {policy: by_policy_hash[policy][scenario_hash] for policy in required_policies}, {
            "selection": "first_common_static_scenario",
            "bc_episode_id": by_policy_hash.get("bc_actor", {}).get(scenario_hash, {}).get("episode_id"),
        }
    raise ValueError("no common static scenario across requested policies")


def _obstacle_patches(ax: Any, scenario: Mapping[str, Any]) -> None:
    generation = scenario.get("scenario_generation", {})
    for obstacle in generation.get("obstacle_specs", []) or []:
        if str(obstacle.get("obstacle_type", "")) != "static":
            continue
        center = obstacle.get("center_m", [float("nan"), float("nan"), float("nan")])
        x, y = _number(center[0]), _number(center[1])
        geometry = str(obstacle.get("geometry_type", ""))
        if geometry == "sphere":
            radius = _number(obstacle.get("radius"), 0.0)
            ax.add_patch(Circle((x, y), radius, facecolor="#9ca3af", edgecolor="#374151", alpha=0.35, label="Static obstacle"))
        elif geometry == "box":
            x_min, x_max = _number(obstacle.get("x_min")), _number(obstacle.get("x_max"))
            y_min, y_max = _number(obstacle.get("y_min")), _number(obstacle.get("y_max"))
            if all(np.isfinite(v) for v in (x_min, x_max, y_min, y_max)):
                ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, facecolor="#9ca3af", edgecolor="#374151", alpha=0.35, label="Static obstacle"))


def _world_bounds(ax: Any, scenario: Mapping[str, Any]) -> None:
    generation = scenario.get("scenario_generation", {})
    low = generation.get("bounds_min_m")
    high = generation.get("bounds_max_m")
    if isinstance(low, list) and isinstance(high, list) and len(low) >= 2 and len(high) >= 2:
        x_min, x_max = float(low[0]), float(high[0])
        y_min, y_max = float(low[1]), float(high[1])
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="#111827", linewidth=1.0, linestyle=":", label="World boundary"))


def _goal_start(scenario: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    generation = scenario.get("scenario_generation", {})
    positions = generation.get("initial_positions", {})
    goals = generation.get("goals", {})
    start = np.asarray(positions.get("uav_00", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
    goal = np.asarray(goals.get("uav_00", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
    return start, goal


def _phase_label(row: Mapping[str, Any]) -> str:
    value = str(row.get("phase") or row.get("base_phase") or "")
    return value if value else "unknown"


def _action_components(row: Mapping[str, Any], prefix: str) -> tuple[float, float, float, float]:
    direction = _goal_direction(row)
    action = vector_from_row(row, prefix)
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(direction)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    radial, tangential = radial_tangential(action, direction)
    magnitude = float(np.linalg.norm(action))
    velocity = np.asarray([
        _number(row.get("velocity_before_x_mps")),
        _number(row.get("velocity_before_y_mps")),
        _number(row.get("velocity_before_z_mps")),
    ], dtype=float)
    speed = float(np.linalg.norm(velocity))
    braking = max(0.0, -float(np.dot(action, velocity / speed))) if speed > 1e-12 else 0.0
    return radial, tangential, braking, magnitude


def _static_episode_rows(episode_rows: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    return [
        dict(row) for row in episode_rows
        if str(row.get("policy", "")) == policy and str(row.get("kind", "")) == "static_obstacle" and str(row.get("split", "")) == "test"
    ]


def _plot_reward(
    mappo_run: Path,
    out_dir: Path,
    baseline_checkpoint: Path,
) -> dict[str, Any]:
    path = mappo_run / "episode_metrics.csv"
    rows = _read_csv(path, {"episode_id", "split", "episode_return", "termination_reason", "collision", "stable_goal"})
    train = [row for row in rows if str(row.get("split", "")) == "train" and np.isfinite(_number(row.get("episode_return")))]
    train.sort(key=lambda row: _int(str(row.get("episode_id", "")).split("_")[-1], 10**9))
    returns = [_number(row.get("episode_return")) for row in train]
    means = aggregate_episode_returns(returns)
    output = []
    for index, (row, mean) in enumerate(zip(train, means), start=1):
        output.append({
            "episode": index,
            "split": "train",
            "episode_return": returns[index - 1],
            "cumulative_mean_return": mean,
            "termination_reason": row.get("termination_reason", ""),
            "collision": row.get("collision", ""),
            "stable_goal": row.get("stable_goal", ""),
        })
    csv_path = out_dir / "01_cumulative_reward_curve.csv"
    _write_csv(csv_path, output, ("episode", "split", "episode_return", "cumulative_mean_return", "termination_reason", "collision", "stable_goal"))
    figure, axis = plt.subplots(figsize=(16, 9))
    if output:
        x = [row["episode"] for row in output]
        axis.plot(x, [row["episode_return"] for row in output], color="#9ca3af", marker="o", markersize=2.8, linewidth=1.0, alpha=0.75, label="Raw episode return")
        axis.plot(x, [row["cumulative_mean_return"] for row in output], color="#2563eb", linewidth=2.2, label="Cumulative mean return")
    axis.set_title("Baseline episode return history")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode return (sum of step rewards)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    _save_figure(figure, out_dir / "01_cumulative_reward_curve.png")
    return {
        "available": bool(output),
        "rows": len(output),
        "source": str(path.resolve()),
        "checkpoint": str(baseline_checkpoint.resolve()),
        "first_return": returns[0] if returns else None,
        "last_return": returns[-1] if returns else None,
        "mean_return": float(np.mean(returns)) if returns else None,
        "min_return": float(np.min(returns)) if returns else None,
        "max_return": float(np.max(returns)) if returns else None,
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "01_cumulative_reward_curve.png").resolve()),
    }


def _plot_losses(mappo_run: Path, out_dir: Path, baseline_checkpoint: Path) -> dict[str, Any]:
    path = mappo_run / "training_metrics.csv"
    source_rows = _read_csv(path)
    rows = parse_mappo_loss_rows(source_rows)
    fields = ("actor_loss", "critic_loss", "entropy", "approximate_kl", "clip_fraction", "loss")
    output = []
    for row in rows:
        item = {
            "update": row.get("update", ""),
            "episode": row.get("episode", ""),
            "source_training_run": str(mappo_run.resolve()),
            "source_checkpoint": str(baseline_checkpoint.resolve()),
            "source_log_file": str(path.resolve()),
        }
        for field in fields:
            item[field] = row.get(field, "")
        output.append(item)
    csv_path = out_dir / "02_mappo_training_loss.csv"
    _write_csv(csv_path, output, ("update", "episode", *fields, "source_training_run", "source_checkpoint", "source_log_file"))
    available = [field for field in fields if any(np.isfinite(_number(row.get(field))) for row in output)]
    figure, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.ravel()
    labels = {
        "actor_loss": "Actor / policy loss",
        "critic_loss": "Critic / value loss",
        "entropy": "Entropy",
        "approximate_kl": "Approximate KL",
        "clip_fraction": "Clip fraction",
        "loss": "Trainer total loss",
    }
    x = [_int(row.get("update"), 0) for row in output]
    for axis, field in zip(axes_flat, fields):
        y = [_number(row.get(field)) for row in output]
        mask = np.isfinite(y)
        if np.any(mask):
            axis.plot(np.asarray(x)[mask], np.asarray(y)[mask], color="#2563eb", linewidth=1.8)
        else:
            axis.text(0.5, 0.5, "unavailable", transform=axis.transAxes, ha="center", va="center")
        axis.set_title(labels[field])
        axis.set_xlabel("PPO update")
        axis.grid(alpha=0.25)
    figure.suptitle("Baseline MAPPO/PPO training history", fontsize=16)
    _save_figure(figure, out_dir / "02_mappo_training_loss.png")
    return {
        "available": bool(output),
        "rows": len(output),
        "fields": available,
        "source": str(path.resolve()),
        "source_checkpoint": str(baseline_checkpoint.resolve()),
        "plot": str((out_dir / "02_mappo_training_loss.png").resolve()),
        "csv": str(csv_path.resolve()),
        "current_run_ppo_updates": 0,
    }


def _trajectory_markers(
    step_rows: Sequence[Mapping[str, Any]],
    episode_row: Mapping[str, Any],
) -> dict[str, int | None]:
    ordered = sorted(step_rows, key=lambda row: _int(row.get("step"), 10**9))
    return {
        "first_obstacle_risk": _first_step(
            ordered,
            lambda row: _bool(row.get("pre_candidate_static_swept_conflict")) or _bool(row.get("pre_candidate_static_unsafe")),
        ),
        "obstacle_pass": _first_step(ordered, lambda row: _bool(row.get("passed_obstacle_after"))),
        "recovery": _first_step(
            ordered,
            lambda row: str(row.get("phase", "")).lower() in {"recovery", "post_bypass_recovery"},
        ),
        "entered_8m": _first_step(ordered, lambda row: _number(row.get("estimated_goal_distance_m")) <= 8.0),
        "entered_3m": _first_step(ordered, lambda row: _number(row.get("estimated_goal_distance_m")) <= 3.0),
        "entered_1m": _first_step(ordered, lambda row: _number(row.get("estimated_goal_distance_m")) <= 1.0),
    }


def _plot_trajectory(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    required = ("baseline_rl", "bc_actor", "oracle_teacher")
    scenario_hash, selected, selection = _choose_static_pair(episode_rows, required)
    step_groups = _group_steps(step_rows)
    trajectory_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    for axis, policy in zip(axes, required):
        episode_row = selected[policy]
        trace_dir = _writer_path(episode_row, data_root)
        scenario = _load_scenario(trace_dir)
        trajectory = _load_trajectory(trace_dir)
        if not trajectory:
            # Use step-level post-state coordinates when a trace is absent.
            trajectory = [
                {
                    "timestamp": row.get("step", ""),
                    "x": row.get("position_after_x_m", ""),
                    "y": row.get("position_after_y_m", ""),
                    "z": row.get("position_after_z_m", ""),
                }
                for row in step_groups.get((policy, "static_obstacle", str(episode_row.get("episode_id", ""))), [])
            ]
        x = [_number(row.get("x")) for row in trajectory]
        y = [_number(row.get("y")) for row in trajectory]
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            axis.plot(np.asarray(x)[mask], np.asarray(y)[mask], color=POLICY_COLORS[policy], linewidth=2.0, label=POLICY_LABELS[policy])
        start, goal = _goal_start(scenario)
        if np.all(np.isfinite(start[:2])):
            axis.scatter([start[0]], [start[1]], color="#16a34a", s=55, marker="o", zorder=5, label="Start")
        if np.all(np.isfinite(goal[:2])):
            axis.scatter([goal[0]], [goal[1]], color="#dc2626", s=80, marker="*", zorder=5, label="Goal")
            axis.add_patch(Circle((goal[0], goal[1]), 1.0, fill=False, color="#dc2626", linestyle="--", linewidth=1.0, label="Goal radius (1 m)"))
        _obstacle_patches(axis, scenario)
        _world_bounds(axis, scenario)
        rows_for_policy = step_groups.get((policy, "static_obstacle", str(episode_row.get("episode_id", ""))), [])
        markers = _trajectory_markers(rows_for_policy, episode_row)
        marker_specs = (
            ("first_obstacle_risk", "Risk", "#b91c1c", "x"),
            ("obstacle_pass", "Pass", "#7c3aed", "P"),
            ("recovery", "Recovery", "#0891b2", "D"),
            ("entered_8m", "8 m", "#ca8a04", "1"),
            ("entered_3m", "3 m", "#ea580c", "2"),
            ("entered_1m", "1 m", "#15803d", "3"),
        )
        for marker_name, label, color, marker in marker_specs:
            step = markers[marker_name]
            if step is None:
                continue
            matching = next((row for row in rows_for_policy if _int(row.get("step"), -1) == step), None)
            if matching is None:
                continue
            px, py = _number(matching.get("position_after_x_m")), _number(matching.get("position_after_y_m"))
            if np.isfinite(px) and np.isfinite(py):
                axis.scatter([px], [py], color=color, marker=marker, s=42, zorder=6, label=f"{label} step {step}")
        for row in rows_for_policy:
            px, py = _number(row.get("position_after_x_m")), _number(row.get("position_after_y_m"))
            if np.isfinite(px) and np.isfinite(py) and (_bool(row.get("safety_projected")) or _bool(row.get("boundary_projected"))):
                marker = "s" if _bool(row.get("safety_projected")) else "^"
                color = "#be123c" if _bool(row.get("safety_projected")) else "#9333ea"
                axis.scatter([px], [py], marker=marker, color=color, s=22, zorder=4)
        axis.set_title(
            f"{POLICY_LABELS[policy]} | {episode_row.get('episode_id')} | "
            f"pass={_bool(episode_row.get('passed_obstacle'))} "
            f"recovery={_bool(episode_row.get('post_bypass_recovery'))}\n"
            f"1m={_bool(episode_row.get('entered_1m'))} stable={_bool(episode_row.get('stable_goal'))} "
            f"safe={_int(episode_row.get('safety_projection_steps'), 0)} "
            f"boundary={_int(episode_row.get('boundary_projection_steps'), 0)}",
            fontsize=10,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        handles, labels = axis.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        axis.legend(dedup.values(), dedup.keys(), fontsize=7, loc="best")
        for index, row in enumerate(rows_for_policy):
            trajectory_rows.append({
                "step": row.get("step", index),
                "x": row.get("position_after_x_m", ""),
                "y": row.get("position_after_y_m", ""),
                "goal_x": goal[0] if np.isfinite(goal[0]) else "",
                "goal_y": goal[1] if np.isfinite(goal[1]) else "",
                "policy": policy,
                "phase": _phase_label(row),
                "safety_projected": row.get("safety_projected", ""),
                "boundary_projected": row.get("boundary_projected", ""),
                "scenario_hash": scenario_hash,
                "episode_id": episode_row.get("episode_id", ""),
                "first_obstacle_risk": markers["first_obstacle_risk"],
                "obstacle_pass_step": markers["obstacle_pass"],
                "recovery_step": markers["recovery"],
                "entered_8m_step": markers["entered_8m"],
                "entered_3m_step": markers["entered_3m"],
                "entered_1m_step": markers["entered_1m"],
                "stable_goal": episode_row.get("stable_goal", ""),
            })
    csv_path = out_dir / "03_xy_trajectory_overview.csv"
    _write_csv(csv_path, trajectory_rows)
    _save_figure(figure, out_dir / "03_xy_trajectory_overview.png")
    return {
        "available": True,
        "scenario_hash": scenario_hash,
        "selection": selection,
        "policies": list(required),
        "pairing": same_scenario_pairing(
            [row for row in episode_rows if str(row.get("kind", "")) == "static_obstacle"],
            required,
        ),
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "03_xy_trajectory_overview.png").resolve()),
    }


def _phase_spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, str]]:
    if not rows:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    current = _phase_label(rows[0])
    for index, row in enumerate(rows[1:], start=1):
        phase = _phase_label(row)
        if phase != current:
            spans.append((start, index - 1, current))
            start, current = index, phase
    spans.append((start, len(rows) - 1, current))
    return spans


def _plot_rollout(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    scenario_hash, selected, selection = _choose_static_pair(
        episode_rows, ("baseline_rl", "bc_actor", "oracle_teacher")
    )
    bc_episode = selected["bc_actor"]
    episode_id = str(bc_episode.get("episode_id", ""))
    rows = _group_steps(step_rows).get(("bc_actor", "static_obstacle", episode_id), [])
    if not rows:
        raise ValueError("selected BC representative has no step rows")
    rows.sort(key=lambda row: _int(row.get("step"), 10**9))
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        actor_radial, actor_tangent, actor_braking, actor_norm = _action_components(row, "raw_actor_delta_v")
        teacher_radial, teacher_tangent, teacher_braking, teacher_norm = _action_components(row, "teacher_delta_v")
        plot_rows.append({
            "step": row.get("step", ""),
            "timestamp_s": row.get("timestamp_s", ""),
            "goal_distance_m": row.get("estimated_goal_distance_m", ""),
            "obstacle_clearance_m": row.get("relevant_obstacle_clearance_m") or row.get("pre_candidate_static_swept_minimum_clearance_m", ""),
            "radial_velocity_mps": row.get("true_radial_velocity_after_mps", row.get("estimated_radial_velocity_mps", "")),
            "tangential_velocity_mps": row.get("tangential_speed_after_mps", ""),
            "actor_raw_radial_action_mps": actor_radial,
            "actor_raw_tangential_action_mps": actor_tangent,
            "actor_braking_magnitude_mps": actor_braking,
            "actor_action_norm_mps": actor_norm,
            "teacher_radial_action_mps": teacher_radial,
            "teacher_tangential_action_mps": teacher_tangent,
            "teacher_braking_magnitude_mps": teacher_braking,
            "teacher_action_norm_mps": teacher_norm,
            "safety_projected": row.get("safety_projected", ""),
            "boundary_projected": row.get("boundary_projected", ""),
            "phase": _phase_label(row),
            "passed_obstacle_after": row.get("passed_obstacle_after", ""),
            "stable_condition": row.get("stable_condition", ""),
            "scenario_hash": scenario_hash,
            "episode_id": episode_id,
        })
    csv_path = out_dir / "04_rollout_diagnostics.csv"
    _write_csv(csv_path, plot_rows)
    x = np.arange(len(plot_rows))
    figure, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    series = (
        ("goal_distance_m", "Goal distance (m)"),
        ("obstacle_clearance_m", "Obstacle clearance (m)"),
        ("radial_velocity_mps", "Radial velocity toward goal (m/s)"),
        ("tangential_velocity_mps", "Tangential velocity (m/s)"),
        ("actor_raw_radial_action_mps", "Actor raw radial action (m/s)"),
        ("actor_raw_tangential_action_mps", "Actor raw tangential action (m/s)"),
        ("teacher_radial_action_mps", "Teacher radial action (m/s)"),
        ("teacher_tangential_action_mps", "Teacher tangential action (m/s)"),
    )
    for axis, (field, label) in zip(axes.flat, series):
        y = np.asarray([_number(row.get(field)) for row in plot_rows])
        mask = np.isfinite(y)
        if np.any(mask):
            axis.plot(x[mask], y[mask], color="#2563eb" if "Actor" in label else "#16a34a", linewidth=1.7)
        axis.set_title(label)
        axis.set_xlabel("Step")
        axis.grid(alpha=0.22)
        for start, end, phase in _phase_spans(plot_rows):
            axis.axvspan(start - 0.5, end + 0.5, color=PHASE_COLORS.get(phase, "#f3f4f6"), alpha=0.16)
    flag_axis = axes.flat[-1]
    safety_steps = [i for i, row in enumerate(plot_rows) if _bool(row.get("safety_projected"))]
    boundary_steps = [i for i, row in enumerate(plot_rows) if _bool(row.get("boundary_projected"))]
    if safety_steps:
        flag_axis.vlines(safety_steps, 0.55, 1.0, color="#be123c", linewidth=2.0, label="Safety projection")
    if boundary_steps:
        flag_axis.vlines(boundary_steps, 0.0, 0.45, color="#9333ea", linewidth=2.0, label="Boundary projection")
    flag_axis.set_ylim(-0.05, 1.05)
    flag_axis.set_yticks([0.25, 0.75])
    flag_axis.set_yticklabels(["Boundary", "Safety"])
    flag_axis.set_xlabel("Step")
    flag_axis.set_title("Execution intervention flags")
    flag_axis.grid(alpha=0.22)
    if safety_steps or boundary_steps:
        flag_axis.legend(loc="best", fontsize=8)
    markers = _trajectory_markers(rows, bc_episode)
    for axis in axes.flat:
        if markers["obstacle_pass"] is not None:
            axis.axvline(markers["obstacle_pass"], color="#7c3aed", linestyle="--", linewidth=1.0, label="Obstacle pass")
        if markers["recovery"] is not None:
            axis.axvline(markers["recovery"], color="#0891b2", linestyle=":", linewidth=1.0, label="Recovery")
    figure.suptitle(
        f"BC Actor rollout diagnostics | {episode_id} | scenario {scenario_hash[:12]} | "
        f"pass={_bool(bc_episode.get('passed_obstacle'))} "
        f"recovery={_bool(bc_episode.get('post_bypass_recovery'))} "
        f"entered_1m={_bool(bc_episode.get('entered_1m'))} "
        f"stable={_bool(bc_episode.get('stable_goal'))}",
        fontsize=15,
    )
    _save_figure(figure, out_dir / "04_rollout_diagnostics.png")
    return {
        "available": True,
        "scenario_hash": scenario_hash,
        "episode_id": episode_id,
        "selection": selection,
        "markers": markers,
        "steps": len(plot_rows),
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "04_rollout_diagnostics.png").resolve()),
    }


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _finite(x), _finite(y)
    if xa.size < 2 or ya.size != xa.size or np.std(xa) <= 1e-12 or np.std(ya) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _plot_action_scatter(
    step_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for row in step_rows:
        if str(row.get("policy", "")) != "bc_actor" or str(row.get("kind", "")) != "static_obstacle":
            continue
        if not _bool(row.get("teacher_query")):
            continue
        direction = _goal_direction(row)
        actor = _action_components(row, "raw_actor_delta_v")
        teacher = _action_components(row, "teacher_delta_v")
        if not np.all(np.isfinite(direction)) or not np.isfinite(actor[3]) or not np.isfinite(teacher[3]):
            continue
        points.append({
            "record_type": "point",
            "policy": "bc_actor",
            "episode_id": row.get("episode_id", ""),
            "scenario_hash": row.get("scenario_hash", ""),
            "step": row.get("step", ""),
            "distance_bin": _distance_bin(row.get("estimated_goal_distance_m")) or "",
            "teacher_radial_action_mps": teacher[0],
            "actor_radial_action_mps": actor[0],
            "teacher_tangential_action_mps": teacher[1],
            "actor_tangential_action_mps": actor[1],
            "teacher_braking_magnitude_mps": teacher[2],
            "actor_braking_magnitude_mps": actor[2],
            "teacher_action_norm_mps": teacher[3],
            "actor_action_norm_mps": actor[3],
        })
    metrics = (
        ("radial_action", "teacher_radial_action_mps", "actor_radial_action_mps"),
        ("tangential_action", "teacher_tangential_action_mps", "actor_tangential_action_mps"),
        ("braking_magnitude", "teacher_braking_magnitude_mps", "actor_braking_magnitude_mps"),
        ("action_norm", "teacher_action_norm_mps", "actor_action_norm_mps"),
    )
    csv_rows = list(points)
    summary_rows: list[dict[str, Any]] = []
    for distance_bin in DISTANCE_BIN_ORDER:
        subset = [point for point in points if point["distance_bin"] == distance_bin]
        for metric, teacher_field, actor_field in metrics:
            teacher_values = [_number(point[teacher_field]) for point in subset]
            actor_values = [_number(point[actor_field]) for point in subset]
            mask = np.isfinite(teacher_values) & np.isfinite(actor_values)
            tv = np.asarray(teacher_values)[mask]
            av = np.asarray(actor_values)[mask]
            ratio = float(np.mean(np.abs(av) / np.maximum(np.abs(tv), 1e-8))) if tv.size else float("nan")
            summary_rows.append({
                "record_type": "summary",
                "policy": "bc_actor",
                "distance_bin": distance_bin,
                "metric": metric,
                "n": int(tv.size),
                "pearson": _correlation(tv, av),
                "mae": float(np.mean(np.abs(av - tv))) if tv.size else float("nan"),
                "rmse": float(np.sqrt(np.mean((av - tv) ** 2))) if tv.size else float("nan"),
                "mean_actor_teacher_magnitude_ratio": ratio,
            })
    csv_rows.extend(summary_rows)
    csv_path = out_dir / "05_teacher_vs_actor_action_scatter.csv"
    _write_csv(csv_path, csv_rows)
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    labels = {
        "radial_action": "Signed radial action (m/s)",
        "tangential_action": "Tangential action magnitude (m/s)",
        "braking_magnitude": "Braking magnitude (m/s)",
        "action_norm": "Action norm (m/s)",
    }
    for axis, (metric, teacher_field, actor_field) in zip(axes.flat, metrics):
        for distance_bin in DISTANCE_BIN_ORDER:
            subset = [point for point in points if point["distance_bin"] == distance_bin]
            if not subset:
                continue
            color = {
                ">8m": "#2563eb", "8–5m": "#16a34a", "5–3m": "#f59e0b",
                "3–1m": "#dc2626", "≤1m": "#7c3aed",
            }[distance_bin]
            tx = np.asarray([_number(point[teacher_field]) for point in subset])
            ay = np.asarray([_number(point[actor_field]) for point in subset])
            mask = np.isfinite(tx) & np.isfinite(ay)
            axis.scatter(tx[mask], ay[mask], s=8, alpha=0.18, color=color, label=distance_bin)
        values = np.asarray([
            _number(point[field])
            for point in points
            for field in (teacher_field, actor_field)
        ])
        values = values[np.isfinite(values)]
        if values.size:
            lower, upper = float(np.min(values)), float(np.max(values))
            if math.isclose(lower, upper):
                lower, upper = lower - 1.0, upper + 1.0
            axis.plot([lower, upper], [lower, upper], color="#111827", linestyle="--", linewidth=1.0, label="y=x")
        axis.set_title(labels[metric])
        axis.set_xlabel("Teacher action")
        axis.set_ylabel("Actor action")
        axis.grid(alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=7, loc="best")
    figure.suptitle("Teacher versus BC Actor action fidelity", fontsize=16)
    _save_figure(figure, out_dir / "05_teacher_vs_actor_action_scatter.png")
    return {
        "available": bool(points),
        "points": len(points),
        "summary": summary_rows,
        "source": "bc_actor evaluation rows with teacher_query=true; same observation record",
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "05_teacher_vs_actor_action_scatter.png").resolve()),
    }


def _plot_raw_applied(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    _scenario_hash, selected, _selection = _choose_static_pair(
        episode_rows, ("baseline_rl", "bc_actor", "oracle_teacher")
    )
    episode_row = selected["bc_actor"]
    episode_id = str(episode_row.get("episode_id", ""))
    rows = _group_steps(step_rows).get(("bc_actor", "static_obstacle", episode_id), [])
    rows.sort(key=lambda row: _int(row.get("step"), 10**9))
    records: list[dict[str, Any]] = []
    for row in rows:
        direction = _goal_direction(row)
        vectors = {
            "raw_actor": vector_from_row(row, "raw_actor_delta_v"),
            "after_braking": vector_from_row(row, "after_braking_delta_v"),
            "after_boundary": vector_from_row(row, "after_boundary_delta_v"),
            "applied": vector_from_row(row, "applied_delta_v"),
        }
        values: dict[str, Any] = {
            "step": row.get("step", ""),
            "timestamp_s": row.get("timestamp_s", ""),
            "safety_projected": row.get("safety_projected", ""),
            "boundary_projected": row.get("boundary_projected", ""),
            "action_clipped": row.get("action_clipped", ""),
            "episode_id": episode_id,
        }
        for name, vector in vectors.items():
            if np.all(np.isfinite(vector)) and np.all(np.isfinite(direction)):
                radial, tangent = radial_tangential(vector, direction)
                values[f"{name}_radial_mps"] = radial
                values[f"{name}_tangential_mps"] = tangent
                values[f"{name}_norm_mps"] = float(np.linalg.norm(vector))
            else:
                values[f"{name}_radial_mps"] = ""
                values[f"{name}_tangential_mps"] = ""
                values[f"{name}_norm_mps"] = ""
        applied, raw = vectors["applied"], vectors["raw_actor"]
        values["correction_norm_mps"] = float(np.linalg.norm(applied - raw)) if np.all(np.isfinite(applied)) and np.all(np.isfinite(raw)) else ""
        records.append(values)
    csv_path = out_dir / "06_raw_vs_applied_action.csv"
    _write_csv(csv_path, records)
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    x = np.arange(len(records))
    styles = {
        "raw_actor": ("Raw Actor", "#2563eb"),
        "after_braking": ("After braking supervisor", "#f59e0b"),
        "after_boundary": ("After boundary projection", "#9333ea"),
        "applied": ("Applied action", "#16a34a"),
    }
    for axis, component, label in (
        (axes[0, 0], "radial_mps", "Radial delta-v (m/s)"),
        (axes[0, 1], "tangential_mps", "Tangential delta-v (m/s)"),
        (axes[1, 0], "norm_mps", "Action norm (m/s)"),
    ):
        for name, (display, color) in styles.items():
            y = np.asarray([_number(row.get(f"{name}_{component}")) for row in records])
            mask = np.isfinite(y)
            if np.any(mask):
                axis.plot(x[mask], y[mask], linewidth=1.5, color=color, label=display)
        axis.set_title(label)
        axis.set_xlabel("Step")
        axis.grid(alpha=0.22)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=8, loc="best")
    correction = np.asarray([_number(row.get("correction_norm_mps")) for row in records])
    axes[1, 1].plot(x[np.isfinite(correction)], correction[np.isfinite(correction)], color="#dc2626", linewidth=1.7)
    axes[1, 1].set_title("Applied − raw correction norm (m/s)")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(alpha=0.22)
    for axis in axes.flat:
        safety = [i for i, row in enumerate(records) if _bool(row.get("safety_projected"))]
        boundary = [i for i, row in enumerate(records) if _bool(row.get("boundary_projected"))]
        if safety:
            axis.vlines(safety, *axis.get_ylim(), color="#be123c", alpha=0.18, linewidth=1.0)
        if boundary:
            axis.vlines(boundary, *axis.get_ylim(), color="#9333ea", alpha=0.18, linewidth=1.0)
    figure.suptitle(f"Raw Actor action versus applied action | BC episode {episode_id}", fontsize=16)
    _save_figure(figure, out_dir / "06_raw_vs_applied_action.png")
    total = len(records)
    raw_equal = sum(
        abs(_number(row.get("correction_norm_mps"))) <= 1e-9
        for row in records if np.isfinite(_number(row.get("correction_norm_mps")))
    )
    safety_steps = sum(_bool(row.get("safety_projected")) for row in records)
    boundary_steps = sum(_bool(row.get("boundary_projected")) for row in records)
    corrections = _finite(row.get("correction_norm_mps") for row in records)
    return {
        "available": bool(records),
        "episode_id": episode_id,
        "total_steps": total,
        "raw_equals_applied_fraction": _rate(raw_equal, total),
        "safety_projection_steps": safety_steps,
        "boundary_projection_steps": boundary_steps,
        "mean_action_correction_norm_mps": float(np.mean(corrections)) if corrections.size else None,
        "max_action_correction_norm_mps": float(np.max(corrections)) if corrections.size else None,
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "06_raw_vs_applied_action.png").resolve()),
    }


def _heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    title: str,
    *,
    normalized: bool,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="Blues" if not normalized else "YlGnBu")
    axis.set_xticks(np.arange(len(DISTANCE_BIN_ORDER)))
    axis.set_xticklabels(DISTANCE_BIN_ORDER)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xlabel("Goal-distance bin")
    axis.set_ylabel("Observed phase")
    axis.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_value = f"{value:.1%}" if normalized else f"{int(value)}"
            axis.text(j, i, text_value, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Fraction" if normalized else "Sample count")
    _save_figure(figure, path)


def _plot_dataset_heatmaps(
    demonstration_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    counts = phase_distance_counts(demonstration_rows)
    csv_path = out_dir / "07_dataset_distance_phase_counts.csv"
    _write_csv(csv_path, counts, ("split", "collection_mode", "phase", "distance_bin", "count", "denominator", "fraction"))
    phases = sorted({str(row["phase"]) for row in counts})
    modes = sorted({str(row["collection_mode"]) for row in counts})
    rows_n = 2
    cols_n = max(1, len(modes))
    if not phases:
        phases = ["unavailable"]
    if not modes:
        modes = ["unavailable"]
    figure, axes = plt.subplots(rows_n, cols_n, figsize=(16, 12), constrained_layout=True, squeeze=False)
    for split_index, split in enumerate(("train", "validation")):
        for mode_index, mode in enumerate(modes):
            rows_subset = [row for row in counts if row["split"] == split and row["collection_mode"] == mode]
            local_phases = sorted({str(row["phase"]) for row in rows_subset}) or phases
            matrix = np.zeros((len(local_phases), len(DISTANCE_BIN_ORDER)))
            for row in rows_subset:
                matrix[local_phases.index(str(row["phase"])), DISTANCE_BIN_ORDER.index(str(row["distance_bin"]))] = float(row["count"])
            axis = axes[split_index, mode_index]
            image = axis.imshow(matrix, aspect="auto", cmap="Blues")
            axis.set_xticks(np.arange(len(DISTANCE_BIN_ORDER)))
            axis.set_xticklabels(DISTANCE_BIN_ORDER, rotation=20)
            axis.set_yticks(np.arange(len(local_phases)))
            axis.set_yticklabels(local_phases)
            axis.set_title(f"{split} | {mode}")
            axis.set_xlabel("Distance bin")
            axis.set_ylabel("Phase")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Dataset distance × phase counts", fontsize=16)
    _save_figure(figure, out_dir / "07_dataset_distance_phase_heatmap.png")

    normalized_path = out_dir / "07_dataset_distance_phase_heatmap_normalized.png"
    figure, axes = plt.subplots(rows_n, cols_n, figsize=(16, 12), constrained_layout=True, squeeze=False)
    for split_index, split in enumerate(("train", "validation")):
        for mode_index, mode in enumerate(modes):
            rows_subset = [row for row in counts if row["split"] == split and row["collection_mode"] == mode]
            local_phases = sorted({str(row["phase"]) for row in rows_subset}) or phases
            matrix = np.zeros((len(local_phases), len(DISTANCE_BIN_ORDER)))
            denominator = sum(float(row["count"]) for row in rows_subset)
            for row in rows_subset:
                matrix[local_phases.index(str(row["phase"])), DISTANCE_BIN_ORDER.index(str(row["distance_bin"]))] = (
                    float(row["count"]) / denominator if denominator else 0.0
                )
            axis = axes[split_index, mode_index]
            vmax = max(0.01, float(np.max(matrix)) if matrix.size else 0.01)
            image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
            axis.set_xticks(np.arange(len(DISTANCE_BIN_ORDER)))
            axis.set_xticklabels(DISTANCE_BIN_ORDER, rotation=20)
            axis.set_yticks(np.arange(len(local_phases)))
            axis.set_yticklabels(local_phases)
            axis.set_title(f"{split} | {mode}")
            axis.set_xlabel("Distance bin")
            axis.set_ylabel("Phase")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center", fontsize=7)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Dataset distance × phase proportions", fontsize=16)
    _save_figure(figure, normalized_path)
    return {
        "available": bool(counts),
        "rows": len(counts),
        "splits": sorted({str(row["split"]) for row in counts}),
        "collection_modes": modes,
        "phases": phases,
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "07_dataset_distance_phase_heatmap.png").resolve()),
        "normalized_plot": str(normalized_path.resolve()),
    }


def _plot_funnel(
    episode_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    static_rows = [
        row for row in episode_rows
        if str(row.get("kind", "")) == "static_obstacle" and str(row.get("split", "")) == "test"
    ]
    policies = ("bc_actor", "oracle_teacher")
    rows: list[dict[str, Any]] = []
    for policy in policies:
        rows.extend(funnel_counts(static_rows, policy))
    csv_path = out_dir / "08_task_completion_funnel.csv"
    _write_csv(csv_path, rows, ("policy", "stage", "episode_count", "rate"))
    figure, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    stages = [row["stage"] for row in funnel_counts(static_rows, "bc_actor")]
    x = np.arange(len(stages))
    width = 0.38
    for offset, policy in ((-width / 2, "bc_actor"), (width / 2, "oracle_teacher")):
        values = [row["episode_count"] for row in funnel_counts(static_rows, policy)]
        axis.bar(x + offset, values, width=width, color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
        for pos, value in zip(x + offset, values):
            axis.text(pos, value + 0.2, str(value), ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(stages, rotation=25, ha="right")
    axis.set_ylabel("Episodes (sequential intersection)")
    axis.set_title("Static holdout task-completion funnel")
    axis.set_ylim(0, max(1, len(static_rows) + 3))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    _save_figure(figure, out_dir / "08_task_completion_funnel.png")
    return {
        "available": bool(static_rows),
        "episodes_per_policy": {policy: len(_static_episode_rows(episode_rows, policy)) for policy in policies},
        "rows": rows,
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "08_task_completion_funnel.png").resolve()),
    }


def _config_from_trace(episode_rows: Sequence[Mapping[str, Any]], data_root: Path) -> tuple[dict[str, Any], Path | None]:
    candidates = [row for row in episode_rows if str(row.get("kind", "")) == "static_obstacle"] or list(episode_rows)
    for row in candidates:
        path = _writer_path(row, data_root)
        if path is not None:
            scenario = _load_scenario(path)
            if scenario:
                return scenario, path
    return {}, None


def _manifest(
    *,
    input_root: Path,
    dagger_root: Path,
    data_root: Path,
    mappo_run: Path,
    baseline_checkpoint: Path,
    top_summary: Mapping[str, Any],
    bc_summary: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    algorithm_files = [
        _repo_root() / "src" / "experiments" / "single_uav_external_recovery_method_study.py",
        _repo_root() / "src" / "experiments" / "single_uav_static_obstacle_dagger_recovery_goal.py",
        _repo_root() / "src" / "simulation" / "environment.py",
        _repo_root() / "src" / "simulation" / "motion_model.py",
    ]
    explicit = [
        baseline_checkpoint,
        data_root / "dagger_bc" / "bc_final" / "scale_1" / "checkpoints" / "best_validation.pt",
        data_root / "dagger_bc" / "bc_round0" / "scale_1" / "checkpoints" / "best_validation.pt",
    ]
    checkpoints = _snapshot_checkpoints((top_summary, bc_summary), explicit)
    eval_path = dagger_root / "evaluation_episode_metrics.csv"
    eval_seed_rows = _read_csv(eval_path, {"scenario_seed"}) if eval_path.is_file() else []
    return {
        "visualization_task": "single_uav_current_run_visualization",
        "run_id": input_root.name,
        "input_run": str(input_root.resolve()),
        "dagger_log_root": str(dagger_root.resolve()),
        "data_root": str(data_root.resolve()),
        "mappo_training_run": str(mappo_run.resolve()),
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "git": {
            "commit": _subprocess(("git", "rev-parse", "HEAD")),
            "branch": _subprocess(("git", "branch", "--show-current")),
            "status": _subprocess(("git", "status", "--short", "--untracked-files=no")).splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "randomness": {
            "evaluation_seed_values": sorted({
                _int(row.get("scenario_seed"))
                for row in eval_seed_rows
                if _int(row.get("scenario_seed")) >= 0
            }),
            "scenario_seed_policy": "taken from actual evaluation rows; no new seed selected for plotting",
        },
        "checkpoints": checkpoints,
        "scenario_config": _scenario_snapshot(scenario),
        "run_summary": {
            "status": top_summary.get("status"),
            "training_performed": top_summary.get("training_performed"),
            "ppo_updates": top_summary.get("ppo_updates"),
            "reward_or_environment_modified": top_summary.get("reward_or_environment_modified"),
            "dynamic_obstacles_used": top_summary.get("dynamic_obstacles_used"),
            "bc_gates": top_summary.get("bc_gates"),
            "contracts": bc_summary.get("contracts"),
        },
        "model_contract": {
            "observation_dimension": "actual checkpoint migration and scenario metadata retained in source run summary",
            "encoder_and_gat": "read from source metadata where present; visualization did not construct or change a network",
            "trainable_groups_during_source_run": "Actor-only BC in source run; critic, normalizer and log_std frozen; PPO not run",
            "visualization_updates": 0,
        },
        "source_files_sha256_at_visualization": {
            str(path.relative_to(_repo_root())): _sha256(path) for path in algorithm_files if path.is_file()
        },
        "provenance_notes": [
            "Current rerun was executed through the existing external-recovery entry point in a new independent output directory.",
            "The rerun recorded ppo_updates=0; the loss figure uses the original baseline MAPPO/PPO training log.",
            "Figures and CSVs are diagnostic artifacts and are not automatically manuscript evidence.",
        ],
    }


def _metric_text(value: Any, digits: int = 3) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _role_metric_table(
    episode_rows: Sequence[Mapping[str, Any]],
    kind: str,
    split: str = "test",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        rows = [
            row for row in episode_rows
            if str(row.get("policy", "")) == policy
            and str(row.get("kind", "")) == kind
            and str(row.get("split", "")) == split
        ]
        if not rows:
            continue
        total = len(rows)
        output.append({
            "policy": policy,
            "label": POLICY_LABELS.get(policy, policy),
            "episodes": total,
            "entered_8m": sum(_bool(row.get("entered_8m")) for row in rows),
            "entered_5m": sum(_bool(row.get("entered_5m")) for row in rows),
            "entered_3m": sum(_bool(row.get("entered_3m")) for row in rows),
            "entered_1m": sum(_bool(row.get("entered_1m")) for row in rows),
            "stable_goal": sum(_bool(row.get("stable_goal")) for row in rows),
            "passed_obstacle": sum(_bool(row.get("passed_obstacle")) for row in rows),
            "post_bypass_recovery": sum(_bool(row.get("post_bypass_recovery")) for row in rows),
            "mean_min_distance": _mean(row.get("minimum_goal_distance_m") for row in rows),
            "mean_speed_at_8m": _mean(row.get("mean_speed_at_8m_entry_mps") for row in rows),
            "mean_speed_at_5m": _mean(row.get("mean_speed_at_5m_entry_mps") for row in rows),
            "mean_speed_at_3m": _mean(row.get("mean_speed_at_3m_entry_mps") for row in rows),
            "mean_speed_at_1m": _mean(row.get("mean_speed_at_1m_entry_mps") for row in rows),
            "mean_speed_within_1m": _mean(row.get("mean_speed_within_1m_mps") for row in rows),
            "mean_hold": _mean(row.get("maximum_consecutive_goal_hold_steps") for row in rows),
            "max_hold": max((_int(row.get("maximum_consecutive_goal_hold_steps"), 0) for row in rows), default=0),
            "hold_ge_1": sum(_bool(row.get("hold_steps_ge_1")) for row in rows),
            "hold_ge_3": sum(_bool(row.get("hold_steps_ge_3")) for row in rows),
            "hold_ge_5": sum(_bool(row.get("hold_steps_ge_5")) for row in rows),
            "collision": sum(_bool(row.get("collision")) for row in rows),
            "safety_steps": sum(_int(row.get("safety_projection_steps"), 0) for row in rows),
            "boundary_steps": sum(_int(row.get("boundary_projection_steps"), 0) for row in rows),
            "pre_risk_steps": sum(_int(row.get("pre_candidate_static_unsafe_steps"), 0) for row in rows),
        })
    return output


def _format_metric_table(table: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Policy | Episodes | 8m | 5m | 3m | 1m | Stable | Min distance (m) | Mean max hold | Hold >=1 | Hold >=3 | Hold >=5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        total = int(row["episodes"])
        lines.append(
            f"| {row['label']} | {total} | "
            f"{row['entered_8m']}/{total} | {row['entered_5m']}/{total} | "
            f"{row['entered_3m']}/{total} | {row['entered_1m']}/{total} | "
            f"{row['stable_goal']}/{total} | {_metric_text(row['mean_min_distance'])} | "
            f"{_metric_text(row['mean_hold'])} | {row['hold_ge_1']}/{total} | "
            f"{row['hold_ge_3']}/{total} | {row['hold_ge_5']}/{total} |"
        )
    return "\n".join(lines)


def _reward_definition(
    mappo_run: Path,
    baseline_data_root: Path,
    out_dirs: Sequence[Path],
) -> dict[str, Any]:
    episode_path = mappo_run / "episode_metrics.csv"
    episode_rows = _read_csv(episode_path, {"episode_id", "split", "episode_return"})
    train_rows = [row for row in episode_rows if str(row.get("split", "")) == "train"]
    differences: list[float] = []
    missing = 0
    for row in train_rows:
        episode_id = str(row.get("episode_id", ""))
        reward_path = baseline_data_root / "scale_1" / "train" / episode_id / "rewards.csv"
        if not reward_path.is_file():
            missing += 1
            continue
        reward_rows = _read_csv(reward_path, {"reward_total"})
        step_sum = float(np.sum([_number(item.get("reward_total")) for item in reward_rows]))
        differences.append(abs(step_sum - _number(row.get("episode_return"))))
    max_difference = max(differences, default=float("nan"))
    text = (
        "# Reward definition\n\n"
        "The episode-return curve uses the baseline training episode_metrics.csv "
        "field episode_return for split=train. It is the sum of step rewards and "
        "is not cumulatively summed across episodes. The cumulative mean is only "
        "a display statistic.\n\n"
        f"- episode source: {episode_path.resolve()}\n"
        f"- baseline trace root: {baseline_data_root.resolve()}\n"
        f"- traces checked: {len(differences)}\n"
        f"- missing reward traces: {missing}\n"
        f"- maximum absolute difference between episode_return and step-reward sum: {_metric_text(max_difference, 9)}\n"
        "- MAPPO loss is read from trainer training_metrics.csv and is never replaced by BC MSE.\n"
    )
    for directory in out_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reward_definition.md").write_text(text, encoding="utf-8")
    return {
        "episode_source": str(episode_path.resolve()),
        "checked": len(differences),
        "missing": missing,
        "max_abs_difference": max_difference,
    }


def _integrity_audit(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    required_policies: Sequence[str],
) -> dict[str, Any]:
    episode_groups: Counter[tuple[str, str]] = Counter(
        (str(row.get("policy", "")), str(row.get("kind", ""))) for row in episode_rows
    )
    step_groups = _group_steps(step_rows)
    continuity_issues: list[dict[str, Any]] = []
    for key, rows in step_groups.items():
        observed = [_int(row.get("step"), -1) for row in rows]
        if observed != list(range(len(observed))):
            continuity_issues.append({"key": key, "first_steps": observed[:10], "row_count": len(observed)})
    static = [row for row in episode_rows if str(row.get("kind", "")) == "static_obstacle"]
    return {
        "episode_groups": {f"{policy}/{kind}": count for (policy, kind), count in sorted(episode_groups.items())},
        "step_group_count": len(step_groups),
        "step_continuity_issues": continuity_issues,
        "same_static_scenario_pairing": same_scenario_pairing(static, required_policies),
        "episode_count": len(episode_rows),
        "step_count": len(step_rows),
    }


def _diagnostic_evidence(
    episode_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    dataset_counts: Sequence[Mapping[str, Any]],
    scatter: Mapping[str, Any],
    raw_applied: Mapping[str, Any],
    funnel: Mapping[str, Any],
) -> dict[str, Any]:
    static_bc = _static_episode_rows(episode_rows, "bc_actor")
    static_teacher = _static_episode_rows(episode_rows, "oracle_teacher")
    bc_steps = [
        row for row in step_rows
        if str(row.get("policy", "")) == "bc_actor" and str(row.get("kind", "")) == "static_obstacle"
    ]
    post_pass = [row for row in bc_steps if _bool(row.get("passed_obstacle_before")) or _bool(row.get("passed_obstacle_after"))]
    recovery_rows = [row for row in post_pass if _phase_label(row) in {"post_bypass", "recovery"}]
    actor_radial = [_action_components(row, "raw_actor_delta_v")[0] for row in recovery_rows]
    teacher_radial = [_action_components(row, "teacher_delta_v")[0] for row in recovery_rows]
    tangential_velocity = [_number(row.get("tangential_speed_after_mps")) for row in recovery_rows]
    radial_values = [_number(row.get("raw_radial_action_projection_mps")) for row in bc_steps]
    radial_values = [value for value in radial_values if np.isfinite(value)]
    negative_fraction = _rate(sum(value < 0.0 for value in radial_values), len(radial_values))
    phase_totals: Counter[str] = Counter()
    distance_totals: Counter[str] = Counter()
    for row in dataset_counts:
        phase_totals[str(row["phase"])] += int(row["count"])
        distance_totals[str(row["distance_bin"])] += int(row["count"])
    dataset_total = sum(distance_totals.values())
    top_distance = max(distance_totals.items(), key=lambda item: item[1], default=("", 0))
    funnel_rows = funnel.get("rows", [])
    drops = []
    for first, second in zip(funnel_rows, funnel_rows[1:]):
        if first["policy"] == second["policy"]:
            drops.append({
                "policy": first["policy"],
                "from": first["stage"],
                "to": second["stage"],
                "drop": int(first["episode_count"]) - int(second["episode_count"]),
            })
    return {
        "bc_static_episodes": len(static_bc),
        "teacher_static_episodes": len(static_teacher),
        "bc_static_passed": sum(_bool(row.get("passed_obstacle")) for row in static_bc),
        "bc_static_recovery": sum(_bool(row.get("post_bypass_recovery")) for row in static_bc),
        "bc_static_entered_1m": sum(_bool(row.get("entered_1m")) for row in static_bc),
        "bc_static_stable": sum(_bool(row.get("stable_goal")) for row in static_bc),
        "bc_post_pass_recovery_rows": len(recovery_rows),
        "bc_post_pass_mean_radial_action": _mean(actor_radial),
        "teacher_post_pass_mean_radial_action": _mean(teacher_radial),
        "bc_post_pass_mean_tangential_velocity": _mean(tangential_velocity),
        "negative_radial_action_fraction": negative_fraction,
        "bc_static_safety_projection_steps": sum(_int(row.get("safety_projection_steps"), 0) for row in static_bc),
        "bc_static_boundary_projection_steps": sum(_int(row.get("boundary_projection_steps"), 0) for row in static_bc),
        "bc_static_pre_risk_steps": sum(_int(row.get("pre_candidate_static_unsafe_steps"), 0) for row in static_bc),
        "dataset_distance_total": dataset_total,
        "dataset_top_distance_bin": top_distance[0],
        "dataset_top_distance_fraction": _rate(top_distance[1], dataset_total),
        "dataset_phase_totals": dict(phase_totals),
        "dataset_distance_totals": dict(distance_totals),
        "largest_funnel_drop": max(drops, key=lambda item: item["drop"], default=None),
        "scatter_summary": scatter.get("summary", []),
        "raw_applied": dict(raw_applied),
    }


def _report_text(
    *,
    manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
    reward: Mapping[str, Any],
    loss: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    rollout: Mapping[str, Any],
    scatter: Mapping[str, Any],
    raw_applied: Mapping[str, Any],
    dataset: Mapping[str, Any],
    funnel: Mapping[str, Any],
    evidence: Mapping[str, Any],
    normal_table: Sequence[Mapping[str, Any]],
    static_table: Sequence[Mapping[str, Any]],
    input_root: Path,
    dagger_root: Path,
) -> str:
    largest_drop = evidence.get("largest_funnel_drop")
    scatter_lines = []
    for item in evidence.get("scatter_summary", []):
        if item.get("metric") == "radial_action" and item.get("distance_bin") in {"5–3m", "3–1m", "≤1m"}:
            scatter_lines.append(
                f"{item.get('distance_bin')}: n={item.get('n')}, MAE={_metric_text(item.get('mae'))}, "
                f"actor/teacher magnitude ratio={_metric_text(item.get('mean_actor_teacher_magnitude_ratio'))}"
            )
    missing = [
        name for name, value in {
            "reward": reward.get("available"),
            "MAPPO loss": loss.get("available"),
            "trajectory": trajectory.get("available"),
            "rollout": rollout.get("available"),
            "Teacher/Actor scatter": scatter.get("available"),
            "raw/applied": raw_applied.get("available"),
            "dataset": dataset.get("available"),
            "funnel": funnel.get("available"),
        }.items() if not value
    ]
    return f"""# Current single-UAV visualization diagnostic report

## Scope and provenance

This is a diagnostic-only reproduction and visualization run. It reads the new
external-recovery rerun at {input_root.resolve()} and its evaluation data at
{dagger_root.resolve()}. It does not modify reward, environment, network,
checkpoint, PPO/MAPPO, DAgger, BC, or Teacher logic.

- git commit: {manifest.get("git", {}).get("commit")}
- branch: {manifest.get("git", {}).get("branch")}
- current rerun PPO updates: {manifest.get("run_summary", {}).get("ppo_updates")}
- current rerun training_performed: {manifest.get("run_summary", {}).get("training_performed")}
- baseline MAPPO/PPO source: {loss.get("source")}
- evaluation seed values: {manifest.get("randomness", {}).get("evaluation_seed_values")}
- historical directories overwritten: no

The project did not contain the optional docs/current_state.md,
docs/next_task.md, or docs/claim_gate.md files at audit time.

## Reproduction integrity

- episode rows: {integrity.get("episode_count")}
- step rows: {integrity.get("step_count")}
- step groups: {integrity.get("step_group_count")}
- step-index continuity issues: {len(integrity.get("step_continuity_issues", []))}
- static same-scenario pairing: {integrity.get("same_static_scenario_pairing", {}).get("paired")}
- static pairing intersection: {integrity.get("same_static_scenario_pairing", {}).get("intersection_count")} scenarios
- reward-return checks: {reward.get("checked")} traces, missing {reward.get("missing")}, max absolute difference {_metric_text(reward.get("max_abs_difference"), 9)}
- requested plots unavailable because of missing fields: {", ".join(missing) if missing else "none"}

## Unified normal far-distance test comparison

{_format_metric_table(normal_table)}

## Unified static holdout comparison

{_format_metric_table(static_table)}

The funnel is a sequential intersection, so its stage counts are non-increasing;
it is not a set of unrelated marginal event counts.

## Figure/data mapping

1. 01_cumulative_reward_curve.png uses raw baseline train episode returns and cumulative mean.
2. 02_mappo_training_loss.png uses real baseline trainer metrics, one row per PPO update; it is not BC MSE.
3. 03_xy_trajectory_overview.png uses three policies on one paired static scenario with obstacle, goal, boundary, and event markers.
4. 04_rollout_diagnostics.png uses one BC episode that passed/recovered but did not enter 1 m where such an episode exists.
5. 05_teacher_vs_actor_action_scatter.png uses finite Teacher/Actor values attached to the same BC evaluation observation record.
6. 06_raw_vs_applied_action.png shows raw, supervisor/boundary/applied action paths and correction norm.
7. 07_dataset_distance_phase_heatmap*.png uses actual demonstration split, collection mode, phase and distance bins.
8. 08_task_completion_funnel.png shows BC Actor and Oracle/Teacher sequential static-holdout stages.

## Current behavior diagnosis

- obstacle bypass: BC passed obstacle {evidence.get("bc_static_passed")}/{evidence.get("bc_static_episodes")} and post-bypass recovery {evidence.get("bc_static_recovery")}/{evidence.get("bc_static_episodes")}.
- post-bypass radial return: mean Actor radial action {_metric_text(evidence.get("bc_post_pass_mean_radial_action"))} m/s versus Teacher {_metric_text(evidence.get("teacher_post_pass_mean_radial_action"))} m/s; negative radial-action fraction={_metric_text(evidence.get("negative_radial_action_fraction"))}.
- post-bypass lateral-persistence proxy: mean tangential velocity {_metric_text(evidence.get("bc_post_pass_mean_tangential_velocity"))} m/s.
- execution intervention: BC static safety steps={evidence.get("bc_static_safety_projection_steps")}, boundary steps={evidence.get("bc_static_boundary_projection_steps")}, pre-risk unsafe steps={evidence.get("bc_static_pre_risk_steps")}.
- largest funnel drop: {largest_drop}

Teacher/Actor radial-action comparison in terminal bins:

{chr(10).join("- " + line for line in scatter_lines) if scatter_lines else "- No finite terminal-bin comparison was available."}

Dataset distance totals: {evidence.get("dataset_distance_totals")}.
Largest distance-bin share: {evidence.get("dataset_top_distance_bin")} at {_metric_text(evidence.get("dataset_top_distance_fraction"), 1)}.
Phase totals: {evidence.get("dataset_phase_totals")}.

## Answers to requested questions

1. Episode return first/last: {_metric_text(reward.get("first_return"))} to {_metric_text(reward.get("last_return"))}; mean {_metric_text(reward.get("mean_return"))}. Raw points are retained.
2. MAPPO loss: {loss.get("rows")} trainer updates, fields {loss.get("fields")}; source is baseline history, not the current BC/DAgger rerun.
3. BC failure location: static entered 1 m {evidence.get("bc_static_entered_1m")}/{evidence.get("bc_static_episodes")}; stable {evidence.get("bc_static_stable")}/{evidence.get("bc_static_episodes")}.
4. Failure category: the direct measurable candidates are bypass, post-bypass radial recovery, lateral persistence and execution intervention; see the values above rather than treating any proxy as causal.
5. Teacher/Actor radial differences are in the scatter CSV by distance bin.
6. Action shrinkage is indicated by Actor/Teacher magnitude ratios below 1 in the scatter CSV; no statistical threshold is implied.
7. Static success attribution requires reading raw/applied correction and projection counts together; nonzero correction prevents Actor-only attribution.
8. Dataset dominance is quantified by the distance and phase totals above.
9. Funnel maximum drop: {largest_drop}.
10. The next evidence-led issues are to preserve far-distance navigation, reduce measured Actor/Teacher mismatch in the failing distance bins, and cover under-represented transition/recovery states before changing reward or safety semantics.

## Evidence boundary

This single-seed fixed-holdout output is a local diagnostic dashboard. It does
not establish multi-seed generalization, causal superiority, or manuscript-level
evidence. Missing optional fields and pairing caveats are recorded in the
manifest and output CSV files.
"""


def run_visualization(
    *,
    input_run: str | Path,
    output_dir: str | Path,
    log_output_dir: str | Path | None = None,
    data_run: str | Path | None = None,
    mappo_run: str | Path | None = None,
    baseline_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete read-only dashboard generation."""

    input_root, dagger_root = _resolve_input_run(Path(input_run))
    data_root = _infer_data_root(input_root, Path(data_run) if data_run else None)
    if mappo_run is None:
        mappo_root = _repo_root() / "logs" / "preliminary" / "single_uav_braking_progress025_seed2026"
    else:
        mappo_root = Path(mappo_run).resolve()
    if baseline_checkpoint is None:
        baseline_path = (
            _repo_root() / "data" / "synthetic" / "single_uav_braking_progress025_seed2026"
            / "scale_1" / "checkpoints" / "final.pt"
        )
    else:
        baseline_path = Path(baseline_checkpoint).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    output_root = Path(output_dir).resolve()
    log_root = Path(log_output_dir).resolve() if log_output_dir else (
        _repo_root() / "logs" / "visualization" / output_root.name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    top_summary = _read_json(input_root / "summary.json") if (input_root / "summary.json").is_file() else {}
    bc_summary = _read_json(dagger_root / "summary.json") if (dagger_root / "summary.json").is_file() else {}
    episode_path = dagger_root / "evaluation_episode_metrics.csv"
    step_path = dagger_root / "evaluation_step_metrics.csv"
    episode_rows = _read_csv(episode_path, EPISODE_FIELDS)
    step_rows = _read_csv(step_path, STEP_FIELDS)
    demonstration_path = dagger_root / "demonstration_step_audit.csv"
    demonstration_rows = _read_csv(
        demonstration_path,
        {"split", "episode_id", "scenario_hash", "scenario_seed", "noise_seed", "round_index", "pair_id", "kind", "phase", "base_phase", "collection_mode", "student_closed_loop", "teacher_query", "goal_distance_m", "speed_mps"},
    ) if demonstration_path.is_file() else []
    scenario, _trace = _config_from_trace(episode_rows, data_root)

    manifest = _manifest(
        input_root=input_root,
        dagger_root=dagger_root,
        data_root=data_root,
        mappo_run=mappo_root,
        baseline_checkpoint=baseline_path,
        top_summary=top_summary,
        bc_summary=bc_summary,
        scenario=scenario,
    )
    _write_json(output_root / "visualization_run_manifest.json", manifest)
    _write_json(log_root / "visualization_run_manifest.json", manifest)

    reward_dir = output_root / "01_reward"
    loss_dir = output_root / "02_mappo_loss"
    trajectory_dir = output_root / "03_trajectory"
    rollout_dir = output_root / "04_rollout_diagnostics"
    scatter_dir = output_root / "05_teacher_actor"
    safety_dir = output_root / "06_safety"
    dataset_dir = output_root / "07_dataset"
    funnel_dir = output_root / "08_funnel"

    reward_result = _plot_reward(mappo_root, reward_dir, baseline_path)
    loss_result = _plot_losses(mappo_root, loss_dir, baseline_path)
    trajectory_result = _plot_trajectory(episode_rows, step_rows, data_root, trajectory_dir)
    rollout_result = _plot_rollout(episode_rows, step_rows, data_root, rollout_dir)
    scatter_result = _plot_action_scatter(step_rows, scatter_dir)
    raw_applied_result = _plot_raw_applied(episode_rows, step_rows, safety_dir)
    dataset_result = _plot_dataset_heatmaps(demonstration_rows, dataset_dir)
    funnel_result = _plot_funnel(episode_rows, funnel_dir)

    baseline_data_root = _repo_root() / "data" / "synthetic" / mappo_root.name
    reward_definition = _reward_definition(mappo_root, baseline_data_root, (output_root, log_root))
    integrity = _integrity_audit(episode_rows, step_rows, ("baseline_rl", "bc_actor", "oracle_teacher"))
    normal_table = _role_metric_table(episode_rows, "normal_far_distance")
    static_table = _role_metric_table(episode_rows, "static_obstacle")
    evidence = _diagnostic_evidence(
        episode_rows,
        step_rows,
        phase_distance_counts(demonstration_rows),
        scatter_result,
        raw_applied_result,
        funnel_result,
    )
    report = _report_text(
        manifest=manifest,
        integrity=integrity,
        reward=reward_definition | reward_result,
        loss=loss_result,
        trajectory=trajectory_result,
        rollout=rollout_result,
        scatter=scatter_result,
        raw_applied=raw_applied_result,
        dataset=dataset_result,
        funnel=funnel_result,
        evidence=evidence,
        normal_table=normal_table,
        static_table=static_table,
        input_root=input_root,
        dagger_root=dagger_root,
    )
    (output_root / "visualization_diagnostic_report.md").write_text(report, encoding="utf-8")
    (log_root / "visualization_diagnostic_report.md").write_text(report, encoding="utf-8")
    result = {
        "input_run": str(input_root),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "log_output_root": str(log_root),
        "manifest": str((output_root / "visualization_run_manifest.json").resolve()),
        "report": str((output_root / "visualization_diagnostic_report.md").resolve()),
        "reward_definition": reward_definition,
        "integrity": integrity,
        "figures": {
            "reward": reward_result,
            "mappo_loss": loss_result,
            "trajectory": trajectory_result,
            "rollout": rollout_result,
            "teacher_actor": scatter_result,
            "raw_applied": raw_applied_result,
            "dataset": dataset_result,
            "funnel": funnel_result,
        },
        "normal_table": normal_table,
        "static_table": static_table,
        "evidence": evidence,
    }
    _write_json(output_root / "visualization_summary.json", result)
    _write_json(log_root / "visualization_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only single-UAV visualization diagnostic")
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-output-dir", type=Path)
    parser.add_argument("--data-run", type=Path)
    parser.add_argument("--mappo-run", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    args = parser.parse_args(argv)
    result = run_visualization(
        input_run=args.input_run,
        output_dir=args.output_dir,
        log_output_dir=args.log_output_dir,
        data_run=args.data_run,
        mappo_run=args.mappo_run,
        baseline_checkpoint=args.baseline_checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DISTANCE_BIN_ORDER",
    "aggregate_episode_returns",
    "funnel_counts",
    "main",
    "parse_mappo_loss_rows",
    "phase_distance_counts",
    "radial_tangential",
    "run_visualization",
    "same_scenario_pairing",
]


def _legacy_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    title: str,
    *,
    normalized: bool,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="Blues" if not normalized else "YlGnBu")
    axis.set_xticks(np.arange(len(DISTANCE_BIN_ORDER)))
    axis.set_xticklabels(DISTANCE_BIN_ORDER)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xlabel("Goal-distance bin")
    axis.set_ylabel("Observed phase")
    axis.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_value = f"{value:.1%}" if normalized else f"{int(value)}"
            axis.text(j, i, text_value, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Fraction" if normalized else "Sample count")
    _save_figure(figure, path)


def _legacy_plot_dataset_heatmaps(
    demonstration_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    counts = phase_distance_counts(demonstration_rows)
    csv_path = out_dir / "07_dataset_distance_phase_counts.csv"
    _write_csv(csv_path, counts, ("split", "collection_mode", "phase", "distance_bin", "count", "denominator", "fraction"))
    phases = sorted({str(row["phase"]) for row in counts})
    modes = sorted({str(row["collection_mode"]) for row in counts})
    figure, axes = plt.subplots(
        max(1, len(phases) and 2), max(1, len(modes)), figsize=(16, 12), constrained_layout=True
    )
    axes_array = np.asarray(axes, dtype=object).reshape(max(1, len(phases) and 2), max(1, len(modes)))
    # The first figure is intentionally a compact 2-row split x mode view.
    split_names = ("train", "validation")
    if not modes:
        modes = [""]
    for split_index, split in enumerate(split_names):
        for mode_index, mode in enumerate(modes):
            rows_subset = [row for row in counts if row["split"] == split and row["collection_mode"] == mode]
            local_phases = sorted({str(row["phase"]) for row in rows_subset}) or phases
            matrix = np.zeros((len(local_phases), len(DISTANCE_BIN_ORDER)))
            for row in rows_subset:
                matrix[local_phases.index(str(row["phase"])), DISTANCE_BIN_ORDER.index(str(row["distance_bin"]))] = float(row["count"])
            axis = axes_array[split_index, mode_index]
            image = axis.imshow(matrix, aspect="auto", cmap="Blues")
            axis.set_xticks(np.arange(len(DISTANCE_BIN_ORDER)))
            axis.set_xticklabels(DISTANCE_BIN_ORDER, rotation=20)
            axis.set_yticks(np.arange(len(local_phases)))
            axis.set_yticklabels(local_phases)
            axis.set_title(f"{split} | {mode or 'unknown mode'}")
            axis.set_xlabel("Distance bin")
            axis.set_ylabel("Phase")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Dataset distance × phase counts", fontsize=16)
    _save_figure(figure, out_dir / "07_dataset_distance_phase_heatmap.png")

    normalized_path = out_dir / "07_dataset_distance_phase_heatmap_normalized.png"
    figure, axes = plt.subplots(
        max(1, len(phases) and 2), max(1, len(modes)), figsize=(16, 12), constrained_layout=True
    )
    axes_array = np.asarray(axes, dtype=object).reshape(max(1, len(phases) and 2), max(1, len(modes)))
    for split_index, split in enumerate(split_names):
        for mode_index, mode in enumerate(modes):
            rows_subset = [row for row in counts if row["split"] == split and row["collection_mode"] == mode]
            local_phases = sorted({str(row["phase"]) for row in rows_subset}) or phases
            matrix = np.zeros((len(local_phases), len(DISTANCE_BIN_ORDER)))
            denominator = sum(float(row["count"]) for row in rows_subset)
            for row in rows_subset:
                matrix[local_phases.index(str(row["phase"])), DISTANCE_BIN_ORDER.index(str(row["distance_bin"]))] = (
                    float(row["count"]) / denominator if denominator else 0.0
                )
            axis = axes_array[split_index, mode_index]
            image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.01, float(np.max(matrix)) if matrix.size else 0.01))
            axis.set_xticks(np.arange(len(DISTANCE_BIN_ORDER)))
            axis.set_xticklabels(DISTANCE_BIN_ORDER, rotation=20)
            axis.set_yticks(np.arange(len(local_phases)))
            axis.set_yticklabels(local_phases)
            axis.set_title(f"{split} | {mode or 'unknown mode'}")
            axis.set_xlabel("Distance bin")
            axis.set_ylabel("Phase")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center", fontsize=7)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Dataset distance × phase proportions", fontsize=16)
    _save_figure(figure, normalized_path)
    return {
        "available": bool(counts),
        "rows": len(counts),
        "splits": sorted({str(row["split"]) for row in counts}),
        "collection_modes": modes,
        "phases": phases,
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "07_dataset_distance_phase_heatmap.png").resolve()),
        "normalized_plot": str(normalized_path.resolve()),
    }


def _legacy_plot_funnel(
    episode_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    static_rows = [row for row in episode_rows if str(row.get("kind", "")) == "static_obstacle" and str(row.get("split", "")) == "test"]
    policies = ("bc_actor", "oracle_teacher")
    rows: list[dict[str, Any]] = []
    for policy in policies:
        rows.extend(funnel_counts(static_rows, policy))
    csv_path = out_dir / "08_task_completion_funnel.csv"
    _write_csv(csv_path, rows, ("policy", "stage", "episode_count", "rate"))
    figure, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    stages = [row["stage"] for row in funnel_counts(static_rows, "bc_actor")]
    x = np.arange(len(stages))
    width = 0.38
    for offset, policy in ((-width / 2, "bc_actor"), (width / 2, "oracle_teacher")):
        values = [row["episode_count"] for row in funnel_counts(static_rows, policy)]
        axis.bar(x + offset, values, width=width, color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
        for pos, value in zip(x + offset, values):
            axis.text(pos, value + 0.2, str(value), ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(stages, rotation=25, ha="right")
    axis.set_ylabel("Episodes (sequential intersection)")
    axis.set_title("Static holdout task-completion funnel")
    axis.set_ylim(0, max(1, len(static_rows) + 3))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    _save_figure(figure, out_dir / "08_task_completion_funnel.png")
    return {
        "available": bool(static_rows),
        "episodes_per_policy": {policy: len(_static_episode_rows(episode_rows, policy)) for policy in policies},
        "rows": rows,
        "csv": str(csv_path.resolve()),
        "plot": str((out_dir / "08_task_completion_funnel.png").resolve()),
    }


def _legacy_config_from_trace(episode_rows: Sequence[Mapping[str, Any]], data_root: Path) -> tuple[dict[str, Any], Path | None]:
    static = [row for row in episode_rows if str(row.get("kind", "")) == "static_obstacle"]
    if not static:
        static = list(episode_rows)
    for row in static:
        path = _writer_path(row, data_root)
        if path is not None:
            scenario = _load_scenario(path)
            if scenario:
                return scenario, path
    return {}, None


def _legacy_manifest(
    *,
    input_root: Path,
    dagger_root: Path,
    data_root: Path,
    mappo_run: Path,
    baseline_checkpoint: Path,
    top_summary: Mapping[str, Any],
    bc_summary: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    algorithm_files = [
        _repo_root() / "src" / "experiments" / "single_uav_external_recovery_method_study.py",
        _repo_root() / "src" / "experiments" / "single_uav_static_obstacle_dagger_recovery_goal.py",
        _repo_root() / "src" / "simulation" / "environment.py",
        _repo_root() / "src" / "simulation" / "motion_model.py",
    ]
    explicit = [baseline_checkpoint]
    for path in (
        data_root / "dagger_bc" / "bc_final" / "scale_1" / "checkpoints" / "best_validation.pt",
        data_root / "dagger_bc" / "bc_round0" / "scale_1" / "checkpoints" / "best_validation.pt",
    ):
        explicit.append(path)
    checkpoints = _snapshot_checkpoints((top_summary, bc_summary), explicit)
    return {
        "visualization_task": "single_uav_current_run_visualization",
        "run_id": input_root.name,
        "input_run": str(input_root.resolve()),
        "dagger_log_root": str(dagger_root.resolve()),
        "data_root": str(data_root.resolve()),
        "mappo_training_run": str(mappo_run.resolve()),
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "git": {
            "commit": _subprocess(("git", "rev-parse", "HEAD")),
            "branch": _subprocess(("git", "branch", "--show-current")),
            "status": _subprocess(("git", "status", "--short", "--untracked-files=no")).splitlines(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "randomness": {
            "evaluation_seed_values": sorted({
                _int(row.get("scenario_seed"))
                for row in _read_csv(dagger_root / "evaluation_episode_metrics.csv", {"scenario_seed"})
                if _int(row.get("scenario_seed")) >= 0
            }),
            "scenario_seed_policy": "taken from actual evaluation episode rows; no new seed was selected for plotting",
        },
        "checkpoints": checkpoints,
        "scenario_config": _scenario_snapshot(scenario),
        "run_summary": {
            "status": top_summary.get("status"),
            "training_performed": top_summary.get("training_performed"),
            "ppo_updates": top_summary.get("ppo_updates"),
            "reward_or_environment_modified": top_summary.get("reward_or_environment_modified"),
            "dynamic_obstacles_used": top_summary.get("dynamic_obstacles_used"),
            "bc_gates": top_summary.get("bc_gates"),
            "contracts": bc_summary.get("contracts"),
        },
        "model_contract": {
            "observation_dimension": "from actual checkpoint/scenario metadata; node feature dimension is recorded in migration summary",
            "encoder_and_gat": "from actual checkpoint metadata; no network was constructed or changed by visualization",
            "trainable_groups_during_source_run": "Actor-only BC in source run; critic, normalizer and log_std frozen; PPO not run",
            "visualization_updates": 0,
        },
        "source_files_sha256_at_visualization": {
            str(path.relative_to(_repo_root())): _sha256(path) for path in algorithm_files if path.is_file()
        },
        "provenance_notes": [
            "The current rerun was executed through the existing external-recovery entry point in an independent output directory.",
            "The rerun recorded ppo_updates=0; the loss figure therefore uses the original baseline MAPPO/PPO training log.",
            "All requested visualization outputs are diagnostic artifacts, not automatically admitted manuscript evidence.",
        ],
    }
