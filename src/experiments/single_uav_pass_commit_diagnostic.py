"""Prepare the frozen PASS_COMMIT causal diagnostic; human runs are opt-in.

The preparation path reads only existing logs/manifests/checkpoints and pure
geometry.  The evaluator is imported lazily only from the explicit human-run
entry point.
"""

from __future__ import annotations

if __package__ in {None, ""}:  # PyCharm "Run File" support.
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "src.experiments"

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..control.pass_commit_non_regression import (
    PASS_COMMIT_HARD_CAP_MPS,
    PASS_COMMIT_TRIGGER_DEADBAND_M,
    PassCommitNonRegression,
)


STAGE_NAME = "single_uav_pass_commit_diagnostic"
PARENT_RADIAL_INTENT_PREREGISTRATION_HASH = (
    "bf70fb95188ca73c1ba87997a090d22331a3ac76a27daaaea1d86da09f59e40b"
)
PREPARATION_ROOT = Path(
    "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/preparation_action_decomposition_fix_v1"
)
PREVIOUS_PREPARATION_ROOT = Path(
    "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/preparation"
)
PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH = (
    "74b999edb1158fc960374025f418d33c7812a57653c07f6d0e73fab8b69e02ac"
)
EXPECTED_OFF_BASELINE_FINGERPRINT = (
    "de53e383264db1c46ea793519f7b627f3ef83fbb447ca33d9d44cf72a603c97e"
)
PASS_COMMIT_ATTEMPT_ID = "action_decomposition_fix_v1"
PREREGISTRATION_FILE = "preregistered_pass_commit_endpoints.json"
PREREGISTRATION_LOCK_FILE = "pass_commit_preregistration_lock.json"
RUNTIME_SOURCE_LOCK_FILE = "pass_commit_runtime_source_lock.json"
CAP_DERIVATION_FILE = "pass_commit_cap_derivation.json"
CAP_ROWS_FILE = "pass_commit_cap_derivation_rows.csv"
OFF_BASELINE_CONTRACT_FILE = "pass_commit_off_baseline_contract.json"
PREPARATION_SUMMARY_FILE = "pass_commit_preparation_summary.json"
CANONICAL_ARTIFACT_FILES = (
    PREREGISTRATION_FILE,
    PREREGISTRATION_LOCK_FILE,
    RUNTIME_SOURCE_LOCK_FILE,
    CAP_DERIVATION_FILE,
    CAP_ROWS_FILE,
    OFF_BASELINE_CONTRACT_FILE,
    PREPARATION_SUMMARY_FILE,
)

ALL_EPISODE_IDS = tuple(f"test_r99_episode_{index:04d}" for index in range(1, 21))
MAIN_FAILURE_EPISODE_IDS = tuple(
    f"test_r99_episode_{index:04d}" for index in (2, 4, 5, 6, 11, 20)
)
RESCUE_DESCRIPTIVE_EPISODE_ID = "test_r99_episode_0012"
SAFETY_SPECIAL_EPISODE_ID = "test_r99_episode_0013"
BOUNDARY_SPECIAL_EPISODE_ID = "test_r99_episode_0014"
HARD_CAP_MPS = 0.075
PASS_BUFFER_M = 0.75
CLEARANCE_FLOOR_M = 2.5
LATERAL_EXIT_THRESHOLD_MPS = 0.5
LATERAL_EXIT_CONTIGUOUS_STEPS = 3
GEOMETRY_TOLERANCE = 1.0e-8
OFF_BASELINE_MISMATCH = "PASS_COMMIT_OFF_BASELINE_MISMATCH"
CAP_DERIVATION_FAIL_CLOSED = "PASS_COMMIT_CAP_DERIVATION_FAIL_CLOSED"
SOURCE_LOCK_MISMATCH = "PASS_COMMIT_RUNTIME_SOURCE_LOCK_MISMATCH"

PRIOR_LOG_ROOT = Path(
    "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
PRIOR_ON_STEPS = PRIOR_LOG_ROOT / "on" / "steps.csv"
PRIOR_ON_EPISODES = PRIOR_LOG_ROOT / "on" / "episodes.csv"
PRIOR_SUMMARY = PRIOR_LOG_ROOT / "evaluation_summary.json"
PRIOR_PREREGISTRATION = PRIOR_LOG_ROOT / "preregistered_endpoints.json"
SCENARIO_MANIFEST = Path(
    "data/synthetic/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026/"
    "scenario_manifest.json"
)
CHECKPOINT = Path(
    "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
    "dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt"
)
SOFT_BUFFER_SOURCE = Path("src/experiments/single_uav_external_recovery_method_study.py")
EVALUATOR_SOURCE = Path(
    "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
)
ASSISTANCE_SOURCE = Path("src/control/lateral_exit_braking_assistance.py")
LATERAL_DIAGNOSTIC_SOURCE = Path("src/experiments/single_uav_lateral_exit_braking_assistance.py")
SIGNED_GEOMETRY_SOURCE = Path("src/experiments/single_uav_signed_bypass_observability.py")
OBSTACLE_GEOMETRY_SOURCE = Path("src/experiments/single_uav_obstacle_dagger_stage2b.py")
ACTION_TRANSFORM_SOURCE = Path("src/experiments/single_uav_actor_constrained_oracle.py")
FORMAL_SOURCE = Path("src/experiments/formal.py")
ENVIRONMENT_SOURCE = Path("src/simulation/environment.py")
HELPER_SOURCE = Path("src/control/pass_commit_non_regression.py")
ACTION_DECOMPOSITION_SOURCE = Path("src/control/pass_commit_action_decomposition.py")
PREVIOUS_PREREGISTRATION_PATH = PREVIOUS_PREPARATION_ROOT / PREREGISTRATION_FILE
PREVIOUS_PREREGISTRATION_LOCK_PATH = PREVIOUS_PREPARATION_ROOT / PREREGISTRATION_LOCK_FILE
PREVIOUS_OFF_BASELINE_CONTRACT_PATH = PREVIOUS_PREPARATION_ROOT / OFF_BASELINE_CONTRACT_FILE
ABORTED_RUN_STATUS_PATH = Path(
    "logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run/attempt_status.json"
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric field {field}")
    return result


def _vector(value: Any, *, field: str) -> tuple[float, float, float]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        result = tuple(float(item) for item in decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid vector field {field}") from exc
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"invalid vector field {field}")
    return result  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in a))


def _normal(value: Any) -> Any:
    if value is None or value == "":
        return {"__scalar__": "none" if value is None else "empty"}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return _normal(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        if math.isnan(number):
            return {"__float__": "nan"}
        if math.isinf(number):
            return {"__float__": "+inf" if number > 0.0 else "-inf"}
        return number
    if isinstance(value, Mapping):
        return {str(key): _normal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return {"__float__": "nan"}
        if math.isinf(number):
            return {"__float__": "+inf" if number > 0.0 else "-inf"}
        return number
    if hasattr(value, "tolist"):
        return _normal(value.tolist())
    return str(value)


def _normalized_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    normalize_condition: bool,
) -> list[list[Any]]:
    output: list[list[Any]] = []
    for row in rows:
        if any(key not in row for key in keys):
            raise RuntimeError(OFF_BASELINE_MISMATCH)
        current = []
        for key in keys:
            value = "off" if key == "condition" and normalize_condition else row[key]
            current.append(_normal(value))
        output.append(current)
    return output


def _hash_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, normalize_condition: bool,
) -> str:
    payload = {"key_order": list(keys), "rows": _normalized_rows(
        rows, keys, normalize_condition=normalize_condition
    )}
    return _sha256_bytes(_canonical(payload).encode("utf-8"))


def _type7(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("type-7 quantiles require values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    h = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    fraction = h - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _spec_extent(spec: Mapping[str, Any], direction: Sequence[float]) -> float:
    shape = str(spec.get("geometry_type", "")).lower()
    if shape == "sphere":
        return _finite_float(spec["radius_m"], field="radius_m")
    if shape in {"box", "cuboid"}:
        half = tuple(float(item) / 2.0 for item in spec["size_m"])
        return sum(abs(float(direction[index])) * half[index] for index in range(3))
    raise ValueError(f"unsupported obstacle geometry {shape!r}")


def _spec_clearance(point: Sequence[float], spec: Mapping[str, Any]) -> float:
    center = tuple(float(item) for item in spec["center_m"])
    offset = tuple(float(point[index]) - center[index] for index in range(3))
    shape = str(spec.get("geometry_type", "")).lower()
    if shape == "sphere":
        return max(0.0, _norm(offset) - float(spec["radius_m"]))
    if shape in {"box", "cuboid"}:
        half = tuple(float(item) / 2.0 for item in spec["size_m"])
        closest = tuple(max(abs(offset[i]) - half[i], 0.0) for i in range(3))
        return _norm(closest)
    raise ValueError(f"unsupported obstacle geometry {shape!r}")


def _manifest_scenarios(root: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    manifest = json.loads((root / SCENARIO_MANIFEST).read_text(encoding="utf-8"))
    ordered = manifest["identity"]["ordered_scenarios"]
    if [entry["episode_id"] for entry in ordered] != list(ALL_EPISODE_IDS):
        raise RuntimeError(CAP_DERIVATION_FAIL_CLOSED)
    return {str(item["episode_id"]): item for item in ordered}, manifest


def _source_hashes(root: Path) -> dict[str, str]:
    relative = (
        EVALUATOR_SOURCE,
        HELPER_SOURCE,
        Path(__file__).resolve().relative_to(root),
        ASSISTANCE_SOURCE,
        LATERAL_DIAGNOSTIC_SOURCE,
        SIGNED_GEOMETRY_SOURCE,
        OBSTACLE_GEOMETRY_SOURCE,
        SOFT_BUFFER_SOURCE,
        ACTION_TRANSFORM_SOURCE,
        FORMAL_SOURCE,
        ENVIRONMENT_SOURCE,
        ACTION_DECOMPOSITION_SOURCE,
        PRIOR_ON_STEPS,
        PRIOR_ON_EPISODES,
        PRIOR_SUMMARY,
        PRIOR_PREREGISTRATION,
        SCENARIO_MANIFEST,
        PREVIOUS_PREREGISTRATION_PATH,
        PREVIOUS_PREREGISTRATION_LOCK_PATH,
        PREVIOUS_OFF_BASELINE_CONTRACT_PATH,
    )
    result: dict[str, str] = {}
    for path in relative:
        full = root / path
        if not full.is_file():
            raise FileNotFoundError(full)
        result[path.as_posix()] = _sha256_file(full)
    checkpoint = root / CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result["checkpoint/best_validation.pt"] = _sha256_file(checkpoint)
    return dict(sorted(result.items()))


def _goal_recovery_active(row: Mapping[str, Any]) -> bool:
    keys = (
        "goal_recovery_correction_active", "terminal_capture_active",
        "terminal_v2_governor_active", "terminal_v21_radial_conflict_active",
        "terminal_v22_restorative_active", "terminal_v23_authority_active",
        "terminal_v23_preboundary_authority_repair_active",
    )
    return any(_truth(row.get(key, "0")) for key in keys)


def derive_cap_rows(
    prior_rows: Sequence[Mapping[str, str]],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay the runtime helper chronologically against the prior lateral-ON log."""

    grouped: dict[str, list[Mapping[str, str]]] = {
        episode_id: [] for episode_id in MAIN_FAILURE_EPISODE_IDS
    }
    for row in prior_rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id in grouped:
            grouped[episode_id].append(row)
    output: list[dict[str, Any]] = []
    included_total = 0
    for episode_id in MAIN_FAILURE_EPISODE_IDS:
        episode_rows = grouped[episode_id]
        if not episode_rows:
            raise RuntimeError(f"{CAP_DERIVATION_FAIL_CLOSED}: missing {episode_id}")
        episode_rows.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in episode_rows] != list(range(len(episode_rows))):
            raise RuntimeError(f"{CAP_DERIVATION_FAIL_CLOSED}: non-contiguous {episode_id}")
        scenario = scenarios[episode_id]
        goal = tuple(float(value) for value in scenario["goal"])
        specs = {
            str(spec["obstacle_id"]): spec
            for spec in scenario["obstacle_specs"]
            if bool(spec.get("physical_collision_enabled", True))
        }
        controller = PassCommitNonRegression(enabled=True, cap_mps=HARD_CAP_MPS)
        for row in episode_rows:
            step = int(row["step"])
            relevant_id = str(row.get("relevant_obstacle_id", ""))
            frozen_id = str(row.get("lateral_exit_assist_frozen_obstacle_id", ""))
            spec = specs.get(relevant_id)
            frozen_available = _truth(row.get("lateral_exit_assist_frozen_obstacle_available", "0"))
            obstacle_available = spec is not None
            matches = bool(frozen_available and frozen_id and relevant_id == frozen_id)
            reason = ""
            position_values: tuple[float, float, float] | None = None
            try:
                if spec is None:
                    raise ValueError("manifest obstacle unavailable")
                center = tuple(float(value) for value in spec["center_m"])
                relative = _vector(row.get("obstacle_relative_vector_m", ""), field="obstacle_relative_vector_m")
                position = tuple(center[index] - relative[index] for index in range(3))
                position_values = position
                goal_delta = tuple(goal[index] - position[index] for index in range(3))
                goal_norm = _norm(goal_delta)
                if goal_norm <= 1.0e-12:
                    raise ValueError("degenerate goal ray")
                direction = tuple(value / goal_norm for value in goal_delta)
                extent = _spec_extent(spec, direction)
                clearance = _spec_clearance(position, spec)
                logged_clearance = _finite_float(
                    row["relevant_obstacle_clearance_m"], field="relevant_obstacle_clearance_m"
                )
                if abs(clearance - logged_clearance) > 1.0e-6:
                    raise ValueError("static-map clearance mismatch")
                logged_extent = _finite_float(row["signed_obstacle_extent_m"], field="signed_obstacle_extent_m")
                if abs(extent - logged_extent) > GEOMETRY_TOLERANCE:
                    raise ValueError("logged obstacle extent mismatch")
                # The prior trace's current_radial_velocity_mps is the exact
                # pre-step EKF velocity projection on the current goal unit.
                # Only this scalar projection enters the cancellation target;
                # representing it as scalar*u is therefore exact for this
                # dot product and avoids borrowing any post-step signed vector.
                radial_velocity = _finite_float(
                    row["current_radial_velocity_mps"],
                    field="current_radial_velocity_mps",
                )
                velocity = tuple(radial_velocity * direction[index] for index in range(3))
                actor = _vector(row["actor_raw_delta_v_mps"], field="actor_raw_delta_v_mps")
                frozen_lateral = _vector(
                    row["lateral_exit_assist_lateral_unit"],
                    field="lateral_exit_assist_lateral_unit",
                ) if row.get("lateral_exit_assist_lateral_unit", "") not in {"", "[]"} else (0.0, 0.0, 0.0)
                estimated_lateral = _finite_float(
                    row["estimated_lateral_velocity_mps"],
                    field="estimated_lateral_velocity_mps",
                )
                passed_margin = _finite_float(row["passed_margin_m"], field="passed_margin_m")
                along = _finite_float(row["signed_obstacle_along_goal_m"], field="signed_obstacle_along_goal_m")
                if abs(-(along + extent + PASS_BUFFER_M) - passed_margin) > GEOMETRY_TOLERANCE:
                    raise ValueError("logged passed-margin identity mismatch")
                exact_lateral_velocity = _finite_float(
                    row["estimated_lateral_velocity_mps"], field="estimated_lateral_velocity_mps"
                )
                encounter_established = bool(
                    row.get("lateral_exit_assist_encounter_step", "") != ""
                    and row.get("lateral_exit_assist_lateral_unit", "") not in {"", "[]"}
                )
                decision = controller.evaluate(
                    step=step,
                    estimated_position_m=position,
                    goal_position_m=goal,
                    obstacle_center_m=center,
                    obstacle_extent_m=extent,
                    pass_buffer_m=PASS_BUFFER_M,
                    logged_along_obstacle_m=along,
                    logged_passed_margin_m=passed_margin,
                    estimated_velocity_mps=velocity,
                    baseline_candidate_delta_v_mps=actor,
                    estimated_lateral_velocity_mps=exact_lateral_velocity,
                    encounter_context_established=encounter_established,
                    frozen_obstacle_matches=matches,
                    relevant_obstacle_available=obstacle_available,
                    relevant_obstacle_clearance_m=logged_clearance,
                    passed_obstacle=_truth(row.get("passed_obstacle", "0")),
                    passed_condition_now=_truth(row.get("passed_condition_now", "false")),
                    release_latched=_truth(row.get("passed_obstacle", "0")),
                    post_bypass_recovery=_truth(row.get("post_bypass_recovery", "0")),
                    goal_recovery_active=_goal_recovery_active(row),
                    lateral_exit_assist_active=_truth(row.get("lateral_exit_assist_active", "0")),
                    past_closest_approach=_truth(row.get("past_closest_approach", "0")),
                    clearance_increasing=_truth(
                        row.get("lateral_exit_assist_clearance_increasing_gate", "0")
                    ),
                    episode_done=_truth(row.get("terminated", "0")) or _truth(row.get("truncated", "0")),
                )
                if decision.eligible and decision.pass_progress_velocity_before_mps < 0.0:
                    reason = "included_runtime_equivalent_negative_cancellation_row"
                    required = decision.requested_delta_v_mps
                    included_total += 1
                else:
                    required = 0.0
                    reason = decision.inactive_reason
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                reason = f"excluded_geometry_or_source_invalid:{type(exc).__name__}:{exc}"
                required = 0.0
                decision = None
            output.append({
                "episode_id": episode_id,
                "step": step,
                "included": int(reason == "included_runtime_equivalent_negative_cancellation_row"),
                "inclusion_exclusion_reason": reason,
                "relevant_obstacle_id": relevant_id,
                "frozen_obstacle_id": frozen_id,
                "obstacle_matches": int(matches),
                "estimated_position_m": list(position_values) if position_values is not None else [],
                "goal_unit": list(decision.direction_unit) if decision else [],
                "pass_margin_m": decision.passed_margin_m if decision else "",
                "pass_margin_delta_m": decision.margin_delta_m if decision else "",
                "pass_margin_regression_run": decision.regression_run if decision else "",
                "pass_commit_triggered": int(decision.triggered) if decision else 0,
                "pass_commit_latched": int(decision.latched) if decision else 0,
                "estimated_lateral_exit_ready": int(decision.estimated_lateral_exit_ready) if decision else 0,
                "past_closest_approach": int(_truth(row.get("past_closest_approach", "0"))),
                "clearance_increasing_gate": int(_truth(row.get("lateral_exit_assist_clearance_increasing_gate", "0"))),
                "lateral_exit_assist_active": int(_truth(row.get("lateral_exit_assist_active", "0"))),
                "goal_recovery_active": int(_goal_recovery_active(row)),
                "relevant_obstacle_clearance_m": row.get("relevant_obstacle_clearance_m", ""),
                "pass_progress_velocity_before_mps": decision.pass_progress_velocity_before_mps if decision else "",
                "required_cancellation_mps": required,
            })
    return output


def derive_cap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    included = [
        float(row["required_cancellation_mps"])
        for row in rows if int(row["included"]) == 1
    ]
    if not included or not all(math.isfinite(value) and value > 0.0 for value in included):
        raise RuntimeError(CAP_DERIVATION_FAIL_CLOSED)
    q1 = _type7(included, 0.25)
    median = _type7(included, 0.5)
    q3 = _type7(included, 0.75)
    maximum = max(included)
    cap = min(q3, HARD_CAP_MPS)
    if not math.isfinite(cap) or cap <= 0.0:
        raise RuntimeError(CAP_DERIVATION_FAIL_CLOSED)
    per_episode = {
        episode_id: sum(
            int(row["included"] == 1 and row["episode_id"] == episode_id)
            for row in rows
        )
        for episode_id in MAIN_FAILURE_EPISODE_IDS
    }
    return {
        "schema_version": "pass_commit_cap_derivation_v1",
        "source": "prior authoritative lateral-exit braking assistance ON steps.csv",
        "population": list(MAIN_FAILURE_EPISODE_IDS),
        "eligible_row_count": len(included),
        "eligible_rows_per_episode": per_episode,
        "required_cancellation_statistics_mps": {
            "q1": q1,
            "median": median,
            "q3": q3,
            "maximum": maximum,
        },
        "quantile_method": "type-7 linear interpolation",
        "hard_maximum_diagnostic_authority_mps": HARD_CAP_MPS,
        "cap_rule": "min(Q3_required_cancellation, 0.075 m/s)",
        "pass_commit_max_delta_v_mps": cap,
        "sweep_performed": False,
        "fail_closed": True,
        "fail_closed_triggered": False,
        "zero_rows_in_one_episode_permitted": True,
    }


def _baseline_episode_contract(
    source_episodes: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    if not source_episodes:
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    keys = [key for key in source_episodes[0] if not str(key).startswith("pass_commit_")]
    required = {"condition", "episode_index", "episode_id", "scenario_seed", "noise_seed", "scenario_hash", "passed_obstacle", "post_bypass_recovery", "collision"}
    if not required.issubset(keys):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    ordered = sorted(source_episodes, key=lambda row: int(row["episode_index"]))
    if [str(row["episode_id"]) for row in ordered] != list(ALL_EPISODE_IDS):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    return keys, {
        str(row["episode_id"]): _hash_rows([row], keys, normalize_condition=True)
        for row in ordered
    }


def build_off_baseline_contract(
    *,
    prior_rows: Sequence[Mapping[str, str]],
    prior_episodes: Sequence[Mapping[str, str]],
    prior_aggregate: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    prior_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not prior_rows or not prior_episodes:
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    row_keys = [key for key in prior_rows[0] if not str(key).startswith("pass_commit_")]
    if not {"condition", "episode_index", "episode_id", "scenario_seed", "noise_seed", "step", "lateral_exit_assist_active", "passed_margin_m", "passed_obstacle", "action_transform_applied", "safety_projection_applied", "boundary_projection_applied"}.issubset(row_keys):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    ordered_rows = sorted(prior_rows, key=lambda row: (int(row["episode_index"]), int(row["step"])))
    if any(str(row.get("condition")) != "on" for row in ordered_rows):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    if [str(row["episode_id"]) for row in ordered_rows if int(row["step"]) == 0] != list(ALL_EPISODE_IDS):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    for episode_id in ALL_EPISODE_IDS:
        local = [row for row in ordered_rows if str(row["episode_id"]) == episode_id]
        if not local or [int(row["step"]) for row in local] != list(range(len(local))):
            raise RuntimeError(OFF_BASELINE_MISMATCH)
    trace_hashes = {
        episode_id: _hash_rows(
            [row for row in ordered_rows if str(row["episode_id"]) == episode_id],
            row_keys,
            normalize_condition=True,
        )
        for episode_id in ALL_EPISODE_IDS
    }
    episode_keys, outcome_hashes = _baseline_episode_contract(prior_episodes)
    aggregate_keys = [key for key in prior_aggregate if not str(key).startswith("pass_commit_")]
    aggregate_hash = _sha256_bytes(_canonical({
        "key_order": aggregate_keys,
        "values": [_normal(prior_aggregate[key]) for key in aggregate_keys],
    }).encode("utf-8"))
    summary_results = prior_summary["results"]["on"]
    provenance = summary_results["provenance"]
    scenario_order = [
        [int(row["episode_index"]), str(row["episode_id"]), int(row["scenario_seed"]),
         int(row["noise_seed"]), str(row["scenario_hash"])]
        for row in sorted(prior_episodes, key=lambda item: int(item["episode_index"]))
    ]
    identity = {
        "episode_order": list(ALL_EPISODE_IDS),
        "episode_identity": scenario_order,
        "scenario_manifest_hash": provenance["scenario_manifest_hash"],
        "ordered_pool_identity": provenance["ordered_pool_identity"],
        "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        "policy_hash": provenance["policy_hash"],
        "observation_dimension": provenance["observation_dimension"],
        "controller_config_hash": provenance["controller_config_hash"],
        "goal_recovery_config_hash": provenance["goal_recovery_config_hash"],
        "frozen_runtime_component_hashes": prior_summary["frozen_runtime_component_hashes"],
        "source_hashes": dict(sorted(source_hashes.items())),
        "episode_trace_hashes": trace_hashes,
        "episode_outcome_hashes": outcome_hashes,
        "aggregate_key_order": aggregate_keys,
        "aggregate_sha256": aggregate_hash,
        "row_count": len(ordered_rows),
    }
    fingerprint = _sha256_bytes(
        _canonical({"domain": "PASS_COMMIT_OFF_BASELINE_FINGERPRINT_v1", **identity}).encode("utf-8")
    )
    return {
        "schema_version": "pass_commit_off_baseline_contract_v1",
        "fingerprint_name": "PASS_COMMIT_OFF_BASELINE_FINGERPRINT",
        "fingerprint": fingerprint,
        "baseline_source": PRIOR_ON_STEPS.as_posix(),
        "baseline_condition": "prior authoritative lateral-exit braking ON; condition alone normalized to off",
        "normalized_field": "condition",
        "normalized_value": "off",
        "excluded_new_field_prefixes": ["pass_commit_"],
        "inherited_step_key_order": row_keys,
        "inherited_episode_key_order": episode_keys,
        "identity": identity,
        "mismatch_action": OFF_BASELINE_MISMATCH,
        "on_condition_directory_must_not_be_created_until_match": True,
    }


def validate_off_against_baseline(
    *,
    current_rows: Sequence[Mapping[str, Any]],
    current_episodes: Sequence[Mapping[str, Any]],
    current_aggregate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    identity = contract["identity"]
    row_keys = tuple(contract["inherited_step_key_order"])
    episode_keys = tuple(contract["inherited_episode_key_order"])
    if len(current_rows) != int(identity["row_count"]):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    ordered_rows = sorted(current_rows, key=lambda row: (int(row["episode_index"]), int(row["step"])))
    current_trace = {
        episode_id: _hash_rows(
            [row for row in ordered_rows if str(row.get("episode_id")) == episode_id],
            row_keys,
            normalize_condition=False,
        )
        for episode_id in ALL_EPISODE_IDS
    }
    if current_trace != identity["episode_trace_hashes"]:
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    if len(current_episodes) != len(ALL_EPISODE_IDS):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    ordered_episodes = sorted(current_episodes, key=lambda row: int(row["episode_index"]))
    if [str(row["episode_id"]) for row in ordered_episodes] != list(ALL_EPISODE_IDS):
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    current_outcomes = {
        str(row["episode_id"]): _hash_rows([row], episode_keys, normalize_condition=False)
        for row in ordered_episodes
    }
    if current_outcomes != identity["episode_outcome_hashes"]:
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    aggregate_keys = tuple(identity["aggregate_key_order"])
    current_aggregate_hash = _sha256_bytes(_canonical({
        "key_order": list(aggregate_keys),
        "values": [_normal(current_aggregate[key]) for key in aggregate_keys],
    }).encode("utf-8"))
    if current_aggregate_hash != identity["aggregate_sha256"]:
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    return {
        "status": "PASS",
        "baseline_fingerprint": contract["fingerprint"],
        "row_count": len(current_rows),
        "episode_count": len(current_episodes),
        "episode_trace_hashes_match": True,
        "episode_outcome_hashes_match": True,
        "aggregate_hash_matches": True,
        "only_normalized_field": "condition",
        "new_pass_commit_fields_excluded": True,
    }


def _load_previous_active_contract(project_root: Path) -> dict[str, Any]:
    previous_folder = project_root / PREVIOUS_PREPARATION_ROOT
    try:
        previous_payload = json.loads(
            (previous_folder / PREREGISTRATION_FILE).read_text(encoding="utf-8")
        )
        previous_lock = json.loads(
            (previous_folder / PREREGISTRATION_LOCK_FILE).read_text(encoding="utf-8")
        )
        baseline = json.loads(
            (previous_folder / OFF_BASELINE_CONTRACT_FILE).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(SOURCE_LOCK_MISMATCH) from exc
    active_hash = _sha256_bytes(_canonical(previous_payload).encode("utf-8"))
    if (
        active_hash != PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH
        or previous_lock.get("active_pass_commit_preregistration_hash") != active_hash
        or baseline.get("fingerprint") != EXPECTED_OFF_BASELINE_FINGERPRINT
    ):
        raise RuntimeError(SOURCE_LOCK_MISMATCH)
    return baseline


def verify_historical_off_baseline(
    *,
    prior_rows: Sequence[Mapping[str, Any]],
    prior_episodes: Sequence[Mapping[str, Any]],
    prior_aggregate: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the immutable frozen ON trace under its original OFF contract."""

    normalized_rows = [dict(row, condition="off") for row in prior_rows]
    normalized_episodes = [dict(row, condition="off") for row in prior_episodes]
    result = validate_off_against_baseline(
        current_rows=normalized_rows,
        current_episodes=normalized_episodes,
        current_aggregate=prior_aggregate,
        contract=contract,
    )
    if result["baseline_fingerprint"] != EXPECTED_OFF_BASELINE_FINGERPRINT:
        raise RuntimeError(OFF_BASELINE_MISMATCH)
    return result


def _preregistration_payload(
    *,
    cap: Mapping[str, Any],
    baseline: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "pass_commit_endpoints_v2_action_decomposition_fix",
        "stage": STAGE_NAME,
        "attempt_id": PASS_COMMIT_ATTEMPT_ID,
        "mechanism_diagnostic_only": True,
        "candidate_final_controller": False,
        "parent_radial_intent_diagnostic_identity": PARENT_RADIAL_INTENT_PREREGISTRATION_HASH,
        "parent_pass_commit_preregistration_hash": PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH,
        "parent_identity_role": "historical diagnostic identity only; not reused as active preregistration",
        "previous_attempt": {
            "status": "ABORTED_IN_ON_EPISODE_0001",
            "reason": "ACTION_DECOMPOSITION_ASSERTION",
            "off_baseline_fingerprint": EXPECTED_OFF_BASELINE_FINGERPRINT,
            "next_attempt_requires_fresh_paired_off_and_on": True,
        },
        "active_new_preregistration_required": True,
        "estimand": "paired_ON_minus_OFF_intention_to_treat",
        "population": {
            "primary": "fixed all-20 episode paired ITT",
            "ordered_episode_ids": list(ALL_EPISODE_IDS),
            "main_mechanism_subgroup": list(MAIN_FAILURE_EPISODE_IDS),
            "rescue_descriptive_control": RESCUE_DESCRIPTIVE_EPISODE_ID,
            "safety_special": SAFETY_SPECIAL_EPISODE_ID,
            "boundary_special": BOUNDARY_SPECIAL_EPISODE_ID,
        },
        "frozen_base_in_both_arms": {
            "checkpoint": True,
            "actor": True,
            "critic": True,
            "gat": True,
            "normalizer": True,
            "ordered_scenarios_and_seeds": True,
            "goal_recovery_and_terminal_stack": "unchanged frozen existing behavior",
            "reward": True,
            "dynamics": True,
            "action_transform": True,
            "safety_projection": True,
            "boundary_projection": True,
            "lateral_exit_braking_assistance_enabled": True,
            "radial_intent_persistence_assistance_enabled": False,
            "max_steps": 120,
        },
        "single_treatment_difference": {
            "pass_commit_enabled": {"OFF": False, "ON": True},
            "radial_intent_assistance": "OFF in both arms",
            "lateral_exit_braking_assistance": "ON in both arms",
        },
        "runtime_definition": {
            "state_source": "pre-step EKF estimate plus static map and existing tracker context only",
            "pass_geometry": {
                "direction": "u=(goal-estimated_position)/norm(goal-estimated_position)",
                "along": "(obstacle_center-estimated_position) dot u",
                "margin": "-(along+existing_obstacle_extent+existing_soft_recovery_buffer)",
                "direction_source": "current goal unit recomputed from the same exact release-gate geometry",
                "local_direction_proof": "p'=p+epsilon*u leaves u and extent unchanged and increases passed_margin by epsilon",
                "geometry_mismatch": "fail closed",
                "never_use_encounter_frozen_radial_axis": True,
            },
            "estimated_lateral_exit_gate": {
                "signal": "EKF estimated velocity projected onto existing encounter lateral unit",
                "absolute_threshold_mps": LATERAL_EXIT_THRESHOLD_MPS,
                "contiguous_steps": LATERAL_EXIT_CONTIGUOUS_STEPS,
                "margin_history_begins": "first ready row",
            },
            "regression_detector": {
                "delta": "current_passed_margin - previous_passed_margin",
                "deadband_m": PASS_COMMIT_TRIGGER_DEADBAND_M,
                "regression_rule": "delta < -1e-8",
                "trigger": "two consecutive regression rows",
                "latch_end": ["first exact pass/release", "episode end", "fail-closed condition"],
                "future_best_or_outcome_used_online": False,
            },
            "runtime_gates": [
                "valid frozen relevant obstacle identity",
                "three consecutive abs estimated lateral velocity <= 0.5 m/s",
                "passed_obstacle=false and release_latched=false",
                "post_bypass_recovery=false",
                "Goal Recovery inactive",
                "relevant obstacle clearance >= 2.5 m",
                "episode not done",
                "existing current-row past_closest_approach gate true",
                "existing current-row lateral_exit_assist_clearance_increasing_gate true",
            ],
            "mutual_exclusions": [
                "lateral-exit braking active makes PASS_COMMIT inactive for that row while preserving the latch",
                "Goal Recovery active permanently closes PASS_COMMIT for that episode",
                "radial-intent assistance OFF in both arms",
            ],
            "correction": {
                "component_before": "dot(estimated_velocity + baseline_candidate_delta_v, current_pass_direction_unit)",
                "requested_magnitude_mps": "max(0,-component_before), uncapped",
                "applied_magnitude_mps": "min(requested_magnitude, frozen cap)",
                "target": "zero pass-progress component; no positive floor",
                "cap_mps": cap["pass_commit_max_delta_v_mps"],
            },
            "action_decomposition": {
                "combined_pre_transform_action": "actor_base_candidate + goal_recovery_applied + lateral_exit_applied + pass_commit_applied + radial_intent_applied",
                "pass_commit_component_source": "applied/capped vector only; requested vector is telemetry and is never composed",
                "pass_commit_mutual_exclusion_asserted": ["goal_recovery", "lateral_exit_assistance", "radial_intent_assistance"],
                "action_transform_call_count": "at most once per decision row; equal to existing transform_applied flag",
            },
            "pipeline": "Actor raw Δv -> existing lateral braking -> PASS_COMMIT -> existing action transform exactly once -> existing safety/boundary/dynamics",
            "exact_pass_semantics": "existing passed-obstacle latch, passed_margin_m > 0, and same-row relevant obstacle clearance >= 2.5 m",
        },
        "authority_cap_derivation": cap,
        "off_baseline": {
            "fingerprint_name": baseline["fingerprint_name"],
            "fingerprint": baseline["fingerprint"],
            "baseline_source": baseline["baseline_source"],
            "normalized_only": "condition=on to condition=off",
            "new_pass_commit_fields_excluded": True,
        },
        "endpoint_hierarchy": [
            {"tier": 1, "name": "safety", "endpoints": ["collision", "minimum_clearance_m", "safety_projection_steps", "boundary_projection_steps"]},
            {"tier": 2, "name": "system", "endpoints": ["passed_obstacle", "post_bypass_recovery"]},
            {"tier": 3, "name": "mechanism", "endpoints": ["first_pass_step", "first_pass_commit_trigger_step", "first_pass_commit_active_step", "pass_commit_active_steps", "best_passed_margin_m", "final_pre_pass_passed_margin_m", "pass_margin_regression_count", "longest_regression_run", "time_exit_to_pass", "time_commit_to_pass"]},
            {"tier": 4, "name": "downstream_descriptive", "endpoints": ["entered_8m", "entered_5m", "entered_3m", "entered_1m", "stable_goal"]},
        ],
        "special_case_reporting": {
            RESCUE_DESCRIPTIVE_EPISODE_ID: "predefined rescue descriptive control only; never used to tune gate/cap",
            SAFETY_SPECIAL_EPISODE_ID: "report OFF/ON safety projections, PASS_COMMIT overlap, minimum clearance, and first overlap step; excluded from main mechanism vote",
            BOUNDARY_SPECIAL_EPISODE_ID: "report OFF/ON boundary projection steps; excluded from mechanism vote",
        },
        "interpretation_rules": {
            "support": "If ON passed_obstacle exceeds OFF with no collision increase, no material minimum-clearance deterioration, and no new safety-projection concentration, describe pass-progress non-regression as a causal contributing mechanism only.",
            "equal_pass_rate": "If OFF pass equals ON pass, conclude local pass-progress stability may improve but is insufficient to rescue additional bypasses; do not tune cap.",
            "safety_deterioration": "Reject/refine the gate if collision increases, clearance materially worsens, or safety projection increases materially, even if pass improves.",
            "unique_cause_claim": False,
        },
        "source_hashes": dict(sorted(source_hashes.items())),
        "scenario_manifest_hash": manifest["scenario_manifest_hash"],
        "preregistration_hash_is_domain_separated_from_parent": True,
        "preparation_flags": {
            "pass_commit_diagnostic_prepared": True,
            "pass_commit_diagnostic_executed": False,
            "actor_modified": False,
            "critic_modified": False,
            "gat_modified": False,
            "goal_recovery_modified": False,
            "terminal_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "safety_modified": False,
            "boundary_modified": False,
            "training_performed": False,
            "rollout_executed": False,
            "environment_step_called": False,
        },
    }


def build_preparation_bundle(project_root: Path | None = None) -> dict[str, str]:
    root = _project_root() if project_root is None else Path(project_root).resolve()
    prior_rows = _csv_rows(root / PRIOR_ON_STEPS)
    prior_episodes = _csv_rows(root / PRIOR_ON_EPISODES)
    prior_summary = json.loads((root / PRIOR_SUMMARY).read_text(encoding="utf-8"))
    prior_aggregate = prior_summary["results"]["on"]["aggregate"]
    source_hashes = _source_hashes(root)
    scenarios, manifest = _manifest_scenarios(root)
    cap_rows = derive_cap_rows(prior_rows, scenarios)
    cap = derive_cap(cap_rows)
    baseline = _load_previous_active_contract(root)
    baseline_verification = verify_historical_off_baseline(
        prior_rows=prior_rows,
        prior_episodes=prior_episodes,
        prior_aggregate=prior_aggregate,
        contract=baseline,
    )
    payload = _preregistration_payload(
        cap=cap,
        baseline=baseline,
        source_hashes=source_hashes,
        manifest=manifest,
    )
    active_hash = _sha256_bytes(_canonical(payload).encode("utf-8"))
    if active_hash == PARENT_RADIAL_INTENT_PREREGISTRATION_HASH:
        raise AssertionError("active PASS_COMMIT preregistration must be new")
    prereg_lock = {
        "schema_version": "pass_commit_preregistration_lock_v1",
        "parent_radial_intent_diagnostic_identity": PARENT_RADIAL_INTENT_PREREGISTRATION_HASH,
        "parent_pass_commit_preregistration_hash": PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH,
        "active_pass_commit_preregistration_hash": active_hash,
        "active_hash_algorithm": "sha256(canonical-json(preregistered_pass_commit_endpoints.json payload))",
        "payload_file": PREREGISTRATION_FILE,
        "payload_sha256": active_hash,
    }
    source_lock = {
        "schema_version": "pass_commit_runtime_source_lock_v2_action_decomposition_fix",
        "parent_radial_intent_diagnostic_identity": PARENT_RADIAL_INTENT_PREREGISTRATION_HASH,
        "parent_pass_commit_preregistration_hash": PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH,
        "active_pass_commit_preregistration_hash": active_hash,
        "preregistration_payload_sha256": active_hash,
        "source_hashes": source_hashes,
        "non_cyclic_lock_contract": "hash concrete runtime code and immutable inputs; exclude generated preparation artifacts and both lock payloads",
        "runtime_check": "validate every source hash, regenerated cap, baseline fingerprint and active preregistration before condition output layouts exist",
        "attempt_id": PASS_COMMIT_ATTEMPT_ID,
    }
    cap_artifact = {**cap, "source_steps_sha256": source_hashes[PRIOR_ON_STEPS.as_posix()], "source_manifest_sha256": source_hashes[SCENARIO_MANIFEST.as_posix()]}
    summary = {
        "status": "PASS_COMMIT_CAUSAL_DIAGNOSTIC_PREPARED",
        "attempt_id": PASS_COMMIT_ATTEMPT_ID,
        "parent_pass_commit_preregistration_hash": PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH,
        "pass_commit_diagnostic_prepared": True,
        "pass_commit_diagnostic_executed": False,
        "parent_radial_intent_preregistration_hash_identity": PARENT_RADIAL_INTENT_PREREGISTRATION_HASH,
        "new_active_preregistration_hash": active_hash,
        "pass_commit_off_baseline_fingerprint": baseline["fingerprint"],
        "off_baseline_offline_verification": baseline_verification,
        "pass_progress_direction_source": "current estimated goal unit derived from exact existing release geometry",
        "regression_trigger": "two consecutive passed-margin deltas below -1e-8 after estimated lateral exit is ready",
        "cap_eligible_rows": cap["eligible_row_count"],
        "eligible_rows_per_episode": cap["eligible_rows_per_episode"],
        "cap_statistics_mps": cap["required_cancellation_statistics_mps"],
        "frozen_cap_mps": cap["pass_commit_max_delta_v_mps"],
        "source_hashes": source_hashes,
        "canonical_artifact_files": list(CANONICAL_ARTIFACT_FILES),
        "execution_flags": payload["preparation_flags"],
    }
    columns = [
        "episode_id", "step", "included", "inclusion_exclusion_reason",
        "relevant_obstacle_id", "frozen_obstacle_id", "obstacle_matches",
        "estimated_position_m", "goal_unit", "pass_margin_m", "pass_margin_delta_m",
        "pass_margin_regression_run", "pass_commit_triggered", "pass_commit_latched",
        "estimated_lateral_exit_ready", "past_closest_approach", "clearance_increasing_gate",
        "lateral_exit_assist_active", "goal_recovery_active",
        "relevant_obstacle_clearance_m", "pass_progress_velocity_before_mps",
        "required_cancellation_mps",
    ]
    cap_csv = _csv_text(cap_rows, columns)
    return {
        PREREGISTRATION_FILE: _canonical(payload) + "\n",
        PREREGISTRATION_LOCK_FILE: _canonical(prereg_lock) + "\n",
        RUNTIME_SOURCE_LOCK_FILE: _canonical(source_lock) + "\n",
        CAP_DERIVATION_FILE: _canonical(cap_artifact) + "\n",
        CAP_ROWS_FILE: cap_csv,
        OFF_BASELINE_CONTRACT_FILE: _canonical(baseline) + "\n",
        PREPARATION_SUMMARY_FILE: _canonical(summary) + "\n",
    }


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = {
            key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        }
        writer.writerow(normalized)
    return output.getvalue()


def write_preparation_artifacts(
    project_root: Path | None = None,
    output_root: Path = PREPARATION_ROOT,
) -> dict[str, Any]:
    root = _project_root() if project_root is None else Path(project_root).resolve()
    destination = root / output_root
    files = build_preparation_bundle(root)
    if tuple(files) != CANONICAL_ARTIFACT_FILES:
        raise AssertionError("preparation bundle differs from canonical seven-file contract")
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(destination)
        existing_names = {item.name for item in destination.iterdir() if item.is_file()}
        if existing_names != set(CANONICAL_ARTIFACT_FILES):
            raise FileExistsError(f"refusing non-canonical output directory {destination}")
        if all((destination / name).read_text(encoding="utf-8") == files[name] for name in files):
            return {"status": "IDENTICAL_PREPARATION_ALREADY_PRESENT", "output_root": str(destination), "files": list(CANONICAL_ARTIFACT_FILES)}
        raise FileExistsError(f"refusing to overwrite non-identical preparation files: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".pass_commit_stage_", dir=destination.parent))
    staged = stage_parent / "bundle"
    try:
        staged.mkdir()
        for name in CANONICAL_ARTIFACT_FILES:
            (staged / name).write_text(files[name], encoding="utf-8", newline="\n")
        if {item.name for item in staged.iterdir() if item.is_file()} != set(CANONICAL_ARTIFACT_FILES):
            raise AssertionError("staged output differs from canonical seven-file contract")
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(staged, destination)
    finally:
        if stage_parent.exists():
            shutil.rmtree(stage_parent)
    return {"status": "PASS_COMMIT_CAUSAL_DIAGNOSTIC_PREPARED", "output_root": str(destination), "files": list(CANONICAL_ARTIFACT_FILES)}


def load_runtime_locks(project_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _project_root() if project_root is None else Path(project_root).resolve()
    folder = root / PREPARATION_ROOT
    try:
        prereg_lock = json.loads((folder / PREREGISTRATION_LOCK_FILE).read_text(encoding="utf-8"))
        source_lock = json.loads((folder / RUNTIME_SOURCE_LOCK_FILE).read_text(encoding="utf-8"))
        baseline = json.loads((folder / OFF_BASELINE_CONTRACT_FILE).read_text(encoding="utf-8"))
        cap = json.loads((folder / CAP_DERIVATION_FILE).read_text(encoding="utf-8"))
        payload = json.loads((folder / PREREGISTRATION_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(SOURCE_LOCK_MISMATCH) from exc
    try:
        current_hashes = _source_hashes(root)
        if current_hashes != source_lock.get("source_hashes"):
            raise RuntimeError(SOURCE_LOCK_MISMATCH)
        if (
            prereg_lock.get("parent_pass_commit_preregistration_hash")
            != PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH
            or source_lock.get("parent_pass_commit_preregistration_hash")
            != PREVIOUS_PASS_COMMIT_PREREGISTRATION_HASH
            or source_lock.get("attempt_id") != PASS_COMMIT_ATTEMPT_ID
            or source_lock.get("schema_version")
            != "pass_commit_runtime_source_lock_v2_action_decomposition_fix"
        ):
            raise RuntimeError(SOURCE_LOCK_MISMATCH)
        active_hash = _sha256_bytes(_canonical(payload).encode("utf-8"))
        if active_hash != prereg_lock.get("active_pass_commit_preregistration_hash") or active_hash != source_lock.get("active_pass_commit_preregistration_hash"):
            raise RuntimeError(SOURCE_LOCK_MISMATCH)
        prior_rows = _csv_rows(root / PRIOR_ON_STEPS)
        prior_episodes = _csv_rows(root / PRIOR_ON_EPISODES)
        summary = json.loads((root / PRIOR_SUMMARY).read_text(encoding="utf-8"))
        scenarios, manifest = _manifest_scenarios(root)
        cap_rows = derive_cap_rows(prior_rows, scenarios)
        expected_cap = derive_cap(cap_rows)
        expected_baseline = _load_previous_active_contract(root)
        baseline_verification = verify_historical_off_baseline(
            prior_rows=prior_rows,
            prior_episodes=prior_episodes,
            prior_aggregate=summary["results"]["on"]["aggregate"],
            contract=expected_baseline,
        )
        expected_payload = _preregistration_payload(
            cap=expected_cap,
            baseline=expected_baseline,
            source_hashes=current_hashes,
            manifest=manifest,
        )
        expected_cap_artifact = {**expected_cap, "source_steps_sha256": current_hashes[PRIOR_ON_STEPS.as_posix()], "source_manifest_sha256": current_hashes[SCENARIO_MANIFEST.as_posix()]}
        if (
            payload != expected_payload
            or baseline != expected_baseline
            or cap != expected_cap_artifact
            or baseline_verification["baseline_fingerprint"]
            != EXPECTED_OFF_BASELINE_FINGERPRINT
        ):
            raise RuntimeError(SOURCE_LOCK_MISMATCH)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeError) and str(exc) == SOURCE_LOCK_MISMATCH:
            raise
        raise RuntimeError(SOURCE_LOCK_MISMATCH) from exc
    return baseline, cap, source_lock, payload


def run_human_evaluation(**kwargs: Any) -> dict[str, Any]:
    """Explicit human-run path; deliberately lazy-imports the evaluator."""

    from .single_uav_goal_recovery_correction_evaluation import (
        run_goal_recovery_correction_evaluation,
    )

    return run_goal_recovery_correction_evaluation(
        pass_commit_diagnostic=True,
        formal=True,
        condition="paired",
        **kwargs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="write frozen offline artifacts")
    mode.add_argument("--human-run", action="store_true", help="explicit human-authorized paired evaluation")
    args = parser.parse_args(argv)
    if args.prepare_only:
        result = write_preparation_artifacts()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.human_run:
        run_human_evaluation()
        return 0
    parser.error("select exactly one mode")
    return 2


__all__ = [
    "ALL_EPISODE_IDS",
    "BOUNDARY_SPECIAL_EPISODE_ID",
    "CANONICAL_ARTIFACT_FILES",
    "CAP_DERIVATION_FAIL_CLOSED",
    "CLEARANCE_FLOOR_M",
    "HARD_CAP_MPS",
    "MAIN_FAILURE_EPISODE_IDS",
    "OFF_BASELINE_MISMATCH",
    "PARENT_RADIAL_INTENT_PREREGISTRATION_HASH",
    "PREPARATION_ROOT",
    "PREREGISTRATION_FILE",
    "RESCUE_DESCRIPTIVE_EPISODE_ID",
    "SAFETY_SPECIAL_EPISODE_ID",
    "_type7",
    "build_off_baseline_contract",
    "build_preparation_bundle",
    "derive_cap",
    "derive_cap_rows",
    "load_runtime_locks",
    "main",
    "run_human_evaluation",
    "validate_off_against_baseline",
    "write_preparation_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
