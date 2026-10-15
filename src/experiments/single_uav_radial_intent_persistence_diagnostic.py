"""Prepare and human-run the paired radial-intent persistence diagnostic.

Importing this module is offline and side-effect free.  The only runtime path
lazy-imports the evaluator after a human explicitly requests a formal run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


STAGE_NAME = "single_uav_radial_intent_persistence_diagnostic"
PARENT_LATERAL_EXIT_PREREGISTRATION_HASH = (
    "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2"
)
FRAME_SOURCE = "encounter_time_ekf_estimated_goal_ray_projected_to_xy"
RADIAL_COMMAND_DEADBAND_MPS = 1.0e-8
ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS = 0.5
ESTIMATED_LATERAL_EXIT_CONTIGUOUS_STEPS = 3
ACTOR_HISTORY_STEPS = 5
ACTOR_GOALWARD_MINIMUM_COUNT = 4
TARGET_FLOOR_MPS = 0.031192317367950993
RADIAL_ASSIST_HARD_CAP_MPS = 0.075
MAX_STEPS = 120
FORMAL_EPISODES = 20
FAILURE_EPISODE_IDS = tuple(
    f"test_r99_episode_{index:04d}" for index in (2, 4, 5, 6, 11, 12, 20)
)
ALL_EPISODE_IDS = tuple(
    f"test_r99_episode_{index:04d}" for index in range(1, FORMAL_EPISODES + 1)
)
PREPARATION_ROOT = Path(
    "logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/preparation"
)
PREREGISTRATION_FILE = "preregistered_radial_intent_persistence_endpoints.json"
PREREGISTRATION_LOCK_FILE = "radial_intent_preregistration_lock.json"
RUNTIME_SOURCE_LOCK_FILE = "radial_intent_runtime_source_lock.json"
CAP_DERIVATION_FILE = "radial_intent_cap_derivation.json"
CAP_ROWS_FILE = "radial_intent_cap_derivation_rows.csv"
OFF_BASELINE_CONTRACT_FILE = "radial_intent_off_baseline_contract.json"
PREPARATION_SUMMARY_FILE = "radial_intent_persistence_preparation_summary.json"
CANONICAL_ARTIFACT_FILES = (
    PREREGISTRATION_FILE,
    PREREGISTRATION_LOCK_FILE,
    RUNTIME_SOURCE_LOCK_FILE,
    CAP_DERIVATION_FILE,
    CAP_ROWS_FILE,
    OFF_BASELINE_CONTRACT_FILE,
    PREPARATION_SUMMARY_FILE,
)
PRIOR_LATERAL_LOG_ROOT = Path(
    "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
PRIOR_ON_STEPS = PRIOR_LATERAL_LOG_ROOT / "on" / "steps.csv"
PRIOR_ON_EPISODES = PRIOR_LATERAL_LOG_ROOT / "on" / "episodes.csv"
PRIOR_PARENT_PREREGISTRATION = PRIOR_LATERAL_LOG_ROOT / "preregistered_endpoints.json"
PRIOR_SUMMARY = PRIOR_LATERAL_LOG_ROOT / "evaluation_summary.json"
EVALUATOR_SOURCE = Path(
    "src/experiments/single_uav_goal_recovery_correction_evaluation.py"
)
LATERAL_DIAGNOSTIC_SOURCE = Path("src/experiments/single_uav_lateral_exit_braking_assistance.py")
LATERAL_ASSIST_SOURCE = Path("src/control/lateral_exit_braking_assistance.py")
SIGNED_OBSERVABILITY_SOURCE = Path(
    "src/experiments/single_uav_signed_bypass_observability.py"
)
GOAL_RECOVERY_SOURCE = Path("src/control/goal_recovery_correction.py")
ACTION_TRANSFORM_SOURCE = Path("src/experiments/single_uav_actor_constrained_oracle.py")
FORMAL_SOURCE = Path("src/experiments/formal.py")
ENVIRONMENT_SOURCE = Path("src/simulation/environment.py")
PREPASS_SUMMARY_SOURCE = Path(
    "logs/preliminary/single_uav_prepass_reorientation_reference_audit_seed2026/"
    "prepass_reorientation_summary.json"
)
PREPASS_REFERENCE_SOURCE = Path(
    "logs/preliminary/single_uav_prepass_reorientation_reference_audit_seed2026/"
    "prepass_success_reference.csv"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_radial_intent_persistence_diagnostic_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/run"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026"
)

RADIAL_INTENT_OFF_BASELINE_MISMATCH = "RADIAL_INTENT_OFF_BASELINE_MISMATCH"
RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED = "RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED"
RADIAL_INTENT_SOURCE_LOCK_MISMATCH = "RADIAL_INTENT_RUNTIME_SOURCE_LOCK_MISMATCH"


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


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric field {field}")
    return number


def _parse_vector(value: Any, *, field: str) -> tuple[float, float, float]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        vector = tuple(float(item) for item in parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid vector field {field}") from exc
    if len(vector) != 3 or not all(math.isfinite(item) for item in vector):
        raise ValueError(f"invalid vector field {field}")
    return vector  # type: ignore[return-value]


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class RadialIntentDecision:
    enabled: bool
    eligible: bool
    active: bool
    inactive_reason: str
    frame_source: str
    radial_unit: tuple[float, float, float]
    actor_radial_delta_v_mps: float
    actor_goalward_history_5: tuple[int, ...]
    actor_goalward_count_5: int
    actor_radial_persistence_deficient: bool
    estimated_lateral_velocity_mps: float
    estimated_lateral_exit_run: int
    estimated_lateral_exit_ready: bool
    requested_radial_assist_mps: float
    applied_radial_assist_mps: float
    requested_radial_assist_delta_v_mps: tuple[float, float, float]
    applied_radial_assist_delta_v_mps: tuple[float, float, float]


class RadialIntentPersistenceAssistance:
    """Evaluator-local stateful gate; inputs are estimated and policy values only."""

    def __init__(
        self,
        *,
        enabled: bool,
        cap_mps: float,
        deadband_mps: float = RADIAL_COMMAND_DEADBAND_MPS,
        target_floor_mps: float = TARGET_FLOOR_MPS,
    ) -> None:
        if not math.isfinite(float(cap_mps)) or float(cap_mps) < 0.0:
            raise ValueError("cap_mps must be finite and non-negative")
        if float(deadband_mps) != RADIAL_COMMAND_DEADBAND_MPS:
            raise ValueError("radial command deadband is preregistered and immutable")
        if float(target_floor_mps) != TARGET_FLOOR_MPS:
            raise ValueError("radial target floor is preregistered and immutable")
        self.enabled = bool(enabled)
        self.cap_mps = float(cap_mps)
        self.deadband_mps = float(deadband_mps)
        self.target_floor_mps = float(target_floor_mps)
        self._last_step: int | None = None
        self._lateral_exit_run = 0
        self._actor_history: deque[tuple[int, int]] = deque(maxlen=ACTOR_HISTORY_STEPS)

    def reset(self) -> None:
        self._last_step = None
        self._lateral_exit_run = 0
        self._actor_history.clear()

    def evaluate(
        self,
        *,
        step: int,
        actor_raw_delta_v_mps: Iterable[float],
        radial_unit: Iterable[float],
        estimated_lateral_velocity_mps: float,
        frame_frozen: bool,
        relevant_obstacle_match: bool,
        release_latched: bool,
        post_bypass_recovery: bool,
        passed_obstacle: bool,
        goal_recovery_active: bool,
        lateral_assist_active: bool,
        done: bool,
    ) -> RadialIntentDecision:
        """Evaluate one pre-action row, never accepting simulator truth."""

        action = tuple(float(item) for item in actor_raw_delta_v_mps)
        radial = tuple(float(item) for item in radial_unit)
        if len(action) != 3 or len(radial) != 3:
            raise ValueError("actor command and frozen radial unit must be 3-vectors")
        if not all(math.isfinite(item) for item in action + radial):
            raise ValueError("actor command and frozen radial unit must be finite")
        radial_norm = math.sqrt(sum(item * item for item in radial))
        if abs(radial_norm - 1.0) > 1.0e-8 or abs(radial[2]) > 1.0e-8:
            raise ValueError("radial unit must be frozen, horizontal, and unit length")
        current_step = int(step)
        contiguous = self._last_step is None or current_step == self._last_step + 1
        if not contiguous:
            self._lateral_exit_run = 0
            self._actor_history.clear()
        self._last_step = current_step
        actor_radial = sum(action[index] * radial[index] for index in range(3))
        lateral_velocity = float(estimated_lateral_velocity_mps)
        if frame_frozen and math.isfinite(lateral_velocity) and abs(lateral_velocity) <= ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS:
            self._lateral_exit_run += 1
        else:
            self._lateral_exit_run = 0
        lateral_exit_ready = self._lateral_exit_run >= ESTIMATED_LATERAL_EXIT_CONTIGUOUS_STEPS
        goalward = int(actor_radial > self.deadband_mps)
        if frame_frozen:
            self._actor_history.append((current_step, goalward))
        else:
            self._actor_history.clear()
        history_is_contiguous = (
            len(self._actor_history) == ACTOR_HISTORY_STEPS
            and all(
                self._actor_history[index][0] + 1 == self._actor_history[index + 1][0]
                for index in range(ACTOR_HISTORY_STEPS - 1)
            )
        )
        history = tuple(value for _, value in self._actor_history)
        goalward_count = sum(history)
        persistence_deficient = history_is_contiguous and goalward_count < ACTOR_GOALWARD_MINIMUM_COUNT
        prepass = not bool(release_latched) and not bool(post_bypass_recovery) and not bool(passed_obstacle)
        gates = (
            bool(frame_frozen),
            bool(relevant_obstacle_match),
            not bool(release_latched),
            not bool(post_bypass_recovery),
            prepass,
            not bool(goal_recovery_active),
            lateral_exit_ready,
            not bool(lateral_assist_active),
            history_is_contiguous,
            persistence_deficient,
            not bool(done),
        )
        eligible = all(gates)
        candidate_request = max(
            0.0,
            max(actor_radial, self.target_floor_mps) - actor_radial,
        )
        active = self.enabled and eligible and candidate_request > 0.0
        if not self.enabled:
            reason = "disabled"
        elif not frame_frozen:
            reason = "encounter_frame_not_frozen"
        elif not relevant_obstacle_match:
            reason = "frozen_obstacle_mismatch_or_unavailable"
        elif release_latched or post_bypass_recovery or passed_obstacle:
            reason = "pass_or_post_bypass_latched"
        elif goal_recovery_active:
            reason = "goal_recovery_active"
        elif not lateral_exit_ready:
            reason = "estimated_lateral_exit_not_ready"
        elif lateral_assist_active:
            reason = "lateral_exit_assistance_active"
        elif not history_is_contiguous:
            reason = "actor_history_not_ready"
        elif not persistence_deficient:
            reason = "actor_persistence_not_deficient"
        elif done:
            reason = "episode_done"
        elif candidate_request <= 0.0:
            reason = "actor_radial_at_or_above_target_floor"
        else:
            reason = "active"
        requested = candidate_request if active else 0.0
        applied = min(requested, self.cap_mps) if active else 0.0
        requested_vector = tuple(applied_scale * axis for applied_scale, axis in ((requested, radial[0]), (requested, radial[1]), (requested, radial[2])))
        applied_vector = tuple(applied * axis for axis in radial)
        if active and (goal_recovery_active or lateral_assist_active or release_latched or passed_obstacle or post_bypass_recovery):
            raise AssertionError("radial-intent assistance overlapped an excluded control state")
        return RadialIntentDecision(
            enabled=self.enabled,
            eligible=eligible,
            active=active,
            inactive_reason=reason,
            frame_source=FRAME_SOURCE,
            radial_unit=radial,
            actor_radial_delta_v_mps=actor_radial,
            actor_goalward_history_5=history,
            actor_goalward_count_5=goalward_count,
            actor_radial_persistence_deficient=persistence_deficient,
            estimated_lateral_velocity_mps=lateral_velocity,
            estimated_lateral_exit_run=self._lateral_exit_run,
            estimated_lateral_exit_ready=lateral_exit_ready,
            requested_radial_assist_mps=requested,
            applied_radial_assist_mps=applied,
            requested_radial_assist_delta_v_mps=requested_vector,
            applied_radial_assist_delta_v_mps=applied_vector,
        )


def assert_radial_intent_exclusive(
    *,
    radial_intent_active: bool,
    goal_recovery_active: bool,
    lateral_assist_active: bool,
) -> None:
    if radial_intent_active and goal_recovery_active:
        raise AssertionError("radial-intent assistance overlapped Goal Recovery")
    if radial_intent_active and lateral_assist_active:
        raise AssertionError("radial-intent assistance overlapped lateral-exit assistance")


def transform_radial_assisted_delta_v_once(
    *,
    base_delta_v_mps: Iterable[float],
    radial_assist_delta_v_mps: Iterable[float],
    action_transform: Callable[[Iterable[float], float], Mapping[str, Any]],
    action_limit_mps: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    base = tuple(float(value) for value in base_delta_v_mps)
    assist = tuple(float(value) for value in radial_assist_delta_v_mps)
    if len(base) != 3 or len(assist) != 3:
        raise ValueError("pre-transform vectors must be 3-vectors")
    combined = tuple(base[index] + assist[index] for index in range(3))
    result = action_transform(combined, float(action_limit_mps))
    constrained = tuple(float(value) for value in result["constrained_delta_v_mps"])
    if len(constrained) != 3:
        raise ValueError("action transform must return a 3-vector")
    return combined, constrained


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("type-7 quantile requires a non-empty finite sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _lateral_frame_from_prior_row(row: Mapping[str, Any]) -> tuple[float, float, float]:
    if row.get("lateral_exit_assist_frame_source") != FRAME_SOURCE:
        raise ValueError("prior row frame source does not match the frozen encounter frame")
    vector = _parse_vector(row.get("lateral_exit_assist_horizontal_goal_ray_m"), field="horizontal goal ray")
    horizontal_norm = math.hypot(vector[0], vector[1])
    if horizontal_norm <= 1.0e-9 or abs(vector[2]) > 1.0e-8:
        raise ValueError("prior encounter horizontal goal ray is degenerate")
    radial = (vector[0] / horizontal_norm, vector[1] / horizontal_norm, 0.0)
    return radial


def _offline_runtime_equivalent_gates(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Derive eligible cap rows from chronological EKF-visible prior ON rows."""

    grouped: dict[str, list[Mapping[str, str]]] = {episode_id: [] for episode_id in FAILURE_EPISODE_IDS}
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id in grouped:
            grouped[episode_id].append(row)
    accepted: list[dict[str, Any]] = []
    excluded_first: dict[str, int] = {}
    exclusion_flags: dict[str, int] = {}
    considered = 0
    for episode_id in FAILURE_EPISODE_IDS:
        episode_rows = grouped[episode_id]
        if not episode_rows:
            raise ValueError(f"missing required prior failure episode: {episode_id}")
        episode_rows.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in episode_rows] != list(range(len(episode_rows))):
            raise ValueError(f"non-contiguous prior step rows: {episode_id}")
        previous_radial_unit: tuple[float, float, float] | None = None
        lateral_run = 0
        actor_history: deque[tuple[int, int]] = deque(maxlen=ACTOR_HISTORY_STEPS)
        for row in episode_rows:
            considered += 1
            step = int(row["step"])
            frame_ready = bool(row.get("lateral_exit_assist_encounter_step", "")) and bool(
                row.get("lateral_exit_assist_lateral_unit", "") not in {"", "[]"}
            )
            obstacle_match = (
                frame_ready
                and _truth(row.get("lateral_exit_assist_frozen_obstacle_available", "0"))
                and str(row.get("lateral_exit_assist_frozen_obstacle_id", ""))
                == str(row.get("relevant_obstacle_id", ""))
            )
            if frame_ready:
                radial_unit = _lateral_frame_from_prior_row(row)
                if previous_radial_unit is not None and max(
                    abs(radial_unit[index] - previous_radial_unit[index]) for index in range(3)
                ) > 1.0e-8:
                    raise ValueError(f"encounter frame changed within episode: {episode_id}")
                previous_radial_unit = radial_unit
                estimated_lateral = _finite_float(row["estimated_lateral_velocity_mps"], field="estimated_lateral_velocity_mps")
                lateral_run = lateral_run + 1 if abs(estimated_lateral) <= ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS else 0
                actor_vector = _parse_vector(row["actor_raw_delta_v_mps"], field="actor_raw_delta_v_mps")
                actor_radial = _dot(actor_vector, radial_unit)
                lateral_unit = _parse_vector(row["lateral_exit_assist_lateral_unit"], field="lateral unit")
                vertical_unit = (0.0, 0.0, 1.0)
                actor_lateral = _dot(actor_vector, lateral_unit)
                actor_vertical = actor_vector[2]
                residual = math.sqrt(sum(
                    (actor_vector[index] - actor_radial * radial_unit[index] - actor_lateral * lateral_unit[index] - actor_vertical * vertical_unit[index]) ** 2
                    for index in range(3)
                ))
                if residual > 1.0e-8:
                    raise ValueError(f"frozen-frame radial dot reconstruction failed: {episode_id}:{step}")
                signed_axis = _parse_vector(row["signed_axis_radial_unit"], field="signed radial unit")
                signed_logged = _finite_float(row["actor_raw_radial_delta_v_mps"], field="logged radial actor command")
                signed_recomputed = _dot(actor_vector, signed_axis)
                if abs(signed_logged - signed_recomputed) > 1.0e-8:
                    raise ValueError(f"source radial-dot cross-check failed: {episode_id}:{step}")
                actor_history.append((step, int(actor_radial > RADIAL_COMMAND_DEADBAND_MPS)))
            else:
                radial_unit = (0.0, 0.0, 0.0)
                actor_radial = float("nan")
                lateral_run = 0
                actor_history.clear()
            history_ready = (
                len(actor_history) == ACTOR_HISTORY_STEPS
                and all(actor_history[index][0] + 1 == actor_history[index + 1][0] for index in range(ACTOR_HISTORY_STEPS - 1))
            )
            goalward_count = sum(flag for _, flag in actor_history)
            deficient = history_ready and goalward_count < ACTOR_GOALWARD_MINIMUM_COUNT
            release = _truth(row.get("passed_obstacle", "0"))
            post = _truth(row.get("post_bypass_recovery", "0"))
            passed = release or _truth(row.get("passed_condition_now", "False"))
            recovery_active = _truth(row.get("goal_recovery_correction_active", "0")) or _truth(row.get("terminal_capture_active", "0")) or _truth(row.get("terminal_v2_governor_active", "0")) or _truth(row.get("terminal_v21_radial_conflict_active", "0")) or _truth(row.get("terminal_v22_restorative_active", "0")) or _truth(row.get("terminal_v23_authority_active", "False"))
            lateral_active = _truth(row.get("lateral_exit_assist_active", "0"))
            done = _truth(row.get("terminated", "0")) or _truth(row.get("truncated", "0"))
            flags = {
                "frame_not_frozen": not frame_ready,
                "frozen_obstacle_unavailable_or_mismatch": not obstacle_match,
                "release_latched": release,
                "post_bypass_recovery": post,
                "passed_or_pass_condition": passed,
                "goal_recovery_active": recovery_active,
                "estimated_lateral_exit_not_ready": not (lateral_run >= ESTIMATED_LATERAL_EXIT_CONTIGUOUS_STEPS),
                "lateral_exit_assistance_active": lateral_active,
                "actor_history_not_ready": not history_ready,
                "actor_persistence_not_deficient": not deficient,
                "episode_done": done,
            }
            reason = next((name for name, failed in flags.items() if failed), "eligible")
            if reason == "eligible":
                target = max(0.0, TARGET_FLOOR_MPS - actor_radial)
                accepted.append({
                    "episode_id": episode_id,
                    "step": step,
                    "actor_radial_delta_v_mps": actor_radial,
                    "target_floor_mps": TARGET_FLOOR_MPS,
                    "required_assistance_mps": target,
                    "actor_goalward_history_5": [flag for _, flag in actor_history],
                    "actor_goalward_count_5": goalward_count,
                    "estimated_lateral_velocity_mps": estimated_lateral,
                    "estimated_lateral_exit_run": lateral_run,
                    "radial_unit": list(radial_unit),
                })
            else:
                excluded_first[reason] = excluded_first.get(reason, 0) + 1
            for name, failed in flags.items():
                if failed:
                    exclusion_flags[name] = exclusion_flags.get(name, 0) + 1
    return {
        "considered_row_count": considered,
        "eligible_row_count": len(accepted),
        "first_exclusion_counts": dict(sorted(excluded_first.items())),
        "all_gate_exclusion_counts": dict(sorted(exclusion_flags.items())),
        "eligible_rows": accepted,
    }


def derive_assistance_cap(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    gate_result = _offline_runtime_equivalent_gates(rows)
    required = [float(row["required_assistance_mps"]) for row in gate_result["eligible_rows"]]
    if not required or not all(math.isfinite(value) for value in required):
        raise RuntimeError(RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED)
    median = _type7(required, 0.5)
    q1 = _type7(required, 0.25)
    q3 = _type7(required, 0.75)
    maximum = max(required)
    cap = min(q3, RADIAL_ASSIST_HARD_CAP_MPS)
    if not math.isfinite(cap) or cap <= 0.0:
        raise RuntimeError(RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED)
    return {
        "schema_version": "radial_intent_cap_derivation_v1",
        "population": list(FAILURE_EPISODE_IDS),
        "source_file": str(PRIOR_ON_STEPS).replace("\\", "/"),
        "selection": "chronological rows satisfying reconstructed estimated-only runtime gates and deficient 5-step Actor history",
        "frozen_frame_source": FRAME_SOURCE,
        "deadband_mps": RADIAL_COMMAND_DEADBAND_MPS,
        "target_floor_mps": TARGET_FLOOR_MPS,
        "quantile_method": "numpy_linear_type7",
        "required_assistance_statistics_mps": {
            "median": median,
            "q1": q1,
            "q3": q3,
            "maximum": maximum,
        },
        "hard_cap_mps": RADIAL_ASSIST_HARD_CAP_MPS,
        "cap_rule": "min(Q3_required_assistance, 0.075 m/s)",
        "max_radial_intent_assist_mps": cap,
        "exclusion_counts": {
            "considered_row_count": gate_result["considered_row_count"],
            "eligible_row_count": gate_result["eligible_row_count"],
            "first_exclusion_counts": gate_result["first_exclusion_counts"],
            "all_gate_exclusion_counts": gate_result["all_gate_exclusion_counts"],
        },
        "frozen_radial_dot_cross_checks": {
            "encounter_frame_reconstruction_residual_max_mps": "<=1e-8",
            "logged_start_frame_radial_dot_residual_max_mps": "<=1e-8",
            "assistance_cap_uses": "recomputed_actor_dot_encounter_frozen_radial_unit",
        },
        "eligible_rows": gate_result["eligible_rows"],
        "fail_closed_policy": True,
        "fail_closed_triggered": False,
        "sweep_performed": False,
    }


def inherited_off_baseline_keys(source_rows: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    if not source_rows:
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    keys = tuple(source_rows[0].keys())
    required_families = (
        ("condition", "episode_index", "episode_id", "scenario_seed", "noise_seed", "step"),
        ("lateral_exit_assist_enabled", "lateral_exit_assist_active", "estimated_lateral_velocity_mps"),
        ("goal_recovery_correction_active", "terminal_capture_active", "terminal_v2_governor_active"),
        ("action_transform_applied", "corrected_delta_v_pre_transform_mps", "corrected_delta_v_pre_safety_mps"),
        ("safety_projection_applied", "boundary_projection_applied", "applied_delta_v_mps"),
        ("passed_obstacle", "post_bypass_recovery", "passed_margin_m", "relevant_obstacle_clearance_m"),
        ("collision", "terminated", "truncated", "stable_goal", "minimum_clearance_m"),
    )
    if any(any(key not in keys for key in family) for family in required_families):
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    if any(key.startswith("radial_intent_") for key in keys):
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    return keys


def _normalize_baseline_value(value: Any) -> Any:
    if value is None:
        return {"__scalar__": "none"}
    if value == "":
        return {"__scalar__": "empty"}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return _normalize_baseline_value(json.loads(stripped))
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
        return {
            str(key): _normalize_baseline_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_baseline_value(item) for item in value]
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
        return _normalize_baseline_value(value.tolist())
    return str(value)


def _baseline_row_values(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, normalize_condition: bool,
) -> list[list[Any]]:
    output: list[list[Any]] = []
    for row in rows:
        if any(key not in row for key in keys):
            raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
        values: list[Any] = []
        for key in keys:
            value = row[key]
            if key == "condition" and normalize_condition:
                value = "off"
            values.append(_normalize_baseline_value(value))
        output.append(values)
    return output


def baseline_trace_fingerprint(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, normalize_condition: bool,
) -> str:
    payload = {"key_order": list(keys), "rows": _baseline_row_values(rows, keys, normalize_condition=normalize_condition)}
    return _sha256_bytes(_canonical(payload).encode("utf-8"))


def build_off_baseline_contract(source_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    keys = inherited_off_baseline_keys(source_rows)
    ordered = sorted(source_rows, key=lambda row: (int(row["episode_index"]), int(row["step"])))
    if any(str(row.get("condition")) != "on" for row in ordered):
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    if [str(row.get("episode_id")) for row in ordered if int(row["step"]) == 0] != list(ALL_EPISODE_IDS):
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    for episode_id in ALL_EPISODE_IDS:
        episode = [row for row in ordered if row["episode_id"] == episode_id]
        if not episode or [int(row["step"]) for row in episode] != list(range(len(episode))):
            raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    return {
        "schema_version": "radial_intent_off_baseline_contract_v1",
        "baseline_source": str(PRIOR_ON_STEPS).replace("\\", "/"),
        "baseline_condition": "prior lateral-exit assistance ON trajectory, normalized only from condition=on to condition=off",
        "normalized_field": "condition",
        "normalized_value": "off",
        "inherited_key_order": list(keys),
        "excluded_new_fields_prefix": "radial_intent_",
        "row_count": len(ordered),
        "episode_order": list(ALL_EPISODE_IDS),
        "trace_sha256": baseline_trace_fingerprint(ordered, keys, normalize_condition=True),
        "source_steps_sha256": _sha256_file(_project_root() / PRIOR_ON_STEPS),
        "canonical_representation": "SHA256(canonical-json({key_order, rows_as_csv_cell_strings}); condition alone normalized to off)",
        "required_inherited_behavior_families": [
            "row identity and ordered episode/step index",
            "lateral-exit assistance telemetry and EKF estimated lateral velocity",
            "Goal Recovery and terminal v2/v2.1/v2.2/v2.3 state",
            "transform input/output and one-transform telemetry",
            "safety and boundary projections",
            "pass/release, recovery, margin, and obstacle clearance",
            "collision, termination, truncation, stable-goal downstream, minimum clearance",
        ],
        "mismatch_action": RADIAL_INTENT_OFF_BASELINE_MISMATCH,
        "on_condition_directory_must_not_be_created_until_match": True,
    }


def _source_hashes(project_root: Path) -> dict[str, str]:
    relative_paths = (
        EVALUATOR_SOURCE,
        Path(__file__).resolve().relative_to(project_root),
        LATERAL_DIAGNOSTIC_SOURCE,
        LATERAL_ASSIST_SOURCE,
        SIGNED_OBSERVABILITY_SOURCE,
        GOAL_RECOVERY_SOURCE,
        ACTION_TRANSFORM_SOURCE,
        FORMAL_SOURCE,
        ENVIRONMENT_SOURCE,
        PREPASS_SUMMARY_SOURCE,
        PREPASS_REFERENCE_SOURCE,
        PRIOR_PARENT_PREREGISTRATION,
        PRIOR_ON_STEPS,
        PRIOR_ON_EPISODES,
        PRIOR_SUMMARY,
    )
    result: dict[str, str] = {}
    for path in relative_paths:
        absolute = project_root / path
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        result[path.as_posix()] = _sha256_file(absolute)
    checkpoint = project_root / Path(
        "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
        "dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result["checkpoint/best_validation.pt"] = _sha256_file(checkpoint)
    return result


def _preregistration_payload(
    *, cap: Mapping[str, Any], baseline: Mapping[str, Any], source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "radial_intent_persistence_endpoints_v1",
        "stage": STAGE_NAME,
        "mechanism_diagnostic_only": True,
        "candidate_final_controller": False,
        "parent_lateral_exit_preregistration_hash": PARENT_LATERAL_EXIT_PREREGISTRATION_HASH,
        "new_radial_intent_preregistration_hash_is_distinct_from_parent": True,
        "estimand": "paired_ON_minus_OFF_intention_to_treat",
        "population": {
            "primary": "all 20 fixed ordered episodes; paired ITT",
            "ordered_episode_ids": list(ALL_EPISODE_IDS),
            "historical_main_failure_mechanism_subgroup_only": list(FAILURE_EPISODE_IDS),
            "episode_0008": "downstream-only descriptive case; excluded from pre-pass mechanism confirmation/denial",
        },
        "frozen_base_in_both_arms": {
            "actor": True,
            "critic": True,
            "gat": True,
            "normalizer": True,
            "checkpoint": True,
            "observation": True,
            "reward": True,
            "dynamics": True,
            "action_transform": True,
            "lateral_exit_braking_assistance_enabled": True,
            "goal_recovery_and_terminal_stack": "frozen v2/v2.1/v2.2/v2.3 and authority repair",
            "safety_projection": True,
            "boundary_projection": True,
            "ordered_scenarios_and_seeds": True,
            "max_steps": MAX_STEPS,
            "correction_enabled": True,
            "signed_observability_enabled": True,
        },
        "single_treatment_difference": {
            "radial_intent_assist_enabled": {"OFF": False, "ON": True},
            "actor_critic_gat_normalizer_checkpoint_scenarios_seeds": "identical",
        },
        "runtime_definition": {
            "frame_source": FRAME_SOURCE,
            "radial_unit_positive_direction": "goalward",
            "lateral_exit_gate": {
                "signal": "EKF estimated velocity projected on encounter-frozen lateral unit",
                "threshold_abs_mps": ESTIMATED_LATERAL_EXIT_THRESHOLD_MPS,
                "required_contiguous_steps": ESTIMATED_LATERAL_EXIT_CONTIGUOUS_STEPS,
            },
            "deadband_mps": RADIAL_COMMAND_DEADBAND_MPS,
            "actor_persistence_gate": {
                "history_steps": ACTOR_HISTORY_STEPS,
                "goalward_rule": "actor_radial_delta_v_mps > deadband",
                "deficient_rule": "goalward count in full contiguous window < 4",
            },
            "all_required_gates": [
                "encounter frame frozen",
                "relevant frozen obstacle available and matching",
                "release_latched=false",
                "post_bypass_recovery=false",
                "passed_obstacle=false and pre-pass stage",
                "Goal Recovery inactive",
                "estimated lateral exit ready",
                "lateral-exit assistance inactive this step",
                "full contiguous 5-step Actor history ready and deficient",
                "episode not done",
            ],
            "assist_rule": "requested=max(0,max(actor_radial,target_floor)-actor_radial); applied=min(requested,cap)",
            "assistance_vector": "applied_radial_assist_mps * frozen radial_unit; lateral and vertical components exactly zero",
            "target_floor_mps": TARGET_FLOOR_MPS,
            "max_radial_intent_assist_mps": cap["max_radial_intent_assist_mps"],
            "assistance_cap_derivation": {
                "source": "existing frozen lateral-exit-assistance ON steps.csv only",
                "episodes": list(FAILURE_EPISODE_IDS),
                "population": "runtime-equivalent eligible estimated-only pre-pass rows with deficient full Actor history",
                "statistics": cap["required_assistance_statistics_mps"],
                "rule": cap["cap_rule"],
                "no_sweep": True,
                "fail_closed": True,
                "fail_closed_policy": cap["fail_closed_policy"],
                "fail_closed_triggered_at_preparation": cap[
                    "fail_closed_triggered"
                ],
            },
            "post-pass_inactive_immediately": True,
            "goal_recovery_and_lateral_assistance_overlap_assertion": True,
            "action_transform": "actor -> existing lateral assist -> radial assist -> exactly one existing action transform -> safety/boundary",
        },
        "primary_endpoint_hierarchy": [
            {
                "tier": 1,
                "name": "safety",
                "endpoints": ["collision", "minimum_clearance_m", "safety_projection_steps", "boundary_projection_steps"],
            },
            {
                "tier": 2,
                "name": "system_causal",
                "endpoints": ["passed_obstacle", "post_bypass_recovery"],
            },
            {
                "tier": 3,
                "name": "mechanism",
                "endpoints": [
                    "first_pass_step", "best_passed_margin_m", "final_passed_margin_m",
                    "pass_progress_regression", "time_from_runtime_lateral_exit_to_pass",
                    "final_goalward_radial_fraction", "integrated_positive_radial_assist",
                ],
            },
        ],
        "paired_transition_categories": [
            "OFF fail -> ON pass", "OFF pass -> ON fail", "OFF fail -> ON fail", "OFF pass -> ON pass",
        ],
        "transition_table_population": "all 20 episodes",
        "fixed_failure_episode_rows_to_report": list(FAILURE_EPISODE_IDS),
        "downstream_descriptive_only": ["stable_goal", "episode_0008 downstream terminal adverse case"],
        "interpretation_rules": {
            "support": "If passed_obstacle or post_bypass_recovery increases with no collision increase, no material minimum-clearance deterioration, and no new concentration of safety/boundary projection, support insufficient pre-pass radial persistence as a causal contributing bottleneck only.",
            "passed_11_to_11": "If passed_obstacle is 11 in OFF and 11 in ON, even if radial fraction or passed margin improves, conclude local radial mechanism improved but radial-intent persistence is not sufficient to rescue bypass; do not tune cap or sweep; next inspect pass-progress state transition/geometry regression.",
            "safety_deterioration": "Reject or refine intervention gates if collision increases, minimum clearance deteriorates, or safety projection materially increases; pass improvement alone is insufficient.",
            "not_a_unique_cause": True,
        },
        "source_hashes": dict(sorted(source_hashes.items())),
        "off_baseline": {
            "fingerprint": baseline["trace_sha256"],
            "row_count": baseline["row_count"],
            "source_steps_sha256": baseline["source_steps_sha256"],
        },
        "preparation_only_flags": {
            "radial_intent_diagnostic_prepared": True,
            "radial_intent_diagnostic_executed": False,
            "actor_modified": False,
            "critic_modified": False,
            "gat_modified": False,
            "goal_recovery_modified": False,
            "terminal_controller_modified": False,
            "lateral_exit_assistance_modified": False,
            "reward_modified": False,
            "dynamics_modified": False,
            "safety_modified": False,
            "boundary_modified": False,
            "training_performed": False,
            "rollout_executed": False,
            "environment_step_called": False,
        },
    }


def build_preparation_bundle(project_root: Path | None = None) -> dict[str, Any]:
    """Read frozen offline inputs and build the seven canonical preparation artifacts."""

    root = _project_root() if project_root is None else Path(project_root).resolve()
    prior_steps_path = root / PRIOR_ON_STEPS
    prior_rows = _csv_rows(prior_steps_path)
    if not prior_rows:
        raise RuntimeError(RADIAL_INTENT_CAP_DERIVATION_FAIL_CLOSED)
    cap = derive_assistance_cap(prior_rows)
    baseline = build_off_baseline_contract(prior_rows)
    source_hashes = _source_hashes(root)
    payload = _preregistration_payload(cap=cap, baseline=baseline, source_hashes=source_hashes)
    prereg_hash = _sha256_bytes(_canonical(payload).encode("utf-8"))
    if prereg_hash == PARENT_LATERAL_EXIT_PREREGISTRATION_HASH:
        raise AssertionError("new preregistration identity must differ from its parent")
    source_lock = {
        "schema_version": "radial_intent_runtime_source_lock_v1",
        "parent_lateral_exit_preregistration_hash": PARENT_LATERAL_EXIT_PREREGISTRATION_HASH,
        "new_radial_intent_preregistration_hash": prereg_hash,
        "preregistration_payload_sha256": prereg_hash,
        "cap_derivation_fail_closed_policy": cap["fail_closed_policy"],
        "cap_derivation_fail_closed_triggered": cap["fail_closed_triggered"],
        "source_hashes": dict(sorted(source_hashes.items())),
        "non_cyclic_lock_contract": "hashes concrete code/input files; excludes this lock and all generated preparation artifacts",
        "runtime_check": "exact source hash equality required before loading the frozen policy or creating output roots",
    }
    prereg_lock = {
        "schema_version": "radial_intent_preregistration_lock_v1",
        "parent_lateral_exit_preregistration_hash": PARENT_LATERAL_EXIT_PREREGISTRATION_HASH,
        "active_radial_intent_preregistration_hash": prereg_hash,
        "active_hash_algorithm": "sha256(canonical-json(preregistered_radial_intent_persistence_endpoints.json payload))",
        "payload_file": PREREGISTRATION_FILE,
        "payload_sha256": prereg_hash,
        "cap_derivation_fail_closed_policy": cap["fail_closed_policy"],
        "cap_derivation_fail_closed_triggered": cap["fail_closed_triggered"],
        "source_lateral_preregistration_file": str(PRIOR_PARENT_PREREGISTRATION).replace("\\", "/"),
        "source_lateral_preregistration_sha256": source_hashes[PRIOR_PARENT_PREREGISTRATION.as_posix()],
    }
    cap["source_steps_sha256"] = source_hashes[PRIOR_ON_STEPS.as_posix()]
    cap["parent_preregistration_sha256"] = source_hashes[PRIOR_PARENT_PREREGISTRATION.as_posix()]
    summary = {
        "status": "RADIAL_INTENT_PERSISTENCE_DIAGNOSTIC_PREPARED",
        "radial_intent_diagnostic_prepared": True,
        "radial_intent_diagnostic_executed": False,
        "mechanism_diagnostic_only": True,
        "candidate_final_controller": False,
        "parent_lateral_exit_preregistration_hash": PARENT_LATERAL_EXIT_PREREGISTRATION_HASH,
        "new_radial_intent_preregistration_hash": prereg_hash,
        "new_off_baseline_fingerprint": baseline["trace_sha256"],
        "target_floor_mps": TARGET_FLOOR_MPS,
        "max_radial_intent_assist_mps": cap["max_radial_intent_assist_mps"],
        "cap_derivation_statistics_mps": cap["required_assistance_statistics_mps"],
        "cap_derivation_exclusion_counts": cap["exclusion_counts"],
        "cap_derivation_fail_closed_policy": cap["fail_closed_policy"],
        "cap_derivation_fail_closed_triggered": cap["fail_closed_triggered"],
        "gates": payload["runtime_definition"]["all_required_gates"],
        "insertion_point": payload["runtime_definition"]["action_transform"],
        "mutual_exclusion_assertions": ["Goal Recovery inactive", "lateral-exit braking assistance inactive"],
        "primary_endpoints": payload["primary_endpoint_hierarchy"],
        "paired_transition_categories": payload["paired_transition_categories"],
        "source_hashes": dict(sorted(source_hashes.items())),
        "canonical_artifact_files": list(CANONICAL_ARTIFACT_FILES),
        "no_data_copies": True,
        "execution_flags": payload["preparation_only_flags"],
    }
    cap_rows = cap.pop("eligible_rows")
    cap_csv = _csv_text(
        cap_rows,
        ("episode_id", "step", "actor_radial_delta_v_mps", "target_floor_mps", "required_assistance_mps", "actor_goalward_history_5", "actor_goalward_count_5", "estimated_lateral_velocity_mps", "estimated_lateral_exit_run", "radial_unit"),
    )
    return {
        PREREGISTRATION_FILE: _canonical(payload) + "\n",
        PREREGISTRATION_LOCK_FILE: _canonical(prereg_lock) + "\n",
        RUNTIME_SOURCE_LOCK_FILE: _canonical(source_lock) + "\n",
        CAP_DERIVATION_FILE: _canonical(cap) + "\n",
        CAP_ROWS_FILE: cap_csv,
        OFF_BASELINE_CONTRACT_FILE: _canonical(baseline) + "\n",
        PREPARATION_SUMMARY_FILE: _canonical(summary) + "\n",
    }


def _csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        output: dict[str, Any] = {}
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            output[key] = value
        writer.writerow(output)
    return buffer.getvalue()


def write_preparation_artifacts(
    project_root: Path | None = None,
    output_root: Path = PREPARATION_ROOT,
) -> dict[str, Any]:
    """Atomically write exact canonical files, refusing non-identical overwrite."""

    root = _project_root() if project_root is None else Path(project_root).resolve()
    destination = root / output_root
    files = build_preparation_bundle(root)
    if set(files) != set(CANONICAL_ARTIFACT_FILES) or len(files) != len(CANONICAL_ARTIFACT_FILES):
        raise AssertionError("preparation output file set differs from the canonical contract")
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(destination)
        existing = {path.name for path in destination.iterdir() if path.is_file()}
        if existing != set(CANONICAL_ARTIFACT_FILES):
            raise FileExistsError(f"refusing non-canonical preparation output directory: {destination}")
        if all((destination / name).read_text(encoding="utf-8") == files[name] for name in files):
            return {"status": "IDENTICAL_PREPARATION_ALREADY_PRESENT", "output_root": str(destination), "files": list(CANONICAL_ARTIFACT_FILES)}
        raise FileExistsError(f"refusing to overwrite non-identical preparation artifacts: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".radial_intent_stage_", dir=destination.parent))
    bundle = stage_parent / "bundle"
    try:
        bundle.mkdir()
        for name in CANONICAL_ARTIFACT_FILES:
            (bundle / name).write_text(files[name], encoding="utf-8", newline="\n")
        if {path.name for path in bundle.iterdir() if path.is_file()} != set(CANONICAL_ARTIFACT_FILES):
            raise AssertionError("staged preparation file set differs from the canonical contract")
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(bundle, destination)
    finally:
        if stage_parent.exists():
            shutil.rmtree(stage_parent)
    return {"status": "RADIAL_INTENT_PERSISTENCE_DIAGNOSTIC_PREPARED", "output_root": str(destination), "files": list(CANONICAL_ARTIFACT_FILES)}


def load_runtime_locks(project_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _project_root() if project_root is None else Path(project_root).resolve()
    folder = root / PREPARATION_ROOT
    try:
        prereg_lock = json.loads((folder / PREREGISTRATION_LOCK_FILE).read_text(encoding="utf-8"))
        source_lock = json.loads((folder / RUNTIME_SOURCE_LOCK_FILE).read_text(encoding="utf-8"))
        baseline = json.loads((folder / OFF_BASELINE_CONTRACT_FILE).read_text(encoding="utf-8"))
        cap = json.loads((folder / CAP_DERIVATION_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH) from exc
    expected_hashes = source_lock.get("source_hashes", {})
    try:
        current_hashes = _source_hashes(root)
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH) from exc
    if current_hashes != expected_hashes:
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH)
    if prereg_lock.get("active_radial_intent_preregistration_hash") != source_lock.get("new_radial_intent_preregistration_hash"):
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH)
    payload = json.loads((folder / PREREGISTRATION_FILE).read_text(encoding="utf-8"))
    payload_hash = _sha256_bytes(_canonical(payload).encode("utf-8"))
    if payload_hash != prereg_lock.get("active_radial_intent_preregistration_hash") or payload_hash == PARENT_LATERAL_EXIT_PREREGISTRATION_HASH:
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH)
    try:
        prior_rows = _csv_rows(root / PRIOR_ON_STEPS)
        expected_cap = derive_assistance_cap(prior_rows)
        expected_baseline = build_off_baseline_contract(prior_rows)
        expected_payload = _preregistration_payload(
            cap=expected_cap,
            baseline=expected_baseline,
            source_hashes=current_hashes,
        )
        expected_cap.pop("eligible_rows")
        expected_cap["source_steps_sha256"] = current_hashes[PRIOR_ON_STEPS.as_posix()]
        expected_cap["parent_preregistration_sha256"] = current_hashes[
            PRIOR_PARENT_PREREGISTRATION.as_posix()
        ]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH) from exc
    if payload != expected_payload or cap != expected_cap or baseline != expected_baseline:
        raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH)
    for lock in (prereg_lock, source_lock):
        if (
            lock.get("cap_derivation_fail_closed_policy") is not True
            or lock.get("cap_derivation_fail_closed_triggered") is not False
        ):
            raise RuntimeError(RADIAL_INTENT_SOURCE_LOCK_MISMATCH)
    return baseline, cap, source_lock


def validate_off_against_frozen_baseline(
    *,
    current_rows: Sequence[Mapping[str, Any]],
    baseline_contract: Mapping[str, Any],
) -> dict[str, Any]:
    keys = tuple(str(value) for value in baseline_contract.get("inherited_key_order", ()))
    if not keys or len(current_rows) != int(baseline_contract.get("row_count", -1)):
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    # New radial telemetry is deliberately ignored: comparison projects only
    # the frozen inherited key list from the prior lateral-ON trace.
    fingerprint = baseline_trace_fingerprint(current_rows, keys, normalize_condition=False)
    if fingerprint != baseline_contract.get("trace_sha256"):
        raise RuntimeError(RADIAL_INTENT_OFF_BASELINE_MISMATCH)
    return {
        "status": "PASS",
        "baseline_fingerprint": fingerprint,
        "row_count": len(current_rows),
        "inherited_key_count": len(keys),
        "only_normalized_field": "condition",
        "new_radial_fields_excluded": True,
    }


def assert_paired_identity(
    *,
    off_rows: Sequence[Mapping[str, Any]],
    on_rows: Sequence[Mapping[str, Any]],
    off_component_hashes: Mapping[str, Any],
    on_component_hashes: Mapping[str, Any],
    off_provenance: Mapping[str, Any],
    on_provenance: Mapping[str, Any],
) -> None:
    identity_keys = ("episode_index", "episode_id", "scenario_seed", "noise_seed", "scenario_hash")
    identity = lambda rows: [tuple(str(row.get(key, "")) for key in identity_keys) for row in rows]
    if identity(off_rows) != identity(on_rows):
        raise AssertionError("RADIAL_INTENT_PAIRED_IDENTITY_MISMATCH")
    if off_component_hashes != on_component_hashes:
        raise AssertionError("RADIAL_INTENT_FROZEN_COMPONENT_HASH_MISMATCH")
    for key in ("scenario_manifest_hash", "ordered_pool_identity", "checkpoint", "policy_hash", "observation_dimension", "controller_config_hash", "goal_recovery_config_hash"):
        if off_provenance.get(key) != on_provenance.get(key):
            raise AssertionError(f"RADIAL_INTENT_FROZEN_IDENTITY_MISMATCH:{key}")
    if off_provenance.get("correction_enabled") is not True or on_provenance.get("correction_enabled") is not True:
        raise AssertionError("RADIAL_INTENT_GOAL_RECOVERY_BASELINE_NOT_FROZEN_ON")
    if off_provenance.get("lateral_exit_braking_assistance_enabled") is not True or on_provenance.get("lateral_exit_braking_assistance_enabled") is not True:
        raise AssertionError("RADIAL_INTENT_LATERAL_BASELINE_NOT_FROZEN_ON")
    if off_provenance.get("radial_intent_assistance_enabled") is not False or on_provenance.get("radial_intent_assistance_enabled") is not True:
        raise AssertionError("RADIAL_INTENT_NOT_SOLE_TREATMENT_DIFFERENCE")


def _number_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def paired_itt_endpoint_summary(
    *,
    off_episodes: Sequence[Mapping[str, Any]],
    on_episodes: Sequence[Mapping[str, Any]],
    off_steps: Sequence[Mapping[str, Any]],
    on_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize all-20 paired outcomes, transitions, and fixed subgroups."""

    def fixed_episode_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        ids = [str(row.get("episode_id", "")) for row in rows]
        if len(rows) != FORMAL_EPISODES or ids != list(ALL_EPISODE_IDS) or len(set(ids)) != FORMAL_EPISODES:
            raise AssertionError("RADIAL_INTENT_PRIMARY_POPULATION_MUST_BE_FIXED_ALL_20")
        return {episode_id: row for episode_id, row in zip(ids, rows)}

    off_by_id = fixed_episode_map(off_episodes)
    on_by_id = fixed_episode_map(on_episodes)
    if tuple(off_by_id) != ALL_EPISODE_IDS or tuple(on_by_id) != ALL_EPISODE_IDS:
        raise AssertionError("RADIAL_INTENT_PRIMARY_POPULATION_MUST_BE_FIXED_ALL_20")
    by_condition = {
        "off": {episode_id: [] for episode_id in ALL_EPISODE_IDS},
        "on": {episode_id: [] for episode_id in ALL_EPISODE_IDS},
    }
    for condition, rows in (("off", off_steps), ("on", on_steps)):
        for row in rows:
            episode_id = str(row.get("episode_id", ""))
            if episode_id not in by_condition[condition]:
                raise AssertionError("RADIAL_INTENT_UNEXPECTED_EPISODE")
            by_condition[condition][episode_id].append(row)
        for episode_id, local_rows in by_condition[condition].items():
            local_rows.sort(key=lambda row: int(row["step"]))
            if not local_rows or [int(row["step"]) for row in local_rows] != list(range(len(local_rows))):
                raise AssertionError(f"RADIAL_INTENT_NONCONTIGUOUS_STEPS:{condition}:{episode_id}")

    def passed(row: Mapping[str, Any]) -> bool:
        return bool(int(float(row.get("passed_obstacle", 0))))

    transitions: dict[str, list[str]] = {
        "OFF fail -> ON pass": [],
        "OFF pass -> ON fail": [],
        "OFF fail -> ON fail": [],
        "OFF pass -> ON pass": [],
    }
    per_episode: list[dict[str, Any]] = []
    for episode_id in ALL_EPISODE_IDS:
        off_episode, on_episode = off_by_id[episode_id], on_by_id[episode_id]
        off_pass, on_pass = passed(off_episode), passed(on_episode)
        category = (
            "OFF pass -> ON pass" if off_pass and on_pass else
            "OFF pass -> ON fail" if off_pass and not on_pass else
            "OFF fail -> ON pass" if not off_pass and on_pass else
            "OFF fail -> ON fail"
        )
        transitions[category].append(episode_id)

        condition_detail: dict[str, Any] = {}
        for condition, episode, local_rows in (
            ("off", off_episode, by_condition["off"][episode_id]),
            ("on", on_episode, by_condition["on"][episode_id]),
        ):
            pass_rows = [row for row in local_rows if int(float(row.get("passed_obstacle", 0)))]
            first_pass_row = pass_rows[0] if pass_rows else None
            first_exit = next(
                (row for row in local_rows if int(float(row.get("estimated_lateral_exit_ready", 0)))),
                None,
            )
            last_prepass = [
                row for row in local_rows
                if first_pass_row is None or int(row["step"]) < int(first_pass_row["step"])
            ][-ACTOR_HISTORY_STEPS:]
            radial_commands = [
                value for row in last_prepass
                if (value := _number_or_none(row.get("actor_radial_delta_v_mps"))) is not None
            ]
            margins = [
                value for row in pass_rows
                if (value := _number_or_none(row.get("passed_margin_m"))) is not None
            ]
            progress_values = [
                value for row in local_rows
                if (value := _number_or_none(row.get("soft_pass_progress"))) is not None
            ]
            running_best = -math.inf
            regression = False
            for value in progress_values:
                if value < running_best - 1.0e-12:
                    regression = True
                running_best = max(running_best, value)
            lateral_exit_to_pass = (
                int(first_pass_row["step"]) - int(first_exit["step"])
                if first_pass_row is not None and first_exit is not None
                else None
            )
            final_margin = _number_or_none(local_rows[-1].get("passed_margin_m"))
            condition_detail[condition] = {
                "passed_obstacle": int(off_pass if condition == "off" else on_pass),
                "post_bypass_recovery": int(bool(int(float(episode.get("post_bypass_recovery", 0))))),
                "collision": int(bool(int(float(episode.get("collision", 0))))),
                "minimum_clearance_m": _number_or_none(
                    episode.get("minimum_clearance_m", min(
                        [value for row in local_rows if (value := _number_or_none(row.get("minimum_clearance_m"))) is not None],
                        default=None,
                    ))
                ),
                "safety_projection_steps": sum(int(float(row.get("safety_projection_applied", 0))) for row in local_rows),
                "boundary_projection_steps": sum(int(float(row.get("boundary_projection_applied", 0))) for row in local_rows),
                "first_pass_step": int(first_pass_row["step"]) if first_pass_row is not None else None,
                "best_passed_margin_m": max(margins) if margins else None,
                "final_passed_margin_m": final_margin,
                "pass_progress_regression": regression,
                "time_from_runtime_lateral_exit_to_pass_steps": lateral_exit_to_pass,
                "final_goalward_radial_fraction": (
                    sum(value > RADIAL_COMMAND_DEADBAND_MPS for value in radial_commands) / len(radial_commands)
                    if radial_commands else None
                ),
                "integrated_positive_radial_assist_mps_steps": sum(
                    max(0.0, value)
                    for row in local_rows
                    if (value := _number_or_none(row.get("applied_radial_assist_mps"))) is not None
                ),
                "stable_goal_downstream_descriptive": int(bool(int(float(episode.get("stable_goal", 0))))),
            }
        per_episode.append({"episode_id": episode_id, "paired_transition": category, **condition_detail})
    if sum(len(ids) for ids in transitions.values()) != FORMAL_EPISODES:
        raise AssertionError("RADIAL_INTENT_TRANSITION_TABLE_MUST_COVER_ALL_20")
    return {
        "schema_version": "radial_intent_persistence_paired_itt_results_v1",
        "population": "all 20 ordered episodes; paired ITT",
        "primary_endpoint_hierarchy": [
            {"tier": 1, "name": "safety", "endpoints": ["collision", "minimum_clearance_m", "safety_projection_steps", "boundary_projection_steps"]},
            {"tier": 2, "name": "system_causal", "endpoints": ["passed_obstacle", "post_bypass_recovery"]},
            {"tier": 3, "name": "mechanism", "endpoints": ["first_pass_step", "best_passed_margin_m", "final_passed_margin_m", "pass_progress_regression", "time_from_runtime_lateral_exit_to_pass_steps", "final_goalward_radial_fraction", "integrated_positive_radial_assist_mps_steps"]},
        ],
        "episode_count": FORMAL_EPISODES,
        "episode_order": list(ALL_EPISODE_IDS),
        "paired_transitions": transitions,
        "historical_main_failure_subgroup_descriptive_only": {
            episode_id.removeprefix("test_r99_episode_"): next(
                row for row in per_episode if row["episode_id"] == episode_id
            )
            for episode_id in FAILURE_EPISODE_IDS
        },
        "episode_0008_role": (
            "test_r99_episode_0008 is a downstream terminal adverse case only; "
            "not used to confirm or deny the pre-pass mechanism"
        ),
        "stable_goal_role": "downstream descriptive endpoint only",
        "per_episode_paired_itt": per_episode,
    }


def run_human_evaluation(**kwargs: Any) -> dict[str, Any]:
    """Human-only, lazy evaluator entry point; never used by preparation."""

    evaluator_module = __import__(
        "src.experiments.single_uav_goal_recovery_correction_evaluation",
        fromlist=["run_goal_recovery_correction_evaluation"],
    )
    return evaluator_module.run_goal_recovery_correction_evaluation(
        radial_intent_persistence_diagnostic=True,
        **kwargs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="write frozen offline artifacts; no evaluator import")
    mode.add_argument("--human-run", action="store_true", help="human-authorized formal paired evaluation")
    parser.add_argument("--condition", choices=("paired", "off", "on"), default="paired")
    parser.add_argument("--quiet", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.prepare_only:
        if remaining:
            parser.error("--prepare-only accepts no runtime options")
        result = write_preparation_artifacts()
        print("RADIAL_INTENT_PERSISTENCE_DIAGNOSTIC_PREPARED")
        print(_canonical(result))
        return 0
    if args.condition != "paired":
        parser.error("human run is fixed to paired formal20 mode")
    if remaining:
        parser.error("--human-run accepts only --condition paired and --quiet")
    result = run_human_evaluation(
        condition="paired",
        formal=True,
        progress=not args.quiet,
    )
    evaluator_module = __import__(
        "src.experiments.single_uav_goal_recovery_correction_evaluation",
        fromlist=["_jsonable"],
    )
    print(json.dumps(evaluator_module._jsonable(result), ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "ALL_EPISODE_IDS",
    "CANONICAL_ARTIFACT_FILES",
    "FAILURE_EPISODE_IDS",
    "FRAME_SOURCE",
    "MAX_STEPS",
    "PARENT_LATERAL_EXIT_PREREGISTRATION_HASH",
    "PREPARATION_ROOT",
    "RADIAL_ASSIST_HARD_CAP_MPS",
    "RADIAL_COMMAND_DEADBAND_MPS",
    "RADIAL_INTENT_OFF_BASELINE_MISMATCH",
    "RadialIntentDecision",
    "RadialIntentPersistenceAssistance",
    "TARGET_FLOOR_MPS",
    "assert_paired_identity",
    "assert_radial_intent_exclusive",
    "baseline_trace_fingerprint",
    "build_off_baseline_contract",
    "build_preparation_bundle",
    "derive_assistance_cap",
    "load_runtime_locks",
    "paired_itt_endpoint_summary",
    "run_human_evaluation",
    "validate_off_against_frozen_baseline",
    "transform_radial_assisted_delta_v_once",
    "write_preparation_artifacts",
]


if __name__ == "__main__":  # pragma: no cover - human-invoked only.
    raise SystemExit(main())
