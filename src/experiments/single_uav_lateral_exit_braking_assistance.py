"""Preparation contract for the paired lateral-exit braking diagnostic.

The CLI is intentionally runnable by a human later, but importing this module is
pure: it constructs no environment, loads no checkpoint, and performs no rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE_NAME = "single_uav_lateral_exit_braking_assistance_diagnostic"
SCHEMA_VERSION = "lateral_exit_braking_assistance_v1"
OFF_TRACE_SCHEMA_VERSION = "lateral_exit_off_baseline_trace_v1"
OFF_BASELINE_MISMATCH = "LATERAL_EXIT_OFF_BASELINE_MISMATCH"
NOT_INDEPENDENTLY_VERIFIED = "NOT_INDEPENDENTLY_VERIFIED"
SUPERSEDED_PRE_RUN_PREREGISTRATION_HASH = (
    "6804dc2e8207da57bbcfb202a7fb936379ea5a815adae771b74ceefd060a1679"
)
FROZEN_OFF_BASELINE_FINGERPRINT = (
    "d0d6ec4891d314923c01be839cc3def1d071462cbccff1971a0f0e22ebcfd07b"
)
ASSIST_CAP_MPS = 0.075
LATERAL_EXIT_THRESHOLD_MPS = 0.5
FORMAL_EPISODES = 20
MAX_STEPS = 120

SIGNED_BASELINE_ROOT = Path(
    "logs/preliminary/single_uav_signed_bypass_observability_diagnostic_seed2026"
)
AUTHORITY_AUDIT_SUMMARY = Path(
    "logs/preliminary/single_uav_lateral_exit_authority_audit_seed2026/"
    "lateral_exit_authority_audit_summary.json"
)
LOG_ROOT = Path(
    "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
DATA_ROOT = Path(
    "data/synthetic/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
PLOT_ROOT = Path(
    "plots/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)

BASELINE_FILES = (
    "signed_bypass_summary.json",
    "signed_bypass_episodes.csv",
    "signed_bypass_steps.csv",
    "signed_bypass_focus_failures.csv",
    "signed_bypass_direction_transitions.csv",
)

# Versioned projection of every pre-existing behavior field needed for the OFF
# equivalence claim.  Newly introduced assistance-only columns are absent by design.
OFF_TRACE_KEYS = (
    "episode_index", "episode_id", "scenario_seed", "noise_seed", "step",
    "actor_raw_delta_v_mps", "goal_recovery_correction_enabled",
    "goal_recovery_correction_eligible", "goal_recovery_correction_active",
    "goal_recovery_correction_reason", "goal_recovery_correction_alpha",
    "goal_recovery_correction_delta_v_mps", "goal_recovery_applied_delta_v_mps",
    "corrected_delta_v_pre_transform_mps", "corrected_delta_v_pre_safety_mps",
    "applied_delta_v_mps", "motion_boundary_delta_v_mps",
    "safety_projection_delta_v_mps", "safety_projection_applied",
    "boundary_projection_applied", "action_clipped", "action_transform_applied",
    "estimated_radial_velocity_mps", "estimated_lateral_velocity_mps",
    "estimated_vertical_velocity_mps", "estimated_radial_displacement_step_m",
    "estimated_lateral_displacement_step_m", "estimated_vertical_displacement_step_m",
    "goal_distance_m", "goal_distance_after_m", "passed_obstacle",
    "post_bypass_recovery", "stable_goal", "reward", "collision", "terminated",
    "truncated", "termination_reason",
)

PREREGISTERED_ENDPOINTS: dict[str, Any] = {
    "schema_version": "lateral_exit_braking_endpoints_v2_pre_run_validity_fix",
    "estimand": "paired_ON_minus_OFF",
    "population": "all_20_ordered_episodes_intention_to_treat",
    "primary": {
        "passed_obstacle": {"type": "binary", "better": "higher"},
        "collision": {"type": "binary", "better": "lower"},
        "maximum_initial_direction_lateral_excursion_from_encounter_m": {
            "type": "continuous", "better": "lower"
        },
        "steps_to_lateral_exit_threshold_0p5_mps": {
            "type": "discrete_time",
            "threshold_mps": LATERAL_EXIT_THRESHOLD_MPS,
            "censoring": "right_censored_at_termination_or_horizon",
            "better": "lower",
        },
        "minimum_frozen_obstacle_clearance_m": {
            "type": "continuous", "better": "higher"
        },
    },
    "secondary_off_triggered": {
        "selection": "OFF_first_fully_eligible_step_only",
        "on_anchor": "same_absolute_step_as_OFF",
        "common_observation_window": "[OFF_anchor,min(OFF_last_step,ON_last_step)]",
        "no_on_reselection": True,
        "outcomes": (
            "extra_lateral_drift_after_assist_m",
            "steps_to_lateral_exit_threshold_0p5_mps_after_common_anchor",
            "after_common_anchor_threshold_censored",
            "after_common_anchor_threshold_status",
            "integrated_abs_lateral_velocity_after_assist",
            "mean_lateral_velocity_decay_rate",
            "peak_lateral_velocity",
            "lateral_displacement_after_assistance_activation",
            "time_to_50pct_lateral_speed_reduction",
            "passed_obstacle", "post_bypass_recovery", "entered_1m", "stable_goal",
            "collision", "timeout", "safety_projection_steps",
            "boundary_projection_steps",
        ),
        "missing_categories": (
            "OFF_NEVER_ELIGIBLE", "ON_TERMINATED_BEFORE_OFF_ANCHOR",
            "RIGHT_CENSORED_AT_COMMON_WINDOW_END",
            "THRESHOLD_REACHED_WITHIN_COMMON_WINDOW",
        ),
    },
    "prior_special_cases": ("test_r99_episode_0013", "test_r99_episode_0014"),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def preregistration_hash() -> str:
    return hashlib.sha256(_canonical(PREREGISTERED_ENDPOINTS).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def load_frozen_baseline(
    baseline_root: Path = SIGNED_BASELINE_ROOT,
    audit_summary_path: Path = AUTHORITY_AUDIT_SUMMARY,
) -> dict[str, Any]:
    """Load and hash-check the immutable signed-observability reference."""

    try:
        audit = json.loads(audit_summary_path.read_text(encoding="utf-8"))
        expected = audit["source_hashes"]["before"]
        actual = {name: _sha256(baseline_root / name) for name in BASELINE_FILES}
        if actual != expected or audit.get("source_hashes_unchanged") is not True:
            raise ValueError("source hash mismatch")
        summary = json.loads(
            (baseline_root / "signed_bypass_summary.json").read_text(encoding="utf-8")
        )
        episodes = _csv_rows(baseline_root / "signed_bypass_episodes.csv")
        steps = _csv_rows(baseline_root / "signed_bypass_steps.csv")
        if (
            int(summary.get("episode_count", -1)) != FORMAL_EPISODES
            or len(episodes) != FORMAL_EPISODES
            or int(summary.get("step_count", -1)) != len(steps)
            or summary.get("only_logging_changed") is not True
        ):
            raise ValueError("baseline shape or identity mismatch")
        return {
            "summary": summary,
            "episodes": episodes,
            "steps": steps,
            "hashes": actual,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(OFF_BASELINE_MISMATCH) from exc


def _normalize(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return _normalize(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        try:
            number = float(stripped)
            return round(number, 12) if math.isfinite(number) else None
        except ValueError:
            return stripped
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        return round(number, 12) if math.isfinite(number) else None
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    return str(value)


def canonical_off_trace(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [[_normalize(row.get(key)) for key in OFF_TRACE_KEYS] for row in rows]


def off_trace_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    envelope = {
        "schema_version": OFF_TRACE_SCHEMA_VERSION,
        "keys": OFF_TRACE_KEYS,
        "rows": canonical_off_trace(rows),
    }
    return hashlib.sha256(_canonical(envelope).encode("utf-8")).hexdigest()


def _episode_identity(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            int(float(row["episode_index"])), str(row["episode_id"]),
            int(float(row["scenario_seed"])), int(float(row["noise_seed"])),
            str(row.get("scenario_hash", "")), int(float(row.get("max_steps", MAX_STEPS))),
        )
        for row in rows
    ]


def validate_off_against_frozen_baseline(
    *,
    current_episodes: Sequence[Mapping[str, Any]],
    current_steps: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        baseline_episodes = baseline["episodes"]
        baseline_steps = baseline["steps"]
        if len(current_episodes) != FORMAL_EPISODES:
            raise ValueError("not formal20")
        if _episode_identity(current_episodes) != _episode_identity(baseline_episodes):
            raise ValueError("episode identity mismatch")
        current_fp = off_trace_fingerprint(current_steps)
        baseline_fp = off_trace_fingerprint(baseline_steps)
        if current_fp != baseline_fp:
            raise ValueError("step trace mismatch")
        return {
            "status": "PASS",
            "schema_version": OFF_TRACE_SCHEMA_VERSION,
            "row_count": len(current_steps),
            "fingerprint": current_fp,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(OFF_BASELINE_MISMATCH) from exc


def assert_paired_identity(
    off_episodes: Sequence[Mapping[str, Any]],
    on_episodes: Sequence[Mapping[str, Any]],
) -> None:
    if _episode_identity(off_episodes) != _episode_identity(on_episodes):
        raise RuntimeError("LATERAL_EXIT_PAIRED_IDENTITY_MISMATCH")


def build_paired_identity_report(
    *,
    off_result: Mapping[str, Any],
    on_result: Mapping[str, Any],
    runtime_hashes: Mapping[str, Any],
    expected_observation_dimension: int,
    expected_max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Separate evidence statuses from booleans used for fail-fast decisions."""

    off_provenance = off_result.get("provenance", {})
    on_provenance = on_result.get("provenance", {})
    off_checkpoint = off_provenance.get("checkpoint", {})
    on_checkpoint = on_provenance.get("checkpoint", {})
    off_episodes = off_result.get("episodes", ())
    on_episodes = on_result.get("episodes", ())

    off_checkpoint_sha = str(off_checkpoint.get("sha256", ""))
    on_checkpoint_sha = str(on_checkpoint.get("sha256", ""))
    off_policy_hash = str(off_provenance.get("policy_hash", ""))
    on_policy_hash = str(on_provenance.get("policy_hash", ""))
    off_manifest_hash = str(off_provenance.get("scenario_manifest_hash", ""))
    on_manifest_hash = str(on_provenance.get("scenario_manifest_hash", ""))
    off_controller_hash = str(off_provenance.get("controller_config_hash", ""))
    on_controller_hash = str(on_provenance.get("controller_config_hash", ""))
    off_goal_recovery_hash = str(off_provenance.get("goal_recovery_config_hash", ""))
    on_goal_recovery_hash = str(on_provenance.get("goal_recovery_config_hash", ""))
    off_observation_dimension = _integer(
        off_provenance, "observation_dimension", default=-1
    )
    on_observation_dimension = _integer(
        on_provenance, "observation_dimension", default=-1
    )
    off_max_steps = {_integer(row, "max_steps", default=-1) for row in off_episodes}
    on_max_steps = {_integer(row, "max_steps", default=-1) for row in on_episodes}
    before_runtime = runtime_hashes.get("before_off_or_first_condition")
    after_off_runtime = runtime_hashes.get("after_off")
    after_on_runtime = runtime_hashes.get("after_on")
    off_baseline_check = off_provenance.get("off_baseline_check", {})
    off_fingerprint = str(off_baseline_check.get("fingerprint", ""))
    off_treatment = off_provenance.get("lateral_exit_treatment_config", {})
    on_treatment = on_provenance.get("lateral_exit_treatment_config", {})

    guards = {
        "same_checkpoint_sha256": bool(off_checkpoint_sha)
        and off_checkpoint_sha == on_checkpoint_sha,
        "same_policy_hash": bool(off_policy_hash)
        and off_policy_hash == on_policy_hash,
        "same_observation_dimension": (
            off_observation_dimension
            == on_observation_dimension
            == int(expected_observation_dimension)
        ),
        "same_ordered_episode_identity": bool(off_episodes)
        and _episode_identity(off_episodes) == _episode_identity(on_episodes),
        "same_scenario_manifest": bool(off_manifest_hash)
        and off_manifest_hash == on_manifest_hash,
        "same_max_steps": (
            off_max_steps == on_max_steps == {int(expected_max_steps)}
        ),
        "same_controller_config": bool(off_controller_hash)
        and off_controller_hash == on_controller_hash,
        "same_goal_recovery_config": bool(off_goal_recovery_hash)
        and off_goal_recovery_hash == on_goal_recovery_hash,
        "same_frozen_runtime_components": before_runtime is not None
        and before_runtime == after_off_runtime == after_on_runtime,
        "off_trajectory_fingerprint_match": (
            off_baseline_check.get("status") == "PASS"
            and off_fingerprint == FROZEN_OFF_BASELINE_FINGERPRINT
        ),
        "on_off_treatment_config_diff_is_exact": (
            off_treatment
            == {
                "enabled": False,
                "assist_cap_mps": ASSIST_CAP_MPS,
            }
            and on_treatment
            == {
                "enabled": True,
                "assist_cap_mps": ASSIST_CAP_MPS,
            }
        ),
    }
    statuses: dict[str, Any] = dict(guards)
    statuses.update(
        {
            "reward_identity": NOT_INDEPENDENTLY_VERIFIED,
            "dynamics_identity": NOT_INDEPENDENTLY_VERIFIED,
            "action_transform_identity": NOT_INDEPENDENTLY_VERIFIED,
            "safety_projection_identity": NOT_INDEPENDENTLY_VERIFIED,
            "boundary_projection_identity": NOT_INDEPENDENTLY_VERIFIED,
            "behavioral_equivalence_protection": (
                "OFF_EXACT_TRAJECTORY_FINGERPRINT"
            ),
        }
    )
    only_treatment_changed = all(guards.values())
    return {
        "identity_status": statuses,
        "fail_fast_boolean_guards": guards,
        "only_lateral_exit_braking_assistance_changed": only_treatment_changed,
    }


def assert_paired_identity_report(report: Mapping[str, Any]) -> None:
    """Fail only on explicit booleans; descriptive strings never enter truth tests."""

    guards = report.get("fail_fast_boolean_guards", {})
    if (
        not guards
        or not all(value is True for value in guards.values())
        or report.get("only_lateral_exit_braking_assistance_changed") is not True
    ):
        raise RuntimeError("LATERAL_EXIT_PAIRED_IDENTITY_MISMATCH")


def _number(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _flag(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(str(value).strip().lower() == "true")


def _integer(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def aggregate_episode_endpoints(
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor_step: int | None = None,
    common_end_step: int | None = None,
    dt_s: float = 0.2,
) -> dict[str, Any]:
    """Aggregate preregistered endpoints without selecting on ON outcomes."""

    ordered = sorted(rows, key=lambda row: int(float(row.get("step", 0))))
    if not ordered:
        return {"status": "NO_ROWS"}
    encounter_values = [
        int(float(row["lateral_exit_assist_encounter_step"]))
        for row in ordered
        if str(row.get("lateral_exit_assist_encounter_step", "")) not in {"", "None"}
    ]
    encounter_step = min(encounter_values) if encounter_values else None
    initial_sign = next(
        (
            _integer(row, "lateral_exit_assist_initial_lateral_sign")
            for row in ordered
            if _integer(row, "lateral_exit_assist_initial_lateral_sign") != 0
        ),
        0,
    )
    encounter_tail = [
        row for row in ordered
        if encounter_step is not None and int(float(row.get("step", 0))) >= encounter_step
    ]
    velocities = [
        _number(row, "lateral_exit_outcome_actual_lateral_velocity_mps")
        for row in encounter_tail
    ]
    displacements = [
        _number(
            row,
            "lateral_exit_outcome_actual_lateral_displacement_from_encounter_m",
        )
        for row in encounter_tail
    ]
    finite_velocity_rows = [
        (row, value) for row, value in zip(encounter_tail, velocities)
        if math.isfinite(value)
    ]
    threshold_row = next(
        (
            row for row, value in finite_velocity_rows
            if abs(value) <= LATERAL_EXIT_THRESHOLD_MPS
        ),
        None,
    )
    signed_excursions = [
        initial_sign * value for value in displacements if math.isfinite(value)
    ] if initial_sign else []
    last_step = int(float(ordered[-1].get("step", 0)))
    effective_common_end = (
        min(int(common_end_step), last_step)
        if common_end_step is not None else last_step
    )
    common_tail = (
        [
            row for row in ordered
            if anchor_step <= int(float(row.get("step", 0))) <= effective_common_end
        ]
        if anchor_step is not None and effective_common_end >= anchor_step else []
    )
    common_velocities = [
        _number(row, "lateral_exit_outcome_actual_lateral_velocity_mps")
        for row in common_tail
    ]
    common_velocities = [value for value in common_velocities if math.isfinite(value)]
    common_displacements = [
        _number(
            row,
            "lateral_exit_outcome_actual_lateral_displacement_from_encounter_m",
        )
        for row in common_tail
    ]
    common_displacements = [value for value in common_displacements if math.isfinite(value)]
    half_step = None
    if common_velocities:
        half_threshold = 0.5 * abs(common_velocities[0])
        for row in common_tail:
            value = _number(row, "lateral_exit_outcome_actual_lateral_velocity_mps")
            if math.isfinite(value) and abs(value) <= half_threshold:
                half_step = int(float(row["step"])) - int(anchor_step)
                break
    post_anchor_threshold = None
    if anchor_step is not None:
        for row in common_tail:
            value = _number(row, "lateral_exit_outcome_actual_lateral_velocity_mps")
            if math.isfinite(value) and abs(value) <= LATERAL_EXIT_THRESHOLD_MPS:
                post_anchor_threshold = int(float(row["step"])) - int(anchor_step)
                break
    min_clearance_values = [
        _number(row, "lateral_exit_assist_minimum_clearance_m") for row in ordered
    ]
    min_clearance_values = [value for value in min_clearance_values if math.isfinite(value)]
    common_threshold_status = (
        "THRESHOLD_REACHED_WITHIN_COMMON_WINDOW"
        if post_anchor_threshold is not None
        else "RIGHT_CENSORED_AT_COMMON_WINDOW_END"
        if anchor_step is not None and common_tail
        else ""
    )
    return {
        "status": "OK",
        "encounter_step": encounter_step if encounter_step is not None else "",
        "initial_lateral_sign": initial_sign,
        "passed_obstacle": int(any(_flag(row, "passed_obstacle") for row in ordered)),
        "post_bypass_recovery": int(any(_flag(row, "post_bypass_recovery") for row in ordered)),
        "entered_1m": int(any(_number(row, "goal_distance_after_m") <= 1.0 for row in ordered)),
        "stable_goal": int(any(_flag(row, "stable_goal") for row in ordered)),
        "collision": int(any(_flag(row, "collision") for row in ordered)),
        "timeout": int(
            not any(_flag(row, "collision") or _flag(row, "stable_goal") for row in ordered)
            and (len(ordered) >= MAX_STEPS or str(ordered[-1].get("termination_reason")) == "time_limit")
        ),
        "safety_projection_steps": sum(_flag(row, "safety_projection_applied") for row in ordered),
        "boundary_projection_steps": sum(_flag(row, "boundary_projection_applied") for row in ordered),
        "maximum_initial_direction_lateral_excursion_from_encounter_m": (
            max(signed_excursions, default=float("nan"))
        ),
        "steps_to_lateral_exit_threshold_0p5_mps": (
            int(float(threshold_row["step"])) - int(encounter_step)
            if threshold_row is not None and encounter_step is not None else ""
        ),
        "lateral_exit_threshold_0p5_mps_censored": int(threshold_row is None),
        "lateral_exit_threshold_0p5_mps_censor_step": last_step,
        "minimum_frozen_obstacle_clearance_m": min(
            min_clearance_values, default=float("nan")
        ),
        "anchor_step": anchor_step if anchor_step is not None else "",
        "common_window_end_step": (
            effective_common_end if anchor_step is not None and common_tail else ""
        ),
        "integrated_abs_lateral_velocity_after_assist": (
            sum(abs(value) for value in common_velocities) * float(dt_s)
            if common_velocities else float("nan")
        ),
        "mean_lateral_velocity_decay_rate": (
            (abs(common_velocities[0]) - abs(common_velocities[-1]))
            / max(len(common_velocities) - 1, 1)
            if common_velocities else float("nan")
        ),
        "peak_lateral_velocity": (
            max((abs(value) for value in common_velocities), default=float("nan"))
        ),
        "lateral_displacement_after_assistance_activation": (
            common_displacements[-1] - common_displacements[0]
            if len(common_displacements) >= 2 else float("nan")
        ),
        "extra_lateral_drift_after_assist_m": (
            max(
                (
                    initial_sign * (value - common_displacements[0])
                    for value in common_displacements
                ),
                default=float("nan"),
            )
            if initial_sign and common_displacements else float("nan")
        ),
        "time_to_50pct_lateral_speed_reduction": (
            half_step if half_step is not None else ""
        ),
        "steps_to_lateral_exit_threshold_0p5_mps_after_common_anchor": (
            post_anchor_threshold if post_anchor_threshold is not None else ""
        ),
        "after_common_anchor_threshold_censored": (
            int(post_anchor_threshold is None) if common_tail else ""
        ),
        "after_common_anchor_threshold_status": common_threshold_status,
    }


def build_paired_endpoint_rows(
    off_episodes: Sequence[Mapping[str, Any]],
    on_episodes: Sequence[Mapping[str, Any]],
    off_steps: Sequence[Mapping[str, Any]],
    on_steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build ITT pairs and the OFF-triggered secondary cohort."""

    assert_paired_identity(off_episodes, on_episodes)
    off_by_id: dict[str, list[Mapping[str, Any]]] = {}
    on_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in off_steps:
        off_by_id.setdefault(str(row["episode_id"]), []).append(row)
    for row in on_steps:
        on_by_id.setdefault(str(row["episode_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for off_episode, on_episode in zip(off_episodes, on_episodes):
        episode_id = str(off_episode["episode_id"])
        off_rows = off_by_id.get(episode_id, [])
        on_rows = on_by_id.get(episode_id, [])
        eligible = [
            int(float(row["step"])) for row in off_rows
            if _flag(row, "lateral_exit_assist_eligible")
        ]
        anchor = min(eligible) if eligible else None
        off_last = max((int(float(row["step"])) for row in off_rows), default=-1)
        on_last = max((int(float(row["step"])) for row in on_rows), default=-1)
        missing = (
            "OFF_NEVER_ELIGIBLE" if anchor is None
            else "ON_TERMINATED_BEFORE_OFF_ANCHOR" if on_last < anchor
            else ""
        )
        common_end = min(off_last, on_last) if not missing else None
        secondary_anchor = anchor if not missing else None
        off_metrics = aggregate_episode_endpoints(
            off_rows,
            anchor_step=secondary_anchor,
            common_end_step=common_end,
        )
        on_metrics = aggregate_episode_endpoints(
            on_rows,
            anchor_step=secondary_anchor,
            common_end_step=common_end,
        )
        pair: dict[str, Any] = {
            "episode_index": int(float(off_episode["episode_index"])),
            "episode_id": episode_id,
            "scenario_seed": int(float(off_episode["scenario_seed"])),
            "noise_seed": int(float(off_episode["noise_seed"])),
            "off_defined_anchor_step": anchor if anchor is not None else "",
            "common_window_end_step": common_end if common_end is not None else "",
            "triggered_cohort_status": missing or "INCLUDED",
            "prior_special_case": int(episode_id in PREREGISTERED_ENDPOINTS["prior_special_cases"]),
        }
        for name in PREREGISTERED_ENDPOINTS["primary"]:
            off_value = off_metrics.get(name)
            on_value = on_metrics.get(name)
            pair[f"{name}_OFF"] = off_value
            pair[f"{name}_ON"] = on_value
            if isinstance(off_value, (int, float)) and isinstance(on_value, (int, float)):
                if math.isfinite(float(off_value)) and math.isfinite(float(on_value)):
                    pair[f"{name}_ON_minus_OFF"] = float(on_value) - float(off_value)
        for name in PREREGISTERED_ENDPOINTS["secondary_off_triggered"]["outcomes"]:
            pair[f"{name}_OFF"] = off_metrics.get(name, off_episode.get(name, ""))
            pair[f"{name}_ON"] = on_metrics.get(name, on_episode.get(name, ""))
        output.append(pair)
    if len(output) != FORMAL_EPISODES:
        raise RuntimeError("LATERAL_EXIT_PAIRED_IDENTITY_MISMATCH")
    return output


def write_diagnostic_summary(
    *,
    output_root: Path,
    results: Mapping[str, Mapping[str, Any]],
    identity_report: Mapping[str, Any],
    runtime_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the paired summary after both conditions have completed."""

    root = Path(output_root)
    if "off" not in results or "on" not in results:
        return {
            "stage": STAGE_NAME,
            "condition": next(iter(results), ""),
            "paired_summary_written": False,
        }
    pairs = build_paired_endpoint_rows(
        results["off"]["episodes"], results["on"]["episodes"],
        results["off"]["step_rows"], results["on"]["step_rows"],
    )
    endpoints = {
        **PREREGISTERED_ENDPOINTS,
        "active_preregistration_hash": preregistration_hash(),
        "superseded_pre_run_preregistration_hash": (
            SUPERSEDED_PRE_RUN_PREREGISTRATION_HASH
        ),
        "preregistration_revised_before_any_off_on_rollout": True,
    }
    payload = {
        "stage": STAGE_NAME,
        "schema_version": SCHEMA_VERSION,
        "condition": "paired",
        "episode_count": len(pairs),
        "lateral_exit_assist_cap_mps": ASSIST_CAP_MPS,
        "active_preregistration_hash": preregistration_hash(),
        "superseded_pre_run_preregistration_hash": (
            SUPERSEDED_PRE_RUN_PREREGISTRATION_HASH
        ),
        "preregistration_revised_before_any_off_on_rollout": True,
        "only_lateral_exit_braking_assistance_changed": bool(
            identity_report.get("only_lateral_exit_braking_assistance_changed", False)
        ),
        "identity_lock": dict(identity_report.get("identity_status", {})),
        "identity_fail_fast_boolean_guards": dict(
            identity_report.get("fail_fast_boolean_guards", {})
        ),
        "frozen_runtime_component_hashes": dict(runtime_hashes),
        "flags": {
            "lateral_exit_braking_diagnostic_prepared": True,
            "actor_modified": False, "reward_modified": False,
            "dynamics_modified": False, "goal_recovery_modified": False,
            "terminal_controller_modified": False, "safety_modified": False,
            "boundary_modified": False, "training_performed": False,
        },
    }
    with (root / "lateral_exit_braking_assistance_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = list(dict.fromkeys(key for row in pairs for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    (root / "preregistered_endpoints.json").write_text(
        json.dumps(endpoints, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "lateral_exit_braking_assistance_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--condition", choices=("paired", "off", "on"), default="paired")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Lazy import keeps parser/tests pure and makes runtime execution explicit.
    from .single_uav_goal_recovery_correction_evaluation import (
        run_goal_recovery_correction_evaluation,
    )

    result = run_goal_recovery_correction_evaluation(
        condition=args.condition,
        formal=True,
        lateral_exit_braking_assistance_diagnostic=True,
        progress=True,
    )
    print("LATERAL_EXIT_BRAKING_DIAGNOSTIC_READY")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSIST_CAP_MPS", "LATERAL_EXIT_THRESHOLD_MPS", "OFF_BASELINE_MISMATCH",
    "NOT_INDEPENDENTLY_VERIFIED", "SUPERSEDED_PRE_RUN_PREREGISTRATION_HASH",
    "FROZEN_OFF_BASELINE_FINGERPRINT", "OFF_TRACE_KEYS", "PREREGISTERED_ENDPOINTS",
    "assert_paired_identity", "build_paired_identity_report",
    "assert_paired_identity_report",
    "build_parser", "canonical_off_trace", "load_frozen_baseline",
    "off_trace_fingerprint", "preregistration_hash", "aggregate_episode_endpoints",
    "build_paired_endpoint_rows", "write_diagnostic_summary",
    "validate_off_against_frozen_baseline",
]
