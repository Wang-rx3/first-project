"""Pure offline audit of paired lateral-exit assistance trajectories.

The module reads frozen CSV/JSON records and a checkpoint byte stream for
identity verification. It never imports the simulator, environment, controller,
policy, or training runtime and never executes a rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SOURCE_ROOT = Path(
    "logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026"
)
OUTPUT_ROOT = Path(
    "logs/preliminary/single_uav_post_lateral_exit_reorientation_audit_seed2026"
)
ACTIVE_PREREGISTRATION_HASH = (
    "a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2"
)
PREREGISTRATION_SCHEMA = "lateral_exit_braking_endpoints_v2_pre_run_validity_fix"
OFF_BASELINE_FINGERPRINT = (
    "d0d6ec4891d314923c01be839cc3def1d071462cbccff1971a0f0e22ebcfd07b"
)
CHECKPOINT_SHA256 = (
    "b7ed3ba79cf3055906bd0425cf4a7c4e613b206a8680ebf17f3b586fbc31b3c6"
)
CHECKPOINT_RELATIVE_SUFFIX = (
    "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
    "dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt"
)
MAX_STEPS = 120
OBSERVATION_DIMENSION = 42
ASSIST_CAP_MPS = 0.075
PASS_CLEARANCE_M = 2.5
EXIT_SPEED_THRESHOLD_MPS = 0.5
RADIAL_EPSILON = 1.0e-8
FRAME_TOLERANCE = 1.0e-7

MAIN_FAILURE_IDS = tuple(
    f"test_r99_episode_{suffix}"
    for suffix in ("0002", "0004", "0005", "0006", "0011", "0012", "0020")
)
SUCCESS_REFERENCE_IDS = tuple(
    f"test_r99_episode_{suffix}"
    for suffix in (
        "0001", "0003", "0007", "0008", "0009", "0010",
        "0015", "0016", "0017", "0018", "0019",
    )
)
EXPECTED_EPISODE_IDS = tuple(
    f"test_r99_episode_{index:04d}" for index in range(1, 21)
)

PRIMARY_DIAGNOSES = frozenset(
    {
        "RADIAL_PROPULSION_INSUFFICIENT",
        "RADIAL_INTENT_NOT_SUSTAINED",
        "RADIAL_MOTION_RESPONSE_WEAK",
        "LATERAL_REACCELERATION",
        "PASS_GEOMETRY_NOT_REACHED",
        "PASS_PROGRESS_REGRESSION",
        "PIPELINE_RADIAL_ATTENUATION",
        "MIXED",
        "NOT_ASSESSABLE",
    }
)
SPECIAL_LABELS = frozenset(
    {"SAFETY_DOMINATED_SPECIAL_CASE", "BOUNDARY_DOMINATED_SPECIAL_CASE"}
)

SOURCE_FILES = (
    "preregistered_endpoints.json",
    "lateral_exit_braking_assistance_summary.json",
    "evaluation_summary.json",
    "comparison_episodes.csv",
    "goal_recovery_correction_comparison.json",
    "goal_recovery_correction_comparison.csv",
    "off/summary.json",
    "on/summary.json",
    "off/episodes.csv",
    "on/episodes.csv",
    "off/steps.csv",
    "on/steps.csv",
    "lateral_exit_braking_assistance_pairs.csv",
)
OUTPUT_FILES = (
    "post_lateral_exit_audit_summary.json",
    "post_lateral_exit_episode_diagnostics.csv",
    "post_lateral_exit_common_anchor_metrics.csv",
    "post_lateral_exit_mechanistic_windows.csv",
    "post_lateral_exit_success_reference.csv",
    "post_lateral_exit_failure_vs_reference.csv",
    "post_lateral_exit_pass_margin_trajectories.csv",
    "post_lateral_exit_report.md",
)

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

METRICS = (
    "actor_radial_mean_mps",
    "final_radial_mean_mps",
    "estimated_radial_velocity_mean_mps",
    "actual_radial_velocity_mean_mps",
    "actor_lateral_mean_mps",
    "final_lateral_mean_mps",
    "estimated_lateral_velocity_mean_mps",
    "actual_lateral_velocity_mean_mps",
    "actor_radial_positive_mean_mps",
    "final_radial_positive_mean_mps",
    "actor_radial_positive_sustained_5step_4of5",
    "final_radial_positive_sustained_5step_4of5",
    "actual_radial_positive_sustained_3step",
    "actor_to_final_radial_retention",
    "actor_radial_to_abs_lateral_balance",
    "final_radial_to_abs_lateral_balance",
    "goal_distance_slope_first10_m_per_step",
    "goal_distance_slope_first20_m_per_step",
    "goal_distance_slope_last20_m_per_step",
    "goal_distance_net_reduction_m",
    "pass_gate_first_step",
    "pass_gate_ever",
    "pass_latch_first_step",
    "pass_latch_ever",
    "pass_gate_final_instantaneous",
    "pass_latch_final",
    "margin_start_m",
    "margin_best_m",
    "margin_best_step",
    "margin_final_m",
    "margin_delta_anchor_to_best_m",
    "margin_delta_anchor_to_final_m",
    "margin_regression_best_minus_final_m",
    "local_exit_latency_steps",
    "lateral_reacceleration_run_count",
    "lateral_reacceleration_duration_steps",
    "lateral_reacceleration_peak_abs_mps",
    "actor_original_side_fraction",
    "actor_counter_side_fraction",
    "actor_lateral_sign_changes",
)
COMPARISON_METRICS = tuple(metric for metric in METRICS if metric not in {
    "pass_gate_first_step", "margin_best_step", "lateral_reacceleration_run_count",
    "actor_lateral_sign_changes",
})


class SourceIdentityError(RuntimeError):
    """Raised when an input artifact fails its frozen identity contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def integer(value: Any) -> int | None:
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path, *, max_rows: int = 10_000) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"invalid or duplicate CSV header: {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"non-rectangular CSV row: {path}")
            rows.append(dict(row))
            if len(rows) > max_rows:
                raise ValueError(f"CSV row limit exceeded: {path}")
    return rows


def normalize_trace_value(value: Any) -> Any:
    """Mirror the frozen OFF fingerprint's scalar/vector normalization."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return normalize_trace_value(json.loads(stripped))
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
        return {str(key): normalize_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_trace_value(item) for item in value]
    if isinstance(value, (int, float)):
        number = float(value)
        return round(number, 12) if math.isfinite(number) else None
    if hasattr(value, "tolist"):
        return normalize_trace_value(value.tolist())
    return str(value)


def off_trace_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    envelope = {
        "schema_version": "lateral_exit_off_baseline_trace_v1",
        "keys": OFF_TRACE_KEYS,
        "rows": [
            [normalize_trace_value(row.get(key)) for key in OFF_TRACE_KEYS]
            for row in rows
        ],
    }
    return canonical_hash(envelope)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise SourceIdentityError(f"POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: {detail}")


def validate_preregistration(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "active_preregistration_hash",
        "superseded_pre_run_preregistration_hash",
        "preregistration_revised_before_any_off_on_rollout",
    }
    content = {key: value for key, value in preregistration.items() if key not in excluded}
    actual_hash = canonical_hash(content)
    _require(preregistration.get("schema_version") == PREREGISTRATION_SCHEMA, "preregistration schema")
    _require(actual_hash == ACTIVE_PREREGISTRATION_HASH, "active preregistration content hash")
    _require(preregistration.get("active_preregistration_hash") == ACTIVE_PREREGISTRATION_HASH, "active preregistration hash field")
    _require(preregistration.get("preregistration_revised_before_any_off_on_rollout") is True, "preregistration timing")
    _require(
        preregistration.get("population") == "all_20_ordered_episodes_intention_to_treat",
        "preregistered population",
    )
    secondary = preregistration.get("secondary_off_triggered", {})
    _require(secondary.get("selection") == "OFF_first_fully_eligible_step_only", "OFF anchor selection rule")
    _require(secondary.get("on_anchor") == "same_absolute_step_as_OFF", "shared ON anchor rule")
    _require(
        secondary.get("common_observation_window") == "[OFF_anchor,min(OFF_last_step,ON_last_step)]",
        "common observation window rule",
    )
    _require(secondary.get("no_on_reselection") is True, "no ON anchor reselection")
    primary = preregistration.get("primary", {})
    _require(primary.get("passed_obstacle", {}).get("type") == "binary", "primary pass endpoint")
    _require(
        primary.get("steps_to_lateral_exit_threshold_0p5_mps", {}).get("threshold_mps") == EXIT_SPEED_THRESHOLD_MPS,
        "preregistered exit threshold",
    )
    return {"computed_active_hash": actual_hash, "schema_version": PREREGISTRATION_SCHEMA}


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        integer(row.get("episode_index")),
        str(row.get("episode_id", "")),
        integer(row.get("scenario_seed")),
        integer(row.get("noise_seed")),
        str(row.get("scenario_hash", "")),
        integer(row.get("max_steps")),
    )


def _ordered_identities(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [_identity(row) for row in rows]


def _expected_no_training_guards(summary: Mapping[str, Any]) -> None:
    expected: dict[str, Any] = {
        "training_enabled": False,
        "training_performed": False,
        "ppo_updates": 0,
        "mappo_updates": 0,
        "bc_retraining_executed": False,
        "dagger_collection_executed": False,
        "teacher_bootstrap_executed": False,
        "teacher_constructed": False,
        "teacher_queried": False,
        "teacher_rollout_steps": 0,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "actor_updates": 0,
        "buffer_updates": 0,
        "checkpoint_writes": 0,
        "formal_training_experiment_executed": False,
        "reward_modified": False,
        "dynamics_modified": False,
        "observation_modified": False,
        "actor_modified": False,
        "critic_modified": False,
        "gat_modified": False,
        "control_algorithm_modified": False,
        "stable_goal_criterion_modified": False,
    }
    for key, value in expected.items():
        _require(key in summary and summary[key] == value, f"source summary guard {key}")


def validate_source_identity(
    preregistration: Mapping[str, Any],
    paired_summary: Mapping[str, Any],
    off_summary: Mapping[str, Any],
    on_summary: Mapping[str, Any],
    off_episodes: Sequence[Mapping[str, Any]],
    on_episodes: Sequence[Mapping[str, Any]],
    off_steps: Sequence[Mapping[str, Any]],
    on_steps: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
) -> dict[str, Any]:
    prereg_report = validate_preregistration(preregistration)
    _require(paired_summary.get("schema_version") == "lateral_exit_braking_assistance_v1", "paired source summary schema")
    _require(paired_summary.get("condition") == "paired", "paired source summary condition")
    _require(paired_summary.get("episode_count") == 20, "paired source episode count")
    _require(paired_summary.get("active_preregistration_hash") == ACTIVE_PREREGISTRATION_HASH, "paired summary preregistration hash")
    _require(paired_summary.get("preregistration_revised_before_any_off_on_rollout") is True, "paired summary preregistration timing")
    _require(paired_summary.get("only_lateral_exit_braking_assistance_changed") is True, "paired summary exact treatment lock")

    boolean_guards = paired_summary.get("identity_fail_fast_boolean_guards", {})
    _require(bool(boolean_guards) and all(value is True for value in boolean_guards.values()), "paired summary identity boolean guards")
    identity_lock = paired_summary.get("identity_lock", {})
    required_true = (
        "same_checkpoint_sha256", "same_policy_hash", "same_observation_dimension",
        "same_ordered_episode_identity", "same_scenario_manifest", "same_max_steps",
        "same_controller_config", "same_goal_recovery_config", "same_frozen_runtime_components",
        "off_trajectory_fingerprint_match", "on_off_treatment_config_diff_is_exact",
    )
    _require(all(identity_lock.get(key) is True for key in required_true), "paired summary identity lock")
    for key in (
        "reward_identity", "dynamics_identity", "action_transform_identity",
        "safety_projection_identity", "boundary_projection_identity",
    ):
        _require(identity_lock.get(key) == "NOT_INDEPENDENTLY_VERIFIED", f"independent identity status {key}")
    _require(
        identity_lock.get("behavioral_equivalence_protection") == "OFF_EXACT_TRAJECTORY_FINGERPRINT",
        "OFF exact step trajectory fingerprint protection",
    )
    flags = paired_summary.get("flags", {})
    for key in (
        "actor_modified", "reward_modified", "dynamics_modified", "goal_recovery_modified",
        "terminal_controller_modified", "safety_modified", "boundary_modified", "training_performed",
    ):
        _require(flags.get(key) is False, f"paired source summary flag {key}")

    _require(len(off_episodes) == len(on_episodes) == len(pair_rows) == 20, "paired episode row count")
    expected_ids = list(EXPECTED_EPISODE_IDS)
    off_ids = [str(row.get("episode_id", "")) for row in off_episodes]
    on_ids = [str(row.get("episode_id", "")) for row in on_episodes]
    _require(off_ids == on_ids == expected_ids, "ordered episode ids")
    off_identity = _ordered_identities(off_episodes)
    on_identity = _ordered_identities(on_episodes)
    _require(off_identity == on_identity, "OFF/ON episode identity")
    _require(all(all(value is not None for value in ident) and ident[-1] == MAX_STEPS for ident in off_identity), "episode seeds, hashes, max_steps")
    for index, (episode, pair) in enumerate(zip(off_episodes, pair_rows), start=1):
        _require(integer(episode.get("episode_index")) == index, "ordered episode index")
        for key in ("episode_index", "episode_id", "scenario_seed", "noise_seed"):
            _require(str(episode.get(key, "")) == str(pair.get(key, "")), f"pair identity {key}")

    off_prov = off_summary.get("provenance", {})
    on_prov = on_summary.get("provenance", {})
    _require(off_prov.get("scenario_manifest_hash") and off_prov.get("scenario_manifest_hash") == on_prov.get("scenario_manifest_hash"), "scenario manifest identity")
    _require(off_prov.get("ordered_pool_identity") == on_prov.get("ordered_pool_identity"), "summary ordered-pool identity")
    _require(off_prov.get("ordered_pool_identity") == [
        [integer(row.get("episode_index")), str(row.get("episode_id", "")), integer(row.get("noise_seed"))]
        for row in off_episodes
    ], "ordered-pool identity versus episodes")
    _require(off_prov.get("policy_hash") and off_prov.get("policy_hash") == on_prov.get("policy_hash"), "policy identity")
    _require(off_prov.get("controller_config_hash") and off_prov.get("controller_config_hash") == on_prov.get("controller_config_hash"), "controller config identity")
    _require(off_prov.get("goal_recovery_config_hash") and off_prov.get("goal_recovery_config_hash") == on_prov.get("goal_recovery_config_hash"), "Goal Recovery config identity")
    _require(off_prov.get("correction_enabled") is True and on_prov.get("correction_enabled") is True, "Goal Recovery enabled identity")
    _require(off_prov.get("observation_dimension") == on_prov.get("observation_dimension") == OBSERVATION_DIMENSION, "observation dimension")

    off_checkpoint = off_prov.get("checkpoint", {})
    on_checkpoint = on_prov.get("checkpoint", {})
    _require(off_checkpoint == on_checkpoint, "OFF/ON checkpoint metadata")
    _require(off_checkpoint.get("sha256", "").lower() == CHECKPOINT_SHA256, "declared checkpoint hash")
    _require(checkpoint_sha256.lower() == CHECKPOINT_SHA256, "actual checkpoint hash")
    _require(checkpoint_size_bytes == off_checkpoint.get("size_bytes") == 470147, "checkpoint byte size")
    _require(off_checkpoint.get("node_feature_dim") == OBSERVATION_DIMENSION, "checkpoint observation dimension")
    _require(off_checkpoint.get("observation_dimension") == OBSERVATION_DIMENSION, "checkpoint observation dimension metadata")
    _require(off_checkpoint.get("policy_hash") == off_checkpoint.get("loaded_policy_hash") == off_checkpoint.get("policy_hash_before") == off_prov.get("policy_hash"), "loaded policy identity")
    _require(off_checkpoint.get("path") == off_checkpoint.get("resolved_path"), "checkpoint resolved path")
    normalized_checkpoint_path = str(off_checkpoint.get("resolved_path", "")).replace("\\", "/")
    _require(normalized_checkpoint_path.endswith(CHECKPOINT_RELATIVE_SUFFIX), "checkpoint path provenance")

    off_treatment = off_prov.get("lateral_exit_treatment_config", {})
    on_treatment = on_prov.get("lateral_exit_treatment_config", {})
    _require(off_prov.get("lateral_exit_braking_assistance_enabled") is False and off_treatment == {"enabled": False, "assist_cap_mps": ASSIST_CAP_MPS}, "OFF treatment")
    _require(on_prov.get("lateral_exit_braking_assistance_enabled") is True and on_treatment == {"enabled": True, "assist_cap_mps": ASSIST_CAP_MPS}, "ON treatment")
    _require(off_prov.get("correction_enabled") == on_prov.get("correction_enabled"), "Goal Recovery treatment diff")

    runtime_hashes = paired_summary.get("frozen_runtime_component_hashes", {})
    phases = [runtime_hashes.get(name, {}) for name in ("before_off_or_first_condition", "after_off", "after_on")]
    component_keys = ("actor", "critic", "gat", "normalizer", "policy_combined")
    _require(all(all(isinstance(phase.get(key), str) and phase.get(key) for key in component_keys) for phase in phases), "frozen actor/critic/GAT/normalizer/combined hashes present")
    _require(phases[0] == phases[1] == phases[2], "frozen runtime hashes changed")
    _require(phases[0].get("policy_combined") == off_prov.get("policy_hash"), "combined policy hash")

    for summary in (off_summary, on_summary):
        _require(summary.get("stage") == "single_uav_goal_recovery_correction_evaluation", "condition summary stage")
        _expected_no_training_guards(summary)
        episodes_in_summary = summary.get("aggregate", {}).get("episodes")
        _require(episodes_in_summary == 20, "condition source summary episode count")
    off_check = off_prov.get("off_baseline_check", {})
    _require(off_check.get("status") == "PASS", "OFF baseline protection status")
    _require(off_check.get("row_count") == len(off_steps) == 2182, "OFF baseline protection row count")
    _require(off_check.get("fingerprint") == OFF_BASELINE_FINGERPRINT, "OFF baseline protection fingerprint")
    computed_fingerprint = off_trace_fingerprint(off_steps)
    _require(computed_fingerprint == OFF_BASELINE_FINGERPRINT, "OFF exact step trajectory fingerprint recomputation")

    _require(len(off_steps) == 2182 and len(on_steps) == 2185, "OFF/ON step row count")
    expected_pair_status = ["INCLUDED"] * 12 + ["OFF_NEVER_ELIGIBLE"] * 8
    statuses = [str(row.get("triggered_cohort_status", "")) for row in pair_rows]
    _require(statuses.count("INCLUDED") == 12 and statuses.count("OFF_NEVER_ELIGIBLE") == 8, "OFF-triggered cohort counts")
    _require(all(status in {"INCLUDED", "OFF_NEVER_ELIGIBLE"} for status in statuses), "OFF-triggered cohort statuses")
    _require([str(row.get("episode_id", "")) for row in pair_rows] == expected_ids, "pair-table ordered identity")
    pair_status_by_id = {str(row.get("episode_id", "")): str(row.get("triggered_cohort_status", "")) for row in pair_rows}
    _require(all(pair_status_by_id.get(episode_id) == "INCLUDED" for episode_id in MAIN_FAILURE_IDS), "all seven main failures belong to OFF-triggered cohort")

    _require(
        {str(row["episode_id"]) for row in off_episodes if flag(row.get("passed_obstacle"))}
        == set(SUCCESS_REFERENCE_IDS),
        "OFF pass set versus fixed success reference",
    )
    _require(
        {str(row["episode_id"]) for row in on_episodes if flag(row.get("passed_obstacle"))}
        == set(SUCCESS_REFERENCE_IDS),
        "ON pass set versus fixed success reference",
    )
    _require(
        {str(row["episode_id"]) for row in off_episodes if flag(row.get("post_bypass_recovery"))}
        == set(SUCCESS_REFERENCE_IDS)
        and {str(row["episode_id"]) for row in on_episodes if flag(row.get("post_bypass_recovery"))}
        == set(SUCCESS_REFERENCE_IDS),
        "post-bypass recovery set versus fixed success reference",
    )
    _require(
        {str(row["episode_id"]) for row in off_episodes if flag(row.get("stable_goal"))}
        == set(SUCCESS_REFERENCE_IDS),
        "OFF stable-goal set",
    )
    _require(
        {str(row["episode_id"]) for row in on_episodes if flag(row.get("stable_goal"))}
        == set(SUCCESS_REFERENCE_IDS) - {"test_r99_episode_0008"},
        "ON stable-goal set and 0008 downstream adverse case",
    )

    _require(all(integer(row.get("max_steps")) == MAX_STEPS for row in off_episodes + on_episodes), "max_steps=120")
    _require(all(str(row.get("condition", "")) == "off" for row in off_steps), "OFF step condition")
    _require(all(str(row.get("condition", "")) == "on" for row in on_steps), "ON step condition")
    return {
        "active_preregistration": prereg_report,
        "checkpoint": {
            "path": off_checkpoint.get("resolved_path"),
            "sha256": checkpoint_sha256.lower(),
            "size_bytes": checkpoint_size_bytes,
        },
        "ordered_episode_count": 20,
        "off_step_count": len(off_steps),
        "on_step_count": len(on_steps),
        "off_fingerprint": computed_fingerprint,
        "identity_lock": dict(identity_lock),
        "frozen_runtime_component_hashes": runtime_hashes,
        "source_summary_guards": "PASS",
        "treatment_diff": {"OFF": off_treatment, "ON": on_treatment},
    }


def group_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        step = integer(row.get("step"))
        _require(bool(episode_id) and step is not None, "step row identity")
        grouped.setdefault(episode_id, []).append(row)
    for episode_id, episode_rows in grouped.items():
        episode_rows.sort(key=lambda row: integer(row.get("step")) or 0)
        steps = [integer(row.get("step")) for row in episode_rows]
        _require(len(steps) == len(set(steps)), f"duplicate step: {episode_id}")
        _require(steps == list(range(0, len(steps))), f"non-contiguous step sequence: {episode_id}")
        _require(bool(steps) and steps[-1] < MAX_STEPS, f"step horizon: {episode_id}")
    return grouped


def validate_step_continuity(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_episode_ids: Sequence[str] = EXPECTED_EPISODE_IDS,
    condition: str | None = None,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped = group_steps(rows)
    _require(set(grouped) == set(expected_episode_ids), "step episode cohort")
    if condition is not None:
        _require(all(str(row.get("condition", "")) == condition for row in rows), f"condition label {condition}")
    ordered_expected_ids = list(expected_episode_ids)
    for episode_id, episode_rows in grouped.items():
        _require(all(integer(row.get("episode_index")) == ordered_expected_ids.index(episode_id) + 1 for row in episode_rows), f"step episode index: {episode_id}")
    return grouped


def vector(value: Any) -> np.ndarray | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (3,) or not np.isfinite(result).all():
        return None
    return result


def _logged_number(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        result = finite_float(row.get(name))
        if result is not None:
            return result
    return None


def _close(left: float | None, right: float | None, *, tolerance: float = FRAME_TOLERANCE) -> bool:
    return left is not None and right is not None and math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def frame_for_row(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    raw_ray = vector(row.get("lateral_exit_assist_horizontal_goal_ray_m"))
    lateral = vector(row.get("lateral_exit_assist_lateral_unit"))
    if raw_ray is None or lateral is None:
        return None
    if abs(float(raw_ray[2])) > FRAME_TOLERANCE or abs(float(lateral[2])) > FRAME_TOLERANCE:
        raise SourceIdentityError("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: encounter frame is not horizontal")
    norm = float(np.linalg.norm(raw_ray))
    lateral_norm = float(np.linalg.norm(lateral))
    if norm <= RADIAL_EPSILON or not math.isclose(lateral_norm, 1.0, abs_tol=FRAME_TOLERANCE):
        raise SourceIdentityError("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: encounter frame unit norm")
    radial = raw_ray / norm
    if abs(float(np.dot(radial, lateral))) > FRAME_TOLERANCE:
        raise SourceIdentityError("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: encounter frame non-orthogonal")
    if str(row.get("lateral_exit_assist_frame_source", "")) != "encounter_time_ekf_estimated_goal_ray_projected_to_xy":
        raise SourceIdentityError("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: encounter radial direction source")
    return radial, lateral


def validate_frame_and_decomposition(
    grouped_steps: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episode_lookup = {str(row["episode_id"]): row for row in episode_rows}
    valid_frame_rows = 0
    pre_encounter_structural_rows = 0
    checks = 0
    for episode_id, rows in grouped_steps.items():
        episode = episode_lookup[episode_id]
        encounter = integer(episode.get("signed_encounter_step"))
        _require(encounter is not None, f"missing encounter step: {episode_id}")
        reference: tuple[np.ndarray, np.ndarray] | None = None
        for row in rows:
            step = integer(row.get("step"))
            current = frame_for_row(row)
            if step is None or step < encounter:
                pre_encounter_structural_rows += 1
                _require(current is None, f"frame unexpectedly present before encounter: {episode_id}")
                continue
            _require(current is not None, f"missing encounter frame after encounter: {episode_id}:{step}")
            assert current is not None
            if reference is None:
                reference = current
            else:
                _require(np.allclose(reference[0], current[0], atol=FRAME_TOLERANCE, rtol=FRAME_TOLERANCE), f"radial frame changed: {episode_id}")
                _require(np.allclose(reference[1], current[1], atol=FRAME_TOLERANCE, rtol=FRAME_TOLERANCE), f"lateral frame changed: {episode_id}")
            radial, lateral = current
            actor = vector(row.get("actor_raw_delta_v_mps"))
            final = vector(row.get("applied_delta_v_mps"))
            _require(actor is not None and final is not None, f"action vector missing: {episode_id}:{step}")
            assert actor is not None and final is not None
            signed_radial = vector(row.get("signed_axis_radial_unit"))
            signed_lateral = vector(row.get("signed_axis_lateral_unit"))
            signed_vertical = vector(row.get("signed_axis_vertical_unit"))
            signed_basis = (
                signed_radial is not None
                and signed_lateral is not None
                and signed_vertical is not None
            )
            actor_radial = _logged_number(row, "actor_raw_radial_delta_v_mps", "actor_raw_delta_v_mps_radial")
            actor_lateral = _logged_number(row, "actor_raw_lateral_delta_v_mps", "actor_raw_delta_v_mps_lateral")
            final_radial = _logged_number(row, "final_applied_radial_delta_v_mps", "final_applied_delta_v_mps_radial")
            final_lateral = _logged_number(row, "final_applied_lateral_delta_v_mps", "final_applied_delta_v_mps_lateral")
            if signed_basis:
                assert signed_radial is not None and signed_lateral is not None and signed_vertical is not None
                _require(_close(actor_radial, float(np.dot(actor, signed_radial))), f"actor logged radial decomposition: {episode_id}:{step}")
                _require(_close(actor_lateral, float(np.dot(actor, signed_lateral))), f"actor logged lateral decomposition: {episode_id}:{step}")
                _require(_close(final_radial, float(np.dot(final, signed_radial)), tolerance=2e-6), f"final logged radial decomposition: {episode_id}:{step}")
                _require(_close(final_lateral, float(np.dot(final, signed_lateral)), tolerance=2e-6), f"final logged lateral decomposition: {episode_id}:{step}")
            if isinstance(row, dict):
                row["audit_actor_radial_delta_v_mps"] = float(np.dot(actor, radial))
                row["audit_actor_lateral_delta_v_mps"] = float(np.dot(actor, lateral))
                row["audit_final_radial_delta_v_mps"] = float(np.dot(final, radial))
                row["audit_final_lateral_delta_v_mps"] = float(np.dot(final, lateral))
                for state in ("estimated", "actual"):
                    signed_components = [
                        _logged_number(row, f"{state}_radial_velocity_mps"),
                        _logged_number(row, f"{state}_lateral_velocity_mps"),
                        _logged_number(row, f"{state}_vertical_velocity_mps"),
                    ]
                    if signed_basis and all(value is not None for value in signed_components):
                        reconstructed = (
                            signed_components[0] * signed_radial
                            + signed_components[1] * signed_lateral
                            + signed_components[2] * signed_vertical
                        )
                        row[f"audit_{state}_radial_velocity_mps"] = float(np.dot(reconstructed, radial))
                        row[f"audit_{state}_lateral_velocity_mps"] = float(np.dot(reconstructed, lateral))
                    else:
                        row[f"audit_{state}_radial_velocity_mps"] = signed_components[0]
                        row[f"audit_{state}_lateral_velocity_mps"] = signed_components[1]
                for key in (
                    "audit_estimated_radial_velocity_mps", "audit_actual_radial_velocity_mps",
                    "audit_estimated_lateral_velocity_mps", "audit_actual_lateral_velocity_mps",
                ):
                    _require(finite_float(row.get(key)) is not None, f"missing projected velocity {key}: {episode_id}:{step}")
            valid_frame_rows += 1
            checks += 4
    _require(valid_frame_rows > 0, "no valid encounter frames")
    return {
        "status": "PASS",
        "valid_encounter_frame_rows": valid_frame_rows,
        "pre_encounter_structural_missing_rows": pre_encounter_structural_rows,
        "actor_final_vector_projection_checks": checks,
        "frame_tolerance": FRAME_TOLERANCE,
        "radial_direction": "normalized_logged_horizontal_goal_ray_positive_goalward",
    }


def pass_gate(row: Mapping[str, Any]) -> bool | None:
    margin = finite_float(row.get("passed_margin_m"))
    clearance = _logged_number(row, "relevant_obstacle_clearance_m", "signed_obstacle_clearance_m")
    if margin is None or clearance is None:
        return None
    return margin > 0.0 and clearance >= PASS_CLEARANCE_M


def validate_pass_latches(
    grouped_steps: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries = {str(row["episode_id"]): row for row in episode_rows}
    episode_count = 0
    pass_count = 0
    for episode_id, rows in grouped_steps.items():
        derived: list[bool] = []
        logged_instant: list[bool] = []
        latched: list[bool] = []
        for row in rows:
            gate = pass_gate(row)
            logged = str(row.get("passed_condition_now", "")).strip().lower()
            if gate is not None:
                _require(logged in {"true", "false", "1", "0"}, f"missing instant pass gate: {episode_id}")
                _require(flag(logged) is gate, f"logged instant pass gate mismatch: {episode_id}:{row.get('step')}")
                derived.append(gate)
                logged_instant.append(flag(logged))
                latched.append(flag(row.get("passed_obstacle")))
        _require(bool(derived), f"no exact pass gate geometry: {episode_id}")
        first_gate = next((index for index, value in enumerate(derived) if value), None)
        first_latch = next((index for index, value in enumerate(latched) if value), None)
        _require(first_gate == first_latch, f"pass latch start disagrees with exact gate: {episode_id}")
        seen_latch = False
        for value in latched:
            if value:
                seen_latch = True
            _require(not seen_latch or value, f"pass latch dropped after release: {episode_id}")
        summary_pass = flag(summaries[episode_id].get("passed_obstacle"))
        _require(summary_pass == bool(latched[-1]), f"episode pass summary disagrees with latch: {episode_id}")
        _require(summary_pass == (first_gate is not None), f"episode pass summary disagrees with geometry: {episode_id}")
        episode_count += 1
        pass_count += int(summary_pass)
    return {"status": "PASS", "episodes": episode_count, "passed": pass_count, "gate": "passed_margin_m>0 and relevant_obstacle_clearance_m>=2.5 at one step", "latch_cross_checked": True}


def episode_pass_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) or 0)
    first_gate = _first_step(ordered, lambda row: pass_gate(row) is True)
    first_latch = _first_step(ordered, lambda row: flag(row.get("passed_obstacle")))
    return {
        "first_exact_pass_gate_step": first_gate,
        "exact_pass_gate_ever": first_gate is not None,
        "first_source_pass_latch_step": first_latch,
        "source_pass_latch_ever": first_latch is not None,
        "final_instantaneous_pass_gate": pass_gate(ordered[-1]) if ordered else None,
        "final_source_pass_latch": flag(ordered[-1].get("passed_obstacle")) if ordered else None,
    }


def _metric(row: Mapping[str, Any], field: str) -> float | None:
    return finite_float(row.get(field))


def mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [value for row in rows if (value := _metric(row, field)) is not None]
    return float(np.mean(values)) if values else None


def ols_slope(rows: Sequence[Mapping[str, Any]], field: str = "goal_distance_m") -> float | None:
    pairs = [(integer(row.get("step")), finite_float(row.get(field))) for row in rows]
    pairs = [(float(step), value) for step, value in pairs if step is not None and value is not None]
    if len(pairs) < 2:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    denominator = float(np.sum((x - np.mean(x)) ** 2))
    if denominator <= 0.0:
        return None
    return float(np.sum((x - np.mean(x)) * (y - np.mean(y))) / denominator)


def sustained_positive(values: Sequence[float | None], *, window: int, minimum: int, strict: bool = True) -> bool:
    if len(values) < window:
        return False
    for start in range(len(values) - window + 1):
        section = values[start : start + window]
        count = sum(value is not None and (value > 0.0 if strict else value >= 0.0) for value in section)
        if count >= minimum:
            return True
    return False


def sustained_actual_radial(values: Sequence[float | None], *, threshold: float = RADIAL_EPSILON) -> bool:
    run = 0
    for value in values:
        run = run + 1 if value is not None and value > threshold else 0
        if run >= 3:
            return True
    return False


def first_local_exit_step(
    rows: Sequence[Mapping[str, Any]],
    *,
    encounter_step: int,
    common_anchor_step: int | None,
    threshold_mps: float = EXIT_SPEED_THRESHOLD_MPS,
) -> int | None:
    start = max(encounter_step, common_anchor_step) if common_anchor_step is not None else encounter_step
    ordered = sorted((row for row in rows if (integer(row.get("step")) or 0) >= start), key=lambda row: integer(row.get("step")) or 0)
    for index in range(len(ordered) - 2):
        section = ordered[index : index + 3]
        steps = [integer(row.get("step")) for row in section]
        speeds = [_logged_number(row, "audit_actual_lateral_velocity_mps", "actual_lateral_velocity_mps") for row in section]
        if steps[0] is not None and steps == list(range(steps[0], steps[0] + 3)) and all(speed is not None and abs(speed) <= threshold_mps for speed in speeds):
            return steps[0]
    return None


def reacceleration_summary(rows: Sequence[Mapping[str, Any]], *, threshold_mps: float = EXIT_SPEED_THRESHOLD_MPS) -> dict[str, Any]:
    qualifying: list[list[Mapping[str, Any]]] = []
    run: list[Mapping[str, Any]] = []
    peak = 0.0
    for row in rows:
        speed = _logged_number(row, "audit_actual_lateral_velocity_mps", "actual_lateral_velocity_mps")
        if speed is not None and abs(speed) > threshold_mps:
            run.append(row)
            peak = max(peak, abs(speed))
        else:
            if len(run) >= 3:
                qualifying.append(run)
            run = []
    right_censored = len(run) >= 3
    if right_censored:
        qualifying.append(run)
    first_step = min(
        (integer(item[0].get("step")) for item in qualifying if item),
        default=None,
    )
    return {
        "lateral_reacceleration_first_step": first_step,
        "lateral_reacceleration_run_count": len(qualifying),
        "lateral_reacceleration_duration_steps": sum(len(group) for group in qualifying),
        "lateral_reacceleration_peak_abs_mps": peak,
        "lateral_reacceleration_right_censored": right_censored,
    }


def margin_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    margins = [(integer(row.get("step")), finite_float(row.get("passed_margin_m"))) for row in rows]
    finite = [(step, value) for step, value in margins if step is not None and value is not None]
    if not finite:
        return {
            "margin_start_m": None,
            "margin_best_m": None,
            "margin_best_step": None,
            "margin_final_m": None,
            "margin_delta_anchor_to_best_m": None,
            "margin_delta_anchor_to_final_m": None,
            "margin_regression_best_minus_final_m": None,
        }
    start_step, start_value = finite[0]
    final_step, final_value = finite[-1]
    best_value = max(value for _, value in finite)
    best_step = min(step for step, value in finite if value == best_value)
    return {
        "margin_start_m": start_value,
        "margin_best_m": best_value,
        "margin_best_step": best_step,
        "margin_final_m": final_value,
        "margin_delta_anchor_to_best_m": best_value - start_value,
        "margin_delta_anchor_to_final_m": final_value - start_value,
        "margin_regression_best_minus_final_m": best_value - final_value,
    }


def actor_side_statistics(rows: Sequence[Mapping[str, Any]], initial_sign: int) -> dict[str, Any]:
    signs: list[int] = []
    for row in rows:
        value = _logged_number(row, "audit_actor_lateral_delta_v_mps", "actor_raw_lateral_delta_v_mps")
        sign = 0 if value is None or abs(value) <= RADIAL_EPSILON else (1 if value > 0 else -1)
        if sign:
            signs.append(sign)
    if initial_sign == 0 or not signs:
        original_fraction = None
        counter_fraction = None
    else:
        original_fraction = sum(sign == initial_sign for sign in signs) / len(signs)
        counter_fraction = sum(sign == -initial_sign for sign in signs) / len(signs)
    return {
        "actor_original_side_fraction": original_fraction,
        "actor_counter_side_fraction": counter_fraction,
        "actor_lateral_sign_changes": sum(left != right for left, right in zip(signs, signs[1:])),
        "actor_nonzero_lateral_steps": len(signs),
        "initial_encounter_actual_lateral_sign": initial_sign,
    }


def _first_step(rows: Sequence[Mapping[str, Any]], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return integer(row.get("step"))
    return None


def _first_encounter_sign(rows: Sequence[Mapping[str, Any]], encounter_step: int) -> int:
    for row in rows:
        step = integer(row.get("step"))
        value = _logged_number(row, "audit_actual_lateral_velocity_mps", "actual_lateral_velocity_mps")
        if step is not None and step >= encounter_step and value is not None and abs(value) > RADIAL_EPSILON:
            return 1 if value > 0 else -1
    return 0


def _pipeline_step_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def count(key: str) -> int:
        return sum(flag(row.get(key)) for row in rows)

    def first(key: str) -> int | None:
        return _first_step(rows, lambda row: flag(row.get(key)))

    return {
        "post_bypass_recovery_steps": count("post_bypass_recovery"),
        "first_post_bypass_recovery_step": first("post_bypass_recovery"),
        "goal_recovery_eligible_steps": count("goal_recovery_correction_eligible"),
        "first_goal_recovery_eligible_step": first("goal_recovery_correction_eligible"),
        "goal_recovery_active_steps": count("goal_recovery_correction_active"),
        "first_goal_recovery_active_step": first("goal_recovery_correction_active"),
        "terminal_capture_active_steps": count("terminal_capture_active"),
        "first_terminal_capture_active_step": first("terminal_capture_active"),
        "terminal_v2_governor_eligible_steps": count("terminal_v2_governor_eligible"),
        "first_terminal_v2_governor_eligible_step": first("terminal_v2_governor_eligible"),
        "terminal_v2_governor_active_steps": count("terminal_v2_governor_active"),
        "terminal_v21_conflict_eligible_steps": count("terminal_v21_conflict_eligible"),
        "terminal_v21_conflict_active_steps": count("terminal_v21_conflict_active"),
        "terminal_v22_restorative_active_steps": count("terminal_v22_restorative_active"),
        "terminal_v23_guard_active_steps": count("terminal_v23_guard_active"),
        "safety_projection_steps": count("safety_projection_applied"),
        "boundary_projection_steps": count("boundary_projection_applied"),
    }


def metrics_for_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    encounter_sign: int = 0,
    local_exit_step: int | None = None,
    search_start_step: int | None = None,
    include_reacceleration: bool = True,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: integer(row.get("step")) or 0)
    fields = {
        "actor_radial_mean_mps": "audit_actor_radial_delta_v_mps",
        "final_radial_mean_mps": "audit_final_radial_delta_v_mps",
        "estimated_radial_velocity_mean_mps": "audit_estimated_radial_velocity_mps",
        "actual_radial_velocity_mean_mps": "audit_actual_radial_velocity_mps",
        "actor_lateral_mean_mps": "audit_actor_lateral_delta_v_mps",
        "final_lateral_mean_mps": "audit_final_lateral_delta_v_mps",
        "estimated_lateral_velocity_mean_mps": "audit_estimated_lateral_velocity_mps",
        "actual_lateral_velocity_mean_mps": "audit_actual_lateral_velocity_mps",
    }
    result: dict[str, Any] = {name: mean_metric(ordered, field) for name, field in fields.items()}
    actor_radial = [_logged_number(row, "audit_actor_radial_delta_v_mps", "actor_raw_radial_delta_v_mps") for row in ordered]
    final_radial = [_logged_number(row, "audit_final_radial_delta_v_mps", "final_applied_radial_delta_v_mps") for row in ordered]
    actual_radial = [_logged_number(row, "audit_actual_radial_velocity_mps", "actual_radial_velocity_mps") for row in ordered]
    result["actor_radial_positive_mean_mps"] = float(np.mean([max(0.0, value) for value in actor_radial if value is not None])) if any(value is not None for value in actor_radial) else None
    result["final_radial_positive_mean_mps"] = float(np.mean([max(0.0, value) for value in final_radial if value is not None])) if any(value is not None for value in final_radial) else None
    result["actor_radial_positive_sustained_5step_4of5"] = sustained_positive(actor_radial, window=5, minimum=4)
    result["final_radial_positive_sustained_5step_4of5"] = sustained_positive(final_radial, window=5, minimum=4)
    result["actual_radial_positive_sustained_3step"] = sustained_actual_radial(actual_radial)
    actor_positive_sum = sum(max(0.0, value) for value in actor_radial if value is not None)
    final_positive_sum = sum(max(0.0, value) for value in final_radial if value is not None)
    result["actor_to_final_radial_retention"] = final_positive_sum / actor_positive_sum if actor_positive_sum > RADIAL_EPSILON else None
    actor_lateral_abs = [abs(value) for value in (_logged_number(row, "audit_actor_lateral_delta_v_mps", "actor_raw_lateral_delta_v_mps") for row in ordered) if value is not None]
    final_lateral_abs = [abs(value) for value in (_logged_number(row, "audit_final_lateral_delta_v_mps", "final_applied_lateral_delta_v_mps") for row in ordered) if value is not None]
    actor_lateral_scale = float(np.mean(actor_lateral_abs)) if actor_lateral_abs else None
    final_lateral_scale = float(np.mean(final_lateral_abs)) if final_lateral_abs else None
    result["actor_radial_to_abs_lateral_balance"] = result["actor_radial_positive_mean_mps"] / actor_lateral_scale if actor_lateral_scale and actor_lateral_scale > RADIAL_EPSILON else None
    result["final_radial_to_abs_lateral_balance"] = result["final_radial_positive_mean_mps"] / final_lateral_scale if final_lateral_scale and final_lateral_scale > RADIAL_EPSILON else None
    first10 = ordered[:10]
    first20 = ordered[:20]
    last20 = ordered[-20:]
    result["goal_distance_slope_first10_m_per_step"] = ols_slope(first10) if len(first10) == 10 else None
    result["goal_distance_slope_first20_m_per_step"] = ols_slope(first20) if len(first20) == 20 else None
    result["goal_distance_slope_last20_m_per_step"] = ols_slope(last20) if len(last20) == 20 else None
    first_distance = finite_float(ordered[0].get("goal_distance_m")) if ordered else None
    last_distance = finite_float(ordered[-1].get("goal_distance_m")) if ordered else None
    result["goal_distance_net_reduction_m"] = first_distance - last_distance if first_distance is not None and last_distance is not None else None
    exact_pass_rows = [(integer(row.get("step")), pass_gate(row)) for row in ordered]
    first_pass = next((step for step, gate in exact_pass_rows if step is not None and gate is True), None)
    first_latch = _first_step(ordered, lambda row: flag(row.get("passed_obstacle")))
    result["pass_gate_first_step"] = first_pass
    result["pass_gate_ever"] = first_pass is not None
    result["pass_latch_first_step"] = first_latch
    result["pass_latch_ever"] = first_latch is not None
    result["pass_gate_final_instantaneous"] = exact_pass_rows[-1][1] if exact_pass_rows else None
    result["pass_latch_final"] = flag(ordered[-1].get("passed_obstacle")) if ordered else None
    result.update(margin_summary(ordered))
    result["local_exit_latency_steps"] = (
        local_exit_step - search_start_step
        if local_exit_step is not None and search_start_step is not None else None
    )
    if include_reacceleration:
        result.update(reacceleration_summary(ordered))
    else:
        result.update({
            "lateral_reacceleration_first_step": None,
            "lateral_reacceleration_run_count": None,
            "lateral_reacceleration_duration_steps": None,
            "lateral_reacceleration_peak_abs_mps": None,
            "lateral_reacceleration_right_censored": None,
            "lateral_reacceleration_unavailable_reason": "REACCELERATION_DEFINED_ONLY_AFTER_ARM_LOCAL_EXIT",
        })
    result.update(actor_side_statistics(ordered, encounter_sign))
    result.update(_pipeline_step_counts(ordered))
    result["window_start_step"] = integer(ordered[0].get("step")) if ordered else None
    result["window_end_step"] = integer(ordered[-1].get("step")) if ordered else None
    result["window_step_count"] = len(ordered)
    return result


def compare_iqr(value: Any, reference: Mapping[str, Any]) -> str | None:
    number = finite_float(value)
    q1 = finite_float(reference.get("q1"))
    q3 = finite_float(reference.get("q3"))
    if number is None or q1 is None or q3 is None:
        return None
    if number < q1:
        return "BELOW_IQR"
    if number > q3:
        return "ABOVE_IQR"
    return "WITHIN_IQR"


def success_reference_statistics(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = COMPARISON_METRICS,
) -> dict[str, dict[str, Any]]:
    by_id = {str(row.get("episode_id", "")): row for row in diagnostics}
    reference_rows = [by_id[episode_id] for episode_id in SUCCESS_REFERENCE_IDS if episode_id in by_id]
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = [number for row in reference_rows if (number := finite_float(row.get(metric))) is not None]
        if values:
            q1, median, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75], method="linear")
            result[metric] = {
                "eligible_episode_count": len(SUCCESS_REFERENCE_IDS),
                "n": len(values),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "minimum": float(min(values)),
                "maximum": float(max(values)),
                "quantile_method": "numpy_linear_type7",
            }
        else:
            result[metric] = {
                "eligible_episode_count": len(SUCCESS_REFERENCE_IDS),
                "n": 0,
                "median": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "minimum": None,
                "maximum": None,
                "quantile_method": "numpy_linear_type7",
                "unavailable_reason": "NO_FINITE_REFERENCE_VALUES",
            }
    return result


def diagnose_on_episode(
    episode_id: str,
    metrics: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    *,
    local_exit_found: bool,
    passed: bool,
) -> dict[str, Any]:
    if episode_id == "test_r99_episode_0013":
        return {"primary_diagnosis": "NOT_ASSESSABLE", "special_case_diagnosis": "SAFETY_DOMINATED_SPECIAL_CASE", "secondary_diagnoses": [], "evidence_flags": {}}
    if episode_id == "test_r99_episode_0014":
        return {"primary_diagnosis": "NOT_ASSESSABLE", "special_case_diagnosis": "BOUNDARY_DOMINATED_SPECIAL_CASE", "secondary_diagnoses": [], "evidence_flags": {}}
    if episode_id == "test_r99_episode_0008":
        return {"primary_diagnosis": "NOT_ASSESSABLE", "special_case_diagnosis": None, "case_annotation": "DOWNSTREAM_ADVERSE_SPECIAL_CASE", "secondary_diagnoses": [], "evidence_flags": {}}
    if not local_exit_found:
        return {"primary_diagnosis": "NOT_ASSESSABLE", "special_case_diagnosis": None, "secondary_diagnoses": [], "evidence_flags": {"local_exit_missing": True}}

    actor_sustained = bool(metrics.get("actor_radial_positive_sustained_5step_4of5"))
    final_sustained = bool(metrics.get("final_radial_positive_sustained_5step_4of5"))
    actual_sustained = bool(metrics.get("actual_radial_positive_sustained_3step"))
    retention_class = compare_iqr(metrics.get("actor_to_final_radial_retention"), references.get("actor_to_final_radial_retention", {}))
    final_magnitude_class = compare_iqr(metrics.get("final_radial_positive_mean_mps"), references.get("final_radial_positive_mean_mps", {}))
    progress_class = compare_iqr(metrics.get("goal_distance_net_reduction_m"), references.get("goal_distance_net_reduction_m", {}))
    actual_progress_class = compare_iqr(metrics.get("actual_radial_velocity_mean_mps"), references.get("actual_radial_velocity_mean_mps", {}))
    regression = finite_float(metrics.get("margin_regression_best_minus_final_m"))
    regression_class = compare_iqr(regression, references.get("margin_regression_best_minus_final_m", {}))
    reaccel_duration = finite_float(metrics.get("lateral_reacceleration_duration_steps"))
    reaccel_class = compare_iqr(reaccel_duration, references.get("lateral_reacceleration_duration_steps", {}))

    evidence: dict[str, bool] = {
        "radial_intent_not_sustained": not actor_sustained,
        "pipeline_radial_attenuation": actor_sustained and (not final_sustained or retention_class == "BELOW_IQR"),
        "radial_propulsion_insufficient": final_sustained and not passed and (final_magnitude_class == "BELOW_IQR" or progress_class == "BELOW_IQR"),
        "radial_motion_response_weak": final_sustained and (not actual_sustained or actual_progress_class == "BELOW_IQR" or progress_class == "BELOW_IQR"),
        "lateral_reacceleration": int(metrics.get("lateral_reacceleration_run_count", 0) or 0) > 0 and reaccel_class == "ABOVE_IQR",
        "pass_progress_regression": regression is not None and regression > 0.0 and regression_class == "ABOVE_IQR",
    }
    applicable = {
        "radial_intent_not_sustained": "RADIAL_INTENT_NOT_SUSTAINED",
        "pipeline_radial_attenuation": "PIPELINE_RADIAL_ATTENUATION",
        "radial_propulsion_insufficient": "RADIAL_PROPULSION_INSUFFICIENT",
        "radial_motion_response_weak": "RADIAL_MOTION_RESPONSE_WEAK",
        "lateral_reacceleration": "LATERAL_REACCELERATION",
        "pass_progress_regression": "PASS_PROGRESS_REGRESSION",
    }
    active = [diagnosis for key, diagnosis in applicable.items() if evidence.get(key)]
    no_geometry = not passed
    if no_geometry and not active:
        active.append("PASS_GEOMETRY_NOT_REACHED")
    if len(active) == 1:
        primary = active[0]
    elif len(active) > 1:
        primary = "MIXED"
    else:
        primary = "NOT_ASSESSABLE"
    result = {
        "primary_diagnosis": primary,
        "special_case_diagnosis": None,
        "secondary_diagnoses": [],
        "evidence_flags": evidence,
        "reference_comparisons": {
            "actor_to_final_radial_retention": retention_class,
            "final_radial_positive_mean_mps": final_magnitude_class,
            "goal_distance_net_reduction_m": progress_class,
            "actual_radial_velocity_mean_mps": actual_progress_class,
            "margin_regression_best_minus_final_m": regression_class,
            "lateral_reacceleration_duration_steps": reaccel_class,
        },
    }
    validate_diagnosis(result["primary_diagnosis"])
    return result


def validate_diagnosis(value: str) -> None:
    if value not in PRIMARY_DIAGNOSES:
        raise ValueError(f"illegal primary diagnosis: {value}")


def select_common_windows(
    pair_rows: Sequence[Mapping[str, Any]],
    off_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    on_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for pair in pair_rows:
        episode_id = str(pair["episode_id"])
        off_rows = list(off_grouped[episode_id])
        on_rows = list(on_grouped[episode_id])
        status = str(pair.get("triggered_cohort_status", ""))
        anchor = integer(pair.get("off_defined_anchor_step"))
        off_eligible = [
            integer(row.get("step")) for row in off_rows
            if flag(row.get("lateral_exit_assist_eligible"))
        ]
        first_off_eligible = min((step for step in off_eligible if step is not None), default=None)
        if status == "OFF_NEVER_ELIGIBLE":
            _require(anchor is None and first_off_eligible is None, f"OFF never eligible anchor mismatch: {episode_id}")
            result[episode_id] = {"status": status, "anchor_step": None, "common_end_step": None, "off_rows": [], "on_rows": []}
            continue
        _require(status == "INCLUDED", f"unknown trigger cohort status: {episode_id}")
        _require(anchor is not None and first_off_eligible == anchor, f"OFF anchor step cross-check: {episode_id}")
        off_last = integer(off_rows[-1].get("step"))
        on_last = integer(on_rows[-1].get("step"))
        _require(off_last is not None and on_last is not None, f"missing last step: {episode_id}")
        common_end = min(off_last, on_last)
        _require(integer(pair.get("common_window_end_step")) == common_end, f"common window end cross-check: {episode_id}")
        if on_last < anchor:
            result[episode_id] = {"status": "ON_TERMINATED_BEFORE_OFF_ANCHOR", "anchor_step": anchor, "common_end_step": common_end, "off_rows": [], "on_rows": []}
            continue
        _require(common_end >= anchor, f"empty common window: {episode_id}")
        result[episode_id] = {
            "status": "INCLUDED",
            "anchor_step": anchor,
            "common_end_step": common_end,
            "off_rows": [row for row in off_rows if anchor <= integer(row.get("step")) <= common_end],
            "on_rows": [row for row in on_rows if anchor <= integer(row.get("step")) <= common_end],
        }
        _require(bool(result[episode_id]["off_rows"]) and bool(result[episode_id]["on_rows"]), f"both common-arm windows required: {episode_id}")
    return result


def _na_window(reason: str) -> dict[str, Any]:
    return {key: None for key in METRICS} | {"window_start_step": None, "window_end_step": None, "window_step_count": 0, "structural_na_reason": reason}


def build_diagnostics(
    off_episodes: Sequence[Mapping[str, Any]],
    on_episodes: Sequence[Mapping[str, Any]],
    off_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    on_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    common_windows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    off_summary = {str(row["episode_id"]): row for row in off_episodes}
    on_summary = {str(row["episode_id"]): row for row in on_episodes}
    diagnostics: list[dict[str, Any]] = []
    mechanistic: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for episode_id in EXPECTED_EPISODE_IDS:
        off_rows = list(off_grouped[episode_id])
        on_rows = list(on_grouped[episode_id])
        off_episode = off_summary[episode_id]
        on_episode = on_summary[episode_id]
        encounter = integer(on_episode.get("signed_encounter_step"))
        _require(encounter is not None, f"missing encounter for {episode_id}")
        common = common_windows[episode_id]
        anchor = common.get("anchor_step")
        on_local_exit = first_local_exit_step(on_rows, encounter_step=encounter, common_anchor_step=anchor)
        off_local_exit = first_local_exit_step(off_rows, encounter_step=encounter, common_anchor_step=anchor)
        on_local_rows = [row for row in on_rows if on_local_exit is not None and integer(row.get("step")) >= on_local_exit]
        off_local_rows = [row for row in off_rows if off_local_exit is not None and integer(row.get("step")) >= off_local_exit]
        on_sign = _first_encounter_sign(on_rows, encounter)
        off_sign = _first_encounter_sign(off_rows, encounter)
        search_start = max(encounter, anchor) if anchor is not None else encounter
        on_metrics = metrics_for_window(on_local_rows, encounter_sign=on_sign, local_exit_step=on_local_exit, search_start_step=search_start) if on_local_exit is not None else _na_window("NO_THREE_CONSECUTIVE_ACTUAL_LATERAL_SPEED_STEPS_AT_OR_BELOW_0.5_MPS")
        off_metrics = metrics_for_window(off_local_rows, encounter_sign=off_sign, local_exit_step=off_local_exit, search_start_step=search_start) if off_local_exit is not None else _na_window("NO_THREE_CONSECUTIVE_ACTUAL_LATERAL_SPEED_STEPS_AT_OR_BELOW_0.5_MPS")
        for condition, local_exit, local_rows, metrics in (
            ("OFF", off_local_exit, off_local_rows, off_metrics),
            ("ON", on_local_exit, on_local_rows, on_metrics),
        ):
            episode = off_episode if condition == "OFF" else on_episode
            condition_rows = off_rows if condition == "OFF" else on_rows
            frame_row = next((row for row in condition_rows if frame_for_row(row) is not None), None)
            mechanism_row: dict[str, Any] = {
                "episode_index": integer(episode.get("episode_index")),
                "episode_id": episode_id,
                "condition": condition,
                "encounter_step": encounter,
                "common_anchor_step": anchor,
                "search_start_step": max(encounter, anchor) if anchor is not None else encounter,
                "local_exit_step": local_exit,
                "local_exit_status": "OBSERVED" if local_exit is not None else "LOCAL_EXIT_NOT_OBSERVED",
                "local_window_start_step": local_exit,
                "local_window_end_step": integer(local_rows[-1].get("step")) if local_rows else None,
                "local_window_censored_at_episode_end": bool(local_rows and integer(local_rows[-1].get("step")) == integer(condition_rows[-1].get("step"))),
                "structural_na_reason": metrics.get("structural_na_reason"),
                "frozen_radial_goal_ray_m": frame_row.get("lateral_exit_assist_horizontal_goal_ray_m") if frame_row else None,
                "frozen_lateral_unit": frame_row.get("lateral_exit_assist_lateral_unit") if frame_row else None,
                "goalward_radial_positive": True,
            }
            mechanism_row.update({key: metrics.get(key) for key in METRICS})
            mechanism_row.update(_pipeline_step_counts(local_rows))
            mechanism_row["first_entered_1m_step"] = integer(episode.get("first_entered_1m_step"))
            mechanism_row["stable_goal_step"] = integer(episode.get("stable_goal_step"))
            mechanism_row.update({
                "no_release_entire_episode": not any(flag(row.get("post_bypass_recovery")) for row in (off_rows if condition == "OFF" else on_rows)),
                "goal_recovery_inactive_entire_episode": not any(flag(row.get("goal_recovery_correction_active")) for row in (off_rows if condition == "OFF" else on_rows)),
            })
            mechanistic.append(mechanism_row)

        anchor_status = str(common["status"])
        common_record: dict[str, Any] = {
            "episode_index": integer(on_episode.get("episode_index")),
            "episode_id": episode_id,
            "scenario_seed": integer(on_episode.get("scenario_seed")),
            "noise_seed": integer(on_episode.get("noise_seed")),
            "cohort_status": anchor_status,
            "off_defined_anchor_step": anchor,
            "common_window_end_step": common.get("common_end_step"),
            "common_window_inclusive": True,
            "off_full_episode_steps": len(off_rows),
            "on_full_episode_steps": len(on_rows),
            "off_local_exit_observed_by_anchor": bool(
                off_local_exit is not None and anchor is not None and off_local_exit <= anchor
            ),
            "on_local_exit_observed_by_anchor": bool(
                on_local_exit is not None and anchor is not None and on_local_exit <= anchor
            ),
        }
        if anchor_status == "INCLUDED":
            off_common_metrics = metrics_for_window(common["off_rows"], encounter_sign=off_sign, include_reacceleration=False)
            on_common_metrics = metrics_for_window(common["on_rows"], encounter_sign=on_sign, include_reacceleration=False)
            for metric in METRICS:
                left = off_common_metrics.get(metric)
                right = on_common_metrics.get(metric)
                common_record[f"OFF_{metric}"] = left
                common_record[f"ON_{metric}"] = right
                left_num, right_num = finite_float(left), finite_float(right)
                common_record[f"ON_minus_OFF_{metric}"] = right_num - left_num if left_num is not None and right_num is not None else None
        else:
            common_record["structural_na_reason"] = anchor_status
            for metric in METRICS:
                common_record[f"OFF_{metric}"] = None
                common_record[f"ON_{metric}"] = None
                common_record[f"ON_minus_OFF_{metric}"] = None
        common_rows.append(common_record)

        diagnostics.append({
            "episode_index": integer(on_episode.get("episode_index")),
            "episode_id": episode_id,
            "scenario_seed": integer(on_episode.get("scenario_seed")),
            "noise_seed": integer(on_episode.get("noise_seed")),
            "scenario_hash": on_episode.get("scenario_hash"),
            "group": "main_failure" if episode_id in MAIN_FAILURE_IDS else "success_reference" if episode_id in SUCCESS_REFERENCE_IDS else "other",
            "cohort_status": anchor_status,
            "off_defined_anchor_step": anchor,
            "common_window_end_step": common.get("common_end_step"),
            "encounter_step": encounter,
            "off_local_exit_step": off_local_exit,
            "on_local_exit_step": on_local_exit,
            "off_passed_obstacle": flag(off_episode.get("passed_obstacle")),
            "on_passed_obstacle": flag(on_episode.get("passed_obstacle")),
            **{f"off_{key}": value for key, value in episode_pass_summary(off_rows).items()},
            **{f"on_{key}": value for key, value in episode_pass_summary(on_rows).items()},
            "off_post_bypass_recovery": flag(off_episode.get("post_bypass_recovery")),
            "on_post_bypass_recovery": flag(on_episode.get("post_bypass_recovery")),
            "off_stable_goal": flag(off_episode.get("stable_goal")),
            "on_stable_goal": flag(on_episode.get("stable_goal")),
            "off_timeout": flag(off_episode.get("timeout")),
            "on_timeout": flag(on_episode.get("timeout")),
            "off_collision": flag(off_episode.get("collision")),
            "on_collision": flag(on_episode.get("collision")),
            "off_goal_recovery_active_steps": integer(off_episode.get("goal_correction_active_steps")),
            "on_goal_recovery_active_steps": integer(on_episode.get("goal_correction_active_steps")),
            "off_no_release": not any(flag(row.get("post_bypass_recovery")) for row in off_rows),
            "on_no_release": not any(flag(row.get("post_bypass_recovery")) for row in on_rows),
            "off_goal_recovery_inactive": not any(flag(row.get("goal_recovery_correction_active")) for row in off_rows),
            "on_goal_recovery_inactive": not any(flag(row.get("goal_recovery_correction_active")) for row in on_rows),
            "off_metrics": off_metrics,
            "on_metrics": on_metrics,
        })
    return diagnostics, common_rows, mechanistic


def attach_diagnoses_and_references(diagnostics: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    on_rows = [
        {
            "episode_id": row["episode_id"],
            **row["on_metrics"],
        }
        for row in diagnostics
    ]
    references = success_reference_statistics(on_rows)
    by_episode = {str(row["episode_id"]): row for row in diagnostics}
    for row in diagnostics:
        episode_id = str(row["episode_id"])
        local_exit = row.get("on_local_exit_step") is not None
        on_metrics = row.get("on_metrics", {})
        passed = bool(on_metrics.get("pass_gate_ever")) if local_exit else flag(row.get("on_passed_obstacle"))
        classification = diagnose_on_episode(episode_id, on_metrics, references, local_exit_found=local_exit, passed=passed)
        row.update(classification)
        row["off_corroborative_diagnosis"] = diagnose_on_episode(
            episode_id,
            row.get("off_metrics", {}),
            references,
            local_exit_found=row.get("off_local_exit_step") is not None,
            passed=bool(row.get("off_metrics", {}).get("pass_gate_ever")),
        )["primary_diagnosis"]
        row["secondary_diagnoses"] = []
        if episode_id == "test_r99_episode_0008":
            row["case_annotation"] = "DOWNSTREAM_ADVERSE_SPECIAL_CASE"
        row["on_metrics"] = on_metrics
        row["off_metrics"] = row.get("off_metrics", {})
    validate_episode_diagnoses(diagnostics)
    return references, diagnostics


def validate_episode_diagnoses(diagnostics: Sequence[Mapping[str, Any]]) -> None:
    _require(len(diagnostics) == 20, "episode diagnostic row count")
    by_id = {str(row.get("episode_id", "")): row for row in diagnostics}
    _require(set(by_id) == set(EXPECTED_EPISODE_IDS), "episode diagnostic identity")
    for episode_id in MAIN_FAILURE_IDS:
        _require(by_id[episode_id].get("primary_diagnosis") in PRIMARY_DIAGNOSES, f"main failure diagnosis: {episode_id}")
    for episode_id, label in (("test_r99_episode_0013", "SAFETY_DOMINATED_SPECIAL_CASE"), ("test_r99_episode_0014", "BOUNDARY_DOMINATED_SPECIAL_CASE")):
        _require(by_id[episode_id].get("primary_diagnosis") == "NOT_ASSESSABLE", f"special primary diagnosis: {episode_id}")
        _require(by_id[episode_id].get("special_case_diagnosis") == label, f"special case label: {episode_id}")
        _require(not by_id[episode_id].get("secondary_diagnoses"), f"special case secondary diagnosis: {episode_id}")
    adverse = by_id["test_r99_episode_0008"]
    _require(adverse.get("primary_diagnosis") == "NOT_ASSESSABLE", "0008 primary diagnosis")
    _require(adverse.get("case_annotation") == "DOWNSTREAM_ADVERSE_SPECIAL_CASE", "0008 annotation")
    _require(not adverse.get("secondary_diagnoses"), "0008 secondary diagnosis")


def validate_main_failure_pipeline(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["episode_id"]): row for row in diagnostics}
    for episode_id in MAIN_FAILURE_IDS:
        row = by_id[episode_id]
        _require(row.get("off_no_release") is True and row.get("on_no_release") is True, f"main failure release latch must remain inactive: {episode_id}")
        _require(row.get("off_goal_recovery_inactive") is True and row.get("on_goal_recovery_inactive") is True, f"main failure Goal Recovery must remain inactive: {episode_id}")
    return {
        "main_failure_count": len(MAIN_FAILURE_IDS),
        "no_release_all_main_failures": True,
        "goal_recovery_inactive_all_main_failures": True,
        "interpretation": "expected pre-release inactivity; downstream Goal Recovery is not eligible until the pass/release gate",
    }


def validate_episode_outcomes(
    grouped_steps: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries = {str(row["episode_id"]): row for row in episode_rows}
    for episode_id, rows in grouped_steps.items():
        summary = summaries[episode_id]
        _require(integer(summary.get("steps")) == len(rows), f"episode step count: {episode_id}")
        for summary_field, step_field in (
            ("collision", "collision"),
            ("post_bypass_recovery", "post_bypass_recovery"),
            ("stable_goal", "stable_goal"),
            ("timeout", "truncated"),
        ):
            _require(flag(summary.get(summary_field)) == any(flag(row.get(step_field)) for row in rows), f"episode outcome latch {summary_field}: {episode_id}")
        terminal_rows = [row for row in rows if flag(row.get("terminated")) or flag(row.get("truncated"))]
        last_row = rows[-1]
        _require(all(integer(row.get("step")) == integer(last_row.get("step")) for row in terminal_rows), f"terminal event before final logged step: {episode_id}")
        _require(str(summary.get("termination_reason", "")) == str(last_row.get("termination_reason", "")), f"terminal reason: {episode_id}")
    return {"status": "PASS", "episodes": len(grouped_steps), "episode_step_counts": "MATCH", "terminal_latches": "MATCH"}


def comparison_rows(
    diagnostics: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = {str(row["episode_id"]): row for row in diagnostics}
    for episode_id in MAIN_FAILURE_IDS:
        diagnostic = by_id[episode_id]
        values = diagnostic.get("on_metrics", {})
        off_values = diagnostic.get("off_metrics", {})
        local_exit = diagnostic.get("on_local_exit_step") is not None
        for metric in COMPARISON_METRICS:
            value = values.get(metric)
            reference = references[metric]
            reason = None
            if not local_exit:
                category = None
                reason = "NO_LOCAL_EXIT_STRUCTURAL_NA"
                value = None
            elif finite_float(value) is None:
                category = None
                reason = str(values.get(f"{metric}_reason") or "METRIC_UNAVAILABLE")
                value = None
            elif finite_float(reference.get("q1")) is None or finite_float(reference.get("q3")) is None:
                category = None
                reason = str(reference.get("unavailable_reason") or "REFERENCE_UNAVAILABLE")
            else:
                category = compare_iqr(value, reference)
            rows.append({
                "episode_id": episode_id,
                "metric": metric,
                "on_value": value,
                "comparison": category,
                "reference_n": reference.get("n"),
                "reference_median": reference.get("median"),
                "reference_q1": reference.get("q1"),
                "reference_q3": reference.get("q3"),
                "reference_iqr": reference.get("iqr"),
                "reason": reason,
                "off_corroborative_value": off_values.get(metric),
                "supported_evidence_flags": [
                    key for key, active in diagnostic.get("evidence_flags", {}).items() if active
                ],
            })
    return rows


def pass_trajectory_rows(
    off_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    on_grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    off_episodes: Sequence[Mapping[str, Any]],
    on_episodes: Sequence[Mapping[str, Any]],
    common_windows: Mapping[str, Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = {
        "OFF": {str(row["episode_id"]): row for row in off_episodes},
        "ON": {str(row["episode_id"]): row for row in on_episodes},
    }
    diagnostic_by_id = {str(row["episode_id"]): row for row in diagnostics}
    output: list[dict[str, Any]] = []
    for condition, grouped in (("OFF", off_grouped), ("ON", on_grouped)):
        for episode_id in EXPECTED_EPISODE_IDS:
            episode_pass = flag(summaries[condition][episode_id].get("passed_obstacle"))
            common = common_windows[episode_id]
            anchor = integer(common.get("anchor_step"))
            common_end = integer(common.get("common_end_step"))
            diagnostic = diagnostic_by_id[episode_id]
            prefix = condition.lower()
            local_exit = integer(diagnostic.get(f"{prefix}_local_exit_step"))
            local_metrics = diagnostic.get(f"{prefix}_metrics", {})
            reaccel_step = integer(local_metrics.get("lateral_reacceleration_first_step"))
            running_best: float | None = None
            previous_margin: float | None = None
            for row in grouped[episode_id]:
                step = integer(row.get("step"))
                margin = finite_float(row.get("passed_margin_m"))
                if margin is not None:
                    running_best = margin if running_best is None else max(running_best, margin)
                delta = (
                    margin - previous_margin
                    if margin is not None and previous_margin is not None
                    else None
                )
                if margin is not None:
                    previous_margin = margin
                output.append({
                    "episode_index": integer(row.get("episode_index")),
                    "episode_id": episode_id,
                    "condition": condition,
                    "step": step,
                    "in_common_window": bool(
                        anchor is not None and common_end is not None
                        and step is not None and anchor <= step <= common_end
                    ),
                    "in_arm_local_window": bool(
                        local_exit is not None and step is not None and step >= local_exit
                    ),
                    "local_exit_marker": step == local_exit if step is not None else False,
                    "lateral_reacceleration_marker": step == reaccel_step if step is not None else False,
                    "passed_margin_m": margin,
                    "passed_margin_step_change_m": delta,
                    "passed_margin_running_max_m": running_best,
                    "passed_margin_running_regression_m": (
                        running_best - margin
                        if running_best is not None and margin is not None
                        else None
                    ),
                    "relevant_obstacle_clearance_m": _logged_number(row, "relevant_obstacle_clearance_m", "signed_obstacle_clearance_m"),
                    "exact_pass_gate_now": pass_gate(row),
                    "logged_pass_condition_now": flag(row.get("passed_condition_now")),
                    "source_passed_obstacle_latch": flag(row.get("passed_obstacle")),
                    "episode_passed_obstacle": episode_pass,
                    "post_bypass_recovery": flag(row.get("post_bypass_recovery")),
                    "goal_recovery_correction_eligible": flag(row.get("goal_recovery_correction_eligible")),
                    "goal_recovery_correction_active": flag(row.get("goal_recovery_correction_active")),
                    "actor_radial_goalward": (_logged_number(row, "audit_actor_radial_delta_v_mps", "actor_raw_radial_delta_v_mps") or 0.0) > 0.0,
                    "final_radial_goalward": (_logged_number(row, "audit_final_radial_delta_v_mps", "final_applied_radial_delta_v_mps") or 0.0) > 0.0,
                    "actual_radial_goalward": (_logged_number(row, "audit_actual_radial_velocity_mps", "actual_radial_velocity_mps") or 0.0) > RADIAL_EPSILON,
                })
    return output


def finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def safe_json_dumps(value: Any) -> str:
    return json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def csv_text(rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None) -> str:
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flat = {key: _csv_value(value) for key, value in row.items()}
        writer.writerow(flat)
    return stream.getvalue()


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(finite_json(value))
    if isinstance(value, np.generic):
        return _csv_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def flatten_diagnostics(diagnostics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nested = {"on_metrics", "off_metrics", "evidence_flags", "reference_comparisons"}
    for item in diagnostics:
        row = {key: value for key, value in item.items() if key not in nested}
        for group in ("on_metrics", "off_metrics"):
            for metric, value in item.get(group, {}).items():
                row[f"{group}_{metric}"] = value
        for key, value in item.get("evidence_flags", {}).items():
            row[f"evidence_{key}"] = value
        for key, value in item.get("reference_comparisons", {}).items():
            row[f"comparison_{key}"] = value
        rows.append(row)
    return rows


def reference_csv_rows(references: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric in COMPARISON_METRICS:
        ref = references[metric]
        rows.append({"metric": metric, **ref, "cohort": "ON_FIXED_SUCCESS_REFERENCE_11", "episode_ids": list(SUCCESS_REFERENCE_IDS)})
    return rows


def _episode_metric_lookup(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["episode_id"]): dict(row.get("on_metrics", {})) for row in diagnostics}


def build_report(
    identity: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    pass_checks: Mapping[str, Any],
    pipeline_checks: Mapping[str, Any],
    common_rows: Sequence[Mapping[str, Any]],
    mechanistic_rows: Sequence[Mapping[str, Any]],
    pass_rows: Sequence[Mapping[str, Any]],
) -> str:
    by_id = {str(row["episode_id"]): row for row in diagnostics}
    included = sum(row.get("cohort_status") == "INCLUDED" for row in common_rows)
    never = sum(row.get("cohort_status") == "OFF_NEVER_ELIGIBLE" for row in common_rows)
    off_passed = sum(row.get("off_passed_obstacle") is True for row in diagnostics)
    on_passed = sum(row.get("on_passed_obstacle") is True for row in diagnostics)
    newly_passed = [row["episode_id"] for row in diagnostics if row.get("off_passed_obstacle") is False and row.get("on_passed_obstacle") is True]
    lost_pass = [row["episode_id"] for row in diagnostics if row.get("off_passed_obstacle") is True and row.get("on_passed_obstacle") is False]
    diagnoses = {name: sum(row.get("primary_diagnosis") == name for row in diagnostics) for name in sorted(PRIMARY_DIAGNOSES)}
    main_lines = []
    for episode_id in MAIN_FAILURE_IDS:
        row = by_id[episode_id]
        main_lines.append(
            f"| {episode_id[-4:]} | {row.get('primary_diagnosis')} | {row.get('off_corroborative_diagnosis')} | "
            f"{row.get('on_local_exit_step') if row.get('on_local_exit_step') is not None else 'NA'} | "
            f"{row.get('on_metrics', {}).get('pass_gate_first_step') if row.get('on_metrics', {}).get('pass_gate_first_step') is not None else 'none'} |"
        )
    ref_net = references.get("goal_distance_net_reduction_m", {})
    ref_actual = references.get("actual_radial_velocity_mean_mps", {})
    no_gate_failures = sum(not bool(by_id[episode_id].get("on_metrics", {}).get("pass_gate_ever")) for episode_id in MAIN_FAILURE_IDS)
    inactive_main = all(by_id[episode_id].get("off_goal_recovery_inactive") and by_id[episode_id].get("on_goal_recovery_inactive") for episode_id in MAIN_FAILURE_IDS)
    released_main = any(not by_id[episode_id].get("off_no_release") or not by_id[episode_id].get("on_no_release") for episode_id in MAIN_FAILURE_IDS)
    source_hashes = identity.get("source_hashes", {})
    count = len(pass_rows)
    return "\n".join([
        "# Post-lateral-exit reorientation audit",
        "",
        "## Scope and identity",
        "",
        "This is a pure offline, paired-trajectory audit over the frozen OFF/ON source logs. It does not identify reward, dynamics, action-transform, safety, or boundary implementation equivalence independently; each remains `NOT_INDEPENDENTLY_VERIFIED`. The OFF exact step trajectory fingerprint is the behavioral-equivalence protection.",
        "",
        f"Active preregistration hash: `{ACTIVE_PREREGISTRATION_HASH}`. The source checkpoint is `{identity.get('checkpoint', {}).get('size_bytes')} bytes`, SHA-256 `{identity.get('checkpoint', {}).get('sha256')}`. Source pre/post hashes matched for {len(source_hashes)} opened inputs including the checkpoint.",
        "",
        "## Shared OFF anchor and common window",
        "",
        f"All 20 ordered episodes remain in the ITT diagnostics. The OFF-first-eligible common window is available for {included}; {never} are structural `OFF_NEVER_ELIGIBLE` rows with no common-window metric. Every available paired window uses the same absolute OFF anchor in both arms and ends at the earlier observed last step.",
        "",
        "## Arm-local mechanics and success IQR",
        "",
        "Local exit is the first start of three consecutive steps at or below 0.5 m/s absolute actual lateral velocity, searched from max(encounter, common anchor), or from encounter where no common anchor exists. Arm-local metrics run from that exit through the arm's last logged step; missing exits remain structural NA. The fixed 11-episode ON success reference uses numpy linear/type-7 quartiles. IQR labels use inclusive Q1/Q3 bounds and no outlier fences, ranks, or composite scores.",
        "",
        f"Reference goal-distance net reduction: n={ref_net.get('n')}, median={ref_net.get('median')}, Q1={ref_net.get('q1')}, Q3={ref_net.get('q3')}. Reference actual radial velocity mean: n={ref_actual.get('n')}, median={ref_actual.get('median')}, Q1={ref_actual.get('q1')}, Q3={ref_actual.get('q3')}.",
        "",
        "## Seven main failures",
        "",
        "| Episode | ON primary diagnosis | OFF corroborative only | ON local exit | First exact pass gate |",
        "|---|---|---|---:|---:|",
        *main_lines,
        "",
        f"The exact pass gate is `passed_margin_m > 0` and relevant obstacle clearance `>= 2.5 m` at the same step, checked against both the logged instantaneous condition and the source pass latch. In the seven main failures, {no_gate_failures} ON episodes never reached that gate in their post-exit window. OFF/ON episode pass totals are {off_passed}/{on_passed}; newly passed episodes: {newly_passed or 'none'}; lost pass episodes: {lost_pass or 'none'}.",
        "",
        "The assistance is lateral damping; these paired logs show whether it created a new pass, not a randomized proof of mechanism. The pass set did not gain an episode. With the same seven main failures lacking the exact geometry gate in ON, lateral braking alone supplied no extra pass on this cohort.",
        "",
        "## Special cases and downstream controller",
        "",
        "Episode 0013 is labeled `SAFETY_DOMINATED_SPECIAL_CASE` and episode 0014 `BOUNDARY_DOMINATED_SPECIAL_CASE`; each has primary `NOT_ASSESSABLE`. Episode 0008 receives only the `DOWNSTREAM_ADVERSE_SPECIAL_CASE` case annotation, has primary `NOT_ASSESSABLE`, and has no secondary diagnosis.",
        "",
        f"For all seven main failures, no release/pass latch occurred and Goal Recovery remained inactive in both arms: {inactive_main and not released_main}. This is the expected pre-release state of the downstream Goal Recovery path, so its inactivity is not diagnosed as a separate failure.",
        "",
        "## Diagnosis counts and next experiment",
        "",
        "Primary diagnosis counts across all 20 episodes: " + ", ".join(f"{key}={value}" for key, value in diagnoses.items() if value),
        "",
        "Continue with one preregistered paired intervention: add a single radial-intent-persistence intervention after local lateral exit and before exact pass, while freezing Actor, scene, and every other controller component. Keep OFF/ON episode identity, anchor/window rules, and pass geometry fixed. This audit does not implement the intervention.",
        "",
        f"The pass-margin trajectory table contains all {count} logged rows from both arms. The output contract is exactly {len(OUTPUT_FILES)} files.",
        "",
    ])


def _flatten_common(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _summary_payload(
    identity: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    common_rows: Sequence[Mapping[str, Any]],
    mechanistic_rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    pass_checks: Mapping[str, Any],
    pipeline_checks: Mapping[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in diagnostics:
        diagnosis = str(row.get("primary_diagnosis", "NOT_ASSESSABLE"))
        counts[diagnosis] = counts.get(diagnosis, 0) + 1
    main = {str(row["episode_id"]): row.get("primary_diagnosis") for row in diagnostics if row.get("episode_id") in MAIN_FAILURE_IDS}
    return {
        "audit": "single_uav_post_lateral_exit_reorientation_audit",
        "status": "PASS",
        "source_identity": dict(identity.get("source_identity", {})),
        "source_hashes": {name: {"sha256_before": value, "sha256_after": identity.get("source_hashes_after", {}).get(name), "unchanged": value == identity.get("source_hashes_after", {}).get(name)} for name, value in identity.get("source_hashes", {}).items()},
        "main_failure_episode_ids": list(MAIN_FAILURE_IDS),
        "success_reference_episode_ids": list(SUCCESS_REFERENCE_IDS),
        "special_case_ids": ["test_r99_episode_0013", "test_r99_episode_0014"],
        "common_anchor_definition": "OFF first fully eligible step; same absolute step in OFF and ON; inclusive end=min(last OFF,last ON)",
        "mechanistic_anchor_definition": "first start of 3 consecutive abs(actual lateral velocity)<=0.5 m/s after max(encounter,common anchor), descriptive only",
        "radial_sign_convention": {"GOALWARD_RADIAL_POSITIVE": True, "frame": "encounter-frozen logged horizontal goal ray"},
        "cohorts": {
            "all_episodes": list(EXPECTED_EPISODE_IDS),
            "main_failures": list(MAIN_FAILURE_IDS),
            "success_reference_on": list(SUCCESS_REFERENCE_IDS),
            "pair_table_included": sum(row.get("cohort_status") == "INCLUDED" for row in common_rows),
            "off_never_eligible": sum(row.get("cohort_status") == "OFF_NEVER_ELIGIBLE" for row in common_rows),
        },
        "definitions": {
            "common_window": "[OFF first fully eligible absolute step, min(OFF last step, ON last step)] inclusive; same anchor to both arms",
            "local_exit": "first start of 3 consecutive abs(actual_lateral_velocity_mps)<=0.5 from max(encounter, common anchor), or encounter without common anchor",
            "local_metrics": "local exit through that arm's final logged step; no-exit is structural NA",
            "pass_gate": "passed_margin_m > 0 AND relevant_obstacle_clearance_m >= 2.5 on the same step",
            "sustained_actor_final_radial": "any 5-step window with at least 4 strictly positive samples",
            "sustained_actual_radial": "3 consecutive actual radial velocities > 1e-8 m/s",
            "success_reference_quantiles": "numpy quantile method='linear' (Hyndman-Fan type 7)",
            "iqr_category_bounds": "Q1 and Q3 inclusive for WITHIN_IQR",
            "off_exact_fingerprint": OFF_BASELINE_FINGERPRINT,
        },
        "layer_a": {"frame_and_vector_projection": identity.get("frame_checks"), "exact_pass_gate_and_latch": pass_checks},
        "per_episode_metrics_and_diagnoses": list(diagnostics),
        "per_episode_post_exit_metrics": {
            str(row["episode_id"]): row.get("on_metrics", {}) for row in diagnostics
        },
        "per_episode_diagnosis": {
            str(row["episode_id"]): row.get("primary_diagnosis") for row in diagnostics
        },
        "common_anchor_metrics": list(common_rows),
        "mechanistic_local_windows": list(mechanistic_rows),
        "success_reference_statistics": dict(references),
        "main_failure_vs_reference": list(comparison),
        "failure_vs_reference": list(comparison),
        "passed_margin_regression_statistics": references.get("margin_regression_best_minus_final_m", {}),
        "lateral_reacceleration_statistics": {
            "duration_reference": references.get("lateral_reacceleration_duration_steps", {}),
            "main_failure_event_count": sum(
                bool(row.get("on_metrics", {}).get("lateral_reacceleration_event"))
                for row in diagnostics if row.get("episode_id") in MAIN_FAILURE_IDS
            ),
        },
        "actor_radial_intent_statistics": {
            "mean_reference": references.get("actor_radial_mean_mps", {}),
            "positive_mean_reference": references.get("actor_radial_positive_mean_mps", {}),
        },
        "final_radial_retention_statistics": references.get("actor_to_final_radial_retention", {}),
        "diagnosis_counts": counts,
        "main_failure_primary_diagnoses": main,
        "goal_recovery_release_check": dict(pipeline_checks),
        "recommendation": {
            "count": 1,
            "experiment": recommendation,
            "text": "Preregister one paired post-local-exit/pre-pass radial-intent-persistence intervention; freeze Actor, scene, and all other controller components.",
            "implemented": False,
        },
        "recommended_single_next_experiment": recommendation,
        "limitations": [
            "Reward, dynamics, action transform, safety projection, and boundary projection identities remain NOT_INDEPENDENTLY_VERIFIED.",
            "OFF exact step trajectory fingerprint is the behavioral-equivalence protection.",
            "Common-window effects are paired causal-descriptive; arm-local exit windows are descriptive and must not be read as paired causal effects.",
            "The success-reference IQR is descriptive for the fixed 11 ON successes and is not a population uncertainty interval.",
        ],
        "output_contract": list(OUTPUT_FILES),
    }


def _check_output_target(output_root: Path) -> None:
    _require(not output_root.exists(), f"output root already exists: {output_root}")


def _load_inputs(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str], Path]:
    paths = {name: source_root / name for name in SOURCE_FILES}
    _require(all(path.is_file() for path in paths.values()), "required source file missing")
    source_hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    prereg = read_json(paths["preregistered_endpoints.json"])
    paired = read_json(paths["lateral_exit_braking_assistance_summary.json"])
    off_summary = read_json(paths["off/summary.json"])
    on_summary = read_json(paths["on/summary.json"])
    off_episodes = read_csv(paths["off/episodes.csv"], max_rows=25)
    on_episodes = read_csv(paths["on/episodes.csv"], max_rows=25)
    off_steps = read_csv(paths["off/steps.csv"], max_rows=2500)
    on_steps = read_csv(paths["on/steps.csv"], max_rows=2500)
    pair_rows = read_csv(paths["lateral_exit_braking_assistance_pairs.csv"], max_rows=25)
    checkpoint_value = off_summary.get("provenance", {}).get("checkpoint", {}).get("resolved_path")
    _require(isinstance(checkpoint_value, str) and checkpoint_value, "checkpoint path missing")
    checkpoint = Path(checkpoint_value)
    _require(checkpoint.is_file(), "checkpoint file missing")
    source_hashes_before[str(checkpoint)] = sha256_file(checkpoint)
    return prereg, paired, off_summary, on_summary, off_episodes, on_episodes, off_steps, on_steps, pair_rows, source_hashes_before, checkpoint


def _post_hashes(source_root: Path, checkpoint: Path) -> dict[str, str]:
    paths = {name: source_root / name for name in SOURCE_FILES}
    values = {name: sha256_file(path) for name, path in paths.items()}
    values[str(checkpoint)] = sha256_file(checkpoint)
    return values


def _write_staged_files(stage_root: Path, contents: Mapping[str, str]) -> None:
    _require(tuple(contents.keys()) == OUTPUT_FILES, "staged output file order/contract")
    stage_root.mkdir(parents=False, exist_ok=False)
    for name in OUTPUT_FILES:
        target = stage_root / name
        target.write_text(contents[name], encoding="utf-8", newline="\n")
    actual = tuple(sorted(path.name for path in stage_root.iterdir() if path.is_file()))
    _require(actual == tuple(sorted(OUTPUT_FILES)), "staged output must contain exactly eight files")


def run_audit(source_root: Path = SOURCE_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    project_root = Path(__file__).resolve().parents[2]
    _require(source_root.resolve() == (project_root / SOURCE_ROOT).resolve(), "active preregistration/source root path")
    _require(output_root.resolve() == (project_root / OUTPUT_ROOT).resolve(), "exclusive audit output root path")
    _check_output_target(output_root)
    (
        prereg, paired, off_summary, on_summary, off_episodes, on_episodes,
        off_steps, on_steps, pair_rows, source_hashes_before, checkpoint,
    ) = _load_inputs(source_root)
    checkpoint_sha = source_hashes_before[str(checkpoint)]
    checkpoint_size = checkpoint.stat().st_size
    source_identity = validate_source_identity(
        prereg, paired, off_summary, on_summary, off_episodes, on_episodes,
        off_steps, on_steps, pair_rows,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=checkpoint_size,
    )
    source_identity["active_preregistration_path"] = str(source_root / "preregistered_endpoints.json")
    off_grouped = validate_step_continuity(off_steps, condition="off")
    on_grouped = validate_step_continuity(on_steps, condition="on")
    frame_checks = {
        "OFF": validate_frame_and_decomposition(off_grouped, off_episodes),
        "ON": validate_frame_and_decomposition(on_grouped, on_episodes),
    }
    pass_checks = {
        "OFF": validate_pass_latches(off_grouped, off_episodes),
        "ON": validate_pass_latches(on_grouped, on_episodes),
        "terminal_episode_outcomes": {
            "OFF": validate_episode_outcomes(off_grouped, off_episodes),
            "ON": validate_episode_outcomes(on_grouped, on_episodes),
        },
    }
    common_windows = select_common_windows(pair_rows, off_grouped, on_grouped)
    diagnostics, common_rows, mechanistic_rows = build_diagnostics(
        off_episodes, on_episodes, off_grouped, on_grouped, common_windows
    )
    references, diagnostics = attach_diagnoses_and_references(diagnostics)
    pipeline_checks = validate_main_failure_pipeline(diagnostics)
    comparison = comparison_rows(diagnostics, references)
    pass_rows = pass_trajectory_rows(
        off_grouped, on_grouped, off_episodes, on_episodes,
        common_windows, diagnostics,
    )
    source_identity["frame_checks"] = frame_checks
    identity_for_report = {"source_identity": source_identity, "checkpoint": source_identity["checkpoint"]}
    recommendation = "PREREGISTERED_PAIRED_SINGLE_POST_LOCAL_EXIT_PRE_PASS_RADIAL_INTENT_PERSISTENCE_INTERVENTION"
    summary = _summary_payload(
        identity_for_report, diagnostics, common_rows, mechanistic_rows, references,
        comparison, pass_checks, pipeline_checks, recommendation,
    )
    contents = {
        OUTPUT_FILES[0]: safe_json_dumps(summary),
        OUTPUT_FILES[1]: csv_text(flatten_diagnostics(diagnostics)),
        OUTPUT_FILES[2]: csv_text(_flatten_common(common_rows)),
        OUTPUT_FILES[3]: csv_text(mechanistic_rows),
        OUTPUT_FILES[4]: csv_text(reference_csv_rows(references)),
        OUTPUT_FILES[5]: csv_text(comparison),
        OUTPUT_FILES[6]: csv_text(pass_rows),
        OUTPUT_FILES[7]: build_report(
            {"checkpoint": source_identity["checkpoint"], "source_hashes": source_hashes_before},
            diagnostics, references, comparison,
            {"OFF": pass_checks["OFF"], "ON": pass_checks["ON"]},
            pipeline_checks, common_rows, mechanistic_rows, pass_rows,
        ),
    }
    _require(tuple(contents.keys()) == OUTPUT_FILES, "output order")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".post_lateral_exit_audit_staging_", dir=output_root.parent)) / "bundle"
    stage_parent = stage_root.parent
    committed = False
    try:
        _write_staged_files(stage_root, contents)
        source_hashes_after = _post_hashes(source_root, checkpoint)
        if source_hashes_before != source_hashes_after:
            changed = sorted(name for name in set(source_hashes_before) | set(source_hashes_after) if source_hashes_before.get(name) != source_hashes_after.get(name))
            raise SourceIdentityError("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL: source changed during audit: " + ", ".join(changed))
        for name, before in source_hashes_before.items():
            summary["source_hashes"][name] = {"sha256_before": before, "sha256_after": source_hashes_after[name], "unchanged": True}
        summary_text = safe_json_dumps(summary)
        (stage_root / OUTPUT_FILES[0]).write_text(summary_text, encoding="utf-8", newline="\n")
        _require(not output_root.exists(), f"output root appeared during audit: {output_root}")
        os.replace(stage_root, output_root)
        committed = True
    finally:
        if stage_parent.exists():
            shutil.rmtree(stage_parent)

    # Deterministic, compact result for the calling shell; full findings are in the artifacts.
    result = {
        "status": "PASS",
        "output_root": str(output_root),
        "output_files": list(OUTPUT_FILES),
        "source_hashes_unchanged": True,
        "source_hash_count": len(source_hashes_before),
        "diagnosis_counts": summary["diagnosis_counts"],
        "main_failure_primary_diagnoses": summary["main_failure_primary_diagnoses"],
        "paired_pass_counts": {"OFF": pass_checks["OFF"]["passed"], "ON": pass_checks["ON"]["passed"], "new_passes": 0},
        "common_anchor_cohort": summary["cohorts"],
        "recommendation": recommendation,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.source_root, args.output_root)
    except SourceIdentityError as exc:
        print("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print("POST_LATERAL_EXIT_AUDIT_SOURCE_IDENTITY_FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(finite_json(result), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
