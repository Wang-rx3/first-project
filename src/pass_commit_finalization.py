"""Pure validation and offline recovery for the single-UAV pass-commit result.

This module deliberately has no project imports.  It consumes already recorded
CSV/JSON artifacts and can therefore be used by the offline recovery command
without importing an environment, controller, evaluator, or training runtime.
"""

from __future__ import annotations

import argparse
import csv
import copy
import datetime as _datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence


_STAGE_ROOT = Path("logs/preliminary/single_uav_pass_commit_diagnostic_seed2026")
DEFAULT_SOURCE_ROOT = _STAGE_ROOT / "run_action_decomposition_fix_v1"
DEFAULT_PREPARATION_ROOT = _STAGE_ROOT / "preparation_action_decomposition_fix_v1"
DEFAULT_OUTPUT_ROOT = _STAGE_ROOT / "offline_recovery_action_decomposition_fix_v1_20261010"
DEFAULT_BASELINE_MANIFEST = Path(
    "logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/workspace_baseline.json"
)

_IDENTITY_FIELDS = ("episode_index", "episode_id", "scenario_seed", "noise_seed", "scenario_hash")
_STEP_IDENTITY_FIELDS = ("episode_index", "episode_id", "scenario_id", "scenario_seed", "noise_seed", "step")
_CORE_AGGREGATE_FIELDS = (
    "episodes", "entered_8m", "entered_5m", "entered_3m", "entered_1m",
    "stable_goal", "timeout", "collision", "passed_obstacle",
    "post_bypass_recovery", "safety_projection_steps", "boundary_projection_steps",
)
_EPISODE_SUM_FIELDS = ("entered_8m", "entered_5m", "entered_3m", "entered_1m", "stable_goal", "timeout")
_EPISODE_FINAL_FIELDS = ("passed_obstacle", "post_bypass_recovery")
_STEP_SUM_FIELDS = {
    "collision": "collision",
    "safety_projection_steps": "safety_projection_applied",
    "boundary_projection_steps": "boundary_projection_applied",
    "pass_commit_active_steps": "pass_commit_active",
}


class PassCommitValidationError(ValueError):
    """Raised when recorded artifacts do not satisfy the paired contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise PassCommitValidationError(message)


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    return value


def _rows(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{name} must be a sequence of row mappings")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            _fail(f"{name}[{index}] must be a mapping")
        result.append(row)
    return result


def _text(value: Any, field: str) -> str:
    if value is None:
        _fail(f"missing {field}")
    text = str(value).strip()
    if not text:
        _fail(f"empty {field}")
    return text


def _integer(value: Any, field: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool):
        number = int(value)
    else:
        text = _text(value, field)
        try:
            number_float = float(text)
        except (TypeError, ValueError):
            _fail(f"{field} is not an integer: {value!r}")
        if not math.isfinite(number_float) or not number_float.is_integer():
            _fail(f"{field} is not an integer: {value!r}")
        number = int(number_float)
    if nonnegative and number < 0:
        _fail(f"{field} must be nonnegative")
    return number


def _flag(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    text = _text(value, field).lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    number = _integer(text, field)
    if number not in (0, 1):
        _fail(f"{field} must be a 0/1 flag")
    return number


def _finite(value: Any, field: str) -> float:
    text = _text(value, field)
    try:
        number = float(text)
    except (TypeError, ValueError):
        _fail(f"{field} is not numeric: {value!r}")
    if not math.isfinite(number):
        _fail(f"{field} must be finite")
    return number


def _expected_ids(preregistration: Mapping[str, Any]) -> list[str]:
    population = _mapping(preregistration.get("population"), "preregistration.population")
    ordered = population.get("ordered_episode_ids")
    if not isinstance(ordered, Sequence) or isinstance(ordered, (str, bytes, bytearray)):
        _fail("preregistration.population.ordered_episode_ids must be a sequence")
    identifiers = [_text(item, "episode_id") for item in ordered]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        _fail("preregistration episode IDs must be nonempty and unique")
    return identifiers


def _extract_arm(results: Mapping[str, Any], condition: str) -> Mapping[str, Any]:
    if condition not in results:
        _fail(f"missing paired arm {condition!r}")
    arm = _mapping(results[condition], f"results[{condition!r}]")
    # The public API accepts rows directly.  Supporting the evaluator's
    # summary-shaped result here keeps recovery independent of evaluator code.
    if "episodes" in arm and "step_rows" in arm:
        return arm
    nested = arm.get("results")
    if isinstance(nested, Mapping) and condition in nested:
        return _extract_arm({condition: nested[condition]}, condition)
    _fail(f"results[{condition!r}] requires episodes and step_rows")


def _identity_tuple(row: Mapping[str, Any], fields: Sequence[str], prefix: str) -> tuple[str, ...]:
    return tuple(_text(row.get(field), f"{prefix}.{field}") for field in fields)


def _validate_arm(
    condition: str,
    arm: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, list[Mapping[str, Any]]], dict[str, int]]:
    episodes = _rows(arm.get("episodes"), f"{condition}.episodes")
    steps = _rows(arm.get("step_rows"), f"{condition}.step_rows")
    if [str(row.get("episode_id", "")) for row in episodes] != list(expected_ids):
        seen = [str(row.get("episode_id", "")) for row in episodes]
        _fail(f"{condition} episodes must exactly follow preregistered order; got {seen!r}")
    if len(episodes) != len(expected_ids):
        _fail(f"{condition} episode count mismatch")
    episode_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(episodes, start=1):
        episode_id = _text(row.get("episode_id"), f"{condition}.episodes[{index}].episode_id")
        if episode_id in episode_by_id:
            _fail(f"duplicate {condition} episode ID {episode_id}")
        episode_by_id[episode_id] = row
        if _integer(row.get("episode_index"), f"{condition}.{episode_id}.episode_index") != index:
            _fail(f"{condition}.{episode_id} episode_index is not {index}")
        if _text(row.get("condition"), f"{condition}.{episode_id}.condition").lower() != condition:
            _fail(f"{condition}.{episode_id} has the wrong condition")
        for field in ("scenario_seed", "noise_seed"):
            _integer(row.get(field), f"{condition}.{episode_id}.{field}")
        _text(row.get("scenario_hash"), f"{condition}.{episode_id}.scenario_hash")
        _integer(row.get("steps"), f"{condition}.{episode_id}.steps", nonnegative=True)

    steps_by_id: dict[str, list[Mapping[str, Any]]] = {episode_id: [] for episode_id in expected_ids}
    seen_step_keys: set[tuple[str, int]] = set()
    for row_index, row in enumerate(steps):
        episode_id = _text(row.get("episode_id"), f"{condition}.step_rows[{row_index}].episode_id")
        if episode_id not in episode_by_id:
            _fail(f"{condition} step references missing or extra episode {episode_id!r}")
        if _text(row.get("condition"), f"{condition}.{episode_id}.step.condition").lower() != condition:
            _fail(f"{condition}.{episode_id} step has the wrong condition")
        step = _integer(row.get("step"), f"{condition}.{episode_id}.step", nonnegative=True)
        key = (episode_id, step)
        if key in seen_step_keys:
            _fail(f"duplicate {condition} step key {key!r}")
        seen_step_keys.add(key)
        episode = episode_by_id[episode_id]
        for field in ("episode_index", "scenario_id", "scenario_seed", "noise_seed"):
            _text(row.get(field), f"{condition}.{episode_id}.step.{field}")
        if _integer(row.get("episode_index"), "step.episode_index") != _integer(episode.get("episode_index"), "episode_index"):
            _fail(f"{condition}.{episode_id} step episode_index mismatch")
        for field in ("scenario_seed", "noise_seed"):
            if _integer(row.get(field), f"step.{field}") != _integer(episode.get(field), f"episode.{field}"):
                _fail(f"{condition}.{episode_id} step {field} mismatch")
        _finite(row.get("minimum_clearance_m"), f"{condition}.{episode_id}.step[{step}].minimum_clearance_m")
        for field in ("passed_obstacle", "post_bypass_recovery", "collision", "safety_projection_applied", "boundary_projection_applied", "pass_commit_active"):
            _flag(row.get(field), f"{condition}.{episode_id}.step[{step}].{field}")
        steps_by_id[episode_id].append(row)

    counters: dict[str, int] = {}
    for episode_id in expected_ids:
        episode = episode_by_id[episode_id]
        local = steps_by_id[episode_id]
        expected_count = _integer(episode.get("steps"), f"{condition}.{episode_id}.steps", nonnegative=True)
        if len(local) != expected_count:
            _fail(f"{condition}.{episode_id} step count {len(local)} != episode.steps {expected_count}")
        local.sort(key=lambda row: _integer(row.get("step"), "step"))
        if [_integer(row.get("step"), "step") for row in local] != list(range(expected_count)):
            _fail(f"{condition}.{episode_id} steps are not contiguous from zero")
        if not local:
            _fail(f"{condition}.{episode_id} has no step rows")
        for field in _EPISODE_FINAL_FIELDS:
            if _flag(episode.get(field), f"{condition}.{episode_id}.{field}") != _flag(local[-1].get(field), f"last.{field}"):
                _fail(f"{condition}.{episode_id}.{field} does not agree with its final step")
        for field, step_field in _STEP_SUM_FIELDS.items():
            total = sum(_flag(row.get(step_field), f"{condition}.{episode_id}.{step_field}") for row in local)
            if field not in episode:
                _fail(f"{condition}.{episode_id} missing counter {field}")
            observed = _integer(episode.get(field), f"{condition}.{episode_id}.{field}", nonnegative=True)
            if observed != total:
                _fail(f"{condition}.{episode_id}.{field} {observed} != step total {total}")
            counters[field] = counters.get(field, 0) + total
        for field in _EPISODE_SUM_FIELDS:
            _integer(episode.get(field), f"{condition}.{episode_id}.{field}", nonnegative=True)
        # Episode collision is a counter in the recorded summary and should
        # equal the number of collision flags, even when the final row is not
        # the collision row in a hand-built fixture.
        collision_total = sum(_flag(row.get("collision"), "collision") for row in local)
        if _integer(episode.get("collision"), f"{condition}.{episode_id}.collision", nonnegative=True) != collision_total:
            _fail(f"{condition}.{episode_id}.collision does not agree with steps")
    return episodes, steps, episode_by_id, steps_by_id, counters


def _aggregate_check(
    condition: str,
    arm: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    counters: Mapping[str, int],
) -> Mapping[str, Any]:
    aggregate = _mapping(arm.get("aggregate"), f"{condition}.aggregate")
    expected: dict[str, int] = {"episodes": len(episodes)}
    for field in _EPISODE_SUM_FIELDS:
        expected[field] = sum(_flag(row.get(field), f"{condition}.{field}") for row in episodes)
    for field in ("passed_obstacle", "post_bypass_recovery"):
        expected[field] = sum(_flag(row.get(field), f"{condition}.{field}") for row in episodes)
    expected["collision"] = counters["collision"]
    expected["safety_projection_steps"] = counters["safety_projection_steps"]
    expected["boundary_projection_steps"] = counters["boundary_projection_steps"]
    for field in _CORE_AGGREGATE_FIELDS:
        if field not in aggregate:
            _fail(f"{condition}.aggregate missing required core field {field}")
        observed = _integer(aggregate[field], f"{condition}.aggregate.{field}", nonnegative=True)
        if observed != expected[field]:
            _fail(f"{condition}.aggregate.{field} {observed} != derived {expected[field]}")
    if "pass_commit_active_steps" in aggregate:
        observed = _integer(aggregate["pass_commit_active_steps"], f"{condition}.aggregate.pass_commit_active_steps", nonnegative=True)
        if observed != counters["pass_commit_active_steps"]:
            _fail(f"{condition}.aggregate.pass_commit_active_steps disagrees with steps")
    return aggregate


def _paired_identity_check(
    expected_ids: Sequence[str],
    off_episodes: Mapping[str, Mapping[str, Any]],
    on_episodes: Mapping[str, Mapping[str, Any]],
    off_steps: Mapping[str, Sequence[Mapping[str, Any]]],
    on_steps: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for episode_id in expected_ids:
        off = off_episodes[episode_id]
        on = on_episodes[episode_id]
        if _identity_tuple(off, _IDENTITY_FIELDS, f"off.{episode_id}") != _identity_tuple(on, _IDENTITY_FIELDS, f"on.{episode_id}"):
            _fail(f"paired identity mismatch for {episode_id}")
        off_local = off_steps[episode_id]
        on_local = on_steps[episode_id]
        for off_row, on_row in zip(off_local, on_local):
            for field in _STEP_IDENTITY_FIELDS:
                if _text(off_row.get(field), f"off.{episode_id}.step.{field}") != _text(on_row.get(field), f"on.{episode_id}.step.{field}"):
                    _fail(f"paired step identity mismatch for {episode_id} step {off_row.get('step')}: {field}")


def _suffix(episode_id: str) -> str:
    return episode_id.rsplit("_", 1)[-1] if "_" in episode_id else episode_id


def build_pass_commit_paired_summary(
    results: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    attempt_id: str,
    off_baseline_fingerprint_match: bool,
) -> dict[str, Any]:
    """Validate two recorded arms and build the compact paired result.

    The function only reads its arguments and deep-copies returned structures.
    It does not import runtime code, mutate rows, or write files.
    """
    if not isinstance(results, Mapping):
        _fail("results must be a mapping")
    if not isinstance(preregistration, Mapping):
        _fail("preregistration must be a mapping")
    attempt = _text(attempt_id, "attempt_id")
    if not isinstance(off_baseline_fingerprint_match, bool):
        _fail("off_baseline_fingerprint_match must be bool")
    expected_ids = _expected_ids(preregistration)
    off = _extract_arm(results, "off")
    on = _extract_arm(results, "on")
    off_episodes, off_step_rows, off_by_id, off_steps_by_id, off_counters = _validate_arm("off", off, expected_ids)
    on_episodes, on_step_rows, on_by_id, on_steps_by_id, on_counters = _validate_arm("on", on, expected_ids)
    _paired_identity_check(expected_ids, off_by_id, on_by_id, off_steps_by_id, on_steps_by_id)
    off_aggregate = _aggregate_check("off", off, off_episodes, off_counters)
    on_aggregate = _aggregate_check("on", on, on_episodes, on_counters)

    paired_outcomes: list[dict[str, Any]] = []
    for episode_id in expected_ids:
        off_episode = off_by_id[episode_id]
        on_episode = on_by_id[episode_id]
        off_clearances = [_finite(row.get("minimum_clearance_m"), f"off.{episode_id}.minimum_clearance_m") for row in off_steps_by_id[episode_id]]
        on_clearances = [_finite(row.get("minimum_clearance_m"), f"on.{episode_id}.minimum_clearance_m") for row in on_steps_by_id[episode_id]]
        paired_outcomes.append({
            "episode_id": episode_id,
            "off_passed_obstacle": _flag(off_episode.get("passed_obstacle"), "off.passed_obstacle"),
            "on_passed_obstacle": _flag(on_episode.get("passed_obstacle"), "on.passed_obstacle"),
            "off_post_bypass_recovery": _flag(off_episode.get("post_bypass_recovery"), "off.post_bypass_recovery"),
            "on_post_bypass_recovery": _flag(on_episode.get("post_bypass_recovery"), "on.post_bypass_recovery"),
            "off_collision": _integer(off_episode.get("collision"), "off.collision", nonnegative=True),
            "on_collision": _integer(on_episode.get("collision"), "on.collision", nonnegative=True),
            "off_minimum_clearance_m": min(off_clearances),
            "on_minimum_clearance_m": min(on_clearances),
            "off_safety_projection_steps": _integer(off_episode.get("safety_projection_steps"), "off.safety_projection_steps", nonnegative=True),
            "on_safety_projection_steps": _integer(on_episode.get("safety_projection_steps"), "on.safety_projection_steps", nonnegative=True),
            "off_boundary_projection_steps": _integer(off_episode.get("boundary_projection_steps"), "off.boundary_projection_steps", nonnegative=True),
            "on_boundary_projection_steps": _integer(on_episode.get("boundary_projection_steps"), "on.boundary_projection_steps", nonnegative=True),
            "off_pass_commit_active_steps": _integer(off_episode.get("pass_commit_active_steps", 0), "off.pass_commit_active_steps", nonnegative=True),
            "on_pass_commit_active_steps": _integer(on_episode.get("pass_commit_active_steps", 0), "on.pass_commit_active_steps", nonnegative=True),
            "off_first_pass_commit_trigger_step": _copy(off_episode.get("first_pass_commit_trigger_step", "")),
            "on_first_pass_commit_trigger_step": _copy(on_episode.get("first_pass_commit_trigger_step", "")),
            "off_first_pass_commit_active_step": _copy(off_episode.get("first_pass_commit_active_step", "")),
            "on_first_pass_commit_active_step": _copy(on_episode.get("first_pass_commit_active_step", "")),
        })

    population = _mapping(preregistration.get("population"), "preregistration.population")
    tiers = _copy(preregistration.get("endpoint_hierarchy", []))
    special: dict[str, Any] = {}
    safety_id = population.get("safety_special")
    boundary_id = population.get("boundary_special")
    rescue_id = population.get("rescue_descriptive_control")
    if safety_id is not None:
        sid = _text(safety_id, "population.safety_special")
        if sid not in off_by_id:
            _fail(f"safety special ID {sid!r} is outside preregistered population")
        overlap_rows = [
            row for row in on_steps_by_id[sid]
            if _flag(row.get("pass_commit_active"), "pass_commit_active") and _flag(row.get("safety_projection_applied"), "safety_projection_applied")
        ]
        special[f"safety_special_{_suffix(sid)}"] = {
            "episode_id": sid,
            "off_safety_projection_steps": _integer(off_by_id[sid].get("safety_projection_steps"), "safety off projection", nonnegative=True),
            "on_safety_projection_steps": _integer(on_by_id[sid].get("safety_projection_steps"), "safety on projection", nonnegative=True),
            "pass_commit_safety_overlap_steps": len(overlap_rows),
            "first_overlap_step": _integer(overlap_rows[0].get("step"), "first overlap step", nonnegative=True) if overlap_rows else "",
            "off_minimum_clearance_m": min(_finite(row.get("minimum_clearance_m"), "safety off clearance") for row in off_steps_by_id[sid]),
            "on_minimum_clearance_m": min(_finite(row.get("minimum_clearance_m"), "safety on clearance") for row in on_steps_by_id[sid]),
            "included_in_main_mechanism_vote": False,
        }
    if boundary_id is not None:
        bid = _text(boundary_id, "population.boundary_special")
        if bid not in off_by_id:
            _fail(f"boundary special ID {bid!r} is outside preregistered population")
        special[f"boundary_special_{_suffix(bid)}"] = {
            "episode_id": bid,
            "off_boundary_projection_steps": _integer(off_by_id[bid].get("boundary_projection_steps"), "boundary off projection", nonnegative=True),
            "on_boundary_projection_steps": _integer(on_by_id[bid].get("boundary_projection_steps"), "boundary on projection", nonnegative=True),
            "included_in_main_mechanism_vote": False,
        }
    if rescue_id is not None:
        rid = _text(rescue_id, "population.rescue_descriptive_control")
        if rid not in off_by_id:
            _fail(f"rescue special ID {rid!r} is outside preregistered population")
        special[f"rescue_descriptive_control_{_suffix(rid)}"] = {
            "episode_id": rid,
            "off_passed_obstacle": _flag(off_by_id[rid].get("passed_obstacle"), "rescue off passed_obstacle"),
            "on_passed_obstacle": _flag(on_by_id[rid].get("passed_obstacle"), "rescue on passed_obstacle"),
            "used_to_tune_gate_or_cap": False,
        }

    cohorts = {
        "main_mechanism_subgroup": _copy(population.get("main_mechanism_subgroup", [])),
        "rescue_descriptive_control": _copy(rescue_id) if rescue_id is not None else None,
        "safety_special": _copy(safety_id) if safety_id is not None else None,
        "boundary_special": _copy(boundary_id) if boundary_id is not None else None,
    }
    for item in cohorts["main_mechanism_subgroup"]:
        if item not in expected_ids:
            _fail(f"main mechanism cohort contains unregistered ID {item!r}")

    off_provenance = _copy(off.get("provenance", {}))
    on_provenance = _copy(on.get("provenance", {}))
    provenance = {
        "off": off_provenance,
        "on": on_provenance,
        "scenario_manifest_hash": on_provenance.get("scenario_manifest_hash", off_provenance.get("scenario_manifest_hash")),
        "ordered_pool_identity": _copy(on_provenance.get("ordered_pool_identity", off_provenance.get("ordered_pool_identity"))),
        "attempt_id": attempt,
        "off_baseline_fingerprint_match": off_baseline_fingerprint_match,
    }
    summary: dict[str, Any] = {
        "schema_version": "pass_commit_paired_itt_summary_v1",
        "attempt_id": attempt,
        "population": list(expected_ids),
        "paired_identity_match": True,
        "off_baseline_fingerprint_match": off_baseline_fingerprint_match,
        "only_treatment_difference": "PASS_COMMIT ON versus OFF; lateral braking ON and radial-intent OFF in both arms",
        "paired_episode_outcomes": paired_outcomes,
        "tiers": tiers,
        "aggregates": {"OFF": _copy(off_aggregate), "ON": _copy(on_aggregate)},
        "cohorts": cohorts,
        "special_cohorts": special,
        "provenance": provenance,
        "off_pass_commit_active_steps": off_counters["pass_commit_active_steps"],
        "on_pass_commit_active_steps": on_counters["pass_commit_active_steps"],
        "pass_commit_active_steps_delta": on_counters["pass_commit_active_steps"] - off_counters["pass_commit_active_steps"],
    }
    # Keep the evaluator's historical field names for the canonical population.
    summary.update(special)
    return summary


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot load JSON {path}: {exc}")


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                _fail(f"invalid or duplicate CSV header in {path}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        _fail(f"cannot load CSV {path}: {exc}")


def _resolve(project_root: Path, path: Path | str | None, default: Path) -> Path:
    candidate = default if path is None else Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _assert_disjoint_output(output: Path, source: Path, preparation: Path) -> None:
    output = output.resolve()
    historical = (
        source.parent / "run",
        source.parent / "preparation",
        source.parent / "run_action_decomposition_fix_v1",
        source.parent / "preparation_action_decomposition_fix_v1",
    )
    guarded = (("source", source.resolve()), ("preparation", preparation.resolve())) + tuple(
        ("historical", item.resolve()) for item in historical
    )
    for name, other in guarded:
        if output == other or other in output.parents or output in other.parents:
            _fail(f"output path overlaps {name} path: {output}")
    if output.exists() or output.is_symlink():
        _fail(f"refusing existing output path: {output}")


def _read_arm_files(source_root: Path, condition: str) -> dict[str, Any]:
    arm_root = source_root / condition
    episodes = _load_csv(arm_root / "episodes.csv")
    step_rows = _load_csv(arm_root / "steps.csv")
    summary = _load_json(arm_root / "summary.json")
    summary_map = _mapping(summary, f"{condition}.summary")
    return {
        "episodes": episodes,
        "step_rows": step_rows,
        "aggregate": _mapping(summary_map.get("aggregate"), f"{condition}.summary.aggregate"),
        "provenance": _mapping(summary_map.get("provenance", {}), f"{condition}.summary.provenance"),
        "summary": summary_map,
    }


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
            return {"__float__": "+inf" if number > 0 else "-inf"}
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
            return {"__float__": "+inf" if number > 0 else "-inf"}
        return number
    return str(value)


def _hash_rows(rows: Iterable[Mapping[str, Any]], keys: Sequence[str], *, normalize_condition: bool = False) -> str:
    normalized = []
    for row in rows:
        values = []
        for key in keys:
            if key not in row:
                _fail(f"baseline row missing locked field {key}")
            values.append(_normal("off" if normalize_condition and key == "condition" else row[key]))
        normalized.append(values)
    return _sha256_bytes(_canonical({"key_order": list(keys), "rows": normalized}).encode("utf-8"))


def _baseline_fingerprint_check(
    off: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    identity = _mapping(contract.get("identity"), "off baseline contract.identity")
    row_keys = list(identity.get("step_key_order", contract.get("inherited_step_key_order", [])))
    episode_keys = list(identity.get("episode_key_order", contract.get("inherited_episode_key_order", [])))
    if not row_keys or not episode_keys:
        return False, {"status": "UNAVAILABLE", "reason": "contract lacks inherited key order"}
    rows = _rows(off.get("step_rows"), "off.step_rows")
    episodes = _rows(off.get("episodes"), "off.episodes")
    ordered_rows = sorted(rows, key=lambda row: (_integer(row.get("episode_index"), "episode_index"), _integer(row.get("step"), "step")))
    expected_traces = identity.get("episode_trace_hashes")
    expected_outcomes = identity.get("episode_outcome_hashes")
    current_traces = {
        str(episode_id): _hash_rows([row for row in ordered_rows if str(row.get("episode_id")) == str(episode_id)], row_keys)
        for episode_id in identity.get("episode_order", [])
    }
    ordered_episodes = sorted(episodes, key=lambda row: _integer(row.get("episode_index"), "episode_index"))
    current_outcomes = {
        str(row.get("episode_id")): _hash_rows([row], episode_keys)
        for row in ordered_episodes
    }
    aggregate = _mapping(off.get("aggregate"), "off.aggregate")
    aggregate_keys = list(identity.get("aggregate_key_order", []))
    aggregate_hash = _sha256_bytes(_canonical({"key_order": aggregate_keys, "values": [_normal(aggregate[key]) for key in aggregate_keys]}).encode("utf-8")) if all(key in aggregate for key in aggregate_keys) else None
    checks = {
        "status": "PASS" if current_traces == expected_traces and current_outcomes == expected_outcomes and aggregate_hash == identity.get("aggregate_sha256") else "FAIL",
        "baseline_fingerprint": contract.get("fingerprint"),
        "row_count": len(rows),
        "episode_count": len(episodes),
        "episode_trace_hashes_match": current_traces == expected_traces,
        "episode_outcome_hashes_match": current_outcomes == expected_outcomes,
        "aggregate_hash_matches": aggregate_hash == identity.get("aggregate_sha256"),
        "only_normalized_field": "condition",
        "new_pass_commit_fields_excluded": True,
    }
    return checks["status"] == "PASS", checks


def _comparison_checks(source_root: Path, expected_ids: Sequence[str], results: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    path = source_root / "goal_recovery_correction_comparison.json"
    if not path.exists():
        return {"status": "UNAVAILABLE", "path": str(path)}
    comparison = _mapping(_load_json(path), "comparison")
    checks: dict[str, Any] = {"path": str(path), "status": "PASS", "episode_count": comparison.get("episode_count")}
    if comparison.get("episode_count") is not None and _integer(comparison["episode_count"], "comparison.episode_count") != len(expected_ids):
        _fail("recorded comparison episode_count mismatch")
    recorded_order = comparison.get("ordered_pool_identity")
    if recorded_order is not None:
        observed_order = [[_integer(row["episode_index"], "episode_index"), str(row["episode_id"]), _integer(row["noise_seed"], "noise_seed")] for row in results["off"]["episodes"]]
        if recorded_order != observed_order:
            _fail("recorded comparison ordered_pool_identity mismatch")
        checks["ordered_pool_identity_match"] = True
    if comparison.get("paired_same_scenarios") is False:
        _fail("recorded comparison says paired scenarios differ")
    recorded_fp = comparison.get("pass_commit_off_baseline_fingerprint")
    if recorded_fp is not None and recorded_fp != contract.get("fingerprint"):
        _fail("recorded comparison OFF baseline fingerprint mismatch")
    frozen = comparison.get("frozen_runtime_component_hashes")
    if isinstance(frozen, Mapping):
        snapshots = [frozen.get(key) for key in ("before_off_or_first_condition", "after_off", "after_on")]
        if all(isinstance(snapshot, Mapping) for snapshot in snapshots):
            if not (snapshots[0] == snapshots[1] == snapshots[2]):
                _fail("recorded frozen runtime component hashes changed between arms")
            checks["frozen_runtime_components_unchanged"] = True
    recorded_results = comparison.get("results")
    if isinstance(recorded_results, Mapping):
        episode_fields = _IDENTITY_FIELDS + (
            "steps", "passed_obstacle", "post_bypass_recovery", "collision",
            "safety_projection_steps", "boundary_projection_steps", "pass_commit_active_steps",
        )
        step_fields = _STEP_IDENTITY_FIELDS + (
            "passed_obstacle", "post_bypass_recovery", "collision",
            "safety_projection_applied", "boundary_projection_applied", "pass_commit_active",
            "minimum_clearance_m",
        )
        for condition in ("off", "on"):
            recorded_arm = recorded_results.get(condition)
            if not isinstance(recorded_arm, Mapping):
                continue
            current_arm = results[condition]
            for field in ("episodes", "step_rows"):
                recorded_rows = recorded_arm.get(field)
                current_rows = current_arm.get(field)
                if isinstance(recorded_rows, Sequence) and isinstance(current_rows, Sequence):
                    if len(recorded_rows) != len(current_rows):
                        _fail(f"recorded comparison {condition}.{field} count mismatch")
                    compare_fields = episode_fields if field == "episodes" else step_fields
                    for index, (recorded_row, current_row) in enumerate(zip(recorded_rows, current_rows)):
                        if not isinstance(recorded_row, Mapping) or not isinstance(current_row, Mapping):
                            _fail(f"recorded comparison {condition}.{field}[{index}] malformed")
                        for key in compare_fields:
                            if key in recorded_row and key in current_row and _normal(recorded_row[key]) != _normal(current_row[key]):
                                _fail(f"recorded comparison {condition}.{field}[{index}] mismatch at {key}")
                    checks[f"{condition}_{field}_evidence_match"] = True
    # Verify the compact comparison metrics against the recorded arm aggregates.
    metrics = {"passed_obstacle", "post_bypass_recovery", "entered_8m", "entered_5m", "entered_3m", "entered_1m", "stable_goal", "boundary_projection_steps", "safety_projection_steps"}
    for row in comparison.get("comparison", []):
        if not isinstance(row, Mapping) or row.get("metric") not in metrics:
            continue
        metric = str(row["metric"])
        off_value = float(results["off"]["aggregate"][metric])
        on_value = float(results["on"]["aggregate"][metric])
        if float(row.get("OFF")) != off_value or float(row.get("ON")) != on_value or float(row.get("delta")) != on_value - off_value:
            _fail(f"recorded comparison metric mismatch for {metric}")
    return checks


def _manifest_hash_checks(project_root: Path, manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {"status": "UNAVAILABLE", "reason": "no baseline manifest supplied"}
    manifest = _mapping(_load_json(manifest_path), "baseline manifest")
    result: dict[str, Any] = {"manifest": str(manifest_path), "protected_file_sha256": {}, "source_sha256_before": {}}
    protected_mismatches = []
    for field in ("protected_file_sha256", "source_sha256_before"):
        registered = _mapping(manifest.get(field, {}), field)
        for raw, expected in registered.items():
            relative = Path(str(raw).replace("\\", "/"))
            path = project_root / relative
            if not path.exists() or not path.is_file():
                result[field][str(raw)] = {"expected": expected, "current": None, "status": "MISSING"}
                if field == "protected_file_sha256":
                    protected_mismatches.append(str(raw))
                continue
            current = _sha256_file(path)
            status = "MATCH" if current == expected else "MISMATCH"
            result[field][str(raw)] = {"expected": expected, "current": current, "status": status}
            if field == "protected_file_sha256" and status != "MATCH":
                protected_mismatches.append(str(raw))
    result["status"] = "PASS" if not protected_mismatches else "FAIL"
    result["protected_mismatches"] = protected_mismatches
    if protected_mismatches:
        _fail("protected baseline files changed before offline recovery: " + ", ".join(protected_mismatches[:5]))
    return result


def _runtime_source_hash_checks(project_root: Path, source_lock: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    registered = _mapping(source_lock.get("source_hashes", {}), "runtime source lock.source_hashes")
    result: dict[str, Any] = {}
    aliases: dict[str, Path] = {}
    for arm_name in ("off", "on"):
        provenance = results.get(arm_name, {}).get("provenance", {})
        checkpoint = provenance.get("checkpoint") if isinstance(provenance, Mapping) else None
        if isinstance(checkpoint, Mapping) and checkpoint.get("path"):
            aliases["checkpoint/best_validation.pt"] = Path(str(checkpoint["path"]))
    for raw, expected in registered.items():
        path = project_root / Path(str(raw).replace("\\", "/"))
        if not path.exists() and str(raw) in aliases:
            path = aliases[str(raw)]
        if not path.exists():
            result[str(raw)] = {"expected": expected, "current": None, "status": "MISSING"}
        else:
            current = _sha256_file(path)
            is_code = str(raw).replace("\\", "/").startswith("src/")
            if current == expected:
                status = "MATCH"
            elif is_code:
                status = "CODE_DRIFT"
            else:
                status = "IMMUTABLE_INPUT_DRIFT"
            result[str(raw)] = {"expected": expected, "current": current, "status": status, "absolute_path": str(path.resolve())}
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_input_manifest(project_root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    output = []
    for path in sorted({path.resolve() for path in paths}):
        if path.exists() and path.is_file():
            relative = str(path.relative_to(project_root)).replace("\\", "/")
            output.append({"path": relative, "relative_path": relative, "absolute_path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return output


def recover_pass_commit_summary(
    project_root: Path | str = Path("."),
    source_root: Path | str | None = None,
    preparation_root: Path | str | None = None,
    output_root: Path | str | None = None,
    baseline_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Recover compact paired summary files from immutable offline artifacts."""
    project = Path(project_root).resolve()
    source = _resolve(project, source_root, DEFAULT_SOURCE_ROOT)
    preparation = _resolve(project, preparation_root, DEFAULT_PREPARATION_ROOT)
    output = _resolve(project, output_root, DEFAULT_OUTPUT_ROOT)
    manifest = None if baseline_manifest is None else _resolve(project, baseline_manifest, DEFAULT_BASELINE_MANIFEST)
    _assert_disjoint_output(output, source, preparation)
    prereg_path = preparation / "preregistered_pass_commit_endpoints.json"
    lock_path = preparation / "pass_commit_preregistration_lock.json"
    source_lock_path = preparation / "pass_commit_runtime_source_lock.json"
    contract_path = preparation / "pass_commit_off_baseline_contract.json"
    preregistration = _mapping(_load_json(prereg_path), "preregistration")
    lock = _mapping(_load_json(lock_path), "preregistration lock")
    source_lock = _mapping(_load_json(source_lock_path), "runtime source lock")
    contract = _mapping(_load_json(contract_path), "OFF baseline contract")
    attempt_id = _text(source_lock.get("attempt_id", preregistration.get("attempt_id")), "attempt_id")
    prereg_attempt = _text(preregistration.get("attempt_id"), "preregistration.attempt_id")
    if attempt_id != prereg_attempt:
        _fail("runtime source lock attempt_id differs from preregistration")
    current_prereg_hash = _sha256_bytes(_canonical(preregistration).encode("utf-8"))
    locked_prereg_hash = lock.get("active_pass_commit_preregistration_hash")
    if current_prereg_hash != locked_prereg_hash:
        _fail("canonical preregistration hash does not match the immutable lock")
    if source_lock.get("active_pass_commit_preregistration_hash") != current_prereg_hash:
        _fail("runtime source lock preregistration hash mismatch")
    if source_lock.get("preregistration_payload_sha256") not in (None, current_prereg_hash):
        _fail("runtime source lock payload hash mismatch")
    results = {condition: _read_arm_files(source, condition) for condition in ("off", "on")}
    # Validate all core data before creating any output directory.
    paired = build_pass_commit_paired_summary(
        results,
        preregistration=preregistration,
        attempt_id=attempt_id,
        off_baseline_fingerprint_match=False,
    )
    baseline_match, baseline_checks = _baseline_fingerprint_check(results["off"], contract)
    if not baseline_match:
        _fail("OFF baseline fingerprint validation failed")
    paired["off_baseline_fingerprint_match"] = True
    paired["provenance"]["off_baseline_fingerprint_match"] = True
    paired["off_baseline_fingerprint"] = contract.get("fingerprint")
    comparison_checks = _comparison_checks(source, paired["population"], results, contract)
    manifest_checks = _manifest_hash_checks(project, manifest)
    source_hash_checks = _runtime_source_hash_checks(project, source_lock, results)
    immutable_drift = [key for key, item in source_hash_checks.items() if item.get("status") in {"IMMUTABLE_INPUT_DRIFT", "MISSING"}]
    if immutable_drift:
        _fail("immutable runtime inputs changed or are unavailable: " + ", ".join(immutable_drift[:5]))
    prereg_current_hash = current_prereg_hash
    prereg_lock_hash = locked_prereg_hash
    lock_status = "MATCH"
    input_paths = [prereg_path, lock_path, source_lock_path, contract_path]
    for condition in ("off", "on"):
        input_paths.extend(source / condition / name for name in ("episodes.csv", "steps.csv", "summary.json"))
    comparison_path = source / "goal_recovery_correction_comparison.json"
    if comparison_path.exists():
        input_paths.append(comparison_path)
    inputs_before = _source_input_manifest(project, input_paths)
    provenance = {
        "offline_only": True,
        "environment_step_called": False,
        "training_performed": False,
        "rollout_executed": False,
        "historical_rollout_source_root": str(source),
        "current_recovery_module": str(Path(__file__).resolve()),
        "current_recovery_module_sha256": _sha256_file(Path(__file__).resolve()),
        "invocation": {
            "project_root": str(project), "source_root": str(source), "preparation_root": str(preparation), "output_root": str(output),
        },
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "preregistration": {"current_sha256": prereg_current_hash, "locked_sha256": prereg_lock_hash, "status": lock_status},
        "expected_episode_ids_sha256": _sha256_bytes(_canonical(paired["population"]).encode("utf-8")),
        "runtime_source_hashes": source_hash_checks,
        "baseline_manifest_checks": manifest_checks,
        "off_baseline_checks": baseline_checks,
        "comparison_checks": comparison_checks,
        "input_manifest_before": inputs_before,
    }
    # A final guard closes the read/validate window before staging output.
    inputs_after = _source_input_manifest(project, input_paths)
    if inputs_after != inputs_before:
        _fail("source artifacts changed during offline recovery")
    provenance["input_manifest_after"] = inputs_after
    evaluation = {
        "stage": "single_uav_pass_commit_diagnostic",
        "condition": "paired",
        "pass_commit_paired_itt_summary": _copy(paired),
        "aggregates": _copy(paired["aggregates"]),
        "cohorts": _copy(paired["cohorts"]),
        "provenance": _copy(provenance),
        "offline_only": True,
        "training_performed": False,
        "rollout_executed": False,
        "environment_step_called": False,
    }
    checks_rows = [
        {"check": "paired_identity", "status": "PASS", "detail": "exact preregistered IDs, seeds, scenario hashes and step identities"},
        {"check": "step_contiguity_and_counters", "status": "PASS", "detail": "per-episode rows and counters agree"},
        {"check": "off_baseline_fingerprint", "status": baseline_checks["status"], "detail": contract.get("fingerprint", "")},
        {"check": "recorded_comparison", "status": comparison_checks.get("status", "UNAVAILABLE"), "detail": comparison_checks.get("path", "")},
        {"check": "protected_inputs", "status": manifest_checks.get("status", "UNAVAILABLE"), "detail": manifest_checks.get("manifest", "")},
    ]
    paired_csv_rows = [
        {"episode_id": row["episode_id"], "off_passed_obstacle": row["off_passed_obstacle"], "on_passed_obstacle": row["on_passed_obstacle"], "off_post_bypass_recovery": row["off_post_bypass_recovery"], "on_post_bypass_recovery": row["on_post_bypass_recovery"], "off_collision": row["off_collision"], "on_collision": row["on_collision"], "off_minimum_clearance_m": row["off_minimum_clearance_m"], "on_minimum_clearance_m": row["on_minimum_clearance_m"], "off_safety_projection_steps": row["off_safety_projection_steps"], "on_safety_projection_steps": row["on_safety_projection_steps"], "off_boundary_projection_steps": row["off_boundary_projection_steps"], "on_boundary_projection_steps": row["on_boundary_projection_steps"], "off_pass_commit_active_steps": row["off_pass_commit_active_steps"], "on_pass_commit_active_steps": row["on_pass_commit_active_steps"]}
        for row in paired["paired_episode_outcomes"]
    ]
    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging_prefix = f".{output.name}."
    staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=str(output_parent)))
    staging_parent = output_parent.resolve()

    def _validate_staging_path() -> None:
        if staging.is_symlink() or not staging.is_dir():
            _fail("staging directory was replaced or is not a directory")
        if staging.parent.resolve() != staging_parent or not staging.name.startswith(staging_prefix):
            _fail("staging directory escaped the output parent")

    try:
        _validate_staging_path()
        _write_json(staging / "evaluation_summary.json", evaluation)
        _write_json(staging / "recovery_provenance.json", provenance)
        _write_csv(staging / "recovery_checks.csv", checks_rows)
        _write_csv(staging / "paired_summary.csv", paired_csv_rows)
        # Recheck source hashes immediately before the atomic publish.
        if _source_input_manifest(project, input_paths) != inputs_before:
            _fail("source artifacts changed before atomic publish")
        _validate_staging_path()
        if output.exists() or output.is_symlink() or output.parent.resolve() != staging_parent:
            _fail("output target appeared or moved before atomic publish")
        # os.rename maps to an atomic directory move and refuses a late
        # existing target on Windows, preserving the no-overwrite contract.
        os.rename(str(staging), str(output))
    except Exception:
        if staging.exists() and not staging.is_symlink():
            try:
                _validate_staging_path()
            except Exception:
                pass
            else:
                shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output_root": str(output), "evaluation_summary": evaluation, "recovery_provenance": provenance}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="offline pass-commit paired summary recovery")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--preparation-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--baseline-manifest", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = recover_pass_commit_summary(
        args.project_root,
        source_root=args.source_root,
        preparation_root=args.preparation_root,
        output_root=args.output_root,
        baseline_manifest=args.baseline_manifest,
    )
    print(json.dumps({"output_root": result["output_root"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
