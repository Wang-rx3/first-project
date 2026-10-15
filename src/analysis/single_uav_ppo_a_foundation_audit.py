"""Independent, read-only audit for the single-UAV PPO-A foundation.

The foundation runner owns simulation, policy actions, optimizer updates, and
checkpoint creation.  This module deliberately does none of those things: it
only reads JSON/CSV artifacts and computes file hashes.  In particular it does
not import the runner, call ``env.step``/``PPOTrainer.update``, or deserialize a
checkpoint.  A missing downstream phase is reported as ``pending`` so an
incomplete staged run cannot be mistaken for a completed experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_ppo_a_foundation_v1_seed2026")
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_ppo_a_foundation_v1_seed2026")
EXPECTED_MAX_TRANSITIONS = 16252
EXPECTED_NORMAL_TRAIN_CASES = 6
EXPECTED_NORMAL_VALIDATION_CASES = 8
EXPECTED_ROLLOUT_STEPS = 120
EXPECTED_UPDATES = 6
EXPECTED_OPTIMIZER_STEPS_PER_UPDATE = 8
EXPECTED_OPTIMIZER_STEPS = EXPECTED_UPDATES * EXPECTED_OPTIMIZER_STEPS_PER_UPDATE


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value: Any, default: int = -1) -> int:
    number = _number(value, float("nan"))
    return int(number) if math.isfinite(number) else default


def _payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def _vector(value: Any, size: int = 3) -> list[float] | None:
    parsed = _payload(value)
    if not isinstance(parsed, (list, tuple)):
        return None
    try:
        values = [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None
    return values if len(values) == size and all(math.isfinite(item) for item in values) else None


def _close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    a, b = _number(left), _number(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def _resolve(value: Any, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (project_root / path)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _canonical(value: Any) -> Any:
    """Convert nested JSON/tensor state to a deterministic comparison shape."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else str(value)
    if isinstance(value, (int, bool)) or value is None or isinstance(value, str):
        return value
    # Torch/numpy scalars expose item(); importing either library is unnecessary.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonical(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _canonical(tolist())
        except Exception:
            pass
    return str(value)


def _tensor_hash(tensor: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _Audit:
    def __init__(self, project_root: Path, log_root: Path, data_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.log_root = log_root.resolve()
        self.data_root = data_root.resolve()
        self.checks: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.failures: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.diagnostics: list[str] = []

    def check(self, name: str, passed: bool, details: Any = None) -> None:
        self.checks[name] = bool(passed)
        if details is not None:
            self.details[name] = details
        if not passed:
            self.failures.append({"check": name, "details": details})

    def mark_pending(self, name: str, details: Any) -> None:
        self.pending.append({"check": name, "details": details})

    def guarded(self, name: str, function: Any) -> Any:
        try:
            value = function()
            if value is not None:
                self.details[name] = value
            self.checks[name] = True
            return value
        except Exception as exc:  # Keep the report machine-readable.
            self.check(name, False, f"{type(exc).__name__}: {exc}")
            self.diagnostics.append(f"{name}: {type(exc).__name__}: {exc}")
            return None


def _discover_data_root(project_root: Path, log_root: Path) -> Path:
    candidate = project_root / DEFAULT_DATA_ROOT
    if candidate.is_dir():
        return candidate
    name = log_root.name
    candidate = project_root / "data" / "synthetic" / name
    if candidate.is_dir():
        return candidate
    return candidate


def _optional_json(audit: _Audit, relative: str, *, pending_name: str | None = None) -> Any | None:
    path = audit.log_root / relative
    if not path.is_file():
        audit.mark_pending(pending_name or relative, f"missing artifact: {path}")
        return None
    return _read_json(path)


def _optional_csv(audit: _Audit, relative: str, *, pending_name: str | None = None) -> list[dict[str, str]] | None:
    path = audit.log_root / relative
    if not path.is_file():
        audit.mark_pending(pending_name or relative, f"missing artifact: {path}")
        return None
    return _read_csv(path)


def _manifest(audit: _Audit) -> Mapping[str, Any] | None:
    path = audit.data_root / "run_specification.lock.json"
    if not path.is_file():
        audit.mark_pending("run_specification", f"missing artifact: {path}")
        return None
    value = _read_json(path)
    if not isinstance(value, Mapping):
        audit.check("run_specification", False, "run specification must be an object")
        return None
    audit.check("run_specification", True, {"schema_version": value.get("schema_version"), "plan_version": value.get("plan_version")})
    return value


def _scenario_ids(audit: _Audit) -> tuple[set[str], set[str]] | None:
    path = audit.data_root / "scenario_manifest.json"
    if not path.is_file():
        audit.mark_pending("scenario_manifest", f"missing artifact: {path}")
        return None
    manifest = _read_json(path)
    if not isinstance(manifest, Mapping):
        audit.check("scenario_manifest", False, "scenario manifest must be an object")
        return None
    train = {str(row.get("scenario_id")) for row in manifest.get("normal_train", []) if isinstance(row, Mapping)}
    validation = {str(row.get("scenario_id")) for row in manifest.get("validation_normal", []) if isinstance(row, Mapping)}
    passed = len(train) == EXPECTED_NORMAL_TRAIN_CASES and len(validation) == EXPECTED_NORMAL_VALIDATION_CASES
    audit.check("normal_scenario_cardinality", passed, {"train": sorted(train), "validation": sorted(validation)})
    return train, validation


def _check_feasibility_pair(audit: _Audit) -> None:
    legacy_manifest = _optional_json(audit, "legacy_feasibility_manifest.json", pending_name="legacy_feasibility")
    legacy_result = _optional_json(audit, "legacy_feasibility_result.json", pending_name="legacy_feasibility_result")
    semantics_manifest = _optional_json(audit, "new_semantics_feasibility_manifest.json", pending_name="new_semantics_feasibility")
    legacy_rows = _optional_csv(audit, "legacy_feasibility_episodes.csv", pending_name="legacy_feasibility_episodes")
    semantics_rows = _optional_csv(audit, "new_semantics_feasibility_episodes.csv", pending_name="new_semantics_feasibility_episodes")
    if any(value is None for value in (legacy_manifest, legacy_result, semantics_manifest, legacy_rows, semantics_rows)):
        return
    assert isinstance(legacy_manifest, Mapping) and isinstance(legacy_result, Mapping) and isinstance(semantics_manifest, Mapping)
    assert isinstance(legacy_rows, list) and isinstance(semantics_rows, list)
    basic = (
        legacy_result.get("passed_all") is True
        and semantics_manifest.get("passed_all") is True
        and _integer(legacy_manifest.get("case_count")) == 14
        and _integer(semantics_manifest.get("case_count")) == 14
        and _integer(legacy_manifest.get("actual_interaction_steps")) == 490
        and _integer(semantics_manifest.get("actual_interaction_steps")) == len(_optional_csv(audit, "new_semantics_feasibility_steps.csv") or [])
        and len(legacy_rows) == 14
        and len(semantics_rows) == 14
    )
    legacy_by_id = {str(row.get("scenario_id")): row for row in legacy_rows}
    semantics_by_id = {str(row.get("scenario_id")): row for row in semantics_rows}
    paired = set(legacy_by_id) == set(semantics_by_id) and set(legacy_by_id)
    if paired:
        for scenario_id in sorted(paired):
            left, right = legacy_by_id[scenario_id], semantics_by_id[scenario_id]
            for field in ("actual_goal_distance_m_final", "actual_speed_mps_final", "maximum_acceleration_mps2"):
                if not _close(left.get(field), right.get(field), 1e-8):
                    paired = False
                    break
            if not _truth(left.get("criterion_pass")) or not _truth(right.get("criterion_pass")):
                paired = False
                break
            if not _truth(right.get("public_trajectory_match")):
                paired = False
                break
    audit.check("physics_gate_14_pairs_consistent", bool(basic and paired), {
        "legacy_cases": len(legacy_rows), "semantics_cases": len(semantics_rows),
        "legacy_steps": _integer(legacy_manifest.get("actual_interaction_steps")),
        "semantics_steps": _integer(semantics_manifest.get("actual_interaction_steps")),
        "pair_ids_match": bool(paired),
    })


def _check_protected(audit: _Audit) -> None:
    path = audit.log_root / "protected_inputs_before.json"
    if not path.is_file():
        audit.mark_pending("protected_inputs_135", f"missing artifact: {path}")
        return
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        audit.check("protected_inputs_135", False, "protected input manifest must be an object")
        return
    mismatches: list[str] = []
    for raw_path, expected in payload.items():
        protected_path = _resolve(raw_path, audit.project_root)
        if not protected_path.is_file() or _sha256(protected_path) != str(expected):
            mismatches.append(str(protected_path))
    audit.check("protected_inputs_135", len(payload) == 135 and not mismatches, {
        "expected_count": len(payload), "mismatch_count": len(mismatches), "mismatches": mismatches[:12],
    })


def _iter_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def _walk_values(value: Any, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _walk_values(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child, key)
    else:
        yield key, value


def _check_b_pause_and_checkpoint_paths(audit: _Audit, spec: Mapping[str, Any] | None) -> None:
    status_path = audit.log_root / "branch_status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    branch_marker = status.get("B_status", status.get("B")) if isinstance(status, Mapping) else None
    status_ok = (
        not status or (
            str(branch_marker or "").lower() == "paused"
            and _integer(status.get("B_model_loads"), 0) == 0
            and _integer(status.get("B_training_runs"), 0) == 0
        )
    )
    allowed: set[Path] = set()
    if isinstance(spec, Mapping):
        for key in ("A_source", "historical_A_initializer"):
            record = spec.get(key, {})
            if isinstance(record, Mapping):
                value = record.get("checkpoint", record.get("path"))
                if value:
                    allowed.add(_resolve(value, audit.project_root).resolve())
    allowed.update({(audit.data_root / "G" / "normal_final.pt").resolve(), (audit.data_root / "P" / "normal_final.pt").resolve()})
    unknown: list[str] = []
    b_nonzero: list[str] = []
    for path in _iter_json_files(audit.log_root):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for key, value in _walk_values(payload):
            lowered = key.lower()
            if lowered in {"b_checkpoint_loads", "b_model_loads", "b_training_runs"} and _integer(value, 0) != 0:
                b_nonzero.append(f"{path.name}:{key}={value}")
            if lowered in {"checkpoint_load_paths", "checkpoint_load_path", "source_path", "checkpoint"}:
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if not isinstance(item, str) or not item:
                        continue
                    resolved = _resolve(item, audit.project_root).resolve()
                    if resolved not in allowed:
                        unknown.append(str(resolved))
    audit.check("B_paused_zero_model_loads", bool(status_ok and not b_nonzero), {"branch_status_present": bool(status), "nonzero_records": b_nonzero})
    audit.check("checkpoint_loads_are_A_or_new_GP", not unknown, {"unknown_paths": sorted(set(unknown)), "allowed_count": len(allowed)})


def _check_runtime_and_budget(audit: _Audit) -> None:
    components: dict[str, int] = {}
    for name, relative in (
        ("legacy", "legacy_feasibility_steps.csv"),
        ("semantics", "new_semantics_feasibility_steps.csv"),
        ("diagnostics", "diagnostics_trajectory_steps.csv"),
        ("fixtures", "action_reward_fixtures_steps.csv"),
    ):
        path = audit.log_root / relative
        if path.is_file():
            components[name] = len(_read_csv(path))
    for arm in ("G", "P"):
        path = audit.log_root / arm / "steps.csv"
        if path.is_file():
            components[f"training_{arm}"] = len(_read_csv(path))
    observed = sum(components.values())
    audit.check("research_transition_budget", observed <= EXPECTED_MAX_TRANSITIONS, {
        "observed_transitions": observed, "maximum_transitions": EXPECTED_MAX_TRANSITIONS, "components": components,
    })
    runtime_checks: dict[str, Any] = {}
    for phase, expected in (
        ("semantics", components.get("semantics")),
        ("diagnostics", None if "diagnostics" not in components else components.get("diagnostics", 0) + components.get("fixtures", 0)),
        ("train", sum(components.get(f"training_{arm}", 0) for arm in ("G", "P")) or None),
    ):
        path = audit.log_root / f"{phase}_runtime_audit.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        actual = _integer(payload.get("actual_env_transitions"), -1) if isinstance(payload, Mapping) else -1
        runtime_checks[phase] = {"reported": actual, "csv_expected": expected}
        if expected is not None and actual != expected:
            audit.check(f"{phase}_runtime_count_matches_csv", False, runtime_checks[phase])
    audit.details["runtime_counts"] = runtime_checks
    if not components:
        audit.mark_pending("runtime_and_budget", "no phase CSV artifacts exist yet")


def _allowed_final_checkpoint(audit: _Audit, arm: str) -> Path:
    return (audit.data_root / arm / "normal_final.pt").resolve()


def _load_final_checkpoint(path: Path) -> Any:
    """The sole checkpoint deserialization entry point; callers pass only G/P finals."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_audit(audit: _Audit, spec: Mapping[str, Any] | None) -> int:
    """Load exactly the two new G/P finals and verify state, provenance and deltas."""
    paths = {arm: _allowed_final_checkpoint(audit, arm) for arm in ("G", "P")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        audit.mark_pending("new_GP_checkpoint_audit", {"missing": missing})
        return 0
    shared_path: Path | None = None
    if isinstance(spec, Mapping):
        record = spec.get("shared_A_train_statistics", {})
        if isinstance(record, Mapping) and record.get("path"):
            shared_path = _resolve(record["path"], audit.project_root).resolve()
    shared = _read_json(shared_path) if shared_path and shared_path.is_file() else None
    if not isinstance(shared, Mapping):
        audit.mark_pending("shared_statistics_for_checkpoint_audit", f"missing shared statistics: {shared_path}")
        return 0
    report: dict[str, Any] = {}
    loaded = 0
    all_ok = True
    for arm, path in paths.items():
        # This is intentionally the only call site of torch.load in this module.
        payload = _load_final_checkpoint(path)
        loaded += 1
        if not isinstance(payload, Mapping):
            all_ok = False
            report[arm] = {"error": "checkpoint payload is not a mapping"}
            continue
        policy = payload.get("policy_state_dict")
        metadata_ok = (
            _integer(payload.get("online_steps")) == EXPECTED_ROLLOUT_STEPS
            and _integer(payload.get("new_ppo_updates")) == EXPECTED_UPDATES
            and str(payload.get("arm", arm)) == arm
            and _integer(payload.get("uav_count"), -1) == 1
            and _integer(payload.get("teacher_data_count"), 0) == 0
        )
        finite_names: list[str] = []
        nonfinite_names: list[str] = []
        if isinstance(policy, Mapping):
            for name, tensor in policy.items():
                is_finite = bool(getattr(tensor, "isfinite", lambda: False)().all().item()) if hasattr(tensor, "isfinite") else False
                (finite_names if is_finite else nonfinite_names).append(str(name))
        policy_finite = bool(isinstance(policy, Mapping) and finite_names and not nonfinite_names)

        optimizer = payload.get("optimizer_state_dict")
        optimizer_state = optimizer.get("state", {}) if isinstance(optimizer, Mapping) else {}
        state_steps: list[int] = []
        state_finite = True
        if isinstance(optimizer_state, Mapping):
            for value in optimizer_state.values():
                if not isinstance(value, Mapping):
                    state_finite = False
                    continue
                step = _number(value.get("step"), float("nan"))
                state_steps.append(int(step) if math.isfinite(step) else -1)
                for key in ("exp_avg", "exp_avg_sq"):
                    tensor = value.get(key)
                    if tensor is None or not hasattr(tensor, "isfinite") or not bool(tensor.isfinite().all().item()):
                        state_finite = False
        adam_ok = bool(state_steps and state_finite and all(step == EXPECTED_OPTIMIZER_STEPS for step in state_steps))

        init = payload.get("initialization", {})
        initial_hashes = init.get("parameter_hashes_after", {}) if isinstance(init, Mapping) else {}
        final_hashes = {str(name): _tensor_hash(tensor) for name, tensor in policy.items()} if isinstance(policy, Mapping) else {}
        groups = {
            "delta_mean": [name for name in final_hashes if name.startswith("delta_mean_head.")],
            "first_gat": [name for name in final_hashes if name.startswith("gat.layers.0.")],
            "deep_gat": [name for name in final_hashes if name.startswith("gat.layers.") and not name.startswith("gat.layers.0.")],
            "critic": [name for name in final_hashes if name.startswith("critic.")],
        }
        changed = {group: sum(1 for name in names if str(initial_hashes.get(name)) != final_hashes.get(name)) for group, names in groups.items()}
        deltas_ok = bool(all(names for names in groups.values()) and all(changed[group] > 0 for group in groups))

        normalization = payload.get("normalization_stats")
        central = payload.get("central_normalization_stats")
        # Shared statistics store node normalizer inside normalizers.node_features.
        expected_normalization = shared.get("observation_normalizer")
        expected_central = shared.get("central_normalizer")
        normalizers_ok = _canonical(normalization) == _canonical(expected_normalization) and _canonical(central) == _canonical(expected_central)
        arm_ok = bool(metadata_ok and policy_finite and adam_ok and deltas_ok and normalizers_ok)
        all_ok = all_ok and arm_ok
        report[arm] = {
            "path": str(path), "metadata_ok": metadata_ok, "policy_finite": policy_finite,
            "nonfinite_policy_parameters": nonfinite_names[:12], "adam_state_steps": sorted(set(state_steps)),
            "adam_active_parameter_count": len(state_steps), "adam_state_finite": state_finite,
            "normalizers_match_shared_statistics": normalizers_ok, "changed_parameter_counts": changed,
            "delta_groups": {group: names for group, names in groups.items()}, "arm_passed": arm_ok,
        }
    audit.check("new_GP_checkpoint_state_audit", all_ok, report)
    audit.details["checkpoint_load_policy"] = {"allowed": [str(path) for path in paths.values()], "actual_loads": loaded, "B_loaded": False}
    return loaded


def _check_action_log_join(audit: _Audit, arm: str, steps: list[dict[str, str]]) -> None:
    path = audit.log_root / arm / "sampled_execution.csv"
    if not path.is_file():
        audit.check(f"{arm}_sampled_execution_join", False, f"missing artifact: {path}")
        return
    sampled = _read_csv(path)
    left: dict[tuple[str, int], dict[str, str]] = {}
    duplicate = 0
    for row in sampled:
        key = (str(row.get("episode_id")), _integer(row.get("step")))
        if key in left:
            duplicate += 1
        left[key] = row
    right: dict[tuple[str, int], dict[str, str]] = {}
    for row in steps:
        key = (str(row.get("episode_id")), _integer(row.get("episode_step")) - 1)
        if key in right:
            duplicate += 1
        right[key] = row
    errors: list[str] = []
    if len(left) != len(right) or duplicate or set(left) != set(right):
        errors.append(f"key_mismatch sampled={len(left)} steps={len(right)} duplicate={duplicate}")
    for key in sorted(set(left) & set(right)):
        source, step = left[key], right[key]
        for field in ("old_logprob", "logprob_action"):
            if field == "logprob_action":
                if source.get(field) != step.get(field):
                    errors.append(f"{key}:logprob_action")
            elif not _close(source.get(field), step.get(field), 1e-6):
                errors.append(f"{key}:{field}")
        for source_names, step_name in ((["sampled_delta_v_x", "sampled_delta_v_y", "sampled_delta_v_z"], "sampled_delta_v"), (["executed_delta_v_x", "executed_delta_v_y", "executed_delta_v_z"], "executed_delta_v"), (["total_correction_x", "total_correction_y", "total_correction_z"], "total_correction_delta_v")):
            source_vector = [_number(source.get(name)) for name in source_names]
            step_vector = _vector(step.get(step_name))
            if step_vector is None or any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(source_vector, step_vector)):
                errors.append(f"{key}:{step_name}")
        if _truth(source.get("reason_safety_projection")) != _truth(step.get("safety_projected")):
            errors.append(f"{key}:safety")
        if _truth(source.get("reason_boundary_projection")) != _truth(step.get("boundary_projected")):
            errors.append(f"{key}:boundary")
    audit.check(f"{arm}_sampled_execution_join", not errors, {"sampled_rows": len(sampled), "step_rows": len(steps), "errors": errors[:20], "logger_step_is_zero_based": True})


def _episode_metrics_from_steps(rows: list[dict[str, str]], episodes: list[dict[str, str]]) -> tuple[dict[str, Any], list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("episode_id"))].append(row)
    errors: list[str] = []
    recomputed: list[dict[str, Any]] = []
    episode_by_id = {str(row.get("episode_id")): row for row in episodes}
    for episode_id, step_rows in grouped.items():
        step_rows.sort(key=lambda row: _integer(row.get("episode_step")))
        hold = 0
        bonus_count = 0
        for row in step_rows:
            goal = _number(row.get("strict_goal_distance_m_actual"), _number(row.get("goal_distance_m_actual")))
            speed = _number(row.get("strict_speed_mps_actual"), _number(row.get("speed_mps_actual")))
            qualifies = goal <= 1.0 and speed <= 0.5
            hold = hold + 1 if qualifies else 0
            if _integer(row.get("strict_hold_steps")) != hold:
                errors.append(f"{episode_id}:hold")
            bonus_count += int(_close(row.get("strict_success_bonus"), 5.0, 1e-7))
            if _truth(row.get("physical_bypass")):
                errors.append(f"{episode_id}:normal_physical_bypass")
        success = hold >= 5 or any(_truth(row.get("strict_success")) for row in step_rows)
        terminal = any(_truth(row.get("terminated")) for row in step_rows)
        collision = any(_truth(row.get("collision")) or _truth(row.get("physical_collision")) for row in step_rows)
        boundary_attempt = sum(_truth(row.get("boundary_projected")) or _truth(row.get("raw_action_boundary_danger")) for row in step_rows)
        boundary_projection = sum(_truth(row.get("boundary_projected")) for row in step_rows)
        safety_projection = sum(_truth(row.get("safety_projected")) for row in step_rows)
        physical_contact = sum(_truth(row.get("raw_action_contact")) for row in step_rows)
        raw_danger = sum(_truth(row.get("raw_action_physical_danger")) for row in step_rows)
        timeout = bool(step_rows and _truth(step_rows[-1].get("truncated")) and not terminal)
        expected = episode_by_id.get(episode_id)
        if expected is None:
            errors.append(f"{episode_id}:missing_episode_row")
            continue
        comparisons = {
            "steps": len(step_rows), "strict_success": success, "task_success": success,
            "terminated": terminal, "collision": collision, "boundary_attempt_steps": boundary_attempt,
            "boundary_projection_steps": boundary_projection, "safety_projection_steps": safety_projection,
            "physical_contact_steps": physical_contact, "raw_action_danger_steps": raw_danger,
            "timeout_failure": timeout,
        }
        for field, expected_value in comparisons.items():
            raw = expected.get(field)
            if field == "task_success":
                actual = _truth(raw)
            elif field in {"strict_success", "terminated", "collision", "timeout_failure"}:
                actual = _truth(raw)
            else:
                actual = _integer(raw)
            if actual != expected_value:
                errors.append(f"{episode_id}:{field} expected={expected_value} actual={actual}")
        if (success and bonus_count != 1) or (not success and bonus_count != 0):
            errors.append(f"{episode_id}:bonus_count={bonus_count}")
        recomputed.append({"episode_id": episode_id, "steps": len(step_rows), "success": success, "collision": collision, "timeout": timeout, "boundary_attempt_steps": boundary_attempt, "boundary_projection_steps": boundary_projection, "safety_projection_steps": safety_projection, "physical_contact_steps": physical_contact, "raw_action_danger_steps": raw_danger, "final_goal": _number(step_rows[-1].get("strict_goal_distance_m_actual"), _number(step_rows[-1].get("goal_distance_m_actual"))), "final_speed": _number(step_rows[-1].get("strict_speed_mps_actual"), _number(step_rows[-1].get("speed_mps_actual")))})
    return {"episodes": recomputed, "episode_count": len(recomputed)}, errors


def _check_episode_recomputation(audit: _Audit, arm: str, result: Mapping[str, Any] | None) -> None:
    steps_path, episodes_path = audit.log_root / arm / "steps.csv", audit.log_root / arm / "episodes.csv"
    if not steps_path.is_file() or not episodes_path.is_file():
        audit.mark_pending(f"{arm}_episode_recomputation", "steps.csv or episodes.csv missing")
        return
    steps, episodes = _read_csv(steps_path), _read_csv(episodes_path)
    all_errors: list[str] = []
    phase_details: dict[str, Any] = {}
    for phase in ("initial_train_eval", "initial_validation", "final_train_eval", "final_validation"):
        phase_steps = [row for row in steps if row.get("phase") == phase]
        phase_episodes = [row for row in episodes if row.get("phase") == phase]
        recomputed, errors = _episode_metrics_from_steps(phase_steps, phase_episodes)
        all_errors.extend(f"{phase}:{error}" for error in errors)
        phase_details[phase] = recomputed
        if isinstance(result, Mapping):
            result_phase = result.get({"initial_train_eval": "initial_train", "initial_validation": "initial_validation", "final_train_eval": "final_train", "final_validation": "final_validation"}.get(phase, phase), {})
            result_metrics = result_phase.get("metrics", {}).get("all", {}) if isinstance(result_phase, Mapping) else {}
            expected_count = len(phase_episodes)
            if _integer(result_phase.get("steps"), -1) != len(phase_steps) or _integer(result_metrics.get("episode_count"), -1) != expected_count or _integer(result_metrics.get("actual_steps"), -1) != len(phase_steps):
                all_errors.append(f"{phase}:result_metric_count_mismatch")
            # Compare the primary claims directly to the CSV-derived episode rows.
            success_count = sum(1 for row in recomputed["episodes"] if row["success"])
            collision_count = sum(1 for row in recomputed["episodes"] if row["collision"])
            timeout_count = sum(1 for row in recomputed["episodes"] if row["timeout"])
            for name, actual, key in (("success", success_count, "task_success_count"), ("collision", collision_count, "collision_episode_count"), ("timeout", timeout_count, "timeout_failure_count")):
                if _integer(result_metrics.get(key), -1) != actual:
                    all_errors.append(f"{phase}:result_{name}_mismatch")
    audit.check(f"{arm}_episode_metrics_recomputed_from_csv", not all_errors, {"errors": all_errors[:30], "phases": phase_details})


def _check_gae_recurrence(audit: _Audit, arm: str, updates: list[Mapping[str, Any]], online_rows: list[dict[str, str]]) -> None:
    gamma, gae_lambda = 0.99, 0.95
    errors: list[str] = []
    update_details: list[dict[str, Any]] = []
    for index, update in enumerate(updates, 1):
        records = update.get("per_record_audit") if isinstance(update, Mapping) else None
        if not isinstance(records, list) or len(records) != EXPECTED_ROLLOUT_STEPS:
            errors.append(f"update_{index:02d}:missing_120_per_record_audit")
            continue
        ordered = sorted(records, key=lambda row: _integer(row.get("record_index"), 10**9))
        if [_integer(row.get("record_index")) for row in ordered] != list(range(EXPECTED_ROLLOUT_STEPS)):
            errors.append(f"update_{index:02d}:record_index")
        raw_advantages = [0.0] * len(ordered)
        raw_returns = [0.0] * len(ordered)
        raw_deltas = [0.0] * len(ordered)
        next_gae = 0.0
        for position in range(len(ordered) - 1, -1, -1):
            row = ordered[position]
            transition_index = _integer(row.get("transition_index"), position)
            reward = _number(online_rows[(index - 1) * EXPECTED_ROLLOUT_STEPS + transition_index].get("learning_reward")) if 0 <= transition_index < EXPECTED_ROLLOUT_STEPS and len(online_rows) >= index * EXPECTED_ROLLOUT_STEPS else float("nan")
            old_value, next_value = (_number(row.get(name)) for name in ("old_value", "next_value"))
            if not all(math.isfinite(value) for value in (reward, old_value, next_value)):
                errors.append(f"update_{index:02d}:{position}:nonfinite")
                continue
            terminated, truncated, done = (_truth(row.get(name)) for name in ("terminated", "truncated", "done"))
            terminal_for_bootstrap = _truth(row.get("terminal_for_bootstrap"))
            expected_bootstrap = not terminal_for_bootstrap
            expected_continuation = not done
            if expected_bootstrap != (not _truth(row.get("terminal_for_bootstrap"))) or expected_continuation != _truth(row.get("gae_recursion_continues_to_previous_transition")):
                errors.append(f"update_{index:02d}:{position}:boundary_flags")
            # The runner's audit defines terminal_for_bootstrap from termination/truncation;
            # its stored next value is zero at a true terminal and retained at a time limit.
            bootstrap = 0.0 if terminal_for_bootstrap else 1.0
            continuation = 0.0 if done else 1.0
            delta = reward + gamma * next_value * bootstrap - old_value
            raw_deltas[position] = delta
            next_gae = delta + gamma * gae_lambda * continuation * next_gae
            raw_advantages[position] = next_gae
            raw_returns[position] = old_value + next_gae
            expected_ret = _number(row.get("expected_return_from_recurrence"))
            if not math.isclose(expected_ret, raw_returns[position], rel_tol=0.0, abs_tol=3e-5):
                errors.append(f"update_{index:02d}:{position}:return_recurrence")
            if not math.isclose(_number(row.get("return")), raw_returns[position], rel_tol=0.0, abs_tol=3e-5):
                errors.append(f"update_{index:02d}:{position}:return")
        # The stored advantage is standardized over the 120-record update.
        mean = sum(raw_advantages) / len(raw_advantages)
        variance = sum((value - mean) ** 2 for value in raw_advantages) / len(raw_advantages)
        std = math.sqrt(variance)
        if std <= 1e-12:
            errors.append(f"update_{index:02d}:zero_advantage_std")
        else:
            for position, row in enumerate(ordered):
                expected_normalized = (raw_advantages[position] - mean) / max(std, 1e-8)
                if not math.isclose(_number(row.get("expected_advantage_from_recurrence")), expected_normalized, rel_tol=0.0, abs_tol=3e-5):
                    errors.append(f"update_{index:02d}:{position}:normalized_expected_advantage")
                if not math.isclose(_number(row.get("advantage")), expected_normalized, rel_tol=0.0, abs_tol=3e-5):
                    errors.append(f"update_{index:02d}:{position}:normalized_advantage")
        update_details.append({"index": index, "records": len(records), "terminal_records": sum(_truth(row.get("done")) for row in ordered), "raw_advantage_mean": mean, "raw_advantage_std": std, "raw_delta_finite": all(math.isfinite(value) for value in raw_deltas), "raw_delta_min": min(raw_deltas), "raw_delta_max": max(raw_deltas), "done_reset_indices": [position for position, row in enumerate(ordered) if _truth(row.get("done"))]})
    audit.check(f"{arm}_gae_recurrence_independently_recomputed", not errors, {"gamma": gamma, "gae_lambda": gae_lambda, "updates": update_details, "errors": errors[:40]})


def _check_action_and_updates(audit: _Audit, arm: str, train_ids: set[str], validation_ids: set[str]) -> None:
    arm_root = audit.log_root / arm
    steps_path = arm_root / "steps.csv"
    if not steps_path.is_file():
        audit.mark_pending(f"{arm}_training", f"missing artifact: {steps_path}")
        return
    steps = _read_csv(steps_path)
    updates = []
    for index in range(1, EXPECTED_UPDATES + 1):
        path = arm_root / f"update_{index:02d}.json"
        if not path.is_file():
            audit.check(f"{arm}_update_{index:02d}", False, f"missing artifact: {path}")
            continue
        updates.append(_read_json(path))
    update_ok = len(updates) == EXPECTED_UPDATES
    update_details: list[dict[str, Any]] = []
    for index, update in enumerate(updates, 1):
        expected_optimizer_steps = index * EXPECTED_OPTIMIZER_STEPS_PER_UPDATE
        row_ok = isinstance(update, Mapping) and all(
            update.get(key) is True for key in ("on_policy_logprob_within_tolerance", "normalizers_unchanged", "all_weights_finite", "rollout_empty_after_update")
        ) and _integer(update.get("expected_samples")) == EXPECTED_ROLLOUT_STEPS and _integer(update.get("record_count")) == EXPECTED_ROLLOUT_STEPS and _integer(update.get("optimizer_steps")) == expected_optimizer_steps
        update_ok = update_ok and row_ok
        update_details.append({"index": index, "row_ok": row_ok, "expected_samples": update.get("expected_samples") if isinstance(update, Mapping) else None, "record_count": update.get("record_count") if isinstance(update, Mapping) else None, "optimizer_steps": update.get("optimizer_steps") if isinstance(update, Mapping) else None})
    online = [row for row in steps if row.get("phase") == "online"]
    phase_sets = {phase: {str(row.get("scenario_id")) for row in steps if row.get("phase") == phase} for phase in ("online", "initial_train_eval", "final_train_eval", "initial_validation", "final_validation")}
    scenario_ok = phase_sets["online"] == train_ids and phase_sets["initial_train_eval"] == train_ids and phase_sets["final_train_eval"] == train_ids and phase_sets["initial_validation"] == validation_ids and phase_sets["final_validation"] == validation_ids
    action_ok = True
    malformed_rows: list[int] = []
    for line, row in enumerate(steps, 2):
        if row.get("logprob_action") != "sampled" or _vector(row.get("sampled_delta_v")) is None or not math.isfinite(_number(row.get("old_logprob"))):
            action_ok = False
            malformed_rows.append(line)
    audit.check(f"{arm}_sampled_action_logprob", action_ok, {"rows": len(steps), "malformed_rows": malformed_rows[:12]})
    audit.check(f"{arm}_120_samples_x_6_updates_x_8_optimizer", bool(update_ok and len(online) == EXPECTED_UPDATES * EXPECTED_ROLLOUT_STEPS), {"online_rows": len(online), "updates": update_details})
    audit.check(f"{arm}_shared_six_train_fresh_eight_validation", bool(scenario_ok), {"phase_sets": {key: sorted(value) for key, value in phase_sets.items()}})

    result_path = arm_root / "normal_result.json"
    result: Mapping[str, Any] | None = None
    config_path = audit.data_root / arm / "effective_training_config.json"
    if result_path.is_file():
        result = _read_json(result_path)
        deltas = result.get("parameter_deltas", {}) if isinstance(result, Mapping) else {}
        changed = all(_number(deltas.get(group, {}).get("l2"), 0.0) > 0.0 for group in ("total_actor", "critic")) and any(_number(deltas.get(group, {}).get("l2"), 0.0) > 0.0 for group in ("first_gat", "deep_gat"))
        audit.check(f"{arm}_parameters_changed", bool(changed), {"parameter_deltas": deltas})
        metrics_train = result.get("final_train", {}).get("metrics", {}) if isinstance(result, Mapping) else {}
        metrics_val = result.get("final_validation", {}).get("metrics", {}) if isinstance(result, Mapping) else {}
        train_all = metrics_train.get("all", metrics_train) if isinstance(metrics_train, Mapping) else {}
        val_all = metrics_val.get("all", metrics_val) if isinstance(metrics_val, Mapping) else {}
        strict_ok = (_integer(train_all.get("episode_count")) == EXPECTED_NORMAL_TRAIN_CASES and _integer(val_all.get("episode_count")) == EXPECTED_NORMAL_VALIDATION_CASES and _integer(train_all.get("collision_episode_count"), 0) == 0 and _integer(val_all.get("collision_episode_count"), 0) == 0)
        audit.check(f"{arm}_normal_strict_metrics", strict_ok, {"train": train_all, "validation": val_all})
    else:
        audit.mark_pending(f"{arm}_normal_result", f"missing artifact: {result_path}")
    if config_path.is_file():
        config = _read_json(config_path)
        provenance = config.get("statistics_provenance", {}) if isinstance(config, Mapping) else {}
        audit.check(f"{arm}_normalizers_frozen", bool(isinstance(provenance, Mapping) and provenance.get("statistics_frozen") is True and provenance.get("refit_performed") is False), provenance)
    else:
        audit.mark_pending(f"{arm}_effective_config", f"missing artifact: {config_path}")
    _check_action_log_join(audit, arm, steps)
    _check_episode_recomputation(audit, arm, result)
    _check_gae_recurrence(audit, arm, [update for update in updates if isinstance(update, Mapping)], online)


def _check_conditional_gate(audit: _Audit) -> None:
    report_path = audit.log_root / "normal_training_result.json"
    if not report_path.is_file():
        audit.mark_pending("conditional_gate", f"missing artifact: {report_path}")
        return
    report = _read_json(report_path)
    gate = bool(report.get("normal_release_gate_passed")) if isinstance(report, Mapping) else False
    children = [path.name for path in audit.log_root.iterdir() if path.name.lower() in {"single", "full", "conditional_single", "conditional_full"} or "single_stage" in path.name.lower() or "full_stage" in path.name.lower()]
    explicit = {
        "single_stage_executed": bool(report.get("single_stage_executed")) if isinstance(report, Mapping) else False,
        "full_stage_executed": bool(report.get("full_stage_executed")) if isinstance(report, Mapping) else False,
    }
    non_normal_stages: dict[str, list[str]] = {}
    for arm in ("G", "P"):
        path = audit.log_root / arm / "steps.csv"
        if path.is_file():
            stages = sorted({str(row.get("stage")) for row in _read_csv(path)})
            non_normal_stages[arm] = [stage for stage in stages if stage != "normal"]
    blocked = gate or (not children and not any(explicit.values()) and not any(non_normal_stages.values()))
    audit.check("conditional_gate_blocks_single_full_when_failed", blocked, {"normal_release_gate_passed": gate, "conditional_artifacts": children, "explicit": explicit, "non_normal_stages": non_normal_stages})
    audit.check("conditional_gate_report_explicit_single_full_false", bool(gate or (explicit["single_stage_executed"] is False and explicit["full_stage_executed"] is False)), explicit)


def audit_artifacts(log_root: str | Path = DEFAULT_LOG_ROOT, *, output: str | Path | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
    """Read and audit currently existing A-foundation artifacts."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    log = Path(log_root)
    if not log.is_absolute():
        log = root / log
    data = _discover_data_root(root, log)
    audit = _Audit(root, log, data)
    audit.check("artifact_roots_exist", audit.log_root.is_dir() and audit.data_root.is_dir(), {"log_root": str(audit.log_root), "data_root": str(audit.data_root)})
    spec = _manifest(audit) if audit.data_root.is_dir() else None
    ids = _scenario_ids(audit) if audit.data_root.is_dir() else None
    _check_feasibility_pair(audit)
    _check_protected(audit)
    _check_b_pause_and_checkpoint_paths(audit, spec)
    _check_runtime_and_budget(audit)
    checkpoint_loads = 0
    if ids is not None:
        train_ids, validation_ids = ids
        for arm in ("G", "P"):
            _check_action_and_updates(audit, arm, train_ids, validation_ids)
    else:
        for arm in ("G", "P"):
            audit.mark_pending(f"{arm}_training", "scenario manifest unavailable")
    checkpoint_loads = audit.guarded("checkpoint_audit_execution", lambda: _checkpoint_audit(audit, spec)) or 0
    _check_conditional_gate(audit)
    has_failures = bool(audit.failures)
    status = "failed" if has_failures else ("pending" if audit.pending else "passed")
    report = {
        "schema_version": "single_uav_ppo_a_foundation_independent_audit_v1",
        "status": status,
        "passed": status == "passed",
        "project_root": str(audit.project_root),
        "data_root": str(audit.data_root),
        "log_root": str(audit.log_root),
        "checks": dict(audit.checks),
        "details": dict(audit.details),
        "failures": list(audit.failures),
        "pending": list(audit.pending),
        "diagnostics": list(audit.diagnostics),
        "read_only": True,
        "env_steps_executed": 0,
        "ppo_updates_executed": 0,
        "checkpoint_loads_executed": int(checkpoint_loads),
    }
    if output is not None:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Audits are immutable evidence artifacts: never silently overwrite a prior report.
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        report["output"] = str(output_path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit_artifacts(args.log_root, output=args.output), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = ["audit_artifacts", "DEFAULT_LOG_ROOT", "_check_action_and_updates", "_check_conditional_gate", "_check_protected", "_sha256"]


if __name__ == "__main__":
    main()
