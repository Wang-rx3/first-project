"""Independent, offline audit for the normalization-transfer PPO artifact set.

The transfer runner owns all simulation and optimization.  This module only
reads the resulting JSON/CSV/NPZ files (and, when present, checkpoint files)
and recomputes inexpensive consistency checks.  It never imports the runner,
creates an environment, calls ``step`` or performs an optimizer update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_ppo_a_norm_transfer_v2_seed2026_20261011")
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_ppo_a_norm_transfer_v2_seed2026_20261011")
EXPECTED_ARMS = ("C", "N", "G", "R")
EXPECTED_NODE_DIM = 42
EXPECTED_CENTRAL_DIM = 174
EXPECTED_CALIBRATION_SCENES = 12
EXPECTED_TRAIN_SCENES = 6
EXPECTED_VALIDATION_SCENES = 8
EXPECTED_CHALLENGE_SCENES = 4
EXPECTED_HORIZON = 120
EXPECTED_ONLINE_STEPS = 720
EXPECTED_UPDATES = 6
EXPECTED_ADAM_STEPS = 48
EXPECTED_CALIBRATION_STEPS = 2_892
EXPECTED_TRAIN_RUNTIME_STEPS = 20_160
EXPECTED_TOTAL_STEPS = 23_052
EXPECTED_CALIBRATION_BANK_SHA256 = "54043ea2f2abd8d25b0bd4cc2af1eccc604049118d00c3a63f581954519dc287"
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95
MOTION_DT_SECONDS = 0.2
MOTION_LOWER_BOUND_M = (0.0, 0.0, 2.0)
MOTION_UPPER_BOUND_M = (200.0, 200.0, 60.0)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def parse_vector(value: Any, size: int | None = 3) -> list[float] | None:
    parsed = _payload(value)
    if not isinstance(parsed, (list, tuple)):
        return None
    try:
        result = [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None
    if size is not None and len(result) != size:
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _vector(value: Any, size: int, *, allow_empty: bool = False) -> list[float] | None:
    result = parse_vector(value, size)
    if result is not None:
        return result
    if allow_empty and (_payload(value) in (None, [], "")):
        return []
    return None


def _close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    a, b = _number(left), _number(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _canonical(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonical(item())
        except Exception:
            pass
    return str(value)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class _Audit:
    """Small report collector shared by the public audit and pure tests."""

    def __init__(self, project_root: Path, log_root: Path, data_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.log_root = Path(log_root).resolve()
        self.data_root = Path(data_root).resolve()
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
            self.checks[name] = True
            if value is not None:
                self.details[name] = value
            return value
        except Exception as exc:  # Machine-readable and audit should continue.
            self.check(name, False, f"{type(exc).__name__}: {exc}")
            self.diagnostics.append(f"{name}: {type(exc).__name__}: {exc}")
            return None


def _discover_data_root(project_root: Path, log_root: Path) -> Path:
    candidate = project_root / DEFAULT_DATA_ROOT
    if candidate.is_dir():
        return candidate
    sibling = project_root / "data" / "synthetic" / log_root.name
    return sibling


def _optional_json(audit: _Audit, relative: str, *, pending_name: str | None = None) -> Any | None:
    path = audit.log_root / relative
    if not path.is_file():
        audit.mark_pending(pending_name or relative, f"missing artifact: {path}")
        return None
    try:
        return _read_json(path)
    except Exception as exc:
        audit.check(pending_name or relative, False, f"{type(exc).__name__}: {exc}")
        return None


def _optional_csv(audit: _Audit, relative: str, *, pending_name: str | None = None) -> list[dict[str, str]] | None:
    path = audit.log_root / relative
    if not path.is_file():
        audit.mark_pending(pending_name or relative, f"missing artifact: {path}")
        return None
    try:
        return _read_csv(path)
    except Exception as exc:
        audit.check(pending_name or relative, False, f"{type(exc).__name__}: {exc}")
        return None


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_strings(child)
    elif isinstance(value, str):
        yield value


def _check_spec(audit: _Audit) -> Mapping[str, Any] | None:
    path = audit.data_root / "run_specification.lock.json"
    if not path.is_file():
        audit.mark_pending("run_specification", f"missing artifact: {path}")
        return None
    spec = _read_json(path)
    if not isinstance(spec, Mapping):
        audit.check("run_specification", False, "run specification must be an object")
        return None
    expected = {
        "schema_version": "single_uav_ppo_a_norm_transfer_v2",
        "plan_version": 3,
        "calibration_scenarios": EXPECTED_CALIBRATION_SCENES,
        "calibration_observations": 2904,
        "calibration_env_steps": 2880,
        "online_steps_per_condition": EXPECTED_ONLINE_STEPS,
        "updates_per_condition": EXPECTED_UPDATES,
        "optimizer_steps_per_condition": EXPECTED_ADAM_STEPS,
        "maximum_research_env_steps": EXPECTED_TOTAL_STEPS,
    }
    mismatches = {key: {"actual": spec.get(key), "expected": value} for key, value in expected.items() if spec.get(key) != value}
    cond = tuple(str(item) for item in spec.get("conditions", []))
    if cond != EXPECTED_ARMS:
        mismatches["conditions"] = {"actual": cond, "expected": EXPECTED_ARMS}
    gate = spec.get("gate", {})
    gate_ok = isinstance(gate, Mapping) and all(spec.get("gate", {}).get(k) == v for k, v in {
        "train_success_min": 4, "validation_success_min": 5, "validation_median_final_distance_max_m": 2.0,
        "challenge_success_min": 3, "stationary_success_min": 1, "fast_success_min": 1,
        "collision_and_contact_max": 0,
    }.items())
    if not gate_ok:
        mismatches["gate"] = gate
    audit.check("run_specification_contract", not mismatches, {"mismatches": mismatches, "schema_version": spec.get("schema_version"), "plan_version": spec.get("plan_version")})
    audit.check("branch_B_paused_and_no_dynamics_change", spec.get("B_status") == "paused" and spec.get("reward_or_dynamics_changes") is False and spec.get("no_obstacle_curriculum") is True and spec.get("no_multiseed") is True, {key: spec.get(key) for key in ("B_status", "reward_or_dynamics_changes", "no_obstacle_curriculum", "no_multiseed")})
    return spec


def _split_entries(manifest: Mapping[str, Any], names: Sequence[str]) -> list[Mapping[str, Any]]:
    for name in names:
        value = manifest.get(name)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("scenario_id", row.get("id")), row.get("noise_seed", row.get("seed")), row.get("geometry_sha256", row.get("geometry_hash")))


def _check_splits(audit: _Audit) -> tuple[dict[str, list[Mapping[str, Any]]], Mapping[str, Any] | None]:
    path = audit.data_root / "scenario_manifest.json"
    if not path.is_file():
        audit.mark_pending("scenario_manifest", f"missing artifact: {path}")
        return {}, None
    manifest = _read_json(path)
    if not isinstance(manifest, Mapping):
        audit.check("scenario_manifest", False, "scenario manifest must be an object")
        return {}, None
    splits = {
        "calibration": _split_entries(manifest, ("calibration", "calibration_train")),
        "train": _split_entries(manifest, ("train", "normal_train")),
        "validation": _split_entries(manifest, ("validation", "validation_normal", "val")),
        "challenge": _split_entries(manifest, ("challenge", "validation_challenge")),
    }
    expected = {"calibration": EXPECTED_CALIBRATION_SCENES, "train": EXPECTED_TRAIN_SCENES, "validation": EXPECTED_VALIDATION_SCENES, "challenge": EXPECTED_CHALLENGE_SCENES}
    cardinality = {name: len(rows) for name, rows in splits.items()}
    identities: dict[str, set[tuple[Any, Any, Any]]] = {name: {_identity(row) for row in rows} for name, rows in splits.items()}
    overlap = {f"{left}:{right}": sorted(identities[left] & identities[right], key=str) for left in identities for right in identities if left < right and identities[left] & identities[right]}
    missing_identity = {name: sum(any(item is None for item in _identity(row)) for row in rows) for name, rows in splits.items()}
    audit.check("scenario_manifest_contract", cardinality == expected and not overlap and not any(missing_identity.values()), {"cardinality": cardinality, "expected": expected, "overlap": overlap, "missing_identity_fields": missing_identity})
    return splits, manifest


def _check_protected(audit: _Audit) -> None:
    path = audit.log_root / "protected_inputs_before.json"
    if not path.is_file():
        audit.mark_pending("protected_inputs_208", f"missing artifact: {path}")
        return
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        audit.check("protected_inputs_208", False, "protected manifest must be an object")
        return
    mismatches: list[str] = []
    for raw_path, expected in payload.items():
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = audit.project_root / candidate
        if not candidate.is_file() or _sha256(candidate) != str(expected):
            mismatches.append(str(candidate))
    audit.check("protected_inputs_208", len(payload) == 208 and not mismatches, {"manifest_count": len(payload), "mismatch_count": len(mismatches), "mismatches": mismatches[:20]})


def _check_calibration_bank(audit: _Audit) -> None:
    path = audit.data_root / "calibration_bank.npz"
    if not path.is_file():
        audit.mark_pending("calibration_bank", f"missing artifact: {path}")
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {str(key): np.asarray(archive[key]) for key in archive.files}
        nodes = next((arrays[key] for key in ("nodes", "node_features", "raw_nodes") if key in arrays), None)
        central = next((arrays[key] for key in ("central", "central_vectors", "central_features") if key in arrays), None)
        node_shape = None if nodes is None else tuple(nodes.reshape(-1, EXPECTED_NODE_DIM).shape)
        central_shape = None if central is None else tuple(central.reshape(-1, EXPECTED_CENTRAL_DIM).shape)
        required = {"node_features", "central_features", "scenario_id", "step", "arm", "split"}
        passed = nodes is not None and central is not None and node_shape == (2904, EXPECTED_NODE_DIM) and central_shape == (2904, EXPECTED_CENTRAL_DIM) and required.issubset(arrays)
        audit.check("calibration_bank_2904x42_174_no_pickle", bool(passed), {"arrays": sorted(arrays), "required_keys": sorted(required), "nodes_shape": node_shape, "central_shape": central_shape})
    except Exception as exc:
        audit.check("calibration_bank_2904x42_174_no_pickle", False, f"{type(exc).__name__}: {exc}")


def _check_calibration_chain(audit: _Audit) -> None:
    """Validate the bank/completion evidence used to fit the frozen stats.

    The bank intentionally stores 121 observations per 120-step trajectory:
    the reset observation at step 0 and the post-step observation at step
    120.  This is the source of 24*121=2904, and is stronger evidence than a
    redundant ``24 x 121`` label in the statistics JSON.
    """
    bank_path = audit.data_root / "calibration_bank.npz"
    completion_path = audit.log_root / "calibration_completion.json"
    if not bank_path.is_file() or not completion_path.is_file():
        if not bank_path.is_file():
            audit.mark_pending("calibration_bank_metadata", f"missing artifact: {bank_path}")
        if not completion_path.is_file():
            audit.mark_pending("calibration_completion", f"missing artifact: {completion_path}")
        return
    errors: list[str] = []
    bank_hash = _sha256(bank_path)
    if bank_hash != EXPECTED_CALIBRATION_BANK_SHA256:
        errors.append(f"bank SHA256 {bank_hash} != locked {EXPECTED_CALIBRATION_BANK_SHA256}")
    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            arrays = {str(key): np.asarray(archive[key]) for key in archive.files}
        required = ("episode_id", "scenario_id", "arm", "source", "split", "kind", "noise_seed", "step")
        if any(name not in arrays for name in required):
            errors.append("calibration bank metadata keys are incomplete")
        lengths = {name: len(arrays[name].reshape(-1)) for name in required if name in arrays}
        if set(lengths.values()) != {2904}:
            errors.append(f"metadata array lengths are not 2904: {lengths}")
        if not errors or all(name in arrays for name in required):
            metadata_rows = []
            for index in range(2904):
                metadata_rows.append({name: arrays[name].reshape(-1)[index].item() if hasattr(arrays[name].reshape(-1)[index], "item") else arrays[name].reshape(-1)[index] for name in required})
            groups: dict[tuple[str, str], list[int]] = {}
            for row in metadata_rows:
                if str(row["split"]) != "calibration" or str(row["kind"]) != "normal" or str(row["arm"]) not in {"P", "G"} or str(row["source"]) not in {"P", "G"}:
                    errors.append("calibration bank contains non-calibration/non-normal/non-PG metadata")
                    break
                groups.setdefault((str(row["arm"]), str(row["episode_id"])), []).append(int(row["step"]))
            if len(groups) != 24 or sum(len(value) for value in groups.values()) != 2904:
                errors.append(f"expected 24 calibration episodes, found {len(groups)}")
            if {arm: sum(1 for key in groups if key[0] == arm) for arm in ("P", "G")} != {"P": 12, "G": 12}:
                errors.append("expected 12 episodes for each frozen arm P/G")
            for key, steps in groups.items():
                if sorted(steps) != list(range(121)) or len(set(steps)) != 121:
                    errors.append(f"{key} does not contain each step 0..120 exactly once")
                    break
    except Exception as exc:
        errors.append(f"bank metadata read failed: {type(exc).__name__}: {exc}")
    try:
        completion = _read_json(completion_path)
        if not isinstance(completion, Mapping):
            errors.append("calibration completion must be an object")
        else:
            for key, expected in (("passed", True), ("all_complete", True), ("teacher_data_count", 0), ("optimizer_steps", 0), ("rollout_length", 0), ("env_steps", 2880), ("observations", 2904), ("episodes", 24)):
                if completion.get(key) != expected:
                    errors.append(f"completion {key}={completion.get(key)!r}, expected {expected!r}")
            details = completion.get("episodes_detail", [])
            if not isinstance(details, list) or len(details) != 24:
                errors.append("completion episodes_detail must contain 24 rows")
            else:
                for row in details:
                    if not isinstance(row, Mapping) or row.get("steps") != 120 or row.get("observations") != 121 or row.get("truncated_at_120") is not True:
                        errors.append("completion episode is not 120 steps + 121 observations")
                        break
                if {str(row.get("arm")) for row in details if isinstance(row, Mapping)} != {"P", "G"}:
                    errors.append("completion episode arms are not exactly P/G")
            completion_bank = completion.get("bank_path")
            if completion_bank and Path(str(completion_bank)).resolve() != bank_path.resolve():
                errors.append("completion bank_path does not resolve to calibration_bank.npz")
    except Exception as exc:
        errors.append(f"completion read failed: {type(exc).__name__}: {exc}")
    audit.check("calibration_bank_completion_chain", not errors, {"bank_sha256": bank_hash, "expected_bank_sha256": EXPECTED_CALIBRATION_BANK_SHA256, "errors": errors})


def _find_mapping(root: Any, names: Sequence[str]) -> Mapping[str, Any] | None:
    if isinstance(root, Mapping):
        for name in names:
            candidate = root.get(name)
            if isinstance(candidate, Mapping):
                return candidate
        for value in root.values():
            found = _find_mapping(value, names)
            if found is not None:
                return found
    return None


def _check_statistics(audit: _Audit) -> None:
    path = audit.data_root / "full_train_statistics.json"
    if not path.is_file():
        audit.mark_pending("full_train_statistics", f"missing artifact: {path}")
        return
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        audit.check("full_train_statistics", False, "statistics must be an object")
        return
    node = _find_mapping(payload, ("node_features",))
    if node is None and isinstance(payload.get("observation_normalizer"), Mapping):
        state = payload["observation_normalizer"]
        node = _find_mapping(state, ("node_features",))
    central = payload.get("central_normalizer", payload.get("central_statistics"))
    if isinstance(central, Mapping) and isinstance(central.get("state_dict"), Mapping):
        central = central["state_dict"]
    errors: list[str] = []
    mean = scale = count = None
    if not isinstance(node, Mapping):
        errors.append("missing node normalizer")
    else:
        try:
            mean, scale = np.asarray(node.get("mean"), dtype=float).reshape(-1), np.asarray(node.get("scale"), dtype=float).reshape(-1)
            count = np.asarray(node.get("count", np.full(mean.shape, 2904)), dtype=float).reshape(-1)
            if mean.size != 42 or scale.size != 42 or count.size != 42:
                errors.append("node dimension/count must be 42")
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(count).all() or np.any(scale <= 0) or np.any(count <= 0):
                errors.append("node statistics non-finite/non-positive")
            if mean.size == 42 and (not np.allclose(mean[30:33], 0.0, atol=1e-10) or not np.allclose(mean[3:6], 0.0, atol=1e-10)):
                errors.append("goal/velocity means are not zero")
            if scale.size == 42 and (np.any(scale[30:33] < 2.4) or np.any(scale[3:6] < 6.0)):
                errors.append("goal/velocity scale floors violated")
            clip = node.get("clip", payload.get("clip", 10.0))
            if not _close(clip, 10.0, 1e-8):
                errors.append("clip is not 10")
        except (TypeError, ValueError):
            errors.append("node mean/scale/count malformed")
    if not isinstance(central, Mapping):
        errors.append("missing central normalizer")
    else:
        try:
            cmean = np.asarray(central.get("mean"), dtype=float).reshape(-1)
            cscale = np.asarray(central.get("scale"), dtype=float).reshape(-1)
            if cmean.size != 174 or cscale.size != 174 or not np.isfinite(cmean).all() or not np.isfinite(cscale).all() or np.any(cscale <= 0):
                errors.append("central normalizer must be finite 174-vector")
        except (TypeError, ValueError):
            errors.append("central normalizer malformed")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    fit_split = str(provenance.get("fit_split", payload.get("fit_split", ""))).lower()
    if fit_split not in {"train", "calibration_train_only", "calibration"}:
        errors.append("fit split is not calibration train only")
    if provenance.get("statistics_frozen") is False or payload.get("frozen_during_ppo_and_validation") is False:
        errors.append("statistics not frozen")
    # The 24x121 fit contract is independently established from the bank and
    # completion artifact by _check_calibration_chain.  Here only verify that
    # the statistics artifact binds itself to that exact bank hash when the
    # provenance field is present (the fit function stores it at top level).
    bank_path = audit.data_root / "calibration_bank.npz"
    expected_bank_hash = _sha256(bank_path) if bank_path.is_file() else None
    source_bank_hash = payload.get("source_rawbank_sha256", payload.get("source_bank_sha256"))
    if expected_bank_hash is not None and source_bank_hash != expected_bank_hash:
        errors.append("statistics source_rawbank_sha256 is not the calibration bank SHA256")
    if payload.get("count") not in (None, 2904) or payload.get("node_count") not in (None, 2904) or payload.get("central_count") not in (None, 2904):
        errors.append("statistics count fields are not 2904")
    audit.check("full_statistics_frozen_train_only", not errors, {"errors": errors, "node_count_values": None if count is None else sorted(set(count.astype(int).tolist()))[:6], "artifact_sha256": _sha256(path), "source_rawbank_sha256": source_bank_hash, "expected_bank_sha256": expected_bank_hash, "provenance": dict(provenance)})


def _check_boundary_and_calibration_logs(audit: _Audit) -> None:
    boundary = _optional_json(audit, "boundary_probe_result.json", pending_name="boundary_probes")
    calibration = _optional_json(audit, "calibration_phase_result.json", pending_name="calibration_phase")
    rows = _optional_csv(audit, "boundary_probes.csv", pending_name="boundary_probe_rows")
    if boundary is not None:
        audit.check("twelve_boundary_probes", _integer(boundary.get("env_steps")) == 12 and boundary.get("passed") is True and boundary.get("reward_or_dynamics_modified") is False, boundary)
    if rows is not None:
        audit.check("boundary_probe_csv_count", len(rows) == 12, {"rows": len(rows)})
    if calibration is not None:
        payload = calibration.get("calibration", calibration) if isinstance(calibration, Mapping) else {}
        values = payload if isinstance(payload, Mapping) else {}
        steps = values.get("env_steps", values.get("calibration_env_steps", values.get("steps")))
        observations = values.get("observations", values.get("calibration_observations"))
        audit.check("calibration_2880_steps_2904_observations", _integer(steps) == 2880 and (_integer(observations) in {-1, 2904}), {"steps": steps, "observations": observations})


def _actual_train_steps_from_logs(audit: _Audit) -> tuple[int, dict[str, int], list[str]]:
    """Count the actual emitted train/eval rows, without assuming full episodes.

    ``steps.csv`` contains one row per environment transition for initial and
    final evaluation, online collection, and challenge evaluation.  Evaluation
    episodes may terminate early, so this row count is the independent runtime
    denominator for the train phase.
    """
    counts: dict[str, int] = {}
    missing: list[str] = []
    total = 0
    for arm in EXPECTED_ARMS:
        path = audit.log_root / arm / "steps.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            count = len(_read_csv(path))
        except Exception as exc:
            missing.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        counts[arm] = count
        total += count
    return total, counts, missing


def _phase_runtime(audit: _Audit, name: str, expected: int | None) -> None:
    payload = _optional_json(audit, f"{name}_runtime_audit.json", pending_name=f"{name}_runtime_audit")
    if payload is None:
        return
    if not isinstance(payload, Mapping):
        audit.check(f"{name}_runtime_budget", False, "runtime audit must be object")
        return
    loads = payload.get("checkpoint_loads", [])
    bloads = payload.get("B_loads", 0)
    row_counts: dict[str, int] = {}
    missing_rows: list[str] = []
    if expected is None and name == "train":
        expected, row_counts, missing_rows = _actual_train_steps_from_logs(audit)
        if missing_rows:
            audit.mark_pending("train_runtime_rows", {"missing": missing_rows})
    actual = _integer(payload.get("env_steps"))
    if expected is None:
        env_ok = False
    elif name == "train":
        env_ok = not missing_rows and actual == expected and 0 <= actual <= EXPECTED_TRAIN_RUNTIME_STEPS
    else:
        env_ok = actual == expected
    audit.check(f"{name}_runtime_env_steps", env_ok, {"actual": payload.get("env_steps"), "expected": expected, "row_counts": row_counts, "missing_rows": missing_rows})
    audit.check(f"{name}_runtime_no_B_loads", _integer(bloads, -1) == 0 and isinstance(loads, list), {"B_loads": bloads, "checkpoint_loads": loads})
    bad = [value for value in loads if re.search(r"(?:^|[\\/])B(?:[_.\\/]|$)", str(value), re.IGNORECASE)] if isinstance(loads, list) else ["checkpoint_loads_not_list"]
    audit.check(f"{name}_runtime_checkpoint_whitelist", not bad, {"unexpected_B_paths": bad})


def _phase_names(split: str, initial: bool) -> str:
    return ("initial_" if initial else "final_") + split


def recompute_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute gate metrics from episode rows only."""
    rows = list(episodes)
    def summarize(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        actual_steps = sum(max(0, _integer(row.get("steps"), 0)) for row in selected)
        successes = sum(_truth(row.get("task_success", row.get("strict_success"))) for row in selected)
        collisions = sum(_truth(row.get("collision")) for row in selected)
        contacts = sum(_integer(row.get("physical_contact_steps"), 0) for row in selected)
        attempts = sum(_integer(row.get("boundary_attempt_steps"), 0) for row in selected)
        projections = sum(_integer(row.get("boundary_projection_steps"), 0) for row in selected)
        distances = [_number(row.get("final_actual_goal_distance_m"), float("nan")) for row in selected]
        distances = [value for value in distances if math.isfinite(value)]
        return {"episode_count": len(selected), "actual_steps": actual_steps, "task_success_count": successes,
                "success_rate": successes / len(selected) if selected else 0.0, "collision_episode_count": collisions,
                "physical_contact_steps": contacts, "boundary_attempt_steps": attempts, "boundary_projection_steps": projections,
                "median_final_distance_m": float(np.median(distances)) if distances else float("nan"),
                "boundary_attempt_fraction": attempts / actual_steps if actual_steps else float("inf"),
                "boundary_projection_fraction": projections / actual_steps if actual_steps else float("inf")}
    result = {"all": summarize(rows)}
    for key in ("normal", "challenge", "stationary", "fast"):
        selected = [row for row in rows if str(row.get("kind", row.get("challenge_type", ""))).lower() == key]
        if key == "normal":
            selected = [row for row in rows if str(row.get("split", "")).lower() in {"train", "validation", "normal"} and str(row.get("kind", "normal")).lower() == "normal"]
        result[key] = summarize(selected)
    return result


def normal_release_gate(train: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    train = train.get("all", train) if isinstance(train, Mapping) else {}
    validation = validation.get("all", validation) if isinstance(validation, Mapping) else {}
    train_limit = math.floor(0.01 * max(0, _integer(train.get("actual_steps"), 0)))
    val_limit = math.floor(0.01 * max(0, _integer(validation.get("actual_steps"), 0)))
    checks = {
        "train_success": _integer(train.get("task_success_count"), -1) >= 4 and _integer(train.get("episode_count"), -1) == 6,
        "validation_success": _integer(validation.get("task_success_count"), -1) >= 5 and _integer(validation.get("episode_count"), -1) == 8,
        "validation_median_distance": _number(validation.get("median_final_distance_m")) <= 2.0,
        "train_boundary_attempt": _integer(train.get("boundary_attempt_steps"), -1) <= train_limit,
        "train_boundary_projection": _integer(train.get("boundary_projection_steps"), -1) <= train_limit,
        "validation_boundary_attempt": _integer(validation.get("boundary_attempt_steps"), -1) <= val_limit,
        "validation_boundary_projection": _integer(validation.get("boundary_projection_steps"), -1) <= val_limit,
        "zero_train_collisions": _integer(train.get("collision_episode_count"), -1) == 0 and _integer(train.get("physical_contact_steps"), -1) == 0,
        "zero_validation_collisions": _integer(validation.get("collision_episode_count"), -1) == 0 and _integer(validation.get("physical_contact_steps"), -1) == 0,
    }
    return {"passed": bool(all(checks.values())), "checks": checks, "train_boundary_limit": train_limit, "validation_boundary_limit": val_limit}


def challenge_release_gate(metrics: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_metrics = metrics.get("all", metrics) if isinstance(metrics, Mapping) else {}
    stationary = sum(_truth(row.get("task_success", row.get("strict_success"))) for row in episodes if str(row.get("challenge_type", row.get("kind", ""))).lower() == "stationary")
    fast = sum(_truth(row.get("task_success", row.get("strict_success"))) for row in episodes if str(row.get("challenge_type", row.get("kind", ""))).lower() == "fast")
    limit = math.floor(0.01 * max(0, _integer(all_metrics.get("actual_steps"), 0)))
    checks = {"success": _integer(all_metrics.get("task_success_count"), -1) >= 3 and _integer(all_metrics.get("episode_count"), -1) == 4,
              "stationary": stationary >= 1, "fast": fast >= 1,
              "zero_collision": _integer(all_metrics.get("collision_episode_count"), -1) == 0 and _integer(all_metrics.get("physical_contact_steps"), -1) == 0,
              "boundary_attempt": _integer(all_metrics.get("boundary_attempt_steps"), -1) <= limit,
              "boundary_projection": _integer(all_metrics.get("boundary_projection_steps"), -1) <= limit}
    return {"passed": bool(all(checks.values())), "checks": checks, "success_by_type": {"stationary": stationary, "fast": fast}, "boundary_limit": limit}


def recompute_gae(records: Sequence[Mapping[str, Any]], rewards: Sequence[float] | None = None, *, gamma: float = GAE_GAMMA, gae_lambda: float = GAE_LAMBDA, bootstrap_time_limits: bool = False) -> dict[str, Any]:
    """Replay the runner's reverse GAE recurrence without a trainer."""
    ordered = sorted(records, key=lambda row: _integer(row.get("record_index"), 10**9))
    reward_values = list(rewards) if rewards is not None else [_number(row.get("reward", row.get("learning_reward"))) for row in ordered]
    if len(reward_values) != len(ordered):
        raise ValueError("reward count must match records")
    raw_adv: list[float] = [0.0] * len(ordered)
    deltas: list[float] = [0.0] * len(ordered)
    next_advantage = 0.0
    for i in range(len(ordered) - 1, -1, -1):
        row = ordered[i]
        terminated, truncated, done = _truth(row.get("terminated")), _truth(row.get("truncated")), _truth(row.get("done"))
        terminal_for_bootstrap = terminated or (done and not truncated) or (truncated and not bootstrap_time_limits)
        old_value, next_value = _number(row.get("old_value")), _number(row.get("next_value"))
        if not all(math.isfinite(v) for v in (reward_values[i], old_value, next_value)):
            raise ValueError(f"non-finite GAE input at record {i}")
        delta = reward_values[i] + gamma * next_value * (0.0 if terminal_for_bootstrap else 1.0) - old_value
        deltas[i] = delta
        raw_adv[i] = delta + gamma * gae_lambda * (0.0 if done else 1.0) * next_advantage
        next_advantage = raw_adv[i]
    mean = float(np.mean(raw_adv)) if raw_adv else 0.0
    std = float(np.std(raw_adv)) if raw_adv else 0.0
    normalized = [(value - mean) / max(std, 1e-8) for value in raw_adv]
    residuals: list[float] = []
    for i, row in enumerate(ordered):
        for key, expected in (("expected_advantage_from_recurrence", normalized[i]), ("advantage", normalized[i]), ("expected_return_from_recurrence", _number(row.get("old_value")) + raw_adv[i]), ("return", _number(row.get("old_value")) + raw_adv[i])):
            if key in row and math.isfinite(_number(row.get(key))):
                residuals.append(abs(_number(row.get(key)) - expected))
    return {"count": len(ordered), "raw_advantages": raw_adv, "normalized_advantages": normalized,
            "deltas": deltas, "raw_advantage_mean": mean, "raw_advantage_std": std,
            "max_residual": max(residuals, default=0.0), "passed": bool(residuals == [] or max(residuals) < 1e-5),
            "bootstrap_time_limits": bool(bootstrap_time_limits), "done_indices": [i for i, row in enumerate(ordered) if _truth(row.get("done"))]}


def _check_condition_audit(audit: _Audit) -> None:
    payload = _optional_json(audit, "condition_tensor_audit.json", pending_name="condition_tensor_audit")
    if payload is None:
        return
    changed = payload.get("actual_changed_names", {}) if isinstance(payload, Mapping) else {}
    cn = changed.get("C_N", []) if isinstance(changed, Mapping) else []
    ng = changed.get("N_G", []) if isinstance(changed, Mapping) else []
    nr = changed.get("N_R", []) if isinstance(changed, Mapping) else []
    allowed_g = {str(name) for name in (payload.get("allowed_changed_scope", {}).get("N_G", []) if isinstance(payload.get("allowed_changed_scope"), Mapping) else [])}
    allowed_r = {"delta_mean_head.weight", "delta_mean_head.bias"}
    passed = payload.get("passed") is True and not cn and set(ng).issubset(allowed_g) and set(nr).issubset(allowed_r) and payload.get("critic_all_identical") is True
    audit.check("condition_tensor_pair_contract", bool(passed), {"payload_passed": payload.get("passed") if isinstance(payload, Mapping) else None, "C_N": cn, "N_G": ng, "N_R": nr})


def _check_output_head_license(audit: _Audit) -> Mapping[str, Any] | None:
    """Validate the pre-PPO probe that independently licenses the R arm."""
    payload = _optional_json(audit, "output_head_license.json", pending_name="output_head_license")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        audit.check("output_head_license_contract", False, "license artifact must be an object")
        return None
    norms_raw = payload.get("probe_norms_mps", payload.get("probe_norms", []))
    norms = []
    if isinstance(norms_raw, (list, tuple)):
        norms = [_number(value) for value in norms_raw]
    licensed = _truth(payload.get("licensed"))
    primary = str(payload.get("primary", ""))
    reset_names = {str(value) for value in payload.get("resetparameter_names", [])} if isinstance(payload.get("resetparameter_names"), list) else set()
    expected_reset_names = {"delta_mean_head.weight", "delta_mean_head.bias"}
    sanity = payload.get("post_reset_sanity", [])
    sanity_ok = isinstance(sanity, list) and len(sanity) == 6 and all(isinstance(row, Mapping) and _number(row.get("max_abs"), float("nan")) <= 1e-9 for row in sanity)
    errors: list[str] = []
    if payload.get("probe_count") != 6 or len(norms) != 6 or not all(math.isfinite(value) and value > 0.05 for value in norms):
        errors.append("six finite source-control probe norms must all exceed 0.05")
    if payload.get("source_control") != "N_unreset":
        errors.append("source_control must be N_unreset")
    if payload.get("env_steps") != 0 or payload.get("ppo_updates") != 0:
        errors.append("license probes must use zero environment and PPO steps")
    if primary != ("R" if licensed else "N"):
        errors.append("primary does not match the licensed flag")
    if licensed and reset_names != expected_reset_names:
        errors.append("licensed reset scope is not exactly the motion mean head")
    if licensed and not sanity_ok:
        errors.append("post-reset six-probe zero sanity is missing or nonzero")
    audit.check("output_head_license_contract", not errors, {"errors": errors, "primary": primary, "licensed": licensed, "probe_count": payload.get("probe_count"), "probe_norms_mps": norms, "source_control": payload.get("source_control"), "resetparameter_names": sorted(reset_names), "post_reset_sanity_zero": sanity_ok})
    return payload


def _typed_challenge_rows(rows: Sequence[Mapping[str, Any]], splits: Mapping[str, list[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach challenge stationary/fast labels from the frozen manifest.

    The episode logger intentionally records ``kind=normal`` for these cases,
    so using the logger's kind would silently turn the challenge type gate into
    a count over no stationary or fast episodes.
    """
    challenge_types = {
        str(entry.get("scenario_id")): str(entry.get("challenge_type", entry.get("side", ""))).lower()
        for entry in splits.get("challenge", [])
        if isinstance(entry, Mapping)
    }
    typed: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        copied = dict(row)
        scenario_id = str(row.get("scenario_id", ""))
        challenge_type = challenge_types.get(scenario_id, "")
        if challenge_type not in {"stationary", "fast"}:
            missing.append(scenario_id)
        copied["challenge_type"] = challenge_type
        typed.append(copied)
    return typed, missing


def _check_arm_logs(audit: _Audit, arm: str, splits: Mapping[str, list[Mapping[str, Any]]]) -> None:
    arm_root = audit.log_root / arm
    steps_path, episodes_path, action_path = arm_root / "steps.csv", arm_root / "episodes.csv", arm_root / "sampled_execution.csv"
    if not steps_path.is_file() or not episodes_path.is_file():
        audit.mark_pending(f"{arm}_logs", f"missing steps.csv or episodes.csv under {arm_root}")
        return
    steps, episodes = _read_csv(steps_path), _read_csv(episodes_path)
    train_ids = {str(row.get("scenario_id")) for row in splits.get("train", [])}
    val_ids = {str(row.get("scenario_id")) for row in splits.get("validation", [])}
    chal_ids = {str(row.get("scenario_id")) for row in splits.get("challenge", [])}
    expected_phases = {"online": train_ids, "initial_train": train_ids, "final_train": train_ids,
                       "initial_validation": val_ids, "final_validation": val_ids,
                       "initial_challenge": chal_ids, "final_challenge": chal_ids}
    phase_sets = {phase: {str(row.get("scenario_id")) for row in steps if row.get("phase") == phase} for phase in expected_phases}
    phase_counts = {phase: sum(1 for row in episodes if row.get("phase") == phase) for phase in expected_phases}
    action_ok = all(row.get("logprob_action") == "sampled" and _vector(row.get("sampled_delta_v"), 3) is not None and math.isfinite(_number(row.get("old_logprob"))) for row in steps)
    online_rows = [row for row in steps if row.get("phase") == "online"]
    online_ok = len(online_rows) == EXPECTED_ONLINE_STEPS and all(_integer(row.get("episode_step"), -1) >= 1 for row in online_rows)
    phase_ok = all(phase_sets.get(phase, set()) == ids for phase, ids in expected_phases.items()) and all(phase_counts.get(f"initial_{split}") == len(splits.get(split, [])) and phase_counts.get(f"final_{split}") == len(splits.get(split, [])) for split in ("train", "validation", "challenge"))
    audit.check(f"{arm}_phase_and_online_counts", bool(phase_ok and online_ok), {"rows": len(steps), "online_rows": len(online_rows), "phase_sets": {key: sorted(value) for key, value in phase_sets.items()}, "episode_phase_counts": phase_counts})
    audit.check(f"{arm}_sampled_actions_and_logprobs", bool(action_ok), {"rows": len(steps)})
    # The enriched runner records telemetry as a JSON column.  Validate all
    # required fields for online decisions while leaving optional historical
    # evaluation rows inspectable.
    telemetry_errors: list[str] = []
    motion_formula_errors: list[str] = []
    required = ("pre_position_m", "post_position_m", "pre_velocity_mps", "post_velocity_mps", "candidate_position_m", "candidate_velocity_mps", "nominal_acceleration_mps2", "boundary_attempt_axes", "boundary_projected_axes", "env_rng_before", "env_rng_after", "noise_seed", "reset_ordinal")
    for line, row in enumerate(online_rows, 2):
        telemetry = _payload(row.get("motion_telemetry"))
        if not isinstance(telemetry, Mapping) or any(key not in telemetry for key in required):
            telemetry_errors.append(f"line {line}:missing motion telemetry")
            continue
        pre_position = _vector(telemetry.get("pre_position_m"), 3)
        post_position = _vector(telemetry.get("post_position_m"), 3)
        pre_velocity = _vector(telemetry.get("pre_velocity_mps"), 3)
        post_velocity = _vector(telemetry.get("post_velocity_mps"), 3)
        candidate_position = _vector(telemetry.get("candidate_position_m"), 3)
        candidate_velocity = _vector(telemetry.get("candidate_velocity_mps"), 3)
        if pre_position is None or post_position is None or pre_velocity is None or post_velocity is None or candidate_position is None or candidate_velocity is None or not _finite(telemetry.get("nominal_acceleration_mps2")):
            telemetry_errors.append(f"line {line}:non-finite telemetry vectors")
        else:
            # apply_velocity_increment integrates position with the average
            # pre/post velocity, clips the candidate position to the physical
            # box, and, when a boundary is projected, reconciles every velocity
            # component as displacement/dt.  Without a projection the logged
            # post velocity remains the candidate velocity.
            pre_position_np = np.asarray(pre_position, dtype=float)
            post_position_np = np.asarray(post_position, dtype=float)
            pre_velocity_np = np.asarray(pre_velocity, dtype=float)
            post_velocity_np = np.asarray(post_velocity, dtype=float)
            candidate_position_np = np.asarray(candidate_position, dtype=float)
            candidate_velocity_np = np.asarray(candidate_velocity, dtype=float)
            projected_axes = _payload(telemetry.get("boundary_projected_axes"))
            projected = isinstance(projected_axes, (list, tuple)) and any(bool(value) for value in projected_axes)
            expected_candidate = pre_position_np + 0.5 * (pre_velocity_np + candidate_velocity_np) * MOTION_DT_SECONDS
            expected_position = (np.clip(candidate_position_np, np.asarray(MOTION_LOWER_BOUND_M), np.asarray(MOTION_UPPER_BOUND_M)) if projected else candidate_position_np)
            expected_velocity = ((expected_position - pre_position_np) / MOTION_DT_SECONDS) if projected else candidate_velocity_np
            if not np.allclose(candidate_position_np, expected_candidate, rtol=0.0, atol=1e-6):
                motion_formula_errors.append(f"line {line}:candidate position integration")
            if not np.allclose(post_position_np, expected_position, rtol=0.0, atol=1e-6):
                motion_formula_errors.append(f"line {line}:post position clip/reconcile")
            if not np.allclose(post_velocity_np, expected_velocity, rtol=0.0, atol=1e-6):
                motion_formula_errors.append(f"line {line}:post velocity reconciliation formula")
        critic = _payload(row.get("critic_input_normalized"))
        if not isinstance(critic, (list, tuple)) or len(critic) != EXPECTED_CENTRAL_DIM:
            telemetry_errors.append(f"line {line}:critic dimension")
        if "value" not in row or not math.isfinite(_number(row.get("value"))):
            telemetry_errors.append(f"line {line}:missing value")
        if "torch_rng_before" not in row or not row.get("torch_rng_before"):
            telemetry_errors.append(f"line {line}:missing torch rng")
    audit.check(f"{arm}_online_motion_telemetry", not telemetry_errors, {"errors": telemetry_errors[:30], "rows": len(online_rows)})
    audit.check(f"{arm}_boundary_displacement_dt_formula", not motion_formula_errors, {"errors": motion_formula_errors[:30], "rows": len(online_rows), "dt_seconds": MOTION_DT_SECONDS, "lower_bound_m": list(MOTION_LOWER_BOUND_M), "upper_bound_m": list(MOTION_UPPER_BOUND_M)})
    if action_path.is_file():
        action_rows = _read_csv(action_path)
        by_key = {(str(row.get("episode_id")), _integer(row.get("step"), -1)): row for row in action_rows}
        joins = [row for row in online_rows if (str(row.get("episode_id")), _integer(row.get("episode_step"), -1) - 1) in by_key]
        numeric_errors: list[str] = []
        for row in online_rows:
            key = (str(row.get("episode_id")), _integer(row.get("episode_step"), -1) - 1)
            action = by_key.get(key)
            if action is None:
                continue
            sampled = _vector(row.get("sampled_delta_v"), 3)
            executed = _vector(row.get("executed_delta_v"), 3)
            action_sampled = [_number(action.get(f"sampled_delta_v_{axis}")) for axis in "xyz"]
            action_executed = [_number(action.get(f"executed_delta_v_{axis}")) for axis in "xyz"]
            if sampled is None or executed is None or not all(math.isfinite(value) for value in action_sampled + action_executed):
                numeric_errors.append(f"{key}:non-finite action vector")
                continue
            if not np.allclose(sampled, action_sampled, rtol=0.0, atol=1e-7):
                numeric_errors.append(f"{key}:sampled_delta_v mismatch")
            if not np.allclose(executed, action_executed, rtol=0.0, atol=1e-7):
                numeric_errors.append(f"{key}:executed_delta_v mismatch")
            if not _close(row.get("old_logprob"), action.get("old_logprob"), 1e-7):
                numeric_errors.append(f"{key}:old_logprob mismatch")
            if action.get("logprob_action") != "sampled":
                numeric_errors.append(f"{key}:action logger is not sampled")
        audit.check(f"{arm}_sampled_execution_join", len(joins) == len(online_rows) and not numeric_errors, {"online_rows": len(online_rows), "joined": len(joins), "action_rows": len(action_rows), "numeric_errors": numeric_errors[:30]})
    else:
        audit.mark_pending(f"{arm}_sampled_execution_join", f"missing artifact: {action_path}")
    result_path = arm_root / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    if result is None:
        audit.mark_pending(f"{arm}_result", f"missing artifact: {result_path}")
    else:
        updates = [result.get("updates"), result.get("actual_Adam_steps")]
        audit.check(f"{arm}_result_online_contract", result.get("online_steps") == 720 and result.get("updates") == 6 and result.get("actual_Adam_steps") == 48, {"online_steps": result.get("online_steps"), "updates": result.get("updates"), "actual_Adam_steps": result.get("actual_Adam_steps")})
        final_train_rows = [row for row in episodes if row.get("phase") == "final_train"]
        final_val_rows = [row for row in episodes if row.get("phase") == "final_validation"]
        final_challenge_rows, missing_challenge_types = _typed_challenge_rows([row for row in episodes if row.get("phase") == "final_challenge"], splits)
        final_train = recompute_metrics(final_train_rows)
        final_val = recompute_metrics(final_val_rows)
        final_chal = recompute_metrics(final_challenge_rows)
        gate = normal_release_gate(final_train, final_val)
        challenge = challenge_release_gate(final_chal, final_challenge_rows)
        audit.check(f"{arm}_challenge_type_manifest_join", not missing_challenge_types and len(final_challenge_rows) == EXPECTED_CHALLENGE_SCENES, {"missing_types": missing_challenge_types, "rows": len(final_challenge_rows), "types": {key: sum(row.get("challenge_type") == key for row in final_challenge_rows) for key in ("stationary", "fast")}})
        audit.check(f"{arm}_gate_recomputed", result.get("normal_gate", {}).get("passed") == gate["passed"] and result.get("challenge_gate", {}).get("passed") == challenge["passed"], {"normal": gate, "challenge": challenge})
        online_episodes = [row for row in episodes if row.get("phase") == "online"]
        audit.details[f"{arm}_online_metrics_recomputed"] = recompute_metrics(online_episodes)
    for index in range(1, EXPECTED_UPDATES + 1):
        path = arm_root / f"update_{index:02d}.json"
        if not path.is_file():
            audit.mark_pending(f"{arm}_update_{index:02d}", f"missing artifact: {path}")
            continue
        update = _read_json(path)
        records = update.get("per_record_audit", []) if isinstance(update, Mapping) else []
        online_for_update = [row for row in online_rows[(index - 1) * 120:index * 120]]
        rewards = [_number(row.get("learning_reward")) for row in online_for_update]
        if isinstance(records, list) and len(records) == 120 and len(rewards) == 120:
            gae = recompute_gae(records, rewards, bootstrap_time_limits=False)
            audit.check(f"{arm}_update_{index:02d}_gae", bool(gae["passed"] and update.get("bootstrap_time_limits") is False and update.get("gae_recurrence_matches_training_records") is True and update.get("gae_does_not_cross_done_boundary") is True), {"max_residual": gae["max_residual"], "stored": {key: update.get(key) for key in ("bootstrap_time_limits", "gae_recurrence_matches_training_records", "gae_does_not_cross_done_boundary")}})
        else:
            audit.check(f"{arm}_update_{index:02d}_gae", False, {"records": len(records) if isinstance(records, list) else None, "rewards": len(rewards)})
    # Checkpoint paths are existence checks only; the audit never deserializes
    # model files, preserving complete independence from PyTorch execution.
    for checkpoint in ("initial.pt", "final.pt"):
        path = audit.data_root / arm / checkpoint
        if not path.is_file():
            audit.mark_pending(f"{arm}_{checkpoint}", f"missing artifact: {path}")


def _check_rng_prefix(audit: _Audit) -> None:
    """Compare shared case/ordinal/timestep RNG prefixes across conditions."""
    rows_by_arm: dict[str, list[dict[str, str]]] = {}
    missing: list[str] = []
    for arm in EXPECTED_ARMS:
        path = audit.log_root / arm / "steps.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            rows_by_arm[arm] = _read_csv(path)
        except Exception as exc:
            missing.append(f"{path}: {type(exc).__name__}: {exc}")
    if missing:
        audit.mark_pending("rng_common_prefix", {"missing": missing})
        return

    env_by_arm: dict[str, dict[tuple[str, str, str, str], str]] = {}
    torch_by_arm: dict[str, dict[tuple[str, str], str]] = {}
    for arm, rows in rows_by_arm.items():
        env_values: dict[tuple[str, str, str, str], str] = {}
        torch_values: dict[tuple[str, str], str] = {}
        for row in rows:
            if row.get("phase") != "online":
                continue
            telemetry = _payload(row.get("motion_telemetry"))
            if isinstance(telemetry, Mapping) and telemetry.get("env_rng_before"):
                # case, reset ordinal, episode timestep, and phase are the
                # contract grouping.  The ordinal keeps filler resets from
                # being falsely treated as the same noise stream.
                key = (str(row.get("scenario_id", "")), str(row.get("episode_ordinal", "")), str(row.get("episode_step", "")), str(row.get("phase", "")))
                env_values[key] = str(telemetry.get("env_rng_before"))
            if row.get("torch_rng_before"):
                # block/block_step is the 120-call PPO action prefix.  It is
                # deliberately separate from the environment filler key.
                key = (str(row.get("block", "")), str(row.get("block_step", "")))
                torch_values[key] = str(row.get("torch_rng_before"))
        env_by_arm[arm] = env_values
        torch_by_arm[arm] = torch_values

    incomplete = {arm: len([row for row in rows if row.get("phase") == "online"]) for arm, rows in rows_by_arm.items() if len([row for row in rows if row.get("phase") == "online"]) != EXPECTED_ONLINE_STEPS}
    if incomplete:
        audit.mark_pending("rng_common_prefix", {"incomplete_online_rows": incomplete})
        return

    env_common = set.intersection(*(set(values) for values in env_by_arm.values())) if env_by_arm else set()
    torch_common = set.intersection(*(set(values) for values in torch_by_arm.values())) if torch_by_arm else set()
    env_mismatches = [key for key in sorted(env_common) if len({env_by_arm[arm][key] for arm in EXPECTED_ARMS}) != 1]
    torch_mismatches = [key for key in sorted(torch_common) if len({torch_by_arm[arm][key] for arm in EXPECTED_ARMS}) != 1]
    # Six blocks x 120 action calls are required once all four arms finish.
    passed = len(env_common) > 0 and len(torch_common) == EXPECTED_UPDATES * 120 and not env_mismatches and not torch_mismatches
    audit.check("rng_common_case_ordinal_timestep_prefix", passed, {"env_common_keys": len(env_common), "env_mismatches": [list(key) for key in env_mismatches[:20]], "torch_common_keys": len(torch_common), "torch_mismatches": [list(key) for key in torch_mismatches[:20]], "per_arm_env_keys": {arm: len(values) for arm, values in env_by_arm.items()}, "per_arm_torch_keys": {arm: len(values) for arm, values in torch_by_arm.items()}})


def _check_pair_fingerprints(audit: _Audit) -> None:
    payload = _optional_json(audit, "condition_tensor_audit.json", pending_name="pair_tensor_audit")
    if payload is None:
        return
    for key, expected in (("C_N", []), ("N_G", None), ("N_R", None)):
        details = payload.get("actual_changed_names", {}).get(key) if isinstance(payload.get("actual_changed_names"), Mapping) else None
        if key == "C_N":
            audit.check("condition_C_N_exact", details == expected, {"actual": details})
        elif details is not None:
            audit.check(f"condition_{key}_scope", all(str(name).startswith("gat.layers.0.") for name in details) if key == "N_G" else set(details).issubset({"delta_mean_head.weight", "delta_mean_head.bias"}), {"actual": details})


def audit_artifacts(log_root: str | Path = DEFAULT_LOG_ROOT, *, output: str | Path | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
    """Audit a completed or partially completed transfer run without running it."""
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    log = Path(log_root)
    if not log.is_absolute():
        log = root / log
    data = _discover_data_root(root, log)
    audit = _Audit(root, log, data)
    audit.check("artifact_roots_exist", audit.log_root.is_dir() and audit.data_root.is_dir(), {"log_root": str(audit.log_root), "data_root": str(audit.data_root)})
    spec = _check_spec(audit) if audit.data_root.is_dir() else None
    splits, _manifest = _check_splits(audit) if audit.data_root.is_dir() else ({}, None)
    _check_protected(audit)
    _check_calibration_bank(audit)
    _check_calibration_chain(audit)
    _check_statistics(audit)
    _check_boundary_and_calibration_logs(audit)
    _phase_runtime(audit, "calibrate", EXPECTED_CALIBRATION_STEPS)
    # Derive train runtime from the emitted rows after evaluation early
    # termination, rather than assuming four full 120-step eval sweeps.
    _phase_runtime(audit, "train", None)
    _check_condition_audit(audit)
    _check_pair_fingerprints(audit)
    license_payload = _check_output_head_license(audit)
    # A top-level smoke result is the final primary/next-stage claim.  Keep
    # it pending while training artifacts are still being generated.
    smoke = _optional_json(audit, "smoke_result.json", pending_name="smoke_result")
    if smoke is not None:
        arms = smoke.get("arms", {}) if isinstance(smoke, Mapping) else {}
        expected_primary = str(license_payload.get("primary")) if isinstance(license_payload, Mapping) else None
        primary_ok = str(smoke.get("primary")) in {"N", "R"} and (expected_primary is None or str(smoke.get("primary")) == expected_primary)
        required_arms = ("C", "N", "G", "R") if str(smoke.get("primary")) == "R" else ("C", "N", "G")
        audit.check("smoke_result_primary_contract", isinstance(arms, Mapping) and primary_ok and all(arm in arms for arm in required_arms), {"primary": smoke.get("primary") if isinstance(smoke, Mapping) else None, "expected_primary_from_license": expected_primary, "arms": sorted(arms) if isinstance(arms, Mapping) else None, "required_arms": required_arms})
    for arm in EXPECTED_ARMS:
        _check_arm_logs(audit, arm, splits)
    _check_rng_prefix(audit)
    # Config and runtime evidence are useful backstops against accidentally
    # evaluating an old source or a B branch.  Ignore ordinary prose strings,
    # but reject explicit checkpoint B paths and nonzero B loads.
    if spec is not None:
        strings = list(_flatten_strings(spec))
        bad_b = [value for value in strings if re.search(r"(?:^|[\\/])B(?:[_.\\/]|$)", value, re.IGNORECASE)]
        audit.check("spec_contains_no_B_checkpoint_path", not bad_b, {"paths": bad_b[:20]})
    status = "failed" if audit.failures else ("pending" if audit.pending else "passed")
    report: dict[str, Any] = {
        "schema_version": "single_uav_ppo_a_norm_independent_audit_v1",
        "status": status,
        "passed": status == "passed",
        "failure_count": len(audit.failures),
        "pending_count": len(audit.pending),
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
        "checkpoint_loads_executed": 0,
    }
    if output is not None:
        path = Path(output)
        if not path.is_absolute():
            path = root / path
        # Deliberately do not create directories or overwrite evidence.  The
        # caller chooses the already-existing output directory.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        report["output"] = str(path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit_artifacts(args.log_root, output=args.output), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = [
    "_Audit", "audit_artifacts", "challenge_release_gate", "normal_release_gate", "recompute_metrics", "recompute_episode_metrics", "normal_gate", "challenge_gate", "recompute_gae", "parse_vector", "canonical_hash", "DEFAULT_LOG_ROOT", "EXPECTED_ARMS",
]

# Compatibility aliases make the pure recomputation helpers easy to discover
# alongside the similarly named foundation audit helpers.
recompute_episode_metrics = recompute_metrics
normal_gate = normal_release_gate
challenge_gate = challenge_release_gate


if __name__ == "__main__":
    main()
