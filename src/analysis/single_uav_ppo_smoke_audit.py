"""Read-only artifact audit for the paired single-UAV PPO smoke run.

The smoke runner owns data collection and checkpoint creation.  This module
only reads the resulting JSON, CSV, and checkpoint artifacts and returns a
machine-readable report.  It deliberately does not import the runner,
construct an environment, load teacher data, or write an audit artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Torch is needed only for checkpoint inspection.
    import torch
except Exception:  # pragma: no cover - useful for lightweight report parsing
    torch = None  # type: ignore[assignment]


CALIBRATION_STEPS = 288
ONLINE_STEPS = 2160
PPO_UPDATES = 18
ROLLOUT_STEPS = 120
OPTIMIZER_STEPS_PER_UPDATE = 8
OPTIMIZER_STEPS = 144
VALIDATION_STAGE_COUNTS = {0: 10, 1: 2, 2: 6, 3: 10}
VALIDATION_KIND_COUNTS = {"normal": 2, "single": 4, "full": 4}
EXPECTED_DIRECTION_INDICES = [39, 40, 41]
EXPECTED_DIRECTION_FEATURES = [
    "relative_obstacle_dx_body_m",
    "relative_obstacle_dy_body_m",
    "relative_obstacle_dz_body_m",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (_repo_root() / path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def parse_vector(value: Any, *, size: int = 3) -> np.ndarray:
    """Parse a JSON/sequence vector and require the requested finite shape."""

    parsed = _payload(value)
    array = np.asarray(parsed, dtype=float).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"expected a finite vector of length {size}")
    return array


def total_correction(sampled: Sequence[float], executed: Sequence[float]) -> np.ndarray:
    """Return the execution residual ``executed - sampled``."""

    return parse_vector(executed) - parse_vector(sampled)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    a, b = _number(left), _number(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=atol)


def _finite_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_metrics(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_metrics(item) for item in value)
    if isinstance(value, (int, float, np.number)):
        return bool(math.isfinite(float(value)))
    return True


def _json_array(value: Any) -> list[Any]:
    parsed = _payload(value)
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def recompute_episode_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the runner's ``summarize_episodes`` contract from CSV rows."""

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        physical = [row for row in selected if str(row.get("kind")) != "normal"]
        count = len(selected)
        steps = sum(_integer(row.get("steps"), 0) for row in selected)
        physical_steps = sum(_integer(row.get("raw_action_physical_denominator"), 0) for row in selected)
        stable = sum(_truth(row.get("stable_arrival")) for row in selected)
        successes = sum(_truth(row.get("task_success")) for row in selected)
        bypasses = sum(_truth(row.get("physical_bypass")) for row in physical)
        collisions = sum(_truth(row.get("collision")) for row in selected)
        timeout = sum(_truth(row.get("timeout_failure")) for row in selected)
        raw_limits = sum(_truth(row.get("time_limit_reached_raw")) for row in selected)
        cuts = sum(_truth(row.get("rollout_window_cut")) for row in selected)
        danger = sum(_integer(row.get("raw_action_danger_steps"), 0) for row in selected)
        safety = sum(_integer(row.get("safety_projection_steps"), 0) for row in selected)
        boundary = sum(_integer(row.get("boundary_projection_steps"), 0) for row in selected)
        return {
            "episodes": count,
            "steps": steps,
            "stable_arrivals": stable,
            "stable_arrival_rate": _rate(stable, count),
            "task_successes": successes,
            "task_success_rate": _rate(successes, count),
            "physical_bypass_episodes": len(physical),
            "physical_bypasses": bypasses,
            "physical_bypass_rate": _rate(bypasses, len(physical)),
            "collisions": collisions,
            "collision_rate": _rate(collisions, count),
            "timeout_failures": timeout,
            "timeout_rate": _rate(timeout, count),
            "raw_time_limits": raw_limits,
            "rollout_window_cuts": cuts,
            "raw_action_physical_danger_steps": danger,
            "raw_action_physical_denominator": physical_steps,
            "raw_action_danger_rate": _rate(danger, physical_steps),
            "safety_projection_steps": safety,
            "safety_projection_rate": _rate(safety, steps),
            "boundary_projection_steps": boundary,
        }

    return {
        "all": aggregate(rows),
        **{kind: aggregate([row for row in rows if str(row.get("kind")) == kind]) for kind in ("normal", "single", "full")},
    }


def _metrics_match(actual: Any, expected: Any, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected mapping"]
        for key, value in expected.items():
            if key not in actual:
                failures.append(f"{path}.{key}: missing")
            else:
                failures.extend(_metrics_match(actual[key], value, f"{path}.{key}"))
        return failures
    if expected is None:
        if actual is not None and str(actual).strip().lower() not in {"", "none", "null"}:
            failures.append(f"{path}: expected null, got {actual!r}")
        return failures
    if isinstance(expected, (int, float, np.number)):
        if not _close(actual, expected):
            failures.append(f"{path}: expected {expected!r}, got {actual!r}")
        return failures
    if actual != expected:
        failures.append(f"{path}: expected {expected!r}, got {actual!r}")
    return failures


class _Audit:
    def __init__(self, data_root: Path, log_root: Path):
        self.data_root = data_root.resolve()
        self.log_root = log_root.resolve()
        self.results: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.failures: list[dict[str, Any]] = []
        self.diagnostics: list[str] = []
        self.curriculum: Mapping[str, Any] | None = None
        self.data_cache: dict[str, Any] = {}
        self.log_cache: dict[str, Any] = {}
        self.arm_reports: dict[str, dict[str, Any]] = {"A": {}, "B": {}}
        self.costs: dict[str, dict[str, Any]] = {"A": {}, "B": {}}

    def check(self, name: str, passed: bool, details: Any = None) -> None:
        passed = bool(passed)
        self.results[name] = passed
        if details is not None:
            self.details[name] = details
        if not passed:
            message = details if isinstance(details, str) else json.dumps(details, ensure_ascii=False, default=str)
            self.failures.append({"check": name, "details": message})

    def guarded(self, name: str, function: Any) -> Any:
        try:
            value = function()
            self.check(name, True, value if value is not True else None)
            return value
        except Exception as exc:  # Every artifact defect should be reportable.
            self.check(name, False, f"{type(exc).__name__}: {exc}")
            self.diagnostics.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    def data(self, relative: str) -> Any:
        if relative not in self.data_cache:
            path = self.data_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            self.data_cache[relative] = _read_json(path)
        return self.data_cache[relative]

    def log(self, relative: str) -> Any:
        if relative not in self.log_cache:
            path = self.log_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            self.log_cache[relative] = _read_json(path) if path.suffix.lower() == ".json" else _read_csv(path)
        return self.log_cache[relative]


def _check_roots(audit: _Audit) -> None:
    audit.check("independent_roots", audit.data_root != audit.log_root and audit.data_root.is_dir() and audit.log_root.is_dir(), {"data_root": str(audit.data_root), "log_root": str(audit.log_root)})


def _check_curriculum(audit: _Audit) -> None:
    curriculum = audit.data("curriculum.json")
    if not isinstance(curriculum, Mapping):
        raise ValueError("curriculum.json must be an object")
    audit.curriculum = curriculum
    train = list(curriculum.get("train_pool", []))
    validation = list(curriculum.get("validation_pool", []))
    schedule = list(curriculum.get("schedule", []))
    if len(train) != 18 or len(validation) != 10:
        raise AssertionError(f"expected 18 train and 10 validation cases, got {len(train)} and {len(validation)}")
    if Counter(str(row.get("kind")) for row in train) != Counter({"normal": 6, "single": 6, "full": 6}):
        raise AssertionError("train curriculum kind balance drifted")
    if Counter(str(row.get("kind")) for row in validation) != Counter({"normal": 2, "single": 4, "full": 4}):
        raise AssertionError("validation curriculum kind balance drifted")
    for field in ("episode_id", "scenario_id", "scenario_seed", "noise_seed", "geometry_sha256"):
        left = {row.get(field) for row in train}
        right = {row.get(field) for row in validation}
        if left & right:
            raise AssertionError(f"train/validation overlap in {field}")
    if len(schedule) != PPO_UPDATES or sum(_integer(row.get("rollout_steps"), -1) for row in schedule) != ONLINE_STEPS:
        raise AssertionError("schedule budget drifted")
    if [str(row.get("stage_name")) for row in schedule] != ["stage1_normal"] * 6 + ["stage2_single"] * 6 + ["stage3_full"] * 6:
        raise AssertionError("schedule stages drifted")
    if not all(row.get("arms") == ["A", "B"] and _truth(row.get("shared_both_arms")) for row in schedule):
        raise AssertionError("schedule is not shared by both arms")
    provenance = curriculum.get("provenance", {})
    assertions = provenance.get("train_validation_split_assertions", {})
    if any(assertions.get(key) is not True for key in ("ids_disjoint", "scenario_seeds_disjoint", "noise_seeds_disjoint", "geometry_hashes_disjoint")):
        raise AssertionError("curriculum provenance does not preserve split assertions")
    all_ids = [str(row.get("scenario_id", "")) for row in train + validation]
    if any(not (item.startswith("smoke_train_") or item.startswith("smoke_validation_")) for item in all_ids):
        raise AssertionError("curriculum contains a non-smoke or historical pool id")
    audit.check("curriculum_manifest", True, {"train": len(train), "validation": len(validation), "schedule": len(schedule)})
    audit.check("train_validation_geometry_noise_disjoint", True, {"fields": ["geometry_sha256", "noise_seed", "scenario_seed"]})
    audit.check("no_old_20_ids", True, {"scenario_ids": len(all_ids), "historical_pool_ids": 0})


def _check_manifest(audit: _Audit) -> None:
    lock = audit.data("config.lock.json")
    if not isinstance(lock, Mapping):
        raise ValueError("config.lock.json must be an object")
    spec = lock.get("smoke_specification", {})
    expected = {"calibration_steps_per_arm": CALIBRATION_STEPS, "ppo_steps_per_arm": ONLINE_STEPS, "updates_per_arm": PPO_UPDATES, "rollout_steps": ROLLOUT_STEPS}
    if any(_integer(spec.get(key)) != value for key, value in expected.items()):
        raise AssertionError("frozen smoke budget in config.lock.json drifted")
    if lock.get("historical_20_used_for_training_or_validation") is not False or lock.get("sampling_mode") != "sampled" or _integer(lock.get("ppo_online_teacher_samples")) != 0:
        raise AssertionError("config lock permits historical samples, teacher samples, or wrong sampling mode")
    if audit.curriculum is not None and lock.get("curriculum_sha256") != _canonical_hash(audit.curriculum):
        raise AssertionError("config lock curriculum hash mismatch")
    resolved = audit.data("resolved_environment_config.json")
    specification = lock.get("smoke_specification", {})
    frozen = {
        "dt_s": 0.2,
        "max_speed_mps": 12.0,
        "max_acceleration_mps2": 3.0,
        "max_delta_v": 0.6,
        "safety_distance_m": 2.0,
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "safety_projection_enabled": True,
        "safety_execution_mode": "legacy",
        "ppo_action_logprob_mode": "sampled",
        "uav_count": 1,
        "node_feature_dim": 42,
        "max_steps": 120,
    }
    for key, expected in frozen.items():
        actual = resolved.get(key)
        if isinstance(expected, float):
            if not _close(actual, expected, atol=0.0):
                raise AssertionError(f"resolved frozen physics field {key} drifted: {actual!r}")
        elif actual != expected:
            raise AssertionError(f"resolved frozen physics field {key} drifted: {actual!r}")
    for key in (
        "pass_commit_enabled",
        "lateral_exit_enabled",
        "radial_intent_enabled",
        "goal_recovery_correction_enabled",
        "goal_recovery_terminal_capture_enabled",
    ):
        if resolved.get(key) is not False:
            raise AssertionError(f"resolved assistance switch {key} is enabled")
    shared = audit.data("shared_train_statistics.json")
    if _canonical(resolved.get("train_potential_statistics")) != _canonical(shared.get("potential_statistics")):
        raise AssertionError("resolved train potential statistics differ from shared training artifact")

    baseline_value = specification.get("baseline_config")
    baseline_path = _resolve_repo_path(baseline_value)
    if not baseline_path.is_file() or _sha256(baseline_path) != str(specification.get("baseline_config_sha256")):
        raise AssertionError("baseline_config_sha256 does not match the actual frozen baseline file")
    source_hashes: dict[str, str] = {}
    for group, source in dict(specification.get("sources", {})).items():
        if not isinstance(source, Mapping):
            raise AssertionError(f"source {group} metadata is not a mapping")
        source_path = _resolve_repo_path(source.get("checkpoint"))
        if not source_path.is_file() or _sha256(source_path) != str(source.get("sha256")):
            raise AssertionError(f"source checkpoint SHA mismatch for arm {group}")
        source_hashes[str(group)] = str(source.get("sha256"))
    audit.check(
        "config_and_resolved_contract",
        True,
        {
            "sampling_mode": lock.get("sampling_mode"),
            "node_feature_dim": resolved.get("node_feature_dim"),
            "frozen_physics": {key: resolved.get(key) for key in frozen},
            "assistance_switches_disabled": True,
            "potential_statistics_shared": True,
            "baseline_config_sha256": str(specification.get("baseline_config_sha256")),
            "source_checkpoint_sha256": source_hashes,
        },
    )


def _check_calibration(audit: _Audit) -> None:
    a_json, b_json = audit.data("calibration_A.json"), audit.data("calibration_B.json")
    a_csv, b_csv = audit.log("calibration_A.csv"), audit.log("calibration_B.csv")
    train_ids = {str(row.get("scenario_id")) for row in (audit.curriculum or {}).get("train_pool", [])}
    validation_ids = {str(row.get("scenario_id")) for row in (audit.curriculum or {}).get("validation_pool", [])}
    hashes: list[str] = []
    for group, payload, rows in (("A", a_json, a_csv), ("B", b_json, b_csv)):
        if _integer(payload.get("actual_interaction_steps")) != CALIBRATION_STEPS or _integer(payload.get("ppo_samples")) != 0:
            raise AssertionError(f"calibration {group} budget/ppo_samples invalid")
        records = list(payload.get("records", []))
        if len(records) != CALIBRATION_STEPS or len(rows) != CALIBRATION_STEPS:
            raise AssertionError(f"calibration {group} must contain exactly {CALIBRATION_STEPS} rows")
        if any(str(row.get("scenario_id")) not in train_ids for row in records + rows):
            raise AssertionError(f"calibration {group} contains a non-training scene")
        if any(str(row.get("scenario_id")) in validation_ids for row in records + rows):
            raise AssertionError(f"calibration {group} contains a validation scene")
        if any(str(row.get("source", "")) != "train_only_random_calibration" for row in records + rows):
            raise AssertionError(f"calibration {group} source provenance drifted")
        hashes.append(str(payload.get("stream_sha256", "")))
        # Each of the 18 train scenes contributes its 16-step calibration stream.
        counts = Counter(str(row.get("scenario_id")) for row in rows)
        if set(counts) != train_ids or any(count != 16 for count in counts.values()):
            raise AssertionError(f"calibration {group} does not cover each train case exactly 16 times")
    if hashes[0] != hashes[1]:
        raise AssertionError("paired calibration stream hashes differ")
    def action_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row.get("scenario_id"), _integer(row.get("noise_seed")), _integer(row.get("step")), json.dumps(_payload(row.get("delta_v")), sort_keys=True, separators=(",", ":")))
    if sorted(action_key(row) for row in a_csv) != sorted(action_key(row) for row in b_csv):
        raise AssertionError("paired calibration actions/streams differ")
    audit.check("calibration_train_only_288_paired", True, {"steps_per_arm": CALIBRATION_STEPS, "stream_sha256": hashes[0]})


def _normalizer_contract(stats: Mapping[str, Any]) -> dict[str, Any]:
    observation = stats.get("observation_normalizer", {})
    normalizers = observation.get("normalizers", {}) if isinstance(observation, Mapping) else {}
    entries: list[tuple[str, Mapping[str, Any]]] = [(f"observation.{key}", value) for key, value in normalizers.items() if isinstance(value, Mapping)]
    central = stats.get("central_normalizer", {})
    if isinstance(central, Mapping):
        entries.append(("central", central))
    if not entries:
        raise AssertionError("no fitted normalizer entries")
    counts: dict[str, int] = {}
    dims: dict[str, int] = {}
    for name, state in entries:
        mean = np.asarray(state.get("mean"), dtype=float).reshape(-1)
        scale = np.asarray(state.get("scale"), dtype=float).reshape(-1)
        count = np.asarray(state.get("count"), dtype=float).reshape(-1)
        if mean.size == 0 or mean.size != scale.size or mean.size != count.size:
            raise AssertionError(f"normalizer {name} has inconsistent dimensions")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(count).all() or np.any(scale <= 0.0) or np.any(count <= 0.0):
            raise AssertionError(f"normalizer {name} has non-finite, non-positive, or empty statistics")
        counts[name] = int(np.min(count))
        dims[name] = int(mean.size)
    node = normalizers.get("node_features", {})
    direction_counts = np.asarray(node.get("count"), dtype=float).reshape(-1)[39:42]
    if direction_counts.size != 3 or not np.all(direction_counts > 0):
        raise AssertionError("obstacle direction slots 39:42 lack positive calibration counts")
    potential = stats.get("potential_statistics", {})
    for key in ("u_scale", "c_scale", "delta_u_scale", "delta_goal_scale"):
        scale = _number(potential.get(key))
        if not math.isfinite(scale) or scale <= 0:
            raise AssertionError(f"potential statistic {key} is not a finite positive scale")
    if stats.get("fit_split") != "train" or stats.get("frozen_during_ppo_and_validation") is not True or _integer(stats.get("calibration_steps_per_arm")) != CALIBRATION_STEPS:
        raise AssertionError("shared statistics lack train-only frozen provenance")
    if _integer(potential.get("sample_count")) != CALIBRATION_STEPS or potential.get("fit_split") != "train":
        raise AssertionError("potential statistics lack train-only provenance")
    return {"normalizer_dimensions": dims, "minimum_count_by_entry": counts, "direction_counts": direction_counts.tolist(), "potential_scales": {key: potential.get(key) for key in ("u_scale", "c_scale", "delta_u_scale", "delta_goal_scale")}}


def _check_shared_statistics(audit: _Audit) -> None:
    result = _normalizer_contract(audit.data("shared_train_statistics.json"))
    audit.check("shared_train_statistics_finite_positive_train_only", True, result)


def _check_protected(audit: _Audit) -> None:
    before = audit.log("protected_inputs_before.json")
    checks = audit.log("protected_inputs_checks.json")
    if not isinstance(before, Mapping) or not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError("protected input checks are missing or not all true")
    independently_verified = 0
    for raw_path, expected_hash in before.items():
        path = Path(str(raw_path))
        if path.is_file() and _sha256(path) == str(expected_hash):
            independently_verified += 1
        else:
            raise AssertionError(f"protected input changed or is unavailable: {raw_path}")
    if independently_verified != len(before):
        raise AssertionError("not all protected inputs were independently verified")
    audit.log_cache["protected_checks"] = checks
    audit.check("protected_inputs_all_true", True, {"count": len(checks), "independently_verified": independently_verified})


def _check_initialization(audit: _Audit, group: str) -> None:
    report = audit.log(f"{group}/compatibility.json")
    if report.get("initialized_from_checkpoint") is not True or report.get("source_optimizer_ignored") is not True or report.get("rollout_cleared") is not True:
        raise AssertionError(f"{group} initialization did not report clean warm-start semantics")
    if _integer(report.get("target_optimizer_state_entries"), -1) != 0 or _integer(report.get("target_update_count"), -1) != 0 or _integer(report.get("teacher_online_samples"), -1) != 0:
        raise AssertionError(f"{group} initialization retained optimizer, update, or teacher state")
    if report.get("source_checkpoint_unchanged") is not True or report.get("no_environment_steps") is not True or report.get("no_ppo_update") is not True:
        raise AssertionError(f"{group} initialization provenance is incomplete")
    fresh = report.get("parameters", {}).get("fresh_constructor_verification", {})
    if fresh.get("gat.layers.0") is not True or fresh.get("critic") is not True:
        raise AssertionError(f"{group} fresh GAT/critic preservation failed")
    normalization = report.get("normalization", {})
    if normalization.get("normalizer_reused") is not False or normalization.get("source_statistics_loaded") is not False:
        raise AssertionError(f"{group} reused source normalization statistics")
    compatibility = report.get("compatibility", {})
    if compatibility.get("feature_map_direction_indices") != EXPECTED_DIRECTION_INDICES or compatibility.get("feature_map_direction_semantics") != EXPECTED_DIRECTION_FEATURES:
        raise AssertionError(f"{group} direction feature slots drifted")
    audit.check(f"{group}_initialization_fresh_modules_optimizer_rollout", True, {"fresh_modules": fresh, "optimizer_state_entries": report.get("target_optimizer_state_entries"), "rollout_cleared": report.get("rollout_cleared")})


def _check_actor_contract(audit: _Audit, group: str) -> None:
    report = audit.log(f"{group}/actor_observation_contract.json")
    probe = report.get("real_encoder_input", {})
    shape = probe.get("actor_encoder_input_shape")
    if shape != [1, 42] or _integer(report.get("environment_steps"), -1) != 0 or _integer(report.get("training_or_optimizer_steps"), -1) != 0:
        raise AssertionError(f"{group} actor contract probe did not use a reset-only 42-feature input")
    scenes = list(report.get("scenes", []))
    if len(scenes) != 3:
        raise AssertionError(f"{group} actor contract probe must cover three scenes")
    for scene in scenes:
        if _integer(scene.get("actual_actor_node_feature_dim"), -1) != 42 or _integer(scene.get("normalized_actor_node_feature_dim"), -1) != 42 or scene.get("normalization_status") != "fitted":
            raise AssertionError(f"{group} actor contract scene lacks the fitted 42-feature contract")
        feature_map = scene.get("feature_index_map", {})
        if feature_map.get("estimated_velocity") != [3, 6] or feature_map.get("goal_delta") != [30, 33] or feature_map.get("obstacle_body_direction") != [39, 42]:
            raise AssertionError(f"{group} actor contract direction feature map drifted")
        sensitivity = scene.get("normalized_feature_sensitivity", {})
        for feature_name in ("estimated_velocity", "goal_delta", "obstacle_body_direction"):
            response = sensitivity.get(feature_name, {})
            for measure in ("context_change_l2", "pre_tanh_mean_change_l2"):
                value = _number(response.get(measure))
                if not math.isfinite(value) or value <= 0.0:
                    raise AssertionError(f"{group} {feature_name} sensitivity {measure} is not finite and positive")
    audit.check(f"{group}_direction_slots_and_reset_probe", True, {"scenes": len(scenes), "actor_input_shape": shape})


def _load_arm_rows(audit: _Audit, group: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    steps = audit.log(f"{group}/steps.csv")
    episodes = audit.log(f"{group}/episodes.csv")
    if not steps or not episodes:
        raise AssertionError(f"{group} steps/episodes logs are empty")
    if any(str(row.get("group")) != group for row in steps + episodes):
        raise AssertionError(f"{group} log contains rows from another arm")
    if len({(str(row.get("phase")), str(row.get("episode_id"))) for row in episodes}) != len(episodes):
        raise AssertionError(f"{group} episode identifiers are duplicated within a phase")
    by_episode = {str(row.get("episode_id")): row for row in episodes}
    rows_by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in steps:
        episode = str(row.get("episode_id"))
        if episode not in by_episode:
            raise AssertionError(f"{group} step references an unknown episode {episode}")
        rows_by_episode[episode].append(row)
    for episode, episode_rows in rows_by_episode.items():
        numbers = sorted(_integer(row.get("episode_step")) for row in episode_rows)
        if numbers != list(range(1, len(numbers) + 1)):
            raise AssertionError(f"{group} episode {episode} has non-contiguous 1-based step numbers")
        if _integer(by_episode[episode].get("steps"), -1) != len(episode_rows):
            raise AssertionError(f"{group} episode {episode} step count disagrees with episodes.csv")
    if sum(_integer(row.get("steps"), 0) for row in episodes) != len(steps):
        raise AssertionError(f"{group} episode step counts do not sum to steps.csv rows")
    for row in episodes:
        raw_timeout = _truth(row.get("time_limit_reached_raw"))
        cut = _truth(row.get("rollout_window_cut"))
        if raw_timeout and cut:
            raise AssertionError(f"{group} episode marks raw timeout and rollout cut together")
        if cut and (_truth(row.get("timeout_failure")) or _truth(row.get("truncated"))):
            raise AssertionError(f"{group} rollout cut is incorrectly labeled timeout")
    return steps, episodes


def _check_updates(audit: _Audit) -> None:
    rows = audit.log("ppo_updates.csv")
    if len(rows) != 2 * PPO_UPDATES:
        raise AssertionError(f"ppo_updates.csv has {len(rows)} rows, expected {2 * PPO_UPDATES}")
    for group in ("A", "B"):
        group_rows = [row for row in rows if str(row.get("group")) == group]
        if len(group_rows) != PPO_UPDATES or sorted(_integer(row.get("ppo_update_this_run")) for row in group_rows) != list(range(1, PPO_UPDATES + 1)):
            raise AssertionError(f"{group} PPO update numbering/count drifted")
        for row in group_rows:
            if _integer(row.get("rollout_steps")) != ROLLOUT_STEPS or _integer(row.get("samples")) != ROLLOUT_STEPS or _integer(row.get("actual_optimizer_steps")) != OPTIMIZER_STEPS_PER_UPDATE:
                raise AssertionError(f"{group} PPO update budget drifted")
            if _integer(row.get("teacher_online_samples"), -1) != 0 or not _truth(row.get("sampled_action_logprob_preserved")):
                raise AssertionError(f"{group} PPO update has teacher samples or changed logprob contract")
            if any(_number(row.get(key)) <= 0 for key in ("actor_parameter_delta_l2", "gat_parameter_delta_l2", "critic_parameter_delta_l2", "actor_gradient_norm_max", "gat_gradient_norm_max", "critic_gradient_norm_max")):
                raise AssertionError(f"{group} PPO update lacks a finite positive module update/gradient")
            for key, value in row.items():
                if key in {"group", "stage", "scenario_id", "kind", "side", "sample_source", "sampled_action_logprob_preserved", "nonterminal_window_cut"}:
                    continue
                if value not in ("", None) and not math.isfinite(_number(value)):
                    raise AssertionError(f"{group} PPO update has non-finite statistic {key}")
        cumulative = sorted((_integer(row.get("ppo_update_this_run")), _integer(row.get("optimizer_steps_cumulative"))) for row in group_rows)
        if any(total != update * OPTIMIZER_STEPS_PER_UPDATE for update, total in cumulative):
            raise AssertionError(f"{group} optimizer cumulative count drifted")
    audit.check("ppo_updates_18x2_8_optimizer_steps_finite_positive", True, {"rows": len(rows), "optimizer_steps_per_arm": OPTIMIZER_STEPS})


def _check_validation_and_steps(audit: _Audit, group: str) -> None:
    steps, episodes = _load_arm_rows(audit, group)
    curriculum = audit.curriculum or {}
    train_pool = list(curriculum.get("train_pool", []))
    validation_pool = list(curriculum.get("validation_pool", []))
    train_by_id = {str(row.get("scenario_id")): row for row in train_pool}
    validation_by_id = {str(row.get("scenario_id")): row for row in validation_pool}
    schedule = list(curriculum.get("schedule", []))
    expected_schedule = {
        _integer(row.get("global_update")): (
            str(row.get("scenario_id")),
            str(row.get("kind")),
            str(row.get("side")),
            _integer(row.get("stage")),
        )
        for row in schedule
    }
    if set(expected_schedule) != set(range(1, PPO_UPDATES + 1)):
        raise AssertionError("curriculum schedule does not expose global updates 1..18")

    # Validate every recorded scene against the serialized curriculum.  This
    # catches a mutated scene even when the high-level pool counts remain 18/10.
    recorded_train_ids: set[str] = set()
    recorded_validation_ids: set[str] = set()
    for row in episodes:
        scene_id = str(row.get("scenario_id"))
        phase = str(row.get("phase"))
        lookup = train_by_id if phase == "train" else validation_by_id if phase == "validation" else {}
        if scene_id not in lookup:
            raise AssertionError(f"{group} {phase} record references a scene outside its curriculum split: {scene_id}")
        scene = lookup[scene_id]
        if str(row.get("kind")) != str(scene.get("kind")) or str(row.get("side")) != str(scene.get("side")):
            raise AssertionError(f"{group} scene kind/side disagrees with curriculum for {scene_id}")
        if _integer(row.get("noise_seed"), -1) != _integer(scene.get("noise_seed"), -2) or str(row.get("geometry_sha256")) != str(scene.get("geometry_sha256")):
            raise AssertionError(f"{group} noise/geometry provenance disagrees with curriculum for {scene_id}")
        if phase == "train":
            recorded_train_ids.add(scene_id)
            block_match = re.search(r"_block(\d+)_", str(row.get("episode_id")))
            if block_match is None:
                raise AssertionError(f"{group} train episode does not expose its PPO block id: {row.get('episode_id')}")
            block_id = int(block_match.group(1))
            expected = expected_schedule.get(block_id)
            observed = (scene_id, str(scene.get("kind")), str(scene.get("side")), _integer(row.get("stage")))
            if expected != observed:
                raise AssertionError(f"{group} train block {block_id} scene/stage disagrees with curriculum: {observed!r} != {expected!r}")
        elif phase == "validation":
            recorded_validation_ids.add(scene_id)
    if recorded_train_ids & recorded_validation_ids:
        raise AssertionError(f"{group} recorded train/validation scene leakage: {sorted(recorded_train_ids & recorded_validation_ids)}")
    expected_exposed_ids = {str(row.get("scenario_id")) for row in schedule}
    if recorded_train_ids != expected_exposed_ids:
        raise AssertionError(f"{group} recorded train scene IDs do not match the scheduled exposed train set")
    if recorded_validation_ids != set(validation_by_id):
        raise AssertionError(f"{group} recorded validation scene IDs do not cover the canonical validation pool")

    online = [row for row in steps if str(row.get("phase")) == "train"]
    validation = [row for row in steps if str(row.get("phase")) == "validation"]
    if len(online) != ONLINE_STEPS:
        raise AssertionError(f"{group} online step count is {len(online)}, expected {ONLINE_STEPS}")
    block_counts = Counter(_integer(row.get("block_id")) for row in online)
    if sorted(block_counts) != list(range(1, PPO_UPDATES + 1)) or any(count != ROLLOUT_STEPS for count in block_counts.values()):
        raise AssertionError(f"{group} online blocks are not exactly 18x120")
    actual_schedule: list[tuple[int, str, str, str, int]] = []
    for block_id in range(1, PPO_UPDATES + 1):
        block_rows = [row for row in online if _integer(row.get("block_id")) == block_id]
        signatures = {
            (
                str(row.get("scenario_id")),
                str(row.get("kind")),
                str(row.get("side")),
                _integer(row.get("stage")),
            )
            for row in block_rows
        }
        if len(signatures) != 1 or next(iter(signatures)) != expected_schedule[block_id]:
            raise AssertionError(f"{group} online block {block_id} does not match the curriculum schedule")
        scene_id, kind, side, stage = next(iter(signatures))
        actual_schedule.append((block_id, scene_id, kind, side, stage))
    prior_schedule = audit.arm_reports.get("A", {}).get("online_schedule")
    if group == "B" and prior_schedule is not None and actual_schedule != prior_schedule:
        raise AssertionError("A/B online blocks do not share the same scene schedule")
    if any(str(row.get("sample_source")) != "online_policy_rollout" for row in online) or any(str(row.get("sample_source")) != "deterministic_validation" for row in validation):
        raise AssertionError(f"{group} sample source labels drifted")
    val_episodes = [row for row in episodes if str(row.get("phase")) == "validation"]
    stage_counts = Counter(_integer(row.get("stage")) for row in val_episodes)
    if dict(stage_counts) != VALIDATION_STAGE_COUNTS:
        raise AssertionError(f"{group} validation stage episode counts are {dict(stage_counts)}, expected {VALIDATION_STAGE_COUNTS}")
    for stage, expected_count in VALIDATION_STAGE_COUNTS.items():
        selected = [row for row in val_episodes if _integer(row.get("stage")) == stage]
        if len(selected) != expected_count:
            raise AssertionError(f"{group} validation stage {stage} count drifted")
        allowed_kinds = {"normal", "single", "full"} if stage in (0, 3) else {"normal"} if stage == 1 else {"normal", "single"}
        expected_ids = {scene_id for scene_id, scene in validation_by_id.items() if str(scene.get("kind")) in allowed_kinds}
        actual_ids = {str(row.get("scenario_id")) for row in selected}
        if actual_ids != expected_ids:
            raise AssertionError(f"{group} validation stage {stage} scene IDs do not match the canonical pool subset")
        selected_step_ids = {str(row.get("scenario_id")) for row in validation if _integer(row.get("stage")) == stage}
        if selected_step_ids != expected_ids:
            raise AssertionError(f"{group} validation stage {stage} step records leak or omit canonical scene IDs")
    stage3 = [row for row in val_episodes if _integer(row.get("stage")) == 3]
    if Counter(str(row.get("kind")) for row in stage3) != Counter(VALIDATION_KIND_COUNTS):
        raise AssertionError(f"{group} final validation denominator is not normal=2,single=4,full=4")
    summary = audit.log(f"{group}/arm_summary.json")
    eval_cost = summary.get("evaluation_cost", {})
    if _integer(eval_cost.get("steps"), -1) != len(validation) or _integer(eval_cost.get("evaluations"), -1) != 4 or _integer(eval_cost.get("resets"), -1) != len(val_episodes):
        raise AssertionError(f"{group} arm_summary evaluation cost disagrees with validation logs")
    if _integer(summary.get("ppo_online_steps"), -1) != ONLINE_STEPS or _integer(summary.get("ppo_updates"), -1) != PPO_UPDATES or _integer(summary.get("actual_optimizer_steps"), -1) != OPTIMIZER_STEPS:
        raise AssertionError(f"{group} arm summary PPO budget drifted")
    if _integer(summary.get("teacher_online_samples"), -1) != 0 or summary.get("normalization_unchanged") is not True:
        raise AssertionError(f"{group} arm summary provenance drifted")
    actual = recompute_episode_metrics(stage3)
    reported = summary.get("final_validation", {})
    mismatch = _metrics_match(reported, actual, f"{group}.final_validation")
    if mismatch:
        raise AssertionError("; ".join(mismatch[:5]))
    if actual["normal"]["physical_bypass_rate"] is not None or actual["normal"]["raw_action_danger_rate"] is not None:
        raise AssertionError(f"{group} normal validation incorrectly has physical denominators")
    audit.arm_reports[group].update({"online_steps": len(online), "validation_steps": len(validation), "validation_episodes": len(val_episodes), "final_metrics": actual, "episode_rows": len(episodes), "step_rows": len(steps), "online_schedule": actual_schedule, "recorded_train_scene_count": len(recorded_train_ids), "recorded_validation_scene_count": len(recorded_validation_ids)})
    audit.costs[group].update({"calibration_steps": CALIBRATION_STEPS, "ppo_online_steps": len(online), "validation_steps": len(validation), "validation_episodes": len(val_episodes), "optimizer_steps": OPTIMIZER_STEPS})
    audit.check(f"{group}_online_2160_validation_cost_and_stage3_metrics", True, {"online_steps": len(online), "validation_steps": len(validation), "stage3_kind_counts": dict(Counter(str(row.get("kind")) for row in stage3))})


def _check_action_log(audit: _Audit, group: str) -> None:
    steps = audit.log(f"{group}/steps.csv")
    action_rows = audit.log(f"{group}/sampled_execution.csv")
    if len(action_rows) != len(steps):
        raise AssertionError(f"{group} sampled_execution rows {len(action_rows)} != steps rows {len(steps)}")
    step_index: dict[tuple[str, int], dict[str, str]] = {}
    for row in steps:
        key = (str(row.get("episode_id")), _integer(row.get("episode_step")) - 1)
        if key in step_index:
            raise AssertionError(f"{group} duplicate zero-based action join key {key}")
        step_index[key] = row
    seen: set[tuple[str, int]] = set()
    safety_count = 0
    correction_count = 0
    for row in action_rows:
        key = (str(row.get("episode_id")), _integer(row.get("step")))
        if key not in step_index:
            raise AssertionError(f"{group} sampled_execution row has no steps.csv match: {key}")
        if key in seen:
            raise AssertionError(f"{group} sampled_execution duplicate key: {key}")
        seen.add(key)
        step = step_index[key]
        if row.get("logprob_action") != "sampled" or step.get("logprob_action") != "sampled" or not _close(row.get("old_logprob"), step.get("old_logprob")):
            raise AssertionError(f"{group} sampled old log probability contract drifted at {key}")
        sampled = np.asarray([_number(row.get(f"sampled_delta_v_{axis}")) for axis in "xyz"])
        executed = np.asarray([_number(row.get(f"executed_delta_v_{axis}")) for axis in "xyz"])
        total = np.asarray([_number(row.get(f"total_correction_{axis}")) for axis in "xyz"])
        if not np.isfinite(np.concatenate([sampled, executed, total])).all() or not np.allclose(total, executed - sampled, atol=1e-6, rtol=1e-6):
            raise AssertionError(f"{group} total correction is not executed-sampled at {key}")
        logged_sampled = parse_vector(step.get("sampled_delta_v"))
        logged_executed = parse_vector(step.get("executed_delta_v"))
        logged_total = parse_vector(step.get("total_correction_delta_v"))
        if not np.allclose(sampled, logged_sampled, atol=1e-6) or not np.allclose(executed, logged_executed, atol=1e-6) or not np.allclose(total, logged_total, atol=1e-6):
            raise AssertionError(f"{group} sampled/executed vectors disagree at {key}")
        safety = np.asarray([_number(row.get(f"safety_correction_{axis}")) for axis in "xyz"])
        safety_norm = _number(row.get("safety_correction_norm_mps"))
        if not np.isfinite(safety).all() or not _close(safety_norm, np.linalg.norm(safety), atol=1e-6):
            raise AssertionError(f"{group} safety correction norm is inconsistent at {key}")
        safety_flag = _integer(row.get("reason_safety_projection"), 0) == 1
        if safety_flag != _truth(step.get("safety_projected")):
            raise AssertionError(f"{group} safety source flag disagrees with steps.csv at {key}")
        if not safety_flag and not np.allclose(safety, 0.0, atol=1e-6):
            raise AssertionError(f"{group} non-safety correction was counted as safety at {key}")
        # Action norm/speed clips are non-safety residuals unless a safety
        # projection is explicitly sourced on the same row.
        if not safety_flag and (_integer(row.get("reason_action_norm_clip"), 0) or _integer(row.get("reason_speed_clip"), 0)) and safety_norm > 1e-6:
            raise AssertionError(f"{group} norm/speed clip was counted as a safety correction at {key}")
        safety_count += int(safety_flag)
        correction_count += int(np.linalg.norm(total) > 1e-6)
    if len(seen) != len(steps):
        raise AssertionError(f"{group} sampled_execution does not cover every step row")
    audit.costs[group].update({"sampled_execution_rows": len(action_rows), "total_correction_rows": correction_count, "safety_projection_rows": safety_count})
    audit.check(f"{group}_sampled_execution_join_and_correction_accounting", True, {"rows": len(action_rows), "nonzero_total_corrections": correction_count, "safety_projection_rows": safety_count})


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if torch is None:
        raise RuntimeError("torch is unavailable for checkpoint inspection")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint {path} is not a mapping")
    return payload


def _parameter_group(name: str) -> str:
    return "critic" if name.startswith("critic.") else "gat" if name.startswith("gat.") else "actor"


def _check_checkpoints(audit: _Audit, group: str) -> None:
    initial_path, final_path = audit.data_root / group / "initial.pt", audit.data_root / group / "final.pt"
    if not initial_path.is_file() or not final_path.is_file():
        raise FileNotFoundError(f"missing {group} initial.pt/final.pt")
    initial, final = _load_checkpoint(initial_path), _load_checkpoint(final_path)
    shared = audit.data("shared_train_statistics.json")
    shared_observation = shared.get("observation_normalizer")
    shared_central = shared.get("central_normalizer")
    for name, value in (("initial", initial), ("final", final)):
        policy = value.get("policy_state_dict")
        if not isinstance(policy, Mapping) or not policy:
            raise AssertionError(f"{group} {name} checkpoint lacks policy_state_dict")
        if any(not torch.isfinite(tensor.detach()).all().item() for tensor in policy.values() if torch.is_tensor(tensor)):
            raise AssertionError(f"{group} {name} checkpoint contains non-finite parameters")
        if _canonical(value.get("normalization_stats")) != _canonical(shared_observation) or _canonical(value.get("central_normalization_stats")) != _canonical(shared_central):
            raise AssertionError(f"{group} {name} checkpoint normalization state differs from the shared train artifact")
    if _integer(final.get("update_count"), -1) != PPO_UPDATES or _integer(final.get("ppo_updates_this_run"), -1) != PPO_UPDATES or _integer(final.get("ppo_online_steps_this_run"), -1) != ONLINE_STEPS:
        raise AssertionError(f"{group} final checkpoint budget metadata drifted")
    if final.get("online_sample_source") != "fresh_policy_rollout" or _integer(final.get("teacher_online_samples"), -1) != 0 or final.get("smoke_only") is not True:
        raise AssertionError(f"{group} final checkpoint has wrong online provenance")
    if final.get("group") != group:
        raise AssertionError(f"{group} final checkpoint arm identity drifted")
    optimizer_state = final.get("optimizer_state_dict", {}).get("state", {}) if isinstance(final.get("optimizer_state_dict"), Mapping) else {}
    if not isinstance(optimizer_state, Mapping) or not optimizer_state:
        raise AssertionError(f"{group} final checkpoint has no Adam state")
    optimizer_steps: list[float] = []
    for state in optimizer_state.values():
        if not isinstance(state, Mapping) or "step" not in state:
            raise AssertionError(f"{group} final checkpoint has an optimizer state without an Adam step")
        step = state["step"]
        if torch.is_tensor(step):
            step_value = float(step.detach().cpu().item())
        else:
            step_value = _number(step)
        if not math.isfinite(step_value) or not math.isclose(step_value, OPTIMIZER_STEPS, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(f"{group} final checkpoint Adam step is {step_value}, expected {OPTIMIZER_STEPS}")
        optimizer_steps.append(step_value)
    initial_policy, final_policy = initial["policy_state_dict"], final["policy_state_dict"]
    deltas = {"actor": 0.0, "gat": 0.0, "critic": 0.0}
    for name, tensor in final_policy.items():
        if name in initial_policy and torch.is_tensor(tensor) and torch.is_tensor(initial_policy[name]) and tuple(tensor.shape) == tuple(initial_policy[name].shape):
            deltas[_parameter_group(str(name))] += float((tensor.detach().double() - initial_policy[name].detach().double()).square().sum().item())
    deltas = {key: math.sqrt(value) for key, value in deltas.items()}
    if any(value <= 0 or not math.isfinite(value) for value in deltas.values()):
        raise AssertionError(f"{group} final checkpoint has no nonzero actor/GAT/critic hash delta: {deltas}")
    audit.arm_reports[group]["checkpoint_parameter_deltas"] = deltas
    audit.check(f"{group}_final_checkpoint_metadata_and_parameter_deltas", True, {"update_count": final.get("update_count"), "ppo_online_steps": final.get("ppo_online_steps_this_run"), "optimizer_step_values": optimizer_steps, "normalization_matches_shared": True, "parameter_delta_l2": deltas})


def _check_root_report(audit: _Audit) -> None:
    report = audit.log("smoke_report.json")
    if report.get("status") != "engineering_smoke_completed" or report.get("calibration_streams_identical") is not True or report.get("same_training_schedule") is not True or report.get("same_ppo_budget") is not True or report.get("same_validation_rule") is not True or _integer(report.get("historical_20_training_or_validation_samples"), -1) != 0 or _integer(report.get("teacher_online_samples"), -1) != 0:
        raise AssertionError("root smoke report does not preserve completed paired-smoke provenance")
    if not isinstance(report.get("arms"), Mapping) or set(report["arms"]) != {"A", "B"}:
        raise AssertionError("root smoke report lacks both arms")
    protected = audit.log_cache.get("protected_checks")
    if protected is None:
        protected = audit.log("protected_inputs_checks.json")
    audit.check("root_smoke_report_completed_and_protected", True, {"status": report.get("status"), "protected_checks_all_true": bool(protected) and all(value is True for value in protected.values())})


def audit_smoke_artifacts(data_root: str | Path, log_root: str | Path) -> dict[str, Any]:
    """Audit a completed smoke artifact tree without mutating it.

    Missing or malformed artifacts are returned as failed checks with readable
    diagnostics.  No environment step, optimizer update, teacher-data import,
    checkpoint save, or output-file write is performed here.
    """

    audit = _Audit(Path(data_root), Path(log_root))
    _check_roots(audit)
    if audit.data_root.is_dir() and audit.log_root.is_dir():
        audit.guarded("curriculum_manifest", lambda: _check_curriculum(audit))
        audit.guarded("config_and_resolved_contract", lambda: _check_manifest(audit))
        audit.guarded("calibration_train_only_288_paired", lambda: _check_calibration(audit))
        audit.guarded("shared_train_statistics_finite_positive_train_only", lambda: _check_shared_statistics(audit))
        audit.guarded("protected_inputs_all_true", lambda: _check_protected(audit))
        for group in ("A", "B"):
            audit.guarded(f"{group}_initialization_fresh_modules_optimizer_rollout", lambda group=group: _check_initialization(audit, group))
            audit.guarded(f"{group}_direction_slots_and_reset_probe", lambda group=group: _check_actor_contract(audit, group))
        audit.guarded("ppo_updates_18x2_8_optimizer_steps_finite_positive", lambda: _check_updates(audit))
        for group in ("A", "B"):
            audit.guarded(f"{group}_online_2160_validation_cost_and_stage3_metrics", lambda group=group: _check_validation_and_steps(audit, group))
            audit.guarded(f"{group}_sampled_execution_join_and_correction_accounting", lambda group=group: _check_action_log(audit, group))
            audit.guarded(f"{group}_final_checkpoint_metadata_and_parameter_deltas", lambda group=group: _check_checkpoints(audit, group))
        audit.guarded("root_smoke_report_completed_and_protected", lambda: _check_root_report(audit))
    else:
        audit.check("artifacts_available", False, "data_root and log_root must both exist before artifact checks can run")
    checks = dict(audit.results)
    passed_check_count = sum(bool(value) for value in checks.values())
    return {
        "schema_version": "single_uav_static_ppo_smoke_audit_v1",
        "data_root": str(audit.data_root),
        "log_root": str(audit.log_root),
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "check_details": audit.details,
        "failures": audit.failures,
        "diagnostics": audit.diagnostics,
        "arms": audit.arm_reports,
        "final_metrics": {group: report.get("final_metrics") for group, report in audit.arm_reports.items()},
        "actual_cost_audit": audit.costs,
        "failure_count": len(audit.failures),
        "check_count": len(checks),
        "passed_check_count": passed_check_count,
        "read_only_audit_passed_count": passed_check_count,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("log_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_smoke_artifacts(args.data_root, args.log_root), ensure_ascii=False, indent=2, default=str))


__all__ = ["audit_smoke_artifacts", "parse_vector", "recompute_episode_metrics", "total_correction"]


if __name__ == "__main__":
    main()
