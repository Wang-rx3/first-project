"""Pre-registered, read-only cooperative-EKF reliability diagnostic (v5).

The module is intentionally a runner rather than a training entry point.  It
imports the frozen v4 simulator and the pure v3 selection contract, injects the
locked six-dimensional initial errors into fresh local filters, and records
closed-loop navigation, EKF, message, and censoring diagnostics.  Importing
this module does not construct an environment or a learning model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..control.fixed_navigation_execution import compose_fixed_navigation_actions
from ..simulation.environment import MultiUAVJointEnv
from ..simulation.random_streams import indexed_rng
from . import three_uav_measurement_selection as v3
from . import three_uav_measurement_selection_v4 as v4
from .three_uav_fixed_navigation_contract import StrictParkingTracker, joint_closest_approach


ROOT = Path(__file__).resolve().parents[2]
STEM = "three_uav_cooperative_ekf_reliability_diag_v5_seed2026_20261012"
CONFIG = ROOT / "data/configs" / (STEM + ".json")
LOCK = ROOT / "data/synthetic" / STEM / "diagnostic_design.lock.json"
LOCK_SHA = ROOT / "data/synthetic" / STEM / "diagnostic_design.sha256.json"
EXECUTION_SOURCES_LOCK = ROOT / "data/synthetic" / STEM / "execution_sources.lock.json"
V4_CONFIG = ROOT / "data/configs" / "three_uav_measurement_learnability_v4_seed2026_20261012.json"
V4_CALIBRATION_MANIFEST = (
    ROOT / "data/synthetic" / "three_uav_measurement_learnability_v4_seed2026_20261012"
    / "calibration_v4_s2026" / "manifest.json"
)
PROTECTED_INPUTS = ROOT / "logs/preliminary" / STEM / "protected_inputs_before.json"
UIDS = tuple(v3.UIDS)
NONE = "none"
DECISION_STEPS = tuple(range(2, 119, 4))
SOURCE_FILES = tuple(dict.fromkeys((*v4.SOURCE_FILES, "src/experiments/three_uav_cooperative_ekf_reliability_v5.py", "src/experiments/stage3_diagnose_cooperative_ekf_v5.py")))


def episode_id(arm: str, case: Mapping[str, Any]) -> str:
    """Stable join key shared by every per-episode table."""

    return f"{arm}_{case['scene_id']}_s{int(case['seed'])}"


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def normalize(value: Any) -> Any:
    """JSON-normalize arrays and tuple-keyed state without losing structure."""

    if isinstance(value, Mapping):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [normalize(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if is_dataclass(value):
        return normalize(asdict(value))
    if hasattr(value, "state") and value.__class__.__module__.startswith("numpy.random"):
        return normalize(value.state)
    return value


def encode(value: Any) -> str:
    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=native)


def content_hash(value: Any) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class DiagnosticEnv(MultiUAVJointEnv):
    """Native environment with a class-level transmission audit hook.

    A method override keeps ``copy.deepcopy`` branch clones isolated; a
    closure assigned to one instance would share mutable logging state.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.transmitted_edge_history: list[list[tuple[str, str]]] = []
        self.transmitted_payload_history: list[dict[str, Any]] = []

    def _enqueue_messages(self, edges: set[tuple[str, str]], payloads: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.transmitted_edge_history.append(sorted((str(source), str(target)) for source, target in edges))
        self.transmitted_payload_history.append(
            {
                str(uid): {"state": np.asarray(payload[0], dtype=float).copy(), "covariance": np.asarray(payload[1], dtype=float).copy()}
                for uid, payload in payloads.items()
            }
        )
        return super()._enqueue_messages(edges, payloads)


class RunOutput:
    """Exclusive output roots with episode-level flushes."""

    def __init__(self, run_id: str, root: str | Path = ROOT) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", str(run_id)):
            raise ValueError("run-id must contain only letters, digits, underscore or hyphen")
        self.run_id = str(run_id)
        self.data = Path(root) / "data/synthetic" / STEM / self.run_id
        self.logs = Path(root) / "logs/preliminary" / STEM / self.run_id
        self.plots = Path(root) / "plots/preliminary" / STEM / self.run_id
        occupied = [str(path) for path in (self.data, self.logs, self.plots) if path.exists()]
        if occupied:
            raise FileExistsError("Refusing to reuse output directories: " + ", ".join(occupied))
        for path in (self.data, self.logs, self.plots):
            path.mkdir(parents=True, exist_ok=False)
        self.tables: dict[str, list[dict[str, Any]]] = {}

    def row(self, table: str, values: Mapping[str, Any]) -> None:
        self.tables.setdefault(str(table), []).append(copy.deepcopy(dict(values)))

    def flush(self) -> None:
        for table, rows in self.tables.items():
            if not rows:
                continue
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (self.logs / (table + ".csv")).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: encode(value) if isinstance(value, (Mapping, list, tuple, np.ndarray)) else (None if isinstance(value, float) and not math.isfinite(value) else value) for key, value in row.items()})


def source_manifest() -> dict[str, str]:
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_frozen_config(spec: Mapping[str, Any], *, config_path: str | Path = CONFIG) -> None:
    """Validate both the v5 lock and the human v4 source/calibration chain."""

    _require(isinstance(spec, Mapping), "configuration must be a mapping")
    _require(spec.get("schema") == "cooperative_ekf_reliability_diagnostic_v5", "v5 diagnostic schema is required")
    config_path = Path(config_path)
    _require(file_hash(config_path) == file_hash(LOCK), "configuration differs from the locked diagnostic design")
    sidecar = read_json(LOCK_SHA)
    _require(sidecar.get("path") == LOCK.name and sidecar.get("sha256") == file_hash(LOCK), "design lock SHA sidecar mismatch")
    _require(file_hash(V4_CONFIG) == str(spec.get("base_config_sha256")), "base v4 configuration SHA mismatch")
    v4_spec = read_json(V4_CONFIG)
    v4.validate_frozen_config(v4_spec)
    manifest = read_json(V4_CALIBRATION_MANIFEST)
    _require(manifest.get("sources") == v4.source_manifest(), "human v4 source manifest mismatch")
    _require(tuple(spec.get("seeds", ())) == (2026116100, 2026116102), "locked seeds mismatch")
    _require(tuple(spec.get("arms", ())) == ("none", "fixed", "random_peer", "greedy"), "locked arms mismatch")
    _require(tuple(spec.get("decision_steps", ())) == DECISION_STEPS, "decision windows must be 2, 6, ..., 118")
    _require(int(spec.get("horizon")) == 120 and int(spec.get("trajectory_transition_cap")) == 1920, "trajectory cap mismatch")
    _require(int(spec.get("branch_transition_cap")) == 25 and int(spec.get("total_transition_cap")) == 1945, "branch cap mismatch")
    _require(spec.get("deployment_changes_allowed") is False and spec.get("scientific_training_ready") is False, "diagnostic must remain blocked")


def validate_execution_sources_lock() -> None:
    """Require the root-owned final source manifest before any transition."""

    _require(EXECUTION_SOURCES_LOCK.exists(), "execution_sources.lock.json must be created before the actual run")
    locked = read_json(EXECUTION_SOURCES_LOCK)
    _require(locked.get("schema") == "cooperative_ekf_reliability_execution_sources_v5", "execution source lock schema mismatch")
    _require(locked.get("sources") == source_manifest(), "execution source manifest mismatch")


def protected_input_hashes() -> dict[str, str]:
    """Return the locked contents of all pre-existing protected inputs."""

    _require(PROTECTED_INPUTS.exists(), "protected_inputs_before.json is required")
    payload = read_json(PROTECTED_INPUTS)
    _require(isinstance(payload, Mapping) and len(payload) == 787, "protected input manifest must contain exactly 787 files")
    result: dict[str, str] = {}
    for raw_path, expected in payload.items():
        path = Path(str(raw_path))
        _require(path.exists() and path.is_file(), f"protected input missing: {path}")
        actual = file_hash(path)
        _require(actual == str(expected), f"protected input hash mismatch: {path}")
        result[str(path)] = actual
    return result


def verify_protected_inputs(expected: Mapping[str, str]) -> None:
    """Re-hash protected file contents, rather than hashing the manifest itself."""

    for raw_path, expected_hash in expected.items():
        path = Path(raw_path)
        _require(path.exists() and file_hash(path) == str(expected_hash), f"protected input changed: {path}")


def effective_simulation(spec: Mapping[str, Any]) -> Any:
    validate_frozen_config(spec)
    return v4.effective_simulation(read_json(V4_CONFIG))


@contextmanager
def _forbid_training_runtime():
    """Extend the frozen v4 guard to normalizer fitting as well."""

    patches: list[tuple[Any, str, Any]] = []

    def forbid(target: Any, name: str) -> None:
        original = getattr(target, name)
        patches.append((target, name, original))

        def fail(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError(f"Forbidden v5 training operation: {name}")

        setattr(target, name, fail)

    # v4 owns policy, optimizer, torch-load, and trainer guards.  These two
    # classes are imported through the v3 contract and are the remaining fit
    # entry points that must fail closed during this diagnostic.
    forbid(v3.ObservationNormalizer, "fit")
    forbid(v3.FeatureNormalizer, "fit")
    try:
        with v4._forbid_training_runtime():
            yield
    finally:
        for target, name, original in reversed(patches):
            setattr(target, name, original)


def _initial_covariance_check(env: DiagnosticEnv) -> None:
    for uid in UIDS:
        covariance = np.asarray(env.ekfs[uid].covariance, dtype=float)
        _require(np.allclose(covariance[:3, :3], 4.0 * np.eye(3)), f"{uid} position covariance is not 4I")
        _require(np.allclose(covariance[3:6, 3:6], np.eye(3)), f"{uid} velocity covariance is not I")
        _require(np.isfinite(covariance).all(), f"{uid} covariance is nonfinite")


def prepare_environment(case: Mapping[str, Any], sim: Any, spec: Mapping[str, Any]) -> tuple[DiagnosticEnv, dict[str, dict[str, Any]], dict[str, Any]]:
    """Construct/reset one episode and inject only the locked own-filter error."""

    env = DiagnosticEnv(sim)
    obs_stale, info = env.reset(seed=int(case["seed"]), scenario_config=case["scenario"])
    del obs_stale
    _initial_covariance_check(env)
    _require(not env.pending_messages and not env.arrived_messages, "initial error injection requires empty message buffers")
    for uid in UIDS:
        _require(np.allclose(env.ekfs[uid].covariance[:3, 3:6], 0.0), "initial position/velocity cross covariance must be zero")
    errors = spec["initial_errors_by_seed"][str(int(case["seed"]))]
    for uid in UIDS:
        error = np.asarray(errors[uid], dtype=float).reshape(6)
        _require(np.isfinite(error).all(), f"nonfinite locked initial error for {uid}")
        env.ekfs[uid].state[:6] += error
    env.graph_batch = env._build_graph_batch()
    obs = env._observations()
    for uid in UIDS:
        expected = np.asarray(env.states[uid].position_m.tolist() + env.states[uid].velocity_mps.tolist()) + np.asarray(errors[uid], dtype=float)
        _require(np.allclose(np.asarray(obs[uid]["estimate"][:6], dtype=float), expected), f"initial estimate injection mismatch for {uid}")
    return env, obs, info


def _own(local: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = local.get("local") if isinstance(local, Mapping) else None
    return nested if isinstance(nested, Mapping) else local


def legal_sources(local: Mapping[str, Any]) -> tuple[str, ...]:
    """Use native v3 legality, including masks, message validity, budget, and timing."""

    return tuple(v3.contract()._legal_sources(_own(local), apply_base_mask=True))


def select_source(arm: str, local: Mapping[str, Any], *, step: int, env_seed: int, credit_scale: float, range_variance_m2: float) -> str:
    """Pure arm selection; no truth, future outcome, or neural policy input."""

    uid = str(_own(local).get("uav_id", ""))
    legal = legal_sources(local)
    if arm == "none":
        return NONE
    if arm == "fixed":
        preferred = next((candidate for candidate in UIDS if candidate != uid), NONE)
        return preferred if preferred in legal else NONE
    if arm == "random_peer":
        if not legal:
            return NONE
        rng = indexed_rng(int(env_seed), int(step), "diag_v5_random_peer", uid)
        return str(rng.choice(np.asarray(legal, dtype=object)))
    if arm == "greedy":
        return str(v3.contract().greedy_selection(_own(local), float(credit_scale), float(range_variance_m2)))
    raise ValueError(f"unknown arm {arm}")


def selection_masks(local: Mapping[str, Any], selected: str) -> dict[str, Any]:
    view = _own(local)
    native_mask = np.asarray(view.get("native_action_mask", view.get("action_mask", np.zeros(4, dtype=bool))), dtype=bool).reshape(-1)
    requested_mask = np.zeros(4, dtype=bool)
    if selected in UIDS:
        requested_mask[UIDS.index(selected)] = True
    requested_mask[3] = selected == NONE
    return {"native": native_mask.tolist(), "selected": requested_mask.tolist(), "legal": list(legal_sources(view))}


def _array_metric(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nees(error: Any, covariance: Any) -> float:
    vector = np.asarray(error, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (vector.size, vector.size) or not np.isfinite(vector).all() or not np.isfinite(matrix).all():
        return float("nan")
    matrix = 0.5 * (matrix + matrix.T)
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-8:
        return float("nan")
    return float(vector @ np.linalg.pinv(matrix) @ vector)


def _candidate_geometry(
    local: Mapping[str, Any],
    source: str,
    *,
    message: Any = None,
    current_time_s: float | None = None,
) -> dict[str, Any]:
    """Summarize observable estimate/covariance geometry for candidate logs."""

    view = _own(local)
    own = np.asarray(view.get("estimate", np.full(15, np.nan)), dtype=float).reshape(-1)
    messages = view.get("neighbor_messages", {})
    embedded_message = messages.get(source, {}) if isinstance(messages, Mapping) else {}
    other = np.asarray(embedded_message.get("estimate", np.full(15, np.nan)), dtype=float).reshape(-1) if isinstance(embedded_message, Mapping) else np.full(15, np.nan)
    own_cov = np.asarray(view.get("covariance", np.full((15, 15), np.nan)), dtype=float)
    other_cov = np.asarray(embedded_message.get("covariance", np.full((15, 15), np.nan)), dtype=float) if isinstance(embedded_message, Mapping) else np.full((15, 15), np.nan)
    relative = other[:3] - own[:3] if own.size >= 3 and other.size >= 3 else np.full(3, np.nan)
    norm = float(np.linalg.norm(relative)) if np.isfinite(relative).all() else float("nan")
    direction = relative / norm if math.isfinite(norm) and norm > 1e-12 else np.full(3, np.nan)
    own_pos = own_cov[:3, :3] if own_cov.shape == (15, 15) else np.full((3, 3), np.nan)
    other_pos = other_cov[:3, :3] if other_cov.shape == (15, 15) else np.full((3, 3), np.nan)
    gram = float(direction @ own_pos @ direction) if np.isfinite(direction).all() and np.isfinite(own_pos).all() else float("nan")
    cross = float(direction @ (own_pos + other_pos) @ direction) if np.isfinite(direction).all() and np.isfinite(own_pos).all() and np.isfinite(other_pos).all() else float("nan")
    current_time = _number_or_none(current_time_s)
    if current_time is None:
        current_time = _number_or_none(view.get("current_time_s", view.get("timestamp_s", view.get("time_s"))))
    message_time = _number_or_none(
        getattr(message, "send_time_s", None)
        if message is not None and not isinstance(message, Mapping)
        else (message.get("send_time_s", message.get("timestamp_s", message.get("timestamp"))) if isinstance(message, Mapping) else None)
    )
    age = max(0.0, current_time - message_time) if current_time is not None and message_time is not None else None
    return {"source_uid": source, "relative_vector_estimate": relative, "relative_unit_direction": direction, "relative_norm_m": _number_or_none(norm), "gram_directional_variance_m2": _number_or_none(gram), "neighbor_position_trace_m2": _number_or_none(np.trace(other_pos)) if np.isfinite(other_pos).all() else None, "combined_directional_variance_m2": _number_or_none(cross), "message_send_time_s": message_time, "message_age_s": age}


def candidate_angle_cosine(local: Mapping[str, Any], first: str = "uav_01", second: str = "uav_02") -> float | None:
    """Dot product of the two observable relative unit directions (B/C)."""

    first_geometry = _candidate_geometry(local, first)
    second_geometry = _candidate_geometry(local, second)
    first_direction = np.asarray(first_geometry["relative_unit_direction"], dtype=float)
    second_direction = np.asarray(second_geometry["relative_unit_direction"], dtype=float)
    if first_direction.shape != (3,) or second_direction.shape != (3,) or not np.isfinite(first_direction).all() or not np.isfinite(second_direction).all():
        return None
    return _number_or_none(np.dot(first_direction, second_direction))


def _state_hash(env: DiagnosticEnv, *, sender_payloads: Any = None) -> str:
    state = {
        "states": {uid: {"position": env.states[uid].position_m, "velocity": env.states[uid].velocity_mps, "attitude": env.states[uid].attitude_rad, "acceleration": env.states[uid].acceleration_mps2} for uid in UIDS},
        "goals": env.goals,
        "ekfs": {uid: {"state": env.ekfs[uid].state, "covariance": env.ekfs[uid].covariance} for uid in UIDS},
        "queue_state": env.queue_state,
        "aoi_state": env.aoi_state,
        "pending_messages": env.pending_messages,
        "arrived_messages": env.arrived_messages,
        "snapshot": env.snapshot,
        "timestep": env.timestep,
        "episode_random_seed": getattr(env, "_episode_random_seed", None),
        "topology_draw_index": getattr(env, "_topology_draw_index", None),
        "rng_state": env.rng.bit_generator.state,
        "config": env.config,
        "sender_payloads": sender_payloads,
    }
    return content_hash(state)


def capture_snapshot(env: DiagnosticEnv, raw_obs: Mapping[str, Any], *, step: int, source_uids: Sequence[str], actions: Mapping[str, Any], sender_payloads: Any) -> dict[str, Any]:
    """Capture enough state for a replayable one-transition branch."""

    pre_hash = _state_hash(env, sender_payloads=sender_payloads)
    return {
        "step": int(step),
        "source_uids": list(source_uids),
        "raw_observations": copy.deepcopy(raw_obs),
        "actions_none": copy.deepcopy(actions),
        "sender_payloads": copy.deepcopy(sender_payloads),
        "pre_hash": pre_hash,
        "env": copy.deepcopy(env),
    }


def select_branch_windows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return at most the earliest two legal-peer windows in trajectory order."""

    found: list[Mapping[str, Any]] = []
    for row in records:
        if int(row.get("step", -1)) not in DECISION_STEPS or str(row.get("uav_id", "")) != "uav_00":
            continue
        legal = tuple(row.get("legal_sources", ()))
        if len(legal) == 2:
            found.append(row)
        if len(found) == 2:
            break
    return found


def trajectory_summary(rows: Sequence[Mapping[str, Any]], *, horizon: int, censored: bool, censor_reason: str | None = None) -> dict[str, Any]:
    """Aggregate closed-loop metrics with explicit valid denominators."""

    valid = [row for row in rows if bool(row.get("completed_transition", False))]
    valid_steps = sorted({int(row["step"]) for row in valid if row.get("step") is not None})
    result: dict[str, Any] = {"observed_steps": len(valid_steps), "valid_transition_denominator": len(valid_steps), "horizon": int(horizon), "censored": bool(censored), "censor_reason": censor_reason, "censor_step": int(valid_steps[-1]) if censored and valid_steps else None}
    for uid in UIDS:
        uid_rows = [row for row in valid if row.get("uav_id") == uid]
        pos_sq = [float(row["position_error_m"]) ** 2 for row in uid_rows if math.isfinite(float(row["position_error_m"]))]
        vel_sq = [float(row["velocity_error_mps"]) ** 2 for row in uid_rows if math.isfinite(float(row["velocity_error_mps"]))]
        nees3 = [float(row["nees_position_3d"]) for row in uid_rows if math.isfinite(float(row["nees_position_3d"]))]
        nees6 = [float(row["nees_position_velocity_6d"]) for row in uid_rows if math.isfinite(float(row["nees_position_velocity_6d"]))]
        result[uid] = {"position_rmse_m": math.sqrt(float(np.mean(pos_sq))) if pos_sq else None, "velocity_rmse_mps": math.sqrt(float(np.mean(vel_sq))) if vel_sq else None, "position_denominator": len(pos_sq), "velocity_denominator": len(vel_sq), "position_bias_xyz_m": np.mean([row["position_error_vector_m"] for row in uid_rows], axis=0) if uid_rows else None, "nees_position_mean": float(np.mean(nees3)) if nees3 else None, "nees_position_median": float(np.median(nees3)) if nees3 else None, "nees_position_upper95": float(np.quantile(nees3, .95)) if nees3 else None, "nees_position_reference_coverage": float(np.mean(np.asarray(nees3) <= 7.8147)) if nees3 else None, "nees_position_velocity_mean": float(np.mean(nees6)) if nees6 else None, "vo_nis_attempted": int(sum(row.get("vo_nis_attempted", 0) for row in uid_rows)), "vo_nis_accepted": int(sum(row.get("vo_nis_accepted", 0) for row in uid_rows)), "uwb_nis_attempted": int(sum(row.get("uwb_nis_attempted", 0) for row in uid_rows)), "uwb_nis_accepted": int(sum(row.get("uwb_nis_accepted", 0) for row in uid_rows))}
    pair_rows = [row for row in valid if row.get("uav_id") == "team"]
    if pair_rows:
        result["relative_pair_vector_rmse_m"] = float(np.sqrt(np.mean(np.asarray([row["relative_pair_squared_error_m2"] for row in pair_rows], dtype=float))))
        result["relative_pair_range_rmse_m"] = float(np.sqrt(np.mean(np.asarray([row["relative_pair_range_error_m2"] for row in pair_rows], dtype=float))))
    else:
        result["relative_pair_vector_rmse_m"] = None
        result["relative_pair_range_rmse_m"] = None
    return result


def paired_post_improvement(none_post_error: float, branch_post_error: float) -> float:
    """Causal paired contrast: NONE post-transition error minus branch post error."""

    return float(none_post_error) - float(branch_post_error)


def _record_localization(env: DiagnosticEnv, info: Mapping[str, Any], step: int, pre_velocities: Mapping[str, Any], dt_s: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid in UIDS:
        truth = np.asarray(env.states[uid].position_m, dtype=float)
        truth_velocity = np.asarray(env.states[uid].velocity_mps, dtype=float)
        estimate = np.asarray(env.ekfs[uid].state, dtype=float)
        covariance = np.asarray(env.ekfs[uid].covariance, dtype=float)
        diagnostics = info.get("ekf_diagnostics", {}).get(uid, {}) if isinstance(info.get("ekf_diagnostics"), Mapping) else {}
        execution = info.get("action_execution", {}).get(uid, {}) if isinstance(info.get("action_execution"), Mapping) else {}
        position_covariance = 0.5 * (covariance[:3, :3] + covariance[:3, :3].T)
        actual_accel = (truth_velocity - np.asarray(pre_velocities[uid], dtype=float)) / max(float(dt_s), 1e-12)
        row = {"step": int(step), "uav_id": uid, "completed_transition": True, "estimate_state": estimate.copy(), "true_position_m": truth.copy(), "true_velocity_mps": truth_velocity.copy(), "position_error_vector_m": estimate[:3] - truth, "position_error_m": float(np.linalg.norm(estimate[:3] - truth)), "position_error_norm_m": float(np.linalg.norm(estimate[:3] - truth)), "velocity_error_vector_mps": estimate[3:6] - truth_velocity, "velocity_error_mps": float(np.linalg.norm(estimate[3:6] - truth_velocity)), "nees_position_3d": nees(estimate[:3] - truth, position_covariance), "nees_position_velocity_6d": nees(estimate[:6] - np.concatenate([truth, truth_velocity]), covariance[:6, :6]), "position_covariance_eigenvalues": np.linalg.eigvalsh(position_covariance), "reported_position_covariance_trace_m2": float(np.trace(position_covariance)), "position_covariance_trace_m2": float(np.trace(position_covariance)), "velocity_covariance_trace_m2": float(np.trace(covariance[3:6, 3:6])), "actual_speed_mps": float(np.linalg.norm(truth_velocity)), "actual_acceleration_mps2": float(np.linalg.norm(actual_accel)), "actual_acceleration_vector_mps2": actual_accel, "actual_delta_v": np.asarray(execution.get("delta_v_applied", np.zeros(3)) if isinstance(execution, Mapping) else np.zeros(3), dtype=float), "boundary_projected": bool(execution.get("boundary_projected", False)), "safety_projected": bool(execution.get("safety_projected", False)), "safety_clip": bool(execution.get("action_clipped", False)), "goal_distance_m": float(np.linalg.norm(truth - np.asarray(env.goals[uid], dtype=float))), "vo_nis_attempted": int(diagnostics.get("vo_update_attempted", 0)), "vo_nis_accepted": int(diagnostics.get("vo_update_accepted", 0)), "vo_nis_rejected": int(diagnostics.get("vo_update_rejected", 0)), "uwb_nis_attempted": int(diagnostics.get("uwb_update_attempted", 0)), "uwb_nis_accepted": int(diagnostics.get("uwb_update_accepted", 0)), "uwb_nis_rejected": int(diagnostics.get("uwb_update_rejected", 0)), "vo_nis": _number_or_none(diagnostics.get("nis_vo", -1.0)), "uwb_nis": _number_or_none(diagnostics.get("nis_uwb_mean", -1.0)), "ekf_diagnostics": diagnostics}
        rows.append(row)
    pair_errors = []
    pair_ranges = []
    for left_index, left in enumerate(UIDS):
        for right in UIDS[left_index + 1 :]:
            estimate_left = np.asarray(env.ekfs[left].state[:3], dtype=float)
            estimate_right = np.asarray(env.ekfs[right].state[:3], dtype=float)
            truth_left = np.asarray(env.states[left].position_m, dtype=float)
            truth_right = np.asarray(env.states[right].position_m, dtype=float)
            estimate_relative = estimate_right - estimate_left
            true_relative = truth_right - truth_left
            pair_errors.append(float(np.sum((estimate_relative - true_relative) ** 2)))
            pair_ranges.append(float((np.linalg.norm(estimate_relative) - np.linalg.norm(true_relative)) ** 2))
    rows.append({"step": int(step), "uav_id": "team", "completed_transition": True, "relative_pair_squared_error_m2": float(np.mean(pair_errors)), "relative_pair_range_error_m2": float(np.mean(pair_ranges))})
    return rows


def _fixed_actions(masked_obs: Mapping[str, Any], selections: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    communication = v3.contract().communication_actions(masked_obs, selections)
    return compose_fixed_navigation_actions(masked_obs, communication)


def _episode(output: RunOutput, case: Mapping[str, Any], arm: str, sim: Any, spec: Mapping[str, Any], *, branch_capture: bool = True, counters: dict[str, int] | None = None) -> dict[str, Any]:
    env, obs, reset_info = prepare_environment(case, sim, spec)
    scale = float(spec["credit_scale"])
    range_variance = float(read_json(V4_CONFIG)["range_variance_m2"])
    rows: list[dict[str, Any]] = []
    localization: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    tracker = StrictParkingTracker(UIDS)
    minimum_pair_distance = float("inf")
    censor_reason: str | None = None
    censored = False
    for step in range(1, int(spec["horizon"]) + 1):
        if counters is not None:
            _require(counters["trajectory_steps"] < int(spec["trajectory_transition_cap"]), "trajectory transition cap exceeded before step")
            _require(counters["total_steps"] < int(spec["total_transition_cap"]), "total transition cap exceeded before step")
        masked = v4.mask_observations(obs, step, spec["decision_steps"])
        selections: dict[str, str] = {}
        candidate_rows: list[dict[str, Any]] = []
        for uid in UIDS:
            local = masked[uid]
            selected = select_source(arm, local, step=step, env_seed=int(case["seed"]), credit_scale=scale, range_variance_m2=range_variance) if step in DECISION_STEPS else NONE
            selections[uid] = selected
            legal = legal_sources(local)
            geometries: list[dict[str, Any]] = []
            for source in UIDS:
                if source == uid or source not in legal:
                    continue
                message = env.arrived_messages.get((source, uid))
                geometries.append(
                    _candidate_geometry(
                        local,
                        source,
                        message=message,
                        current_time_s=(env.timestep + 1) * float(sim.motion.dt_s),
                    )
                )
            candidate_rows.append({"episode_id": episode_id(arm, case), "step": step, "uav_id": uid, "arm": arm, "selected_source": selected, "legal_sources": legal, "selection_masks": selection_masks(local, selected), "candidate_geometry": geometries, "candidate_angle_cosine_bc": candidate_angle_cosine(local) if uid == "uav_00" else None, "expected_raw_scores": {source: v3.contract().expected_relative_credit(_own(local), source, range_variance) for source in UIDS if source != uid and source in legal}, "greedy_scores": v3.contract().greedy_scores(_own(local), scale, range_variance) if arm == "greedy" else {}})
            output.row("selection", candidate_rows[-1])
        communication = v3.contract().communication_actions(masked, selections)
        actions, navigation = _fixed_actions(masked, selections)
        if branch_capture and arm == "none" and step in DECISION_STEPS:
            legal = legal_sources(masked["uav_00"])
            if len(legal) == 2 and len(snapshots) < 2:
                sender_payloads = {uid: (env.ekfs[uid].state.copy(), env.ekfs[uid].covariance.copy()) for uid in UIDS}
                snapshots.append(capture_snapshot(env, masked, step=step, source_uids=legal, actions=actions, sender_payloads=sender_payloads))
        budget_pre = {uid: masked[uid].get("measurement_budget_remaining", masked[uid].get("current_budget_remaining", float("nan"))) for uid in UIDS}
        pre_positions = {uid: env.states[uid].position_m.copy() for uid in UIDS}
        pre_velocities = {uid: env.states[uid].velocity_mps.copy() for uid in UIDS}
        pre_estimates = {uid: env.ekfs[uid].state.copy() for uid in UIDS}
        pre_covariances = {uid: env.ekfs[uid].covariance.copy() for uid in UIDS}
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        if counters is not None:
            counters["trajectory_steps"] += 1
            counters["total_steps"] += 1
        step_rewards, step_reward_terms = v3.contract().reward_components(info, scale)
        block_id = f"{case['scene_id']}_s{int(case['seed'])}"
        overhead_events = info.get("measurement_overhead_events", [])
        physical_packets = sum(float(event.get("load_packets_applied", 0.0)) for event in overhead_events if isinstance(event, Mapping))
        transition_row = {"block_id": block_id, "episode_id": episode_id(arm, case), "scene_id": case["scene_id"], "seed": int(case["seed"]), "step": step, "arm": arm, "selected": selections, "requested": {uid: selections[uid] != NONE for uid in UIDS}, "budget_pre": budget_pre, "budget_post": copy.deepcopy(info.get("measurement_budget_remaining", {})), "executed_edges": env.transmitted_edge_history[-1] if env.transmitted_edge_history else [], "measurement_gate_events": info.get("measurement_gate_events", []), "measurement_overhead_events": overhead_events, "measurement_overhead_packet_count": physical_packets, "measurement_overhead_bytes_equivalent": physical_packets * 512.0, "measurement_budget_events": info.get("measurement_budget_events", []), "received_messages": info.get("message_buffer", {}).get("events", []) if isinstance(info.get("message_buffer"), Mapping) else [], "ordinary_state_tx_count": len(env.transmitted_edge_history[-1]) if env.transmitted_edge_history else 0, "ordinary_state_tx_edges": env.transmitted_edge_history[-1] if env.transmitted_edge_history else [], "actual_rewards": rewards, "reward_components": step_reward_terms, "terminated": bool(terminated), "truncated": bool(truncated), "minimum_clearance_m": info.get("minimum_clearance"), "sensor_packet_hash": content_hash(info.get("sensor_packets", {})), "topology_uwb_hash": content_hash(getattr(info.get("topology"), "uwb_measurements", info.get("topology", {})))}
        rows.append(transition_row)
        output.row("transitions", transition_row)
        post_positions = {uid: env.states[uid].position_m.copy() for uid in UIDS}
        minimum_pair_distance = min(minimum_pair_distance, min(item["distance_m"] for item in joint_closest_approach(pre_positions, post_positions)))
        localization.extend(_record_localization(env, info, step, pre_velocities, float(sim.motion.dt_s)))
        lower = np.asarray(sim.motion.bounds_min_m, dtype=float)
        upper = np.asarray(sim.motion.bounds_max_m, dtype=float)
        bounds_ok = all(np.isfinite(env.states[uid].position_m).all() and np.all(env.states[uid].position_m >= lower - 1e-9) and np.all(env.states[uid].position_m <= upper + 1e-9) for uid in UIDS)
        speed_ok = all(np.isfinite(env.states[uid].velocity_mps).all() and np.linalg.norm(env.states[uid].velocity_mps) <= float(sim.motion.max_speed_mps) + 1e-8 for uid in UIDS)
        acceleration_ok = all(np.isfinite(env.states[uid].velocity_mps).all() and np.linalg.norm((env.states[uid].velocity_mps - pre_velocities[uid]) / max(float(sim.motion.dt_s), 1e-12)) <= float(sim.motion.max_acceleration_mps2) + 1e-7 for uid in UIDS)
        pair_clearance_ok = minimum_pair_distance >= float(sim.motion.safety_distance_m) - 1e-8
        transition_row.update({"bounds_ok": bounds_ok, "speed_limit_ok": speed_ok, "acceleration_limit_ok": acceleration_ok, "minimum_pair_distance_m": minimum_pair_distance, "pair_clearance_ok": pair_clearance_ok, "completed_transition": True})
        output.tables["transitions"][-1].update(copy.deepcopy(transition_row))
        parking = tracker.update({uid: float(np.linalg.norm(env.states[uid].position_m - env.goals[uid])) for uid in UIDS}, {uid: float(np.linalg.norm(env.states[uid].velocity_mps)) for uid in UIDS}, collision=bool(info.get("collision", False)))
        block_id = f"{case['scene_id']}_s{int(case['seed'])}"
        for row in localization[-4:]:
            row.update({"block_id": block_id, "episode_id": episode_id(arm, case), "arm": arm, "strict_parking_hold_steps": parking["per_uav"].get(row.get("uav_id"), {}).get("hold_steps", 0) if row.get("uav_id") in UIDS else parking["joint_hold_steps"], "strict_parking_ever_success": parking["per_uav"].get(row.get("uav_id"), {}).get("ever_success", False) if row.get("uav_id") in UIDS else parking["joint_ever_success"], "collision": bool(info.get("collision", False)), "path_step_dt_s": float(sim.motion.dt_s)})
            output.row("localization", row)
        for event in info.get("relative_range_update_events", []):
            raw_before = float(event.get("position_covariance_trace_before_m2", float("nan")))
            raw_mid = float(event.get("position_covariance_trace_after_linear_m2", float("nan")))
            _require("position_covariance_trace_after_final_m2" in event, "native accepted update is missing final covariance trace")
            raw_final = float(event["position_covariance_trace_after_final_m2"])
            target_uid = str(event.get("target_uav_id", event.get("uav_id", "")))
            reward_terms = step_reward_terms.get(target_uid, {})
            before_error = float(np.linalg.norm(pre_estimates[target_uid][:3] - pre_positions[target_uid])) if target_uid in pre_estimates else None
            after_error = float(np.linalg.norm(env.ekfs[target_uid].state[:3] - env.states[target_uid].position_m)) if target_uid in env.ekfs else None
            output.row("relative_update", {"block_id": block_id, "episode_id": episode_id(arm, case), "step": step, "arm": arm, "event": event, "accepted": bool(event.get("accepted", False)), "raw_relative_linear_credit": v3.contract().relative_linear_credit(event), "raw_before_trace": raw_before, "raw_mid_trace": raw_mid, "raw_final_trace": raw_final, "true_position_error_before_m": before_error, "true_position_error_after_m": after_error, "reward_terms": reward_terms, "scaled_credit": reward_terms.get("scaled_credit_proxy"), "credit_clip_flags": {"scaled_credit_clipped": bool(reward_terms.get("scaled_credit_proxy", 0.0) in (0.0, 1.0)), "raw_credit_clipped": bool(reward_terms.get("raw_relative_credit_proxy", 0.0) >= 1.0)}, "source_error_label": "truth_diagnostic_only"})
        nonfinite = not all(np.isfinite(np.asarray(env.ekfs[uid].state)).all() and np.isfinite(np.asarray(env.ekfs[uid].covariance)).all() for uid in UIDS)
        hard_violation = (not bounds_ok) or (not speed_ok) or (not acceleration_ok) or (not pair_clearance_ok) or any(float(row.get("position_error_m", 0.0)) > 1e6 for row in localization[-4:] if row.get("uav_id") in UIDS)
        if bool(info.get("collision", False)) or bool(terminated) or nonfinite or hard_violation:
            censor_reason = "collision" if bool(info.get("collision", False)) else str(info.get("termination_reason", "terminated")) if bool(terminated) else "nonfinite" if nonfinite else "hard_limit"
            censored = True
            if step < int(spec["horizon"]):
                break
        obs = next_obs
        if step % 30 == 0 or step == 1 or terminated or truncated:
            print(f"episode={arm}/{case['scene_id']}/{case['seed']} step={step} selected={sum(v != NONE for v in selections.values())} accepted={sum(bool(e.get('accepted', False)) for e in info.get('relative_range_update_events', []))}", flush=True)
    for row in rows:
        row["completed_transition"] = True
    summary = trajectory_summary(localization, horizon=int(spec["horizon"]), censored=censored, censor_reason=censor_reason)
    result = {"episode_id": episode_id(arm, case), "scene_id": case["scene_id"], "seed": int(case["seed"]), "arm": arm, "summary": summary, "steps": len(rows), "censored": censored, "censor_reason": censor_reason, "minimum_pair_distance_m": minimum_pair_distance, "snapshot_count": len(snapshots)}
    output.row("episodes", result)
    output.flush()
    output.row("episode_metrics", {"block_id": f"{case['scene_id']}_s{int(case['seed'])}", "episode_id": result["episode_id"], "scene_id": case["scene_id"], "seed": int(case["seed"]), "arm": arm, "observed_steps": summary["observed_steps"], "valid_transition_denominator": summary["valid_transition_denominator"], "censored": censored, "censor_reason": censor_reason, "minimum_pair_distance_m": minimum_pair_distance, "position_rmse_by_uid": {uid: summary[uid]["position_rmse_m"] for uid in UIDS}, "velocity_rmse_by_uid": {uid: summary[uid]["velocity_rmse_mps"] for uid in UIDS}, "position_bias_by_uid": {uid: summary[uid]["position_bias_xyz_m"] for uid in UIDS}, "nees_position_mean_by_uid": {uid: summary[uid]["nees_position_mean"] for uid in UIDS}, "nees_position_upper95_by_uid": {uid: summary[uid]["nees_position_upper95"] for uid in UIDS}, "vo_nis_attempted_by_uid": {uid: summary[uid]["vo_nis_attempted"] for uid in UIDS}, "vo_nis_accepted_by_uid": {uid: summary[uid]["vo_nis_accepted"] for uid in UIDS}, "uwb_nis_attempted_by_uid": {uid: summary[uid]["uwb_nis_attempted"] for uid in UIDS}, "uwb_nis_accepted_by_uid": {uid: summary[uid]["uwb_nis_accepted"] for uid in UIDS}})
    return {"result": result, "snapshots": snapshots, "localization": localization, "rows": rows}


def branch_replay(output: RunOutput, snapshot: Mapping[str, Any], case: Mapping[str, Any], sim: Any, spec: Mapping[str, Any], branch_id: str, target: str, *, counters: dict[str, int] | None = None) -> dict[str, Any]:
    clone: DiagnosticEnv = copy.deepcopy(snapshot["env"])
    raw = copy.deepcopy(snapshot["raw_observations"])
    if counters is not None:
        _require(counters["branch_steps"] < int(spec["branch_transition_cap"]), "branch transition cap exceeded before step")
        _require(counters["total_steps"] < int(spec["total_transition_cap"]), "total transition cap exceeded before step")
    pre_hash = _state_hash(clone, sender_payloads=snapshot.get("sender_payloads"))
    _require(pre_hash == snapshot["pre_hash"], "branch pre-clone hash mismatch")
    selections = {uid: NONE for uid in UIDS}
    if target in UIDS:
        selections["uav_00"] = target
    actions, _ = _fixed_actions(raw, selections)
    pre_action_hash = content_hash(actions)
    pre_motion_hash = content_hash({uid: actions[uid].get("delta_v", []) for uid in UIDS})
    pre_bandwidth_hash = content_hash({uid: actions[uid].get("bandwidth", {}) for uid in UIDS})
    pre_positions = {uid: clone.states[uid].position_m.copy() for uid in UIDS}
    pre_errors = {uid: float(np.linalg.norm(clone.ekfs[uid].state[:3] - clone.states[uid].position_m)) for uid in UIDS}
    pre_nees = {uid: nees(clone.ekfs[uid].state[:3] - clone.states[uid].position_m, clone.ekfs[uid].covariance[:3, :3]) for uid in UIDS}
    next_obs, _, terminated, truncated, info = clone.step(actions)
    if counters is not None:
        counters["branch_steps"] += 1
        counters["total_steps"] += 1
    post_positions = {uid: clone.states[uid].position_m.copy() for uid in UIDS}
    post_errors = {uid: float(np.linalg.norm(clone.ekfs[uid].state[:3] - clone.states[uid].position_m)) for uid in UIDS}
    post_nees = {uid: nees(clone.ekfs[uid].state[:3] - clone.states[uid].position_m, clone.ekfs[uid].covariance[:3, :3]) for uid in UIDS}
    sensor_hash = content_hash(info.get("sensor_packets", {}))
    post_hash = _state_hash(clone)
    topology = info.get("topology")
    exogenous_hash = content_hash({"sensor_packets": info.get("sensor_packets", {}), "uwb_measurements": getattr(topology, "uwb_measurements", {})})
    result = {"branch_id": str(branch_id), "target": target, "snapshot_step": int(snapshot["step"]), "scene_id": case["scene_id"], "seed": int(case["seed"]), "block_id": f"{case['scene_id']}_s{int(case['seed'])}", "pre_hash": pre_hash, "pre_action_hash": pre_action_hash, "pre_motion_hash": pre_motion_hash, "pre_bandwidth_hash": pre_bandwidth_hash, "post_hash": post_hash, "sensor_packet_hash": sensor_hash, "exogenous_sensor_uwb_hash": exogenous_hash, "terminated": bool(terminated), "truncated": bool(truncated), "pre_positions": pre_positions, "post_positions": post_positions, "pre_position_error_m": pre_errors, "post_position_error_m": post_errors, "pre_nees_position_3d": pre_nees, "post_nees_position_3d": post_nees, "target_post_position_error_m": post_errors["uav_00"], "team_post_squared_error_m2": float(sum(value * value for value in post_errors.values())), "true_geometric_direction": _candidate_geometry(raw["uav_00"], target) if target in UIDS else {}, "control_other_actions_all_none": True, "source_error_label": "truth_diagnostic_only", "relative_update_events": info.get("relative_range_update_events", []), "cost_budget": info.get("measurement_budget_events", []), "message_delay_unknown_if_absent": True}
    result["estimated_geometric_direction"] = result.pop("true_geometric_direction")
    result["post_beliefs"] = {uid: {"state": clone.ekfs[uid].state.copy(), "covariance": clone.ekfs[uid].covariance.copy()} for uid in UIDS}
    result["cost_overhead"] = copy.deepcopy(info.get("measurement_overhead_events", []))
    result["reward_terms"] = v3.contract().reward_components(info, float(spec["credit_scale"]))[1]
    result["expected_credit_pre"] = v3.contract().expected_relative_credit(raw["uav_00"], target, float(read_json(V4_CONFIG)["range_variance_m2"])) if target in UIDS else 0.0
    transmitted = clone.transmitted_payload_history[-1]
    expected = {uid: {"state": snapshot["sender_payloads"][uid][0], "covariance": snapshot["sender_payloads"][uid][1]} for uid in UIDS}
    _require(content_hash(transmitted) == content_hash(expected), "branch sender payload differs from pre-local-update frozen belief")
    result["sender_payload_hash"] = content_hash(transmitted)
    output.row("branches", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v5 cooperative EKF reliability diagnostic; no training")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", required=True)
    return parser


def run(run_id: str, config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Run the locked diagnostic.  Configuration and output checks precede env construction."""

    spec = read_json(config_path)
    validate_frozen_config(spec, config_path=config_path)
    validate_execution_sources_lock()
    output = RunOutput(run_id)
    counters = {"trajectory_steps": 0, "branch_steps": 0, "total_steps": 0, "training_events": 0, "fit_calls": 0, "optimizer_steps": 0, "checkpoint_loads": 0}
    status = "failed"
    protected_before: dict[str, str] = {}
    try:
        protected_before = protected_input_hashes()
        v4_spec = read_json(V4_CONFIG)
        sim = v4.effective_simulation(v4_spec)
        write_json(output.data / "configuration.json", spec)
        write_json(output.data / "manifest.json", {"schema": spec["schema"], "stem": STEM, "run_id": run_id, "sources": source_manifest(), "config_sha256": file_hash(config_path), "design_lock_sha256": file_hash(LOCK), "protected_input_count": len(protected_before), "scientific_training_ready": False, "deployment_changes_allowed": False})
        snapshot_by_block: dict[tuple[str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        episode_outputs: list[dict[str, Any]] = []
        # The v4 runtime guard rejects all trainer/policy/fit/load/optimizer paths.
        with _forbid_training_runtime():
            for matrix_row in spec["run_matrix"]:
                case = copy.deepcopy(matrix_row)  # full crossed scene/seed row, including its own scenario
                episode = _episode(output, case, str(matrix_row["arm"]), sim, spec, branch_capture=str(matrix_row["arm"]) == "none", counters=counters)
                episode_outputs.append(episode["result"])
                if str(matrix_row["arm"]) == "none":
                    block_key = (str(case["scene_id"]), int(case["seed"]))
                    snapshot_by_block.setdefault(block_key, [])
                    snapshot_by_block[block_key].extend((snapshot, case) for snapshot in episode["snapshots"][:2])
            scene_rank = {"train_compact_x": 0, "train_bridge_x": 1}
            branch_order: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for block_key in sorted(snapshot_by_block, key=lambda item: (scene_rank.get(item[0], 99), item[1])):
                branch_order.extend(snapshot_by_block[block_key])
            branch_rows: list[dict[str, Any]] = []
            for index, (snapshot, case) in enumerate(branch_order):
                for target in ("uav_01", "uav_02", NONE):
                    branch_rows.append(branch_replay(output, snapshot, case, sim, spec, f"branch_{index:02d}_{target}", target, counters=counters))
            if branch_order:
                branch_rows.append(branch_replay(output, branch_order[0][0], branch_order[0][1], sim, spec, "branch_00_uav_01_replay", "uav_01", counters=counters))
        _require(len(branch_rows) <= int(spec["branch_transition_cap"]), "branch transition cap exceeded")
        _require(counters["trajectory_steps"] <= int(spec["trajectory_transition_cap"]), "trajectory transition cap exceeded")
        _require(counters["total_steps"] <= int(spec["total_transition_cap"]), "total transition cap exceeded")
        # Compare each branch trio before any paired causal summary is emitted.
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for row in branch_rows:
            grouped.setdefault((str(row["scene_id"]), int(row["seed"]), int(row["snapshot_step"])), []).append(row)
        for key, rows in grouped.items():
            originals = [row for row in rows if not str(row["branch_id"]).endswith("_replay")]
            if len(originals) == 3:
                _require(len({row["pre_hash"] for row in originals}) == 1, f"branch prestate mismatch: {key}")
                _require(len({row["pre_motion_hash"] for row in originals}) == 1 and len({row["pre_bandwidth_hash"] for row in originals}) == 1, f"branch control/BW mismatch: {key}")
                _require(len({row["exogenous_sensor_uwb_hash"] for row in originals}) == 1, f"branch sensor/UWB hash mismatch: {key}")
                _require(len({encode(row["post_positions"]) for row in originals}) == 1, f"branch true pose mismatch: {key}")
                none_row = next(row for row in originals if row["target"] == NONE)
                for row in originals:
                    if row["target"] in {"uav_01", "uav_02"}:
                        row["improvement_vs_none"] = paired_post_improvement(float(none_row["target_post_position_error_m"]), float(row["target_post_position_error_m"]))
                        row["target_post_error_vs_none"] = float(row["target_post_position_error_m"]) - float(none_row["target_post_position_error_m"])
                        row["team_sse_difference_vs_none"] = float(none_row["team_post_squared_error_m2"]) - float(row["team_post_squared_error_m2"])
                replay = next((row for row in rows if str(row["branch_id"]).endswith("_replay")), None)
                if replay is not None:
                    _require(replay["post_hash"] == next(row for row in originals if row["target"] == "uav_01")["post_hash"], f"duplicate B replay mismatch: {key}")
            for row in rows:
                for stored in output.tables.get("branches", []):
                    if stored.get("branch_id") == row.get("branch_id"):
                        stored.update(copy.deepcopy(row))
        verify_protected_inputs(protected_before)
        report = {"schema": spec["schema"], "stem": STEM, "run_id": run_id, "episodes": episode_outputs, "branch_count": len(branch_rows), "trajectory_transition_count": counters["trajectory_steps"], "branch_transition_count": counters["branch_steps"], "total_transition_count": counters["total_steps"], "scientific_training_ready": False, "deployment_changes_allowed": False, "integrity_ok": True, "protected_inputs_unchanged": True, "counters": counters, "interpretation": "Closed-loop diagnostic only; no training, fit, optimizer, checkpoint, or fusion/reward activation."}
        status = "passed"
        write_json(output.logs / "run_status.json", {"status": status, "report": report}, overwrite=True)
        write_json(output.data / "report.json", report, overwrite=True)
        output.flush()
        return report
    except Exception as exc:
        try:
            output.flush()
            write_json(output.logs / "run_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "counters": counters}, overwrite=True)
        finally:
            pass
        raise


def main() -> None:
    args = build_parser().parse_args()
    print(encode(run(args.run_id, args.config)))


if __name__ == "__main__":
    main()


__all__ = ["CONFIG", "DECISION_STEPS", "DiagnosticEnv", "EXECUTION_SOURCES_LOCK", "LOCK", "RunOutput", "STEM", "branch_replay", "build_parser", "capture_snapshot", "content_hash", "effective_simulation", "episode_id", "legal_sources", "nees", "paired_post_improvement", "prepare_environment", "run", "select_branch_windows", "select_source", "selection_masks", "trajectory_summary", "validate_execution_sources_lock", "validate_frozen_config"]
