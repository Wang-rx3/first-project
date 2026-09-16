"""Stage 2B-R3 observation identifiability and demonstration audit.

This module is intentionally diagnostic-only.  It consumes the frozen,
Actor-constrained rerun4 demonstration and evaluation artifacts and writes a
new audit package.  It does not run PPO/MAPPO, change the environment, train a
persistent policy, or alter any existing experiment directory.

The audit treats episode/scenario as the independent unit for splits and
reports step-level measurements only as within-episode diagnostics.  The
observation used for the main analysis is the 42-dimensional normalized
``node_features`` block that enters the current one-UAV Actor; the remaining
one-UAV graph masks/empty edge blocks are constant in this dataset and are
recorded as such in the source lock rather than being allowed to create
spurious phase separability.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STAGE_NAME = "single_uav_stage2br3_observation_identifiability"
DEFAULT_RERUN4_DATA_ROOT = Path(
    "data/synthetic/single_uav_static_obstacle_dagger_recovery_goal_seed2026_"
    "actorconstrained_balanced_rerun4"
)
DEFAULT_RERUN4_LOG_ROOT = Path(
    "logs/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_"
    "actorconstrained_balanced_rerun4"
)
DEFAULT_BASELINE_CHECKPOINT = Path(
    "data/synthetic/single_uav_braking_progress025_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_BC_CHECKPOINT = DEFAULT_RERUN4_DATA_ROOT / "bc_final/scale_1/checkpoints/best_validation.pt"
DEFAULT_MIGRATED_CHECKPOINT = (
    DEFAULT_RERUN4_DATA_ROOT
    / "baseline_observation_migration/checkpoints/stage2a_observation_compatible_init.pt"
)
DEFAULT_DATA_ROOT = Path("data/synthetic/single_uav_stage2br3_observation_identifiability_seed2026")
DEFAULT_LOG_ROOT = Path("logs/preliminary/single_uav_stage2br3_observation_identifiability_seed2026")
DEFAULT_PLOT_ROOT = Path("plots/preliminary/single_uav_stage2br3_observation_identifiability_seed2026")
DEFAULT_SEED = 2026
NODE_DIM = 42
TRANSITION_RADIUS = 10
PHASES = (
    "NORMAL",
    "APPROACH_RISK",
    "BYPASS_LEFT",
    "BYPASS_RIGHT",
    "POST_BYPASS",
    "RECOVERY",
    "GOAL_APPROACH",
    "STABLE",
)
RAW_PHASES = ("normal", "decision", "bypass", "post_bypass", "recovery", "boundary_onset")
CONFLICT_PHASE_PAIRS = (
    ("BYPASS_LEFT", "RECOVERY"),
    ("BYPASS_RIGHT", "RECOVERY"),
    ("BYPASS_LEFT", "BYPASS_RIGHT"),
    ("NORMAL", "RECOVERY"),
)
FEATURE_GROUPS = {
    "obstacle_geometry": tuple([33, 34, 35, 39, 40, 41]),
    "goal_geometry": tuple([30, 31, 32]),
    "velocity": tuple([3, 4, 5]),
    "radial_tangential_velocity": tuple([3, 4, 5]),
    "boundary": tuple(),
    "attitude": tuple([6, 7, 8]),
    "covariance": tuple(range(15, 30)),
    "braking_descriptor": tuple([37, 38]),
}


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _mean(values: Iterable[Any]) -> float:
    numbers = [_finite(value) for value in values]
    numbers = [value for value in numbers if np.isfinite(value)]
    return float(np.mean(numbers)) if numbers else float("nan")


def _median(values: Iterable[Any]) -> float:
    numbers = [_finite(value) for value in values]
    numbers = [value for value in numbers if np.isfinite(value)]
    return float(np.median(numbers)) if numbers else float("nan")


def _quantile(values: Iterable[Any], q: float) -> float:
    numbers = [_finite(value) for value in values]
    numbers = [value for value in numbers if np.isfinite(value)]
    return float(np.quantile(numbers, q)) if numbers else float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_array(value: Any, length: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.empty(0, dtype=float)
    if length is None:
        return array
    output = np.zeros(int(length), dtype=float)
    output[: min(len(array), int(length))] = array[: int(length)]
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(str(key))
                    seen.add(str(key))
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_roots(*roots: Path) -> None:
    for root in roots:
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"R3 audit output root is not empty: {root.resolve()}")
    for root in roots:
        Path(root).mkdir(parents=True, exist_ok=True)


def _episode_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("split", "")), str(record.get("episode_id", ""))


def _scenario_key(record: Mapping[str, Any]) -> str:
    return str(record.get("scenario_hash", ""))


def _canonical_phase(record: Mapping[str, Any]) -> str:
    raw = str(record.get("phase", record.get("base_phase", ""))).strip().lower()
    base = str(record.get("base_phase", raw)).strip().lower()
    side = str(record.get("side", "")).strip().lower()
    if raw == "normal":
        return "NORMAL"
    if raw in {"decision", "approach", "approach_risk"}:
        return "APPROACH_RISK"
    if raw == "bypass":
        return "BYPASS_LEFT" if side == "left" else "BYPASS_RIGHT" if side == "right" else "BYPASS"
    if raw == "boundary_onset":
        # Keep the overlay separate.  It is excluded from the core phase
        # probes so boundary intervention cannot masquerade as recovery.
        return "BOUNDARY_ONSET"
    if raw == "post_bypass":
        return "POST_BYPASS"
    if raw == "recovery":
        return "RECOVERY"
    if base == "recovery":
        return "RECOVERY"
    if _finite(record.get("goal_distance_m"), float("inf")) <= 1.0:
        return "STABLE" if _safe_int(record.get("stable_condition", 0)) else "GOAL_APPROACH"
    return raw.upper() if raw else "UNKNOWN"


def _raw_base_phase(record: Mapping[str, Any]) -> str:
    return str(record.get("base_phase", record.get("phase", ""))).strip().lower()


def _phase_for_analysis(record: Mapping[str, Any]) -> str:
    phase = _canonical_phase(record)
    # The core probes use a single recovery class, but retain the overlay in
    # the record for coverage and boundary diagnostics.
    return phase


def _phase_definition_rows() -> list[dict[str, Any]]:
    return [
        {"phase": "NORMAL", "criterion": "raw phase == normal", "core_probe": 1},
        {"phase": "APPROACH_RISK", "criterion": "raw decision/approach label before bypass", "core_probe": 1},
        {"phase": "BYPASS_LEFT", "criterion": "raw phase == bypass and side == left", "core_probe": 1},
        {"phase": "BYPASS_RIGHT", "criterion": "raw phase == bypass and side == right", "core_probe": 1},
        {"phase": "POST_BYPASS", "criterion": "raw phase == post_bypass after obstacle pass, before recovery", "core_probe": 1},
        {"phase": "RECOVERY", "criterion": "raw/base phase == recovery: pass observed, positive radial progress and lateral action decayed", "core_probe": 1},
        {"phase": "GOAL_APPROACH", "criterion": "goal-directed approach within the goal corridor; not stable", "core_probe": 1},
        {"phase": "STABLE", "criterion": "stable-condition label; used only if explicitly present", "core_probe": 1},
        {"phase": "BOUNDARY_ONSET", "criterion": "boundary_onset overlay retained separately; not counted as recovery", "core_probe": 0},
    ]


def _observation_vector(record: Mapping[str, Any]) -> np.ndarray:
    observation = record.get("observation", {})
    if not isinstance(observation, Mapping):
        raise ValueError("demonstration record has no observation mapping")
    node = np.asarray(observation.get("node_features", []), dtype=float)
    if node.ndim == 2:
        node = node[0]
    node = node.reshape(-1)
    if node.size != NODE_DIM:
        raise ValueError(f"expected {NODE_DIM} node features, got {node.size}")
    if not np.isfinite(node).all():
        raise ValueError("Actor node observation contains non-finite values")
    return node.astype(np.float64)


def _goal_direction(record: Mapping[str, Any]) -> np.ndarray:
    vector = _safe_array(record.get("goal_direction", []), 3)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 and np.isfinite(norm) else np.zeros(3, dtype=float)


def _estimated_velocity_from_observation(record: Mapping[str, Any], node: np.ndarray | None = None) -> np.ndarray:
    values = node if node is not None else _observation_vector(record)
    return _safe_array(values[3:6], 3)


def _action_decomposition(record: Mapping[str, Any]) -> dict[str, float]:
    action = _safe_array(record.get("teacher_action_delta_v_mps", []), 3)
    direction = _goal_direction(record)
    radial = float(np.dot(action, direction))
    velocity = _estimated_velocity_from_observation(record)
    speed = float(np.linalg.norm(velocity))
    velocity_direction = velocity / speed if speed > 1.0e-12 else np.zeros(3, dtype=float)
    braking = max(0.0, -float(np.dot(action, velocity_direction)))
    tangent = action - radial * direction
    tangent_norm = float(np.linalg.norm(tangent))
    lateral_sign = 0
    side = str(record.get("side", "")).lower()
    if tangent_norm > 1.0e-9:
        lateral_sign = 1 if float(tangent[1]) > 0.0 else -1 if float(tangent[1]) < 0.0 else 0
    return {
        "teacher_action_norm_mps": float(np.linalg.norm(action)),
        "teacher_radial_action_mps": radial,
        "teacher_tangential_action_mps": tangent_norm,
        "teacher_braking_magnitude_mps": braking,
        "teacher_lateral_sign": lateral_sign,
        "teacher_radial_positive": int(radial > 1.0e-9),
        "teacher_radial_negative": int(radial < -1.0e-9),
        "teacher_action_y_mps": float(action[1]),
        "side_expected_tangent_sign": 1 if side == "left" else -1 if side == "right" else 0,
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 1.0e-12 and nb > 1.0e-12 else 0.0


def _load_feature_layout(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({**row, "index": _safe_int(row.get("index"))})
    result.sort(key=lambda row: int(row["index"]))
    if len(result) != NODE_DIM or [row["index"] for row in result] != list(range(NODE_DIM)):
        raise ValueError("stage2a feature layout is not a complete 0..41 mapping")
    return result


def _feature_semantics(row: Mapping[str, Any]) -> dict[str, Any]:
    index = _safe_int(row.get("index"))
    name = str(row.get("feature", ""))
    obstacle = index in FEATURE_GROUPS["obstacle_geometry"]
    goal = index in FEATURE_GROUPS["goal_geometry"]
    velocity = index in FEATURE_GROUPS["velocity"]
    boundary = index in FEATURE_GROUPS["boundary"]
    if index >= 39:
        information = "estimated_state + static_map_geometry"
        truth = "no"
        source = "EKF estimated position/yaw plus nearest static-map obstacle center"
    elif index in (33, 34, 35):
        information = "derived from estimated state + static_map_geometry"
        truth = "no"
        source = str(row.get("source", "environment local obstacle feature"))
    elif index in (30, 31, 32):
        information = "derived from estimated state and configured goal"
        truth = "no"
        source = str(row.get("source", "goal minus EKF estimate"))
    else:
        information = "estimated_state" if index <= 29 else "derived_descriptor"
        truth = "no"
        source = str(row.get("source", ""))
    return {
        "index": index,
        "feature_name": name,
        "source": source,
        "estimated_or_truth": information,
        "truth_state_used_for_actor": 0,
        "normalization": "frozen Stage-2A ObservationNormalizer node_features; see source lock",
        "physical_meaning": str(row.get("notes", "")) or name,
        "available_to_actor": 1,
        "obstacle_related": int(obstacle),
        "goal_related": int(goal),
        "velocity_related": int(velocity),
        "boundary_related": int(boundary),
        "direct_radial_velocity": 0,
        "direct_tangential_velocity": 0,
        "notes": "radial/tangential velocity, obstacle bearing, distance and pass state are derived/offline; no explicit boundary-clearance feature is present",
    }


def build_feature_map(layout: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_feature_semantics(row) for row in layout]


def _decorate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add offline-only labels and action diagnostics to JSONL records."""

    decorated: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        record["canonical_phase"] = _phase_for_analysis(record)
        record["raw_base_phase"] = _raw_base_phase(record)
        record["episode_key"] = "|".join(_episode_key(record))
        record["scenario_key"] = _scenario_key(record)
        record.update(_action_decomposition(record))
        base = _raw_base_phase(record)
        record["passed_obstacle_inferred"] = int(base in {"post_bypass", "recovery"})
        record["recovery_condition_inferred"] = int(record["canonical_phase"] == "RECOVERY")
        record["boundary_overlay"] = int(record["canonical_phase"] == "BOUNDARY_ONSET")
        record["goal_distance_numeric"] = _finite(record.get("goal_distance_m"))
        record["speed_numeric"] = _finite(record.get("speed_mps"))
        record["radial_velocity_numeric"] = _finite(record.get("radial_velocity_mps"))
        record["tangential_velocity_numeric"] = _finite(record.get("tangential_velocity_mps"))
        obstacle = _safe_array(record.get("obstacle_direction_body_m", []), 3)
        record["obstacle_bearing_rad"] = float(math.atan2(obstacle[1], obstacle[0])) if np.linalg.norm(obstacle[:2]) > 1.0e-12 else float("nan")
        goal = _safe_array(record.get("goal_direction", []), 3)
        record["goal_bearing_rad"] = float(math.atan2(goal[1], goal[0])) if np.linalg.norm(goal[:2]) > 1.0e-12 else float("nan")
        record["static_clearance_available"] = int(any(key in record for key in ("minimum_pre_clearance_m", "relevant_obstacle_clearance_m", "pre_candidate_static_swept_minimum_clearance_m")))
        decorated.append(record)
    return decorated


def _checkpoint_metadata(path: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Read checkpoint provenance without changing or saving the checkpoint."""

    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": int(path.is_file()),
        "sha256": _sha256_file(path) if path.is_file() else "",
    }
    if not path.is_file():
        return result
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        result["payload_keys"] = sorted(str(key) for key in payload) if isinstance(payload, Mapping) else []
        state = payload.get("policy_state_dict", {}) if isinstance(payload, Mapping) else {}
        result["policy_state_parameter_count"] = int(sum(int(np.prod(tuple(value.shape))) for value in state.values() if hasattr(value, "shape")))
        result["policy_state_keys_count"] = int(len(state))
        if isinstance(payload, Mapping):
            for key in ("config", "training_config", "bc_training", "stage", "command_line"):
                value = payload.get(key)
                if isinstance(value, Mapping):
                    result[f"{key}_keys"] = sorted(str(item) for item in value)
                elif value is not None:
                    result[key] = str(value)
            config_payload = payload.get("config")
            if not isinstance(config_payload, Mapping):
                config_payload = payload.get("trainer_config")
            if isinstance(config_payload, Mapping):
                selected = {}
                for key in ("uav_count", "node_feature_dim", "edge_feature_dim", "gat_hidden_dim", "gat_output_dim", "gat_layers", "actor_hidden_dim", "encoder_type", "max_delta_v", "include_remaining_time", "device"):
                    if key in config_payload:
                        selected[key] = config_payload[key]
                result["config_selected"] = selected
            for config_key in ("trainer_config", "experiment_config"):
                config_value = payload.get(config_key)
                if isinstance(config_value, Mapping):
                    selected = {}
                    for key in (
                        "uav_count", "node_feature_dim", "edge_feature_dim", "gat_hidden_dim",
                        "gat_output_dim", "gat_layers", "gat_heads", "gat_dropout",
                        "actor_hidden_dim", "critic_hidden_dim", "max_delta_v", "encoder_type",
                        "device", "include_remaining_time", "learning_rate", "gamma", "gae_lambda",
                        "clip_epsilon", "policy_epochs", "minibatch_size", "max_grad_norm",
                    ):
                        if key in config_value:
                            selected[key] = config_value[key]
                    result[f"{config_key}_selected"] = selected
            normalization = payload.get("normalization_stats")
            if isinstance(normalization, Mapping):
                result["normalization_stats_keys"] = sorted(str(key) for key in normalization)
                normalizers = normalization.get("normalizers")
                if isinstance(normalizers, Mapping):
                    result["normalizer_blocks"] = sorted(str(key) for key in normalizers)
    except Exception as exc:  # pragma: no cover - environment-dependent checkpoint reader
        result["checkpoint_read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_source_lock(
    rerun4_data_root: Path,
    rerun4_log_root: Path,
    *,
    baseline_checkpoint: Path = DEFAULT_BASELINE_CHECKPOINT,
    bc_checkpoint: Path = DEFAULT_BC_CHECKPOINT,
    migrated_checkpoint: Path = DEFAULT_MIGRATED_CHECKPOINT,
) -> dict[str, Any]:
    """Lock the exact input artifacts used by this diagnostic."""

    manifest_path = rerun4_data_root / "demonstrations/manifest.json"
    summary_path = rerun4_data_root / "summary.json"
    report_path = rerun4_log_root / "stage_report.md"
    layout_path = rerun4_log_root / "baseline_observation_migration/stage2a_feature_layout.csv"
    required = [
        manifest_path,
        summary_path,
        report_path,
        layout_path,
        rerun4_data_root / "demonstrations/train.jsonl",
        rerun4_data_root / "demonstrations/validation.jsonl",
        rerun4_log_root / "evaluation_step_metrics.csv",
        rerun4_log_root / "evaluation_episode_metrics.csv",
        rerun4_log_root / "demonstration_step_audit.csv",
        rerun4_log_root / "bc_final/bc_v2_training_history.csv",
        rerun4_log_root / "bc_final/offline_action_diagnostics_by_distance.csv",
    ]
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Stage 2B-R3 source lock missing: " + "; ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    layout = _load_feature_layout(layout_path)
    train = _read_jsonl(rerun4_data_root / "demonstrations/train.jsonl")
    validation = _read_jsonl(rerun4_data_root / "demonstrations/validation.jsonl")
    if not train or not validation:
        raise ValueError("rerun4 demonstrations must contain non-empty train and validation splits")
    node_dims = sorted({int(_observation_vector(row).size) for row in train[: min(32, len(train))] + validation[: min(32, len(validation))]})
    truth_flags = sorted({bool(row.get("truth_state_read_for_control", False)) for row in train + validation})
    split_episode_sets = {
        "train": sorted({str(row.get("episode_id", "")) for row in train}),
        "validation": sorted({str(row.get("episode_id", "")) for row in validation}),
    }
    split_scenario_sets = {
        "train": sorted({str(row.get("scenario_hash", "")) for row in train}),
        "validation": sorted({str(row.get("scenario_hash", "")) for row in validation}),
    }
    split_pair_sets = {
        "train": sorted({str(row.get("pair_id", "")) for row in train}),
        "validation": sorted({str(row.get("pair_id", "")) for row in validation}),
    }
    source_lock = {
        "stage": STAGE_NAME,
        "seed": DEFAULT_SEED,
        "rerun4_data_root": str(rerun4_data_root.resolve()),
        "rerun4_log_root": str(rerun4_log_root.resolve()),
        "required_input_files": [str(path.resolve()) for path in required],
        "missing_input_files": missing,
        "rerun4_summary_keys": sorted(str(key) for key in summary),
        "manifest_dataset_id": manifest.get("dataset_id", ""),
        "manifest_source": manifest.get("source", ""),
        "manifest_train_samples_raw": manifest.get("train_samples_raw"),
        "manifest_validation_samples_raw": manifest.get("validation_samples_raw"),
        "manifest_train_samples_balanced": manifest.get("train_samples_balanced"),
        "manifest_validation_samples_balanced": manifest.get("validation_samples_balanced"),
        "manifest_train_episode_count": manifest.get("train_episode_count"),
        "manifest_validation_episode_count": manifest.get("validation_episode_count"),
        "manifest_trajectory_leakage": manifest.get("trajectory_leakage"),
        "manifest_pair_leakage": manifest.get("pair_leakage"),
        "manifest_teacher_truth_state_read_for_control": manifest.get("teacher_truth_state_read_for_control"),
        "demonstration_observation_node_dims": node_dims,
        "truth_state_read_for_control_values": truth_flags,
        "train_episode_count_observed": len(split_episode_sets["train"]),
        "validation_episode_count_observed": len(split_episode_sets["validation"]),
        "train_scenario_count_observed": len(split_scenario_sets["train"]),
        "validation_scenario_count_observed": len(split_scenario_sets["validation"]),
        "train_validation_episode_leakage": bool(set(split_episode_sets["train"]) & set(split_episode_sets["validation"])),
        "train_validation_scenario_leakage": bool(set(split_scenario_sets["train"]) & set(split_scenario_sets["validation"])),
        "train_validation_pair_leakage": bool(set(split_pair_sets["train"]) & set(split_pair_sets["validation"])),
        "observation_contract": {
            "actor_node_features_dim": NODE_DIM,
            "actor_node_features_are_normalized": True,
            "actor_context_constant_blocks": ["empty one-UAV edge blocks", "action mask", "bandwidth mask"],
            "analysis_vector": "normalized node_features only; masks/empty edge blocks are invariant in the audited one-UAV dataset",
        },
        "feature_layout_path": str(layout_path.resolve()),
        "feature_layout_sha256": _sha256_file(layout_path),
        "feature_layout_rows": len(layout),
        "baseline_checkpoint": _checkpoint_metadata(Path(baseline_checkpoint)),
        "bc_checkpoint": _checkpoint_metadata(Path(bc_checkpoint)),
        "migrated_stage2a_checkpoint": _checkpoint_metadata(Path(migrated_checkpoint)),
        "action_transform": {
            "actor_output": "post-tanh mean",
            "physical_action": "tanh(mean) * max_delta_v / sqrt(3)",
            "teacher": "existing estimated-state geometric Oracle followed by the same Actor componentwise constraint",
        },
        "teacher_definition": manifest.get("source", ""),
        "phase_definitions": _phase_definition_rows(),
        "raw_phase_counts_manifest": manifest.get("phase_counts_raw", {}),
    }
    return source_lock


def _load_audit_records(rerun4_data_root: Path, rerun4_log_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    train = _decorate_records(_read_jsonl(rerun4_data_root / "demonstrations/train.jsonl"))
    validation = _decorate_records(_read_jsonl(rerun4_data_root / "demonstrations/validation.jsonl"))
    raw_audit = _read_csv(rerun4_log_root / "demonstration_step_audit.csv")
    return train, validation, raw_audit


def _feature_matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not records:
        return np.empty((0, NODE_DIM), dtype=float)
    return np.stack([_observation_vector(record) for record in records]).astype(float)


def _standardize(train_x: np.ndarray, other_x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0) if len(train_x) else np.zeros(NODE_DIM, dtype=float)
    scale = np.nanstd(train_x, axis=0) if len(train_x) else np.ones(NODE_DIM, dtype=float)
    mean = np.nan_to_num(mean, nan=0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-8), scale, 1.0)
    standardized = (np.nan_to_num(train_x, nan=0.0) - mean) / scale
    if other_x is not None:
        other = (np.nan_to_num(other_x, nan=0.0) - mean) / scale
        return standardized, other, np.stack([mean, scale])
    return standardized, mean, scale


def _core_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in records
        if str(record.get("canonical_phase", "")) in PHASES
        and str(record.get("canonical_phase", "")) != "STABLE"
    ]


def _phase_record_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        phase = str(record.get("canonical_phase") or _phase_for_analysis(record))
        if phase in PHASES:
            groups[phase].append(dict(record))
    return dict(groups)


def _unique_count(records: Sequence[Mapping[str, Any]], field: str) -> int:
    return len({str(record.get(field, "")) for record in records if str(record.get(field, ""))})


def _episode_majority_accuracy(records: Sequence[Mapping[str, Any]], predictions: Sequence[str]) -> float:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record, predicted in zip(records, predictions):
        grouped[str(record.get("episode_key", _episode_key(record)))] .append((str(record.get("canonical_phase", "")), str(predicted)))
    if not grouped:
        return float("nan")
    correct = []
    for values in grouped.values():
        true_label = Counter(true for true, _ in values).most_common(1)[0][0]
        predicted_label = Counter(predicted for _, predicted in values).most_common(1)[0][0]
        correct.append(int(true_label == predicted_label))
    return float(np.mean(correct))


def _phase_metric_rows(records: Sequence[Mapping[str, Any]], predictions: Sequence[str], *, model: str, level: str) -> list[dict[str, Any]]:
    labels = sorted(set(PHASES) & {str(record.get("canonical_phase", "")) for record in records})
    rows: list[dict[str, Any]] = []
    for phase in labels:
        true = [str(record.get("canonical_phase", "")) for record in records]
        tp = sum(int(actual == phase and predicted == phase) for actual, predicted in zip(true, predictions))
        fp = sum(int(actual != phase and predicted == phase) for actual, predicted in zip(true, predictions))
        fn = sum(int(actual == phase and predicted != phase) for actual, predicted in zip(true, predictions))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        phase_records = [record for record in records if str(record.get("canonical_phase", "")) == phase]
        rows.append(
            {
                "model": model,
                "level": level,
                "phase": phase,
                "samples": len(phase_records),
                "episodes": _unique_count(phase_records, "episode_key"),
                "precision": precision,
                "recall": recall,
                "support": int(tp + fn),
            }
        )
    return rows


def _fallback_knn_predict(train_z: np.ndarray, train_y: Sequence[str], validation_z: np.ndarray, *, k: int) -> list[str]:
    if not len(train_z):
        return []
    distances = np.linalg.norm(validation_z[:, None, :] - train_z[None, :, :], axis=2)
    predictions: list[str] = []
    for row in distances:
        indices = np.argsort(row)[: min(int(k), len(train_z))]
        predictions.append(str(Counter(str(train_y[index]) for index in indices).most_common(1)[0][0]))
    return predictions


def _fallback_torch_probe_predict(train_z: np.ndarray, train_y: Sequence[str], validation_z: np.ndarray, *, nonlinear: bool, seed: int) -> list[str]:
    """Dependency-light linear/MLP probe used when sklearn is unavailable."""

    import torch
    from torch import nn

    classes = sorted({str(value) for value in train_y})
    if len(classes) < 2:
        return [classes[0]] * len(validation_z) if classes else []
    torch.manual_seed(int(seed))
    layers: list[nn.Module] = [nn.Linear(train_z.shape[1], 32), nn.Tanh(), nn.Linear(32, len(classes))] if nonlinear else [nn.Linear(train_z.shape[1], len(classes))]
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    features = torch.as_tensor(train_z, dtype=torch.float32)
    targets = torch.as_tensor([classes.index(str(value)) for value in train_y], dtype=torch.long)
    model.train()
    for _ in range(180 if nonlinear else 120):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = model(torch.as_tensor(validation_z, dtype=torch.float32)).argmax(dim=1).cpu().numpy()
    return [classes[int(index)] for index in predictions]


def _fit_probes(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fit diagnostic probes on train episodes and evaluate on held-out episodes."""

    train_records = _core_records(train_records)
    validation_records = _core_records(validation_records)
    train_x = _feature_matrix(train_records)
    validation_x = _feature_matrix(validation_records)
    train_z, validation_z, stats = _standardize(train_x, validation_x)
    y_train = np.asarray([str(row["canonical_phase"]) for row in train_records])
    y_validation = np.asarray([str(row["canonical_phase"]) for row in validation_records])
    classes = sorted(set(y_train) & set(y_validation))
    result_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "train_episodes": _unique_count(train_records, "episode_key"),
        "validation_episodes": _unique_count(validation_records, "episode_key"),
        "classes_train": sorted(set(y_train)),
        "classes_validation": sorted(set(y_validation)),
        "classes_common": classes,
        "standardization_mean": stats[0].tolist() if isinstance(stats, np.ndarray) else [],
        "standardization_scale": stats[1].tolist() if isinstance(stats, np.ndarray) else [],
    }
    if len(classes) < 2:
        metadata["status"] = "insufficient_common_phase_classes"
        return result_rows, confusion_rows, metadata

    def record_predictions(model_name: str, predictions: Sequence[str], model_object: str, *, fit_level: str = "step") -> None:
        predictions = [str(value) for value in predictions]
        accuracy = float(np.mean(np.asarray(predictions) == y_validation)) if len(predictions) else float("nan")
        result_rows.extend(
            [
                {
                    "comparison": "all_common_phases",
                    "metric": "accuracy",
                    "value": accuracy,
                    "model": model_name,
                    "level": fit_level,
                    "samples": len(validation_records),
                    "episodes": _unique_count(validation_records, "episode_key"),
                    "notes": model_object,
                },
                {
                    "comparison": "all_common_phases",
                    "metric": "episode_majority_accuracy",
                    "value": _episode_majority_accuracy(validation_records, predictions),
                    "model": model_name,
                    "level": "episode",
                    "samples": len(validation_records),
                    "episodes": _unique_count(validation_records, "episode_key"),
                    "notes": "episode is the independent unit; validation steps are aggregated by majority label",
                },
            ]
        )
        result_rows.extend(_phase_metric_rows(validation_records, predictions, model=model_name, level="step"))
        for actual, predicted in zip(y_validation, predictions):
            confusion_rows.append({"model": model_name, "level": "step", "true_phase": actual, "predicted_phase": predicted, "count": 1})

        episode_predictions: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for record, predicted in zip(validation_records, predictions):
            episode_predictions[str(record["episode_key"])].append((str(record["canonical_phase"]), predicted))
        for values in episode_predictions.values():
            actual = Counter(label for label, _ in values).most_common(1)[0][0]
            predicted = Counter(label for _, label in values).most_common(1)[0][0]
            confusion_rows.append({"model": model_name, "level": "episode", "true_phase": actual, "predicted_phase": predicted, "count": 1})

    # Nearest-neighbor diagnostic: every validation state is compared to the
    # training pool, so no same-trajectory neighbor can leak into evaluation.
    if len(train_z):
        distances = np.linalg.norm(validation_z[:, None, :] - train_z[None, :, :], axis=2)
        nearest_indices = np.argmin(distances, axis=1)
        knn_predictions = [str(y_train[index]) for index in nearest_indices]
        record_predictions("1-NN", knn_predictions, "held-out validation state -> nearest train state")

        try:
            from sklearn.neighbors import KNeighborsClassifier

            k = min(5, max(1, len(train_z)))
            knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
            knn.fit(train_z, y_train)
            knn_predictions = knn.predict(validation_z)
            knn_note = "sklearn distance-weighted kNN"
        except Exception as exc:
            knn_predictions = _fallback_knn_predict(train_z, y_train, validation_z, k=5)
            knn_note = f"NumPy kNN fallback; sklearn unavailable: {type(exc).__name__}"
            metadata["knn_fallback"] = str(exc)
        record_predictions("5-NN", knn_predictions, knn_note)

    try:
        from sklearn.linear_model import LogisticRegression

        linear = LogisticRegression(max_iter=400, class_weight="balanced", random_state=int(seed), solver="lbfgs")
        linear.fit(train_z, y_train)
        record_predictions("linear_probe", linear.predict(validation_z), "standardized 42D node-feature logistic regression")
    except Exception as exc:
        linear_predictions = _fallback_torch_probe_predict(train_z, y_train, validation_z, nonlinear=False, seed=seed)
        record_predictions("linear_probe", linear_predictions, f"PyTorch linear fallback; sklearn unavailable: {type(exc).__name__}")
        metadata["linear_probe_fallback"] = str(exc)

    try:
        from sklearn.neural_network import MLPClassifier

        nonlinear = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=int(seed), early_stopping=False)
        nonlinear.fit(train_z, y_train)
        record_predictions("nonlinear_probe", nonlinear.predict(validation_z), "one-hidden-layer MLP diagnostic probe")
    except Exception as exc:
        nonlinear_predictions = _fallback_torch_probe_predict(train_z, y_train, validation_z, nonlinear=True, seed=seed)
        record_predictions("nonlinear_probe", nonlinear_predictions, f"PyTorch MLP fallback; sklearn unavailable: {type(exc).__name__}")
        metadata["nonlinear_probe_fallback"] = str(exc)

    # Cross-phase standardized distances and train-only cross-episode nearest
    # neighbor purity provide classifier-independent evidence.
    all_groups = _phase_record_groups(train_records)
    train_z_by_phase = {phase: train_z[[index for index, row in enumerate(train_records) if row["canonical_phase"] == phase]] for phase in all_groups}
    for left, right in CONFLICT_PHASE_PAIRS:
        if left not in train_z_by_phase or right not in train_z_by_phase:
            continue
        left_z, right_z = train_z_by_phase[left], train_z_by_phase[right]
        cross = np.linalg.norm(left_z[:, None, :] - right_z[None, :, :], axis=2)
        nearest = np.concatenate([np.min(cross, axis=1), np.min(cross, axis=0)])
        centroid_distance = float(np.linalg.norm(np.mean(left_z, axis=0) - np.mean(right_z, axis=0)))
        result_rows.extend(
            [
                {"comparison": f"{left}__{right}", "metric": "centroid_standardized_distance", "value": centroid_distance, "model": "distribution", "level": "train", "samples": int(left_z.shape[0] + right_z.shape[0]), "episodes": "", "notes": "distance between phase centroids"},
                {"comparison": f"{left}__{right}", "metric": "nearest_cross_phase_standardized_distance_median", "value": float(np.median(nearest)), "model": "distribution", "level": "train", "samples": int(left_z.shape[0] + right_z.shape[0]), "episodes": "", "notes": "nearest opposite-phase cross-distance"},
                {"comparison": f"{left}__{right}", "metric": "nearest_cross_phase_standardized_distance_p10", "value": float(np.quantile(nearest, 0.10)), "model": "distribution", "level": "train", "samples": int(left_z.shape[0] + right_z.shape[0]), "episodes": "", "notes": "small distances are candidates for observation aliasing"},
            ]
        )

    # Neighbor purity on train, excluding all states from the same episode.
    if len(train_records) > 1:
        train_distances = np.linalg.norm(train_z[:, None, :] - train_z[None, :, :], axis=2)
        np.fill_diagonal(train_distances, np.inf)
        for index, record in enumerate(train_records):
            same_episode = np.asarray([str(other["episode_key"]) == str(record["episode_key"]) for other in train_records])
            train_distances[index, same_episode] = np.inf
        for k in (1, 5):
            purities = []
            for index, record in enumerate(train_records):
                neighbors = np.argsort(train_distances[index])[: min(k, len(train_records) - 1)]
                neighbors = [j for j in neighbors if np.isfinite(train_distances[index, j])]
                if neighbors:
                    purities.append(float(np.mean([y_train[j] == y_train[index] for j in neighbors])))
            result_rows.append({"comparison": "all_core_phases", "metric": f"cross_episode_{k}NN_phase_purity", "value": _mean(purities), "model": "nearest_neighbor", "level": "train", "samples": len(train_records), "episodes": _unique_count(train_records, "episode_key"), "notes": "same-episode neighbors excluded"})
    metadata["status"] = "ok"
    return result_rows, confusion_rows, metadata


def _phase_pair(left: str, right: str) -> str:
    if left == right:
        return left
    return "__".join(sorted((str(left), str(right))))


def _target_pair_allowed(left: str, right: str) -> bool:
    return _phase_pair(left, right) in {_phase_pair(*pair) for pair in CONFLICT_PHASE_PAIRS}


def _action_vector(record: Mapping[str, Any]) -> np.ndarray:
    return _safe_array(record.get("teacher_action_delta_v_mps", []), 3)


def _conflict_row(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    obs_distance: float,
    standardized_distance: float,
    rank: int,
    query_split: str,
) -> dict[str, Any]:
    action_left = _action_vector(left)
    action_right = _action_vector(right)
    left_parts = _action_decomposition(left)
    right_parts = _action_decomposition(right)
    action_norms = (float(np.linalg.norm(action_left)), float(np.linalg.norm(action_right)))
    action_distance = float(np.linalg.norm(action_left - action_right))
    tangential_difference = abs(float(left_parts["teacher_tangential_action_mps"] - right_parts["teacher_tangential_action_mps"]))
    radial_difference = abs(float(left_parts["teacher_radial_action_mps"] - right_parts["teacher_radial_action_mps"]))
    sign_left = int(np.sign(left_parts["teacher_action_y_mps"]))
    sign_right = int(np.sign(right_parts["teacher_action_y_mps"]))
    radial_sign_left = int(np.sign(left_parts["teacher_radial_action_mps"]))
    radial_sign_right = int(np.sign(right_parts["teacher_radial_action_mps"]))
    return {
        "episode_i": str(left.get("episode_id", "")),
        "split_i": str(left.get("split", "")),
        "step_i": _safe_int(left.get("step_index", 0)),
        "phase_i": str(left.get("canonical_phase", "")),
        "episode_j": str(right.get("episode_id", "")),
        "split_j": str(right.get("split", "")),
        "step_j": _safe_int(right.get("step_index", 0)),
        "phase_j": str(right.get("canonical_phase", "")),
        "scenario_i": str(left.get("scenario_hash", "")),
        "scenario_j": str(right.get("scenario_hash", "")),
        "side_i": str(left.get("side", "")),
        "side_j": str(right.get("side", "")),
        "query_split": query_split,
        "nearest_neighbor_rank": int(rank),
        "observation_distance_raw_normalized": float(obs_distance),
        "observation_distance_standardized": float(standardized_distance),
        "observation_cosine_similarity": _cosine(_observation_vector(left), _observation_vector(right)),
        "teacher_action_i": json.dumps(action_left.tolist(), separators=(",", ":")),
        "teacher_action_j": json.dumps(action_right.tolist(), separators=(",", ":")),
        "teacher_action_norm_i_mps": action_norms[0],
        "teacher_action_norm_j_mps": action_norms[1],
        "teacher_action_distance_mps": action_distance,
        "teacher_radial_i_mps": left_parts["teacher_radial_action_mps"],
        "teacher_radial_j_mps": right_parts["teacher_radial_action_mps"],
        "teacher_radial_difference_mps": radial_difference,
        "teacher_tangential_i_mps": left_parts["teacher_tangential_action_mps"],
        "teacher_tangential_j_mps": right_parts["teacher_tangential_action_mps"],
        "teacher_tangential_difference_mps": tangential_difference,
        "teacher_action_cosine": _cosine(action_left, action_right),
        "teacher_y_sign_i": sign_left,
        "teacher_y_sign_j": sign_right,
        "tangential_sign_disagreement": int(sign_left != sign_right and sign_left != 0 and sign_right != 0),
        "radial_sign_disagreement": int(radial_sign_left != radial_sign_right and radial_sign_left != 0 and radial_sign_right != 0),
        "left_right_phase_disagreement": int({str(left.get("canonical_phase")), str(right.get("canonical_phase"))} == {"BYPASS_LEFT", "BYPASS_RIGHT"}),
        "goal_distance_i_m": _finite(left.get("goal_distance_m")),
        "goal_distance_j_m": _finite(right.get("goal_distance_m")),
        "conflict_type": _phase_pair(str(left.get("canonical_phase", "")), str(right.get("canonical_phase", ""))),
    }


def _nearest_neighbor_conflicts(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], *, max_neighbors: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Find cross-episode near-observation pairs with different teacher targets."""

    train_records = _core_records(train_records)
    validation_records = _core_records(validation_records)
    train_x = _feature_matrix(train_records)
    validation_x = _feature_matrix(validation_records)
    train_z, validation_z, _stats = _standardize(train_x, validation_x)
    rows: list[dict[str, Any]] = []

    def add_query(query_records: Sequence[Mapping[str, Any]], query_z: np.ndarray, pool_records: Sequence[Mapping[str, Any]], pool_z: np.ndarray, query_split: str, *, exclude_same_episode: bool) -> None:
        if not len(query_records) or not len(pool_records):
            return
        for q_index, query in enumerate(query_records):
            distance = np.linalg.norm(pool_z - query_z[q_index], axis=1)
            eligible = np.ones(len(pool_records), dtype=bool)
            if exclude_same_episode:
                eligible &= np.asarray([str(candidate.get("episode_key")) != str(query.get("episode_key")) for candidate in pool_records])
            eligible &= np.asarray([_target_pair_allowed(str(query.get("canonical_phase", "")), str(candidate.get("canonical_phase", ""))) for candidate in pool_records])
            candidate_indices = np.flatnonzero(eligible)
            if len(candidate_indices) == 0:
                continue
            order = candidate_indices[np.argsort(distance[candidate_indices])[: int(max_neighbors)]]
            for rank, candidate_index in enumerate(order, start=1):
                rows.append(_conflict_row(query, pool_records[int(candidate_index)], obs_distance=float(np.linalg.norm(_observation_vector(query) - _observation_vector(pool_records[int(candidate_index)]))), standardized_distance=float(distance[int(candidate_index)]), rank=rank, query_split=query_split))

    # Train queries use cross-episode neighbors; validation queries use only
    # train states.  This keeps the held-out comparison free of trajectory and
    # pair leakage while still exposing within-train aliasing.
    add_query(train_records, train_z, train_records, train_z, "train_cross_episode", exclude_same_episode=True)
    add_query(validation_records, validation_z, train_records, train_z, "validation_to_train", exclude_same_episode=False)
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["query_split"], row["episode_i"], row["step_i"], row["episode_j"], row["step_j"], row["conflict_type"])
        dedup[key] = row
    rows = list(dedup.values())
    rows.sort(key=lambda row: (str(row["query_split"]), str(row["conflict_type"]), _finite(row["observation_distance_standardized"]), str(row["episode_i"]), _safe_int(row["step_i"])))
    distances = [_finite(row.get("observation_distance_standardized")) for row in rows]
    action_distances = [_finite(row.get("teacher_action_distance_mps")) for row in rows]
    obs_threshold = _quantile(distances, 0.10)
    action_threshold = _quantile(action_distances, 0.75)
    for row in rows:
        row["observation_close_threshold_p10"] = obs_threshold
        row["action_conflict_threshold_p75"] = action_threshold
        row["observation_close"] = int(np.isfinite(obs_threshold) and _finite(row.get("observation_distance_standardized")) <= obs_threshold)
        sign_conflict = int(row.get("tangential_sign_disagreement", 0) or row.get("radial_sign_disagreement", 0) or row.get("left_right_phase_disagreement", 0))
        row["action_conflict"] = int((_finite(row.get("teacher_action_distance_mps")) >= action_threshold if np.isfinite(action_threshold) else False) or sign_conflict)
        row["strong_observation_target_conflict"] = int(row["observation_close"] and row["action_conflict"])
    summary: list[dict[str, Any]] = []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("conflict_type", ""))].append(row)
    for pair in sorted(groups):
        selected = groups[pair]
        summary.append(
            {
                "conflict_type": pair,
                "pairs": len(selected),
                "rank1_pairs": int(sum(_safe_int(row.get("nearest_neighbor_rank")) == 1 for row in selected)),
                "observation_close_pairs": int(sum(_safe_int(row.get("observation_close")) for row in selected)),
                "action_conflict_pairs": int(sum(_safe_int(row.get("action_conflict")) for row in selected)),
                "strong_observation_target_conflict_pairs": int(sum(_safe_int(row.get("strong_observation_target_conflict")) for row in selected)),
                "strong_conflict_rate": _mean(row.get("strong_observation_target_conflict") for row in selected),
                "observation_distance_standardized_median": _median(row.get("observation_distance_standardized") for row in selected),
                "observation_distance_standardized_p10": _quantile((row.get("observation_distance_standardized") for row in selected), 0.10),
                "teacher_action_distance_median_mps": _median(row.get("teacher_action_distance_mps") for row in selected),
                "teacher_action_distance_p75_mps": _quantile((row.get("teacher_action_distance_mps") for row in selected), 0.75),
                "tangential_sign_disagreement_rate": _mean(row.get("tangential_sign_disagreement") for row in selected),
                "radial_sign_disagreement_rate": _mean(row.get("radial_sign_disagreement") for row in selected),
                "query_splits": ",".join(sorted({str(row.get("query_split", "")) for row in selected})),
                "notes": "observation_close is standardized distance <= eligible-pair p10; action_conflict is p75 action distance or sign/left-right conflict",
            }
        )
    overall = {
        "pairs": len(rows),
        "conflict_types": sorted(groups),
        "observation_close_threshold_standardized_p10": obs_threshold,
        "action_conflict_threshold_mps_p75": action_threshold,
        "strong_conflicts": int(sum(_safe_int(row.get("strong_observation_target_conflict")) for row in rows)),
        "strong_conflict_rate": _mean(row.get("strong_observation_target_conflict") for row in rows),
        "bypass_recovery_pairs": int(sum("BYPASS" in str(row.get("conflict_type")) and "RECOVERY" in str(row.get("conflict_type")) for row in rows)),
        "bypass_recovery_strong_conflicts": int(sum(_safe_int(row.get("strong_observation_target_conflict")) for row in rows if "BYPASS" in str(row.get("conflict_type")) and "RECOVERY" in str(row.get("conflict_type")))),
    }
    return rows, summary, overall


def _transition_windows(records: Sequence[Mapping[str, Any]], *, radius: int = TRANSITION_RADIUS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract bypass-to-post-bypass/recovery windows without relabeling input."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("kind", "")) != "static_obstacle":
            continue
        grouped[_episode_key(record)].append(dict(record))
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for episode_key, episode_rows in sorted(grouped.items()):
        episode_rows.sort(key=lambda row: _safe_int(row.get("step_index", 0)))
        bypass_indices = [index for index, row in enumerate(episode_rows) if str(row.get("canonical_phase") or _phase_for_analysis(row)) in {"BYPASS_LEFT", "BYPASS_RIGHT"}]
        if not bypass_indices:
            continue
        transition_index = None
        for index in range(min(bypass_indices), len(episode_rows)):
            current = str(episode_rows[index].get("canonical_phase") or _phase_for_analysis(episode_rows[index]))
            previous = str(episode_rows[index - 1].get("canonical_phase") or _phase_for_analysis(episode_rows[index - 1])) if index else ""
            if previous in {"BYPASS_LEFT", "BYPASS_RIGHT"} and current in {"POST_BYPASS", "RECOVERY", "BOUNDARY_ONSET"}:
                transition_index = index
                break
        if transition_index is None:
            continue
        center_step = _safe_int(episode_rows[transition_index].get("step_index", 0))
        side = str(episode_rows[transition_index].get("side", ""))
        transition_id = f"{episode_key[0]}::{episode_key[1]}::step_{center_step:04d}"
        transitions.append(
            {
                "transition_id": transition_id,
                "split": episode_key[0],
                "episode_id": episode_key[1],
                "scenario_hash": str(episode_rows[transition_index].get("scenario_hash", "")),
                "pair_id": str(episode_rows[transition_index].get("pair_id", "")),
                "side": side,
                "center_step": center_step,
                "from_phase": str(episode_rows[transition_index - 1].get("canonical_phase") or _phase_for_analysis(episode_rows[transition_index - 1])) if transition_index else "",
                "to_phase": str(episode_rows[transition_index].get("canonical_phase") or _phase_for_analysis(episode_rows[transition_index])),
                "window_radius": int(radius),
                "window_rows": 0,
            }
        )
        selected = episode_rows[max(0, transition_index - int(radius)) : min(len(episode_rows), transition_index + int(radius) + 1)]
        for row in selected:
            action = _action_decomposition(row)
            step = _safe_int(row.get("step_index", 0))
            obstacle = _safe_array(row.get("obstacle_direction_body_m", []), 3)
            goal = _safe_array(row.get("goal_direction", []), 3)
            static_clearance = row.get("minimum_pre_clearance_m", row.get("relevant_obstacle_candidate_clearance_m", row.get("pre_candidate_static_swept_minimum_clearance_m", float("nan"))))
            boundary = row.get("boundary_clearance_m", row.get("distance_to_nearest_boundary_m", float("nan")))
            rows.append(
                {
                    "transition_id": transition_id,
                    "split": episode_key[0],
                    "episode_id": episode_key[1],
                    "scenario_hash": str(row.get("scenario_hash", "")),
                    "pair_id": str(row.get("pair_id", "")),
                    "side": side,
                    "center_step": center_step,
                    "step": step,
                    "offset": int(step - center_step),
                    "raw_phase": str(row.get("phase", "")),
                    "base_phase": str(row.get("base_phase", "")),
                    "canonical_phase": str(row.get("canonical_phase") or _phase_for_analysis(row)),
                    "goal_distance_m": _finite(row.get("goal_distance_m")),
                    "obstacle_clearance_m": _finite(static_clearance),
                    "obstacle_bearing_rad": float(math.atan2(obstacle[1], obstacle[0])) if np.linalg.norm(obstacle[:2]) > 1.0e-12 else float("nan"),
                    "goal_bearing_rad": float(math.atan2(goal[1], goal[0])) if np.linalg.norm(goal[:2]) > 1.0e-12 else float("nan"),
                    "radial_velocity_mps": _finite(row.get("radial_velocity_mps")),
                    "tangential_velocity_mps": _finite(row.get("tangential_velocity_mps")),
                    "speed_mps": _finite(row.get("speed_mps")),
                    "teacher_radial_action_mps": action["teacher_radial_action_mps"],
                    "teacher_tangential_action_mps": action["teacher_tangential_action_mps"],
                    "teacher_braking_magnitude_mps": action["teacher_braking_magnitude_mps"],
                    "teacher_action_y_mps": action["teacher_action_y_mps"],
                    "boundary_clearance_m": _finite(boundary),
                    "passed_obstacle_inferred": int(row.get("passed_obstacle_inferred", 0)),
                    "recovery_condition_inferred": int(row.get("recovery_condition_inferred", 0)),
                }
            )
        transitions[-1]["window_rows"] = int(len(selected))
    return rows, transitions


def _transition_feature_changes(
    records: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    feature_map: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Quantify Actor-feature changes at each bypass-to-recovery transition."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_episode_key(record)].append(record)
    for values in grouped.values():
        values.sort(key=lambda row: _safe_int(row.get("step_index", 0)))
    rows: list[dict[str, Any]] = []
    all_deltas: list[np.ndarray] = []
    for transition in transitions:
        key = (str(transition.get("split", "")), str(transition.get("episode_id", "")))
        values = grouped.get(key, [])
        if not values:
            continue
        center = _safe_int(transition.get("center_step", 0))
        by_step = {_safe_int(row.get("step_index", 0)): row for row in values}
        before = by_step.get(center - 1)
        after = by_step.get(center)
        if before is None or after is None:
            continue
        delta = _observation_vector(after) - _observation_vector(before)
        all_deltas.append(np.abs(delta))
        for index, value in enumerate(delta):
            feature = feature_map[index] if index < len(feature_map) else {}
            rows.append(
                {
                    "transition_id": str(transition.get("transition_id", "")),
                    "split": str(transition.get("split", "")),
                    "episode_id": str(transition.get("episode_id", "")),
                    "side": str(transition.get("side", "")),
                    "center_step": center,
                    "feature_index": index,
                    "feature_name": str(feature.get("feature_name", f"feature_{index}")),
                    "feature_group": "obstacle_geometry" if index in FEATURE_GROUPS["obstacle_geometry"] else "goal_geometry" if index in FEATURE_GROUPS["goal_geometry"] else "velocity" if index in FEATURE_GROUPS["velocity"] else "other",
                    "value_before": float(_observation_vector(before)[index]),
                    "value_after": float(_observation_vector(after)[index]),
                    "delta": float(value),
                    "absolute_delta": float(abs(value)),
                }
            )
    summary: list[dict[str, Any]] = []
    if all_deltas:
        matrix = np.stack(all_deltas)
        for index in range(NODE_DIM):
            feature = feature_map[index] if index < len(feature_map) else {}
            summary.append(
                {
                    "feature_index": index,
                    "feature_name": str(feature.get("feature_name", f"feature_{index}")),
                    "mean_absolute_transition_delta": float(np.mean(matrix[:, index])),
                    "median_absolute_transition_delta": float(np.median(matrix[:, index])),
                    "transition_count": int(matrix.shape[0]),
                    "obstacle_related": int(index in FEATURE_GROUPS["obstacle_geometry"]),
                    "goal_related": int(index in FEATURE_GROUPS["goal_geometry"]),
                    "velocity_related": int(index in FEATURE_GROUPS["velocity"]),
                }
            )
        summary.sort(key=lambda row: (-_finite(row.get("mean_absolute_transition_delta")), _safe_int(row.get("feature_index"))))
    return rows, summary


def _coverage_rows(records: Sequence[Mapping[str, Any]], *, split_label: str, source_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return phase and phase×scenario×side coverage at episode granularity."""

    phase_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    detail_groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        phase = _phase_for_analysis(record)
        phase_groups[phase].append(record)
        detail_groups[(phase, str(record.get("kind", "")), str(record.get("category", "")), str(record.get("side", "")), split_label)].append(record)
    phase_rows: list[dict[str, Any]] = []
    for phase in list(PHASES) + ["BOUNDARY_ONSET", "BYPASS"]:
        selected = phase_groups.get(phase, [])
        per_episode = Counter(str(record.get("episode_key", _episode_key(record))) for record in selected)
        phase_rows.append(
            {
                "source": source_label,
                "split": split_label,
                "phase": phase,
                "samples": len(selected),
                "unique_episodes": len(per_episode),
                "unique_scenarios": _unique_count(selected, "scenario_hash"),
                "unique_seeds": len({str(record.get("scenario_seed", "")) for record in selected if str(record.get("scenario_seed", ""))}),
                "mean_samples_per_episode": _mean(per_episode.values()),
                "median_samples_per_episode": _median(per_episode.values()),
                "min_samples_per_episode": min(per_episode.values(), default=0),
                "max_samples_per_episode": max(per_episode.values(), default=0),
                "unique_pair_ids": _unique_count(selected, "pair_id"),
            }
        )
    detail_rows: list[dict[str, Any]] = []
    for (phase, kind, category, side, split), selected in sorted(detail_groups.items()):
        detail_rows.append(
            {
                "source": source_label,
                "split": split,
                "phase": phase,
                "kind": kind,
                "category": category,
                "side": side,
                "samples": len(selected),
                "unique_episodes": _unique_count(selected, "episode_key"),
                "unique_scenarios": _unique_count(selected, "scenario_hash"),
                "unique_seeds": len({str(record.get("scenario_seed", "")) for record in selected if str(record.get("scenario_seed", ""))}),
                "unique_pair_ids": _unique_count(selected, "pair_id"),
                "episodes": ",".join(sorted({str(record.get("episode_id", "")) for record in selected})),
            }
        )
    return phase_rows, detail_rows


def _load_frozen_actor_trainer(checkpoint: Path, *, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    from .single_uav_checkpoint_evaluation import _load_frozen_trainer, _state_dict_hash
    from .single_uav_obstacle_dagger_stage2br import _stage2br_config_from_payload

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _stage2br_config_from_payload(payload, output_dir=output_dir, command_line=f"{STAGE_NAME} tiny overfit audit")
    trainer, identity, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    policy_config = trainer.policy.config
    return trainer, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "policy_hash": _state_dict_hash(trainer.policy.state_dict()),
        "loaded_policy_hash": identity.get("loaded_policy_hash", ""),
        "node_feature_dim": int(config.node_feature_dim),
        "uav_count": int(config.uav_count),
        "encoder_type": str(getattr(policy_config, "encoder_type", "gat")),
        "gat_layers": int(getattr(policy_config, "gat_layers", 0)),
        "gat_hidden_dim": int(getattr(policy_config, "gat_hidden_dim", 0)),
        "gat_output_dim": int(getattr(policy_config, "gat_output_dim", 0)),
        "actor_hidden_dim": int(getattr(policy_config, "actor_hidden_dim", 0)),
        "max_delta_v": float(getattr(policy_config, "max_delta_v", float("nan"))),
        "delta_component_limit": float(trainer.policy._delta_component_limit()),
        "include_remaining_time": bool(config.include_remaining_time),
    }


def _set_tiny_scope(policy: Any, scope: str) -> tuple[str, ...]:
    """Select diagnostic trainable parameters; no persistent policy is saved."""

    names: list[str] = []
    for name, parameter in policy.named_parameters():
        selected = False
        if scope == "current_mean_path":
            selected = name.startswith(("gat.", "mlp_flat_encoder.", "actor_body.", "delta_mean_head."))
        elif scope == "actor_head":
            selected = name.startswith("delta_mean_head.")
        elif scope == "first_gat_layer_plus_actor_body_and_head":
            selected = name.startswith(("gat.layers.0.", "actor_body.", "delta_mean_head."))
        elif scope == "full_actor":
            selected = not name.startswith("critic.") and name != "delta_logstd"
        else:
            raise ValueError(f"unknown tiny-overfit scope: {scope}")
        parameter.requires_grad_(selected)
        if selected:
            names.append(name)
    if not names:
        raise RuntimeError(f"tiny-overfit scope {scope} selected no parameters")
    return tuple(names)


def _select_tiny_records(records: Sequence[Mapping[str, Any]], *, limit_per_phase_side: int = 2) -> list[dict[str, Any]]:
    """Select a tiny, episode-disjoint phase/mirror set for in-memory fitting."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        phase = str(record.get("canonical_phase") or _phase_for_analysis(record))
        if phase not in {"NORMAL", "BYPASS_LEFT", "BYPASS_RIGHT", "POST_BYPASS", "RECOVERY"}:
            continue
        side = str(record.get("side", "none"))
        groups[(phase, side)].append(dict(record))
    selected: list[dict[str, Any]] = []
    used_episodes: set[str] = set()
    order = [
        ("NORMAL", "none"),
        ("BYPASS_LEFT", "left"),
        ("BYPASS_RIGHT", "right"),
        ("POST_BYPASS", "left"),
        ("POST_BYPASS", "right"),
        ("RECOVERY", "left"),
        ("RECOVERY", "right"),
    ]
    for key in order:
        candidates = sorted(groups.get(key, []), key=lambda row: (str(row.get("episode_key", "")), _safe_int(row.get("step_index", 0))))
        episode_count = 0
        for record in candidates:
            episode = str(record.get("episode_key", ""))
            if episode in used_episodes:
                continue
            selected.append(record)
            used_episodes.add(episode)
            episode_count += 1
            if episode_count >= int(limit_per_phase_side):
                break
    # If strict one-record-per-episode selection is too sparse in a phase,
    # fill from distinct steps only after the primary mirror/phase coverage is
    # established.  The first branch normally yields 10-14 samples.
    if len(selected) < 8:
        for record in records:
            if str(record.get("canonical_phase", "")) in {"BYPASS_LEFT", "BYPASS_RIGHT", "RECOVERY", "NORMAL"}:
                selected.append(dict(record))
                if len(selected) >= 12:
                    break
    return selected


def _decompose_action_for_record(record: Mapping[str, Any], action: np.ndarray) -> dict[str, float]:
    direction = _goal_direction(record)
    radial = float(np.dot(action, direction))
    velocity = _estimated_velocity_from_observation(record)
    speed = float(np.linalg.norm(velocity))
    velocity_direction = velocity / speed if speed > 1.0e-12 else np.zeros(3, dtype=float)
    tangent = action - radial * direction
    return {
        "radial": radial,
        "tangential": float(np.linalg.norm(tangent)),
        "braking": max(0.0, -float(np.dot(action, velocity_direction))),
        "radial_positive": int(radial > 1.0e-9),
        "lateral_sign": int(np.sign(action[1])) if abs(float(action[1])) > 1.0e-9 else 0,
    }


def _tiny_overfit(train_records: Sequence[Mapping[str, Any]], *, checkpoint: Path, output_dir: Path, seed: int, smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from .single_uav_bc_v2_dagger import _batch_actor_physical_mean

    samples = _select_tiny_records(train_records, limit_per_phase_side=1 if smoke else 2)
    if len(samples) < 8:
        raise ValueError(f"tiny-overfit requires at least 8 diagnostic samples, got {len(samples)}")
    scopes = (
        "current_mean_path",
        "actor_head",
        "first_gat_layer_plus_actor_body_and_head",
        "full_actor",
    )
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "samples": len(samples),
        "sample_episode_count": _unique_count(samples, "episode_key"),
        "sample_phase_counts": dict(Counter(str(row.get("canonical_phase", "")) for row in samples)),
        "scope_epochs": 80 if smoke else 220,
        "learning_rate": 1.0e-3,
        "persistent_checkpoint_written": False,
    }
    for scope_index, scope in enumerate(scopes):
        torch.manual_seed(int(seed) + scope_index)
        trainer, load_meta = _load_frozen_actor_trainer(checkpoint, output_dir=output_dir)
        selected_names = _set_tiny_scope(trainer.policy, scope)
        parameters = [parameter for parameter in trainer.policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
        observations = [row["observation"] for row in samples]
        targets = torch.as_tensor(np.stack([_action_vector(row) for row in samples]), dtype=torch.float32, device=trainer.policy.device)
        initial_predictions = _batch_actor_physical_mean(trainer, observations).detach().cpu().numpy().astype(float)
        initial_mse = float(np.mean((initial_predictions - targets.detach().cpu().numpy()) ** 2))
        epochs = 80 if smoke else 220
        for _epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            predictions = _batch_actor_physical_mean(trainer, observations)
            loss = torch.mean((predictions - targets) ** 2)
            loss.backward()
            optimizer.step()
        final_predictions = _batch_actor_physical_mean(trainer, observations).detach().cpu().numpy().astype(float)
        final_mse = float(np.mean((final_predictions - targets.detach().cpu().numpy()) ** 2))
        phase_rows: dict[str, list[bool]] = defaultdict(list)
        radial_rows: list[bool] = []
        lateral_rows: list[bool] = []
        for record, prediction in zip(samples, final_predictions):
            predicted = _decompose_action_for_record(record, prediction)
            teacher = _action_decomposition(record)
            phase_rows[str(record.get("canonical_phase", ""))].append(bool(np.linalg.norm(prediction - _action_vector(record)) <= 0.15))
            radial_rows.append(bool(predicted["radial_positive"] == teacher["teacher_radial_positive"] or abs(predicted["radial"]) <= 1.0e-9 and abs(teacher["teacher_radial_action_mps"]) <= 1.0e-9))
            lateral_rows.append(bool(predicted["lateral_sign"] == int(np.sign(teacher["teacher_action_y_mps"]))))
        rows.append(
            {
                "scope": scope,
                "samples": len(samples),
                "unique_episodes": _unique_count(samples, "episode_key"),
                "trainable_parameter_count": int(sum(int(parameter.numel()) for parameter in parameters)),
                "trainable_parameter_names_count": len(selected_names),
                "initial_mse": initial_mse,
                "final_mse": final_mse,
                "mse_reduction": initial_mse - final_mse,
                "action_accuracy_at_0.15mps": _mean([value for values in phase_rows.values() for value in values]),
                "radial_sign_accuracy": _mean(radial_rows),
                "lateral_sign_accuracy": _mean(lateral_rows),
                "phase_action_accuracy": json.dumps({phase: _mean(values) for phase, values in sorted(phase_rows.items())}, sort_keys=True),
                "checkpoint_sha256": load_meta["checkpoint_sha256"],
                "policy_hash_before": load_meta["policy_hash"],
                "persistent_checkpoint_written": 0,
                "notes": "in-memory diagnostic only; critic, log_std, normalizer and existing checkpoint remain untouched",
            }
        )
    return rows, meta


def _probe_score(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], train_z: np.ndarray, validation_z: np.ndarray, *, model_name: str, seed: int) -> tuple[float, float, str]:
    y_train = np.asarray([str(row.get("canonical_phase", "")) for row in train_records])
    y_validation = np.asarray([str(row.get("canonical_phase", "")) for row in validation_records])
    if len(set(y_train)) < 2 or len(set(y_validation)) < 2:
        return float("nan"), float("nan"), "insufficient phase classes"
    try:
        if model_name == "linear_probe":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=int(seed), solver="lbfgs")
        elif model_name == "5-NN":
            from sklearn.neighbors import KNeighborsClassifier

            model = KNeighborsClassifier(n_neighbors=min(5, len(train_z)), weights="distance")
        else:
            from sklearn.neural_network import MLPClassifier

            model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=int(seed), early_stopping=False)
        model.fit(train_z, y_train)
        predictions = [str(value) for value in model.predict(validation_z)]
        accuracy = float(np.mean(np.asarray(predictions) == y_validation))
        episode_accuracy = _episode_majority_accuracy(validation_records, predictions)
        return accuracy, episode_accuracy, "ok"
    except Exception as exc:
        if model_name == "5-NN":
            predictions = _fallback_knn_predict(train_z, y_train, validation_z, k=5)
        else:
            predictions = _fallback_torch_probe_predict(train_z, y_train, validation_z, nonlinear=model_name == "nonlinear_probe", seed=seed)
        accuracy = float(np.mean(np.asarray(predictions) == y_validation)) if predictions else float("nan")
        return accuracy, _episode_majority_accuracy(validation_records, predictions), f"fallback after {type(exc).__name__}: {exc}"


def _feature_ablation(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_records = _core_records(train_records)
    validation_records = _core_records(validation_records)
    train_x = _feature_matrix(train_records)
    validation_x = _feature_matrix(validation_records)
    train_z, validation_z, _ = _standardize(train_x, validation_x)
    rows: list[dict[str, Any]] = []
    ablations = {"none": tuple(), **FEATURE_GROUPS}
    for removed_name, removed_indices in ablations.items():
        masked_train = train_z.copy()
        masked_validation = validation_z.copy()
        for index in removed_indices:
            if 0 <= int(index) < NODE_DIM:
                masked_train[:, int(index)] = 0.0
                masked_validation[:, int(index)] = 0.0
        for model_name in ("linear_probe", "5-NN", "nonlinear_probe"):
            accuracy, episode_accuracy, status = _probe_score(train_records, validation_records, masked_train, masked_validation, model_name=model_name, seed=seed)
            rows.append(
                {
                    "removed_group": removed_name,
                    "removed_feature_indices": ",".join(str(index) for index in removed_indices),
                    "remaining_feature_count": NODE_DIM - len(set(removed_indices)),
                    "model": model_name,
                    "validation_accuracy": accuracy,
                    "validation_episode_majority_accuracy": episode_accuracy,
                    "status": status,
                    "notes": "features are masked to the train standardization mean; boundary group is empty because no explicit boundary-clearance node feature exists",
                }
            )
    metadata = {
        "groups": {name: list(indices) for name, indices in FEATURE_GROUPS.items()},
        "boundary_direct_feature_available": False,
        "radial_tangential_velocity_direct_feature_available": False,
        "radial_tangential_velocity_ablation_uses_velocity_indices": True,
    }
    return rows, metadata


def _mirror_recovery_audit(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        pair = str(record.get("pair_id", ""))
        if not pair:
            continue
        grouped[(str(record.get("split", "")), pair)][str(record.get("side", ""))].append(record)
    rows: list[dict[str, Any]] = []
    for (split, pair_id), sides in sorted(grouped.items()):
        left = sorted(sides.get("left", []), key=lambda row: _safe_int(row.get("step_index", 0)))
        right = sorted(sides.get("right", []), key=lambda row: _safe_int(row.get("step_index", 0)))
        if not left or not right:
            continue
        right_by_step = {_safe_int(row.get("step_index", 0)): row for row in right}
        compared = []
        bypass_compared = []
        recovery_compared = []
        for left_row in left:
            step = _safe_int(left_row.get("step_index", 0))
            right_row = right_by_step.get(step)
            if right_row is None:
                continue
            left_obs = _observation_vector(left_row)
            right_obs = _observation_vector(right_row)
            expected = left_obs.copy()
            # Stage-2A's body-frame obstacle lateral feature is the explicit
            # mirror coordinate; the other 41 features are not assumed to be
            # exact mirrors because EKF/noise and world y can differ.
            expected[40] *= -1.0
            obs_error = float(np.mean(np.abs(right_obs[[39, 40, 41]] - expected[[39, 40, 41]])))
            left_action = _action_decomposition(left_row)
            right_action = _action_decomposition(right_row)
            left_y = int(np.sign(left_action["teacher_action_y_mps"]))
            right_y = int(np.sign(right_action["teacher_action_y_mps"]))
            tangential_opposite = int(left_y != 0 and right_y != 0 and left_y == -right_y)
            radial_difference = abs(float(left_action["teacher_radial_action_mps"] - right_action["teacher_radial_action_mps"]))
            phase_left = str(left_row.get("canonical_phase", ""))
            phase_right = str(right_row.get("canonical_phase", ""))
            phase_class = "bypass" if phase_left in {"BYPASS_LEFT", "BYPASS_RIGHT"} and phase_right in {"BYPASS_LEFT", "BYPASS_RIGHT"} else "recovery" if phase_left == "RECOVERY" and phase_right == "RECOVERY" else "other"
            compared.append((obs_error, tangential_opposite, radial_difference))
            if phase_class == "bypass":
                bypass_compared.append((obs_error, tangential_opposite, radial_difference))
            if phase_class == "recovery":
                recovery_compared.append((obs_error, tangential_opposite, radial_difference))
        if not compared:
            continue
        transition_rows = [row for row in records if str(row.get("split", "")) == split and str(row.get("pair_id", "")) == pair_id]
        transition_by_episode: dict[str, int] = {}
        for side in ("left", "right"):
            side_rows = [row for row in transition_rows if str(row.get("side", "")) == side]
            side_rows.sort(key=lambda row: _safe_int(row.get("step_index", 0)))
            for index in range(1, len(side_rows)):
                previous = str(side_rows[index - 1].get("canonical_phase", ""))
                current = str(side_rows[index].get("canonical_phase", ""))
                if previous in {"BYPASS_LEFT", "BYPASS_RIGHT"} and current in {"POST_BYPASS", "RECOVERY", "BOUNDARY_ONSET"}:
                    transition_by_episode[side] = _safe_int(side_rows[index].get("step_index", 0))
                    break
        transition_delta = float("nan")
        if "left" in transition_by_episode and "right" in transition_by_episode:
            transition_delta = abs(float(transition_by_episode["left"] - transition_by_episode["right"]))
        rows.append(
            {
                "split": split,
                "pair_id": pair_id,
                "left_episode_id": str(left[0].get("episode_id", "")),
                "right_episode_id": str(right[0].get("episode_id", "")),
                "aligned_steps": len(compared),
                "bypass_aligned_steps": len(bypass_compared),
                "recovery_aligned_steps": len(recovery_compared),
                "mean_obstacle_geometry_mirror_error": _mean(value[0] for value in compared),
                "mean_bypass_obstacle_geometry_mirror_error": _mean(value[0] for value in bypass_compared),
                "mean_recovery_obstacle_geometry_mirror_error": _mean(value[0] for value in recovery_compared),
                "bypass_teacher_tangential_sign_opposite_rate": _mean(value[1] for value in bypass_compared),
                "recovery_teacher_tangential_sign_opposite_rate": _mean(value[1] for value in recovery_compared),
                "mean_radial_action_difference_mps": _mean(value[2] for value in compared),
                "left_transition_step": transition_by_episode.get("left", ""),
                "right_transition_step": transition_by_episode.get("right", ""),
                "absolute_transition_timing_difference_steps": transition_delta,
                "notes": "mirror error is computed on Stage-2A obstacle geometry indices 39:42; phase-aligned rows are diagnostic, not training samples",
            }
        )
    metadata = {
        "paired_split_count": len(rows),
        "paired_train_count": sum(row.get("split") == "train" for row in rows),
        "paired_validation_count": sum(row.get("split") == "validation" for row in rows),
    }
    return rows, metadata


def _plot_phase_embedding(records: Sequence[Mapping[str, Any]], plot_path: Path, *, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = _core_records(records)
    if not selected:
        return
    x = _feature_matrix(selected)
    z, _, _ = _standardize(x)
    try:
        from sklearn.decomposition import PCA

        embedded = PCA(n_components=2, random_state=int(seed)).fit_transform(z)
    except Exception:
        _, _, vh = np.linalg.svd(z - np.mean(z, axis=0), full_matrices=False)
        embedded = (z - np.mean(z, axis=0)) @ vh[:2].T
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    palette = {
        "NORMAL": "#0072B2",
        "APPROACH_RISK": "#56B4E9",
        "BYPASS_LEFT": "#D55E00",
        "BYPASS_RIGHT": "#E69F00",
        "POST_BYPASS": "#CC79A7",
        "RECOVERY": "#009E73",
        "GOAL_APPROACH": "#F0E442",
    }
    for phase in PHASES:
        indices = [index for index, row in enumerate(selected) if str(row.get("canonical_phase")) == phase]
        if not indices:
            continue
        train_indices = [index for index in indices if str(selected[index].get("split")) == "train"]
        validation_indices = [index for index in indices if str(selected[index].get("split")) == "validation"]
        color = palette.get(phase, "#666666")
        if train_indices:
            axis.scatter(embedded[train_indices, 0], embedded[train_indices, 1], s=12, alpha=0.35, color=color, label=f"train {phase}")
        if validation_indices:
            axis.scatter(embedded[validation_indices, 0], embedded[validation_indices, 1], s=30, alpha=0.85, marker="x", color=color, label=f"validation {phase}")
    axis.set_title("Actor-visible phase embedding (PCA diagnostic)")
    axis.set_xlabel("PC1 of standardized 42D node features")
    axis.set_ylabel("PC2 of standardized 42D node features")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2, loc="best")
    figure.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_confusion(confusion_rows: Sequence[Mapping[str, Any]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = "linear_probe"
    selected = [row for row in confusion_rows if row.get("model") == model and row.get("level") == "step"]
    labels = [phase for phase in PHASES if any(str(row.get("true_phase")) == phase or str(row.get("predicted_phase")) == phase for row in selected)]
    if not labels:
        labels = list(PHASES)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    positions = {label: index for index, label in enumerate(labels)}
    for row in selected:
        actual = str(row.get("true_phase"))
        predicted = str(row.get("predicted_phase"))
        if actual in positions and predicted in positions:
            matrix[positions[actual], positions[predicted]] += _safe_int(row.get("count", 1), 1)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, shrink=0.8)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Predicted phase")
    axis.set_ylabel("True phase")
    axis.set_title("Bypass/recovery phase confusion (linear probe)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.5 else "black", fontsize=8)
    figure.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_conflicts(conflict_rows: Sequence[Mapping[str, Any]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = {"BYPASS_LEFT__RECOVERY": "#D55E00", "BYPASS_RIGHT__RECOVERY": "#E69F00", "BYPASS_LEFT__BYPASS_RIGHT": "#56B4E9", "NORMAL__RECOVERY": "#009E73"}
    for conflict_type in sorted({str(row.get("conflict_type", "")) for row in conflict_rows}):
        selected = [row for row in conflict_rows if str(row.get("conflict_type", "")) == conflict_type]
        if not selected:
            continue
        x = [_finite(row.get("observation_distance_standardized")) for row in selected]
        y = [_finite(row.get("teacher_action_distance_mps")) for row in selected]
        close = [bool(_safe_int(row.get("strong_observation_target_conflict", 0))) for row in selected]
        axis.scatter(x, y, s=[38 if item else 14 for item in close], alpha=0.65, color=colors.get(conflict_type, "#666666"), label=conflict_type)
    axis.set_xlabel("Standardized observation distance")
    axis.set_ylabel("Teacher action distance (m/s delta-v)")
    axis.set_title("Nearest-neighbor teacher-target conflict audit")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _phase_code(phase: str) -> int:
    order = {phase: index for index, phase in enumerate(("BYPASS_LEFT", "BYPASS_RIGHT", "POST_BYPASS", "RECOVERY", "BOUNDARY_ONSET"))}
    return order.get(str(phase), -1)


def _plot_transition_windows(transition_rows: Sequence[Mapping[str, Any]], plot_path: Path, *, side: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in transition_rows if str(row.get("side")) == side]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    metrics = (
        ("goal_distance_m", "Goal distance (m)"),
        ("speed_mps", "Speed (m/s)"),
        ("teacher_radial_action_mps", "Teacher radial action (m/s delta-v)"),
        ("teacher_tangential_action_mps", "Teacher tangential action (m/s delta-v)"),
    )
    for axis, (field, ylabel) in zip(axes.flat, metrics):
        by_offset: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            value = _finite(row.get(field))
            if np.isfinite(value):
                by_offset[_safe_int(row.get("offset"))].append(value)
        offsets = sorted(by_offset)
        means = [_mean(by_offset[offset]) for offset in offsets]
        if offsets:
            axis.plot(offsets, means, marker="o", linewidth=1.8, color="#0072B2")
        axis.axvline(0.0, color="#D55E00", linestyle="--", linewidth=1.2)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Transition offset (step)")
    axes[-1, 1].set_xlabel("Transition offset (step)")
    figure.suptitle(f"Bypass → post-bypass/recovery transition window ({side})")
    figure.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_coverage(coverage_rows: Sequence[Mapping[str, Any]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [phase for phase in list(PHASES) + ["BOUNDARY_ONSET"] if any(str(row.get("phase")) == phase and str(row.get("source")) == "balanced_jsonl" for row in coverage_rows)]
    train = {str(row.get("phase")): _safe_int(row.get("samples")) for row in coverage_rows if row.get("source") == "balanced_jsonl" and row.get("split") == "train"}
    validation = {str(row.get("phase")): _safe_int(row.get("samples")) for row in coverage_rows if row.get("source") == "balanced_jsonl" and row.get("split") == "validation"}
    x = np.arange(len(phases))
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.bar(x - 0.18, [train.get(phase, 0) for phase in phases], width=0.36, label="train")
    axis.bar(x + 0.18, [validation.get(phase, 0) for phase in phases], width=0.36, label="validation")
    axis.set_xticks(x, phases, rotation=35, ha="right")
    axis.set_ylabel("Balanced demonstration samples")
    axis.set_title("Phase/state coverage")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_tiny_overfit(tiny_rows: Sequence[Mapping[str, Any]], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scopes = [str(row.get("scope")) for row in tiny_rows]
    initial = [_finite(row.get("initial_mse")) for row in tiny_rows]
    final = [_finite(row.get("final_mse")) for row in tiny_rows]
    x = np.arange(len(scopes))
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    axis.bar(x - 0.18, initial, width=0.36, label="initial MSE")
    axis.bar(x + 0.18, final, width=0.36, label="final MSE")
    axis.set_xticks(x, scopes, rotation=30, ha="right")
    axis.set_ylabel("Physical delta-v MSE")
    axis.set_title("Tiny-overfit diagnostic by Actor update scope")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _binary_probe_summary(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], left: str, right: str, *, seed: int) -> dict[str, Any]:
    selected_train = [record for record in train_records if str(record.get("canonical_phase", "")) in {left, right}]
    selected_validation = [record for record in validation_records if str(record.get("canonical_phase", "")) in {left, right}]
    if not selected_train or not selected_validation:
        return {"left": left, "right": right, "status": "missing_phase", "train_samples": len(selected_train), "validation_samples": len(selected_validation)}
    train_x = _feature_matrix(selected_train)
    validation_x = _feature_matrix(selected_validation)
    train_z, validation_z, _ = _standardize(train_x, validation_x)
    y_train = np.asarray([str(record.get("canonical_phase", "")) for record in selected_train])
    y_validation = np.asarray([str(record.get("canonical_phase", "")) for record in selected_validation])
    result: dict[str, Any] = {
        "left": left,
        "right": right,
        "status": "ok",
        "train_samples": len(selected_train),
        "validation_samples": len(selected_validation),
        "train_episodes": _unique_count(selected_train, "episode_key"),
        "validation_episodes": _unique_count(selected_validation, "episode_key"),
    }
    for model_name in ("linear_probe", "nonlinear_probe"):
        accuracy, episode_accuracy, status = _probe_score(selected_train, selected_validation, train_z, validation_z, model_name=model_name, seed=seed)
        result[f"{model_name}_accuracy"] = accuracy
        result[f"{model_name}_episode_majority_accuracy"] = episode_accuracy
        result[f"{model_name}_status"] = status
    return result


def _aggregate_benchmark_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the existing rerun4 frozen benchmark for context only."""

    output: list[dict[str, Any]] = []
    policies = sorted({str(row.get("policy", "")) for row in rows})
    for policy in policies:
        for scenario in ("normal_far_distance", "static_obstacle"):
            selected = [row for row in rows if str(row.get("policy", "")) == policy and str(row.get("scenario", "")) == scenario]
            if not selected:
                continue
            count = len(selected)
            item: dict[str, Any] = {
                "policy": policy,
                "scenario": scenario,
                "episodes": count,
                "entered_8m": sum(_safe_int(row.get("entered_8m")) for row in selected),
                "entered_5m": sum(_safe_int(row.get("entered_5m")) for row in selected),
                "entered_3m": sum(_safe_int(row.get("entered_3m")) for row in selected),
                "entered_1m": sum(_safe_int(row.get("entered_1m")) for row in selected),
                "stable_goal": sum(_safe_int(row.get("stable_goal")) for row in selected),
                "minimum_goal_distance_m_mean": _mean(row.get("minimum_goal_distance_m") for row in selected),
                "mean_speed_at_8m_entry_mps": _mean(row.get("mean_speed_at_8m_entry_mps") for row in selected),
                "mean_speed_at_3m_entry_mps": _mean(row.get("mean_speed_at_3m_entry_mps") for row in selected),
                "mean_speed_at_1m_entry_mps": _mean(row.get("mean_speed_at_1m_entry_mps") for row in selected),
                "mean_speed_within_1m_mps": _mean(row.get("mean_speed_within_1m_mps") for row in selected),
                "mean_maximum_consecutive_goal_hold_steps": _mean(row.get("maximum_consecutive_goal_hold_steps") for row in selected),
                "hold_steps_ge_1": sum(_safe_int(row.get("hold_steps_ge_1")) for row in selected),
                "hold_steps_ge_3": sum(_safe_int(row.get("hold_steps_ge_3")) for row in selected),
                "hold_steps_ge_5": sum(_safe_int(row.get("hold_steps_ge_5")) for row in selected),
                "pre_candidate_static_swept_conflict_steps": sum(_safe_int(row.get("pre_candidate_static_swept_conflict_steps")) for row in selected),
                "safety_projection_steps": sum(_safe_int(row.get("safety_projection_steps")) for row in selected),
                "boundary_projection_steps": sum(_safe_int(row.get("boundary_projection_steps")) for row in selected),
            }
            for key in ("entered_8m", "entered_5m", "entered_3m", "entered_1m", "stable_goal", "hold_steps_ge_1", "hold_steps_ge_3", "hold_steps_ge_5"):
                item[f"{key}_rate"] = item[key] / count if count else float("nan")
            output.append(item)
    return output


def _classify_causes(
    binary_metrics: Sequence[Mapping[str, Any]],
    conflict_summary: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    tiny_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_pair = {f"{row.get('left')}__{row.get('right')}": row for row in binary_metrics}
    pair_scores = []
    for left, right in (("BYPASS_LEFT", "RECOVERY"), ("BYPASS_RIGHT", "RECOVERY")):
        row = by_pair.get(f"{left}__{right}")
        if row:
            pair_scores.extend([_finite(row.get("linear_probe_accuracy")), _finite(row.get("nonlinear_probe_accuracy"))])
    pair_scores = [value for value in pair_scores if np.isfinite(value)]
    best_pair_probe = max(pair_scores, default=float("nan"))
    br_conflicts = [row for row in conflict_summary if "BYPASS" in str(row.get("conflict_type", "")) and "RECOVERY" in str(row.get("conflict_type", ""))]
    strong_rate = _mean(row.get("strong_conflict_rate") for row in br_conflicts)
    recovery_coverage = [row for row in coverage_rows if str(row.get("source")) == "balanced_jsonl" and str(row.get("split")) == "train" and str(row.get("phase")) == "RECOVERY"]
    recovery_unique_episodes = _safe_int(recovery_coverage[0].get("unique_episodes")) if recovery_coverage else 0
    recovery_samples = _safe_int(recovery_coverage[0].get("samples")) if recovery_coverage else 0
    static_samples = sum(_safe_int(row.get("samples")) for row in coverage_rows if str(row.get("source")) == "balanced_jsonl" and str(row.get("split")) == "train" and str(row.get("kind", "")) == "static_obstacle")
    if static_samples == 0:
        static_samples = sum(_safe_int(row.get("samples")) for row in coverage_rows if str(row.get("source")) == "balanced_jsonl" and str(row.get("split")) == "train" and str(row.get("phase")) not in {"NORMAL"})
    recovery_share = recovery_samples / max(static_samples, 1)
    current = next((row for row in tiny_rows if str(row.get("scope")) == "current_mean_path"), None)
    full = next((row for row in tiny_rows if str(row.get("scope")) == "full_actor"), None)
    current_final = _finite(current.get("final_mse")) if current else float("nan")
    full_final = _finite(full.get("final_mse")) if full else float("nan")
    aliasing = bool((np.isfinite(best_pair_probe) and best_pair_probe < 0.65) or (np.isfinite(strong_rate) and strong_rate >= 0.25))
    coverage = bool(recovery_coverage and (recovery_unique_episodes < 8 or recovery_share < 0.12))
    capacity = bool(np.isfinite(current_final) and np.isfinite(full_final) and current_final > 0.25 and full_final < current_final * 0.6)
    flags = {"observation_aliasing": aliasing, "demonstration_coverage": coverage, "actor_representation_or_update_scope": capacity}
    active = [name for name, enabled in flags.items() if enabled]
    if len(active) > 1:
        case = "Case D"
        primary = "observation_aliasing" if aliasing else active[0]
        secondary = "demonstration_coverage" if primary != "demonstration_coverage" and coverage else "actor_representation_or_update_scope" if primary != "actor_representation_or_update_scope" and capacity else "none"
        tertiary = next((name for name in active if name not in {primary, secondary}), "none")
    elif aliasing:
        case, primary, secondary, tertiary = "Case A", "observation_aliasing", "none", "none"
    elif coverage:
        case, primary, secondary, tertiary = "Case B", "demonstration_coverage", "none", "none"
    elif capacity:
        case, primary, secondary, tertiary = "Case C", "actor_representation_or_update_scope", "none", "none"
    else:
        case, primary, secondary, tertiary = "INCONCLUSIVE", "inconclusive", "none", "none"
    if case in {"Case A", "INCONCLUSIVE"}:
        next_step = "A" if case == "Case A" else "E"
    elif case == "Case B":
        next_step = "B"
    elif case == "Case C":
        next_step = "C"
    else:
        next_step = "D"
    ppo_bc_anchor = bool(case in {"Case B", "Case C", "Case D"} and not aliasing and np.isfinite(full_final) and full_final < 0.25)
    return {
        "case": case,
        "primary_cause": primary,
        "secondary_cause": secondary,
        "tertiary_cause": tertiary,
        "next_step_choice": next_step,
        "observation_aliasing_flag": aliasing,
        "demonstration_coverage_flag": coverage,
        "actor_representation_update_scope_flag": capacity,
        "bypass_recovery_probe_best_accuracy": best_pair_probe,
        "bypass_recovery_strong_conflict_rate": strong_rate,
        "recovery_unique_train_episodes": recovery_unique_episodes,
        "recovery_train_samples": recovery_samples,
        "recovery_train_share_of_static_balanced_samples": recovery_share,
        "tiny_current_mean_path_final_mse": current_final,
        "tiny_full_actor_final_mse": full_final,
        "ppo_plus_bc_anchor_now": ppo_bc_anchor,
        "interpretation": "This is a diagnostic classification from held-out probes, cross-episode conflicts, coverage and in-memory tiny-overfit; it is not a causal proof.",
    }


def _smoke_subset(records: Sequence[Mapping[str, Any]], *, max_episodes_per_phase: int = 4, max_rows_per_episode_phase: int = 8) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        phase = str(record.get("canonical_phase") or _phase_for_analysis(record))
        if phase in PHASES:
            grouped[(phase, str(record.get("episode_key", _episode_key(record))))].append(dict(record))
    selected: list[dict[str, Any]] = []
    episodes_by_phase: dict[str, int] = Counter()
    for (phase, _episode), values in sorted(grouped.items()):
        if episodes_by_phase[phase] >= int(max_episodes_per_phase):
            continue
        values.sort(key=lambda row: _safe_int(row.get("step_index", 0)))
        selected.extend(values[: int(max_rows_per_episode_phase)])
        episodes_by_phase[phase] += 1
    return selected


def _schema_validation(output_log_root: Path, output_plot_root: Path) -> dict[str, Any]:
    required_csv = (
        "observation_feature_map.csv",
        "phase_separability.csv",
        "phase_confusion_matrix.csv",
        "nearest_neighbor_conflicts.csv",
        "teacher_target_conflict_summary.csv",
        "phase_coverage.csv",
        "phase_scenario_coverage.csv",
        "transition_coverage.csv",
        "transition_feature_changes.csv",
        "transition_feature_change_summary.csv",
        "tiny_overfit_results.csv",
        "feature_ablation.csv",
        "mirror_recovery_audit.csv",
    )
    required_plots = (
        "phase_embedding.png",
        "bypass_recovery_confusion.png",
        "teacher_target_conflict.png",
        "transition_window_left.png",
        "transition_window_right.png",
        "phase_coverage.png",
        "tiny_overfit_comparison.png",
    )
    csv_status = {name: int((output_log_root / name).is_file() and (output_log_root / name).stat().st_size > 0) for name in required_csv}
    plot_status = {name: int((output_plot_root / name).is_file() and (output_plot_root / name).stat().st_size > 0) for name in required_plots}
    return {
        "required_csv": list(required_csv),
        "required_plots": list(required_plots),
        "csv_status": csv_status,
        "plot_status": plot_status,
        "all_required_csv_present": bool(all(csv_status.values())),
        "all_required_plots_present": bool(all(plot_status.values())),
        "ok": bool(all(csv_status.values()) and all(plot_status.values())),
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], *, limit: int = 50) -> str:
    if not rows:
        return "（无记录）"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in list(rows)[:limit]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float) and not np.isfinite(value):
                value = "nan"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_stage_report(
    path: Path,
    *,
    source_lock: Mapping[str, Any],
    phase_metadata: Mapping[str, Any],
    binary_metrics: Sequence[Mapping[str, Any]],
    conflict_summary: Sequence[Mapping[str, Any]],
    transition_summary: Sequence[Mapping[str, Any]],
    transition_feature_summary: Sequence[Mapping[str, Any]],
    tiny_rows: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
    benchmark_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    smoke: bool,
) -> None:
    report = [
        f"# {STAGE_NAME}",
        "",
        f"- mode: `{'smoke' if smoke else 'formal read-only audit'}`",
        "- fixed seed: `2026`",
        "- training/PPO/MAPPO updates: **0**",
        "- reward, curriculum, dynamics, action mapping, safety, boundary, collision, stable判据、horizon: **unchanged**",
        "- independent unit: episode/scenario; step rows are used only as within-episode diagnostics",
        "",
        "## Stage 0 source lock",
        "",
        f"- rerun4 data: `{source_lock.get('rerun4_data_root')}`",
        f"- rerun4 logs: `{source_lock.get('rerun4_log_root')}`",
        f"- baseline checkpoint: `{source_lock.get('baseline_checkpoint', {}).get('path')}`",
        f"- baseline SHA256: `{source_lock.get('baseline_checkpoint', {}).get('sha256')}`",
        f"- BC checkpoint: `{source_lock.get('bc_checkpoint', {}).get('path')}`",
        f"- BC SHA256: `{source_lock.get('bc_checkpoint', {}).get('sha256')}`",
        f"- migrated 42D checkpoint: `{source_lock.get('migrated_stage2a_checkpoint', {}).get('path')}`",
        f"- migrated 42D SHA256: `{source_lock.get('migrated_stage2a_checkpoint', {}).get('sha256')}`",
        f"- observation node dimensions: `{source_lock.get('demonstration_observation_node_dims')}`",
        f"- train/validation episode leakage: `{source_lock.get('train_validation_episode_leakage')}`",
        f"- train/validation scenario leakage: `{source_lock.get('train_validation_scenario_leakage')}`",
        f"- train/validation pair leakage: `{source_lock.get('train_validation_pair_leakage')}`",
        f"- truth-state control flags: `{source_lock.get('truth_state_read_for_control_values')}`",
        "",
        "## Observation contract",
        "",
        "主分析使用当前 one-UAV Actor 实际接收的冻结归一化 `node_features[42]`。空边特征以及 one-UAV action/bandwidth masks 在本数据中不随行为阶段变化，因此没有把它们当作可辨识证据；Actor 前向仍通过既有路径使用这些常量块。goal distance、radial/tangential velocity、obstacle bearing、passed/recovery 状态均没有作为显式 phase 输入，而是由观测或离线轨迹派生。",
        "",
        "## Phase definitions and separability",
        "",
        "`post_bypass` 仅表示已通过障碍物且尚未满足 recovery 条件；`recovery` 要求已有通过迹象、径向进度为正且横向动作衰减。`passed_obstacle` 与 `recovery` 不合并。",
        "",
        _markdown_table(binary_metrics, ("left", "right", "train_samples", "validation_samples", "linear_probe_accuracy", "nonlinear_probe_accuracy")),
        "",
        "Probe metrics are fitted on train episodes and evaluated on validation episodes. Confusion matrices contain both step-level display and episode-majority diagnostic views; episode-level aggregation is the independent-unit check.",
        "",
        "## Teacher-target conflict",
        "",
        _markdown_table(conflict_summary, ("conflict_type", "pairs", "observation_close_pairs", "action_conflict_pairs", "strong_observation_target_conflict_pairs", "strong_conflict_rate", "observation_distance_standardized_median", "teacher_action_distance_median_mps")),
        "",
        "close threshold = eligible cross-episode standardized observation-distance p10; action conflict threshold = eligible teacher-action-distance p75 or sign/left-right conflict. These are diagnostic thresholds, not training rules.",
        "",
        "## Transition and coverage",
        "",
        f"- detected bypass→post/recovery transitions: `{len(transition_summary)}`",
        _markdown_table(transition_summary, ("split", "episode_id", "side", "center_step", "from_phase", "to_phase", "window_rows")),
        "",
        "See `phase_coverage.csv`, `phase_scenario_coverage.csv`, and `transition_coverage.csv` for raw/balanced samples, episode/scenario/seed counts, side and category coverage.",
        "",
        "### Observation changes at the action-switch transition",
        "",
        _markdown_table(transition_feature_summary, ("feature_index", "feature_name", "mean_absolute_transition_delta", "median_absolute_transition_delta", "obstacle_related", "goal_related", "velocity_related"), limit=12),
        "",
        "## Tiny overfit",
        "",
        _markdown_table(tiny_rows, ("scope", "trainable_parameter_count", "initial_mse", "final_mse", "mse_reduction", "radial_sign_accuracy", "lateral_sign_accuracy")),
        "",
        "Tiny-overfit is in-memory only. Each scope reloads the same migrated baseline checkpoint; critic, log_std and normalizer are not updated, and no checkpoint is written.",
        "",
        "## Existing rerun4 benchmark context",
        "",
        _markdown_table(benchmark_rows, ("policy", "scenario", "episodes", "entered_8m", "entered_3m", "entered_1m", "stable_goal", "minimum_goal_distance_m_mean", "mean_maximum_consecutive_goal_hold_steps"), limit=30),
        "",
        "## Evidence-based classification",
        "",
        f"- case: **{classification.get('case')}**",
        f"- primary cause: **{classification.get('primary_cause')}**",
        f"- secondary cause: **{classification.get('secondary_cause')}**",
        f"- tertiary cause: **{classification.get('tertiary_cause')}**",
        f"- next step choice: **{classification.get('next_step_choice')}**",
        f"- observation aliasing flag: `{classification.get('observation_aliasing_flag')}`",
        f"- demonstration coverage flag: `{classification.get('demonstration_coverage_flag')}`",
        f"- Actor representation/update-scope flag: `{classification.get('actor_representation_update_scope_flag')}`",
        f"- PPO + BC anchor now: `{classification.get('ppo_plus_bc_anchor_now')}`",
        "",
        str(classification.get("interpretation", "")),
        "",
        "## Output schema",
        "",
        f"- schema validation: **{schema.get('ok')}**",
        f"- output directory: `{path.parent.resolve()}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def run_stage2br3_audit(
    *,
    rerun4_data_root: str | Path = DEFAULT_RERUN4_DATA_ROOT,
    rerun4_log_root: str | Path = DEFAULT_RERUN4_LOG_ROOT,
    output_data_root: str | Path = DEFAULT_DATA_ROOT,
    output_log_root: str | Path = DEFAULT_LOG_ROOT,
    output_plot_root: str | Path = DEFAULT_PLOT_ROOT,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    bc_checkpoint: str | Path = DEFAULT_BC_CHECKPOINT,
    migrated_checkpoint: str | Path = DEFAULT_MIGRATED_CHECKPOINT,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the complete read-only R3 audit and return the written summary."""

    rerun4_data_root = Path(rerun4_data_root)
    rerun4_log_root = Path(rerun4_log_root)
    output_data_root = Path(output_data_root)
    output_log_root = Path(output_log_root)
    output_plot_root = Path(output_plot_root)
    source_lock = build_source_lock(
        rerun4_data_root,
        rerun4_log_root,
        baseline_checkpoint=Path(baseline_checkpoint),
        bc_checkpoint=Path(bc_checkpoint),
        migrated_checkpoint=Path(migrated_checkpoint),
    )
    _prepare_roots(output_data_root, output_log_root, output_plot_root)
    train, validation, raw_audit = _load_audit_records(rerun4_data_root, rerun4_log_root)
    analysis_train = _smoke_subset(train) if smoke else train
    analysis_validation = _smoke_subset(validation) if smoke else validation
    layout = _load_feature_layout(rerun4_log_root / "baseline_observation_migration/stage2a_feature_layout.csv")
    feature_map = build_feature_map(layout)
    _write_csv(output_log_root / "observation_feature_map.csv", feature_map, [
        "index", "feature_name", "source", "estimated_or_truth", "truth_state_used_for_actor", "normalization", "physical_meaning", "available_to_actor", "obstacle_related", "goal_related", "velocity_related", "boundary_related", "direct_radial_velocity", "direct_tangential_velocity", "notes",
    ])

    phase_rows, confusion_rows, probe_metadata = _fit_probes(analysis_train, analysis_validation, seed=seed)
    binary_metrics = [
        _binary_probe_summary(analysis_train, analysis_validation, "BYPASS_LEFT", "RECOVERY", seed=seed),
        _binary_probe_summary(analysis_train, analysis_validation, "BYPASS_RIGHT", "RECOVERY", seed=seed),
        _binary_probe_summary(analysis_train, analysis_validation, "BYPASS_LEFT", "BYPASS_RIGHT", seed=seed),
        _binary_probe_summary(analysis_train, analysis_validation, "NORMAL", "RECOVERY", seed=seed),
    ]
    _write_csv(output_log_root / "phase_separability.csv", phase_rows)
    _write_csv(output_log_root / "phase_confusion_matrix.csv", confusion_rows)

    conflict_rows, conflict_summary, conflict_metadata = _nearest_neighbor_conflicts(analysis_train, analysis_validation)
    _write_csv(output_log_root / "nearest_neighbor_conflicts.csv", conflict_rows)
    _write_csv(output_log_root / "teacher_target_conflict_summary.csv", conflict_summary)

    transition_rows, transition_summary = _transition_windows(analysis_train + analysis_validation)
    _write_csv(output_log_root / "transition_coverage.csv", transition_rows)
    transition_feature_rows, transition_feature_summary = _transition_feature_changes(analysis_train + analysis_validation, transition_summary, feature_map)
    _write_csv(output_log_root / "transition_feature_changes.csv", transition_feature_rows)
    _write_csv(output_log_root / "transition_feature_change_summary.csv", transition_feature_summary)

    raw_records = []
    for row in raw_audit:
        raw = dict(row)
        raw["canonical_phase"] = _phase_for_analysis(raw)
        raw["episode_key"] = "|".join(_episode_key(raw))
        raw_records.append(raw)
    coverage_rows: list[dict[str, Any]] = []
    phase_scenario_rows: list[dict[str, Any]] = []
    for records, split_label, source_label in ((analysis_train, "train", "balanced_jsonl"), (analysis_validation, "validation", "balanced_jsonl")):
        phase_part, detail_part = _coverage_rows(records, split_label=split_label, source_label=source_label)
        coverage_rows.extend(phase_part)
        phase_scenario_rows.extend(detail_part)
    for split_label in ("train", "validation"):
        selected_raw = [row for row in raw_records if str(row.get("split", "")) == split_label]
        phase_part, detail_part = _coverage_rows(selected_raw, split_label=split_label, source_label="raw_audit_csv")
        coverage_rows.extend(phase_part)
        phase_scenario_rows.extend(detail_part)
    _write_csv(output_log_root / "phase_coverage.csv", coverage_rows)
    _write_csv(output_log_root / "phase_scenario_coverage.csv", phase_scenario_rows)

    tiny_rows, tiny_metadata = _tiny_overfit(analysis_train, checkpoint=Path(migrated_checkpoint), output_dir=output_data_root / "_tiny_overfit_in_memory", seed=seed, smoke=smoke)
    _write_csv(output_log_root / "tiny_overfit_results.csv", tiny_rows)
    ablation_rows, ablation_metadata = _feature_ablation(analysis_train, analysis_validation, seed=seed)
    _write_csv(output_log_root / "feature_ablation.csv", ablation_rows)
    mirror_rows, mirror_metadata = _mirror_recovery_audit(analysis_train + analysis_validation)
    _write_csv(output_log_root / "mirror_recovery_audit.csv", mirror_rows)

    _plot_phase_embedding(analysis_train + analysis_validation, output_plot_root / "phase_embedding.png", seed=seed)
    _plot_confusion(confusion_rows, output_plot_root / "bypass_recovery_confusion.png")
    _plot_conflicts(conflict_rows, output_plot_root / "teacher_target_conflict.png")
    _plot_transition_windows(transition_rows, output_plot_root / "transition_window_left.png", side="left")
    _plot_transition_windows(transition_rows, output_plot_root / "transition_window_right.png", side="right")
    _plot_coverage(coverage_rows, output_plot_root / "phase_coverage.png")
    _plot_tiny_overfit(tiny_rows, output_plot_root / "tiny_overfit_comparison.png")

    binary_metrics_for_classification = [row for row in binary_metrics if row.get("status") == "ok"]
    classification = _classify_causes(binary_metrics_for_classification, conflict_summary, coverage_rows, tiny_rows)
    try:
        evaluation_rows = _read_csv(rerun4_log_root / "evaluation_episode_metrics.csv")
    except Exception:
        evaluation_rows = []
    benchmark_rows = _aggregate_benchmark_rows(evaluation_rows)
    schema = _schema_validation(output_log_root, output_plot_root)
    summary = {
        "stage": STAGE_NAME,
        "status": "completed_read_only_diagnostic",
        "seed": int(seed),
        "smoke": bool(smoke),
        "training_performed": False,
        "ppo_updates": 0,
        "existing_mechanisms_modified": False,
        "source_lock": source_lock,
        "analysis_counts": {
            "train_records": len(analysis_train),
            "validation_records": len(analysis_validation),
            "train_episodes": _unique_count(analysis_train, "episode_key"),
            "validation_episodes": _unique_count(analysis_validation, "episode_key"),
            "raw_audit_records": len(raw_audit),
            "raw_audit_train_episodes": _unique_count([row for row in raw_records if row.get("split") == "train"], "episode_key"),
            "raw_audit_validation_episodes": _unique_count([row for row in raw_records if row.get("split") == "validation"], "episode_key"),
        },
        "probe_metadata": probe_metadata,
        "binary_probe_metrics": binary_metrics,
        "conflict_metadata": conflict_metadata,
        "transition_summary": {
            "transitions": len(transition_summary),
            "train": sum(str(row.get("split")) == "train" for row in transition_summary),
            "validation": sum(str(row.get("split")) == "validation" for row in transition_summary),
            "by_side": dict(Counter(str(row.get("side", "")) for row in transition_summary)),
            "feature_change_rows": len(transition_feature_rows),
            "top_changed_features": transition_feature_summary[:10],
        },
        "tiny_overfit_metadata": tiny_metadata,
        "ablation_metadata": ablation_metadata,
        "mirror_metadata": mirror_metadata,
        "classification": classification,
        "existing_rerun4_benchmark_context": benchmark_rows,
        "output_roots": {
            "data": str(output_data_root.resolve()),
            "logs": str(output_log_root.resolve()),
            "plots": str(output_plot_root.resolve()),
        },
        "schema_validation": schema,
        "required_outputs": {
            "csv": schema["required_csv"],
            "plots": schema["required_plots"],
        },
    }
    _write_json(output_data_root / "source_lock.json", source_lock)
    _write_json(output_data_root / "summary.json", summary)
    _write_json(output_log_root / "summary.json", summary)
    _write_stage_report(
        output_log_root / "stage_report.md",
        source_lock=source_lock,
        phase_metadata=probe_metadata,
        binary_metrics=binary_metrics,
        conflict_summary=conflict_summary,
        transition_summary=transition_summary,
        transition_feature_summary=transition_feature_summary,
        tiny_rows=tiny_rows,
        classification=classification,
        benchmark_rows=benchmark_rows,
        schema=schema,
        smoke=smoke,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--rerun4-data-root", type=Path, default=DEFAULT_RERUN4_DATA_ROOT)
    parser.add_argument("--rerun4-log-root", type=Path, default=DEFAULT_RERUN4_LOG_ROOT)
    parser.add_argument("--output-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--migrated-checkpoint", type=Path, default=DEFAULT_MIGRATED_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_stage2br3_audit(
        rerun4_data_root=args.rerun4_data_root,
        rerun4_log_root=args.rerun4_log_root,
        output_data_root=args.output_data_root,
        output_log_root=args.output_log_root,
        output_plot_root=args.output_plot_root,
        baseline_checkpoint=args.baseline_checkpoint,
        bc_checkpoint=args.bc_checkpoint,
        migrated_checkpoint=args.migrated_checkpoint,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps({"stage": STAGE_NAME, "classification": summary.get("classification"), "schema_ok": summary.get("schema_validation", {}).get("ok")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
