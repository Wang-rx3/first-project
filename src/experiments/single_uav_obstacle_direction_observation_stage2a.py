"""Stage 2A: opt-in UAV-relative static-obstacle direction observation.

This module is deliberately observation-only.  It adds three estimated,
UAV-relative obstacle-center coordinates to a new node-feature contract,
migrates a historical 39-dimensional checkpoint without changing its old
parameter slices, and runs paired probes.  It never calls ``MAPPOTrainer.update``
and never writes to an existing non-empty experiment directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..learning.mappo import MAPPOConfig, MAPPOTrainer, central_state_vector_dim
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..simulation.environment import MultiUAVJointEnv, SimulationConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.obstacle_model import SphereObstacle, point_obstacle_distance
from ..synthetic.episode_generator import _jsonable, _sha256, _write_csv, _write_json
from .formal import FormalExperimentConfig, _make_environment, _scale_seed
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_navigation_preserving_finetune import _read_checkpoint
from .single_uav_obstacle_observation_audit import (
    PROBE_CASES,
    probe_scenarios,
)
from .single_uav_static_obstacle_avoidance import (
    _normal_scenario,
    _static_pressure_scenario,
)


DEFAULT_BASELINE_CHECKPOINT = Path(
    "data/synthetic/single_uav_braking_progress025_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_STAGE1_CHECKPOINT = Path(
    "data/synthetic/single_uav_static_obstacle_avoidance_stage1_formal_seed2026/"
    "scale_1/checkpoints/final.pt"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_obstacle_direction_observation_stage2a_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_obstacle_direction_observation_stage2a_seed2026"
)

OLD_NODE_DIM = 39
NEW_NODE_DIM = 42
NEW_DIRECTION_INDICES = (39, 40, 41)
NEW_DIRECTION_FEATURES = (
    "relative_obstacle_dx_body_m",
    "relative_obstacle_dy_body_m",
    "relative_obstacle_dz_body_m",
)
PROBE_SEED = 2026
TOLERANCE = 1.0e-7


def _array_hash(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _difference(first: Any, second: Any) -> dict[str, float]:
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return {
        "l1": float(np.abs(delta).sum()),
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _ensure_empty_roots(*roots: Path) -> None:
    for root in roots:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty Stage 2A output root: {root.resolve()}"
            )
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)


def _checkpoint_config_payload(config: FormalExperimentConfig) -> dict[str, Any]:
    payload = _jsonable(asdict(config))
    payload["node_feature_dim"] = int(config.node_feature_dim)
    return payload


def _source_contract_report(
    payload: Mapping[str, Any],
    config: FormalExperimentConfig,
    checkpoint: Path,
) -> dict[str, Any]:
    trainer_state = payload.get("trainer_config")
    normalizer_state = payload.get("normalization_stats")
    central_state = payload.get("central_normalization_stats")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("source checkpoint is missing trainer_config")
    if not isinstance(normalizer_state, Mapping):
        raise ValueError("source checkpoint is missing normalization_stats")
    if not isinstance(central_state, Mapping):
        raise ValueError("source checkpoint is missing central_normalization_stats")
    node_state = dict(normalizer_state.get("normalizers", {})).get("node_features")
    node_mean = node_state.get("mean") if isinstance(node_state, Mapping) else None
    central_mean = central_state.get("mean")
    trainer_dim = int(trainer_state.get("node_feature_dim", -1))
    normalizer_dim = len(node_mean) if isinstance(node_mean, list) else -1
    central_dim = len(central_mean) if isinstance(central_mean, list) else -1
    if config.uav_count != 1:
        raise ValueError("Stage 2A requires a one-UAV source checkpoint")
    if config.single_uav_obstacle_direction_features_enabled:
        raise ValueError("source checkpoint must use the historical flag=false contract")
    if trainer_dim != OLD_NODE_DIM or normalizer_dim != OLD_NODE_DIM:
        raise ValueError(
            f"source checkpoint must be 39-dimensional; trainer={trainer_dim}, "
            f"normalizer={normalizer_dim}"
        )
    expected_central = central_state_vector_dim(
        1,
        OLD_NODE_DIM,
        int(trainer_state.get("edge_feature_dim", 11)),
        int(trainer_state.get("max_obstacles", config.max_obstacles)),
        bool(trainer_state.get("include_remaining_time", config.include_remaining_time)),
    )
    if central_dim != expected_central:
        raise ValueError(
            f"source central normalizer dimension mismatch: {central_dim} != {expected_central}"
        )
    return {
        "path": str(checkpoint.resolve()),
        "sha256": _sha256(checkpoint),
        "trainer_node_feature_dim": trainer_dim,
        "normalizer_node_feature_dim": normalizer_dim,
        "central_normalizer_dim": central_dim,
        "expected_central_normalizer_dim": expected_central,
        "direction_features_enabled": False,
        "checkpoint_policy_hash": _state_dict_hash(payload.get("policy_state_dict", {})),
    }


def migrate_observation_normalizer_state(
    source_state: Mapping[str, Any],
    *,
    old_node_dim: int = OLD_NODE_DIM,
    new_node_dim: int = NEW_NODE_DIM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append neutral entries while preserving every historical node statistic."""

    if new_node_dim < old_node_dim:
        raise ValueError("new node dimension cannot be smaller than old dimension")
    result = copy.deepcopy(dict(source_state))
    normalizers = result.get("normalizers")
    if not isinstance(normalizers, Mapping):
        raise ValueError("normalization state is missing normalizers")
    node = normalizers.get("node_features")
    if not isinstance(node, Mapping):
        raise ValueError("normalization state is missing node_features normalizer")
    mean = np.asarray(node.get("mean"), dtype=np.float64).reshape(-1)
    scale = np.asarray(node.get("scale"), dtype=np.float64).reshape(-1)
    count = np.asarray(node.get("count"), dtype=np.int64).reshape(-1)
    if not (mean.size == scale.size == count.size == old_node_dim):
        raise ValueError("source node normalizer has an unexpected dimension")
    added = new_node_dim - old_node_dim
    migrated_node = dict(node)
    migrated_node["mean"] = np.concatenate([mean, np.zeros(added, dtype=np.float64)]).tolist()
    migrated_node["scale"] = np.concatenate([scale, np.ones(added, dtype=np.float64)]).tolist()
    migrated_node["count"] = np.concatenate([count, np.zeros(added, dtype=np.int64)]).tolist()
    result["normalizers"] = dict(normalizers)
    result["normalizers"]["node_features"] = migrated_node
    report = {
        "old_node_dim": int(old_node_dim),
        "new_node_dim": int(new_node_dim),
        "old_mean_preserved_exact": bool(np.array_equal(mean, np.asarray(migrated_node["mean"])[:old_node_dim])),
        "old_scale_preserved_exact": bool(np.array_equal(scale, np.asarray(migrated_node["scale"])[:old_node_dim])),
        "old_count_preserved_exact": bool(np.array_equal(count, np.asarray(migrated_node["count"])[:old_node_dim])),
        "new_mean": [0.0] * added,
        "new_scale": [1.0] * added,
        "new_count": [0] * added,
        "new_entries_neutral": True,
        "other_normalizer_keys_preserved": sorted(
            str(key) for key in result["normalizers"] if str(key) != "node_features"
        ),
    }
    return result, report


def migrate_central_normalizer_state(
    source_state: Mapping[str, Any],
    *,
    uav_count: int,
    old_node_dim: int,
    new_node_dim: int,
    edge_feature_dim: int,
    max_obstacles: int,
    include_remaining_time: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Insert neutral direction slots in every centralized node block."""

    result = copy.deepcopy(dict(source_state))
    mean = np.asarray(result.get("mean"), dtype=np.float64).reshape(-1)
    scale = np.asarray(result.get("scale"), dtype=np.float64).reshape(-1)
    count = np.asarray(result.get("count"), dtype=np.int64).reshape(-1)
    old_dim = central_state_vector_dim(
        uav_count, old_node_dim, edge_feature_dim, max_obstacles, include_remaining_time
    )
    new_dim = central_state_vector_dim(
        uav_count, new_node_dim, edge_feature_dim, max_obstacles, include_remaining_time
    )
    if not (mean.size == scale.size == count.size == old_dim):
        raise ValueError(
            f"central normalizer has unexpected dimension: {mean.size}, expected {old_dim}"
        )
    old_nodes = uav_count * old_node_dim
    new_nodes = uav_count * new_node_dim
    added_per_node = new_node_dim - old_node_dim

    def expand(values: np.ndarray, neutral: float) -> np.ndarray:
        output = np.empty(new_dim, dtype=values.dtype)
        for index in range(uav_count):
            old_slice = slice(index * old_node_dim, (index + 1) * old_node_dim)
            new_slice = slice(index * new_node_dim, index * new_node_dim + old_node_dim)
            output[new_slice] = values[old_slice]
            output[index * new_node_dim + old_node_dim : (index + 1) * new_node_dim] = neutral
        output[new_nodes:] = values[old_nodes:]
        return output

    result["mean"] = expand(mean, 0.0).tolist()
    result["scale"] = expand(scale, 1.0).tolist()
    result["count"] = expand(count, 0).tolist()
    migrated_mean = np.asarray(result["mean"], dtype=np.float64)
    migrated_scale = np.asarray(result["scale"], dtype=np.float64)
    migrated_count = np.asarray(result["count"], dtype=np.int64)
    old_positions = []
    for index in range(uav_count):
        old_positions.extend(range(index * new_node_dim, index * new_node_dim + old_node_dim))
    old_positions.extend(range(new_nodes, new_dim))
    report = {
        "old_central_dim": int(old_dim),
        "new_central_dim": int(new_dim),
        "uav_count": int(uav_count),
        "old_node_dim": int(old_node_dim),
        "new_node_dim": int(new_node_dim),
        "inserted_neutral_slots_per_uav": int(added_per_node),
        "old_mean_preserved_exact": bool(np.array_equal(mean, migrated_mean[old_positions])),
        "old_scale_preserved_exact": bool(np.array_equal(scale, migrated_scale[old_positions])),
        "old_count_preserved_exact": bool(np.array_equal(count, migrated_count[old_positions])),
        "new_mean_slots_neutral": bool(np.all(migrated_mean[[index * new_node_dim + offset for index in range(uav_count) for offset in range(old_node_dim, new_node_dim)]] == 0.0)),
        "new_scale_slots_neutral": bool(np.all(migrated_scale[[index * new_node_dim + offset for index in range(uav_count) for offset in range(old_node_dim, new_node_dim)]] == 1.0)),
        "new_count_slots_neutral": bool(np.all(migrated_count[[index * new_node_dim + offset for index in range(uav_count) for offset in range(old_node_dim, new_node_dim)]] == 0)),
    }
    return result, report


def _migrate_parameter_tensor(
    name: str,
    source: torch.Tensor,
    target_template: torch.Tensor,
    *,
    old_node_dim: int,
    new_node_dim: int,
    uav_count: int,
    old_central_dim: int,
    new_central_dim: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Copy one parameter, expanding only input axes affected by node width."""

    source = source.detach().cpu()
    target = target_template.detach().cpu().clone()
    if tuple(source.shape) == tuple(target.shape):
        return source.clone(), {"name": name, "shape_changed": False, "old_slice_copied": True}

    if name in {
        "gat.layers.0.node_projection.weight",
        "gat.layers.0.residual.weight",
    }:
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(f"unsupported GAT shape migration for {name}: {tuple(source.shape)} -> {tuple(target.shape)}")
        if source.shape[1] != old_node_dim or target.shape[1] != new_node_dim:
            raise ValueError(f"unexpected GAT node input shape for {name}")
        target.zero_()
        target[:, :old_node_dim] = source
        return target, {
            "name": name,
            "shape_changed": True,
            "old_columns_copied": [0, int(old_node_dim)],
            "new_columns_zero_initialized": [int(old_node_dim), int(new_node_dim)],
        }

    if name == "critic.0.weight":
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(f"unsupported critic shape migration: {tuple(source.shape)} -> {tuple(target.shape)}")
        if source.shape[1] != old_central_dim or target.shape[1] != new_central_dim:
            raise ValueError("unexpected critic central input shape")
        target.zero_()
        for index in range(uav_count):
            source_start = index * old_node_dim
            target_start = index * new_node_dim
            target[:, target_start : target_start + old_node_dim] = source[:, source_start : source_start + old_node_dim]
        target[:, uav_count * new_node_dim :] = source[:, uav_count * old_node_dim :]
        return target, {
            "name": name,
            "shape_changed": True,
            "old_node_blocks_copied": int(uav_count),
            "new_node_slots_zero_initialized": int(uav_count * (new_node_dim - old_node_dim)),
            "old_tail_copied": True,
        }

    if name == "mlp_flat_encoder.0.weight":
        old_node_blocks = 2 * uav_count * old_node_dim
        new_node_blocks = 2 * uav_count * new_node_dim
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(f"unsupported MLP flat shape migration for {name}")
        target.zero_()
        for index in range(2 * uav_count):
            old_start = index * old_node_dim
            new_start = index * new_node_dim
            target[:, new_start : new_start + old_node_dim] = source[:, old_start : old_start + old_node_dim]
        target[:, new_node_blocks:] = source[:, old_node_blocks:]
        return target, {
            "name": name,
            "shape_changed": True,
            "old_node_blocks_copied": int(2 * uav_count),
            "new_node_slots_zero_initialized": int(2 * uav_count * (new_node_dim - old_node_dim)),
            "old_non_node_tail_copied": True,
        }

    if name == "measurement_source_encoder.0.weight":
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            raise ValueError(f"unsupported measurement source encoder migration for {name}")
        if source.shape[1] != old_node_dim + 1 or target.shape[1] != new_node_dim + 1:
            raise ValueError(f"unexpected measurement source encoder shape for {name}")
        target.zero_()
        target[:, :old_node_dim] = source[:, :old_node_dim]
        target[:, new_node_dim] = source[:, old_node_dim]
        return target, {
            "name": name,
            "shape_changed": True,
            "old_node_columns_copied": [0, int(old_node_dim)],
            "presence_column_copied_to": int(new_node_dim),
            "new_node_columns_zero_initialized": [int(old_node_dim), int(new_node_dim)],
        }

    raise ValueError(
        f"unexpected shape-changing parameter has no explicit migration rule: "
        f"{name}: {tuple(source.shape)} -> {tuple(target.shape)}"
    )


def migrate_policy_state(
    source_state: Mapping[str, Any],
    target_template: Mapping[str, Any],
    *,
    old_node_dim: int = OLD_NODE_DIM,
    new_node_dim: int = NEW_NODE_DIM,
    uav_count: int = 1,
    old_central_dim: int = 171,
    new_central_dim: int = 174,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate a policy state without reinitializing unrelated modules."""

    if set(source_state) != set(target_template):
        missing = sorted(set(target_template) - set(source_state))
        extra = sorted(set(source_state) - set(target_template))
        raise ValueError(f"policy state key mismatch during migration: missing={missing}, extra={extra}")
    result: dict[str, Any] = {}
    parameter_reports: list[dict[str, Any]] = []
    for name, target_value in target_template.items():
        source_value = source_state[name]
        if not hasattr(source_value, "shape") or not hasattr(target_value, "shape"):
            if source_value != target_value:
                raise ValueError(f"non-tensor policy state changed unexpectedly: {name}")
            result[name] = copy.deepcopy(source_value)
            continue
        migrated, report = _migrate_parameter_tensor(
            str(name),
            torch.as_tensor(source_value),
            torch.as_tensor(target_value),
            old_node_dim=old_node_dim,
            new_node_dim=new_node_dim,
            uav_count=uav_count,
            old_central_dim=old_central_dim,
            new_central_dim=new_central_dim,
        )
        result[name] = migrated
        parameter_reports.append(report)
    changed = [item for item in parameter_reports if item.get("shape_changed")]
    return result, {
        "old_node_dim": int(old_node_dim),
        "new_node_dim": int(new_node_dim),
        "old_central_dim": int(old_central_dim),
        "new_central_dim": int(new_central_dim),
        "parameter_count": int(len(parameter_reports)),
        "shape_changed_parameters": changed,
        "all_old_parameter_slices_explicitly_handled": True,
        "new_input_weights_zero_initialized": True,
    }


def _optimizer_parameter_maps(
    trainer: MAPPOTrainer,
    state: Mapping[str, Any],
) -> dict[str, int]:
    """Map serialized optimizer parameter IDs back to stable parameter names."""

    groups = state.get("param_groups")
    if not isinstance(groups, Sequence) or len(groups) != len(trainer.optimizer.param_groups):
        raise ValueError("optimizer parameter-group structure does not match trainer")
    object_names = {id(parameter): str(name) for name, parameter in trainer.policy.named_parameters()}
    result: dict[str, int] = {}
    for runtime_group, saved_group in zip(trainer.optimizer.param_groups, groups):
        saved_ids = list(saved_group.get("params", []))
        runtime_params = list(runtime_group.get("params", []))
        if len(saved_ids) != len(runtime_params):
            raise ValueError("optimizer parameter-group length mismatch")
        for parameter, parameter_id in zip(runtime_params, saved_ids):
            name = object_names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer parameter is not present in policy named parameters")
            result[name] = int(parameter_id)
    return result


def migrate_optimizer_state(
    source_state: Mapping[str, Any],
    source_trainer: MAPPOTrainer,
    target_trainer: MAPPOTrainer,
    *,
    old_node_dim: int,
    new_node_dim: int,
    uav_count: int,
    old_central_dim: int,
    new_central_dim: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carry Adam moments across the same explicit parameter migration."""

    source_map = _optimizer_parameter_maps(source_trainer, source_state)
    target_state = target_trainer.optimizer.state_dict()
    target_map = _optimizer_parameter_maps(target_trainer, target_state)
    source_named = dict(source_trainer.policy.named_parameters())
    target_named = dict(target_trainer.policy.named_parameters())
    source_states = source_state.get("state", {})
    if not isinstance(source_states, Mapping):
        raise ValueError("checkpoint optimizer state is missing state mapping")
    migrated_states: dict[int, dict[str, Any]] = {}
    changed_entries = 0
    for name, source_id in source_map.items():
        target_id = target_map[name]
        source_parameter = source_named[name]
        target_parameter = target_named[name]
        state_item = source_states.get(source_id, {})
        if not isinstance(state_item, Mapping):
            raise ValueError(f"invalid optimizer state for {name}")
        migrated_item: dict[str, Any] = {}
        for key, value in state_item.items():
            if hasattr(value, "shape") and tuple(value.shape) == tuple(source_parameter.shape):
                migrated_value, _ = _migrate_parameter_tensor(
                    name,
                    torch.as_tensor(value),
                    target_parameter.detach().cpu(),
                    old_node_dim=old_node_dim,
                    new_node_dim=new_node_dim,
                    uav_count=uav_count,
                    old_central_dim=old_central_dim,
                    new_central_dim=new_central_dim,
                )
                migrated_item[key] = migrated_value
                changed_entries += int(tuple(source_parameter.shape) != tuple(target_parameter.shape))
            elif hasattr(value, "shape"):
                migrated_item[key] = value.detach().cpu().clone()
            else:
                migrated_item[key] = copy.deepcopy(value)
        migrated_states[int(target_id)] = migrated_item
    migrated_groups = copy.deepcopy(list(source_state.get("param_groups", [])))
    for group in migrated_groups:
        group["params"] = [int(target_map[next(name for name, value in source_map.items() if value == parameter_id)]) for parameter_id in group.get("params", [])]
    result = {"state": migrated_states, "param_groups": migrated_groups}
    return result, {
        "source_state_entries": int(len(source_states)),
        "target_state_entries": int(len(migrated_states)),
        "shape_changed_moment_entries_migrated": int(changed_entries),
        "parameter_name_mapping_complete": set(source_map) == set(target_map),
    }


def _trainer_from_payload(
    payload: Mapping[str, Any],
    config: FormalExperimentConfig,
    *,
    normalizer_state: Mapping[str, Any],
    central_state: Mapping[str, Any],
    node_feature_dim: int,
) -> MAPPOTrainer:
    trainer_state = payload.get("trainer_config")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("checkpoint is missing trainer_config")
    normalizer = ObservationNormalizer.from_state_dict(normalizer_state)
    central_normalizer = FeatureNormalizer.from_state_dict(central_state)
    model_env = _make_environment(
        config,
        _scale_seed(config.seed, config.uav_count, "train"),
        metric_scales=config.metric_scales,
        visual_odometry_quality_model=config.visual_odometry_quality_model,
    )
    trainer_fields = {field.name for field in fields(MAPPOConfig)}
    kwargs = {
        name: copy.deepcopy(trainer_state[name])
        for name in trainer_fields
        if name in trainer_state and name != "measurement_persistent_proxy_model"
    }
    kwargs["node_feature_dim"] = int(node_feature_dim)
    trainer = MAPPOTrainer.from_environment(
        model_env,
        seed=_scale_seed(config.seed, config.uav_count, "train"),
        normalizer=normalizer,
        central_normalizer=central_normalizer,
        measurement_quality_model=(
            normalizer_state.get("measurement_quality_model")
            if isinstance(normalizer_state.get("measurement_quality_model"), Mapping)
            else None
        ),
        measurement_proxy_model=(
            payload.get("measurement_proxy_model")
            if isinstance(payload.get("measurement_proxy_model"), Mapping)
            else None
        ),
        measurement_persistent_proxy_model=(
            payload.get("measurement_persistent_proxy_model")
            if isinstance(payload.get("measurement_persistent_proxy_model"), Mapping)
            else None
        ),
        **kwargs,
    )
    trainer.load_measurement_rate_dual_state_dict(payload.get("measurement_rate_dual_state"))
    trainer.load_measurement_communication_dual_state_dict(payload.get("measurement_communication_dual_state"))
    trainer.policy.eval()
    return trainer


def _migrate_checkpoint(
    source_payload: Mapping[str, Any],
    source_config: FormalExperimentConfig,
    source_trainer: MAPPOTrainer,
    target_config: FormalExperimentConfig,
    target_trainer: MAPPOTrainer,
    migrated_normalizer_state: Mapping[str, Any],
    migrated_central_state: Mapping[str, Any],
    *,
    source_checkpoint: Path,
    target_checkpoint: Path,
    source_contract: Mapping[str, Any],
    normalizer_report: Mapping[str, Any],
    central_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_policy = source_payload.get("policy_state_dict")
    if not isinstance(source_policy, Mapping):
        raise ValueError("source checkpoint is missing policy_state_dict")
    old_central_dim = int(source_trainer.config.critic_input_dim)
    new_central_dim = int(target_trainer.config.critic_input_dim)
    migrated_policy, policy_report = migrate_policy_state(
        source_policy,
        target_trainer.policy.state_dict(),
        old_node_dim=OLD_NODE_DIM,
        new_node_dim=NEW_NODE_DIM,
        uav_count=1,
        old_central_dim=old_central_dim,
        new_central_dim=new_central_dim,
    )
    source_optimizer = source_payload.get("optimizer_state_dict")
    if not isinstance(source_optimizer, Mapping):
        raise ValueError("source checkpoint is missing optimizer_state_dict")
    migrated_optimizer, optimizer_report = migrate_optimizer_state(
        source_optimizer,
        source_trainer,
        target_trainer,
        old_node_dim=OLD_NODE_DIM,
        new_node_dim=NEW_NODE_DIM,
        uav_count=1,
        old_central_dim=old_central_dim,
        new_central_dim=new_central_dim,
    )
    migrated_payload = copy.deepcopy(dict(source_payload))
    migrated_payload["experiment_config"] = _checkpoint_config_payload(target_config)
    migrated_payload["trainer_config"] = _jsonable(asdict(target_trainer.config))
    migrated_payload["policy_state_dict"] = migrated_policy
    migrated_payload["optimizer_state_dict"] = migrated_optimizer
    migrated_payload["normalization_stats"] = copy.deepcopy(dict(migrated_normalizer_state))
    migrated_payload["central_normalization_stats"] = copy.deepcopy(dict(migrated_central_state))
    migrated_payload["stage2a_migration"] = {
        "stage": "single_uav_obstacle_direction_observation_stage2a",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": str(source_contract["sha256"]),
        "source_policy_hash": str(source_contract["checkpoint_policy_hash"]),
        "old_node_dim": OLD_NODE_DIM,
        "new_node_dim": NEW_NODE_DIM,
        "new_direction_indices": list(NEW_DIRECTION_INDICES),
        "new_direction_features": list(NEW_DIRECTION_FEATURES),
        "old_central_dim": old_central_dim,
        "new_central_dim": new_central_dim,
        "new_input_initialization": "zero",
        "normalizer_new_entries": "mean=0, scale=1, count=0",
        "trainer_update_calls": 0,
        "policy_updates": 0,
        "source_checkpoint_not_modified": True,
        "policy_report": policy_report,
        "optimizer_report": optimizer_report,
        "normalizer_report": normalizer_report,
        "central_normalizer_report": central_report,
    }
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated_payload, target_checkpoint)
    report = {
        "status": "completed",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": str(source_contract["sha256"]),
        "source_policy_hash": str(source_contract["checkpoint_policy_hash"]),
        "migrated_checkpoint": str(target_checkpoint.resolve()),
        "migrated_checkpoint_sha256": _sha256(target_checkpoint),
        "old_node_dim": OLD_NODE_DIM,
        "new_node_dim": NEW_NODE_DIM,
        "old_central_dim": old_central_dim,
        "new_central_dim": new_central_dim,
        "new_direction_indices": list(NEW_DIRECTION_INDICES),
        "new_direction_features": list(NEW_DIRECTION_FEATURES),
        "policy": policy_report,
        "optimizer": optimizer_report,
        "source_checkpoint_hash_after_save": _sha256(source_checkpoint),
        "source_checkpoint_unchanged": _sha256(source_checkpoint) == str(source_contract["sha256"]),
        "trainer_update_calls": 0,
        "policy_updates": 0,
    }
    return migrated_payload, report


def _observation_action_probe(
    trainer: MAPPOTrainer,
    config: FormalExperimentConfig,
    scenario: Mapping[str, Any],
    *,
    role: str,
    case: str,
    seed: int,
) -> dict[str, Any]:
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        visual_odometry_quality_model=config.visual_odometry_quality_model,
    )
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    raw_local = observations["uav_00"]
    decision = trainer.act(observations, info["central_state"], deterministic=True)
    normalized_local = decision.observations["uav_00"]
    context, _action_mask, _bandwidth_mask, _effective, _potential, _embedding, _node_ids = trainer.policy._local_context(normalized_local)
    raw_node = np.asarray(raw_local["node_features"], dtype=np.float64).reshape(-1)
    normalized_node = np.asarray(normalized_local["node_features"], dtype=np.float64).reshape(-1)
    action = np.asarray(decision.actions["uav_00"]["delta_v"], dtype=np.float64).reshape(3)
    position = np.asarray(environment.ekfs["uav_00"].state[:3], dtype=float)
    physical = [obstacle for obstacle in environment.obstacles if getattr(obstacle, "physical_collision_enabled", True)]
    nearest = environment._nearest_physical_obstacle(position, physical)
    nearest_id = str(nearest.obstacle_id) if nearest is not None else ""
    direction = raw_node[list(NEW_DIRECTION_INDICES)] if raw_node.size >= NEW_NODE_DIM else np.empty(0)
    normalized_direction = normalized_node[list(NEW_DIRECTION_INDICES)] if normalized_node.size >= NEW_NODE_DIM else np.empty(0)
    return {
        "role": str(role),
        "case": str(case),
        "seed": int(seed),
        "scenario": _jsonable(scenario),
        "raw_node_features": raw_node.tolist(),
        "normalized_node_features": normalized_node.tolist(),
        "old_prefix": raw_node[:OLD_NODE_DIM].tolist(),
        "normalized_old_prefix": normalized_node[:OLD_NODE_DIM].tolist(),
        "new_direction_features": direction.tolist(),
        "new_normalized_direction_features": normalized_direction.tolist(),
        "new_direction_signs": [int(np.sign(value)) for value in direction],
        "actor_context": context.detach().cpu().numpy().astype(float).tolist(),
        "actor_action": action.tolist(),
        "actor_node_shape": list(np.asarray(raw_local["node_features"]).shape),
        "actor_context_dim": int(context.numel()),
        "nearest_physical_obstacle_id": nearest_id,
        "minimum_obstacle_clearance_m": float(raw_node[33]) if raw_node.size > 33 else 0.0,
        "raw_node_finite": bool(np.isfinite(raw_node).all()),
        "normalized_node_finite": bool(np.isfinite(normalized_node).all()),
        "hashes": {
            "raw_node": _array_hash(raw_node),
            "normalized_node": _array_hash(normalized_node),
            "context": _array_hash(context.detach().cpu().numpy()),
            "action": _array_hash(action),
        },
    }


def _compatibility_rows(
    source_trainer: MAPPOTrainer,
    target_trainer: MAPPOTrainer,
    source_config: FormalExperimentConfig,
    target_config: FormalExperimentConfig,
    scenarios: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, scenario in scenarios.items():
        source = _observation_action_probe(
            source_trainer, source_config, scenario, role="stage1_source_flag_false", case=case, seed=PROBE_SEED
        )
        target = _observation_action_probe(
            target_trainer, target_config, scenario, role="stage2a_migrated_flag_true", case=case, seed=PROBE_SEED
        )
        rows.append(
            {
                "case": case,
                "source_node_dim": source["actor_node_shape"][-1],
                "target_node_dim": target["actor_node_shape"][-1],
                "raw_direction_features": json.dumps(target["new_direction_features"]),
                "normalized_direction_features": json.dumps(target["new_normalized_direction_features"]),
                "action_l1": _difference(source["actor_action"], target["actor_action"])["l1"],
                "action_l2": _difference(source["actor_action"], target["actor_action"])["l2"],
                "action_max_abs": _difference(source["actor_action"], target["actor_action"])["max_abs"],
                "context_max_abs": _difference(source["actor_context"], target["actor_context"])["max_abs"],
                "old_prefix_max_abs": _difference(source["old_prefix"], target["old_prefix"])["max_abs"],
                "compatible_within_tolerance": bool(
                    _difference(source["actor_action"], target["actor_action"])["max_abs"] <= TOLERANCE
                    and _difference(source["actor_context"], target["actor_context"])["max_abs"] <= TOLERANCE
                    and _difference(source["old_prefix"], target["old_prefix"])["max_abs"] <= TOLERANCE
                ),
            }
        )
    return rows


def _direction_difference_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_case = {str(row["case"]): row for row in records}
    pairs = (
        ("left_front", "right_front", "left_front_vs_right_front"),
        ("left", "right", "left_vs_right"),
        ("front", "behind", "front_vs_behind"),
    )
    rows: list[dict[str, Any]] = []
    for case_a, case_b, pair in pairs:
        first = by_case[case_a]
        second = by_case[case_b]
        raw = _difference(first["raw_node_features"], second["raw_node_features"])
        normalized = _difference(first["normalized_node_features"], second["normalized_node_features"])
        new_raw = _difference(first["new_direction_features"], second["new_direction_features"])
        new_normalized = _difference(first["new_normalized_direction_features"], second["new_normalized_direction_features"])
        old_prefix = _difference(first["old_prefix"], second["old_prefix"])
        context = _difference(first["actor_context"], second["actor_context"])
        action = _difference(first["actor_action"], second["actor_action"])
        changed = [
            int(index)
            for index, (a, b) in enumerate(zip(first["raw_node_features"], second["raw_node_features"]))
            if abs(float(a) - float(b)) > TOLERANCE
        ]
        rows.append(
            {
                "pair": pair,
                "case_a": case_a,
                "case_b": case_b,
                "raw_node_l1": raw["l1"],
                "raw_node_l2": raw["l2"],
                "raw_node_max_abs": raw["max_abs"],
                "normalized_node_l1": normalized["l1"],
                "normalized_node_l2": normalized["l2"],
                "normalized_node_max_abs": normalized["max_abs"],
                "new_direction_raw_l1": new_raw["l1"],
                "new_direction_raw_l2": new_raw["l2"],
                "new_direction_raw_max_abs": new_raw["max_abs"],
                "new_direction_normalized_l1": new_normalized["l1"],
                "new_direction_normalized_l2": new_normalized["l2"],
                "new_direction_normalized_max_abs": new_normalized["max_abs"],
                "old_prefix_max_abs": old_prefix["max_abs"],
                "actor_context_l1": context["l1"],
                "actor_context_l2": context["l2"],
                "actor_context_max_abs": context["max_abs"],
                "actor_action_l1": action["l1"],
                "actor_action_l2": action["l2"],
                "actor_action_max_abs": action["max_abs"],
                "changed_feature_indices": json.dumps(changed),
                "direction_only_old_prefix_unchanged": bool(old_prefix["max_abs"] <= TOLERANCE),
            }
        )
    return rows


def _feature_layout_rows(node_dim: int = NEW_NODE_DIM) -> list[dict[str, Any]]:
    old_names = [
        "estimated_position_x", "estimated_position_y", "estimated_position_z",
        "estimated_velocity_x", "estimated_velocity_y", "estimated_velocity_z",
        "estimated_attitude_roll", "estimated_attitude_pitch", "estimated_attitude_yaw",
        "estimated_accel_bias_x", "estimated_accel_bias_y", "estimated_accel_bias_z",
        "estimated_gyro_bias_x", "estimated_gyro_bias_y", "estimated_gyro_bias_z",
        "covariance_position_x", "covariance_position_y", "covariance_position_z",
        "covariance_velocity_x", "covariance_velocity_y", "covariance_velocity_z",
        "covariance_attitude_roll", "covariance_attitude_pitch", "covariance_attitude_yaw",
        "covariance_accel_bias_x", "covariance_accel_bias_y", "covariance_accel_bias_z",
        "covariance_gyro_bias_x", "covariance_gyro_bias_y", "covariance_gyro_bias_z",
        "goal_relative_dx", "goal_relative_dy", "goal_relative_dz",
        "minimum_obstacle_clearance", "obstacle_box_count", "obstacle_sphere_count",
        "environment_timestep", "braking_target_speed_normalized", "braking_margin_normalized",
    ]
    frames = [
        *(["EKF_estimated_world_state"] * 15),
        *(["EKF_covariance_diagonal"] * 15),
        *(["world_relative_goal"] * 3),
        *(["scalar_obstacle_aggregate"] * 4),
        *(["scalar_braking_descriptor"] * 2),
    ]
    sources = [
        *(["LocalEKF.state"] * 15),
        *(["LocalEKF.covariance diagonal"] * 15),
        *(["goal - EKF estimated position"] * 3),
        *(["environment._local_obstacle_features"] * 4),
        *(["environment braking descriptor"] * 2),
    ]
    rows = []
    for index, (name, frame, source) in enumerate(zip(old_names, frames, sources)):
        rows.append({
            "layer": "actor_local_node_features",
            "feature": name,
            "index": index,
            "dimension": 1,
            "coordinate_frame": frame,
            "source": source,
            "present_in_observation": True,
            "enters_actor": index < node_dim,
            "notes": "historical prefix preserved exactly",
        })
    for index, name in zip(NEW_DIRECTION_INDICES, NEW_DIRECTION_FEATURES):
        rows.append({
            "layer": "actor_local_node_features",
            "feature": name,
            "index": index,
            "dimension": 1,
            "coordinate_frame": "UAV_yaw_body_frame_ENU",
            "source": "EKF estimated position/yaw + nearest physical obstacle center",
            "present_in_observation": True,
            "enters_actor": index < node_dim,
            "notes": "appended opt-in Stage 2A feature; neutral migration weight",
        })
    return rows


def _write_probe_plot(records: Sequence[Mapping[str, Any]], path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [name for name, _offset in PROBE_CASES]
    ordered = sorted(records, key=lambda row: order.index(str(row["case"])))
    matrix = np.asarray([row["new_normalized_direction_features"] for row in ordered], dtype=float)
    actions = np.asarray([row["actor_action"] for row in ordered], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    image = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[0].set_yticks(range(len(ordered)), [str(row["case"]) for row in ordered])
    axes[0].set_xticks(range(3), ["dx_body", "dy_body", "dz_body"])
    axes[0].set_title("Stage 2A normalized direction input")
    axes[0].set_xlabel("new feature")
    figure.colorbar(image, ax=axes[0], shrink=0.8)
    axes[1].bar(np.arange(len(ordered)) - 0.25, actions[:, 0], width=0.25, label="Δv_x")
    axes[1].bar(np.arange(len(ordered)), actions[:, 1], width=0.25, label="Δv_y")
    axes[1].bar(np.arange(len(ordered)) + 0.25, actions[:, 2], width=0.25, label="Δv_z")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(range(len(ordered)), [str(row["case"]) for row in ordered], rotation=30, ha="right")
    axes[1].set_title("Migrated actor action (zero new weights expected)")
    axes[1].set_ylabel("Δv (m/s)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.suptitle("Stage 2A observation-only direction probe")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return str(path.resolve())


def _build_probe_scenarios(config: FormalExperimentConfig) -> dict[str, dict[str, Any]]:
    scenarios = {name: copy.deepcopy(value) for name, value in probe_scenarios().items()}
    scenarios["normal_far_distance"] = _normal_scenario(
        config, seed=PROBE_SEED, split="test", episode_index=0
    )
    scenarios["static_front_obstacle"] = _static_pressure_scenario(
        config, seed=PROBE_SEED, split="test", episode_index=0
    )
    return scenarios


def _sanity_rows(
    trainer: MAPPOTrainer,
    config: FormalExperimentConfig,
) -> list[dict[str, Any]]:
    scenarios = {
        "normal_far_distance": _normal_scenario(config, seed=PROBE_SEED, split="test", episode_index=0),
        "static_front_obstacle": _static_pressure_scenario(config, seed=PROBE_SEED, split="test", episode_index=0),
    }
    rows = []
    for name, scenario in scenarios.items():
        row = _observation_action_probe(
            trainer, config, scenario, role="stage2a_sanity", case=name, seed=PROBE_SEED
        )
        rows.append({
            "case": name,
            "node_dim": row["actor_node_shape"][-1],
            "context_dim": row["actor_context_dim"],
            "raw_node_finite": row["raw_node_finite"],
            "normalized_node_finite": row["normalized_node_finite"],
            "action_finite": bool(np.isfinite(np.asarray(row["actor_action"], dtype=float)).all()),
            "normal_or_static_reset_act_ok": True,
        })
    return rows


def _flag_false_checkpoint_probe(
    trainer: MAPPOTrainer,
    config: FormalExperimentConfig,
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
) -> list[dict[str, Any]]:
    """Probe an old checkpoint without enabling the Stage-2A contract."""

    rows: list[dict[str, Any]] = []
    for case, scenario in scenarios.items():
        record = _observation_action_probe(
            trainer,
            config,
            scenario,
            role=role,
            case=case,
            seed=PROBE_SEED,
        )
        rows.append(
            {
                "role": str(role),
                "case": str(case),
                "node_dim": int(record["actor_node_shape"][-1]),
                "context_dim": int(record["actor_context_dim"]),
                "raw_node_finite": bool(record["raw_node_finite"]),
                "normalized_node_finite": bool(record["normalized_node_finite"]),
                "action_finite": bool(np.isfinite(np.asarray(record["actor_action"], dtype=float)).all()),
                "action": json.dumps(record["actor_action"]),
                "flag_false_compatibility_ok": bool(
                    record["actor_node_shape"][-1] == OLD_NODE_DIM
                    and record["raw_node_finite"]
                    and record["normalized_node_finite"]
                ),
            }
        )
    return rows


def run_stage2a(
    *,
    stage1_checkpoint: str | Path = DEFAULT_STAGE1_CHECKPOINT,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    progress: bool = True,
) -> dict[str, Any]:
    """Migrate, probe, and report Stage 2A without any policy update."""

    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    _ensure_empty_roots(data_root, logs_root, plots_root)
    stage1_checkpoint = Path(stage1_checkpoint).resolve()
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    for checkpoint in (stage1_checkpoint, baseline_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage 2A checkpoint does not exist: {checkpoint}")

    stage1_payload, stage1_sha_before = _read_checkpoint(stage1_checkpoint)
    baseline_payload, baseline_sha_before = _read_checkpoint(baseline_checkpoint)
    source_config = _config_from_checkpoint(stage1_payload, output_dir=logs_root, progress=False)
    source_config = replace(
        source_config,
        output_dir=logs_root,
        force=True,
        progress=False,
        command_line="single-UAV obstacle direction observation Stage 2A; observation-only",
        scenario_path=None,
        scenario_pool_root=None,
        scenario_profile_path=None,
        single_uav_obstacle_direction_features_enabled=False,
    )
    source_contract = _source_contract_report(stage1_payload, source_config, stage1_checkpoint)
    source_trainer, source_identity, _ = _load_frozen_trainer(
        stage1_payload, source_config, metric_scales=source_config.metric_scales
    )
    source_trainer.policy.eval()
    if _state_dict_hash(source_trainer.policy.state_dict()) != source_contract["checkpoint_policy_hash"]:
        raise RuntimeError("Stage 1 source policy did not load exactly")

    baseline_config = _config_from_checkpoint(
        baseline_payload, output_dir=logs_root, progress=False
    )
    baseline_config = replace(
        baseline_config,
        output_dir=logs_root,
        force=True,
        progress=False,
        command_line="single-UAV obstacle direction observation Stage 2A; baseline compatibility probe",
        scenario_path=None,
        scenario_pool_root=None,
        scenario_profile_path=None,
        single_uav_obstacle_direction_features_enabled=False,
    )
    baseline_trainer, baseline_identity, _ = _load_frozen_trainer(
        baseline_payload, baseline_config, metric_scales=baseline_config.metric_scales
    )
    baseline_trainer.policy.eval()
    baseline_policy_hash = _state_dict_hash(baseline_trainer.policy.state_dict())
    if baseline_config.node_feature_dim != OLD_NODE_DIM or baseline_trainer.config.node_feature_dim != OLD_NODE_DIM:
        raise RuntimeError("baseline checkpoint did not retain the flag=false 39-dimensional contract")
    if baseline_policy_hash != _state_dict_hash(baseline_payload.get("policy_state_dict", {})):
        raise RuntimeError("baseline checkpoint policy did not load exactly")

    target_config = replace(
        source_config,
        output_dir=data_root,
        force=True,
        single_uav_obstacle_direction_features_enabled=True,
    )
    migrated_normalizer_state, normalizer_report = migrate_observation_normalizer_state(
        stage1_payload["normalization_stats"]
    )
    trainer_state = stage1_payload.get("trainer_config", {})
    edge_dim = int(trainer_state.get("edge_feature_dim", 11))
    max_obstacles = int(trainer_state.get("max_obstacles", source_config.max_obstacles))
    include_remaining_time = bool(trainer_state.get("include_remaining_time", source_config.include_remaining_time))
    migrated_central_state, central_report = migrate_central_normalizer_state(
        stage1_payload["central_normalization_stats"],
        uav_count=1,
        old_node_dim=OLD_NODE_DIM,
        new_node_dim=NEW_NODE_DIM,
        edge_feature_dim=edge_dim,
        max_obstacles=max_obstacles,
        include_remaining_time=include_remaining_time,
    )
    target_trainer = _trainer_from_payload(
        stage1_payload,
        target_config,
        normalizer_state=migrated_normalizer_state,
        central_state=migrated_central_state,
        node_feature_dim=NEW_NODE_DIM,
    )
    migrated_checkpoint = data_root / "checkpoints" / "stage2a_observation_compatible_init.pt"
    migrated_payload, migration_report = _migrate_checkpoint(
        stage1_payload,
        source_config,
        source_trainer,
        target_config,
        target_trainer,
        migrated_normalizer_state,
        migrated_central_state,
        source_checkpoint=stage1_checkpoint,
        target_checkpoint=migrated_checkpoint,
        source_contract=source_contract,
        normalizer_report=normalizer_report,
        central_report=central_report,
    )
    # Re-load the serialized migrated checkpoint once, so all downstream
    # probes exercise the actual saved artifact rather than only an in-memory
    # state dict.
    migrated_payload_loaded, migrated_sha = _read_checkpoint(migrated_checkpoint)
    target_trainer = _trainer_from_payload(
        migrated_payload_loaded,
        target_config,
        normalizer_state=migrated_payload_loaded["normalization_stats"],
        central_state=migrated_payload_loaded["central_normalization_stats"],
        node_feature_dim=NEW_NODE_DIM,
    )
    target_trainer.policy.load_state_dict(migrated_payload_loaded["policy_state_dict"], strict=True)
    target_trainer.policy.eval()

    scenarios = _build_probe_scenarios(source_config)
    probe_scenarios_only = {name: scenarios[name] for name, _offset in PROBE_CASES}
    stage2_records = [
        _observation_action_probe(
            target_trainer,
            target_config,
            scenario,
            role="stage2a_migrated",
            case=case,
            seed=PROBE_SEED,
        )
        for case, scenario in probe_scenarios_only.items()
    ]
    pair_rows = _direction_difference_rows(stage2_records)
    compatibility_scenarios = {
        "normal_far_distance": scenarios["normal_far_distance"],
        "static_front_obstacle": scenarios["static_front_obstacle"],
        **probe_scenarios_only,
    }
    compatibility_rows = _compatibility_rows(
        source_trainer,
        target_trainer,
        source_config,
        target_config,
        compatibility_scenarios,
    )
    baseline_probe_rows = _flag_false_checkpoint_probe(
        baseline_trainer,
        baseline_config,
        {
            "normal_far_distance": scenarios["normal_far_distance"],
            "static_front_obstacle": scenarios["static_front_obstacle"],
        },
        role="baseline_flag_false",
    )
    sanity_rows = _sanity_rows(target_trainer, target_config)
    plot_path = _write_probe_plot(stage2_records, plots_root / "stage2a_direction_probe.png")

    direction_signs = {str(row["case"]): row["new_direction_signs"] for row in stage2_records}
    left_right_sign_ok = bool(
        direction_signs["left"][1] > 0
        and direction_signs["right"][1] < 0
        and direction_signs["left_front"][1] > 0
        and direction_signs["right_front"][1] < 0
    )
    front_behind_sign_ok = bool(
        direction_signs["front"][0] > 0 and direction_signs["behind"][0] < 0
    )
    raw_differences_ok = all(float(row["new_direction_raw_max_abs"]) > TOLERANCE for row in pair_rows)
    normalized_differences_ok = all(float(row["new_direction_normalized_max_abs"]) > TOLERANCE for row in pair_rows)
    old_prefix_ok = all(bool(row["direction_only_old_prefix_unchanged"]) for row in pair_rows)
    compatibility_ok = all(bool(row["compatible_within_tolerance"]) for row in compatibility_rows)
    source_sha_after = _sha256(stage1_checkpoint)
    baseline_sha_after = _sha256(baseline_checkpoint)
    target_updates = 0
    target_policy_hash_before = _state_dict_hash(target_trainer.policy.state_dict())
    target_policy_hash_after = _state_dict_hash(target_trainer.policy.state_dict())
    if target_updates != 0:
        raise RuntimeError("Stage 2A unexpectedly performed a policy update")

    feature_layout = _feature_layout_rows(NEW_NODE_DIM)
    _write_csv(
        logs_root / "stage2a_feature_layout.csv",
        ("layer", "feature", "index", "dimension", "coordinate_frame", "source", "present_in_observation", "enters_actor", "notes"),
        feature_layout,
    )
    _write_csv(
        logs_root / "stage2a_direction_probe.csv",
        (
            "role", "case", "actor_node_shape", "actor_context_dim", "nearest_physical_obstacle_id",
            "minimum_obstacle_clearance_m", "new_direction_features", "new_normalized_direction_features",
            "new_direction_signs", "actor_action", "raw_node_finite", "normalized_node_finite",
        ),
        [
            {
                "role": row["role"],
                "case": row["case"],
                "actor_node_shape": json.dumps(row["actor_node_shape"]),
                "actor_context_dim": row["actor_context_dim"],
                "nearest_physical_obstacle_id": row["nearest_physical_obstacle_id"],
                "minimum_obstacle_clearance_m": row["minimum_obstacle_clearance_m"],
                "new_direction_features": json.dumps(row["new_direction_features"]),
                "new_normalized_direction_features": json.dumps(row["new_normalized_direction_features"]),
                "new_direction_signs": json.dumps(row["new_direction_signs"]),
                "actor_action": json.dumps(row["actor_action"]),
                "raw_node_finite": row["raw_node_finite"],
                "normalized_node_finite": row["normalized_node_finite"],
            }
            for row in stage2_records
        ],
    )
    _write_csv(
        logs_root / "stage2a_input_difference_matrix.csv",
        tuple(pair_rows[0].keys()),
        pair_rows,
    )
    _write_csv(
        logs_root / "compatibility_action_comparison.csv",
        tuple(compatibility_rows[0].keys()),
        compatibility_rows,
    )
    _write_csv(
        logs_root / "baseline_compatibility_probe.csv",
        tuple(baseline_probe_rows[0].keys()),
        baseline_probe_rows,
    )
    _write_csv(
        logs_root / "stage2a_sanity_probe.csv",
        tuple(sanity_rows[0].keys()),
        sanity_rows,
    )

    _write_json(logs_root / "checkpoint_migration_report.json", _jsonable(migration_report))
    _write_json(
        logs_root / "normalizer_migration_report.json",
        _jsonable({"observation": normalizer_report, "central": central_report}),
    )
    _write_json(
        data_root / "stage2a_artifacts_manifest.json",
        _jsonable({
            "stage": "single_uav_obstacle_direction_observation_stage2a",
            "evaluation_only": True,
            "training_performed": False,
            "checkpoint": str(migrated_checkpoint.resolve()),
            "checkpoint_sha256": migrated_sha,
            "source_stage1_checkpoint_sha256": stage1_sha_before,
            "source_baseline_checkpoint_sha256": baseline_sha_before,
        }),
    )
    summary = {
        "status": "completed_observation_only",
        "stage": "single_uav_obstacle_direction_observation_stage2a",
        "stage2a_passed": bool(
            source_sha_after == stage1_sha_before
            and baseline_sha_after == baseline_sha_before
            and compatibility_ok
            and raw_differences_ok
            and normalized_differences_ok
            and left_right_sign_ok
            and front_behind_sign_ok
            and old_prefix_ok
            and target_policy_hash_before == target_policy_hash_after
            and all(bool(row["normal_or_static_reset_act_ok"]) for row in sanity_rows)
            and baseline_config.node_feature_dim == OLD_NODE_DIM
            and all(bool(row["flag_false_compatibility_ok"]) for row in baseline_probe_rows)
        ),
        "old_node_feature_dim": OLD_NODE_DIM,
        "new_node_feature_dim": NEW_NODE_DIM,
        "new_feature_indices": list(NEW_DIRECTION_INDICES),
        "new_feature_names": list(NEW_DIRECTION_FEATURES),
        "coordinate_convention": {
            "world": "ENU; +x east, +y north, +z up",
            "body": "+x forward, +y left, +z up",
            "transform": "yaw-only world-to-body rotation",
            "pose_source": "EKF estimated position and estimated yaw; not true UAV pose",
            "relative_vector": "nearest physical obstacle center minus EKF estimated UAV position",
            "nearest_selection": "minimum point-to-surface clearance, tie-break by obstacle_id ascending",
            "no_physical_obstacle": "[0, 0, 0] finite sentinel; existing box/sphere counts disambiguate absence",
        },
        "direction_probe": {
            "cases": [name for name, _offset in PROBE_CASES],
            "raw_differences_nonzero": raw_differences_ok,
            "normalized_differences_nonzero": normalized_differences_ok,
            "left_right_signs_correct": left_right_sign_ok,
            "front_behind_signs_correct": front_behind_sign_ok,
            "old_prefix_unchanged": old_prefix_ok,
            "pairs": pair_rows,
        },
        "compatibility": {
            "flag_false_historical_node_dim": OLD_NODE_DIM,
            "flag_true_migrated_node_dim": NEW_NODE_DIM,
            "rows": compatibility_rows,
            "all_within_tolerance": compatibility_ok,
            "tolerance": TOLERANCE,
            "source_policy_identity": source_identity,
            "baseline_flag_false": {
                "node_dim": int(baseline_trainer.config.node_feature_dim),
                "policy_hash": baseline_policy_hash,
                "checkpoint_policy_hash": _state_dict_hash(baseline_payload.get("policy_state_dict", {})),
                "policy_loaded_exactly": baseline_policy_hash == _state_dict_hash(baseline_payload.get("policy_state_dict", {})),
                "identity": baseline_identity,
                "probe_rows": baseline_probe_rows,
            },
        },
        "migration": migration_report,
        "normalizer": {
            "observation": normalizer_report,
            "central": central_report,
        },
        "no_training": {
            "trainer_update_calls": 0,
            "policy_updates": 0,
            "ppo_optimizer_steps": 0,
            "target_policy_hash_before_probe": target_policy_hash_before,
            "target_policy_hash_after_probe": target_policy_hash_after,
        },
        "source_integrity": {
            "stage1_sha_before": stage1_sha_before,
            "stage1_sha_after": source_sha_after,
            "stage1_unchanged": source_sha_after == stage1_sha_before,
            "baseline_sha_before": baseline_sha_before,
            "baseline_sha_after": baseline_sha_after,
            "baseline_unchanged": baseline_sha_after == baseline_sha_before,
        },
        "sanity": sanity_rows,
        "outputs": {
            "data_root": str(data_root.resolve()),
            "logs_root": str(logs_root.resolve()),
            "plots_root": str(plots_root.resolve()),
            "plot": plot_path,
            "migrated_checkpoint": str(migrated_checkpoint.resolve()),
        },
        "decision": {
            "stage2a_observation_blocker_fixed": bool(
                raw_differences_ok and normalized_differences_ok and left_right_sign_ok and front_behind_sign_ok
            ),
            "can_design_stage2b_near_obstacle_oracle_dagger": bool(
                source_sha_after == stage1_sha_before
                and compatibility_ok
                and raw_differences_ok
                and normalized_differences_ok
                and left_right_sign_ok
                and front_behind_sign_ok
                and old_prefix_ok
            ),
            "does_not_claim_active_avoidance": True,
            "next_stage_not_executed": True,
        },
    }
    _write_json(logs_root / "stage2a_observation_summary.json", _jsonable(summary))
    if progress:
        print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_stage2a(
        stage1_checkpoint=args.stage1_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        progress=not args.quiet,
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "DEFAULT_BASELINE_CHECKPOINT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "DEFAULT_STAGE1_CHECKPOINT",
    "NEW_DIRECTION_FEATURES",
    "NEW_DIRECTION_INDICES",
    "NEW_NODE_DIM",
    "OLD_NODE_DIM",
    "migrate_central_normalizer_state",
    "migrate_observation_normalizer_state",
    "migrate_optimizer_state",
    "migrate_policy_state",
    "run_stage2a",
]
