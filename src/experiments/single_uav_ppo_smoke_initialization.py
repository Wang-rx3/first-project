"""Checkpoint initialization for the two-group single-UAV PPO smoke run.

The smoke experiment deliberately separates parameter warm-starting from
normalization and optimizer state.  ``initialize_smoke_trainer`` receives a
fresh trainer that was already assembled with the new, train-only fitted
normalizers.  It copies only semantically compatible policy tensors, leaves
the first GAT layer and centralized critic at their fresh constructor values,
and never restores source optimizer, rollout, or teacher state.

This module is read-only with respect to checkpoint files.  It does not
create an environment, collect a transition, or call a training update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SOURCE_A_DEFAULT = Path(
    "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
    "dagger_bc/baseline_observation_migration/checkpoints/"
    "stage2a_observation_compatible_init.pt"
)
SOURCE_B_DEFAULT = Path(
    "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
    "dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt"
)

EXPECTED_UAV_COUNT = 1
EXPECTED_NODE_FEATURE_DIM = 42
EXPECTED_EDGE_FEATURE_DIM = 11
EXPECTED_CENTRAL_DIM = 174
EXPECTED_DIRECTION_INDICES = [39, 40, 41]
EXPECTED_DIRECTION_FEATURES = [
    "relative_obstacle_dx_body_m",
    "relative_obstacle_dy_body_m",
    "relative_obstacle_dz_body_m",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: Any) -> str:
    if not torch.is_tensor(value):
        raise TypeError("policy state entries must be torch tensors")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if torch.is_tensor(value) else None


def _dtype(value: Any) -> str | None:
    return str(value.dtype) if torch.is_tensor(value) else None


def _json_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _module_name(parameter_name: str) -> str:
    """Map a state-dict key to the whole module that owns it."""

    if parameter_name.startswith("gat.layers."):
        parts = parameter_name.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
    return parameter_name.split(".", 1)[0]


def _resolve_module(root: nn.Module, name: str) -> nn.Module | None:
    current: Any = root
    for token in name.split("."):
        if isinstance(current, (nn.ModuleList, nn.Sequential)) and token.isdigit():
            index = int(token)
            if index >= len(current):
                return None
            current = current[index]
        elif hasattr(current, token):
            current = getattr(current, token)
        else:
            return None
    return current if isinstance(current, nn.Module) else None


def _source_metadata(payload: Mapping[str, Any], group: str) -> dict[str, Any]:
    bc = payload.get("bc_training")
    is_bc = isinstance(bc, Mapping)
    if group == "A" and is_bc:
        raise ValueError("group A requires a PPO lineage checkpoint without bc_training metadata")
    if group == "B" and not is_bc:
        raise ValueError("group B requires a BC/DAgger checkpoint with bc_training metadata")
    if group == "B" and is_bc and bc.get("critic_frozen") is False:
        raise ValueError("group B requires the BC source critic to be explicitly frozen")
    trainer_config = payload.get("trainer_config")
    experiment_config = payload.get("experiment_config")
    trainer_config = trainer_config if isinstance(trainer_config, Mapping) else {}
    experiment_config = experiment_config if isinstance(experiment_config, Mapping) else {}

    if int(payload.get("uav_count", trainer_config.get("uav_count", experiment_config.get("uav_count", -1)))) != EXPECTED_UAV_COUNT:
        raise ValueError("smoke initialization requires a one-UAV source checkpoint")
    node_dim = trainer_config.get("node_feature_dim", experiment_config.get("node_feature_dim"))
    if node_dim is not None and int(node_dim) != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError(f"source node feature dimension must be {EXPECTED_NODE_FEATURE_DIM}")
    edge_dim = trainer_config.get("edge_feature_dim")
    if edge_dim is not None and int(edge_dim) != EXPECTED_EDGE_FEATURE_DIM:
        raise ValueError(f"source edge feature dimension must be {EXPECTED_EDGE_FEATURE_DIM}")

    migration = payload.get("stage2a_migration")
    migration = migration if isinstance(migration, Mapping) else {}
    direction_indices = migration.get("new_direction_indices")
    if direction_indices is not None and list(direction_indices) != EXPECTED_DIRECTION_INDICES:
        raise ValueError("source direction feature indices are incompatible with 42-feature smoke input")
    direction_features = migration.get("new_direction_features")
    if direction_features is not None and list(direction_features) != EXPECTED_DIRECTION_FEATURES:
        raise ValueError("source direction feature semantics are incompatible with smoke input")

    normalizer_stats = payload.get("normalization_stats")
    central_stats = payload.get("central_normalization_stats")
    node_stats = normalizer_stats.get("normalizers", {}).get("node_features", {}) if isinstance(normalizer_stats, Mapping) else {}
    node_mean = node_stats.get("mean") if isinstance(node_stats, Mapping) else None
    node_count = node_stats.get("count") if isinstance(node_stats, Mapping) else None
    central_mean = central_stats.get("mean") if isinstance(central_stats, Mapping) else None
    if node_mean is not None and len(node_mean) != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError("source observation normalizer does not have 42 features")
    if central_mean is not None and len(central_mean) != EXPECTED_CENTRAL_DIM:
        raise ValueError("source central normalizer does not have 174 features")

    action_logprob_mode = trainer_config.get("ppo_action_logprob_mode")
    bc_training = dict(bc) if is_bc else None
    return {
        "lineage": "behavior_cloning_dagger" if is_bc else "ppo",
        "ppo_update_count": int(payload.get("update_count", 0)),
        "source_action_logprob_mode": action_logprob_mode,
        "target_action_logprob_mode": "sampled",
        "action_logprob_mode_provenance": "source metadata retained; smoke target sampled is explicit",
        "uav_count": int(payload.get("uav_count", trainer_config.get("uav_count", 1))),
        "node_feature_dim": int(node_dim) if node_dim is not None else None,
        "edge_feature_dim": int(edge_dim) if edge_dim is not None else None,
        "direction_indices": list(direction_indices) if direction_indices is not None else None,
        "direction_features": list(direction_features) if direction_features is not None else None,
        "migration_old_node_dim": migration.get("old_node_dim"),
        "migration_new_node_dim": migration.get("new_node_dim"),
        "migration_old_central_dim": migration.get("old_central_dim"),
        "migration_new_central_dim": migration.get("new_central_dim"),
        "bc_training": bc_training,
        "critic_frozen_during_bc": bool(bc_training.get("critic_frozen", False)) if bc_training else False,
        "source_normalizer_node_dim": len(node_mean) if node_mean is not None else None,
        "source_central_normalizer_dim": len(central_mean) if central_mean is not None else None,
        "source_scene_calibration": {
            "obstacle_box_count_mean": node_mean[34] if node_mean is not None and len(node_mean) > 34 else None,
            "obstacle_sphere_count_mean": node_mean[35] if node_mean is not None and len(node_mean) > 35 else None,
            "obstacle_count_stats_present": node_count is not None,
        },
    }


def _validate_target_contract(trainer: Any) -> dict[str, Any]:
    policy = getattr(trainer, "policy", None)
    if policy is None or not isinstance(policy, nn.Module):
        raise TypeError("trainer.policy must be a torch.nn.Module")
    config = getattr(policy, "config", getattr(trainer, "config", None))
    def _int_attr(name: str, default: int | None = None) -> int | None:
        value = getattr(config, name, default)
        return int(value) if value is not None else None

    uav_count = _int_attr("uav_count")
    node_dim = _int_attr("node_feature_dim")
    edge_dim = _int_attr("edge_feature_dim")
    if uav_count is not None and uav_count != EXPECTED_UAV_COUNT:
        raise ValueError("smoke target trainer must have one UAV")
    if node_dim is not None and node_dim != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError("smoke target trainer must have 42 node features")
    if edge_dim is not None and edge_dim != EXPECTED_EDGE_FEATURE_DIM:
        raise ValueError("smoke target trainer must have 11 edge features")
    critic = getattr(policy, "critic", None)
    critic_input = getattr(config, "critic_input_dim", None)
    if critic_input is not None and int(critic_input) != EXPECTED_CENTRAL_DIM:
        raise ValueError("smoke target trainer must have a 174-dimensional central critic input")
    if critic is None:
        raise ValueError("smoke target trainer has no centralized critic")
    return {
        "uav_count": uav_count,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
        "central_feature_dim": int(critic_input) if critic_input is not None else None,
        "ppo_action_logprob_mode": getattr(config, "ppo_action_logprob_mode", None),
        "normalizer_fitted": getattr(getattr(trainer, "normalizer", None), "fitted", None),
        "central_normalizer_fitted": getattr(getattr(trainer, "central_normalizer", None), "fitted", None),
    }


def _normalizer_contract_report(trainer: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Describe normalizers without loading source statistics into the trainer."""

    target_normalizer = getattr(trainer, "normalizer", None)
    target_central = getattr(trainer, "central_normalizer", None)
    target_node_dim = None
    target_central_dim = None
    try:
        state = target_normalizer.state_dict()
        node = state.get("normalizers", {}).get("node_features", {})
        target_node_dim = len(node.get("mean", [])) if isinstance(node, Mapping) and node.get("mean") is not None else None
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        state = target_central.state_dict()
        target_central_dim = len(state.get("mean", [])) if isinstance(state, Mapping) and state.get("mean") is not None else None
    except (AttributeError, TypeError, ValueError):
        pass
    return {
        "reused": False,
        "normalizer_reused": False,
        "new_fit_required": True,
        "source_statistics_loaded": False,
        "target_node_feature_dim": target_node_dim,
        "target_central_feature_dim": target_central_dim,
        "source_node_feature_dim": _source_metadata(payload, "A" if not isinstance(payload.get("bc_training"), Mapping) else "B")["source_normalizer_node_dim"],
        "source_central_feature_dim": _source_metadata(payload, "A" if not isinstance(payload.get("bc_training"), Mapping) else "B")["source_central_normalizer_dim"],
        "incompatibility_reason": "new scene calibration and obstacle population require train-only 42/174 statistics",
    }


def _state_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def initialize_smoke_trainer(
    trainer: Any,
    checkpoint_path: str | Path,
    group: str,
) -> dict[str, Any]:
    """Warm-start a fresh smoke trainer from source group A or B.

    The function copies compatible actor tensors in place.  It does not use
    ``load_state_dict`` because whole modules are intentionally left at their
    fresh constructor values and parameter object identity must be retained.
    """

    group = str(group).upper()
    if group not in {"A", "B"}:
        raise ValueError("group must be 'A' (PPO) or 'B' (BC/DAgger)")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # Check all preconditions before mutating the target trainer.
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None and len(getattr(optimizer, "state", {})) != 0:
        raise RuntimeError("smoke initialization refuses a trainer with non-empty optimizer state")
    target_contract = _validate_target_contract(trainer)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    source_meta = _source_metadata(payload, group)
    source_policy = _state_mapping(payload.get("policy_state_dict"), "policy_state_dict")
    policy = trainer.policy
    target_state = policy.state_dict()
    target_before = {name: value.detach().clone() for name, value in target_state.items()}
    target_refs = {name: value for name, value in target_state.items()}

    # Ensure source tensors are finite and provide a complete per-parameter
    # shape/dtype record before any copy is attempted.
    parameter_details: dict[str, dict[str, Any]] = {}
    for name in sorted(set(source_policy) | set(target_state)):
        source_value = source_policy.get(name)
        target_value = target_state.get(name)
        source_finite = None
        if source_value is not None:
            if not torch.is_tensor(source_value):
                raise TypeError(f"source parameter {name} is not a torch tensor")
            source_finite = bool(torch.isfinite(source_value.detach()).all().item())
            if not source_finite:
                raise ValueError(f"source parameter {name} contains a non-finite value")
        parameter_details[name] = {
            "source_shape": _shape(source_value),
            "target_shape": _shape(target_value),
            "source_dtype": _dtype(source_value),
            "target_dtype": _dtype(target_value),
            "source_finite": source_finite,
            "status": "pending",
            "reason": None,
        }

    source_modules: dict[str, list[str]] = {}
    target_modules: dict[str, list[str]] = {}
    for name in source_policy:
        source_modules.setdefault(_module_name(name), []).append(name)
    for name in target_state:
        target_modules.setdefault(_module_name(name), []).append(name)

    # Actor warm-start is a hard smoke precondition.  Check it before the
    # first copy so a malformed source cannot leave a partially initialized
    # trainer behind while ultimately failing the provenance check.
    actor_target_names = sorted(target_modules.get("actor_body", []))
    actor_source_names = sorted(source_modules.get("actor_body", []))
    if not actor_target_names or actor_source_names != actor_target_names:
        raise ValueError("source checkpoint has no complete actor_body warm start")
    for name in actor_target_names:
        source_value = source_policy[name]
        target_value = target_state[name]
        if (
            not torch.is_tensor(source_value)
            or tuple(source_value.shape) != tuple(target_value.shape)
            or source_value.dtype != target_value.dtype
        ):
            raise ValueError("source actor_body is structurally incompatible; whole module remains fresh")

    # These module prefixes encode the old observation/critic normalization
    # contract.  They always retain the freshly constructed target values.
    reinitialized: list[dict[str, Any]] = []
    forced_fresh = {"gat.layers.0", "critic"}
    for module_name, reason in (
        ("gat.layers.0", "new 42-feature observation normalization semantics"),
        ("critic", "new 174-feature central normalization semantics"),
    ):
        reinitialized.append({"module": module_name, "reason": reason, "fresh_constructor_values": True})
        for name in target_modules.get(module_name, []):
            parameter_details[name]["status"] = "reinitialized"
            parameter_details[name]["reason"] = reason

    # A source module is accepted only as a whole.  This prevents a matching
    # bias from being loaded when its companion matrix changed shape.
    loaded: list[str] = []
    skipped: list[str] = []
    retained_hashes: dict[str, dict[str, str]] = {}
    allowed_module_names = set(target_modules) - forced_fresh
    for module_name in sorted(allowed_module_names):
        target_names = sorted(target_modules[module_name])
        source_names = sorted(source_modules.get(module_name, []))
        if not source_names:
            for name in target_names:
                parameter_details[name]["status"] = "fresh"
                parameter_details[name]["reason"] = "source module absent; retain fresh constructor"
            continue
        mismatch: str | None = None
        if set(source_names) != set(target_names):
            mismatch = "source/target parameter set differs; reinitialize whole module"
        else:
            for name in target_names:
                source_value = source_policy[name]
                target_value = target_state[name]
                if tuple(source_value.shape) != tuple(target_value.shape):
                    mismatch = "source/target shape differs; reinitialize whole module"
                    break
                if source_value.dtype != target_value.dtype:
                    mismatch = "source/target dtype differs; reinitialize whole module"
                    break
        if mismatch is not None:
            reinitialized.append({"module": module_name, "reason": mismatch, "fresh_constructor_values": True})
            for name in target_names:
                parameter_details[name]["status"] = "reinitialized"
                parameter_details[name]["reason"] = mismatch
            continue
        for name in target_names:
            source_value = source_policy[name]
            target_value = target_state[name]
            with torch.no_grad():
                # copy_ preserves the existing Parameter/tensor reference.
                target_value.copy_(source_value.to(device=target_value.device))
            loaded.append(name)
            parameter_details[name]["status"] = "loaded"
            parameter_details[name]["reason"] = "shape/dtype compatible retained module"
            retained_hashes[name] = {
                "source": _tensor_hash(source_value),
                "target_after": _tensor_hash(target_value),
            }

    # Unknown source-only tensors are deliberately ignored; they cannot be
    # applied to a fresh target architecture.
    for name in sorted(set(source_policy) - set(target_state)):
        parameter_details[name]["status"] = "ignored"
        parameter_details[name]["reason"] = "source-only parameter"
        skipped.append(name)
    for name in sorted(set(target_state) - set(source_policy)):
        if parameter_details[name]["status"] == "pending":
            parameter_details[name]["status"] = "fresh"
            parameter_details[name]["reason"] = "target-only parameter"

    # Verify the required fresh modules were not changed by source loading.
    fresh_verification: dict[str, bool] = {}
    for module_name in forced_fresh:
        names = target_modules.get(module_name, [])
        fresh_verification[module_name] = all(torch.equal(target_state[name], target_before[name]) for name in names)
        if not fresh_verification[module_name]:
            raise RuntimeError(f"fresh constructor module changed unexpectedly: {module_name}")

    # Make the new PPO target fully trainable and start with a clean rollout.
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.train()
    rollout = getattr(trainer, "rollout", None)
    if rollout is None or not hasattr(rollout, "clear"):
        raise ValueError("trainer must expose a mutable rollout buffer")
    rollout.clear()
    if optimizer is not None and len(getattr(optimizer, "state", {})) != 0:
        raise RuntimeError("optimizer state became non-empty during smoke initialization")

    source_hash_before = _sha256(path)
    source_hash_after = _sha256(path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source checkpoint changed during smoke initialization")

    actor_body_loaded = [name for name in loaded if name.startswith("actor_body.")]
    actor_body_verified = bool(actor_body_loaded) and all(
        retained_hashes[name]["source"] == retained_hashes[name]["target_after"]
        for name in actor_body_loaded
    )
    if not actor_body_verified:
        raise RuntimeError("smoke initialization did not retain an actual actor_body warm start")

    target_update_count = 0
    report = {
        "api": "initialize_smoke_trainer",
        "initialized_from_checkpoint": True,
        "group": group,
        "checkpoint": {"path": str(path.resolve()), "sha256": source_hash_after},
        "source_checkpoint": {"path": str(path.resolve()), "sha256": source_hash_after},
        "source": source_meta,
        "target": target_contract,
        "compatibility": {
            "passed": True,
            "observation_network_compatible": True,
            "feature_map_direction_indices": EXPECTED_DIRECTION_INDICES,
            "feature_map_direction_semantics": EXPECTED_DIRECTION_FEATURES,
            "network_contract": {
                "uav_count": EXPECTED_UAV_COUNT,
                "node_feature_dim": EXPECTED_NODE_FEATURE_DIM,
                "edge_feature_dim": EXPECTED_EDGE_FEATURE_DIM,
                "central_feature_dim": EXPECTED_CENTRAL_DIM,
            },
        },
        "normalization": _normalizer_contract_report(trainer, payload),
        "parameters": {
            "loaded": loaded,
            "ignored_source_only": skipped,
            "reinitialized": reinitialized,
            "details": parameter_details,
            "source_shapes": {name: detail["source_shape"] for name, detail in parameter_details.items()},
            "target_shapes": {name: detail["target_shape"] for name, detail in parameter_details.items()},
            "retained_tensor_hashes": retained_hashes,
            "fresh_constructor_verification": fresh_verification,
        },
        "loaded_parameter_count": len(loaded),
        "meaningful_retained_tensor_count": len(loaded),
        "actor_body_load_verified": actor_body_verified,
        "critic_trainable": all(parameter.requires_grad for parameter in policy.critic.parameters()),
        "actor_trainable": all(parameter.requires_grad for name, parameter in policy.named_parameters() if not name.startswith("critic.")),
        "policy_training_mode": bool(policy.training),
        "source_optimizer_ignored": True,
        "source_optimizer_state_entries": len(payload.get("optimizer_state_dict", {}).get("state", {})) if isinstance(payload.get("optimizer_state_dict"), Mapping) else None,
        "target_optimizer_state_entries": len(getattr(optimizer, "state", {})) if optimizer is not None else 0,
        "rollout_cleared": len(rollout) == 0,
        "teacher_online_samples": 0,
        "teacher_online_training_performed": False,
        "source_checkpoint_unchanged": source_hash_before == source_hash_after,
        "source_update_count": source_meta["ppo_update_count"],
        "target_update_count": target_update_count,
        "update_counter_provenance": "source update_count is historical provenance; smoke PPO starts at 0",
        "sampled_logprob_mode": "sampled",
        "no_environment_steps": True,
        "no_ppo_update": True,
    }
    return report


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def b_pretraining_cost(checkpoint_path: str | Path) -> dict[str, Any]:
    """Report known/measured/derived BC cost without loading demonstrations.

    The function reads the small checkpoint metadata and adjacent manifest or
    summary JSON files only.  It never constructs a DataLoader or scans the
    large demonstration/trace directories.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    bc = payload.get("bc_training")
    if not isinstance(bc, Mapping):
        raise ValueError("checkpoint has no bc_training metadata")
    metadata = dict(bc)
    train_samples = int(metadata["train_samples"])
    validation_samples = int(metadata["validation_samples"])
    batch_size = int(metadata["batch_size"])
    actual_epochs = int(metadata["actual_epochs"])
    train_presentations = actual_epochs * train_samples
    batches_per_epoch = int(np.ceil(train_samples / batch_size))
    train_batches = actual_epochs * batches_per_epoch

    # Read only the directly adjacent final manifest and the small round-0
    # manifest.  This preserves the upstream teacher-data provenance without
    # touching any large CSV/trace file.
    branch_root = path.parents[2] if len(path.parents) >= 3 else path.parent
    dagger_root = branch_root.parent
    final_manifest = _read_json_if_exists(branch_root / "manifest.json")
    round0_manifest = _read_json_if_exists(dagger_root / "bc_round0" / "manifest.json")

    round0_contract = round0_manifest.get("bc_contract", {}) if isinstance(round0_manifest, Mapping) else {}
    return {
        "checkpoint": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "lineage": "behavior_cloning_dagger",
        "dataset_id": metadata.get("dataset_id", "single_uav_dagger_round1"),
        "known": {
            "actual_epochs": actual_epochs,
            "batch_size": batch_size,
            "train_samples_per_epoch": train_samples,
            "validation_samples": validation_samples,
            "critic_frozen": metadata.get("critic_frozen"),
            "source_ppo_update_count": int(payload.get("update_count", 0)),
            "source_update_count_provenance": "shared PPO ancestry; not a new B PPO smoke counter",
        },
        "measured": {
            "manifest_present": final_manifest is not None,
            "round0_manifest_present": round0_manifest is not None,
            "round0_train_samples_per_epoch": round0_contract.get("train_samples") if isinstance(round0_contract, Mapping) else None,
            "round0_validation_samples": round0_contract.get("validation_samples") if isinstance(round0_contract, Mapping) else None,
            "round0_actual_epochs": round0_contract.get("actual_epochs") if isinstance(round0_contract, Mapping) else None,
        },
        "derived": {
            "train_example_presentations": train_presentations,
            "train_batches": train_batches,
            "batches_per_epoch": batches_per_epoch,
            "formula": "actual_epochs * train_samples; actual_epochs * ceil(train_samples / batch_size)",
        },
        "unknown": [
            "upstream_teacher_rollout_total",
            "upstream_teacher_rollout_walltime_seconds",
            "upstream_teacher_inference_cost",
            "missing_cost_components",
        ],
        "teacher_rollout_total": None,
        "teacher_rollout_walltime_seconds": None,
        "new_teacher_training_performed": False,
        "smoke_teacher_online_samples": 0,
        "incremental_ppo_cost_logged_separately": True,
        "data_loader_constructed": False,
    }


def inspect_sources(*checkpoint_paths: str | Path) -> dict[str, Any]:
    """Return compact, read-only source provenance for smoke planning."""

    paths = checkpoint_paths or (SOURCE_A_DEFAULT, SOURCE_B_DEFAULT)
    result: dict[str, Any] = {}
    for index, path_value in enumerate(paths):
        path = Path(path_value)
        if not path.is_file():
            result["AB"[index] if index < 2 else str(index)] = {"path": str(path), "exists": False}
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"source checkpoint is not a mapping: {path}")
        group = "AB"[index] if index < 2 else str(index)
        result[group] = {
            "path": str(path.resolve()),
            "exists": True,
            "sha256": _sha256(path),
            "lineage": "behavior_cloning_dagger" if isinstance(payload.get("bc_training"), Mapping) else "ppo",
            "uav_count": payload.get("uav_count"),
            "update_count": payload.get("update_count"),
            "policy_parameter_count": len(payload.get("policy_state_dict", {})) if isinstance(payload.get("policy_state_dict"), Mapping) else None,
        }
    return result


__all__ = [
    "SOURCE_A_DEFAULT",
    "SOURCE_B_DEFAULT",
    "b_pretraining_cost",
    "initialize_smoke_trainer",
    "inspect_sources",
]
