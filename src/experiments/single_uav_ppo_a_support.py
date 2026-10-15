"""A-only foundation and audit helpers for the single-UAV PPO protocol.

This module is intentionally small and side-effect free at import time.  It
loads the migrated PPO checkpoint from lineage A, reuses the frozen train
statistics artifact, and provides diagnostics around a real ``MAPPOTrainer``
update.  It never loads the BC/teacher checkpoint and never collects an
environment transition itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ..learning.mappo import PolicyDecision
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer


EXPECTED_NODE_FEATURE_DIM = 42
EXPECTED_CENTRAL_FEATURE_DIM = 174
EXPECTED_DIRECTION_INDICES = (39, 40, 41)
EXPECTED_DIRECTION_FEATURES = (
    "relative_obstacle_dx_body_m",
    "relative_obstacle_dy_body_m",
    "relative_obstacle_dz_body_m",
)
EXPECTED_A_SHA256 = "6561bf3cbfcf4036db1a13eb118190f415c10d6de3a0d988a85390b6e3b6bfdc"
EXPECTED_A_UPDATE_COUNT = 100
FEATURE_INDEX_MAP = {
    "estimated_state": [0, 15],
    "estimated_velocity": [3, 6],
    "estimated_attitude": [6, 9],
    "diagonal_covariance": [15, 30],
    "goal_delta": [30, 33],
    "obstacle_aggregate": [33, 37],
    "braking_features": [37, 39],
    "obstacle_body_direction": [39, 42],
}

SOURCE_A_DEFAULT = Path(
    "data/synthetic/single_uav_external_recovery_method_study_seed2026/"
    "dagger_bc/baseline_observation_migration/checkpoints/"
    "stage2a_observation_compatible_init.pt"
)
SHARED_STATISTICS_DEFAULT = Path(
    "data/synthetic/single_uav_static_ppo_smoke_v1_seed2026/shared_train_statistics.json"
)


class _FrozenObservationNormalizer(ObservationNormalizer):
    """ObservationNormalizer-compatible read-only view over train statistics."""

    def fit(self, *_args: Any, **_kwargs: Any) -> "_FrozenObservationNormalizer":
        raise RuntimeError("A foundation observation statistics are frozen; refit is forbidden")

    update = fit


class _FrozenFeatureNormalizer(FeatureNormalizer):
    """FeatureNormalizer-compatible read-only view over train statistics."""

    def fit(self, *_args: Any, **_kwargs: Any) -> "_FrozenFeatureNormalizer":
        raise RuntimeError("A foundation central statistics are frozen; refit is forbidden")

    update = fit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _finite_array(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and array.size != int(size):
        raise ValueError(f"{name} must have {size} entries")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalizer_state(normalizer: Any, *, central: bool = False) -> Mapping[str, Any]:
    state = normalizer.state_dict()
    if central:
        return state
    node = state.get("normalizers", {}).get("node_features")
    if not isinstance(node, Mapping):
        raise ValueError("A foundation requires a fitted node_features normalizer")
    return node


def load_shared_a_statistics(
    path: str | Path = SHARED_STATISTICS_DEFAULT,
) -> tuple[ObservationNormalizer, FeatureNormalizer, dict[str, Any], dict[str, Any]]:
    """Load the frozen shared train statistics used by both foundation arms.

    The file is verified in place and then copied into normalizer objects.  No
    fitting, calibration, or mutation of the persisted artifact occurs.
    """

    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    with artifact.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("shared statistics artifact must be a JSON object")
    if str(payload.get("fit_split", "")).lower() != "train":
        raise ValueError("shared statistics must identify fit_split='train'")
    if payload.get("frozen_during_ppo_and_validation") is not True:
        raise ValueError("shared statistics must be marked frozen during PPO and validation")

    observation_state = payload.get("observation_normalizer")
    central_state = payload.get("central_normalizer")
    if not isinstance(observation_state, Mapping) or not isinstance(central_state, Mapping):
        raise ValueError("shared statistics must contain observation and central normalizers")
    observation_copy = ObservationNormalizer.from_state_dict(observation_state)
    normalizer = _FrozenObservationNormalizer(eps=observation_copy.eps, clip=observation_copy.clip)
    normalizer.normalizers = copy.deepcopy(observation_copy.normalizers)
    central_copy = FeatureNormalizer.from_state_dict(central_state)
    central = _FrozenFeatureNormalizer(eps=central_copy.eps, clip=central_copy.clip)
    central.mean = central_copy.mean.copy() if central_copy.mean is not None else None
    central.scale = central_copy.scale.copy() if central_copy.scale is not None else None
    central.count = central_copy.count.copy() if central_copy.count is not None else None
    node = _normalizer_state(normalizer)
    node_mean = _finite_array(node.get("mean"), name="node mean", size=EXPECTED_NODE_FEATURE_DIM)
    node_scale = _finite_array(node.get("scale"), name="node scale", size=EXPECTED_NODE_FEATURE_DIM)
    node_count = _finite_array(node.get("count"), name="node count", size=EXPECTED_NODE_FEATURE_DIM)
    central_mean = _finite_array(central.mean, name="central mean", size=EXPECTED_CENTRAL_FEATURE_DIM)
    central_scale = _finite_array(central.scale, name="central scale", size=EXPECTED_CENTRAL_FEATURE_DIM)
    central_count = _finite_array(central.count, name="central count", size=EXPECTED_CENTRAL_FEATURE_DIM)
    for name, values in (("node scale", node_scale), ("central scale", central_scale)):
        if np.any(values <= 0.0):
            raise ValueError(f"{name} must be strictly positive")
    for name, values in (("node count", node_count), ("central count", central_count)):
        if np.any(values <= 0.0) or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} must contain positive integer counts")
    if not np.all(node_scale[3:6] >= 6.0) or not np.all(node_scale[39:42] > 0.0):
        raise ValueError("velocity and direction scales do not satisfy the A contract")
    if int(node_count[0]) != 306 or int(central_count[0]) != 306:
        raise ValueError("shared statistics must use all 306 train samples")
    if not np.all(node_count == 306) or not np.all(central_count == 306):
        raise ValueError("all node and central statistics must have count 306")
    potential = payload.get("potential_statistics")
    if not isinstance(potential, Mapping):
        raise ValueError("shared statistics must contain potential_statistics")
    potential = copy.deepcopy(dict(potential))
    if str(potential.get("fit_split", "")).lower() != "train":
        raise ValueError("potential statistics must identify fit_split='train'")
    for name, value in potential.items():
        if isinstance(value, (int, float)) and not np.isfinite(float(value)):
            raise ValueError(f"potential statistic {name} is non-finite")
    for key in ("u_scale", "c_scale", "delta_u_scale", "delta_goal_scale"):
        if float(potential.get(key, 0.0)) <= 0.0:
            raise ValueError(f"potential statistic {key} must be positive")
    if int(potential.get("sample_count", 0)) != 288:
        raise ValueError("potential statistics must retain the historical 288 calibration provenance")

    source_sha = _sha256(artifact)
    provenance = {
        "artifact_path": str(artifact),
        "source_sha256": source_sha,
        "fit_split": "train",
        "feature_index_map": copy.deepcopy(FEATURE_INDEX_MAP),
        "node_feature_dim": EXPECTED_NODE_FEATURE_DIM,
        "central_feature_dim": EXPECTED_CENTRAL_FEATURE_DIM,
        "node_count": 306,
        "central_count": 306,
        "velocity_indices": [3, 4, 5],
        "direction_indices": list(EXPECTED_DIRECTION_INDICES),
        "normalizer_state_sha256": _canonical_hash(observation_state),
        "central_normalizer_state_sha256": _canonical_hash(central_state),
        "statistics_frozen": True,
        "refit_performed": False,
    }
    return normalizer, central, potential, provenance


def _source_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if isinstance(payload.get("bc_training"), Mapping):
        raise ValueError("A foundation refuses a checkpoint with bc_training metadata")
    state = payload.get("policy_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("A source checkpoint must contain policy_state_dict")
    for name, value in state.items():
        if not torch.is_tensor(value) or not torch.isfinite(value.detach()).all().item():
            raise ValueError(f"source tensor {name} is invalid or non-finite")
    return state  # type: ignore[return-value]


def _module_for(name: str) -> str:
    if name.startswith("gat.layers.0."):
        return "first_gat"
    if name.startswith("gat.layers."):
        return "deep_gat"
    if name.startswith("actor_body."):
        return "actor_body"
    if name.startswith("delta_mean_head."):
        return "motion_head"
    if name == "delta_logstd":
        return "log_std"
    if name.startswith("measurement_head."):
        return "measurement_head"
    if name.startswith("bandwidth_head."):
        return "bandwidth_head"
    if name.startswith("critic."):
        return "critic"
    return name.split(".", 1)[0]


def parameter_snapshot(policy: Any) -> dict[str, torch.Tensor]:
    """Clone every policy parameter without changing the live objects."""

    if not hasattr(policy, "named_parameters"):
        raise TypeError("policy must expose named_parameters")
    return {name: parameter.detach().cpu().clone() for name, parameter in policy.named_parameters()}


def grouped_parameter_deltas(before: Mapping[str, torch.Tensor], policy: Any) -> dict[str, Any]:
    """Return L2 and changed-tensor counts for the foundation audit groups."""

    groups = ("first_gat", "deep_gat", "actor_body", "motion_head", "log_std", "measurement_head", "bandwidth_head", "critic")
    result: dict[str, Any] = {}
    for group in groups:
        squares = 0.0
        changed = 0
        tensors = 0
        for name, value in policy.named_parameters():
            if _module_for(name) != group:
                continue
            if name not in before:
                raise KeyError(f"parameter snapshot lacks {name}")
            delta = value.detach().cpu().double() - before[name].double()
            squares += float(delta.square().sum())
            changed += int(bool(torch.any(delta != 0.0).item()))
            tensors += 1
        result[group] = {"l2": float(math.sqrt(squares)), "changed_tensor_count": changed, "tensor_count": tensors}
    actor_groups = [name for name in groups if name != "critic"]
    result["total_actor"] = {
        "l2": float(math.sqrt(sum(float(result[name]["l2"]) ** 2 for name in actor_groups))),
        "changed_tensor_count": int(sum(int(result[name]["changed_tensor_count"]) for name in actor_groups)),
        "tensor_count": int(sum(int(result[name]["tensor_count"]) for name in actor_groups)),
    }
    result["gat_plus_critic"] = {
        "l2": float(math.sqrt(float(result["first_gat"]["l2"]) ** 2 + float(result["deep_gat"]["l2"]) ** 2 + float(result["critic"]["l2"]) ** 2)),
        "changed_tensor_count": int(sum(int(result[name]["changed_tensor_count"]) for name in ("first_gat", "deep_gat", "critic"))),
        "tensor_count": int(sum(int(result[name]["tensor_count"]) for name in ("first_gat", "deep_gat", "critic"))),
    }
    return result


def _target_contract(trainer: Any) -> dict[str, Any]:
    policy = getattr(trainer, "policy", None)
    config = getattr(trainer, "config", getattr(policy, "config", None))
    if policy is None or config is None:
        raise TypeError("trainer must expose policy and config")
    checks = {
        "uav_count": int(getattr(config, "uav_count", -1)),
        "node_feature_dim": int(getattr(config, "node_feature_dim", -1)),
        "edge_feature_dim": int(getattr(config, "edge_feature_dim", -1)),
        "critic_input_dim": int(getattr(config, "critic_input_dim", -1)),
        "ppo_action_logprob_mode": str(getattr(config, "ppo_action_logprob_mode", "")),
    }
    if checks["uav_count"] != 1 or checks["node_feature_dim"] != 42 or checks["edge_feature_dim"] != 11 or checks["critic_input_dim"] != 174:
        raise ValueError(f"target trainer is incompatible with the A foundation: {checks}")
    if checks["ppo_action_logprob_mode"] != "sampled":
        raise ValueError("A foundation requires sampled PPO action log probabilities")
    if getattr(trainer, "normalizer", None) is None or not trainer.normalizer.fitted:
        raise ValueError("target trainer must use fitted shared statistics")
    if getattr(trainer, "central_normalizer", None) is None or not trainer.central_normalizer.fitted:
        raise ValueError("target trainer must use fitted central statistics")
    # The approved baseline factory wraps these objects in read-only
    # Frozen*Normalizer views, which intentionally expose only state_dict(),
    # fitted, transform and fit-forbidden behavior.  Verify dimensions from
    # the serialized state so both raw and wrapped normalizers are accepted.
    try:
        node_state = trainer.normalizer.state_dict().get("normalizers", {}).get("node_features", {})
        node_dim = len(node_state.get("mean", []))
        central_state = trainer.central_normalizer.state_dict()
        central_dim = len(central_state.get("mean", []))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("target trainer normalizer state is not inspectable") from exc
    if node_dim != EXPECTED_NODE_FEATURE_DIM or central_dim != EXPECTED_CENTRAL_FEATURE_DIM:
        raise ValueError("target trainer normalizer dimensions are incompatible")
    return checks


def initialize_a_foundation(
    trainer: Any,
    source_path: str | Path = SOURCE_A_DEFAULT,
    arm: str = "G",
    *,
    old_v1_initial_path: str | Path | None = None,
) -> dict[str, Any]:
    """Warm-start a fresh trainer from the migrated PPO lineage-A source.

    ``G`` retains the complete fresh first GAT layer; ``P`` copies it along
    with the rest of the compatible actor.  The critic, optimizer state, and
    rollout state are always fresh.
    """

    arm = str(arm).upper()
    if arm not in {"G", "P"}:
        raise ValueError("arm must be 'G' or 'P'")
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    target_contract = _target_contract(trainer)
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None or len(getattr(optimizer, "state", {})) != 0:
        raise RuntimeError("A foundation requires a fresh optimizer with empty state")
    rollout = getattr(trainer, "rollout", None)
    if rollout is None or not hasattr(rollout, "clear"):
        raise ValueError("trainer must expose a rollout buffer")
    if len(rollout) != 0:
        raise RuntimeError("A foundation refuses a non-empty rollout; caller must create a fresh trainer")

    source_hash_before = _sha256(path)
    default_source = SOURCE_A_DEFAULT.expanduser().resolve()
    if path == default_source and source_hash_before != EXPECTED_A_SHA256:
        raise ValueError("the configured A source checkpoint SHA-256 does not match the locked manifest")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("A source checkpoint must be a mapping")
    source = _source_state(payload)
    if int(payload.get("uav_count", -1)) != 1:
        raise ValueError("A source checkpoint must be single-UAV")
    migration = payload.get("stage2a_migration")
    if not isinstance(migration, Mapping) or int(migration.get("new_node_dim", -1)) != 42 or int(migration.get("new_central_dim", -1)) != 174:
        raise ValueError("A source checkpoint lacks the 42/174 migration contract")
    if list(migration.get("new_direction_indices", [])) != list(EXPECTED_DIRECTION_INDICES):
        raise ValueError("A source checkpoint has incompatible direction indices")
    if list(migration.get("new_direction_features", [])) != list(EXPECTED_DIRECTION_FEATURES):
        raise ValueError("A source checkpoint has incompatible direction feature semantics")
    source_update_count = int(payload.get("update_count", -1))
    if source_update_count != EXPECTED_A_UPDATE_COUNT:
        raise ValueError("A source checkpoint update_count must remain historical provenance 100")
    source_mode = str((payload.get("trainer_config") or {}).get("ppo_action_logprob_mode", ""))
    if source_mode != "executed":
        raise ValueError("the migrated A source must retain executed-mode source provenance")

    target_params = dict(trainer.policy.named_parameters())
    source_names = set(source)
    target_names = set(target_params)
    if source_names != target_names:
        raise ValueError("source and target policy parameter sets differ")
    before = parameter_snapshot(trainer.policy)
    ids_before = {name: id(parameter) for name, parameter in target_params.items()}
    loaded: list[str] = []
    fresh: list[str] = []
    for name in sorted(target_names):
        target_value = target_params[name]
        source_value = source[name]
        if tuple(target_value.shape) != tuple(source_value.shape) or target_value.dtype != source_value.dtype:
            raise ValueError(f"source tensor {name} has incompatible shape or dtype")
        is_critic = name.startswith("critic.")
        is_first_gat = name.startswith("gat.layers.0.")
        if is_critic or (arm == "G" and is_first_gat):
            fresh.append(name)
            continue
        with torch.no_grad():
            target_value.copy_(source_value.to(device=target_value.device))
        loaded.append(name)

    # The migrated source explicitly zeroes only the appended input columns.
    for name in ("gat.layers.0.node_projection.weight", "gat.layers.0.residual.weight"):
        value = source.get(name)
        if value is None or value.shape[-1] < 42 or not torch.equal(value[:, 39:42], torch.zeros_like(value[:, 39:42])):
            raise ValueError(f"source migration tensor {name} does not preserve zero appended columns")

    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(True)
    trainer.policy.train()
    if len(rollout) != 0 or len(getattr(optimizer, "state", {})) != 0:
        raise RuntimeError("foundation initialization changed fresh rollout or optimizer state")
    if any(id(parameter) != ids_before[name] for name, parameter in trainer.policy.named_parameters()):
        raise RuntimeError("foundation initialization replaced a parameter object")
    source_hash_after = _sha256(path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("A source checkpoint changed during initialization")
    after = parameter_snapshot(trainer.policy)
    source_hashes = {name: _tensor_hash(source[name]) for name in loaded}
    target_hashes = {name: _tensor_hash(after[name]) for name in loaded}
    if source_hashes != target_hashes:
        raise RuntimeError("loaded A tensors do not match source hashes")
    old_v1_report: dict[str, Any] | None = None
    if old_v1_initial_path is not None:
        old_path = Path(old_v1_initial_path).expanduser().resolve()
        old_payload = torch.load(old_path, map_location="cpu", weights_only=False)
        old_state = _source_state(old_payload) if isinstance(old_payload, Mapping) else {}
        names = [name for name in old_state if name.startswith("gat.layers.0.")]
        target_first_gat_hashes = {
            name: _tensor_hash(after[name])
            for name in after
            if name.startswith("gat.layers.0.")
        }
        old_hashes = {name: _tensor_hash(old_state[name]) for name in names}
        old_v1_report = {
            "path": str(old_path),
            "first_gat_hashes": old_hashes,
            "target_first_gat_hashes": target_first_gat_hashes,
            "target_matches_old_v1_initial": bool(
                arm == "G" and old_hashes and old_hashes == target_first_gat_hashes
            ),
        }

    source_metadata = {
        "path": str(path),
        "sha256": source_hash_after,
        "update_count": source_update_count,
        "source_action_logprob_mode": source_mode,
        "bc_training_absent": "bc_training" not in payload,
        "migration": copy.deepcopy(dict(migration)),
    }
    return {
        "api": "initialize_a_foundation",
        "arm": arm,
        "target": target_contract,
        "source": source_metadata,
        "source_optimizer_ignored": True,
        "ignored_source_optimizer": True,
        "source_optimizer_state_entries": len((payload.get("optimizer_state_dict") or {}).get("state", {})) if isinstance(payload.get("optimizer_state_dict"), Mapping) else 0,
        "target_optimizer_state_entries": len(getattr(optimizer, "state", {})),
        "rollout_cleared": len(rollout) == 0,
        "loaded_parameter_names": loaded,
        "fresh_parameter_names": fresh,
        "loaded_parameter_count": len(loaded),
        "fresh_constructor_parameter_count": len(fresh),
        "first_gat_fresh": arm == "G",
        "first_gat_source_loaded": arm == "P",
        "critic_fresh": all(name.startswith("critic.") for name in fresh if name.startswith("critic.")),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in trainer.policy.parameters()),
        "policy_training_mode": bool(trainer.policy.training),
        "parameter_object_ids_preserved": True,
        "source_tensor_hashes": source_hashes,
        "target_loaded_tensor_hashes": target_hashes,
        "parameter_hashes_before": {name: _tensor_hash(value) for name, value in before.items()},
        "parameter_hashes_after": {name: _tensor_hash(value) for name, value in after.items()},
        "trainer_config": {
            "action_logprob_mode": target_contract["ppo_action_logprob_mode"],
            "uav_count": 1,
            "node_feature_dim": 42,
            "central_feature_dim": 174,
            "edge_feature_dim": 11,
            "gradient_clip_scope": str(getattr(trainer.config, "gradient_clip_scope", "joint")),
            "policy_epochs": int(getattr(trainer.config, "policy_epochs", 4)),
            "minibatch_size": int(getattr(trainer.config, "minibatch_size", 64)),
            "learning_rate": float(getattr(trainer.config, "learning_rate", float("nan"))),
            "max_grad_norm": float(getattr(trainer.config, "max_grad_norm", float("nan"))),
            "ppo_hyperparameters_preserved": True,
        },
        "migration_direction_columns": {
            "indices": list(EXPECTED_DIRECTION_INDICES),
            "features": list(EXPECTED_DIRECTION_FEATURES),
            "units": "m",
            "source_projection_and_residual_columns_zero": True,
            "target_columns_trainable": True,
        },
        "source_checkpoint_unchanged": source_hash_before == source_hash_after,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_update_count_is_historical": True,
        "target_update_count": 0,
        "normalization_reused": True,
        "normalizer_refit": False,
        "old_v1_initial_verification": old_v1_report,
    }


def capture_policy_distribution(trainer: Any, decision: PolicyDecision) -> dict[str, Any]:
    """Capture sampled-policy distribution data without drawing another action."""

    result: dict[str, Any] = {"action_logprob_mode": "sampled", "uavs": {}, "units": {"delta_v": "m/s", "raw_delta": "dimensionless Gaussian pre-tanh"}}
    for uav_id, observation in decision.observations.items():
        categorical, bandwidth_logits, delta_distribution, _mask = trainer.policy._actor_distributions(observation)
        base = delta_distribution.base_dist
        raw_mean = base.mean.detach().cpu().numpy().astype(float).reshape(-1)
        raw_std = base.scale.detach().cpu().numpy().astype(float).reshape(-1)
        sample = decision.samples[uav_id]
        limit = float(trainer.policy._delta_component_limit())
        result["uavs"][str(uav_id)] = {
            "raw_mean": raw_mean.tolist(),
            "raw_std": raw_std.tolist(),
            "bounded_mean": (np.tanh(raw_mean) * limit).tolist(),
            "sample_delta_raw": np.asarray(sample.delta_raw, dtype=float).reshape(-1).tolist(),
            "sample_delta_v": np.asarray(sample.delta_v, dtype=float).reshape(-1).tolist(),
            "delta_component_limit_mps": limit,
            "raw_delta_norm_bound_mps": float(math.sqrt(3.0) * limit),
            "measurement_probabilities": categorical.probs.detach().cpu().numpy().astype(float).tolist(),
            "bandwidth_logits": bandwidth_logits.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
            "old_logprob": float(sample.old_log_prob),
            "bounded_mean_units": "m/s",
        }
    if len(result["uavs"]) == 1:
        result.update(next(iter(result["uavs"].values())))
    return result


def cosine_alignment_utility(vector: Iterable[float], goal_vector: Iterable[float]) -> float:
    """Return finite cosine alignment for optional root telemetry."""

    value = _finite_array(vector, name="vector")
    goal = _finite_array(goal_vector, name="goal_vector", size=value.size)
    denominator = float(np.linalg.norm(value) * np.linalg.norm(goal))
    return 0.0 if denominator <= 1e-12 else float(np.dot(value, goal) / denominator)


# Short aliases keep root telemetry call sites readable while retaining the
# descriptive public name above.
cosine_utility = cosine_alignment_utility
cosine_goal_utility = cosine_alignment_utility


class PPOUpdateAudit:
    """Observe actual optimizer steps while leaving the trainer update intact."""

    GROUPS = ("first_gat", "deep_gat", "actor_body", "motion_head", "log_std", "measurement_head", "bandwidth_head", "critic")

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.optimizer = trainer.optimizer
        self.steps = 0
        self.gradients: list[dict[str, Any]] = []
        self._original_step = self.optimizer.step
        self._installed = False
        self.install()

    def install(self) -> None:
        if self._installed:
            return
        original = self._original_step

        def observed_step(*args: Any, **kwargs: Any) -> Any:
            grouped: dict[str, dict[str, Any]] = {
                group: {"l2": 0.0, "max_abs": 0.0, "nonzero_parameter_count": 0, "parameter_count": 0, "finite": True}
                for group in self.GROUPS
            }
            for name, parameter in self.trainer.policy.named_parameters():
                group = _module_for(name)
                if group not in grouped:
                    continue
                item = grouped[group]
                item["parameter_count"] += 1
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                finite = bool(torch.isfinite(gradient).all().item())
                item["finite"] = bool(item["finite"] and finite)
                item["l2"] += float(gradient.double().square().sum())
                item["max_abs"] = max(float(item["max_abs"]), float(gradient.abs().max().cpu())) if gradient.numel() else float(item["max_abs"])
                item["nonzero_parameter_count"] += int(bool(torch.any(gradient != 0.0).item()))
            for item in grouped.values():
                item["l2"] = float(math.sqrt(item["l2"]))
            output = original(*args, **kwargs)
            self.steps += 1
            self.gradients.append({"step": self.steps, "groups": grouped})
            return output

        self.optimizer.step = observed_step  # type: ignore[method-assign]
        self._installed = True

    def close(self) -> None:
        if self._installed:
            self.optimizer.step = self._original_step  # type: ignore[method-assign]
            self._installed = False

    def optimizer_state_steps(self) -> dict[str, int]:
        names = dict(self.trainer.policy.named_parameters())
        return {
            name: int(state["step"].item() if torch.is_tensor(state.get("step")) else state.get("step", 0))
            for parameter, state in self.optimizer.state.items()
            for name, candidate in names.items()
            if candidate is parameter and "step" in state
        }


def _normalizer_hash(value: Any) -> str:
    return _canonical_hash(value.state_dict())


def update_with_audit(trainer: Any, audit: PPOUpdateAudit | None = None, *, expected_samples: int = 120) -> dict[str, Any]:
    """Run one ordinary PPO update and return the A foundation audit record."""

    config = trainer.config
    if int(getattr(config, "policy_epochs", -1)) != 4 or int(getattr(config, "minibatch_size", -1)) != 64:
        raise ValueError("foundation update requires policy_epochs=4 and minibatch_size=64")
    if str(getattr(config, "ppo_action_logprob_mode", "")) != "sampled":
        raise ValueError("foundation update requires sampled action log probabilities")
    if any(bool(getattr(config, name, False)) for name in ("measurement_proxy_guidance_enabled", "measurement_persistent_proxy_guidance_enabled")):
        raise ValueError("foundation update refuses teacher/proxy guidance")
    if not trainer.rollout:
        raise ValueError("foundation update requires the fresh online rollout")
    records = trainer._training_records()
    if len(records) != int(expected_samples) * len(trainer.policy.uav_ids):
        raise ValueError(f"expected {expected_samples} samples per UAV, received {len(records)}")
    before = parameter_snapshot(trainer.policy)
    node_hash_before = _normalizer_hash(trainer.normalizer)
    central_hash_before = _normalizer_hash(trainer.central_normalizer)
    old_logprob_errors: list[float] = []
    for record in records:
        current_logprob, _entropy = trainer.policy.evaluate_actions(record["observation"], record["sample"])
        old_logprob_errors.append(abs(float(current_logprob.detach().cpu()) - float(record["sample"].old_log_prob)))
    preupdate_parameters = parameter_snapshot(trainer.policy)
    advantages = np.asarray([float(record["advantage"]) for record in records], dtype=float)
    returns = np.asarray([float(record["return"]) for record in records], dtype=float)
    old_values = np.asarray([float(record["old_value"]) for record in records], dtype=float)
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all() or not np.isfinite(old_values).all():
        raise ValueError("training records contain non-finite GAE/value data")
    transitions = list(trainer.rollout)
    # Reuse the trainer's own bootstrap helper and mirror its terminal rule
    # exactly.  In particular, a truncated transition is bootstrap-able only
    # when bootstrap_time_limits is enabled; a true terminal always receives
    # zero bootstrap value, even if the predicted value happens to be zero.
    transition_next_values = [float(trainer._next_value(item)) for item in transitions]
    last_next_value = transition_next_values[-1]
    bootstrap_time_limits = bool(getattr(config, "bootstrap_time_limits", False))
    terminal_for_bootstrap = [
        bool(
            item.terminated
            or (item.done and not item.truncated)
            or (item.truncated and not bootstrap_time_limits)
        )
        for item in transitions
    ]
    gae_recursion_boundary = [bool(item.done) for item in transitions]
    nonterminal_bootstrap_count = int(sum(not value for value in terminal_for_bootstrap))
    gae_recursion_boundary_count = int(sum(gae_recursion_boundary))

    # Independently replay the small reverse GAE recurrence used by
    # MAPPOTrainer._training_records.  This makes the audit explicit about
    # reset boundaries instead of inferring them from a numeric value.
    raw_advantages = np.zeros(len(transitions), dtype=np.float64)
    next_advantage = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        reward_values = [float(item.rewards.get(agent, 0.0)) for agent in trainer.policy.uav_ids]
        reward = float(np.mean(reward_values)) if reward_values else 0.0
        bootstrap_nonterminal = 0.0 if terminal_for_bootstrap[index] else 1.0
        gae_continuation = 0.0 if gae_recursion_boundary[index] else 1.0
        delta = reward + float(config.gamma) * transition_next_values[index] * bootstrap_nonterminal - float(item.decision.value)
        raw_advantages[index] = delta + float(config.gamma) * float(config.gae_lambda) * gae_continuation * next_advantage
        next_advantage = raw_advantages[index]
    raw_mean = float(np.repeat(raw_advantages, len(trainer.policy.uav_ids)).mean())
    raw_std = float(np.repeat(raw_advantages, len(trainer.policy.uav_ids)).std())
    expected_normalized_advantages = (raw_advantages - raw_mean) / max(raw_std, 1e-8)
    gae_advantage_match = True
    return_match = True
    record_audit: list[dict[str, Any]] = []
    agent_count = len(trainer.policy.uav_ids)
    for record_index, record in enumerate(records):
        transition_index = record_index // agent_count
        item = transitions[transition_index]
        expected_advantage = float(expected_normalized_advantages[transition_index])
        expected_return = float(raw_advantages[transition_index] + item.decision.value)
        gae_advantage_match = gae_advantage_match and bool(
            np.isclose(float(record["advantage"]), expected_advantage, atol=1e-6, rtol=1e-6)
        )
        return_match = return_match and bool(
            np.isclose(float(record["return"]), expected_return, atol=1e-6, rtol=1e-6)
        )
        record_audit.append(
            {
                "record_index": record_index,
                "transition_index": transition_index,
                "uav_id": str(record["uav_id"]),
                "advantage": float(record["advantage"]),
                "return": float(record["return"]),
                "old_value": float(record["old_value"]),
                "expected_advantage_from_recurrence": expected_advantage,
                "expected_return_from_recurrence": expected_return,
                "next_value": transition_next_values[transition_index],
                "terminated": bool(item.terminated),
                "truncated": bool(item.truncated),
                "done": bool(item.done),
                "terminal_for_bootstrap": terminal_for_bootstrap[transition_index],
                "gae_recursion_continues_to_previous_transition": not gae_recursion_boundary[transition_index],
            }
        )
    audit = audit or PPOUpdateAudit(trainer)
    if audit.trainer is not trainer:
        raise ValueError("audit belongs to a different trainer")
    update_stats = trainer.update()
    after = parameter_snapshot(trainer.policy)
    if len(trainer.rollout) != 0:
        raise RuntimeError("PPO update did not empty the rollout buffer")
    node_hash_after = _normalizer_hash(trainer.normalizer)
    central_hash_after = _normalizer_hash(trainer.central_normalizer)
    optimizer_steps = audit.optimizer_state_steps()
    return {
        "api": "update_with_audit",
        "expected_samples": int(expected_samples),
        "record_count": len(records),
        "advantages": {"mean": float(advantages.mean()), "std": float(advantages.std()), "min": float(advantages.min()), "max": float(advantages.max())},
        "returns": {"min": float(returns.min()), "max": float(returns.max())},
        "value_mse_preupdate": float(np.mean((old_values - returns) ** 2)),
        "max_on_policy_logprob_error": float(max(old_logprob_errors, default=0.0)),
        "on_policy_logprob_within_tolerance": bool(max(old_logprob_errors, default=0.0) < 1e-5),
        "last_next_value": last_next_value,
        "last_transition_flags": {
            "terminated": bool(transitions[-1].terminated),
            "truncated": bool(transitions[-1].truncated),
            "done": bool(transitions[-1].done),
            "terminal_for_bootstrap": bool(terminal_for_bootstrap[-1]),
            "next_value": last_next_value,
        },
        "terminal_count": int(sum(bool(item.terminated) for item in transitions)),
        "truncated_count": int(sum(bool(item.truncated) for item in transitions)),
        "gae_recursion_boundary_count": gae_recursion_boundary_count,
        "nonterminal_bootstrap_count": nonterminal_bootstrap_count,
        "bootstrap_time_limits": bootstrap_time_limits,
        "gae_recurrence_matches_training_records": bool(gae_advantage_match and return_match),
        "gae_does_not_cross_done_boundary": bool(gae_advantage_match and return_match),
        "per_record_audit": record_audit,
        "rollout_empty_after_update": len(trainer.rollout) == 0,
        "normalizer_hash_before": node_hash_before,
        "normalizer_hash_after": node_hash_after,
        "central_normalizer_hash_before": central_hash_before,
        "central_normalizer_hash_after": central_hash_after,
        "normalizers_unchanged": node_hash_before == node_hash_after and central_hash_before == central_hash_after,
        "collector_parameters_unchanged_before_update": all(torch.equal(preupdate_parameters[name], before[name]) for name in before),
        "parameter_deltas": grouped_parameter_deltas(before, trainer.policy),
        "update_stats": dict(update_stats),
        "optimizer_steps": int(audit.steps),
        "optimizer_state_steps": optimizer_steps,
        "gradients_per_optimizer_step": copy.deepcopy(audit.gradients),
        "all_weights_finite": all(bool(torch.isfinite(value).all().item()) for value in trainer.policy.state_dict().values()),
    }


def _bool(row: Mapping[str, Any], *names: str) -> bool:
    for name in names:
        if name in row:
            return bool(row[name])
    return False


def _number(row: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] is not None:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return float(default)


def _episode_metrics(rows: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, Any]:
    rows = list(rows)
    count = len(rows)
    actual_steps = sum(max(0.0, _number(row, "actual_steps", "steps")) for row in rows)
    stable = sum(int(_bool(row, "task_success", "success_stable", "strict_success")) for row in rows)
    collision = sum(int(_bool(row, "collision", "collision_episode")) for row in rows)
    contact_steps = sum(max(0.0, _number(row, "physical_contact_steps")) for row in rows)
    timeout = sum(int(_bool(row, "timeout_failure", "timeout")) for row in rows)
    raw_time_limits = sum(int(_bool(row, "time_limit_reached_raw")) for row in rows)
    boundary_attempt = sum(max(0.0, _number(row, "boundary_attempt_steps")) for row in rows)
    boundary_projection = sum(max(0.0, _number(row, "boundary_projection_steps")) for row in rows)
    safety_projection = sum(max(0.0, _number(row, "safety_projection_steps")) for row in rows)
    raw_danger = sum(max(0.0, _number(row, "raw_action_danger_steps")) for row in rows)
    physical_denominator = sum(max(0.0, _number(row, "raw_action_physical_denominator", default=_number(row, "steps"))) for row in rows)
    goals = [_number(row, "final_actual_goal_distance_m", default=float("nan")) for row in rows]
    goals = [value for value in goals if np.isfinite(value)]
    # The scenario manifest calls the obstacle cases ``single`` and ``full``;
    # ``normal`` is the no-obstacle denominator.  Keep this definition tied to
    # the actual manifest kinds instead of inventing a physical-baseline kind.
    physical_rows = [row for row in rows if str(row.get("kind", "")).lower() in {"single", "full"}]
    physical_bypasses = sum(int(_bool(row, "physical_bypass", "physical_baseline_pass", "physical_pass")) for row in physical_rows)
    physical_stable = sum(int(_bool(row, "task_success", "success_stable", "strict_success")) for row in physical_rows)
    physical_collisions = sum(int(_bool(row, "collision", "collision_episode")) for row in physical_rows)
    physical_pass = physical_bypasses
    # A time-limit truncation at the normal 120-step horizon is a complete
    # episode.  Only the explicit rollout-window cut marks a partial record.
    complete = sum(int(not _bool(row, "rollout_window_cut")) for row in rows)
    cut = count - complete
    final_speeds = [_number(row, "final_actual_speed_mps", default=float("nan")) for row in rows]
    final_speeds = [value for value in final_speeds if np.isfinite(value)]
    danger_rate = float(raw_danger / physical_denominator) if physical_denominator else None
    bypass_rate = float(physical_bypasses / len(physical_rows)) if physical_rows else None
    return {
        "kind": kind,
        "episode_count": count,
        "complete_episode_count": complete,
        "rollout_window_cut_count": cut,
        "actual_steps": int(actual_steps) if actual_steps.is_integer() else float(actual_steps),
        "task_success_count": stable,
        "success_stable_count": stable,
        "success_stable_rate": float(stable / count) if count else 0.0,
        "collision_episode_count": collision,
        "collision_episode_rate": float(collision / count) if count else 0.0,
        "physical_contact_steps": int(contact_steps) if contact_steps.is_integer() else float(contact_steps),
        "physical_contact_step_fraction": float(contact_steps / actual_steps) if actual_steps else 0.0,
        "timeout_failure_count": timeout,
        "timeout_failure_rate": float(timeout / count) if count else 0.0,
        "time_limit_reached_raw_count": raw_time_limits,
        "time_limit_reached_raw_rate": float(raw_time_limits / count) if count else 0.0,
        "boundary_attempt_steps": int(boundary_attempt) if boundary_attempt.is_integer() else float(boundary_attempt),
        "boundary_projection_steps": int(boundary_projection) if boundary_projection.is_integer() else float(boundary_projection),
        "safety_projection_steps": int(safety_projection) if safety_projection.is_integer() else float(safety_projection),
        "boundary_attempt_fraction": float(boundary_attempt / actual_steps) if actual_steps else 0.0,
        "boundary_projection_fraction": float(boundary_projection / actual_steps) if actual_steps else 0.0,
        "safety_projection_fraction": float(safety_projection / actual_steps) if actual_steps else 0.0,
        "raw_action_danger_steps": int(raw_danger) if raw_danger.is_integer() else float(raw_danger),
        "raw_action_physical_denominator": int(physical_denominator) if physical_denominator.is_integer() else float(physical_denominator),
        "raw_action_danger_fraction": danger_rate,
        "raw_action_danger_rate": danger_rate,
        "median_final_goal_distance_m": float(np.median(goals)) if goals else None,
        "median_final_actual_speed_mps": float(np.median(final_speeds)) if final_speeds else None,
        "physical_baseline_episode_count": len(physical_rows),
        "physical_baseline_pass_count": physical_pass,
        "physical_baseline_pass_rate": bypass_rate,
        "physical_bypass_episodes": len(physical_rows),
        "physical_bypasses": physical_bypasses,
        "physical_bypass_rate": bypass_rate,
        "physical_baseline_pass": bool(physical_rows and physical_pass == len(physical_rows) and physical_collisions == 0 and physical_stable == len(physical_rows)),
    }


def summarize_foundation_episodes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize normal and physical foundation episodes without selection logic."""

    rows_list = [dict(row) for row in rows]
    kinds = sorted({str(row.get("kind", row.get("arm", "unknown"))) for row in rows_list})
    result = {kind: _episode_metrics([row for row in rows_list if str(row.get("kind", row.get("arm", "unknown"))) == kind], kind=kind) for kind in kinds}
    result["all"] = _episode_metrics(rows_list, kind="all")
    result["train_fraction_includes_partial_episodes"] = True
    return result


def normal_release_gate(train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the numeric release checks; root owns external mechanics gates."""

    train = train_metrics.get("all", train_metrics) if isinstance(train_metrics.get("all", train_metrics), Mapping) else train_metrics
    val = val_metrics.get("all", val_metrics) if isinstance(val_metrics.get("all", val_metrics), Mapping) else val_metrics
    train_steps = int(float(train.get("actual_steps", 0)))
    val_steps = int(float(val.get("actual_steps", 0)))
    train_boundary_limit = train_steps // 100
    val_boundary_limit = val_steps // 100
    checks = {
        "train_success_at_least_4_of_6": int(train.get("success_stable_count", 0)) >= 4 and int(train.get("episode_count", 0)) == 6,
        "validation_success_at_least_5_of_8": int(val.get("success_stable_count", 0)) >= 5 and int(val.get("episode_count", 0)) == 8,
        "train_zero_collisions_and_contacts": int(train.get("collision_episode_count", 0)) == 0 and float(train.get("physical_contact_steps", 0)) == 0.0,
        "validation_zero_collisions_and_contacts": int(val.get("collision_episode_count", 0)) == 0 and float(val.get("physical_contact_steps", 0)) == 0.0,
        "train_boundary_projection_limit": float(train.get("boundary_attempt_steps", 0)) <= train_boundary_limit and float(train.get("boundary_projection_steps", 0)) <= train_boundary_limit,
        "validation_boundary_projection_limit": float(val.get("boundary_attempt_steps", 0)) <= val_boundary_limit and float(val.get("boundary_projection_steps", 0)) <= val_boundary_limit,
        "validation_median_goal_distance_at_most_2m": val.get("median_final_goal_distance_m", None) is not None and float(val["median_final_goal_distance_m"]) <= 2.0,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "train_boundary_limit_steps": train_boundary_limit,
        "validation_boundary_limit_steps": val_boundary_limit,
        "external_mechanics_gate_required": True,
        "validation_selection_performed": False,
    }


__all__ = [
    "EXPECTED_A_SHA256",
    "EXPECTED_CENTRAL_FEATURE_DIM",
    "EXPECTED_DIRECTION_FEATURES",
    "EXPECTED_DIRECTION_INDICES",
    "EXPECTED_NODE_FEATURE_DIM",
    "FEATURE_INDEX_MAP",
    "PPOUpdateAudit",
    "SHARED_STATISTICS_DEFAULT",
    "SOURCE_A_DEFAULT",
    "capture_policy_distribution",
    "cosine_alignment_utility",
    "cosine_goal_utility",
    "cosine_utility",
    "grouped_parameter_deltas",
    "initialize_a_foundation",
    "load_shared_a_statistics",
    "normal_release_gate",
    "parameter_snapshot",
    "summarize_foundation_episodes",
    "update_with_audit",
]
