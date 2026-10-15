"""Normalization-transfer compatibility and output-head licensing probes.

This module is deliberately an offline/reset-only layer.  It assembles fresh
trainers, copies the approved A source through :mod:`single_uav_ppo_a_support`,
and records policy diagnostics without calling ``env.step`` or ``update``.
The public functions are small enough to be used by the parent experiment
runner and by pure unit tests.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from ..analysis.single_uav_ppo_a_norm_history import single_node_observation
from ..learning.mappo import MAPPOConfig, MAPPOTrainer
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from .single_uav_ppo_a_environment import create_foundation_environment
from .single_uav_ppo_a_support import (
    capture_policy_distribution,
    initialize_a_foundation,
    parameter_snapshot,
)
from .single_uav_ppo_baseline import (
    create_single_uav_ppo_trainer,
    load_single_uav_ppo_config,
)


EXPECTED_NODE_FEATURE_DIM = 42
EXPECTED_CENTRAL_FEATURE_DIM = 174
EXPECTED_EDGE_FEATURE_DIM = 11
EXPECTED_SOURCE_ACTOR_COUNT = 23
EXPECTED_SOURCE_OTHER_COUNT = 18
EXPECTED_R_PRESERVED_COUNT = 21
HEAD_RESET_NAMES = ("delta_mean_head.weight", "delta_mean_head.bias")
SOURCE_P_CONDITIONS = {"C", "N"}
CONDITIONS = {"C", "N", "G", "R"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_hashes(policy_or_state: Any) -> dict[str, str]:
    if hasattr(policy_or_state, "named_parameters"):
        return {name: _tensor_hash(value) for name, value in policy_or_state.named_parameters()}
    state = _extract_state(policy_or_state)
    return {str(name): _tensor_hash(value) for name, value in state.items() if torch.is_tensor(value)}


def _extract_state(value: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(value, Mapping) and isinstance(value.get("policy_state_dict"), Mapping):
        value = value["policy_state_dict"]
    if not isinstance(value, Mapping):
        raise TypeError("state must be a policy state mapping")
    state = {str(name): tensor for name, tensor in value.items() if torch.is_tensor(tensor)}
    if not state:
        raise ValueError("state mapping contains no tensors")
    return state


def _stats_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise ValueError("statistics artifact must be a JSON object")
        return payload
    return None


def _coerce_statistics(value: Any) -> tuple[ObservationNormalizer, FeatureNormalizer, dict[str, Any], dict[str, Any]]:
    """Accept support tuples, normalizer objects, or persisted JSON artifacts."""

    if isinstance(value, (tuple, list)) and len(value) >= 2:
        observation, central = value[0], value[1]
        potential = copy.deepcopy(dict(value[2])) if len(value) >= 3 and isinstance(value[2], Mapping) else {}
        provenance = copy.deepcopy(dict(value[3])) if len(value) >= 4 and isinstance(value[3], Mapping) else {}
        if not hasattr(observation, "transform") or not hasattr(central, "transform"):
            raise TypeError("statistics tuple must begin with observation and central normalizers")
        return observation, central, potential, provenance

    payload = _stats_payload(value)
    if payload is None:
        raise TypeError("statistics must be a support tuple, mapping, or JSON path")
    observation_state = payload.get("observation_normalizer", payload.get("observation_statistics"))
    central_state = payload.get("central_normalizer", payload.get("central_statistics"))
    # Small fixtures sometimes expose the node state directly.  Normalize
    # that shape into the persisted ObservationNormalizer contract while
    # keeping the production artifact schema unchanged.
    if observation_state is None and isinstance(payload.get("node_features"), Mapping):
        observation_state = {
            "eps": payload.get("eps", 1e-6),
            "clip": payload.get("clip", 10.0),
            "normalizers": {"node_features": payload["node_features"]},
        }
    if central_state is None and isinstance(payload.get("central"), Mapping):
        central_state = payload["central"]
    if isinstance(observation_state, Mapping) and "state_dict" in observation_state:
        observation_state = observation_state["state_dict"]
    if isinstance(central_state, Mapping) and "state_dict" in central_state:
        central_state = central_state["state_dict"]
    if not isinstance(observation_state, Mapping) or not isinstance(central_state, Mapping):
        raise ValueError("statistics must contain observation_normalizer and central_normalizer")
    observation = ObservationNormalizer.from_state_dict(observation_state)
    central = FeatureNormalizer.from_state_dict(central_state)
    potential = payload.get("potential_statistics", payload.get("potential", {}))
    if not isinstance(potential, Mapping):
        potential = {}
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    provenance = dict(provenance)
    provenance.setdefault("fit_split", payload.get("fit_split", "calibration_train_only"))
    provenance.setdefault("artifact_path", str(value) if isinstance(value, (str, Path)) else "")
    return observation, central, copy.deepcopy(dict(potential)), copy.deepcopy(provenance)


def _statistics_digest(value: Any) -> str:
    payload = _stats_payload(value)
    if payload is not None:
        return _canonical_hash(payload)
    observation, central, potential, provenance = _coerce_statistics(value)
    return _canonical_hash({
        "observation_normalizer": observation.state_dict(),
        "central_normalizer": central.state_dict(),
        "potential_statistics": potential,
        "provenance": provenance,
    })


def _node_arrays(normalizer: Any) -> tuple[np.ndarray, np.ndarray, float | None]:
    state = normalizer.state_dict()
    if isinstance(state, Mapping) and "normalizers" in state:
        state = state.get("normalizers", {}).get("node_features", {})
    if not isinstance(state, Mapping):
        raise ValueError("node normalizer state is not inspectable")
    mean = np.asarray(state.get("mean"), dtype=np.float64).reshape(-1)
    scale = np.asarray(state.get("scale"), dtype=np.float64).reshape(-1)
    if mean.size != EXPECTED_NODE_FEATURE_DIM or scale.size != EXPECTED_NODE_FEATURE_DIM:
        raise ValueError("node normalizer must have 42 features")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("node normalizer state is non-finite or non-positive")
    clip = state.get("clip", 10.0)
    clip_value = None if clip is None else float(clip)
    return mean, scale, clip_value


def _normalizer_transform_node(normalizer: Any, raw: np.ndarray) -> np.ndarray:
    local = single_node_observation(np.asarray(raw, dtype=np.float32).reshape(42))
    transformed = normalizer.transform(local)
    return np.asarray(transformed["node_features"], dtype=np.float32).reshape(42)


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _assert_fresh_trainer(trainer: Any) -> None:
    optimizer = getattr(trainer, "optimizer", None)
    rollout = getattr(trainer, "rollout", None)
    if optimizer is None or len(getattr(optimizer, "state", {})) != 0:
        raise RuntimeError("condition initialization requires a fresh Adam optimizer")
    if rollout is None or len(rollout) != 0:
        raise RuntimeError("condition initialization requires an empty rollout")


def initialize_condition(
    trainer: Any,
    source_path: str | Path,
    condition: str,
    *,
    old_v1_initial_path: str | Path | None = None,
) -> dict[str, Any]:
    """Initialize one normalization-transfer condition from the A source.

    C and N inherit all 23 actor tensors, G keeps a fresh five-tensor first
    GAT and inherits the other 18 actor tensors, and R inherits the complete
    actor before zeroing only the two motion-mean tensors.
    """

    condition = str(condition).upper()
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}")
    _assert_fresh_trainer(trainer)
    arm = "G" if condition == "G" else "P"
    report = initialize_a_foundation(
        trainer,
        source_path,
        arm,
        old_v1_initial_path=old_v1_initial_path if condition == "G" else None,
    )
    source_names = set(report.get("loaded_parameter_names", []))
    before_reset = parameter_snapshot(trainer.policy)
    reset_names: list[str] = []
    if condition == "R":
        for name in HEAD_RESET_NAMES:
            if name not in dict(trainer.policy.named_parameters()):
                raise ValueError(f"R condition cannot find {name}")
            with torch.no_grad():
                dict(trainer.policy.named_parameters())[name].zero_()
            reset_names.append(name)
        source_names.difference_update(reset_names)
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(True)
    trainer.policy.train()
    _assert_fresh_trainer(trainer)
    after = parameter_snapshot(trainer.policy)
    actual_hashes = {name: _tensor_hash(value) for name, value in after.items()}
    source_hashes = {name: _tensor_hash(value) for name, value in before_reset.items() if name in source_names}
    expected_count = {
        "C": EXPECTED_SOURCE_ACTOR_COUNT,
        "N": EXPECTED_SOURCE_ACTOR_COUNT,
        "G": EXPECTED_SOURCE_OTHER_COUNT,
        "R": EXPECTED_R_PRESERVED_COUNT,
    }[condition]
    if len(source_names) != expected_count:
        raise RuntimeError(f"{condition} preserved {len(source_names)} source tensors, expected {expected_count}")
    return {
        "api": "initialize_condition",
        "condition": condition,
        "source": copy.deepcopy(report.get("source", {})),
        "source_optimizer_ignored": True,
        "source_optimizer_state_entries": int(report.get("source_optimizer_state_entries", 0)),
        "target_optimizer_state_entries": len(getattr(trainer.optimizer, "state", {})),
        "rollout_size": len(getattr(trainer, "rollout", [])),
        "actual_source_preserved_tensors": len(source_names),
        "source_preserved_parameter_names": sorted(source_names),
        "resetparameter_names": reset_names,
        "actualparameterhashes_after": actual_hashes,
        "source_preserved_hashes_after": source_hashes,
        "parameter_hashes_before_reset": {name: _tensor_hash(value) for name, value in before_reset.items()},
        "parameter_object_ids_preserved": bool(report.get("parameter_object_ids_preserved", True)),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in trainer.policy.parameters()),
        "policy_training_mode": bool(trainer.policy.training),
        "normalizer_refit": False,
        "trainer_report": report,
    }


def _scenario_payload(entry: Mapping[str, Any]) -> MutableMapping[str, Any]:
    if isinstance(entry.get("scenario"), Mapping):
        return copy.deepcopy(dict(entry["scenario"]))
    return copy.deepcopy(dict(entry))


def _zero_goal_velocity_scene(entry: Mapping[str, Any]) -> dict[str, Any]:
    scene = _scenario_payload(entry)
    positions = scene.get("initial_positions")
    if not isinstance(positions, Mapping) or not positions:
        raise ValueError("licensing scenario requires initial_positions")
    scene["goals"] = {str(uid): [float(x) for x in np.asarray(position, dtype=float).reshape(3)] for uid, position in positions.items()}
    scene["initial_velocities"] = {str(uid): [0.0, 0.0, 0.0] for uid in positions}
    scene["foundation_task_kind"] = "normal"
    scene["obstacles"] = []
    scene["obstacle_specs"] = []
    return scene


def _normal_train_scenarios(scenarios: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("train", "normal_train"):
        candidate = scenarios.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            entries = list(candidate)
            if key == "train":
                normal_entries = [
                    entry
                    for entry in entries
                    if isinstance(entry, Mapping)
                    and str(entry.get("kind", entry.get("foundation_task_kind", "normal"))).lower() == "normal"
                ]
                if len(normal_entries) >= 6:
                    return normal_entries[:6]
            if len(entries) >= 6:
                return entries[:6]
    raise ValueError("scenarios must provide at least six training normal cases")


def _base_with_potential(base: Any, potential: Mapping[str, Any]) -> Any:
    if hasattr(base, "train_potential_statistics"):
        return replace(base, train_potential_statistics=copy.deepcopy(dict(potential)))
    return base


def _prepare_license_trainer(base: Any, source_path: str | Path, stats: Any, action_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    observation, central, potential, provenance = _coerce_statistics(stats)
    runtime_base = _base_with_potential(base, potential)
    environment = create_foundation_environment(runtime_base)
    trainer = create_single_uav_ppo_trainer(
        environment,
        runtime_base,
        normalizer=observation,
        central_normalizer=central,
        normalizer_provenance=provenance,
        action_log_path=action_path,
    )
    initialization = initialize_condition(trainer, source_path, "N")
    return environment, trainer, {"potential_statistics": potential, "provenance": provenance, "initialization": initialization}


def license_output_head(
    base: Any,
    scenarios: Mapping[str, Any],
    source_path: str | Path,
    new_statistics: Any,
    data_root: str | Path,
    log_root: str | Path,
) -> dict[str, Any]:
    """License R from six zero-goal/zero-velocity reset-only probes."""

    data_root, log_root = Path(data_root), Path(log_root)
    decision_path = data_root / "primary_decision.lock.json"
    license_path = log_root / "output_head_license.json"
    for path in (decision_path, license_path):
        if path.exists():
            raise FileExistsError(path)
    observation, central, potential, provenance = _coerce_statistics(new_statistics)
    stats_digest = _statistics_digest(new_statistics)
    action_path = log_root / "_reset_only_unused_action_log.csv"
    if action_path.exists():
        raise FileExistsError(action_path)
    runtime_base = _base_with_potential(base, potential)
    environment = create_foundation_environment(runtime_base)
    trainer = create_single_uav_ppo_trainer(
        environment,
        runtime_base,
        normalizer=observation,
        central_normalizer=central,
        normalizer_provenance=provenance,
        action_log_path=action_path,
    )
    initialization = initialize_condition(trainer, source_path, "N")
    rows: list[dict[str, Any]] = []
    probe_states: list[Mapping[str, Any]] = []
    uid = str(environment.uav_ids[0])
    for index, entry in enumerate(_normal_train_scenarios(scenarios)):
        scene = _zero_goal_velocity_scene(entry)
        observations, info = environment.reset(seed=20260000 + index, scenario_config=scene)
        raw = np.asarray(observations[uid]["node_features"], dtype=np.float64).reshape(-1, EXPECTED_NODE_FEATURE_DIM)[0]
        if not np.allclose(raw[30:33], 0.0, atol=0.0, rtol=0.0):
            raise AssertionError("licensing reset did not produce zero raw goal delta")
        if not np.allclose(raw[3:6], 0.0, atol=0.0, rtol=0.0):
            raise AssertionError("licensing reset did not produce zero estimated velocity")
        with torch.no_grad():
            decision = trainer.act(observations, info.get("central_state"), deterministic=True)
            distribution = capture_policy_distribution(trainer, decision)
        normalized = np.asarray(decision.observations[uid]["node_features"], dtype=float).reshape(-1, EXPECTED_NODE_FEATURE_DIM)[0]
        bounded = np.asarray(distribution["bounded_mean"], dtype=float).reshape(3)
        rows.append({
            "index": index,
            "scenario_id": str(entry.get("scenario_id", entry.get("episode_id", index))) if isinstance(entry, Mapping) else str(index),
            "raw_goal_delta": raw[30:33].tolist(),
            "raw_estimated_velocity": raw[3:6].tolist(),
            "node_raw": raw.tolist(),
            "node_normalized": normalized.tolist(),
            "latent_mean": np.asarray(distribution["raw_mean"], dtype=float).reshape(3).tolist(),
            "bounded_mean": bounded.tolist(),
            "bounded_mean_norm_mps": float(np.linalg.norm(bounded)),
            "delta_component_limit_mps": float(distribution["delta_component_limit_mps"]),
            "action_logprob_mode": str(distribution.get("action_logprob_mode", "sampled")),
        })
        probe_states.append({"observations": copy.deepcopy(observations), "central_state": copy.deepcopy(info.get("central_state"))})
    norms = [float(row["bounded_mean_norm_mps"]) for row in rows]
    licensed = bool(len(rows) == 6 and all(value > 0.05 for value in norms))
    primary = "R" if licensed else "N"
    post_reset_sanity: list[dict[str, Any]] = []
    reset_changed: list[str] = []
    if licensed:
        before = parameter_snapshot(trainer.policy)
        with torch.no_grad():
            trainer.policy.delta_mean_head.weight.zero_()
            trainer.policy.delta_mean_head.bias.zero_()
        after = parameter_snapshot(trainer.policy)
        reset_changed = [name for name in after if not torch.equal(before[name], after[name])]
        if set(reset_changed) != set(HEAD_RESET_NAMES):
            raise RuntimeError(f"R output-head reset changed unexpected tensors: {reset_changed}")
        for index, state in enumerate(probe_states):
            with torch.no_grad():
                decision = trainer.act(state["observations"], state["central_state"], deterministic=True)
                distribution = capture_policy_distribution(trainer, decision)
            mean = np.asarray(distribution["raw_mean"], dtype=float).reshape(3)
            post_reset_sanity.append({"index": index, "raw_mean": mean.tolist(), "max_abs": float(np.max(np.abs(mean)))})
        if any(item["max_abs"] >= 1e-9 for item in post_reset_sanity):
            raise RuntimeError("R reset sanity check failed")
        for name, value in after.items():
            if name not in HEAD_RESET_NAMES and not torch.equal(value, before[name]):
                raise RuntimeError(f"R reset modified non-head tensor {name}")
    decision_payload = {
        "api": "license_output_head",
        "licensed": licensed,
        "primary": primary,
        "source_control": "N_unreset",
        "statistics_sha256": stats_digest,
        "probe_count": len(rows),
        "mean_physical_delta_v": "tanh(raw_gaussian_mean) * 0.6 / sqrt(3)",
        "probe_norms_mps": norms,
        "resetparameter_names": reset_changed,
        "post_reset_sanity": post_reset_sanity,
        "env_steps": 0,
        "ppo_updates": 0,
    }
    license_payload = {**decision_payload, "probe_rows": rows, "initialization": initialization, "statistics_provenance": provenance}
    _write_exclusive_json(decision_path, decision_payload)
    _write_exclusive_json(license_path, license_payload)
    return {
        "licensed": licensed,
        "primary": primary,
        "probe_rows": rows,
        "env_steps": 0,
        "ppo_updates": 0,
        "post_reset_sanity": post_reset_sanity,
        "resetparameter_names": reset_changed,
        "statistics_sha256": stats_digest,
        "output_paths": {"decision": str(decision_path), "license": str(license_path)},
    }


def _load_bank(bank_path: str | Path) -> dict[str, np.ndarray]:
    path = Path(bank_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {str(key): np.asarray(archive[key]) for key in archive.files}
    def pick(*names: str) -> np.ndarray | None:
        for name in names:
            if name in arrays:
                return arrays[name]
        return None
    nodes = pick("nodes", "node_features", "node", "raw_nodes")
    if nodes is None:
        raise ValueError("calibration bank requires nodes/node_features")
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, EXPECTED_NODE_FEATURE_DIM)
    central = pick("central", "central_vectors", "central_features")
    result = {"nodes": nodes}
    if central is not None:
        result["central"] = np.asarray(central, dtype=np.float64).reshape(-1, EXPECTED_CENTRAL_FEATURE_DIM)
    for out, names in {
        "scenario_id": ("scenario_id", "scenario_ids", "scenarios"),
        "episode_step": ("episode_step", "episode_steps", "steps", "step"),
    }.items():
        array = pick(*names)
        if array is not None:
            result[out] = np.asarray(array).reshape(-1)
    count = len(nodes)
    if "central" in result and len(result["central"]) != count:
        raise ValueError("central bank length does not match nodes")
    for name in ("scenario_id", "episode_step"):
        if name in result and len(result[name]) != count:
            raise ValueError(f"{name} bank length does not match nodes")
    return result


def _offline_trainer(stats: Any, *, source_path: str | Path, condition: str) -> tuple[MAPPOTrainer, dict[str, Any]]:
    observation, central, potential, provenance = _coerce_statistics(stats)
    config_path = Path("data/configs/single_uav_ppo_baseline_v1.json")
    base = load_single_uav_ppo_config(config_path)
    map_config = MAPPOConfig(
        uav_count=1,
        node_feature_dim=EXPECTED_NODE_FEATURE_DIM,
        edge_feature_dim=EXPECTED_EDGE_FEATURE_DIM,
        gat_hidden_dim=base.gat_hidden_dim,
        gat_output_dim=base.gat_output_dim,
        gat_heads=base.gat_heads,
        gat_layers=base.gat_layers,
        actor_hidden_dim=base.actor_hidden_dim,
        critic_hidden_dim=base.critic_hidden_dim,
        max_obstacles=base.max_obstacles,
        max_delta_v=base.max_delta_v,
        gamma=base.gamma,
        gae_lambda=base.gae_lambda,
        clip_epsilon=base.clip_epsilon,
        policy_epochs=base.policy_epochs,
        minibatch_size=base.minibatch_size,
        learning_rate=base.learning_rate,
        entropy_coef=base.entropy_coef,
        initial_delta_std=base.initial_delta_std,
        max_grad_norm=base.max_grad_norm,
        ppo_action_logprob_mode="sampled",
        encoder_type=base.encoder_type,
        device=base.device,
    )
    trainer = MAPPOTrainer(map_config, uav_ids=("uav_00",), normalizer=observation, central_normalizer=central, seed=base.seed)
    initialization = initialize_condition(trainer, source_path, condition)
    trainer.policy.eval()
    return trainer, {"potential_statistics": potential, "provenance": provenance, "initialization": initialization}


def _latent_mean(trainer: MAPPOTrainer, normalized: np.ndarray) -> np.ndarray:
    local = single_node_observation(normalized)
    with torch.no_grad():
        distribution = trainer.policy._actor_distributions(local)[2]
        return distribution.base_dist.mean.detach().cpu().numpy().reshape(3).astype(np.float64)


def _runtime_graph_probe(trainer: MAPPOTrainer, normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    layer = trainer.policy.gat.layers[0]
    node = torch.as_tensor(normalized.reshape(1, 42), dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.empty((0, 11), dtype=torch.float32)
    with torch.no_grad():
        output = layer(node, edge_index, edge_features).detach().cpu().numpy().reshape(-1).astype(np.float64)
    node_weight = layer.node_projection.weight.detach().cpu().double().numpy()
    residual_weight = layer.residual.weight.detach().cpu().double().numpy()
    bias = layer.bias.detach().cpu().double().numpy()
    preactivation = (node_weight + residual_weight) @ normalized.astype(np.float64) + bias
    expected = np.where(preactivation >= 0.0, preactivation, np.expm1(preactivation))
    return preactivation, expected, float(np.max(np.abs(output - expected)))


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _goal_direction(action: np.ndarray, goal: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(goal))
    if norm <= 1e-12:
        return None
    action_norm = float(np.linalg.norm(action))
    if action_norm <= 1e-12:
        return 0.0
    return float(np.dot(action, goal) / (action_norm * norm))


def compatibility_probe(
    bank_path: str | Path,
    source_path: str | Path,
    old_stats: Any,
    new_stats: Any,
    data_root: str | Path,
    log_root: str | Path,
) -> dict[str, Any]:
    """Compare old/full-stat policy forwards on one unchanged calibration bank."""

    data_root, log_root = Path(data_root), Path(log_root)
    features_path = log_root / "compat_features.csv"
    actions_path = log_root / "compat_actions.csv"
    for path in (features_path, actions_path):
        if path.exists():
            raise FileExistsError(path)
    bank = _load_bank(bank_path)
    old_observation, _old_central, _old_potential, old_provenance = _coerce_statistics(old_stats)
    new_observation, _new_central, _new_potential, new_provenance = _coerce_statistics(new_stats)
    old_mean, old_scale, old_clip = _node_arrays(old_observation)
    new_mean, new_scale, new_clip = _node_arrays(new_observation)
    raw_nodes = bank["nodes"]
    old_unclipped = (raw_nodes - old_mean) / old_scale
    new_unclipped = (raw_nodes - new_mean) / new_scale
    old_clipped = old_unclipped if old_clip is None else np.clip(old_unclipped, -old_clip, old_clip)
    new_clipped = new_unclipped if new_clip is None else np.clip(new_unclipped, -new_clip, new_clip)
    old_clip_mask = np.abs(old_unclipped) > old_clip if old_clip is not None else np.zeros_like(old_unclipped, dtype=bool)
    new_clip_mask = np.abs(new_unclipped) > new_clip if new_clip is not None else np.zeros_like(new_unclipped, dtype=bool)
    eligible = np.isfinite(old_unclipped).all(axis=1) & np.isfinite(new_unclipped).all(axis=1) & ~old_clip_mask.any(axis=1) & ~new_clip_mask.any(axis=1)
    old_trainer, old_meta = _offline_trainer(old_stats, source_path=source_path, condition="C")
    new_trainer, new_meta = _offline_trainer(new_stats, source_path=source_path, condition="N")
    action_rows: list[dict[str, Any]] = []
    old_preacts: list[np.ndarray] = []
    new_preacts: list[np.ndarray] = []
    rebase_errors: list[float] = []
    graph_errors: list[float] = []
    old_latents: list[np.ndarray] = []
    new_latents: list[np.ndarray] = []
    plus_minus_rows: list[dict[str, Any]] = []
    old_limit = float(old_trainer.policy._delta_component_limit())
    new_limit = float(new_trainer.policy._delta_component_limit())
    for index, (raw, old_norm, new_norm) in enumerate(zip(raw_nodes, old_clipped, new_clipped)):
        old_latent = _latent_mean(old_trainer, old_norm.astype(np.float32))
        new_latent = _latent_mean(new_trainer, new_norm.astype(np.float32))
        old_action = np.tanh(old_latent) * old_limit
        new_action = np.tanh(new_latent) * new_limit
        old_preact, _old_expected, old_graph_error = _runtime_graph_probe(old_trainer, old_norm.astype(np.float32))
        new_preact, _new_expected, new_graph_error = _runtime_graph_probe(new_trainer, new_norm.astype(np.float32))
        old_preacts.append(old_preact)
        new_preacts.append(new_preact)
        graph_errors.extend([old_graph_error, new_graph_error])
        if bool(eligible[index]):
            layer = old_trainer.policy.gat.layers[0]
            w = (layer.node_projection.weight.detach().cpu().double().numpy() + layer.residual.weight.detach().cpu().double().numpy())
            b = layer.bias.detach().cpu().double().numpy()
            rebased_w = w * (new_scale / old_scale)[None, :]
            rebased_b = b + w @ ((new_mean - old_mean) / old_scale)
            rebase_pred = rebased_w @ new_norm.astype(np.float64) + rebased_b
            rebase_errors.append(float(np.max(np.abs(rebase_pred - old_preact))))
        scenario = str(bank.get("scenario_id", np.arange(len(raw_nodes)))[index])
        step = int(bank.get("episode_step", np.arange(len(raw_nodes)))[index])
        action_rows.append({
            "index": index,
            "scenario_id": scenario,
            "episode_step": step,
            "old_latent_mean_x": old_latent[0], "old_latent_mean_y": old_latent[1], "old_latent_mean_z": old_latent[2],
            "new_latent_mean_x": new_latent[0], "new_latent_mean_y": new_latent[1], "new_latent_mean_z": new_latent[2],
            "old_squashed_mean_x": old_action[0], "old_squashed_mean_y": old_action[1], "old_squashed_mean_z": old_action[2],
            "new_squashed_mean_x": new_action[0], "new_squashed_mean_y": new_action[1], "new_squashed_mean_z": new_action[2],
            "action_l2": float(np.linalg.norm(new_action - old_action)),
            "eligible_unclipped": bool(eligible[index]),
            "old_goal_direction_cosine": _goal_direction(old_action, raw[30:33]),
            "new_goal_direction_cosine": _goal_direction(new_action, raw[30:33]),
        })
        # The signed +/-5 probe lives entirely in the training bank.  A zero
        # raw goal is retained as a canonical zero mask and has no cosine.
        plus_old = old_norm.copy(); minus_old = old_norm.copy()
        plus_new = new_norm.copy(); minus_new = new_norm.copy()
        plus_old[30:33] += 5.0; minus_old[30:33] -= 5.0
        plus_new[30:33] += 5.0; minus_new[30:33] -= 5.0
        old_plus_action = np.tanh(_latent_mean(old_trainer, plus_old.astype(np.float32))) * old_limit
        old_minus_action = np.tanh(_latent_mean(old_trainer, minus_old.astype(np.float32))) * old_limit
        new_plus_action = np.tanh(_latent_mean(new_trainer, plus_new.astype(np.float32))) * new_limit
        new_minus_action = np.tanh(_latent_mean(new_trainer, minus_new.astype(np.float32))) * new_limit
        plus_minus_rows.append({
            "index": index,
            "scenario_id": scenario,
            "episode_step": step,
            "goal_norm_zero_mask": int(np.linalg.norm(raw[30:33]) <= 1e-12),
            "old_plus_goal_direction_cosine": _goal_direction(old_plus_action, raw[30:33]),
            "old_minus_goal_direction_cosine": _goal_direction(old_minus_action, raw[30:33]),
            "new_plus_goal_direction_cosine": _goal_direction(new_plus_action, raw[30:33]),
            "new_minus_goal_direction_cosine": _goal_direction(new_minus_action, raw[30:33]),
        })
        old_latents.append(old_latent); new_latents.append(new_latent)
    old_preact_matrix = np.asarray(old_preacts)
    new_preact_matrix = np.asarray(new_preacts)
    feature_rows: list[dict[str, Any]] = []
    action_mean_difference = np.mean(np.abs(np.asarray(new_latents) - np.asarray(old_latents)), axis=0)
    # Attribute first-layer preactivation differences by input feature.  This
    # retains the requested 42-row feature table without pretending the
    # nonlinear downstream actor has a unique per-feature decomposition.
    first_layer = old_trainer.policy.gat.layers[0]
    first_w = (
        first_layer.node_projection.weight.detach().cpu().double().numpy()
        + first_layer.residual.weight.detach().cpu().double().numpy()
    )
    feature_preact_delta = (
        new_unclipped[:, None, :] * first_w[None, :, :]
        - old_unclipped[:, None, :] * first_w[None, :, :]
    )
    for feature in range(EXPECTED_NODE_FEATURE_DIM):
        feature_rows.append({
            "feature_index": feature,
            "old_clip_fraction": float(old_clip_mask[:, feature].mean()),
            "new_clip_fraction": float(new_clip_mask[:, feature].mean()),
            "preactivation_difference_mean": float(np.mean(np.abs(feature_preact_delta[:, :, feature]))),
            "preactivation_difference_max": float(np.max(np.abs(feature_preact_delta[:, :, feature]))),
            "action_mean_difference": float(np.mean(action_mean_difference)),
        })
    _write_csv_exclusive(features_path, feature_rows, ("feature_index", "old_clip_fraction", "new_clip_fraction", "preactivation_difference_mean", "preactivation_difference_max", "action_mean_difference"))
    _write_csv_exclusive(actions_path, action_rows, (
        "index", "scenario_id", "episode_step", "old_latent_mean_x", "old_latent_mean_y", "old_latent_mean_z",
        "new_latent_mean_x", "new_latent_mean_y", "new_latent_mean_z", "old_squashed_mean_x", "old_squashed_mean_y",
        "old_squashed_mean_z", "new_squashed_mean_x", "new_squashed_mean_y", "new_squashed_mean_z", "action_l2",
        "eligible_unclipped", "old_goal_direction_cosine", "new_goal_direction_cosine"))
    return {
        "api": "compatibility_probe",
        "bank_path": str(Path(bank_path).expanduser().resolve()),
        "bank_count": len(raw_nodes),
        "old_statistics_sha256": _statistics_digest(old_stats),
        "new_statistics_sha256": _statistics_digest(new_stats),
        "old_provenance": old_provenance,
        "new_provenance": new_provenance,
        "eligible_unclipped_count": int(eligible.sum()),
        "old_clip_fraction": old_clip_mask.mean(axis=0).tolist(),
        "new_clip_fraction": new_clip_mask.mean(axis=0).tolist(),
        "max_affine_rebase_preactivation_error": float(max(rebase_errors)) if rebase_errors else None,
        "affine_rebase_mechanism_passed": bool(rebase_errors and max(rebase_errors) <= 1e-5),
        "max_verified_layer_output_diff": float(max(graph_errors)) if graph_errors else 0.0,
        "runtime_graph_equivalence_passed": bool(graph_errors and max(graph_errors) <= 1e-5),
        "action_mean_difference_l2": float(np.mean([row["action_l2"] for row in action_rows])) if action_rows else 0.0,
        "goal_perturbation_rows": plus_minus_rows,
        "env_steps": 0,
        "ppo_updates": 0,
        "no_weights_rewritten": True,
        "output_paths": {"features": str(features_path), "actions": str(actions_path)},
        "initializations": {"old": old_meta["initialization"], "new": new_meta["initialization"]},
    }


def tensor_pair_audit(initial_states: Mapping[str, Any], state_dict: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit C/N/G/R initialization deltas and shared fresh Critic tensors.

    ``state_dict`` is accepted as a compatibility alias for callers that pass
    one mapping positionally; the canonical call supplies the condition map
    as the first argument.
    """

    states = initial_states if state_dict is None else state_dict
    if not isinstance(states, Mapping):
        raise TypeError("initial_states must map condition names to state dictionaries")
    normalized = {str(condition).upper(): _extract_state(value) for condition, value in states.items()}
    missing_required = {"C", "N"} - set(normalized)
    if missing_required:
        raise ValueError(f"initial_states missing required conditions: {sorted(missing_required)}")
    hashes = {condition: _state_hashes(state) for condition, state in normalized.items()}
    names = set.intersection(*(set(state) for state in normalized.values()))
    actor_names = {name for name in names if not name.startswith("critic.")}
    critic_names = {name for name in names if name.startswith("critic.")}
    c_n_changed = sorted(name for name in actor_names if not torch.equal(normalized["C"][name], normalized["N"][name]))
    n_g_changed = (
        sorted(name for name in actor_names if not torch.equal(normalized["N"][name], normalized["G"][name]))
        if "G" in normalized
        else None
    )
    n_r_changed = (
        sorted(name for name in actor_names if not torch.equal(normalized["N"][name], normalized["R"][name]))
        if "R" in normalized
        else None
    )
    allowed_g = {name for name in actor_names if name.startswith("gat.layers.0.")}
    allowed_r = set(HEAD_RESET_NAMES)
    critic_identical = all(
        torch.equal(normalized["C"][name], normalized[condition][name])
        for name in critic_names
        for condition in normalized
    )
    passed = bool(
        not c_n_changed
        and (n_g_changed is None or (set(n_g_changed).issubset(allowed_g) and len(n_g_changed) <= len(allowed_g)))
        and (n_r_changed is None or set(n_r_changed).issubset(allowed_r))
        and critic_identical
    )
    return {
        "api": "tensor_pair_audit",
        "passed": passed,
        "actor_C_N_exact": not c_n_changed,
        "critic_all_identical": critic_identical,
        "actual_changed_names": {"C_N": c_n_changed, "N_G": n_g_changed, "N_R": n_r_changed},
        "allowed_changed_scope": {"C_N": [], "N_G": sorted(allowed_g), "N_R": sorted(allowed_r)},
        "source_actor_tensor_counts": {condition: {"C": 23, "N": 23, "G": 18, "R": 21}[condition] for condition in normalized},
        "condition_state_hashes": hashes,
    }


__all__ = [
    "compatibility_probe",
    "initialize_condition",
    "license_output_head",
    "tensor_pair_audit",
]
