"""BC-v2 and conditional DAgger for the single-UAV Oracle teacher.

This module is deliberately isolated from the MAPPO/reward path.  It reuses
the already validated Actor-constrained estimated-state Oracle, the frozen
baseline observation normalizer, and the existing physical action transform.
The only trainable parameters are the Actor mean-path parameters.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _scenario_for,
    _scale_seed,
    _write_scale_manifest,
)
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _append_jsonl,
    _seed_everything,
)
from .single_uav_actor_constrained_oracle import (
    ACTOR_CONSTRAINED_ORACLE,
    ActorConstrainedEstimatedStateOracleController,
)
from .single_uav_checkpoint_evaluation import _config_from_checkpoint, _load_frozen_trainer, _state_dict_hash
from .single_uav_oracle_behavior_cloning import (
    BC_POLICY,
    DEMO_DATASET_ID,
    _canonical_hash,
    _clear_exact_root,
    _copy_observation_for_json,
    _normal_evaluation_scenarios,
    _read_demo_jsonl,
    _run_rl_or_bc_episode,
    _run_teacher_episode,
    _write_csv,
    actor_mean_trainable_parameter_names,
    actor_physical_mean,
)
from .single_uav_rl_equivalent_oracle_audit import (
    BASELINE_RL,
    _base_step_row,
    _complete_geometry_flags,
    _episode_summary,
    _extract_rl_visible_state,
    _mean,
    _runner_geometry,
    _unit,
    _write_rows_csv,
)


OLD_BC_POLICY = "old_bc_actor"
BC_V2_POLICY = "bc_v2_actor"
DAGGER_POLICY = "dagger_actor"
POLICY_LABELS = {
    BASELINE_RL: "coef=0.25 baseline RL",
    OLD_BC_POLICY: "old Behavior-cloned Actor",
    BC_V2_POLICY: "BC-v2 Actor",
    DAGGER_POLICY: "DAgger round-1 Actor",
    ACTOR_CONSTRAINED_ORACLE: "Actor-action-constrained estimated-state oracle",
}
DISTANCE_BINS = (">8m", "8-5m", "5-3m", "3-1m", "<=1m")
_BIN_KEYS = {
    ">8m": "gt_8m",
    "8-5m": "8_5m",
    "5-3m": "5_3m",
    "3-1m": "3_1m",
    "<=1m": "le_1m",
}
BC_V2_DATASET_ID = "single_uav_behavior_cloning_v2"
DAGGER_DATASET_ID = "single_uav_dagger_round1"


def _distance_bin(distance_m: float) -> str:
    distance = float(distance_m)
    if distance > 8.0:
        return ">8m"
    if distance > 5.0:
        return "8-5m"
    if distance > 3.0:
        return "5-3m"
    if distance > 1.0:
        return "3-1m"
    return "<=1m"


def _canonical_distance_bin(value: Any) -> str:
    """Normalize display labels and stored keys to one internal bin key."""

    text = str(value)
    if text in _BIN_KEYS:
        return _BIN_KEYS[text]
    return text


def _finite_mean(values: Sequence[Any]) -> float:
    numeric = []
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            numeric.append(candidate)
    return float(np.mean(numeric)) if numeric else float("nan")


def _finite_quantile(values: Sequence[Any], quantile: float) -> float:
    numeric = []
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            numeric.append(candidate)
    return float(np.quantile(numeric, quantile)) if numeric else float("nan")


def _raw_estimated_velocity(trainer: Any, observation: Mapping[str, Any]) -> np.ndarray:
    """Recover the estimated self velocity from the stored normalized input.

    The inverse uses only the frozen normalizer state and the normalized Actor
    observation.  It is for offline diagnostics, never for policy control.
    """

    node_features = np.asarray(observation.get("node_features"), dtype=float)
    if node_features.ndim != 2 or node_features.shape[0] < 1 or node_features.shape[1] < 6:
        return np.zeros(3, dtype=float)
    node_ids = list(observation.get("node_ids", []))
    own_index = node_ids.index("uav_00") if "uav_00" in node_ids else 0
    row = node_features[own_index]
    feature_normalizer = getattr(trainer.normalizer, "normalizers", {}).get("node_features")
    if feature_normalizer is not None and feature_normalizer.fitted:
        raw = row * np.asarray(feature_normalizer.scale, dtype=float) + np.asarray(feature_normalizer.mean, dtype=float)
    else:
        raw = row
    return np.asarray(raw[3:6], dtype=float)


def _action_components(
    trainer: Any,
    sample: Mapping[str, Any],
    action: np.ndarray,
) -> dict[str, float]:
    goal_direction = _unit(np.asarray(sample.get("goal_direction", (0.0, 0.0, 0.0)), dtype=float))
    velocity = _raw_estimated_velocity(trainer, sample["observation"])
    velocity_direction = _unit(velocity)
    radial = float(np.dot(action, goal_direction))
    braking = max(0.0, -float(np.dot(action, velocity_direction)))
    tangent = action - radial * goal_direction
    return {
        "action_norm_mps": float(np.linalg.norm(action)),
        "radial_action_projection_mps": radial,
        "braking_magnitude_mps": braking,
        "tangential_action_mps": float(np.linalg.norm(tangent)),
    }


def _fit_vector(value: Any, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    output = np.zeros(int(length), dtype=np.float32)
    output[: min(len(array), int(length))] = array[: int(length)]
    return output


def _batch_actor_physical_mean(trainer: Any, observations: Sequence[Mapping[str, Any]]):
    """Batch the exact Actor mean path for the current single-UAV graph.

    The formal task has one visible self node and no effective edges.  In that
    graph, each EdgeGAT layer reduces exactly to its self-loop projection,
    residual, bias, ELU and dropout path.  Reproducing that algebra in batch
    form avoids thousands of Python-level one-sample forwards while preserving
    the existing Actor parameters and final physical action transform.
    """

    import torch
    import torch.nn.functional as F

    policy = trainer.policy
    if not observations:
        return torch.empty((0, 3), dtype=torch.float32, device=policy.device)
    if getattr(policy.config, "encoder_type", "gat") != "gat":
        return torch.stack([actor_physical_mean(policy, observation) for observation in observations])
    node_arrays = [np.asarray(observation["node_features"], dtype=np.float32) for observation in observations]
    if any(array.ndim != 2 or array.shape[0] != 1 for array in node_arrays):
        return torch.stack([actor_physical_mean(policy, observation) for observation in observations])
    node = torch.as_tensor(np.concatenate(node_arrays, axis=0), dtype=torch.float32, device=policy.device)
    embedding = node
    for layer in policy.gat.layers:
        projected = layer.node_projection(embedding).view(-1, layer.heads, layer.out_node_dim)
        aggregated = projected.reshape(projected.shape[0], layer.heads * layer.out_node_dim) if layer.concat else projected.mean(dim=1)
        embedding = layer.dropout(F.elu(aggregated + layer.residual(embedding) + layer.bias))
    edge_dim = int(policy.config.edge_feature_dim)
    zero_edges = torch.zeros((len(observations), 4 * edge_dim), dtype=torch.float32, device=policy.device)
    action_masks = torch.as_tensor(
        np.stack([_fit_vector(observation.get("action_mask", []), int(policy.config.uav_count) + 1) for observation in observations]),
        dtype=torch.float32,
        device=policy.device,
    )
    bandwidth_masks = torch.as_tensor(
        np.stack([_fit_vector(observation.get("bandwidth_mask", []), int(policy.config.uav_count)) for observation in observations]),
        dtype=torch.float32,
        device=policy.device,
    )
    blocks = [embedding, zero_edges, action_masks, bandwidth_masks]
    if bool(getattr(policy.config, "include_remaining_time", False)):
        remaining = torch.as_tensor(
            [[float(observation["remaining_time_fraction"])] for observation in observations],
            dtype=torch.float32,
            device=policy.device,
        )
        blocks.append(remaining)
    context = torch.cat(blocks, dim=1)
    hidden = policy.actor_body(context)
    mean = policy.delta_mean_head(hidden)
    return torch.tanh(mean) * float(policy._delta_component_limit())


def _offline_action_rows(trainer: Any, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    trainer.policy.eval()
    with torch.no_grad():
        observations = [sample["observation"] for sample in samples]
        predictions = _batch_actor_physical_mean(trainer, observations).detach().cpu().numpy().astype(float)
        for index, sample in enumerate(samples):
            predicted = predictions[index]
            teacher = np.asarray(sample["teacher_action_delta_v_mps"], dtype=float).reshape(3)
            predicted_parts = _action_components(trainer, sample, predicted)
            teacher_parts = _action_components(trainer, sample, teacher)
            distance = float(sample.get("estimated_goal_distance_m", sample.get("goal_distance_m", float("nan"))))
            if not math.isfinite(distance):
                distance = float(sample.get("goal_distance_m", float("nan")))
            rows.append(
                {
                    "sample_index": index,
                    "split": str(sample.get("split", "")),
                    "distance_bin": _distance_bin(distance),
                    "estimated_goal_distance_m": distance,
                    "teacher_action_norm_mps": teacher_parts["action_norm_mps"],
                    "predicted_action_norm_mps": predicted_parts["action_norm_mps"],
                    "predicted_teacher_action_norm_ratio": predicted_parts["action_norm_mps"] / max(teacher_parts["action_norm_mps"], 1.0e-8),
                    "teacher_radial_action_projection_mps": teacher_parts["radial_action_projection_mps"],
                    "predicted_radial_action_projection_mps": predicted_parts["radial_action_projection_mps"],
                    "radial_projection_error_mps": abs(predicted_parts["radial_action_projection_mps"] - teacher_parts["radial_action_projection_mps"]),
                    "teacher_braking_magnitude_mps": teacher_parts["braking_magnitude_mps"],
                    "predicted_braking_magnitude_mps": predicted_parts["braking_magnitude_mps"],
                    "braking_projection_error_mps": abs(predicted_parts["braking_magnitude_mps"] - teacher_parts["braking_magnitude_mps"]),
                    "teacher_tangential_action_mps": teacher_parts["tangential_action_mps"],
                    "predicted_tangential_action_mps": predicted_parts["tangential_action_mps"],
                    "tangential_action_error_mps": abs(predicted_parts["tangential_action_mps"] - teacher_parts["tangential_action_mps"]),
                    "action_mse": float(np.mean((predicted - teacher) ** 2)),
                }
            )
    return rows


def _offline_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "mean_predicted_action_norm_mps": _finite_mean([r.get("predicted_action_norm_mps") for r in rows]),
        "mean_teacher_action_norm_mps": _finite_mean([r.get("teacher_action_norm_mps") for r in rows]),
        "mean_predicted_teacher_action_norm_ratio": _finite_mean([r.get("predicted_teacher_action_norm_ratio") for r in rows]),
        "mean_radial_projection_error_mps": _finite_mean([r.get("radial_projection_error_mps") for r in rows]),
        "mean_braking_projection_error_mps": _finite_mean([r.get("braking_projection_error_mps") for r in rows]),
        "mean_tangential_action_error_mps": _finite_mean([r.get("tangential_action_error_mps") for r in rows]),
    }


def _offline_distance_summary(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label in DISTANCE_BINS:
        selected = [row for row in rows if row.get("split") == split and row.get("distance_bin") == label]
        output.append(
            {
                "split": split,
                "distance_bin": label,
                "samples": len(selected),
                "teacher_mean_action_norm_mps": _finite_mean([r.get("teacher_action_norm_mps") for r in selected]),
                "bc_mean_action_norm_mps": _finite_mean([r.get("predicted_action_norm_mps") for r in selected]),
                "bc_teacher_action_norm_ratio": _finite_mean([r.get("predicted_teacher_action_norm_ratio") for r in selected]),
                "teacher_radial_action_projection_mps": _finite_mean([r.get("teacher_radial_action_projection_mps") for r in selected]),
                "bc_radial_action_projection_mps": _finite_mean([r.get("predicted_radial_action_projection_mps") for r in selected]),
                "teacher_braking_magnitude_mps": _finite_mean([r.get("teacher_braking_magnitude_mps") for r in selected]),
                "bc_braking_magnitude_mps": _finite_mean([r.get("predicted_braking_magnitude_mps") for r in selected]),
                "teacher_tangential_magnitude_mps": _finite_mean([r.get("teacher_tangential_action_mps") for r in selected]),
                "bc_tangential_magnitude_mps": _finite_mean([r.get("predicted_tangential_action_mps") for r in selected]),
                "radial_projection_error_mps": _finite_mean([r.get("radial_projection_error_mps") for r in selected]),
                "braking_projection_error_mps": _finite_mean([r.get("braking_projection_error_mps") for r in selected]),
                "tangential_action_error_mps": _finite_mean([r.get("tangential_action_error_mps") for r in selected]),
            }
        )
    return output


def _flatten_epoch_metrics(prefix: str, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_mse": _finite_mean([diagnostics.get("mse")]),
        f"{prefix}_predicted_action_norm_mps": diagnostics.get("mean_predicted_action_norm_mps"),
        f"{prefix}_teacher_action_norm_mps": diagnostics.get("mean_teacher_action_norm_mps"),
        f"{prefix}_predicted_teacher_action_norm_ratio": diagnostics.get("mean_predicted_teacher_action_norm_ratio"),
        f"{prefix}_radial_projection_error_mps": diagnostics.get("mean_radial_projection_error_mps"),
        f"{prefix}_braking_projection_error_mps": diagnostics.get("mean_braking_projection_error_mps"),
        f"{prefix}_tangential_action_error_mps": diagnostics.get("mean_tangential_action_error_mps"),
    }


def _plot_bc_v2_training(history: Sequence[Mapping[str, Any]], distance_rows: Sequence[Mapping[str, Any]], plots_root: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, str] = {}
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.plot([row["epoch"] for row in history], [row["train_mse"] for row in history], label="train MSE", linewidth=2)
    axis.plot([row["epoch"] for row in history], [row["validation_mse"] for row in history], label="validation MSE", linewidth=2)
    axis.set_xlabel("BC-v2 epoch")
    axis.set_ylabel("post-tanh scaled physical delta-v MSE")
    axis.set_title("BC-v2 training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    loss_path = plots_root / "bc_v2_loss_curves.png"
    figure.savefig(loss_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["loss_curves"] = str(loss_path.resolve())

    selected = [row for row in distance_rows if row.get("split") == "validation"]
    positions = np.arange(len(DISTANCE_BINS))
    width = 0.16
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, teacher_field, student_field, title in (
        (axes[0], "teacher_mean_action_norm_mps", "bc_mean_action_norm_mps", "Action norm"),
        (axes[1], "teacher_braking_magnitude_mps", "bc_braking_magnitude_mps", "Braking magnitude"),
        (axes[2], "teacher_radial_action_projection_mps", "bc_radial_action_projection_mps", "Radial action projection"),
    ):
        axis.bar(positions - width / 2, [float(row.get(teacher_field, 0.0) or 0.0) for row in selected], width=width, label="teacher", color="#0072B2")
        axis.bar(positions + width / 2, [float(row.get(student_field, 0.0) or 0.0) for row in selected], width=width, label="BC-v2", color="#CC79A7")
        axis.set_title(title)
        axis.set_xticks(positions, DISTANCE_BINS, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    distance_path = plots_root / "bc_v2_offline_action_distance_diagnostics.png"
    figure.savefig(distance_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["offline_distance"] = str(distance_path.resolve())
    return output


def _save_imitation_checkpoint(
    payload: Mapping[str, Any],
    policy_state: Mapping[str, Any],
    path: Path,
    metadata: Mapping[str, Any],
) -> None:
    import torch

    checkpoint = copy.deepcopy(dict(payload))
    checkpoint["policy_state_dict"] = copy.deepcopy(dict(policy_state))
    checkpoint["bc_training"] = _jsonable(dict(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _train_imitation(
    initial_checkpoint: str | Path,
    demonstrations_dir: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None,
    plots_dir: str | Path | None,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    stage: str,
    force: bool,
    progress: bool,
    trainable_parameter_names: Sequence[str] | None = None,
    anchor_lambda: float = 0.0,
    anchor_field: str = "baseline_actor_action_delta_v_mps",
) -> dict[str, Any]:
    if max_epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("max_epochs, patience, batch_size and learning_rate must be positive")
    if anchor_lambda < 0.0:
        raise ValueError("anchor_lambda must be non-negative")
    checkpoint = Path(initial_checkpoint)
    demo_root = Path(demonstrations_dir)
    output_root = Path(output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _clear_exact_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    _clear_exact_root(logs_root, force=force)
    _clear_exact_root(plots_root, force=force)
    train_samples = _read_demo_jsonl(demo_root / "demonstrations" / "train.jsonl")
    validation_samples = _read_demo_jsonl(demo_root / "demonstrations" / "validation.jsonl")
    if not train_samples or not validation_samples:
        raise ValueError("imitation training requires non-empty train and validation demonstrations")
    train_ids = {(row.get("scenario_hash"), row.get("episode_index")) for row in train_samples}
    validation_ids = {(row.get("scenario_hash"), row.get("episode_index")) for row in validation_samples}
    if train_ids & validation_ids:
        raise ValueError("demonstration train/validation trajectory leakage detected")

    import torch

    checkpoint_sha = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=output_root, progress=False)
    trainer, policy_hashes, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    normalizer_hash_before = _canonical_hash(trainer.normalizer.state_dict())
    critic_hash_before = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_hash_before = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    all_actor_mean_names = actor_mean_trainable_parameter_names(trainer.policy)
    if trainable_parameter_names is None:
        trainable_names = all_actor_mean_names
    else:
        requested = tuple(str(name) for name in trainable_parameter_names)
        available = set(all_actor_mean_names)
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(
                "trainable_parameter_names contains non-Actor-mean parameters: "
                + ", ".join(unknown)
            )
        requested_set = set(requested)
        for name, parameter in trainer.policy.named_parameters():
            parameter.requires_grad_(name in requested_set)
        trainable_names = tuple(
            name for name, parameter in trainer.policy.named_parameters() if parameter.requires_grad
        )
        if not trainable_names:
            raise ValueError("trainable_parameter_names selected no parameters")
    optimizer = torch.optim.Adam([parameter for parameter in trainer.policy.parameters() if parameter.requires_grad], lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_policy_state: dict[str, Any] | None = None
    last_policy_state: dict[str, Any] | None = None
    for epoch in range(int(max_epochs)):
        trainer.policy.eval()
        order = rng.permutation(len(train_samples))
        batch_losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            selected = [train_samples[int(index)] for index in order[start : start + int(batch_size)]]
            optimizer.zero_grad(set_to_none=True)
            predictions = _batch_actor_physical_mean(trainer, [sample["observation"] for sample in selected])
            targets = torch.as_tensor(
                np.stack([sample["teacher_action_delta_v_mps"] for sample in selected]),
                dtype=torch.float32,
                device=predictions.device,
            )
            imitation_loss = torch.mean((predictions - targets) ** 2)
            anchor_indices = [
                index
                for index, sample in enumerate(selected)
                if bool(sample.get("anchor_apply", False)) and anchor_field in sample
            ]
            if float(anchor_lambda) > 0.0 and anchor_indices:
                anchor_targets = torch.as_tensor(
                    np.stack([selected[index][anchor_field] for index in anchor_indices]),
                    dtype=torch.float32,
                    device=predictions.device,
                )
                anchor_predictions = predictions[anchor_indices]
                anchor_loss = torch.mean((anchor_predictions - anchor_targets) ** 2)
                batch_loss = imitation_loss + float(anchor_lambda) * anchor_loss
            else:
                batch_loss = imitation_loss
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(float(batch_loss.detach().cpu().item()))

        train_rows = _offline_action_rows(trainer, train_samples)
        validation_rows = _offline_action_rows(trainer, validation_samples)
        train_diag = _offline_summary(train_rows)
        validation_diag = _offline_summary(validation_rows)
        train_diag["mse"] = _finite_mean([row.get("action_mse") for row in train_rows])
        validation_diag["mse"] = _finite_mean([row.get("action_mse") for row in validation_rows])
        row: dict[str, Any] = {"epoch": epoch + 1, "mean_batch_loss": _finite_mean(batch_losses), "samples_train": len(train_samples), "samples_validation": len(validation_samples)}
        row.update(_flatten_epoch_metrics("train", train_diag))
        row.update(_flatten_epoch_metrics("validation", validation_diag))
        history.append(row)
        current_state = {name: parameter.detach().cpu().clone() for name, parameter in trainer.policy.state_dict().items()}
        last_policy_state = current_state
        validation_loss = float(validation_diag["mse"])
        if validation_loss < best_validation - 1.0e-10:
            best_validation = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_policy_state = copy.deepcopy(current_state)
        else:
            epochs_without_improvement += 1
        if progress:
            print(
                f"[{stage}] epoch {epoch + 1}/{max_epochs} train_mse={train_diag['mse']:.6g} val_mse={validation_diag['mse']:.6g} wait={epochs_without_improvement}/{patience}",
                flush=True,
            )
        if epochs_without_improvement >= int(patience):
            break

    if best_policy_state is None or last_policy_state is None:
        raise RuntimeError("imitation training did not produce a checkpoint")
    trainer.policy.load_state_dict(best_policy_state)
    trainer.policy.eval()
    normalizer_hash_after = _canonical_hash(trainer.normalizer.state_dict())
    critic_hash_after = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_hash_after = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
    metadata = {
        "dataset_id": BC_V2_DATASET_ID if stage == "bc_v2" else DAGGER_DATASET_ID,
        "stage": stage,
        "source_checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
        "source_policy_hash": policy_hashes["loaded_policy_hash"],
        "policy_hash_after_training": policy_hash_after,
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_selection": (
            "all_actor_mean_path"
            if trainable_parameter_names is None
            else "explicit_subset"
        ),
        "anchor_lambda": float(anchor_lambda),
        "anchor_field": str(anchor_field),
        "critic_frozen": critic_hash_before == critic_hash_after,
        "normalizer_frozen": normalizer_hash_before == normalizer_hash_after,
        "logstd_frozen": logstd_hash_before == logstd_hash_after,
        "loss_space": "post_tanh_scaled_physical_delta_v_mse",
        "pre_tanh_target_used": False,
        "ppo_update_performed": False,
        "reward_or_environment_modified": False,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "actual_epochs": len(history),
        "patience": int(patience),
        "best_validation_epoch": int(best_epoch),
        "early_stopped": len(history) < int(max_epochs),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
    }
    checkpoint_root = output_root / "scale_1" / "checkpoints"
    best_checkpoint = checkpoint_root / "best_validation.pt"
    last_checkpoint = checkpoint_root / "last_epoch.pt"
    _save_imitation_checkpoint(payload, best_policy_state, best_checkpoint, metadata)
    _save_imitation_checkpoint(payload, last_policy_state, last_checkpoint, {**metadata, "checkpoint_role": "last_epoch"})
    history_path = logs_root / "bc_v2_training_history.csv"
    _write_csv(history_path, history)
    best_rows = _offline_action_rows(trainer, train_samples + validation_samples)
    distance_rows: list[dict[str, Any]] = []
    distance_rows.extend(_offline_distance_summary(best_rows, split="train"))
    distance_rows.extend(_offline_distance_summary(best_rows, split="validation"))
    distance_path = logs_root / "offline_action_diagnostics_by_distance.csv"
    _write_csv(distance_path, distance_rows)
    plots = _plot_bc_v2_training(history, distance_rows, plots_root)
    summary = {
        "dataset_id": metadata["dataset_id"],
        "stage": stage,
        "source_checkpoint": metadata["source_checkpoint"],
        "demonstrations_dir": str(demo_root.resolve()),
        "best_validation_checkpoint": str(best_checkpoint.resolve()),
        "last_epoch_checkpoint": str(last_checkpoint.resolve()),
        "actual_epochs": len(history),
        "best_validation_epoch": int(best_epoch),
        "best_validation_mse": float(best_validation),
        "final_epoch_validation_mse": float(history[-1]["validation_mse"]),
        "early_stopped": bool(metadata["early_stopped"]),
        "history": history,
        "offline_action_diagnostics": {
            "train": _offline_summary([row for row in best_rows if row.get("split") == "train"]),
            "validation": _offline_summary([row for row in best_rows if row.get("split") == "validation"]),
            "by_distance": distance_rows,
        },
        "bc_contract": metadata,
        "files": {
            "history": str(history_path.resolve()),
            "offline_distance": str(distance_path.resolve()),
            "best_validation_checkpoint": str(best_checkpoint.resolve()),
            "last_epoch_checkpoint": str(last_checkpoint.resolve()),
            **plots,
        },
    }
    _write_json(output_root / "manifest.json", _jsonable(summary))
    _write_json(output_root / "summary.json", _jsonable(summary))
    return summary


def train_behavior_cloning_v2(
    baseline_checkpoint: str | Path,
    demonstrations_dir: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1.0e-4,
    seed: int = 2026,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Train BC-v2 from the original baseline checkpoint."""

    return _train_imitation(
        baseline_checkpoint,
        demonstrations_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        plots_dir=plots_dir,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        stage="bc_v2",
        force=force,
        progress=progress,
    )


def train_dagger_actor(
    bc_v2_checkpoint: str | Path,
    aggregated_demonstrations_dir: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1.0e-4,
    seed: int = 2026,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Retrain Actor-only on an aggregated original+DAgger dataset."""

    return _train_imitation(
        bc_v2_checkpoint,
        aggregated_demonstrations_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        plots_dir=plots_dir,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        stage="dagger_round1",
        force=force,
        progress=progress,
    )


def _role_root_name(role: str) -> str:
    return {
        BASELINE_RL: "baseline_rl",
        OLD_BC_POLICY: "old_bc",
        BC_V2_POLICY: "bc_v2",
        DAGGER_POLICY: "dagger_round1",
        ACTOR_CONSTRAINED_ORACLE: "actor_constrained_oracle",
    }[role]


def _rename_policy(result: dict[str, Any], rows: list[dict[str, Any]], role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for row in rows:
        row["policy"] = role
        row["policy_label"] = POLICY_LABELS[role]
    result = dict(result)
    result["policy"] = role
    result["policy_label"] = POLICY_LABELS[role]
    return result, rows


def _aggregate_role(role: str, summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in summaries if str(row.get("policy")) == role]
    step_rows = [row for row in rows if str(row.get("policy")) == role]
    count = len(selected)

    def count_field(field: str) -> int:
        return int(sum(int(row.get(field, 0) or 0) for row in selected))

    action_norms = [row.get("action_norm_mps") for row in step_rows]
    result: dict[str, Any] = {
        "policy": role,
        "policy_label": POLICY_LABELS[role],
        "episodes": count,
        "entered_8m_episodes": count_field("entered_8m"),
        "entered_8m_rate": count_field("entered_8m") / count if count else 0.0,
        "entered_5m_episodes": count_field("entered_5m"),
        "entered_5m_rate": count_field("entered_5m") / count if count else 0.0,
        "entered_3m_episodes": count_field("entered_3m"),
        "entered_3m_rate": count_field("entered_3m") / count if count else 0.0,
        "entered_1m_episodes": count_field("entered_1m"),
        "entered_1m_rate": count_field("entered_1m") / count if count else 0.0,
        "stable_goal_episodes": count_field("stable_goal"),
        "stable_goal_rate": count_field("stable_goal") / count if count else 0.0,
        "minimum_goal_distance_m": min((float(row.get("minimum_goal_distance_m", float("nan"))) for row in selected), default=float("nan")),
        "mean_minimum_goal_distance_m": _finite_mean([row.get("minimum_goal_distance_m") for row in selected]),
        "mean_speed_at_8m_entry_mps": _finite_mean([row.get("mean_speed_at_8m_entry_mps") for row in selected]),
        "mean_speed_at_5m_entry_mps": _finite_mean([row.get("mean_speed_at_5m_entry_mps") for row in selected]),
        "mean_speed_at_3m_entry_mps": _finite_mean([row.get("mean_speed_at_3m_entry_mps") for row in selected]),
        "mean_speed_at_1m_entry_mps": _finite_mean([row.get("mean_speed_at_1m_entry_mps") for row in selected]),
        "mean_speed_within_1m_mps": _finite_mean([row.get("mean_speed_within_1m_mps") for row in selected]),
        "mean_maximum_consecutive_goal_hold_steps": _finite_mean([row.get("maximum_consecutive_goal_hold_steps") for row in selected]),
        "maximum_consecutive_goal_hold_steps_max": max((int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in selected), default=0),
        "episodes_with_hold_steps_ge_1": count_field("hold_steps_ge_1"),
        "episodes_with_hold_steps_ge_1_rate": count_field("hold_steps_ge_1") / count if count else 0.0,
        "episodes_with_hold_steps_ge_3": count_field("hold_steps_ge_3"),
        "episodes_with_hold_steps_ge_3_rate": count_field("hold_steps_ge_3") / count if count else 0.0,
        "episodes_with_hold_steps_ge_5": count_field("hold_steps_ge_5"),
        "episodes_with_hold_steps_ge_5_rate": count_field("hold_steps_ge_5") / count if count else 0.0,
        "mean_action_norm_mps": _finite_mean(action_norms),
        "p50_action_norm_mps": _finite_quantile(action_norms, 0.50),
        "p90_action_norm_mps": _finite_quantile(action_norms, 0.90),
        "max_action_norm_mps": max((float(value) for value in action_norms if value is not None and math.isfinite(float(value))), default=float("nan")),
        "mean_radial_action_projection_mps": _finite_mean([row.get("action_projection_toward_goal_mps") for row in step_rows]),
        "mean_braking_magnitude_mps": _finite_mean([row.get("braking_projection_mps") for row in step_rows]),
        "mean_tangential_action_mps": _finite_mean([row.get("tangential_action_mps") for row in step_rows]),
        "mean_radial_velocity_mps": _finite_mean([row.get("true_radial_velocity_before_mps") for row in step_rows]),
        "negative_radial_velocity_fraction": _finite_mean([row.get("negative_radial_velocity") for row in step_rows]),
        "timeout_episodes": int(sum(int(row.get("timeout", 0) or 0) for row in selected)),
    }
    return result


def _plot_evaluation(aggregates: Mapping[str, Mapping[str, Any]], output_dir: Path, roles: Sequence[str]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(roles))
    labels = [POLICY_LABELS[role] for role in roles]
    figure, axis = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    width = 0.18
    for offset, field, label, color in (
        (-1.5 * width, "entered_8m_rate", "8 m", "#0072B2"),
        (-0.5 * width, "entered_3m_rate", "3 m", "#E69F00"),
        (0.5 * width, "entered_1m_rate", "1 m", "#CC79A7"),
        (1.5 * width, "stable_goal_rate", "stable", "#009E73"),
    ):
        axis.bar(positions + offset, [100.0 * float(aggregates[role].get(field, 0.0)) for role in roles], width=width, label=label, color=color)
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Episodes (%)")
    axis.set_ylim(0, 100)
    axis.set_title("BC-v2 / DAgger closed-loop evaluation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path = output_dir / "imitation_evaluation_entry_stable.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for axis, field, title in (
        (axes[0], "mean_action_norm_mps", "Mean action norm"),
        (axes[1], "mean_braking_magnitude_mps", "Braking magnitude"),
        (axes[2], "negative_radial_velocity_fraction", "Negative radial velocity"),
    ):
        axis.bar(positions, [float(aggregates[role].get(field, float("nan"))) for role in roles], color=["#0072B2", "#7B2CBF", "#CC79A7", "#009E73", "#D55E00"][: len(roles)])
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
    action_path = output_dir / "imitation_action_diagnostics.png"
    figure.savefig(action_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"entry_stable_plot": str(path.resolve()), "action_diagnostics_plot": str(action_path.resolve())}


def evaluate_imitation_policies(
    baseline_checkpoint: str | Path,
    old_bc_checkpoint: str | Path,
    bc_v2_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    dagger_checkpoint: str | Path | None = None,
    validation_episodes: int = 20,
    test_episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate baseline, old BC, BC-v2, optional DAgger, and Oracle on paired scenarios."""

    if validation_episodes < 1 or test_episodes < 1:
        raise ValueError("validation_episodes and test_episodes must be positive")
    paths: dict[str, Path] = {
        BASELINE_RL: Path(baseline_checkpoint),
        OLD_BC_POLICY: Path(old_bc_checkpoint),
        BC_V2_POLICY: Path(bc_v2_checkpoint),
    }
    if dagger_checkpoint is not None:
        paths[DAGGER_POLICY] = Path(dagger_checkpoint)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = Path(output_dir)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    _clear_exact_root(output_root, force=force)
    _clear_exact_root(logs_root, force=force)
    _clear_exact_root(plots_root, force=force)

    import torch

    payloads = {role: torch.load(path, map_location="cpu", weights_only=False) for role, path in paths.items()}
    configs = {role: _config_from_checkpoint(payload, output_dir=output_root / _role_root_name(role), progress=False) for role, payload in payloads.items()}
    trainers: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    policy_hash_before: dict[str, str] = {}
    for role, payload in payloads.items():
        trainer, hashes, _ = _load_frozen_trainer(payload, configs[role], metric_scales=configs[role].metric_scales)
        trainer.policy.eval()
        trainers[role] = trainer
        checkpoint_hashes[role] = _sha256(paths[role])
        policy_hash_before[role] = hashes["loaded_policy_hash"]
    oracle_config = configs[BASELINE_RL]
    role_order = [BASELINE_RL, OLD_BC_POLICY, BC_V2_POLICY]
    if DAGGER_POLICY in paths:
        role_order.append(DAGGER_POLICY)
    role_order.append(ACTOR_CONSTRAINED_ORACLE)
    aggregate_by_split: dict[str, dict[str, Any]] = {}
    all_summaries: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    role_reports: dict[str, dict[str, Any]] = {role: {} for role in role_order}
    for split, count in (("validation", int(validation_episodes)), ("test", int(test_episodes))):
        scenarios = _normal_evaluation_scenarios(oracle_config, split, count)
        split_summaries: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        for role_index, role in enumerate(role_order):
            role_root = output_root / _role_root_name(role) / "scale_1"
            role_root.mkdir(parents=True, exist_ok=True)
            role_summaries: list[dict[str, Any]] = []
            role_rows: list[dict[str, Any]] = []
            for item in scenarios:
                index = int(item["episode_index"])
                serial = role_index * 100_000 + (0 if split == "validation" else 50_000) + index
                if role == ACTOR_CONSTRAINED_ORACLE:
                    summary, rows = _run_teacher_episode(
                        trainer=trainers[BASELINE_RL],
                        config=oracle_config,
                        scale_root=role_root,
                        split=split,
                        phase="imitation_evaluation",
                        scenario=item["scenario"],
                        scenario_hash=item["scenario_hash"],
                        episode_index=index,
                        seed=int(item["seed"]),
                        serial=serial,
                    )
                else:
                    summary, rows = _run_rl_or_bc_episode(
                        role=BC_POLICY if role in (OLD_BC_POLICY, BC_V2_POLICY, DAGGER_POLICY) else BASELINE_RL,
                        trainer=trainers[role],
                        config=configs[role],
                        scale_root=role_root,
                        split=split,
                        scenario=item["scenario"],
                        scenario_hash=item["scenario_hash"],
                        episode_index=index,
                        seed=int(item["seed"]),
                        serial=serial,
                    )
                    summary, rows = _rename_policy(summary, rows, role)
                summary.update({"split": split, "scenario_hash": item["scenario_hash"], "paired_episode_index": index})
                role_summaries.append(summary)
                role_rows.extend(rows)
                if progress and (index == 1 or index % 10 == 0 or index == count):
                    print(f"[imitation-eval] {split} {role} {index}/{count}", flush=True)
            _copy_calibration_files(role_root, oracle_config)
            manifest = _write_scale_manifest(scale_root=role_root, config=oracle_config, episodes=[s["writer_summary"] for s in role_summaries])
            manifest.update({"dataset_id": BC_V2_DATASET_ID, "evaluation_only": True, "training_performed": False, "policy_updates_performed": False, "policy_source": role, "policy_label": POLICY_LABELS[role], "curriculum_used": False, "stable_criterion_reused": True})
            _write_json(role_root / "manifest.json", _jsonable(manifest))
            validation = validate_episode_benchmark(role_root)
            role_reports[role][split] = {"schema_validation": validation, "episodes": len(role_summaries)}
            split_summaries.extend(role_summaries)
            split_rows.extend(role_rows)
            all_summaries.extend(role_summaries)
            all_step_rows.extend(role_rows)
        aggregate_by_split[split] = {role: _aggregate_role(role, split_summaries, split_rows) for role in role_order}
        _write_csv(logs_root / f"episode_metrics_{split}.csv", split_summaries)
        _write_rows_csv(logs_root / f"step_metrics_{split}.csv", split_rows)
    _write_csv(logs_root / "episode_metrics.csv", all_summaries)
    _write_rows_csv(logs_root / "step_metrics.csv", all_step_rows)
    overall = {role: _aggregate_role(role, all_summaries, all_step_rows) for role in role_order}
    plots = _plot_evaluation(overall, plots_root, role_order)
    unchanged = {role: _sha256(paths[role]) == checkpoint_hashes[role] for role in paths}
    summary = {
        "dataset_id": BC_V2_DATASET_ID if DAGGER_POLICY not in paths else DAGGER_DATASET_ID,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "roles": role_order,
        "checkpoint_hashes": checkpoint_hashes,
        "policy_hash_before": policy_hash_before,
        "checkpoint_files_unchanged": unchanged,
        "curriculum_contract": {"validation": False, "test": False, "scenario_generation": "paired normal far-distance starts"},
        "validation": aggregate_by_split["validation"],
        "test": aggregate_by_split["test"],
        "overall": overall,
        "role_reports": role_reports,
        "plots": plots,
        "bc_evaluation_contract": {"loss_space": "post_tanh_scaled_physical_delta_v", "critic_frozen": True, "normalizer_frozen": True, "logstd_unchanged": True, "ppo_update_performed": False, "reward_or_environment_modified": False},
        "outputs": {"logs_root": str(logs_root.resolve()), "plots_root": str(plots_root.resolve()), "episode_metrics": str((logs_root / "episode_metrics.csv").resolve()), "step_metrics": str((logs_root / "step_metrics.csv").resolve())},
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    _write_json(output_root / "manifest.json", _jsonable(summary))
    return summary


def _dagger_student_control_row(
    *,
    trainer: Any,
    observations: Mapping[str, Any],
    teacher_control: Mapping[str, Any],
    student_action: np.ndarray,
) -> dict[str, Any]:
    visible = _extract_rl_visible_state(observations["uav_00"])
    velocity = np.asarray(visible["velocity_mps"], dtype=float)
    velocity_direction = _unit(velocity)
    demanded = max(0.0, -float(np.dot(velocity_direction, student_action)))
    del trainer
    return {
        "legal_delta_v_mps": student_action.tolist(),
        "raw_delta_v_norm_mps": float(np.linalg.norm(student_action)),
        "pre_velocity_mps": velocity.tolist(),
        "demanded_braking_delta_v_mps": demanded,
        "desired_speed_mps": float(teacher_control.get("desired_speed_mps", float("nan"))),
    }


def _run_dagger_rollout(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    scale_root: Path,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "train" / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="train",
        seed=seed,
        phase="dagger_rollout",
        environment=environment,
        normalization_fitted=bool(trainer.normalizer.fitted),
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source="dagger_student_rollout",
    )
    teacher = ActorConstrainedEstimatedStateOracleController(environment)
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        normalized = trainer.normalizer.transform(observations)
        actor_observation = copy.deepcopy(dict(normalized["uav_00"]))
        actor_observation["uav_id"] = "uav_00"
        with torch.no_grad():
            student_tensor = actor_physical_mean(trainer.policy, actor_observation)
        student_action = student_tensor.detach().cpu().numpy().astype(float)
        teacher_action, teacher_control = teacher.action(observations)
        teacher_delta = np.asarray(teacher_action["uav_00"]["delta_v"], dtype=float)
        pre_state = environment.states["uav_00"].copy()
        _assert_action_feasible(environment, {"uav_00": {"delta_v": student_action.tolist(), "measurement_uav_id": "none", "bandwidth": {}}})
        action = {"uav_00": {"delta_v": student_action.tolist(), "measurement_uav_id": "none", "bandwidth": {}}}
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        applied = np.asarray(execution.get("delta_v_applied", student_action), dtype=float)
        geometry = _runner_geometry(environment, goal=goal, pre_state=pre_state, post_state=environment.states["uav_00"], commanded=student_action, applied=applied)
        visible = _extract_rl_visible_state(observations["uav_00"])
        control = _dagger_student_control_row(trainer=trainer, observations=observations, teacher_control=teacher_control, student_action=student_action)
        row = _base_step_row(
            # _base_step_row is shared with the audit module, whose label
            # registry intentionally contains only its original policy roles.
            # Use a registered role while constructing the common schema, then
            # relabel this independent DAgger rollout below.
            policy=BASELINE_RL,
            episode_index=episode_index,
            seed=seed,
            step_index=step,
            timestamp_s=(step + 1) * environment.config.motion.dt_s,
            scenario_hash=scenario_hash,
            geometry=geometry,
            estimated_before=_extract_rl_visible_state(observations["uav_00"]),
            estimated_after=_extract_rl_visible_state(next_observations["uav_00"]),
            control=control,
            execution=execution,
            next_info=next_info,
            reward=next_info.get("learning_reward_components", {}),
        )
        row = _complete_geometry_flags(row)
        row.update(
            {
                "policy": DAGGER_POLICY,
                "policy_label": POLICY_LABELS[DAGGER_POLICY],
                "actor_observation": actor_observation,
                "student_action_delta_v_mps": student_action.tolist(),
                "teacher_action_delta_v_mps": teacher_delta.tolist(),
                "correction_action_error_norm_mps": float(np.linalg.norm(student_action - teacher_delta)),
                "teacher_action_norm_mps": float(np.linalg.norm(teacher_delta)),
                "estimated_goal_distance_m": float(visible["distance_m"]),
                "goal_direction": np.asarray(visible["goal_direction"], dtype=float).tolist(),
                "teacher_desired_speed_mps": float(teacher_control.get("desired_speed_mps", float("nan"))),
                "teacher_braking_component_mps": float(teacher_control.get("constrained_braking_projection_mps", 0.0)),
                "teacher_state_source": "actor_visible_node_features_only",
                "truth_state_read_for_control": False,
                "dagger_sample_source": "student_visited_state_oracle_relabel",
            }
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=action,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source="dagger_student_rollout",
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    summary = _episode_summary(rows, policy=BASELINE_RL, writer_summary=writer_summary)
    summary.update({"policy": DAGGER_POLICY, "policy_label": POLICY_LABELS[DAGGER_POLICY], "evaluation_only": False, "training_performed": False, "policy_updates_performed": False, "curriculum_used": False, "teacher_state_source": "actor_visible_node_features_only", "truth_state_read_for_control": False, "writer_summary": writer_summary})
    return summary, rows


def _correction_record(row: Mapping[str, Any], *, scenario_seed: int, noise_seed: int, episode_index: int, scenario_hash: str) -> dict[str, Any]:
    return {
        "dataset_id": DAGGER_DATASET_ID,
        "sample_source": "dagger_correction",
        "split": "train",
        "episode_index": int(episode_index),
        "step_index": int(row.get("step_index", 0)),
        "scenario_seed": int(scenario_seed),
        "noise_seed": int(noise_seed),
        "scenario_hash": scenario_hash,
        "observation": _copy_observation_for_json(row["actor_observation"]),
        "teacher_action_delta_v_mps": _jsonable(row["teacher_action_delta_v_mps"]),
        "student_action_delta_v_mps": _jsonable(row["student_action_delta_v_mps"]),
        "correction_action_error_norm_mps": float(row.get("correction_action_error_norm_mps", float("nan"))),
        "goal_distance_m": float(row.get("true_goal_distance_before_m", float("nan"))),
        "estimated_goal_distance_m": float(row.get("estimated_goal_distance_m", float("nan"))),
        "speed_mps": float(row.get("total_speed_before_mps", float("nan"))),
        "radial_velocity_mps": float(row.get("true_radial_velocity_before_mps", float("nan"))),
        "tangential_velocity_mps": float(row.get("tangential_speed_before_mps", float("nan"))),
        "goal_direction": _jsonable(row.get("goal_direction", (0.0, 0.0, 0.0))),
        "braking_component_mps": float(row.get("teacher_braking_component_mps", 0.0)),
        "student_action_norm_mps": float(row.get("action_norm_mps", float("nan"))),
        "distance_bin": str(row.get("distance_bin", "")),
        "negative_radial_velocity": int(row.get("negative_radial_velocity", 0) or 0),
        "tangential_action_mps": float(row.get("tangential_action_mps", float("nan"))),
    }


def _select_correction_records(records: Sequence[Mapping[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples < 1:
        return []
    if len(records) <= max_samples:
        return [dict(record) for record in records]
    bucket_keys = tuple(_BIN_KEYS.values())
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in bucket_keys}
    for record in records:
        key = _canonical_distance_bin(record.get("distance_bin", ""))
        buckets.setdefault(key, []).append(dict(record))
    selected: list[dict[str, Any]] = []
    bucket_index = 0
    while len(selected) < max_samples and any(buckets.values()):
        key = bucket_keys[bucket_index % len(bucket_keys)]
        bucket = buckets.get(key, [])
        if bucket:
            selected.append(bucket.pop(0))
        bucket_index += 1
        if bucket_index > max_samples * len(DISTANCE_BINS) and not any(buckets.values()):
            break
    return selected


def generate_dagger_corrections(
    bc_v2_checkpoint: str | Path,
    original_demonstrations_dir: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    train_episodes: int = 48,
    max_correction_samples: int | None = None,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Roll out BC-v2 on train-only scenarios and relabel visited states with Oracle."""

    if train_episodes < 1:
        raise ValueError("train_episodes must be positive")
    checkpoint = Path(bc_v2_checkpoint)
    original_root = Path(original_demonstrations_dir)
    output_root = Path(output_dir)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _clear_exact_root(output_root, force=force)
    _clear_exact_root(logs_root, force=force)
    scale_root = output_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    (scale_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    original_train = _read_demo_jsonl(original_root / "demonstrations" / "train.jsonl")
    original_validation = _read_demo_jsonl(original_root / "demonstrations" / "validation.jsonl")
    if max_correction_samples is None:
        max_correction_samples = len(original_train)
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=output_root, progress=False)
    if config.uav_count != 1:
        raise ValueError("DAgger requires a 1-UAV checkpoint")
    trainer, policy_hashes, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    _copy_calibration_files(scale_root, config)
    all_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    correction_records: list[dict[str, Any]] = []
    coverage = {"scenario_seeds": [], "noise_seeds": []}
    for episode_index in range(int(train_episodes)):
        scenario_seed = int(51001 + 37 * episode_index)
        noise_seed = int(config.seed + 1_000_000 + 1009 * episode_index)
        scenario_config = _scenario_for(
            replace(config, scenario_seed=scenario_seed, single_uav_braking_curriculum_fraction=0.0, single_uav_terminal_braking_curriculum_fraction=0.0, progress=False, force=True, command_line="DAgger train-only student rollout; no curriculum"),
            noise_seed,
            split="train",
            episode_index=episode_index,
            phase="train",
        )
        if "single_uav_braking_curriculum" in scenario_config:
            raise RuntimeError("DAgger scenario unexpectedly contains curriculum")
        scenario_hash = _canonical_hash(scenario_config)
        summary, rows = _run_dagger_rollout(
            trainer=trainer,
            config=config,
            scale_root=scale_root,
            scenario=scenario_config,
            scenario_hash=scenario_hash,
            episode_index=episode_index + 1,
            seed=noise_seed,
            serial=episode_index + 1,
        )
        summary.update({"scenario_seed": scenario_seed, "noise_seed": noise_seed, "scenario_hash": scenario_hash, "split": "train"})
        all_summaries.append(summary)
        all_rows.extend(rows)
        correction_records.extend(_correction_record(row, scenario_seed=scenario_seed, noise_seed=noise_seed, episode_index=episode_index + 1, scenario_hash=scenario_hash) for row in rows)
        coverage["scenario_seeds"].append(scenario_seed)
        coverage["noise_seeds"].append(noise_seed)
        if progress and (episode_index == 0 or (episode_index + 1) % 10 == 0 or episode_index + 1 == train_episodes):
            print(f"[dagger] train rollout {episode_index + 1}/{train_episodes}", flush=True)
    selected_corrections = _select_correction_records(correction_records, int(max_correction_samples))
    correction_path = output_root / "corrections" / "train.jsonl"
    for record in selected_corrections:
        _append_jsonl(correction_path, record)
    original_train_tagged = [{**dict(record), "sample_source": "original_expert"} for record in original_train]
    original_validation_tagged = [{**dict(record), "sample_source": "original_expert"} for record in original_validation]
    aggregate_root = output_root / "aggregated_dataset"
    aggregate_train_path = aggregate_root / "demonstrations" / "train.jsonl"
    aggregate_validation_path = aggregate_root / "demonstrations" / "validation.jsonl"
    for record in original_train_tagged + selected_corrections:
        _append_jsonl(aggregate_train_path, record)
    for record in original_validation_tagged:
        _append_jsonl(aggregate_validation_path, record)
    _write_csv(logs_root / "dagger_episode_metrics.csv", all_summaries)
    _write_rows_csv(logs_root / "dagger_step_metrics.csv", all_rows)
    manifest_scale = _write_scale_manifest(scale_root=scale_root, config=config, episodes=[s["writer_summary"] for s in all_summaries])
    manifest_scale.update({"dataset_id": DAGGER_DATASET_ID, "evaluation_only": False, "training_performed": False, "policy_updates_performed": False, "policy_source": "dagger_student_rollout", "teacher": {"controller": ACTOR_CONSTRAINED_ORACLE, "state_source": "actor_visible_node_features_only", "truth_state_read_for_control": False}, "curriculum_used": False})
    _write_json(scale_root / "manifest.json", _jsonable(manifest_scale))
    schema_validation = validate_episode_benchmark(scale_root)
    distance_counts = Counter(
        _canonical_distance_bin(record.get("distance_bin", ""))
        for record in selected_corrections
    )
    special_counts = {
        "negative_radial_velocity": int(sum(int(record.get("negative_radial_velocity", 0) or 0) for record in selected_corrections)),
        "entered_8m": int(sum(float(record.get("estimated_goal_distance_m", float("inf"))) <= 8.0 for record in selected_corrections)),
        "entered_1m": int(sum(float(record.get("estimated_goal_distance_m", float("inf"))) <= 1.0 for record in selected_corrections)),
    }
    summary = {
        "dataset_id": DAGGER_DATASET_ID,
        "source_checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)},
        "source_policy_hash": policy_hashes["loaded_policy_hash"],
        "original_demonstrations_dir": str(original_root.resolve()),
        "train_rollout_episodes": int(train_episodes),
        "correction_samples_collected": len(correction_records),
        "correction_samples_selected": len(selected_corrections),
        "original_expert_train_samples": len(original_train),
        "aggregated_train_samples": len(original_train_tagged) + len(selected_corrections),
        "aggregated_validation_samples": len(original_validation_tagged),
        "coverage": {**coverage, "scenario_seed_count": len(set(coverage["scenario_seeds"])), "noise_seed_count": len(set(coverage["noise_seeds"])), "test_scenarios_used": False},
        "correction_distance_distribution": {
            label: int(distance_counts.get(_BIN_KEYS[label], 0))
            for label in DISTANCE_BINS
        },
        "correction_special_state_counts": special_counts,
        "train_only_curriculum": True,
        "schema_validation": schema_validation,
        "aggregated_dataset": str(aggregate_root.resolve()),
        "correction_jsonl": str(correction_path.resolve()),
        "logs_root": str(logs_root.resolve()),
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    _write_json(output_root / "manifest.json", _jsonable(summary))
    _write_json(aggregate_root / "manifest.json", _jsonable({"dataset_id": DAGGER_DATASET_ID, "train_samples": summary["aggregated_train_samples"], "validation_samples": summary["aggregated_validation_samples"], "original_expert_train_samples": len(original_train), "dagger_correction_samples": len(selected_corrections), "validation_source": "original_expert_validation_only"}))
    return summary


__all__ = [
    "OLD_BC_POLICY",
    "BC_V2_POLICY",
    "DAGGER_POLICY",
    "train_behavior_cloning_v2",
    "train_dagger_actor",
    "evaluate_imitation_policies",
    "generate_dagger_corrections",
    "_distance_bin",
    "_offline_action_rows",
    "_select_correction_records",
]
