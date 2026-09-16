"""Stage 2B-R2: diagnose the Actor's left/right obstacle-learning bottleneck.

This module is intentionally a small, isolated supervised diagnostic.  It
does not run PPO, change the environment, or alter any previous experiment.
It starts every ablation from the Stage 2A observation-compatible checkpoint
and asks a narrow question: can the Actor learn the correct mirrored lateral
action when the target, gradient path, and trainable scope are controlled?
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.motion_model import apply_velocity_increment
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import FormalExperimentConfig, _make_environment
from .pilot import _assert_action_feasible, _assert_observation_finite, _seed_everything
from .single_uav_bc_v2_dagger import _batch_actor_physical_mean, _save_imitation_checkpoint
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_obstacle_dagger_stage2b import (
    _episode_step_row,
    _normal_scenario,
    _passed_obstacle,
    _physical_obstacles,
    _static_candidate_diagnostics,
    _summary_from_rows,
    _visible_state,
    _write_rows_csv,
)
from .single_uav_obstacle_dagger_stage2br import (
    BoundaryAwareGeometricObstacleOracleController,
    MIRROR_CATEGORIES,
    _actor_decision,
    _normal_no_obstacle_scenario,
    _stage2br_config_from_payload,
    mirror_static_scenario,
)
from .single_uav_oracle_behavior_cloning import _canonical_hash, _copy_observation_for_json


STAGE_NAME = "single_uav_obstacle_stage2br2_learning_bottleneck"
DEFAULT_STAGE2A_CHECKPOINT = Path(
    "data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/"
    "checkpoints/stage2a_observation_compatible_init.pt"
)
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026"
)

MIRROR_GROUPS = ("side", "front")
MIRROR_PAIR_COUNT = 8
TRAIN_PAIR_COUNT = 6
ABLATION_EPOCHS = 160
ABLATION_BATCH_SIZE = 16
ABLATION_LEARNING_RATE = 3.0e-3
ABLATION_SEED = 20262602
VIRTUAL_GRADIENT_STEP = 0.1
DIRECTION_COLUMNS = tuple(range(39, 42))
FIRST_LAYER_DIRECTION_NAMES = (
    "gat.layers.0.node_projection.weight",
    "gat.layers.0.residual.weight",
)
ALL_ABLATION_SCOPES = (
    "39_42_direction_columns",
    "first_layer",
    "actor_head",
    "full_actor",
)
SCOPE_LABELS = {
    "39_42_direction_columns": "39:42",
    "first_layer": "first_layer",
    "actor_head": "actor_head",
    "full_actor": "full_actor",
}
EXPECTED_SIDE = {"left": -1.0, "right": 1.0, "left_front": -1.0, "right_front": 1.0}


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _mean_finite(values: Iterable[Any]) -> float:
    numeric = [value for value in (_finite(item) for item in values) if np.isfinite(value)]
    return float(np.mean(numeric)) if numeric else float("nan")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _prepare_roots(*roots: Path) -> None:
    """Create only new exact output roots; never overwrite an existing root."""

    for root in roots:
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"R2 output root is not empty: {root}")
    for root in roots:
        Path(root).mkdir(parents=True, exist_ok=True)


def _load_payload_and_trainer(
    checkpoint: Path, *, output_dir: Path
) -> tuple[dict[str, Any], Any, FormalExperimentConfig, dict[str, Any]]:
    import torch

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _stage2br_config_from_payload(
        payload, output_dir=output_dir, command_line="Stage 2B-R2 frozen diagnostic"
    )
    if config.uav_count != 1 or config.node_feature_dim != 42:
        raise ValueError("Stage 2B-R2 requires the one-UAV Stage 2A 42-dimensional checkpoint")
    trainer, identity, _ = _load_frozen_trainer(
        payload, config, metric_scales=config.metric_scales
    )
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    return dict(payload), trainer, config, {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "policy_hash": _state_dict_hash(trainer.policy.state_dict()),
        "loaded_policy_hash": identity.get("loaded_policy_hash", ""),
    }


def _fresh_trainer_from_payload(
    payload: Mapping[str, Any], *, config: FormalExperimentConfig, output_dir: Path
) -> tuple[Any, dict[str, Any]]:
    """Load a fresh trainer from the same payload for every ablation."""

    _seed_everything(ABLATION_SEED)
    values = dict(config.__dict__)
    values["output_dir"] = Path(output_dir)
    values["force"] = True
    values["progress"] = False
    fresh_config = config.__class__(**values)
    trainer, identity, _ = _load_frozen_trainer(
        payload, fresh_config, metric_scales=fresh_config.metric_scales
    )
    trainer.policy.eval()
    for parameter in trainer.policy.parameters():
        parameter.requires_grad_(False)
    return trainer, {
        "loaded_policy_hash": identity.get("loaded_policy_hash", ""),
        "policy_hash": _state_dict_hash(trainer.policy.state_dict()),
    }


def _category_group(category: str) -> str:
    return "front" if "front" in str(category) else "side"


def _scenario_obstacle_center(scenario: Mapping[str, Any]) -> np.ndarray:
    specs = scenario.get("obstacle_specs", [])
    if not isinstance(specs, Sequence) or not specs:
        return np.zeros(3, dtype=float)
    first = specs[0]
    if not isinstance(first, Mapping):
        return np.zeros(3, dtype=float)
    return np.asarray(first.get("center_m", (0.0, 0.0, 0.0)), dtype=float).reshape(3)


def _scenario_obstacle_radius(scenario: Mapping[str, Any]) -> float:
    specs = scenario.get("obstacle_specs", [])
    if not isinstance(specs, Sequence) or not specs or not isinstance(specs[0], Mapping):
        return float("nan")
    return _finite(specs[0].get("radius_m"), float("nan"))


def _mirror_sample(
    trainer: Any,
    config: FormalExperimentConfig,
    *,
    category: str,
    pair_index: int,
    split: str,
) -> dict[str, Any]:
    side = "left" if category.startswith("left") else "right"
    seed = int(2200000 + int(pair_index) * 100 + MIRROR_CATEGORIES.index(category))
    scenario = mirror_static_scenario(
        config,
        category=category,
        pair_index=int(pair_index),
        side=side,
        seed=seed,
        round_index=0,
    )
    scenario["stage2br2"] = {
        "scenario_id": f"stage2br2_pair_{pair_index:04d}_{category}",
        "pair_id": f"pair_{pair_index:04d}",
        "mirror_group": _category_group(category),
        "split": str(split),
        "evaluation_only": True,
        "curriculum_used": False,
    }
    _seed_everything(seed)
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=False,
    )
    observations, info = environment.reset(
        seed=seed, scenario_config=copy.deepcopy(scenario)
    )
    _assert_observation_finite(observations)
    decision, frozen_action, actor_observation = _actor_decision(trainer, observations, info)
    del decision
    oracle = BoundaryAwareGeometricObstacleOracleController(environment)
    target_payload, control = oracle.action(observations)
    target_action = np.asarray(target_payload["uav_00"]["delta_v"], dtype=float).reshape(3)
    _assert_action_feasible(environment, target_payload)
    visible = _visible_state(observations["uav_00"])
    obstacle_center = _scenario_obstacle_center(scenario)
    obstacle_radius = _scenario_obstacle_radius(scenario)
    start = np.asarray(scenario["initial_positions"]["uav_00"], dtype=float).reshape(3)
    goal = np.asarray(scenario["goals"]["uav_00"], dtype=float).reshape(3)
    velocity = np.asarray(scenario["initial_velocities"]["uav_00"], dtype=float).reshape(3)
    return {
        "sample_id": f"pair_{pair_index:04d}_{category}",
        "scenario_id": scenario["stage2br2"]["scenario_id"],
        "pair_id": scenario["stage2br2"]["pair_id"],
        "mirror_group": scenario["stage2br2"]["mirror_group"],
        "category": str(category),
        "side": str(side),
        "split": str(split),
        "scenario_seed": seed,
        "noise_seed": seed,
        "scenario_hash": _canonical_hash(scenario),
        "obstacle_position_world_m": obstacle_center.tolist(),
        "obstacle_radius_m": float(obstacle_radius),
        "start_position_m": start.tolist(),
        "goal_position_m": goal.tolist(),
        "initial_velocity_mps": velocity.tolist(),
        "uav_yaw_rad": float(visible.get("yaw_rad", 0.0)),
        "obstacle_relative_dx_body_m": float(visible["obstacle_direction_body_m"][0]),
        "obstacle_relative_dy_body_m": float(visible["obstacle_direction_body_m"][1]),
        "obstacle_relative_dz_body_m": float(visible["obstacle_direction_body_m"][2]),
        "oracle_delta_v_x_mps": float(target_action[0]),
        "oracle_delta_v_y_mps": float(target_action[1]),
        "oracle_delta_v_z_mps": float(target_action[2]),
        "bc_target_delta_v_x_mps": float(target_action[0]),
        "bc_target_delta_v_y_mps": float(target_action[1]),
        "bc_target_delta_v_z_mps": float(target_action[2]),
        "frozen_stage2a_delta_v_x_mps": float(frozen_action[0]),
        "frozen_stage2a_delta_v_y_mps": float(frozen_action[1]),
        "frozen_stage2a_delta_v_z_mps": float(frozen_action[2]),
        "oracle_route_phase": str(control.get("route_phase", "")),
        "oracle_state_source": str(control.get("state_source", "actor_visible_node_features_only")),
        "oracle_truth_state_read": bool(control.get("truth_state_read_for_control", False)),
        "expected_lateral_sign": float(EXPECTED_SIDE[category]),
        "observation": actor_observation,
        "teacher_action_delta_v_mps": target_action.tolist(),
    }


def build_mirror_samples(
    trainer: Any,
    config: FormalExperimentConfig,
    *,
    pair_count: int = MIRROR_PAIR_COUNT,
) -> list[dict[str, Any]]:
    if int(pair_count) < 2:
        raise ValueError("pair_count must be at least 2 so train and validation pairs are disjoint")
    samples: list[dict[str, Any]] = []
    for pair_index in range(1, int(pair_count) + 1):
        split = "train" if pair_index <= TRAIN_PAIR_COUNT else "validation"
        for category in MIRROR_CATEGORIES:
            samples.append(
                _mirror_sample(
                    trainer,
                    config,
                    category=category,
                    pair_index=pair_index,
                    split=split,
                )
            )
    return samples


def _pair_key(category: str) -> str:
    return _category_group(category)


def audit_mirror_targets(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify exact geometry and target sign symmetry before any training."""

    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in samples:
        groups[(str(sample["pair_id"]), _pair_key(str(sample["category"])))] [
            "left" if str(sample["side"]) == "left" else "right"
        ] = sample
    pair_rows: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    for (pair_id, group), members in sorted(groups.items()):
        left = members.get("left")
        right = members.get("right")
        if left is None or right is None:
            invalid_reasons.append(f"missing_mirror_member:{pair_id}:{group}")
            continue

        def array(row: Mapping[str, Any], key: str) -> np.ndarray:
            return np.asarray(row.get(key, (float("nan"),) * 3), dtype=float).reshape(3)

        obstacle_left = array(left, "obstacle_position_world_m")
        obstacle_right = array(right, "obstacle_position_world_m")
        dx_left = float(left.get("obstacle_relative_dx_body_m", float("nan")))
        dx_right = float(right.get("obstacle_relative_dx_body_m", float("nan")))
        dy_left = float(left.get("obstacle_relative_dy_body_m", float("nan")))
        dy_right = float(right.get("obstacle_relative_dy_body_m", float("nan")))
        dz_left = float(left.get("obstacle_relative_dz_body_m", float("nan")))
        dz_right = float(right.get("obstacle_relative_dz_body_m", float("nan")))
        target_left = array(left, "teacher_action_delta_v_mps")
        target_right = array(right, "teacher_action_delta_v_mps")
        errors = {
            "world_x_abs_error_m": abs(float(obstacle_left[0] - obstacle_right[0])),
            "world_y_mirror_abs_error_m": abs(float(obstacle_left[1] + obstacle_right[1] - 200.0)),
            "world_z_abs_error_m": abs(float(obstacle_left[2] - obstacle_right[2])),
            "body_dx_abs_error_m": abs(dx_left - dx_right),
            "body_dy_mirror_abs_error_m": abs(dy_left + dy_right),
            "body_dz_abs_error_m": abs(dz_left - dz_right),
            "target_x_mirror_abs_error_m": abs(float(target_left[0] - target_right[0])),
            "target_y_mirror_abs_error_m": abs(float(target_left[1] + target_right[1])),
            "target_z_mirror_abs_error_m": abs(float(target_left[2] - target_right[2])),
        }
        left_expected = float(left["expected_lateral_sign"])
        right_expected = float(right["expected_lateral_sign"])
        row = {
            "pair_id": pair_id,
            "mirror_group": group,
            "left_category": str(left["category"]),
            "right_category": str(right["category"]),
            "left_obstacle_dy_body_m": dy_left,
            "right_obstacle_dy_body_m": dy_right,
            "left_oracle_delta_v_y_mps": float(target_left[1]),
            "right_oracle_delta_v_y_mps": float(target_right[1]),
            "left_target_sign_correct": int(target_left[1] * left_expected > 1.0e-8),
            "right_target_sign_correct": int(target_right[1] * right_expected > 1.0e-8),
            "left_action_legal": int(np.linalg.norm(target_left) <= 0.6000001),
            "right_action_legal": int(np.linalg.norm(target_right) <= 0.6000001),
            **errors,
        }
        row["pair_passed"] = int(
            all(float(value) <= 1.0e-6 for value in errors.values())
            and row["left_target_sign_correct"] == 1
            and row["right_target_sign_correct"] == 1
            and row["left_action_legal"] == 1
            and row["right_action_legal"] == 1
        )
        if not row["pair_passed"]:
            invalid_reasons.append(f"pair_failed:{pair_id}:{group}")
        pair_rows.append(row)
    passed = bool(pair_rows) and not invalid_reasons and all(int(row["pair_passed"]) == 1 for row in pair_rows)
    return {
        "audit": "strict_mirror_supervision_target",
        "samples": len(samples),
        "mirror_pairs": len(pair_rows),
        "pair_rows": pair_rows,
        "passed": passed,
        "mirror_target_audit_passed": passed,
        "invalid_reasons": invalid_reasons,
        "tolerance": 1.0e-6,
        "target_source": "Actor-constrained estimated-state Oracle",
        "truth_state_read_for_control": False,
    }


def _scope_parameter_selected(name: str, scope: str) -> bool:
    """Pure selection contract used both by training and unit tests."""

    name = str(name)
    scope = str(scope)
    if scope == "39_42_direction_columns":
        return name in FIRST_LAYER_DIRECTION_NAMES
    if scope == "first_layer":
        return name.startswith("gat.layers.0.")
    if scope == "actor_head":
        return name.startswith("gat.layers.0.") or name.startswith("actor_body.") or name.startswith("delta_mean_head.")
    if scope == "full_actor":
        return not name.startswith("critic.") and name != "delta_logstd"
    raise ValueError(f"unknown Stage 2B-R2 ablation scope: {scope}")


def configure_ablation_scope(policy: Any, scope: str) -> list[str]:
    """Freeze all parameters, then select exactly the requested Actor scope."""

    names: list[str] = []
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(False)
        if _scope_parameter_selected(name, scope):
            parameter.requires_grad_(True)
            names.append(name)
    if not names:
        raise RuntimeError(f"ablation scope has no trainable parameters: {scope}")
    for name, parameter in policy.named_parameters():
        if name.startswith("critic.") and parameter.requires_grad:
            raise AssertionError("critic must remain frozen")
        if name == "delta_logstd" and parameter.requires_grad:
            raise AssertionError("delta_logstd must remain frozen")
    return names


def _mask_direction_gradients(policy: Any, scope: str) -> None:
    if scope != "39_42_direction_columns":
        return
    parameters = dict(policy.named_parameters())
    for name in FIRST_LAYER_DIRECTION_NAMES:
        gradient = parameters[name].grad
        if gradient is not None:
            gradient[:, :39] = 0.0


def _gradient_vector(policy: Any, names: Sequence[str], *, direction_only: bool = False) -> np.ndarray:
    values: list[np.ndarray] = []
    parameters = dict(policy.named_parameters())
    for name in names:
        parameter = parameters[name]
        if parameter.grad is None:
            values.append(np.zeros(int(parameter.numel()), dtype=float))
            continue
        gradient = parameter.grad.detach().cpu().numpy().astype(float)
        if direction_only and name in FIRST_LAYER_DIRECTION_NAMES:
            gradient = gradient[:, 39:42]
        values.append(gradient.reshape(-1))
    return np.concatenate(values) if values else np.zeros(0, dtype=float)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-20 else float("nan")


def _sample_predictions(trainer: Any, samples: Sequence[Mapping[str, Any]]) -> np.ndarray:
    import torch

    if not samples:
        return np.empty((0, 3), dtype=float)
    with torch.no_grad():
        return _batch_actor_physical_mean(
            trainer, [sample["observation"] for sample in samples]
        ).detach().cpu().numpy().astype(float)


def _loss_tensors(trainer: Any, samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any]:
    import torch

    predictions = _batch_actor_physical_mean(
        trainer, [sample["observation"] for sample in samples]
    )
    targets = torch.as_tensor(
        np.asarray([sample["teacher_action_delta_v_mps"] for sample in samples], dtype=np.float32),
        dtype=torch.float32,
        device=predictions.device,
    )
    per_sample = torch.mean((predictions - targets) ** 2, dim=1)
    return per_sample.mean(), per_sample, predictions


def _restore_policy_state(policy: Any, state: Mapping[str, Any]) -> None:
    policy.load_state_dict(state, strict=True)
    policy.eval()


def gradient_audit(
    trainer: Any, samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute per-sample and paired left/right BC gradients from the source Actor."""

    import torch

    configure_ablation_scope(trainer.policy, "39_42_direction_columns")
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_category[str(sample["category"])].append(sample)
    sample_rows: list[dict[str, Any]] = []
    category_gradients: dict[str, np.ndarray] = {}
    category_direction_gradients: dict[str, np.ndarray] = {}
    category_effects: dict[str, dict[str, Any]] = {}
    base_state = {
        name: value.detach().cpu().clone() if hasattr(value, "detach") else copy.deepcopy(value)
        for name, value in trainer.policy.state_dict().items()
    }
    for category in MIRROR_CATEGORIES:
        category_samples = by_category.get(category, [])
        if not category_samples:
            continue
        per_sample_vectors: list[np.ndarray] = []
        per_sample_direction_vectors: list[np.ndarray] = []
        for sample in category_samples:
            trainer.policy.zero_grad(set_to_none=True)
            total_loss, per_sample_loss, prediction = _loss_tensors(trainer, [sample])
            total_loss.backward()
            _mask_direction_gradients(trainer.policy, "39_42_direction_columns")
            names = [name for name, parameter in trainer.policy.named_parameters() if parameter.requires_grad]
            vector = _gradient_vector(trainer.policy, names)
            direction_vector = _gradient_vector(trainer.policy, FIRST_LAYER_DIRECTION_NAMES, direction_only=True)
            per_sample_vectors.append(vector)
            per_sample_direction_vectors.append(direction_vector)
            predicted_y = float(prediction.detach().cpu().numpy()[0, 1])
            target_y = float(sample["teacher_action_delta_v_mps"][1])
            sample_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "pair_id": sample["pair_id"],
                    "category": category,
                    "side": sample["side"],
                    "predicted_delta_v_y_mps": predicted_y,
                    "target_delta_v_y_mps": target_y,
                    "lateral_error_mps": predicted_y - target_y,
                    "lateral_loss": float((predicted_y - target_y) ** 2),
                    "bc_loss": float(per_sample_loss.detach().cpu().numpy()[0]),
                    "trainable_gradient_norm": float(np.linalg.norm(vector)),
                    "direction_39_42_gradient_norm": float(np.linalg.norm(direction_vector)),
                    "gradient_finite": int(np.isfinite(vector).all()),
                    "gradient_nonzero": int(np.linalg.norm(vector) > 1.0e-12),
                }
            )
        category_gradients[category] = np.mean(np.stack(per_sample_vectors), axis=0)
        category_direction_gradients[category] = np.mean(
            np.stack(per_sample_direction_vectors), axis=0
        )
        # Apply one diagnostic gradient step to a fresh clone of the source
        # state, then restore it.  This is not a training update.
        trainer.policy.zero_grad(set_to_none=True)
        total_loss, _per_sample_loss, _prediction = _loss_tensors(trainer, category_samples)
        total_loss.backward()
        _mask_direction_gradients(trainer.policy, "39_42_direction_columns")
        grad_map = {
            name: parameter.grad.detach().clone()
            for name, parameter in trainer.policy.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        before = _sample_predictions(trainer, category_samples)
        with torch.no_grad():
            for name, gradient in grad_map.items():
                dict(trainer.policy.named_parameters())[name].add_(
                    -float(VIRTUAL_GRADIENT_STEP) * gradient
                )
        after = _sample_predictions(trainer, category_samples)
        _restore_policy_state(trainer.policy, base_state)
        changes = after[:, 1] - before[:, 1]
        desired = np.asarray(
            [float(sample["teacher_action_delta_v_mps"][1]) - float(before[index, 1]) for index, sample in enumerate(category_samples)],
            dtype=float,
        )
        category_effects[category] = {
            "action_y_before_mean": float(np.mean(before[:, 1])),
            "action_y_after_virtual_update_mean": float(np.mean(after[:, 1])),
            "action_y_change_mean": float(np.mean(changes)),
            "action_y_change_toward_target_fraction": float(np.mean(changes * desired > 0.0)),
            "virtual_gradient_step": float(VIRTUAL_GRADIENT_STEP),
        }

    pairwise: list[dict[str, Any]] = []
    for left_category, right_category in (("left", "right"), ("left_front", "right_front")):
        if left_category not in category_gradients or right_category not in category_gradients:
            continue
        left_gradient = category_gradients[left_category]
        right_gradient = category_gradients[right_category]
        left_direction = category_direction_gradients[left_category]
        right_direction = category_direction_gradients[right_category]
        denominator = np.linalg.norm(left_direction) + np.linalg.norm(right_direction)
        sum_ratio = float(np.linalg.norm(left_direction + right_direction) / denominator) if denominator > 1.0e-20 else float("nan")
        pairwise.append(
            {
                "left_category": left_category,
                "right_category": right_category,
                "gradient_cosine": _cosine(left_gradient, right_gradient),
                "direction_39_42_gradient_cosine": _cosine(left_direction, right_direction),
                "left_gradient_norm": float(np.linalg.norm(left_gradient)),
                "right_gradient_norm": float(np.linalg.norm(right_gradient)),
                "left_direction_gradient_norm": float(np.linalg.norm(left_direction)),
                "right_direction_gradient_norm": float(np.linalg.norm(right_direction)),
                "left_right_effective_update_sum_ratio": sum_ratio,
                "gradient_cancellation_fraction": float(1.0 - sum_ratio) if np.isfinite(sum_ratio) else float("nan"),
                "right_update_moves_toward_target": bool(
                    category_effects[right_category]["action_y_change_toward_target_fraction"] > 0.5
                ),
            }
        )
    all_vectors = list(category_direction_gradients.values())
    aggregate = np.mean(np.stack(all_vectors), axis=0) if all_vectors else np.zeros(0)
    return {
        "audit": "left_right_bc_gradient_path",
        "scope": "39:42 input columns only",
        "sample_rows": sample_rows,
        "category_effects": category_effects,
        "pairwise": pairwise,
        "aggregate_direction_gradient_norm": float(np.linalg.norm(aggregate)),
        "gradient_audit_completed": bool(len(sample_rows) == len(samples) and len(pairwise) == 2),
        "gradient_nonzero": bool(all(float(row["gradient_nonzero"]) == 1 for row in sample_rows)),
        "right_gradient_toward_target": bool(
            all(bool(row["right_update_moves_toward_target"]) for row in pairwise)
        ),
        "gradient_cancellation_present": bool(
            any(_finite(row.get("gradient_cancellation_fraction"), 0.0) > 0.5 for row in pairwise)
        ),
    }


def _fixed_epoch_orders(sample_count: int, epochs: int) -> list[np.ndarray]:
    rng = np.random.default_rng(ABLATION_SEED)
    return [rng.permutation(int(sample_count)) for _ in range(int(epochs))]


def _direction_metrics(trainer: Any, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = _sample_predictions(trainer, samples)
    rows: list[dict[str, Any]] = []
    category_accuracy: dict[str, int] = {}
    for category in MIRROR_CATEGORIES:
        selected = [index for index, sample in enumerate(samples) if sample["category"] == category]
        if not selected:
            continue
        expected = float(EXPECTED_SIDE[category])
        values = predictions[selected, 1]
        target_values = np.asarray(
            [float(samples[index]["teacher_action_delta_v_mps"][1]) for index in selected], dtype=float
        )
        sign_ok = int(float(np.mean(values)) * expected > 1.0e-8)
        category_accuracy[category] = sign_ok
        rows.append(
            {
                "category": category,
                "side": "left" if category.startswith("left") else "right",
                "expected_lateral_sign": expected,
                "mean_predicted_delta_v_y_mps": float(np.mean(values)),
                "mean_target_delta_v_y_mps": float(np.mean(target_values)),
                "sample_sign_accuracy": float(np.mean(values * expected > 1.0e-8)),
                "sign_correct": sign_ok,
                "samples": len(selected),
            }
        )
    correct = int(sum(category_accuracy.values()))
    return {
        "rows": rows,
        "correct": correct,
        "applicable": len(category_accuracy),
        "accuracy": float(correct / max(1, len(category_accuracy))),
        "four_of_four": bool(correct == 4 and len(category_accuracy) == 4),
    }


def _state_hash_map(policy: Any) -> dict[str, str]:
    return {
        name: _state_dict_hash({name: parameter.detach().cpu()})
        for name, parameter in policy.named_parameters()
    }


def _normalizer_hash(trainer: Any) -> str:
    return _canonical_hash(trainer.normalizer.state_dict())


def train_ablation(
    payload: Mapping[str, Any],
    config: FormalExperimentConfig,
    source_checkpoint: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    output_root: Path,
    max_epochs: int = ABLATION_EPOCHS,
    batch_size: int = ABLATION_BATCH_SIZE,
    learning_rate: float = ABLATION_LEARNING_RATE,
    epoch_orders: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Train one tiny Actor-only BC ablation from the same Stage 2A state."""

    import torch

    train_samples = [sample for sample in samples if sample["split"] == "train"]
    validation_samples = [sample for sample in samples if sample["split"] == "validation"]
    if not train_samples or not validation_samples:
        raise ValueError("R2 ablation requires non-empty train and validation mirror samples")
    trainer, identity = _fresh_trainer_from_payload(
        payload, config=config, output_dir=output_root / scope
    )
    source_policy_hash = identity["policy_hash"]
    initial_state = {
        name: value.detach().cpu().clone() if hasattr(value, "detach") else copy.deepcopy(value)
        for name, value in trainer.policy.state_dict().items()
    }
    initial_parameter_hashes = _state_hash_map(trainer.policy)
    initial_critic_hash = _state_dict_hash(trainer.policy.critic.state_dict())
    initial_logstd_hash = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    initial_normalizer_hash = _normalizer_hash(trainer)
    trainable_names = configure_ablation_scope(trainer.policy, scope)
    optimizer = torch.optim.Adam(
        [parameter for parameter in trainer.policy.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    if epoch_orders is None:
        epoch_orders = _fixed_epoch_orders(len(train_samples), int(max_epochs))
    history: list[dict[str, Any]] = []
    for epoch in range(int(max_epochs)):
        trainer.policy.eval()
        order = np.asarray(epoch_orders[epoch], dtype=int)
        batch_losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            selected = [train_samples[int(index)] for index in order[start : start + int(batch_size)]]
            optimizer.zero_grad(set_to_none=True)
            loss, _per_sample, _predictions = _loss_tensors(trainer, selected)
            loss.backward()
            _mask_direction_gradients(trainer.policy, scope)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            train_loss, train_per_sample, _train_predictions = _loss_tensors(trainer, train_samples)
            validation_loss, validation_per_sample, _validation_predictions = _loss_tensors(trainer, validation_samples)
        history.append(
            {
                "scope": scope,
                "epoch": int(epoch + 1),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "batch_loss_mean": _mean_finite(batch_losses),
                "train_mse": float(train_loss.detach().cpu()),
                "validation_mse": float(validation_loss.detach().cpu()),
                "train_action_mse_mean": float(train_per_sample.mean().detach().cpu()),
                "validation_action_mse_mean": float(validation_per_sample.mean().detach().cpu()),
            }
        )
    final_state = {
        name: value.detach().cpu().clone() if hasattr(value, "detach") else copy.deepcopy(value)
        for name, value in trainer.policy.state_dict().items()
    }
    final_policy_hash = _state_dict_hash(trainer.policy.state_dict())
    final_parameter_hashes = _state_hash_map(trainer.policy)
    changed_names = [
        name for name in initial_parameter_hashes if initial_parameter_hashes[name] != final_parameter_hashes[name]
    ]
    direction = _direction_metrics(trainer, samples)
    metadata = {
        "stage": STAGE_NAME,
        "ablation_scope": scope,
        "scope_label": SCOPE_LABELS[scope],
        "source_checkpoint": {"path": str(Path(source_checkpoint).resolve()), "sha256": _sha256(source_checkpoint)},
        "source_policy_hash": identity["policy_hash"],
        "initial_policy_hash": source_policy_hash,
        "final_policy_hash": final_policy_hash,
        "trainable_parameter_names": trainable_names,
        "critic_frozen": initial_critic_hash == _state_dict_hash(trainer.policy.critic.state_dict()),
        "normalizer_frozen": initial_normalizer_hash == _normalizer_hash(trainer),
        "logstd_frozen": initial_logstd_hash == _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()}),
        "ppo_updates": 0,
        "formal_training_run": False,
        "loss_space": "post_tanh_scaled_physical_delta_v_mse",
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(max_epochs),
        "changed_parameter_names": changed_names,
        "same_initialization_contract": source_policy_hash == identity["policy_hash"],
    }
    checkpoint_path = output_root / "checkpoints" / f"ablation_{scope}.pt"
    _save_imitation_checkpoint(payload, final_state, checkpoint_path, metadata)
    return {
        "scope": scope,
        "scope_label": SCOPE_LABELS[scope],
        "trainer": trainer,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_policy_hash": identity["policy_hash"],
        "initial_policy_hash": source_policy_hash,
        "final_policy_hash": final_policy_hash,
        "trainable_parameter_names": trainable_names,
        "changed_parameter_names": changed_names,
        "history": history,
        "initial_train_loss": float(history[0]["train_mse"]),
        "final_train_loss": float(history[-1]["train_mse"]),
        "final_validation_loss": float(history[-1]["validation_mse"]),
        "direction": direction,
        "checkpoint_contract": metadata,
    }


def _aggregate_navigation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "entered_8m": int(sum(int(row.get("entered_8m", 0) or 0) for row in rows)),
        "entered_5m": int(sum(int(row.get("entered_5m", 0) or 0) for row in rows)),
        "entered_3m": int(sum(int(row.get("entered_3m", 0) or 0) for row in rows)),
        "entered_1m": int(sum(int(row.get("entered_1m", 0) or 0) for row in rows)),
        "stable_goal": int(sum(int(row.get("stable_goal", 0) or 0) for row in rows)),
        "entered_8m_rate": _mean_finite([row.get("entered_8m") for row in rows]),
        "entered_5m_rate": _mean_finite([row.get("entered_5m") for row in rows]),
        "entered_3m_rate": _mean_finite([row.get("entered_3m") for row in rows]),
        "entered_1m_rate": _mean_finite([row.get("entered_1m") for row in rows]),
        "stable_goal_rate": _mean_finite([row.get("stable_goal") for row in rows]),
        "mean_minimum_goal_distance_m": _mean_finite([row.get("minimum_goal_distance_m") for row in rows]),
        "mean_maximum_consecutive_goal_hold_steps": _mean_finite([row.get("maximum_consecutive_goal_hold_steps") for row in rows]),
        "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in rows)),
        "safety_projection_steps": int(sum(int(row.get("safety_projection_steps", 0) or 0) for row in rows)),
        "collision_episodes": int(sum(int(row.get("collision", 0) or 0) for row in rows)),
    }


def normal_navigation_comparison(
    source_trainer: Any,
    candidate_results: Mapping[str, Mapping[str, Any]],
    config: FormalExperimentConfig,
    *,
    logs_root: Path,
    episodes: int = 4,
) -> dict[str, Any]:
    """Run a small common normal-navigation probe only for 4/4 candidates."""

    from .single_uav_obstacle_dagger_stage2br import _run_actor_diagnostic_episode

    roles: list[tuple[str, Any]] = [("stage2a_frozen", source_trainer)]
    roles.extend((str(scope), result["trainer"]) for scope, result in candidate_results.items())
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for role, trainer in roles:
        for episode_index in range(1, int(episodes) + 1):
            scenario = _normal_no_obstacle_scenario(
                config,
                seed=2300000 + episode_index,
                round_index=0,
                episode_index=episode_index,
            )
            summary, rows = _run_actor_diagnostic_episode(
                trainer=trainer,
                config=config,
                scenario=scenario,
                scenario_name="normal_navigation_probe",
                split="probe",
                safety_mode="on",
                episode_index=episode_index,
                serial=(episode_index if role == "stage2a_frozen" else 1000 + episode_index + 100 * roles.index((role, trainer))),
            )
            summary.update({"policy": role, "scenario_hash": _canonical_hash(scenario)})
            for row in rows:
                row.update({"policy": role, "scenario_hash": _canonical_hash(scenario)})
            summaries.append(summary)
            step_rows.extend(rows)
    _write_rows_csv(logs_root / "normal_navigation_comparison.csv", summaries)
    _write_rows_csv(logs_root / "normal_navigation_step_diagnostics.csv", step_rows)
    aggregates = {
        role: _aggregate_navigation_rows([row for row in summaries if row["policy"] == role])
        for role, _trainer in roles
    }
    baseline = aggregates["stage2a_frozen"]
    candidate_gates: dict[str, Any] = {}
    for scope in candidate_results:
        current = aggregates[scope]
        candidate_gates[scope] = {
            "entered_8m_not_collapsed": bool(current["entered_8m"] >= baseline["entered_8m"] - 1),
            "collision_free": bool(current["collision_episodes"] == 0),
            "preserved": bool(current["entered_8m"] >= baseline["entered_8m"] - 1 and current["collision_episodes"] == 0),
        }
    result = {
        "episodes_per_policy": int(episodes),
        "aggregates": aggregates,
        "candidate_gates": candidate_gates,
        "normal_navigation_preserved": bool(candidate_gates) and all(
            bool(gate["preserved"]) for gate in candidate_gates.values()
        ),
        "criterion": "successful candidate entered_8m count >= frozen Stage2A count - 1 and collision_episodes == 0",
        "evaluation_frozen": True,
    }
    _write_json(logs_root / "normal_navigation_comparison.json", _jsonable(result))
    return result


def _run_static_candidate_episode(
    trainer: Any,
    config: FormalExperimentConfig,
    *,
    scenario: Mapping[str, Any],
    policy: str,
    episode_index: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(2400000 + serial)
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    rows: list[dict[str, Any]] = []
    obstacle_center = _scenario_obstacle_center(scenario)
    obstacle_radius = _scenario_obstacle_radius(scenario)
    start_y = float(np.asarray(scenario["initial_positions"]["uav_00"], dtype=float)[1])
    expected_lateral = -1.0 if float(obstacle_center[1]) > start_y else 1.0
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        decision, action, _actor_observation = _actor_decision(trainer, observations, info)
        state_before = environment.states["uav_00"].copy()
        nominal_state, _ = apply_velocity_increment(
            state_before, action, environment.config.motion, safety_projector=None
        )
        nominal_position = np.asarray(nominal_state.position_m, dtype=float)
        pre_diag = _static_candidate_diagnostics(
            environment, state_before.position_m, nominal_position
        )
        next_observations, _rewards, terminated, truncated, next_info = environment.step(
            decision.actions
        )
        row = _episode_step_row(
            policy=policy,
            scenario_name="static_mirror_smoke",
            safety_mode="on",
            episode_index=episode_index,
            step=step,
            environment=environment,
            visible_before=_visible_state(observations["uav_00"]),
            position_before=np.asarray(state_before.position_m, dtype=float),
            velocity_before=np.asarray(state_before.velocity_mps, dtype=float),
            action=action,
            control={"route_phase": "actor_static_smoke", "lateral_action_projection_mps": float(action[1])},
            nominal_position=nominal_position,
            pre_diag=pre_diag,
            next_info=next_info,
        )
        row.update(
            {
                "position_before_x_m": float(state_before.position_m[0]),
                "position_before_y_m": float(state_before.position_m[1]),
                "position_after_x_m": float(environment.states["uav_00"].position_m[0]),
                "position_after_y_m": float(environment.states["uav_00"].position_m[1]),
            }
        )
        rows.append(row)
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    after_x = np.asarray([float(row["position_after_x_m"]) for row in rows], dtype=float)
    after_y = np.asarray([float(row["position_after_y_m"]) for row in rows], dtype=float)
    passed_index = np.flatnonzero(after_x > obstacle_center[0] + obstacle_radius + 0.5)
    passed = bool(len(passed_index))
    post_progress = False
    if passed:
        index = int(passed_index[0])
        before_distance = float(rows[index].get("true_goal_distance_before_m", float("inf")))
        later_distances = [float(row.get("true_goal_distance_after_m", float("inf"))) for row in rows[index:]]
        post_progress = bool(later_distances and min(later_distances) < before_distance - 1.0e-6)
    summary = _summary_from_rows(
        rows,
        policy=policy,
        scenario_name="static_mirror_smoke",
        safety_mode="on",
        episode_index=episode_index,
    )
    lateral_displacement = float(after_y[-1] - start_y) if len(after_y) else 0.0
    summary.update(
        {
            "obstacle_side": "left" if expected_lateral < 0 else "right",
            "expected_lateral_direction": expected_lateral,
            "lateral_displacement_m": lateral_displacement,
            "direction_adaptive": bool(lateral_displacement * expected_lateral > 0.5),
            "passed_obstacle": int(passed),
            "post_bypass_goal_progress": int(post_progress),
            "lateral_action_mean_mps": _mean_finite([row.get("raw_delta_v_y") for row in rows]),
            "policy_updates_performed": False,
            "ppo_updates": 0,
        }
    )
    return summary, rows


def static_smoke(
    source_trainer: Any,
    candidate: Mapping[str, Any],
    config: FormalExperimentConfig,
    *,
    logs_root: Path,
) -> dict[str, Any]:
    candidate_trainer = candidate["trainer"]
    scenarios: list[dict[str, Any]] = []
    for index, category in enumerate(("left", "right", "left", "right"), start=1):
        side = "left" if category == "left" else "right"
        scenarios.append(
            mirror_static_scenario(
                config,
                category=category,
                pair_index=7000 + index,
                side=side,
                seed=2500000 + index,
                round_index=0,
            )
        )
    summaries: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for episode_index, scenario in enumerate(scenarios, start=1):
        for role, trainer in (("stage2a_frozen", source_trainer), (str(candidate["scope"]), candidate_trainer)):
            summary, rows = _run_static_candidate_episode(
                trainer,
                config,
                scenario=scenario,
                policy=role,
                episode_index=episode_index,
                serial=episode_index + (0 if role == "stage2a_frozen" else 100),
            )
            summary["scenario_hash"] = _canonical_hash(scenario)
            for row in rows:
                row["scenario_hash"] = _canonical_hash(scenario)
            summaries.append(summary)
            step_rows.extend(rows)
    _write_rows_csv(logs_root / "static_smoke_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "static_smoke_step_diagnostics.csv", step_rows)
    by_role = {
        role: [row for row in summaries if row["policy"] == role]
        for role in ("stage2a_frozen", str(candidate["scope"]))
    }
    candidate_rows = by_role[str(candidate["scope"])]
    source_rows = by_role["stage2a_frozen"]
    result = {
        "run": True,
        "candidate_scope": str(candidate["scope"]),
        "episodes": len(candidate_rows),
        "by_policy": {
            role: {
                "episodes": len(rows),
                "direction_adaptive_episodes": int(sum(int(row.get("direction_adaptive", 0)) for row in rows)),
                "passed_obstacle_episodes": int(sum(int(row.get("passed_obstacle", 0)) for row in rows)),
                "post_bypass_goal_progress_episodes": int(sum(int(row.get("post_bypass_goal_progress", 0)) for row in rows)),
                "mean_minimum_goal_distance_m": _mean_finite([row.get("minimum_goal_distance_m") for row in rows]),
                "mean_minimum_post_clearance_m": _mean_finite([row.get("minimum_post_clearance_m") for row in rows]),
                "mean_maximum_consecutive_goal_hold_steps": _mean_finite([row.get("maximum_consecutive_goal_hold_steps") for row in rows]),
                "boundary_projection_steps": int(sum(int(row.get("boundary_projection_steps", 0) or 0) for row in rows)),
                "safety_projection_steps": int(sum(int(row.get("safety_projection_steps", 0) or 0) for row in rows)),
                "collision_episodes": int(sum(int(row.get("collision", 0) or 0) for row in rows)),
            }
            for role, rows in by_role.items()
        },
        "static_direction_adaptive": bool(
            candidate_rows and all(int(row.get("direction_adaptive", 0)) == 1 for row in candidate_rows)
        ),
        "passed_obstacle": bool(
            candidate_rows and all(int(row.get("passed_obstacle", 0)) == 1 for row in candidate_rows)
        ),
        "post_bypass_goal_progress": bool(
            candidate_rows and all(int(row.get("post_bypass_goal_progress", 0)) == 1 for row in candidate_rows)
        ),
        "boundary_interference_improved": bool(
            sum(int(row.get("boundary_projection_steps", 0) or 0) for row in candidate_rows)
            <= sum(int(row.get("boundary_projection_steps", 0) or 0) for row in source_rows)
        ),
        "evaluation_frozen": True,
    }
    _write_json(logs_root / "static_smoke.json", _jsonable(result))
    return result


def _write_plots(
    plots_root: Path,
    *,
    gradient: Mapping[str, Any],
    ablations: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, str] = {}
    pair_rows = list(gradient.get("pairwise", []))
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
    labels = [f"{row['left_category']} vs\n{row['right_category']}" for row in pair_rows]
    axes[0].bar(labels, [float(row.get("direction_39_42_gradient_cosine", 0.0) or 0.0) for row in pair_rows], color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title("Mirror direction-gradient cosine")
    axes[0].set_ylabel("cosine")
    axes[0].tick_params(axis="x", rotation=15)
    scope_names = list(ablations)
    axes[1].bar(scope_names, [float(ablations[name]["direction"]["accuracy"]) for name in scope_names], color="#f58518")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Tiny-overfit direction accuracy")
    axes[1].set_ylabel("fraction of four directions")
    axes[1].tick_params(axis="x", rotation=25)
    figure_path = plots_root / "stage2br2_gradient_and_ablation.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["gradient_and_ablation"] = str(figure_path.resolve())

    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for scope, result in ablations.items():
        history = result["history"]
        axis.plot(
            [int(row["epoch"]) for row in history],
            [float(row["train_mse"]) for row in history],
            label=f"{scope} train",
        )
        axis.plot(
            [int(row["epoch"]) for row in history],
            [float(row["validation_mse"]) for row in history],
            linestyle="--",
            label=f"{scope} validation",
        )
    axis.set_xlabel("BC epoch")
    axis.set_ylabel("physical delta-v MSE")
    axis.set_title("Stage 2B-R2 tiny-overfit loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    loss_path = plots_root / "stage2br2_ablation_loss.png"
    figure.savefig(loss_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output["ablation_loss"] = str(loss_path.resolve())
    return output


def _write_schema_manifest(
    data_root: Path,
    logs_root: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    dataset_root = data_root / "mirror_dataset"
    _write_jsonl(dataset_root / "train.jsonl", train)
    _write_jsonl(dataset_root / "validation.jsonl", validation)
    manifest = {
        "schema_version": "stage2br2-v1",
        "dataset": {
            "train_jsonl": str((dataset_root / "train.jsonl").resolve()),
            "validation_jsonl": str((dataset_root / "validation.jsonl").resolve()),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "train_pair_ids": sorted({str(row["pair_id"]) for row in train}),
            "validation_pair_ids": sorted({str(row["pair_id"]) for row in validation}),
            "trajectory_leakage": bool(
                {str(row["pair_id"]) for row in train}
                & {str(row["pair_id"]) for row in validation}
            ),
        },
        "required_log_files": [
            "mirror_target_audit.json",
            "mirror_target_samples.csv",
            "gradient_audit.json",
            "gradient_samples.csv",
            "overfit_ablation_results.json",
            "overfit_ablation_results.csv",
            "normal_navigation_comparison.json",
            "stage2br2_summary.json",
        ],
        "diagnostic_only": True,
        "ppo_updates": 0,
        "formal_training_run": False,
        "logs_root": str(logs_root.resolve()),
    }
    _write_json(data_root / "schema_manifest.json", _jsonable(manifest))
    return manifest


def run_stage2br2(
    *,
    stage2a_checkpoint: str | Path = DEFAULT_STAGE2A_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    pair_count: int = MIRROR_PAIR_COUNT,
    max_epochs: int = ABLATION_EPOCHS,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the complete small R2 diagnostic and write independent outputs."""

    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    _prepare_roots(data_root, logs_root, plots_root)
    checkpoint = Path(stage2a_checkpoint).resolve()
    payload, source_trainer, config, identity = _load_payload_and_trainer(
        checkpoint, output_dir=data_root / "source_policy"
    )
    if progress:
        print("[stage2br2] loaded Stage 2A source checkpoint", flush=True)
    samples = build_mirror_samples(source_trainer, config, pair_count=int(pair_count))
    mirror_audit = audit_mirror_targets(samples)
    _write_json(logs_root / "mirror_target_audit.json", _jsonable(mirror_audit))
    _write_rows_csv(
        logs_root / "mirror_target_samples.csv",
        [
            {key: value for key, value in sample.items() if key != "observation"}
            for sample in samples
        ],
    )
    schema_manifest = _write_schema_manifest(data_root, logs_root, samples=samples)

    summary: dict[str, Any] = {
        "stage": STAGE_NAME,
        "status": "started",
        "source_checkpoint": identity,
        "pair_count": int(pair_count),
        "samples": len(samples),
        "train_samples": int(sum(sample["split"] == "train" for sample in samples)),
        "validation_samples": int(sum(sample["split"] == "validation" for sample in samples)),
        "mirror_target_audit_passed": bool(mirror_audit["passed"]),
        "gradient_audit_completed": False,
        "ablation_39_42_direction_accuracy": None,
        "ablation_first_layer_direction_accuracy": None,
        "ablation_actor_head_direction_accuracy": None,
        "ablation_full_actor_direction_accuracy": None,
        "minimum_successful_unfreeze_scope": None,
        "normal_navigation_preserved": False,
        "static_smoke_run": False,
        "static_direction_adaptive": False,
        "passed_obstacle": False,
        "boundary_interference_improved": False,
        "ppo_updates": 0,
        "formal_training_run": False,
        "stage2br2_passed": False,
        "eligible_for_next_stage": False,
        "identified_blocker": None,
        "recommended_next_step": None,
        "schema_manifest": schema_manifest,
    }
    if not mirror_audit["passed"]:
        summary.update(
            {
                "status": "stopped_before_training",
                "identified_blocker": "mirror_supervision_target_invalid",
                "recommended_next_step": "inspect body/world transform, selector, Oracle sign and target serialization",
            }
        )
        _write_json(logs_root / "stage2br2_summary.json", _jsonable(summary))
        _write_json(data_root / "summary.json", _jsonable(summary))
        return summary

    gradient = gradient_audit(source_trainer, samples)
    _write_json(logs_root / "gradient_audit.json", _jsonable(gradient))
    _write_rows_csv(logs_root / "gradient_samples.csv", gradient["sample_rows"])
    summary["gradient_audit_completed"] = bool(gradient["gradient_audit_completed"])
    orders = _fixed_epoch_orders(
        int(sum(sample["split"] == "train" for sample in samples)), int(max_epochs)
    )
    ablation_results: dict[str, dict[str, Any]] = {}
    for scope in ALL_ABLATION_SCOPES:
        if progress:
            print(f"[stage2br2] tiny BC ablation: {scope}", flush=True)
        result = train_ablation(
            payload,
            config,
            checkpoint,
            samples,
            scope=scope,
            output_root=data_root,
            max_epochs=int(max_epochs),
            epoch_orders=orders,
        )
        ablation_results[scope] = result
        summary_field = {
            "39_42_direction_columns": "ablation_39_42_direction_accuracy",
            "first_layer": "ablation_first_layer_direction_accuracy",
            "actor_head": "ablation_actor_head_direction_accuracy",
            "full_actor": "ablation_full_actor_direction_accuracy",
        }[scope]
        summary[summary_field] = float(result["direction"]["accuracy"])

    ablation_rows: list[dict[str, Any]] = []
    for scope, result in ablation_results.items():
        direction = result["direction"]
        for row in direction["rows"]:
            ablation_rows.append(
                {
                    "ablation": scope,
                    "trainable_parameter_count": len(result["trainable_parameter_names"]),
                    "trainable_parameter_names": result["trainable_parameter_names"],
                    "initial_train_loss": result["initial_train_loss"],
                    "final_train_loss": result["final_train_loss"],
                    "final_validation_loss": result["final_validation_loss"],
                    "direction_accuracy": direction["accuracy"],
                    "four_of_four": int(direction["four_of_four"]),
                    **row,
                }
            )
    _write_rows_csv(logs_root / "overfit_ablation_results.csv", ablation_rows)
    compact_ablation = {
        scope: {
            key: value
            for key, value in result.items()
            if key not in {"trainer", "history"}
        }
        for scope, result in ablation_results.items()
    }
    _write_json(logs_root / "overfit_ablation_results.json", _jsonable(compact_ablation))
    successful = [
        scope
        for scope in ALL_ABLATION_SCOPES
        if bool(ablation_results[scope]["direction"]["four_of_four"])
    ]
    summary["minimum_successful_unfreeze_scope"] = successful[0] if successful else None

    successful_results = {scope: ablation_results[scope] for scope in successful}
    if successful_results:
        navigation = normal_navigation_comparison(
            source_trainer,
            successful_results,
            config,
            logs_root=logs_root,
            episodes=4,
        )
    else:
        navigation = {
            "evaluation_frozen": True,
            "normal_navigation_preserved": False,
            "candidate_gates": {},
            "reason": "no_ablation_reached_4_of_4",
        }
        _write_json(logs_root / "normal_navigation_comparison.json", _jsonable(navigation))
    summary["normal_navigation_preserved"] = bool(navigation.get("normal_navigation_preserved", False))

    static_result: dict[str, Any] | None = None
    if successful_results:
        static_result = static_smoke(
            source_trainer,
            ablation_results[successful[0]],
            config,
            logs_root=logs_root,
        )
        summary["static_smoke_run"] = True
        summary["static_direction_adaptive"] = bool(static_result["static_direction_adaptive"])
        summary["passed_obstacle"] = bool(static_result["passed_obstacle"])
        summary["boundary_interference_improved"] = bool(static_result["boundary_interference_improved"])
    else:
        summary["static_smoke_run"] = False

    plot_paths = _write_plots(plots_root, gradient=gradient, ablations=ablation_results)
    summary["plots"] = plot_paths
    summary["gradient_diagnostic"] = {
        "right_gradient_toward_target": bool(gradient["right_gradient_toward_target"]),
        "gradient_cancellation_present": bool(gradient["gradient_cancellation_present"]),
    }
    summary["ablation_results"] = {
        scope: {
            "four_of_four": bool(result["direction"]["four_of_four"]),
            "direction_accuracy": float(result["direction"]["accuracy"]),
            "initial_train_loss": result["initial_train_loss"],
            "final_train_loss": result["final_train_loss"],
            "final_validation_loss": result["final_validation_loss"],
            "checkpoint": result["checkpoint"],
            "trainable_parameter_names": result["trainable_parameter_names"],
            "critic_frozen": bool(result["checkpoint_contract"]["critic_frozen"]),
            "normalizer_frozen": bool(result["checkpoint_contract"]["normalizer_frozen"]),
            "logstd_frozen": bool(result["checkpoint_contract"]["logstd_frozen"]),
        }
        for scope, result in ablation_results.items()
    }
    if not successful:
        summary["identified_blocker"] = (
            "supervised_learning_path_invalid"
            if float(ablation_results["full_actor"]["direction"]["accuracy"]) < 1.0
            else "actor_trainable_scope_too_restrictive"
        )
        summary["recommended_next_step"] = (
            "inspect target/action indexing, coordinate transform, actor output extraction and post-processing before any new DAgger"
        )
    elif not summary["normal_navigation_preserved"]:
        summary["identified_blocker"] = "normal_navigation_collapse_for_successful_scope"
        summary["recommended_next_step"] = "retain the smallest 4/4 scope only after a navigation-preserving data/regularization design is specified"
    elif not (
        summary["static_direction_adaptive"]
        and summary["passed_obstacle"]
        and summary["boundary_interference_improved"]
    ):
        summary["identified_blocker"] = "tiny_direction_overfit_but_sequential_static_behavior_not_proven"
        summary["recommended_next_step"] = "collect a small balanced sequential DAgger set with the minimum successful Actor scope; do not start formal 100/20/20 yet"
    else:
        summary["identified_blocker"] = None
        summary["recommended_next_step"] = "the minimum successful scope is eligible for a separately approved next-stage DAgger smoke"
    summary["stage2br2_passed"] = bool(
        summary["mirror_target_audit_passed"]
        and summary["gradient_audit_completed"]
        and summary["minimum_successful_unfreeze_scope"] is not None
        and summary["normal_navigation_preserved"]
        and summary["static_smoke_run"]
        and summary["static_direction_adaptive"]
        and summary["passed_obstacle"]
        and summary["boundary_interference_improved"]
    )
    summary["eligible_for_next_stage"] = bool(summary["stage2br2_passed"])
    summary["status"] = "completed"
    summary["static_smoke"] = static_result
    _write_json(logs_root / "stage2br2_summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--stage2a-checkpoint", type=Path, default=DEFAULT_STAGE2A_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--pair-count", type=int, default=MIRROR_PAIR_COUNT)
    parser.add_argument("--max-epochs", type=int, default=ABLATION_EPOCHS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage2br2(
        stage2a_checkpoint=args.stage2a_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        pair_count=args.pair_count,
        max_epochs=args.max_epochs,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALL_ABLATION_SCOPES",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "DEFAULT_STAGE2A_CHECKPOINT",
    "EXPECTED_SIDE",
    "MIRROR_CATEGORIES",
    "audit_mirror_targets",
    "build_mirror_samples",
    "configure_ablation_scope",
    "gradient_audit",
    "run_stage2br2",
    "train_ablation",
]
