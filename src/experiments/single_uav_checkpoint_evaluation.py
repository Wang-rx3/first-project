"""Read-only start-distance diagnostics for a frozen single-UAV checkpoint.

This module deliberately does not call ``MAPPOTrainer.update`` or add any
transition to a rollout.  It loads the policy and the train-fitted
normalizers from a completed formal run, then evaluates the frozen policy on
four independent test-only start-distance groups.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..learning.mappo import MAPPOConfig, MAPPOTrainer
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.motion_model import MotionConfig
from ..synthetic.episode_generator import _jsonable, _sha256, _write_csv, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _run_episode,
    _scale_seed,
    _scenario_for,
)


START_DISTANCE_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "terminal_1p5_3m",
        "label": "1.5-3 m",
        "low_m": 1.5,
        "high_m": 3.0,
        "kind": "controlled_terminal_range",
    },
    {
        "name": "terminal_3_5m",
        "label": "3-5 m",
        "low_m": 3.0,
        "high_m": 5.0,
        "kind": "controlled_terminal_range",
    },
    {
        "name": "terminal_5_8m",
        "label": "5-8 m",
        "low_m": 5.0,
        "high_m": 8.0,
        "kind": "controlled_terminal_range",
    },
    {
        "name": "normal_far_distance",
        "label": "normal far-distance",
        "low_m": None,
        "high_m": None,
        "kind": "original_normal_test_start",
    },
)

_EPISODE_METRIC_COLUMNS = (
    "group",
    "group_label",
    "episode_index",
    "episode_id",
    "seed",
    "requested_start_distance_m",
    "actual_start_distance_m",
    "initial_speed_mps",
    "entered_8m",
    "entered_1m",
    "stable_goal",
    "minimum_goal_distance_m",
    "mean_speed_mps",
    "maximum_consecutive_goal_hold_steps",
    "hold_steps_ge_1",
    "hold_steps_ge_3",
    "hold_steps_ge_5",
)

_GROUP_SUMMARY_COLUMNS = (
    "group",
    "group_label",
    "episodes",
    "distance_range_m",
    "mean_actual_start_distance_m",
    "minimum_actual_start_distance_m",
    "maximum_actual_start_distance_m",
    "entered_8m_episodes",
    "entered_8m_rate",
    "entered_1m_episodes",
    "entered_1m_rate",
    "stable_goal_episodes",
    "stable_goal_rate",
    "minimum_goal_distance_m",
    "mean_minimum_goal_distance_m",
    "mean_speed_mps",
    "mean_maximum_consecutive_goal_hold_steps",
    "maximum_consecutive_goal_hold_steps_max",
    "episodes_with_hold_steps_ge_1",
    "episodes_with_hold_steps_ge_1_rate",
    "episodes_with_hold_steps_ge_3",
    "episodes_with_hold_steps_ge_3_rate",
    "episodes_with_hold_steps_ge_5",
    "episodes_with_hold_steps_ge_5_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _state_dict_hash(state: Mapping[str, Any]) -> str:
    """Hash model parameters without touching optimizer state."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode("utf-8"))
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _distance_grid(low_m: float, high_m: float, count: int) -> list[float]:
    """Return equally spaced, inclusive probe distances for one range."""

    if count < 1:
        raise ValueError("episodes_per_group must be positive")
    if not np.isfinite(low_m) or not np.isfinite(high_m) or high_m < low_m:
        raise ValueError("invalid start-distance range")
    return [float(value) for value in np.linspace(low_m, high_m, count)]


def _terminal_initial_speed(config: FormalExperimentConfig, distance_m: float) -> float:
    """Reuse the formal terminal-start speed rule without changing training."""

    usable_distance = max(
        0.0,
        float(distance_m)
        - float(config.single_uav_goal_threshold_m)
        - float(config.single_uav_braking_margin_m),
    )
    conservative_target = float(
        np.sqrt(
            max(
                0.0,
                float(config.single_uav_goal_speed_threshold_mps) ** 2
                + 2.0
                * float(config.single_uav_braking_acceleration_mps2)
                * usable_distance,
            )
        )
    )
    return float(
        np.clip(
            max(
                float(config.single_uav_goal_speed_threshold_mps) + 1.5,
                conservative_target + 1.0,
            ),
            float(config.single_uav_goal_speed_threshold_mps) + 1.0,
            float(MotionConfig().max_speed_mps),
        )
    )


def _scenario_start_distance(scenario: Mapping[str, Any]) -> float:
    positions = scenario.get("initial_positions", {})
    goals = scenario.get("goals", {})
    if not isinstance(positions, Mapping) or not isinstance(goals, Mapping):
        return float("nan")
    if "uav_00" not in positions or "uav_00" not in goals:
        return float("nan")
    try:
        return float(
            np.linalg.norm(
                np.asarray(goals["uav_00"], dtype=float).reshape(3)
                - np.asarray(positions["uav_00"], dtype=float).reshape(3)
            )
        )
    except (TypeError, ValueError):
        return float("nan")


def _diagnostic_scenario(
    base_scenario: Mapping[str, Any],
    config: FormalExperimentConfig,
    group: Mapping[str, Any],
    requested_distance_m: float | None,
) -> tuple[dict[str, Any], float, float]:
    """Create a test-only controlled start while preserving scene geometry."""

    result = copy.deepcopy(dict(base_scenario))
    source_distance = _scenario_start_distance(result)
    positions = result.get("initial_positions")
    goals = result.get("goals")
    velocities = result.get("initial_velocities")
    actual_distance = source_distance
    initial_speed = 0.0

    if (
        requested_distance_m is not None
        and isinstance(positions, Mapping)
        and isinstance(goals, Mapping)
        and "uav_00" in positions
        and "uav_00" in goals
    ):
        start = np.asarray(positions["uav_00"], dtype=float).reshape(3)
        goal = np.asarray(goals["uav_00"], dtype=float).reshape(3)
        displacement = goal - start
        displacement_norm = float(np.linalg.norm(displacement))
        if not np.isfinite(displacement_norm) or displacement_norm <= 1e-9:
            raise ValueError("cannot construct a controlled start from a zero-length goal vector")
        direction = displacement / displacement_norm
        controlled_position = goal - direction * float(requested_distance_m)
        bounds_min = np.asarray(result.get("bounds_min_m", (0.0, 0.0, 2.0)), dtype=float)
        bounds_max = np.asarray(result.get("bounds_max_m", (200.0, 200.0, 60.0)), dtype=float)
        controlled_position = np.clip(controlled_position, bounds_min, bounds_max)
        actual_distance = float(np.linalg.norm(goal - controlled_position))
        if not np.isfinite(actual_distance) or actual_distance <= 0.0:
            raise ValueError("controlled diagnostic start has an invalid actual distance")
        initial_speed = _terminal_initial_speed(config, actual_distance)
        toward_goal = goal - controlled_position
        toward_goal_norm = float(np.linalg.norm(toward_goal))
        initial_velocity = toward_goal / toward_goal_norm * initial_speed

        updated_positions = dict(positions)
        updated_positions["uav_00"] = controlled_position.tolist()
        result["initial_positions"] = updated_positions
        updated_velocities = dict(velocities) if isinstance(velocities, Mapping) else {}
        updated_velocities["uav_00"] = initial_velocity.tolist()
        result["initial_velocities"] = updated_velocities

    result["evaluation_diagnostic"] = {
        "evaluation_only": True,
        "curriculum_used": False,
        "group": str(group["name"]),
        "group_label": str(group["label"]),
        "start_distance_range_m": (
            [float(group["low_m"]), float(group["high_m"])]
            if group["low_m"] is not None
            else "original_normal_far_distance"
        ),
        "requested_start_distance_m": (
            float(requested_distance_m) if requested_distance_m is not None else None
        ),
        "actual_start_distance_m": float(actual_distance),
        "initial_speed_mps": float(initial_speed),
        "initial_velocity_nonzero": bool(initial_speed > 0.0),
        "initial_velocity_direction": "toward_goal" if initial_speed > 0.0 else "original",
        "scene_geometry_preserved": True,
        "policy_updates_enabled": False,
    }
    return result, float(actual_distance), float(initial_speed)


def _config_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    progress: bool,
) -> FormalExperimentConfig:
    raw = payload.get("experiment_config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint is missing experiment_config")
    field_names = {field.name for field in fields(FormalExperimentConfig)}
    path_fields = {"output_dir", "scenario_profile_path", "scenario_path", "scenario_pool_root"}
    values: dict[str, Any] = {}
    for name in field_names:
        if name not in raw:
            continue
        value = copy.deepcopy(raw[name])
        if name in path_fields and value is not None:
            value = Path(value)
        values[name] = value
    values["output_dir"] = Path(output_dir)
    values["force"] = True
    values["progress"] = bool(progress)
    values["command_line"] = "frozen-checkpoint start-distance diagnostic evaluation"
    if values.get("potential_statistics") is None and isinstance(
        payload.get("potential_statistics"), Mapping
    ):
        values["potential_statistics"] = copy.deepcopy(dict(payload["potential_statistics"]))
    if values.get("visual_odometry_quality_model") is None and isinstance(
        payload.get("visual_odometry_quality_model"), Mapping
    ):
        values["visual_odometry_quality_model"] = copy.deepcopy(
            dict(payload["visual_odometry_quality_model"])
        )
    config = FormalExperimentConfig(**values)
    if config.uav_count != 1:
        raise ValueError(f"the diagnostic evaluator requires a 1-UAV checkpoint, got {config.uav_count}")
    return config


def _load_frozen_trainer(
    checkpoint_payload: Mapping[str, Any],
    config: FormalExperimentConfig,
    *,
    metric_scales: Mapping[str, float] | None,
) -> tuple[MAPPOTrainer, dict[str, str], dict[str, str]]:
    normalizer_state = checkpoint_payload.get("normalization_stats")
    central_state = checkpoint_payload.get("central_normalization_stats")
    trainer_state = checkpoint_payload.get("trainer_config")
    if not isinstance(normalizer_state, Mapping) or not isinstance(central_state, Mapping):
        raise ValueError("checkpoint is missing frozen normalizer state")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("checkpoint is missing trainer_config")

    normalizer = ObservationNormalizer.from_state_dict(normalizer_state)
    central_normalizer = FeatureNormalizer.from_state_dict(central_state)
    model_env = _make_environment(
        config,
        _scale_seed(config.seed, config.uav_count, "train"),
        metric_scales=metric_scales,
        visual_odometry_quality_model=config.visual_odometry_quality_model,
    )
    trainer_field_names = {field.name for field in fields(MAPPOConfig)}
    trainer_kwargs = {
        name: copy.deepcopy(trainer_state[name])
        for name in trainer_field_names
        if name in trainer_state and name != "measurement_persistent_proxy_model"
    }
    quality_model = normalizer_state.get("measurement_quality_model")
    proxy_model = checkpoint_payload.get("measurement_proxy_model")
    persistent_model = checkpoint_payload.get("measurement_persistent_proxy_model")
    trainer = MAPPOTrainer.from_environment(
        model_env,
        seed=_scale_seed(config.seed, config.uav_count, "train"),
        normalizer=normalizer,
        central_normalizer=central_normalizer,
        measurement_quality_model=(quality_model if isinstance(quality_model, Mapping) else None),
        measurement_proxy_model=(proxy_model if isinstance(proxy_model, Mapping) else None),
        measurement_persistent_proxy_model=(
            persistent_model if isinstance(persistent_model, Mapping) else None
        ),
        **trainer_kwargs,
    )
    initial_hash = _state_dict_hash(trainer.policy.state_dict())
    policy_state = checkpoint_payload.get("policy_state_dict")
    if not isinstance(policy_state, Mapping):
        raise ValueError("checkpoint is missing policy_state_dict")
    trainer.policy.load_state_dict(policy_state, strict=True)
    loaded_hash = _state_dict_hash(trainer.policy.state_dict())
    trainer.policy.eval()
    trainer.load_measurement_rate_dual_state_dict(
        checkpoint_payload.get("measurement_rate_dual_state")
    )
    trainer.load_measurement_communication_dual_state_dict(
        checkpoint_payload.get("measurement_communication_dual_state")
    )
    return trainer, {"constructed_policy_hash": initial_hash, "loaded_policy_hash": loaded_hash}, {
        "checkpoint_policy_hash": loaded_hash,
        "checkpoint_sha256": "",
    }


def _reward_rows(path: Path) -> list[dict[str, str]]:
    with (path / "rewards.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _episode_navigation_metrics(
    episode_path: Path,
    *,
    group: Mapping[str, Any],
    episode_index: int,
    seed: int,
    requested_distance_m: float | None,
    actual_start_distance_m: float,
    initial_speed_mps: float,
) -> dict[str, Any]:
    rows = _reward_rows(episode_path)
    distances: list[float] = []
    speeds: list[float] = []
    hold_steps: list[float] = []
    stable_flags: list[bool] = []
    for row in rows:
        distance = _finite_or_none(row.get("current_goal_distance"))
        if distance is not None:
            distances.append(distance)
        speed = _finite_or_none(row.get("single_uav_speed_mps"))
        if speed is not None:
            speeds.append(speed)
        hold = _finite_or_none(row.get("single_uav_stable_steps"))
        if hold is not None:
            hold_steps.append(hold)
        stable = _finite_or_none(row.get("single_uav_goal_stable"))
        if stable is not None:
            stable_flags.append(stable > 0.5)
    if not distances:
        raise ValueError(f"episode has no finite goal-distance samples: {episode_path}")
    maximum_hold = int(round(max(hold_steps, default=0.0)))
    return {
        "group": str(group["name"]),
        "group_label": str(group["label"]),
        "episode_index": int(episode_index),
        "episode_id": episode_path.name,
        "seed": int(seed),
        "requested_start_distance_m": (
            float(requested_distance_m) if requested_distance_m is not None else None
        ),
        "actual_start_distance_m": float(actual_start_distance_m),
        "initial_speed_mps": float(initial_speed_mps),
        "entered_8m": int(min(distances) <= 8.0),
        "entered_1m": int(min(distances) <= 1.0),
        "stable_goal": int(any(stable_flags)),
        "minimum_goal_distance_m": float(min(distances)),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else float("nan"),
        "maximum_consecutive_goal_hold_steps": maximum_hold,
        "hold_steps_ge_1": int(maximum_hold >= 1),
        "hold_steps_ge_3": int(maximum_hold >= 3),
        "hold_steps_ge_5": int(maximum_hold >= 5),
    }


def _group_summary(group: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if str(row["group"]) == str(group["name"])]
    count = len(selected)
    denominator = max(1, count)

    def count_field(name: str) -> int:
        return int(sum(int(row.get(name, 0)) for row in selected))

    starts = [float(row["actual_start_distance_m"]) for row in selected]
    minima = [float(row["minimum_goal_distance_m"]) for row in selected]
    speeds = [float(row["mean_speed_mps"]) for row in selected]
    holds = [int(row["maximum_consecutive_goal_hold_steps"]) for row in selected]
    ranges = (
        [float(group["low_m"]), float(group["high_m"])]
        if group["low_m"] is not None
        else "original_normal_far_distance"
    )
    entry8 = count_field("entered_8m")
    entry1 = count_field("entered_1m")
    stable = count_field("stable_goal")
    hold1 = count_field("hold_steps_ge_1")
    hold3 = count_field("hold_steps_ge_3")
    hold5 = count_field("hold_steps_ge_5")
    return {
        "group": str(group["name"]),
        "group_label": str(group["label"]),
        "episodes": count,
        "distance_range_m": ranges,
        "mean_actual_start_distance_m": float(np.mean(starts)) if starts else float("nan"),
        "minimum_actual_start_distance_m": float(min(starts)) if starts else float("nan"),
        "maximum_actual_start_distance_m": float(max(starts)) if starts else float("nan"),
        "entered_8m_episodes": entry8,
        "entered_8m_rate": entry8 / denominator if count else 0.0,
        "entered_1m_episodes": entry1,
        "entered_1m_rate": entry1 / denominator if count else 0.0,
        "stable_goal_episodes": stable,
        "stable_goal_rate": stable / denominator if count else 0.0,
        "minimum_goal_distance_m": float(min(minima)) if minima else float("nan"),
        "mean_minimum_goal_distance_m": float(np.mean(minima)) if minima else float("nan"),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else float("nan"),
        "mean_maximum_consecutive_goal_hold_steps": (
            float(np.mean(holds)) if holds else float("nan")
        ),
        "maximum_consecutive_goal_hold_steps_max": max(holds, default=0),
        "episodes_with_hold_steps_ge_1": hold1,
        "episodes_with_hold_steps_ge_1_rate": hold1 / denominator if count else 0.0,
        "episodes_with_hold_steps_ge_3": hold3,
        "episodes_with_hold_steps_ge_3_rate": hold3 / denominator if count else 0.0,
        "episodes_with_hold_steps_ge_5": hold5,
        "episodes_with_hold_steps_ge_5_rate": hold5 / denominator if count else 0.0,
    }


def _plot_group_summary(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["group_label"]) for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        ("Entry rate", ("entered_8m_rate", "entered_1m_rate", "stable_goal_rate"), ("8 m", "1 m", "stable"), (0.0, 1.0)),
        ("Minimum goal distance", ("minimum_goal_distance_m", "mean_minimum_goal_distance_m"), ("best", "mean episode min"), None),
        ("Mean speed", ("mean_speed_mps",), ("all steps",), None),
        ("Mean maximum hold", ("mean_maximum_consecutive_goal_hold_steps",), ("mean hold steps",), None),
        ("Hold-step episode counts", ("episodes_with_hold_steps_ge_1", "episodes_with_hold_steps_ge_3", "episodes_with_hold_steps_ge_5"), (">=1", ">=3", ">=5"), None),
        ("Maximum hold", ("maximum_consecutive_goal_hold_steps_max",), ("maximum",), None),
    )
    colors = ("#7c3aed", "#2563eb", "#16a34a")
    for axis, (title, keys, series_labels, ylim) in zip(axes.flat, panels):
        width = 0.75 / max(1, len(keys))
        for index, (key, series_label) in enumerate(zip(keys, series_labels)):
            values = [float(row.get(key, 0.0)) for row in rows]
            axis.bar(
                positions + (index - (len(keys) - 1) / 2.0) * width,
                values,
                width=width,
                label=series_label,
                color=colors[index % len(colors)],
            )
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if ylim is not None:
            axis.set_ylim(*ylim)
        if len(keys) > 1:
            axis.legend()
    figure.suptitle("Frozen terminal-policy start-distance diagnostic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_diagnostic_manifests(
    *,
    output_root: Path,
    scale_root: Path,
    config: FormalExperimentConfig,
    episode_summaries: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> None:
    scale_manifest = {
        "dataset_id": "single_uav_frozen_checkpoint_start_distance_diagnostic",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine": "python_sim",
        "engine_status": "pure_python_action_driven_environment_with_frozen_mappo_policy",
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "uav_count": 1,
        "policy_source": "frozen_checkpoint",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
        },
        "config": _jsonable({**{field.name: getattr(config, field.name) for field in fields(config)}, "node_feature_dim": config.node_feature_dim}),
        "calibration": {
            "localization": "calibration/localization_noise.json",
            "communication": "calibration/communication_model.json",
        },
        "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
        "episodes": [_jsonable(dict(entry)) for entry in episode_summaries],
        "groups": _jsonable(list(START_DISTANCE_GROUPS)),
        "scope": {
            "split": "test",
            "episodes_per_group": len(episode_summaries) // len(START_DISTANCE_GROUPS),
            "controlled_ranges_m": [[1.5, 3.0], [3.0, 5.0], [5.0, 8.0]],
            "normal_group": "original normal far-distance test start",
            "scene_geometry": "fixed by checkpoint scenario_seed when present",
            "independent_episode_seeds": True,
        },
        "read_only_contract": [
            "MAPPOTrainer.update was not called",
            "no rollout transition was added",
            "optimizer_state_dict was not loaded or written",
            "no reward, curriculum, dynamics, action-space, or PPO parameter was changed",
        ],
    }
    _write_json(scale_root / "manifest.json", scale_manifest)
    _write_json(
        output_root / "manifest.json",
        {
            "dataset_id": scale_manifest["dataset_id"],
            "schema_version": EPISODE_SCHEMA_VERSION,
            "evaluation_only": True,
            "training_performed": False,
            "scales": {
                "1": {
                    "uav_count": 1,
                    "path": "scale_1",
                    "manifest": "scale_1/manifest.json",
                    "episode_count": len(episode_summaries),
                }
            },
        },
    )


def evaluate_frozen_single_uav_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes_per_group: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate a completed checkpoint without any training update."""

    if episodes_per_group < 1:
        raise ValueError("episodes_per_group must be positive")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    output_root = Path(output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(
                f"diagnostic output is not empty: {output_root}; use --force for this exact output directory"
            )
        # Only clear the exact requested diagnostic root.  The caller is
        # expected to use a new independent output directory; no source
        # experiment path is ever touched here.
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint_sha256_before = _sha256(checkpoint)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    config = _config_from_checkpoint(checkpoint_payload, output_dir=output_root, progress=progress)
    metric_scales = config.metric_scales
    scale_root = output_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    normalization_payload = checkpoint_payload.get("normalization_stats")
    if isinstance(normalization_payload, Mapping):
        _write_json(scale_root / "normalization_stats.json", _jsonable(dict(normalization_payload)))

    trainer, policy_hashes, _ = _load_frozen_trainer(
        checkpoint_payload,
        config,
        metric_scales=metric_scales,
    )
    episode_summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    total_episodes = episodes_per_group * len(START_DISTANCE_GROUPS)
    serial_base = 600_001

    for group_index, group in enumerate(START_DISTANCE_GROUPS):
        distances = (
            _distance_grid(float(group["low_m"]), float(group["high_m"]), episodes_per_group)
            if group["low_m"] is not None
            else [None] * episodes_per_group
        )
        for within_group_index, requested_distance_m in enumerate(distances):
            # The normal group reuses the formal test seed indices 0..19, so
            # its start state is directly comparable with the held-out test
            # generation.  Controlled groups use disjoint test seed indices.
            seed_index = (
                within_group_index
                if group["kind"] == "original_normal_test_start"
                else episodes_per_group + group_index * episodes_per_group + within_group_index
            )
            seed = _scale_seed(config.seed, config.uav_count, "test", seed_index)
            base_scenario = _scenario_for(
                config,
                seed,
                split="test",
                episode_index=seed_index,
                phase="test",
            )
            diagnostic_scenario, actual_start_distance_m, initial_speed_mps = _diagnostic_scenario(
                base_scenario,
                config,
                group,
                requested_distance_m,
            )
            serial = serial_base + len(episode_rows)
            episode_number = len(episode_rows) + 1
            with torch.no_grad():
                summary, _update, trace_rows = _run_episode(
                    config=config,
                    trainer=trainer,
                    scale_root=scale_root,
                    phase="diagnostic_test",
                    split="test",
                    serial=serial,
                    seed=seed,
                    update_policy=False,
                    update_index=None,
                    episode_index=episode_number,
                    episode_total=total_episodes,
                    metric_scales=metric_scales,
                    flush_update=False,
                    scenario_config=diagnostic_scenario,
                )
            episode_path = scale_root / "test" / f"episode_{serial:06d}"
            metric_row = _episode_navigation_metrics(
                episode_path,
                group=group,
                episode_index=within_group_index + 1,
                seed=seed,
                requested_distance_m=requested_distance_m,
                actual_start_distance_m=actual_start_distance_m,
                initial_speed_mps=initial_speed_mps,
            )
            episode_rows.append(metric_row)
            episode_entry = dict(summary)
            episode_entry.update(
                {
                    "evaluation_group": group["name"],
                    "requested_start_distance_m": requested_distance_m,
                    "actual_start_distance_m": actual_start_distance_m,
                    "initial_speed_mps": initial_speed_mps,
                }
            )
            episode_summaries.append(episode_entry)
            # ``trace_rows`` is intentionally not used as a source of model
            # state; it is returned only to preserve the existing runner API.
            del trace_rows

    _write_csv(logs_root / "episode_metrics.csv", _EPISODE_METRIC_COLUMNS, episode_rows)
    group_rows = [_group_summary(group, episode_rows) for group in START_DISTANCE_GROUPS]
    _write_csv(logs_root / "group_summary.csv", _GROUP_SUMMARY_COLUMNS, group_rows)
    plot_path = plots_root / "start_distance_diagnostic.png"
    _plot_group_summary(group_rows, plot_path)
    _write_diagnostic_manifests(
        output_root=output_root,
        scale_root=scale_root,
        config=config,
        episode_summaries=episode_summaries,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256_before,
    )
    validation = validate_episode_benchmark(scale_root)
    checkpoint_sha256_after = _sha256(checkpoint)
    policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
    group_summary = {str(row["group"]): row for row in group_rows}
    summary = {
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256_before": checkpoint_sha256_before,
            "sha256_after": checkpoint_sha256_after,
            "unchanged": checkpoint_sha256_before == checkpoint_sha256_after,
            "policy_hashes": {
                **policy_hashes,
                "after_evaluation": policy_hash_after,
                "policy_unchanged_during_evaluation": (
                    policy_hashes["loaded_policy_hash"] == policy_hash_after
                ),
            },
        },
        "uav_count": 1,
        "episodes_per_group": episodes_per_group,
        "total_episodes": len(episode_rows),
        "groups": group_summary,
        "files": {
            "episode_metrics_csv": str((logs_root / "episode_metrics.csv").resolve()),
            "group_summary_csv": str((logs_root / "group_summary.csv").resolve()),
            "diagnostic_plot": str(plot_path.resolve()),
            "scale_manifest": str((scale_root / "manifest.json").resolve()),
            "root_manifest": str((output_root / "manifest.json").resolve()),
        },
        "schema_validation": validation,
        "read_only_checks": {
            "checkpoint_unchanged": checkpoint_sha256_before == checkpoint_sha256_after,
            "policy_unchanged": policy_hashes["loaded_policy_hash"] == policy_hash_after,
            "no_training_update_call": True,
            "no_transition_added": True,
            "optimizer_state_not_loaded": True,
        },
        "scope": {
            "controlled_groups": ["terminal_1p5_3m", "terminal_3_5m", "terminal_5_8m"],
            "normal_group": "normal_far_distance",
            "curriculum_training_not_reused": True,
            "validation_and_training_splits_generated": False,
            "test_split_only": True,
            "controlled_start_speed_rule": "same nonzero terminal braking-start speed rule as formal curriculum; evaluation-only reuse",
            "distance_sampling": "20 equally spaced inclusive distances per controlled range",
        },
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "START_DISTANCE_GROUPS",
    "evaluate_frozen_single_uav_checkpoint",
]
