"""Navigation-preserving fine-tuning from the single-UAV baseline checkpoint.

The experiment in this module is intentionally separate from the formal
training runner.  It loads a completed baseline checkpoint, verifies that the
frozen policy reproduces the baseline far-distance test, and only then runs a
short continuation with a deterministic 90/10 schedule:

* 90% normal full-distance training starts;
* 10% terminal training starts in the 5--8 m range.

The terminal starts are supplied as scenario overrides.  This keeps the
existing reward, action, dynamics, and PPO/MAPPO implementations unchanged,
while making the fine-tuning schedule explicit and auditable.  Validation and
test always receive the unmodified full-distance scenario.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..learning.mappo import MAPPOConfig
from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.motion_model import braking_distance_target_speed
from ..synthetic.episode_generator import _jsonable, _sha256, _write_csv, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _parameter_vector,
    _run_episode,
    _save_checkpoint,
    _scale_seed,
    _scenario_for,
    _update_root_manifest,
    _write_attention_summary,
    _write_episode_metrics,
    _write_scale_manifest,
    summarize_preliminary,
    validate_preliminary,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _diagnostic_scenario,
    _episode_navigation_metrics,
    _load_frozen_trainer,
    _scenario_start_distance,
    _state_dict_hash,
)


FINE_TUNING_CYCLE_LENGTH = 10
FINE_TUNING_NORMAL_FRACTION = 0.90
FINE_TUNING_TERMINAL_FRACTION = 0.10
FINE_TUNING_TERMINAL_RANGE_M = (5.0, 8.0)

_NORMAL_GROUP: dict[str, Any] = {
    "name": "normal_far_distance",
    "label": "normal far-distance",
    "low_m": None,
    "high_m": None,
    "kind": "original_normal_test_start",
}
_TERMINAL_GROUP: dict[str, Any] = {
    "name": "fine_tune_terminal_5_8m",
    "label": "fine-tune terminal 5-8 m",
    "low_m": 5.0,
    "high_m": 8.0,
    "kind": "controlled_terminal_range",
}

_SCHEDULE_COLUMNS = (
    "split",
    "phase",
    "episode_index",
    "episode_id",
    "seed",
    "episode_type",
    "curriculum_used",
    "curriculum_scope",
    "requested_start_distance_m",
    "actual_start_distance_m",
    "initial_speed_mps",
    "terminal_distance_range_m",
    "terminal_distance_ordinal",
)


def fine_tuning_episode_kind(episode_index: int) -> str:
    """Return the deterministic 90/10 assignment for a zero-based index."""

    if int(episode_index) < 0:
        raise ValueError("episode_index must be non-negative")
    return (
        "terminal"
        if int(episode_index) % FINE_TUNING_CYCLE_LENGTH == 0
        else "normal_far_distance"
    )


def fine_tuning_schedule_counts(episode_count: int) -> dict[str, Any]:
    """Return actual counts and fractions for the fixed 10-episode schedule."""

    if int(episode_count) < 1:
        raise ValueError("episode_count must be positive")
    count = int(episode_count)
    terminal = sum(
        fine_tuning_episode_kind(index) == "terminal" for index in range(count)
    )
    normal = count - terminal
    return {
        "episodes": count,
        "normal_far_distance": normal,
        "terminal": terminal,
        "standard_braking": 0,
        "normal_far_distance_rate": normal / count,
        "terminal_rate": terminal / count,
        "standard_braking_rate": 0.0,
        "cycle_length_episodes": FINE_TUNING_CYCLE_LENGTH,
        "requested_normal_fraction": FINE_TUNING_NORMAL_FRACTION,
        "requested_terminal_fraction": FINE_TUNING_TERMINAL_FRACTION,
    }


def fine_tuning_terminal_distance_values(terminal_count: int) -> list[float]:
    """Return deterministic 5--8 m terminal starts for one fine-tune round."""

    if int(terminal_count) < 1:
        return []
    return [
        float(value)
        for value in np.linspace(
            FINE_TUNING_TERMINAL_RANGE_M[0],
            FINE_TUNING_TERMINAL_RANGE_M[1],
            int(terminal_count),
        )
    ]


def _prepare_output_root(path: Path, *, force: bool) -> None:
    """Prepare one exact experiment directory, never a parent or workspace."""

    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"output directory is not empty: {path}; use --force for this exact directory"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _target_speed_curve_values(
    config: FormalExperimentConfig,
    distances_m: Sequence[float],
    *,
    speed_profile: str,
) -> list[float]:
    return [
        float(
            braking_distance_target_speed(
                distance,
                goal_threshold_m=config.single_uav_goal_threshold_m,
                near_goal_radius_m=config.single_uav_near_goal_radius_m,
                stable_speed_mps=config.single_uav_goal_speed_threshold_mps,
                braking_acceleration_mps2=config.single_uav_braking_acceleration_mps2,
                braking_margin_m=config.single_uav_braking_margin_m,
                far_target_speed_mps=config.single_uav_target_speed_far_mps,
                max_speed_mps=12.0,
                speed_profile=speed_profile,
            )
        )
        for distance in distances_m
    ]


def _plot_target_speed_curve_comparison(
    config: FormalExperimentConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Save an auditable old/new target-speed curve comparison."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    near_radius = max(
        float(config.single_uav_near_goal_radius_m),
        float(config.single_uav_goal_threshold_m),
    )
    maximum_distance = max(12.0, near_radius * 1.5)
    distances = np.linspace(0.0, maximum_distance, 500)
    legacy = _target_speed_curve_values(config, distances, speed_profile="legacy")
    selected = _target_speed_curve_values(
        config,
        distances,
        speed_profile=config.single_uav_braking_distance_speed_profile,
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.plot(distances, legacy, color="#475569", linewidth=2.0, label="legacy")
    axis.plot(
        distances,
        selected,
        color="#2563eb",
        linewidth=2.2,
        label=str(config.single_uav_braking_distance_speed_profile),
    )
    for distance, label in (
        (float(config.single_uav_goal_threshold_m), "goal 1 m"),
        (3.0, "3 m"),
        (5.0, "5 m"),
        (6.0, "6 m"),
        (near_radius, "near-goal boundary"),
    ):
        if 0.0 <= distance <= maximum_distance:
            axis.axvline(distance, color="#94a3b8", linestyle="--", linewidth=0.8)
            axis.text(
                distance,
                0.98,
                label,
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#475569",
            )
    axis.set_xlabel("Estimated goal distance (m)")
    axis.set_ylabel("Target speed (m/s)")
    axis.set_title("Braking-distance target-speed curve: legacy vs selected")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left")
    path = output_dir / "target_speed_curve_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    probe_distances = [
        float(config.single_uav_goal_threshold_m),
        3.0,
        5.0,
        6.0,
        near_radius,
    ]
    return {
        "path": str(path.resolve()),
        "distance_m": probe_distances,
        "legacy_target_speed_mps": _target_speed_curve_values(
            config, probe_distances, speed_profile="legacy"
        ),
        "selected_target_speed_mps": _target_speed_curve_values(
            config,
            probe_distances,
            speed_profile=config.single_uav_braking_distance_speed_profile,
        ),
        "selected_profile": str(config.single_uav_braking_distance_speed_profile),
    }


def _read_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"baseline checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return dict(payload), _sha256(path)


def _baseline_checkpoint_guard(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Fail closed unless the requested original baseline is actually loaded."""

    raw = payload.get("experiment_config")
    if not isinstance(raw, Mapping):
        raise ValueError("baseline checkpoint is missing experiment_config")

    def number(name: str, default: float = float("nan")) -> float:
        try:
            return float(raw.get(name, default))
        except (TypeError, ValueError):
            return float("nan")

    checks = {
        "checkpoint_name_contains_baseline": "single_uav_braking_progress025" in checkpoint_path.as_posix(),
        "uav_count_is_1": int(payload.get("uav_count", raw.get("uav_count", -1))) == 1,
        "learning_mode_is_loc_goal_v2": raw.get("single_uav_learning_mode") == "loc_goal_v2",
        "speed_control_is_braking_distance": raw.get("single_uav_speed_control_mode") == "braking_distance",
        "target_speed_profile_is_legacy": raw.get(
            "single_uav_braking_distance_speed_profile", "legacy"
        ) == "legacy",
        "braking_progress_reward_coef_is_025": math.isclose(
            number("single_uav_braking_progress_reward_coef"), 0.25, rel_tol=0.0, abs_tol=1e-12
        ),
        "terminal_radial_progress_reward_coef_is_zero": math.isclose(
            number("single_uav_terminal_radial_progress_reward_coef", 0.0),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "terminal_curriculum_is_disabled_in_baseline": math.isclose(
            number("single_uav_terminal_braking_curriculum_fraction"), 0.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "goal_hold_reward_is_zero": math.isclose(
            number("single_uav_goal_hold_reward"), 0.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "stable_hold_steps_is_5": int(raw.get("single_uav_stable_hold_steps", -1)) == 5,
        "scenario_seed_is_2026": int(raw.get("scenario_seed", -1)) == 2026,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(
            "checkpoint is not the requested original coef=0.25 baseline: "
            + ", ".join(failed)
        )
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_role": "original_coef025_baseline",
        "checks": checks,
        "experiment_config_output_dir": str(raw.get("output_dir", "")),
        "checkpoint_update_count": int(payload.get("update_count", 0)),
        "checkpoint_episode": int(payload.get("episode", 0)),
    }


def _write_frozen_normalization_artifacts(
    scale_root: Path,
    payload: Mapping[str, Any],
) -> None:
    normalization = payload.get("normalization_stats")
    central = payload.get("central_normalization_stats")
    if isinstance(normalization, Mapping):
        _write_json(
            scale_root / "normalization_stats.json",
            _jsonable(
                {
                    "normalization_stats": dict(normalization),
                    "central_normalization_stats": (
                        dict(central) if isinstance(central, Mapping) else None
                    ),
                    "source": "baseline_checkpoint_frozen_state",
                }
            ),
        )


def _scenario_initial_speed(scenario: Mapping[str, Any]) -> float:
    velocities = scenario.get("initial_velocities", {})
    if not isinstance(velocities, Mapping) or "uav_00" not in velocities:
        return 0.0
    try:
        return float(np.linalg.norm(np.asarray(velocities["uav_00"], dtype=float).reshape(3)))
    except (TypeError, ValueError):
        return 0.0


def _curriculum_free_scenario_config(
    config: FormalExperimentConfig,
) -> FormalExperimentConfig:
    """Disable only automatic start rewriting for a controlled scenario base."""

    return replace(
        config,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
    )


def _normal_full_distance_scenario(
    config: FormalExperimentConfig,
    seed: int,
    *,
    split: str,
    episode_index: int,
    phase: str,
) -> dict[str, Any]:
    scenario = _scenario_for(
        _curriculum_free_scenario_config(config),
        seed,
        split=split,
        episode_index=episode_index,
        phase=phase,
    )
    scenario["fine_tuning_schedule"] = {
        "episode_type": "normal_far_distance",
        "curriculum_used": False,
        "curriculum_scope": "none",
        "full_distance_start_preserved": True,
    }
    return scenario


def _terminal_training_scenario(
    config: FormalExperimentConfig,
    base_scenario: Mapping[str, Any],
    requested_distance_m: float,
    *,
    episode_index: int,
    terminal_ordinal: int,
) -> tuple[dict[str, Any], float, float]:
    scenario, actual_distance, initial_speed = _diagnostic_scenario(
        base_scenario,
        config,
        _TERMINAL_GROUP,
        requested_distance_m,
    )
    # The shared diagnostic helper constructs the same controlled start and
    # initial-speed rule used by the existing terminal interface.  Remove its
    # evaluation-only label before using the scenario for training and replace
    # it with a training-specific contract.
    scenario.pop("evaluation_diagnostic", None)
    scenario["fine_tuning_schedule"] = {
        "episode_type": "terminal",
        "curriculum_used": True,
        "curriculum_scope": "train_only",
        "terminal_distance_range_m": list(FINE_TUNING_TERMINAL_RANGE_M),
        "requested_start_distance_m": float(requested_distance_m),
        "actual_start_distance_m": float(actual_distance),
        "initial_speed_mps": float(initial_speed),
        "initial_velocity_nonzero": bool(initial_speed > 0.0),
        "initial_velocity_direction": "toward_goal",
        "terminal_distance_ordinal": int(terminal_ordinal),
        "episode_index": int(episode_index),
        "full_distance_start_preserved": False,
    }
    return scenario, float(actual_distance), float(initial_speed)


def _episode_navigation_row(
    episode_path: Path,
    *,
    episode_index: int,
    seed: int,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    return _episode_navigation_metrics(
        episode_path,
        group=_NORMAL_GROUP,
        episode_index=episode_index,
        seed=seed,
        requested_distance_m=None,
        actual_start_distance_m=_scenario_start_distance(scenario),
        initial_speed_mps=_scenario_initial_speed(scenario),
    )


def _load_trainable_trainer(
    payload: Mapping[str, Any],
    config: FormalExperimentConfig,
) -> tuple[Any, dict[str, Any]]:
    """Load policy, frozen normalizers, optimizer, and dual state from baseline."""

    trainer, policy_hashes, _ = _load_frozen_trainer(
        payload,
        config,
        metric_scales=config.metric_scales,
    )
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("baseline checkpoint is missing optimizer_state_dict")
    trainer.optimizer.load_state_dict(optimizer_state)
    after_optimizer_state = trainer.optimizer.state_dict()
    loaded_hash = _state_dict_hash(trainer.policy.state_dict())
    checkpoint_policy_state = payload.get("policy_state_dict")
    if not isinstance(checkpoint_policy_state, Mapping):
        raise ValueError("baseline checkpoint is missing policy_state_dict")
    checkpoint_hash = _state_dict_hash(checkpoint_policy_state)
    if loaded_hash != checkpoint_hash:
        raise RuntimeError("loaded fine-tuning policy hash does not match baseline checkpoint")
    # A newly constructed Adam optimizer has no lazily-created per-parameter
    # state before its first step.  Therefore the correct check is equality
    # with the checkpoint after loading, not equality with the empty
    # pre-load state.
    optimizer_loaded = (
        len(after_optimizer_state.get("state", {}))
        == len(optimizer_state.get("state", {}))
        and len(after_optimizer_state.get("param_groups", []))
        == len(optimizer_state.get("param_groups", []))
    )
    if not optimizer_loaded:
        raise RuntimeError("fine-tuning optimizer state was not restored completely")
    trainer._episode_batch_metadata = []
    trainer.policy.train()
    return trainer, {
        "constructed_policy_hash": policy_hashes["constructed_policy_hash"],
        "loaded_policy_hash": loaded_hash,
        "checkpoint_policy_hash": checkpoint_hash,
        "optimizer_loaded": True,
        "optimizer_state_entries": len(optimizer_state.get("state", {})),
    }


def _write_checkpoint_identity(
    scale_root: Path,
    *,
    baseline_guard: Mapping[str, Any],
    baseline_sha256: str,
    policy_identity: Mapping[str, Any],
    phase: str,
) -> None:
    _write_json(
        scale_root / "diagnostics" / "checkpoint_identity.json",
        _jsonable(
            {
                "phase": phase,
                "baseline": dict(baseline_guard),
                "baseline_checkpoint_sha256": baseline_sha256,
                "policy": dict(policy_identity),
            }
        ),
    )


def _evaluation_manifest_and_summary(
    *,
    output_root: Path,
    scale_root: Path,
    config: FormalExperimentConfig,
    episodes: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    scale_manifest = _write_scale_manifest(
        scale_root=scale_root,
        config=config,
        episodes=episodes,
    )
    scale_manifest = {
        **scale_manifest,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
        },
    }
    _write_json(scale_root / "manifest.json", _jsonable(scale_manifest))
    _update_root_manifest(output_root, scale_manifest)
    _write_json(output_root / "summary.json", _jsonable(dict(summary)))
    return scale_manifest


def evaluate_baseline_normal_test(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes: int = 20,
    force: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Read-only normal far-distance evaluation of the original checkpoint."""

    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    checkpoint = Path(checkpoint_path)
    output_root = Path(output_dir)
    _prepare_output_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    logs_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    payload, checkpoint_sha256 = _read_checkpoint(checkpoint)
    baseline_guard = _baseline_checkpoint_guard(payload, checkpoint_path=checkpoint)
    config = _config_from_checkpoint(payload, output_dir=output_root, progress=progress)
    scale_root = output_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    _write_frozen_normalization_artifacts(scale_root, payload)
    trainer, policy_hashes, _ = _load_frozen_trainer(
        payload,
        config,
        metric_scales=config.metric_scales,
    )
    trainer.policy.eval()
    _write_checkpoint_identity(
        scale_root,
        baseline_guard=baseline_guard,
        baseline_sha256=checkpoint_sha256,
        policy_identity={**policy_hashes, "checkpoint_policy_hash": _state_dict_hash(payload["policy_state_dict"])},
        phase="baseline_read_only_verification",
    )

    episode_summaries: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    navigation_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    for index in range(int(episodes)):
        seed = _scale_seed(config.seed, config.uav_count, "test", index)
        scenario = _normal_full_distance_scenario(
            config,
            seed,
            split="test",
            episode_index=index,
            phase="test",
        )
        summary, _update, rows = _run_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            phase="baseline_verification",
            split="test",
            serial=500001 + index,
            seed=seed,
            update_policy=False,
            update_index=None,
            episode_index=index + 1,
            episode_total=int(episodes),
            metric_scales=config.metric_scales,
            flush_update=False,
            scenario_config=scenario,
        )
        episode_summaries.append(summary)
        episode_metrics.append(rows[-1])
        attention_rows.extend(rows[:-1])
        navigation_rows.append(
            _episode_navigation_row(
                scale_root / "test" / f"episode_{500001 + index:06d}",
                episode_index=index + 1,
                seed=seed,
                scenario=scenario,
            )
        )

    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", episode_metrics)
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", attention_rows)
    _write_csv(logs_root / "navigation_metrics.csv", sorted({key for row in navigation_rows for key in row}), navigation_rows)
    analysis = _run_navigation_analysis(scale_root, logs_root=logs_root, plots_root=plots_root)
    test_summary = (
        analysis.get("summary", {}).get("test", {})
        if isinstance(analysis, Mapping)
        else {}
    )
    if not test_summary:
        test_summary = _fallback_navigation_summary(navigation_rows)
    loaded_hash_after = _state_dict_hash(trainer.policy.state_dict())
    checkpoint_hash_after = _sha256(checkpoint)
    policy_unchanged = loaded_hash_after == policy_hashes["loaded_policy_hash"]
    checkpoint_unchanged = checkpoint_hash_after == checkpoint_sha256
    entered8 = int(test_summary.get("entered_near_goal_episodes", 0))
    entered1 = int(test_summary.get("entered_goal_radius_episodes", 0))
    passed = bool(
        policy_unchanged
        and checkpoint_unchanged
        and entered8 >= max(1, int(episodes) - 1)
        and 0 <= entered1 <= 3
    )
    validation_placeholder = {
        "baseline_reproduction_passed": passed,
        "checkpoint_identity": {
            **baseline_guard,
            "sha256": checkpoint_sha256,
            "sha256_unchanged_after_eval": checkpoint_unchanged,
        },
        "policy_identity": {
            **policy_hashes,
            "loaded_policy_hash_after_eval": loaded_hash_after,
            "policy_unchanged_after_eval": policy_unchanged,
        },
        "normal_far_distance_test": test_summary,
        "expected_reference": {
            "entered_near_goal_episodes": "approximately 20/20",
            "entered_goal_radius_episodes": "approximately 1/20",
        },
        "analysis": analysis,
        "output_dir": str(output_root.resolve()),
    }
    _evaluation_manifest_and_summary(
        output_root=output_root,
        scale_root=scale_root,
        config=config,
        episodes=episode_summaries,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        summary=validation_placeholder,
    )
    validation = validate_preliminary(output_root)
    final_summary = {**validation_placeholder, "validate_preliminary": validation}
    _write_json(output_root / "summary.json", _jsonable(final_summary))
    return final_summary


def _run_navigation_analysis(
    scale_root: Path,
    *,
    logs_root: Path,
    plots_root: Path,
) -> dict[str, Any]:
    from .single_uav_navigation_analysis import analyze_single_uav_navigation

    return analyze_single_uav_navigation(
        scale_root,
        output_dir=plots_root / "single_uav_navigation_analysis",
        metrics_output_path=logs_root / "single_uav_navigation_analysis.csv",
        force=True,
    )


def _fallback_navigation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    denominator = max(1, count)
    holds = [int(row.get("maximum_consecutive_goal_hold_steps", 0)) for row in rows]
    return {
        "episode_count": count,
        "entered_near_goal_episodes": int(sum(int(row.get("entered_8m", 0)) for row in rows)),
        "entered_near_goal_rate": sum(int(row.get("entered_8m", 0)) for row in rows) / denominator,
        "entered_goal_radius_episodes": int(sum(int(row.get("entered_1m", 0)) for row in rows)),
        "entered_goal_radius_rate": sum(int(row.get("entered_1m", 0)) for row in rows) / denominator,
        "stable_goal_episodes": int(sum(int(row.get("stable_goal", 0)) for row in rows)),
        "stable_goal_rate": sum(int(row.get("stable_goal", 0)) for row in rows) / denominator,
        "minimum_goal_distance_m": min((float(row["minimum_goal_distance_m"]) for row in rows), default=float("nan")),
        "mean_maximum_consecutive_goal_hold_steps": float(np.mean(holds)) if holds else float("nan"),
        "maximum_consecutive_goal_hold_steps_max": max(holds, default=0),
        "episodes_with_hold_steps_ge_1": sum(value >= 1 for value in holds),
        "episodes_with_hold_steps_ge_3": sum(value >= 3 for value in holds),
        "episodes_with_hold_steps_ge_5": sum(value >= 5 for value in holds),
    }


def _training_schedule_row(
    *,
    split: str,
    phase: str,
    episode_index: int,
    episode_id: str,
    seed: int,
    episode_type: str,
    scenario: Mapping[str, Any],
    requested_distance_m: float | None,
    actual_distance_m: float,
    initial_speed_mps: float,
    terminal_ordinal: int | None,
) -> dict[str, Any]:
    return {
        "split": split,
        "phase": phase,
        "episode_index": int(episode_index),
        "episode_id": episode_id,
        "seed": int(seed),
        "episode_type": episode_type,
        "curriculum_used": int(episode_type == "terminal" and split == "train"),
        "curriculum_scope": "train_only" if episode_type == "terminal" and split == "train" else "none",
        "requested_start_distance_m": requested_distance_m,
        "actual_start_distance_m": float(actual_distance_m),
        "initial_speed_mps": float(initial_speed_mps),
        "terminal_distance_range_m": (
            list(FINE_TUNING_TERMINAL_RANGE_M) if episode_type == "terminal" else None
        ),
        "terminal_distance_ordinal": terminal_ordinal,
        "scenario_fine_tuning_schedule": scenario.get("fine_tuning_schedule", {}),
    }


def build_terminal_radial_progress_reward_audit(
    episode_metrics: Sequence[Mapping[str, Any]],
    config: FormalExperimentConfig,
) -> dict[str, Any]:
    """Summarize the terminal radial-progress diagnostic without affecting learning."""

    def episode_metric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [
            float(row[key])
            for row in rows
            if key in row
            and isinstance(row[key], (int, float, np.integer, np.floating))
            and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else 0.0

    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in episode_metrics if str(row.get("split", "")) == split]
        by_split[split] = {
            "episodes": len(split_rows),
            "mean_active_fraction": episode_metric_mean(
                split_rows, "terminal_radial_progress_active_fraction"
            ),
            "mean_signal": episode_metric_mean(split_rows, "terminal_radial_progress_signal"),
            "mean_speed_gate": episode_metric_mean(
                split_rows, "terminal_radial_progress_speed_gate"
            ),
            "mean_reward": episode_metric_mean(split_rows, "terminal_radial_progress_reward"),
            "mean_reward_total": episode_metric_mean(
                split_rows, "terminal_radial_progress_reward_total"
            ),
            "max_reward": max(
                [
                    float(row.get("terminal_radial_progress_reward_max", 0.0))
                    for row in split_rows
                    if isinstance(
                        row.get("terminal_radial_progress_reward_max", 0.0),
                        (int, float, np.integer, np.floating),
                    )
                ]
                or [0.0]
            ),
            "mean_learning_reward_clip_saturation_fraction": episode_metric_mean(
                split_rows, "learning_reward_clip_saturation_fraction"
            ),
        }
    return {
        "coefficient": float(config.single_uav_terminal_radial_progress_reward_coef),
        "normalization_scale_m": 0.5,
        "speed_gate_scale_mps": 1.0,
        "activation_band_m": [
            float(config.single_uav_goal_threshold_m),
            float(config.single_uav_terminal_braking_radius_m),
        ],
        "by_split": by_split,
        "clip_saturation_is_diagnostic_only": True,
    }


def build_terminal_target_velocity_vector_reward_audit(
    episode_metrics: Sequence[Mapping[str, Any]],
    config: FormalExperimentConfig,
) -> dict[str, Any]:
    """Summarize the vector-correction channel without affecting learning."""

    def episode_metric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [
            float(row[key])
            for row in rows
            if key in row
            and isinstance(row[key], (int, float, np.integer, np.floating))
            and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else 0.0

    coefficient = float(
        config.single_uav_terminal_target_velocity_vector_reward_coef
    )
    by_split: dict[str, dict[str, Any]] = {}
    all_rows: list[Mapping[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_rows = [
            row for row in episode_metrics if str(row.get("split", "")) == split
        ]
        all_rows.extend(split_rows)
        max_rewards = [
            float(row.get("terminal_target_velocity_vector_reward_max", 0.0))
            for row in split_rows
            if isinstance(
                row.get("terminal_target_velocity_vector_reward_max", 0.0),
                (int, float, np.integer, np.floating),
            )
            and np.isfinite(
                float(row.get("terminal_target_velocity_vector_reward_max", 0.0))
            )
        ]
        by_split[split] = {
            "episodes": len(split_rows),
            "episodes_with_active_steps": int(
                sum(
                    episode_metric_mean(
                        [row], "terminal_target_velocity_vector_active_fraction"
                    )
                    > 0.0
                    for row in split_rows
                )
            ),
            "mean_active_fraction": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_active_fraction"
            ),
            "mean_signal": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_signal"
            ),
            "mean_error_pre_mps": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_error_pre_mps"
            ),
            "mean_error_post_mps": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_error_post_mps"
            ),
            "mean_error_reduction_mps": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_error_reduction_mps"
            ),
            "mean_activation_distance_m": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_activation_distance_m"
            ),
            "mean_post_radial_velocity_mps": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_post_radial_velocity_mps"
            ),
            "mean_radial_gate": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_radial_gate"
            ),
            "mean_reward": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_reward"
            ),
            "mean_reward_total": episode_metric_mean(
                split_rows, "terminal_target_velocity_vector_reward_total"
            ),
            "max_reward": max(max_rewards or [0.0]),
            "mean_learning_reward_clip_saturation_fraction": episode_metric_mean(
                split_rows, "learning_reward_clip_saturation_fraction"
            ),
        }

    observed_max_reward = max(
        [
            float(row.get("terminal_target_velocity_vector_reward_max", 0.0))
            for row in all_rows
            if isinstance(
                row.get("terminal_target_velocity_vector_reward_max", 0.0),
                (int, float, np.integer, np.floating),
            )
            and np.isfinite(
                float(row.get("terminal_target_velocity_vector_reward_max", 0.0))
            )
        ]
        or [0.0]
    )
    return {
        "coefficient": coefficient,
        "activation_band_m": [
            float(config.single_uav_goal_threshold_m),
            float(config.single_uav_terminal_braking_radius_m),
        ],
        "error_scale_mps": 0.5,
        "positive_only": True,
        "radial_gate_enabled": bool(
            config.single_uav_terminal_target_velocity_vector_radial_gate_enabled
        ),
        "activation_distance_source": (
            "previous_estimated_goal_distance_m"
            if config.single_uav_terminal_target_velocity_vector_radial_gate_enabled
            else "current_goal_distance_m"
        ),
        "single_step_reward_upper_bound": max(0.0, coefficient),
        "observed_max_reward": observed_max_reward,
        "reward_bound_ok": bool(
            observed_max_reward <= max(0.0, coefficient) + 1e-9
        ),
        "by_split": by_split,
        "clip_saturation_is_diagnostic_only": True,
    }


def build_terminal_target_velocity_vector_step_audit(
    scale_root: Path,
    config: FormalExperimentConfig,
) -> dict[str, Any]:
    """Validate the vector channel against its step-level distance contract."""

    threshold = float(config.single_uav_goal_threshold_m)
    terminal_radius = float(config.single_uav_terminal_braking_radius_m)
    coefficient = float(
        config.single_uav_terminal_target_velocity_vector_reward_coef
    )
    by_split: dict[str, dict[str, Any]] = {}
    active_outside_band = 0
    reward_bound_violations = 0
    invalid_signal_rows = 0
    negative_reward_rows = 0
    total_rows = 0
    active_rows = 0
    positive_reward_rows = 0
    positive_reward_with_nonpositive_radial_rows = 0
    radial_gate_range_violations = 0
    observed_max_reward = 0.0
    radial_gate_enabled = bool(
        config.single_uav_terminal_target_velocity_vector_radial_gate_enabled
    )
    for split in ("train", "validation", "test"):
        split_total = 0
        split_active = 0
        split_positive = 0
        split_outside = 0
        split_max_reward = 0.0
        split_positive_with_nonpositive_radial = 0
        split_gate_range_violations = 0
        for rewards_path in sorted(
            (scale_root / split).glob("episode_*/rewards.csv")
        ):
            with rewards_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for raw_row in csv.DictReader(handle):
                    split_total += 1
                    total_rows += 1

                    def number(name: str, default: float = 0.0) -> float:
                        try:
                            value = float(raw_row.get(name, default))
                        except (TypeError, ValueError):
                            return float(default)
                        return value if np.isfinite(value) else float(default)

                    distance = number(
                        "single_uav_terminal_target_velocity_vector_activation_distance_m",
                        float("nan"),
                    )
                    if not np.isfinite(distance):
                        distance = number("current_goal_distance", float("nan"))
                    active = number(
                        "single_uav_terminal_target_velocity_vector_active", 0.0
                    ) > 0.5
                    reward = number(
                        "single_uav_terminal_target_velocity_vector_reward", 0.0
                    )
                    signal = number(
                        "single_uav_terminal_target_velocity_vector_signal", 0.0
                    )
                    post_radial_velocity = number(
                        "single_uav_terminal_target_velocity_vector_post_radial_velocity_mps",
                        0.0,
                    )
                    radial_gate = number(
                        "single_uav_terminal_target_velocity_vector_radial_gate", 0.0
                    )
                    split_max_reward = max(split_max_reward, reward)
                    observed_max_reward = max(observed_max_reward, reward)
                    if active:
                        split_active += 1
                        active_rows += 1
                        in_band = (
                            np.isfinite(distance)
                            and distance > threshold
                            and distance <= terminal_radius
                        )
                        if not in_band:
                            split_outside += 1
                            active_outside_band += 1
                    if reward > 1e-12:
                        split_positive += 1
                        positive_reward_rows += 1
                        if radial_gate_enabled and post_radial_velocity <= 1e-9:
                            split_positive_with_nonpositive_radial += 1
                            positive_reward_with_nonpositive_radial_rows += 1
                    if reward < -1e-12:
                        negative_reward_rows += 1
                    if reward > max(0.0, coefficient) + 1e-9:
                        reward_bound_violations += 1
                    if signal < -1e-9 or signal > 1.0 + 1e-9:
                        invalid_signal_rows += 1
                    if radial_gate_enabled and (
                        radial_gate < -1e-9 or radial_gate > 1.0 + 1e-9
                    ):
                        split_gate_range_violations += 1
                        radial_gate_range_violations += 1
        by_split[split] = {
            "rows": split_total,
            "active_rows": split_active,
            "positive_reward_rows": split_positive,
            "active_outside_band": split_outside,
            "positive_reward_with_nonpositive_radial_rows": (
                split_positive_with_nonpositive_radial
            ),
            "radial_gate_range_violations": split_gate_range_violations,
            "max_reward": split_max_reward,
        }
    return {
        "total_rows": total_rows,
        "active_rows": active_rows,
        "positive_reward_rows": positive_reward_rows,
        "positive_vector_reward_step_count": positive_reward_rows,
        "positive_vector_reward_with_nonpositive_radial_step_count": (
            positive_reward_with_nonpositive_radial_rows
        ),
        "positive_vector_reward_with_nonpositive_radial_fraction": float(
            positive_reward_with_nonpositive_radial_rows
            / max(1, positive_reward_rows)
        ),
        "active_outside_band": active_outside_band,
        "reward_bound_violations": reward_bound_violations,
        "negative_reward_rows": negative_reward_rows,
        "invalid_signal_rows": invalid_signal_rows,
        "radial_gate_range_violations": radial_gate_range_violations,
        "observed_max_reward": observed_max_reward,
        "reward_bound_ok": reward_bound_violations == 0,
        "activation_band_ok": active_outside_band == 0,
        "nonnegative_reward_ok": negative_reward_rows == 0,
        "signal_bounded_ok": invalid_signal_rows == 0,
        "radial_gate_range_ok": radial_gate_range_violations == 0,
        "positive_reward_radial_alignment_ok": (
            positive_reward_with_nonpositive_radial_rows == 0
        ),
        "by_split": by_split,
        "activation_band_m": [threshold, terminal_radius],
        "activation_distance_source": (
            "previous_estimated_goal_distance_m"
            if radial_gate_enabled
            else "current_goal_distance_m"
        ),
        "radial_gate_enabled": radial_gate_enabled,
        "coefficient": coefficient,
    }


def refresh_terminal_radial_progress_reward_audit(
    output_dir: str | Path,
    *,
    coefficient: float | None = None,
) -> dict[str, Any]:
    """Repair/recompute the audit fields from an existing run, without training."""

    root = Path(output_dir)
    summary_path = root / "summary.json"
    metrics_path = root / "scale_1" / "diagnostics" / "episode_metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing summary or episode metrics under {root}")
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    numeric_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw_row.items():
            if key == "split":
                row[key] = value
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
        numeric_rows.append(row)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prior_audit = (
        summary.get("fine_tuning", {}).get("terminal_radial_progress_reward", {})
        if isinstance(summary.get("fine_tuning"), Mapping)
        else {}
    )
    selected_coefficient = float(
        coefficient
        if coefficient is not None
        else prior_audit.get("coefficient", 0.0)
    )
    config = type(
        "TerminalRewardAuditConfig",
        (),
        {
            "single_uav_terminal_radial_progress_reward_coef": selected_coefficient,
            "single_uav_goal_threshold_m": float(
                (prior_audit.get("activation_band_m") or [1.0, 3.0])[0]
            ),
            "single_uav_terminal_braking_radius_m": float(
                (prior_audit.get("activation_band_m") or [1.0, 3.0])[1]
            ),
        },
    )()
    audit = build_terminal_radial_progress_reward_audit(numeric_rows, config)
    if not isinstance(summary.get("fine_tuning"), Mapping):
        raise ValueError(f"summary has no fine_tuning section: {summary_path}")
    summary["fine_tuning"]["terminal_radial_progress_reward"] = audit
    summary["fine_tuning"]["reward_audit_refreshed_without_retraining"] = True
    _write_json(summary_path, _jsonable(summary))
    manifest_path = root / "scale_1" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest.get("fine_tuning"), Mapping):
            manifest["fine_tuning"]["terminal_radial_progress_reward"] = audit
            manifest["fine_tuning"]["reward_audit_refreshed_without_retraining"] = True
            _write_json(manifest_path, _jsonable(manifest))
    return audit


def _run_fine_tuning_round(
    payload: Mapping[str, Any],
    *,
    baseline_checkpoint: Path,
    baseline_guard: Mapping[str, Any],
    baseline_sha256: str,
    output_dir: Path,
    logs_dir: Path,
    plots_dir: Path,
    train_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    force: bool,
    progress: bool,
    label: str,
    speed_profile: str = "legacy",
    terminal_radial_progress_reward_coef: float = 0.0,
    terminal_target_velocity_vector_reward_coef: float = 0.0,
    terminal_target_velocity_vector_radial_gate_enabled: bool = False,
) -> dict[str, Any]:
    if int(train_episodes) < 1 or int(validation_episodes) < 1 or int(test_episodes) < 1:
        raise ValueError("fine-tuning train, validation, and test episode counts must be positive")
    _prepare_output_root(output_dir, force=force)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_config = _config_from_checkpoint(payload, output_dir=output_dir, progress=progress)
    config = replace(
        base_config,
        train_episodes=int(train_episodes),
        validation_episodes=int(validation_episodes),
        test_episodes=int(test_episodes),
        # The active fine-tuning schedule is the requested 90/10 split.  The
        # reward and every PPO/MAPPO field still come directly from baseline.
        single_uav_braking_curriculum_fraction=FINE_TUNING_TERMINAL_FRACTION,
        single_uav_terminal_braking_curriculum_fraction=FINE_TUNING_TERMINAL_FRACTION,
        single_uav_braking_distance_speed_profile=str(speed_profile),
        single_uav_terminal_radial_progress_reward_coef=float(
            terminal_radial_progress_reward_coef
        ),
        single_uav_terminal_target_velocity_vector_reward_coef=float(
            terminal_target_velocity_vector_reward_coef
        ),
        single_uav_terminal_target_velocity_vector_radial_gate_enabled=bool(
            terminal_target_velocity_vector_radial_gate_enabled
        ),
        command_line=f"navigation-preserving terminal fine-tuning ({label})",
        force=True,
    )
    if config.uav_count != 1:
        raise ValueError("navigation-preserving fine-tuning requires a 1-UAV baseline")
    scale_root = output_dir / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    _copy_calibration_files(scale_root, config)
    _write_frozen_normalization_artifacts(scale_root, payload)
    target_speed_curve = _plot_target_speed_curve_comparison(
        config,
        output_dir=plots_dir,
    )

    trainer, policy_identity = _load_trainable_trainer(payload, config)
    _write_checkpoint_identity(
        scale_root,
        baseline_guard=baseline_guard,
        baseline_sha256=baseline_sha256,
        policy_identity=policy_identity,
        phase=f"fine_tuning_{label}_before_training",
    )
    initial_parameters = _parameter_vector(trainer)
    baseline_update_count = int(payload.get("update_count", 0))
    schedule_counts = fine_tuning_schedule_counts(int(train_episodes))
    terminal_distances = fine_tuning_terminal_distance_values(
        int(schedule_counts["terminal"])
    )
    terminal_ordinal = 0
    episode_summaries: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    scenario_base_config = _curriculum_free_scenario_config(config)
    trainer._episode_batch_metadata = []

    for index in range(int(train_episodes)):
        episode_type = fine_tuning_episode_kind(index)
        seed = _scale_seed(config.seed, config.uav_count, "train", index)
        base_scenario = _scenario_for(
            scenario_base_config,
            seed,
            split="train",
            episode_index=index,
            phase="train",
        )
        requested_distance_m: float | None = None
        if episode_type == "terminal":
            requested_distance_m = terminal_distances[terminal_ordinal]
            scenario, actual_distance, initial_speed = _terminal_training_scenario(
                config,
                base_scenario,
                requested_distance_m,
                episode_index=index + 1,
                terminal_ordinal=terminal_ordinal,
            )
            terminal_ordinal += 1
        else:
            scenario = _normal_full_distance_scenario(
                config,
                seed,
                split="train",
                episode_index=index,
                phase="train",
            )
            actual_distance = _scenario_start_distance(scenario)
            initial_speed = _scenario_initial_speed(scenario)
        serial = index + 1
        update_index = baseline_update_count + index // int(config.rollout_episodes_per_update) + 1
        flush_update = (
            (index + 1) % int(config.rollout_episodes_per_update) == 0
            or index + 1 == int(train_episodes)
        )
        summary, _stats, rows = _run_episode(
            config=config,
            trainer=trainer,
            scale_root=scale_root,
            phase="fine_tune_train",
            split="train",
            serial=serial,
            seed=seed,
            update_policy=True,
            update_index=update_index,
            episode_index=index + 1,
            episode_total=int(train_episodes),
            metric_scales=config.metric_scales,
            flush_update=flush_update,
            scenario_config=scenario,
        )
        episode_id = f"episode_{serial:06d}"
        summary["fine_tuning_episode_type"] = episode_type
        summary["fine_tuning_curriculum_used"] = bool(episode_type == "terminal")
        episode_summaries.append(summary)
        episode_metrics.append(rows[-1])
        attention_rows.extend(rows[:-1])
        schedule_rows.append(
            _training_schedule_row(
                split="train",
                phase="fine_tune_train",
                episode_index=index + 1,
                episode_id=episode_id,
                seed=seed,
                episode_type=episode_type,
                scenario=scenario,
                requested_distance_m=requested_distance_m,
                actual_distance_m=actual_distance,
                initial_speed_mps=initial_speed,
                terminal_ordinal=(terminal_ordinal - 1 if episode_type == "terminal" else None),
            )
        )

    update_count = baseline_update_count + math.ceil(
        int(train_episodes) / int(config.rollout_episodes_per_update)
    )
    checkpoint_path = scale_root / "checkpoints" / "final.pt"
    _save_checkpoint(checkpoint_path, trainer, config, update_count)
    final_policy_hash = _state_dict_hash(trainer.policy.state_dict())
    parameter_delta = {
        "initial_l2": float(initial_parameters.norm().cpu()),
        "final_l2": float(_parameter_vector(trainer).norm().cpu()),
        "delta_l2": float((_parameter_vector(trainer) - initial_parameters).norm().cpu()),
        "baseline_update_count": baseline_update_count,
        "fine_tuning_updates": update_count - baseline_update_count,
        "final_update_count": update_count,
    }
    if parameter_delta["delta_l2"] <= 0.0:
        raise RuntimeError("fine-tuning parameters did not change from the baseline checkpoint")
    _write_json(scale_root / "diagnostics" / "parameter_delta.json", _jsonable(parameter_delta))

    trainer.policy.eval()
    validation_episodes_summary: list[dict[str, Any]] = []
    test_episodes_summary: list[dict[str, Any]] = []
    test_navigation_rows: list[dict[str, Any]] = []
    for split, phase, count, serial_base, destination in (
        ("validation", "validation", int(validation_episodes), 400001, validation_episodes_summary),
        ("test", "test", int(test_episodes), 500001, test_episodes_summary),
    ):
        for index in range(count):
            seed = _scale_seed(config.seed, config.uav_count, split, index)
            scenario = _normal_full_distance_scenario(
                config,
                seed,
                split=split,
                episode_index=index,
                phase=phase,
            )
            serial = serial_base + index
            summary, _stats, rows = _run_episode(
                config=config,
                trainer=trainer,
                scale_root=scale_root,
                phase=phase,
                split=split,
                serial=serial,
                seed=seed,
                update_policy=False,
                update_index=None,
                episode_index=index + 1,
                episode_total=count,
                metric_scales=config.metric_scales,
                flush_update=False,
                scenario_config=scenario,
            )
            summary["fine_tuning_episode_type"] = "normal_far_distance"
            summary["fine_tuning_curriculum_used"] = False
            destination.append(summary)
            episode_metrics.append(rows[-1])
            attention_rows.extend(rows[:-1])
            schedule_rows.append(
                _training_schedule_row(
                    split=split,
                    phase=phase,
                    episode_index=index + 1,
                    episode_id=f"episode_{serial:06d}",
                    seed=seed,
                    episode_type="normal_far_distance",
                    scenario=scenario,
                    requested_distance_m=None,
                    actual_distance_m=_scenario_start_distance(scenario),
                    initial_speed_mps=_scenario_initial_speed(scenario),
                    terminal_ordinal=None,
                )
            )
            if split == "test":
                test_navigation_rows.append(
                    _episode_navigation_row(
                        scale_root / "test" / f"episode_{serial:06d}",
                        episode_index=index + 1,
                        seed=seed,
                        scenario=scenario,
                    )
                )

    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", episode_metrics)
    _write_episode_metrics(
        scale_root / "diagnostics" / "validation_episode_metrics.csv",
        [row for row in episode_metrics if str(row.get("split", "")) == "validation"],
    )
    _write_attention_summary(scale_root / "diagnostics" / "attention_summary.csv", attention_rows)
    _write_csv(scale_root / "diagnostics" / "fine_tuning_episode_schedule.csv", _SCHEDULE_COLUMNS, schedule_rows)
    _write_csv(logs_dir / "fine_tuning_episode_schedule.csv", _SCHEDULE_COLUMNS, schedule_rows)
    terminal_radial_reward_audit = build_terminal_radial_progress_reward_audit(
        episode_metrics,
        config,
    )
    terminal_target_velocity_vector_reward_audit = (
        build_terminal_target_velocity_vector_reward_audit(episode_metrics, config)
    )
    terminal_target_velocity_vector_step_audit = (
        build_terminal_target_velocity_vector_step_audit(scale_root, config)
    )
    all_episodes = episode_summaries + validation_episodes_summary + test_episodes_summary
    scale_manifest = _write_scale_manifest(scale_root=scale_root, config=config, episodes=all_episodes)
    scale_manifest = {
        **scale_manifest,
        "fine_tuning": {
            "source_checkpoint": {
                "path": str(baseline_checkpoint.resolve()),
                "sha256": baseline_sha256,
            },
            "schedule": schedule_counts,
            "terminal_distance_policy": {
                "range_m": list(FINE_TUNING_TERMINAL_RANGE_M),
                "values_m": terminal_distances,
                "initial_velocity_nonzero_required": True,
            },
            "target_speed_curve": target_speed_curve,
            "terminal_radial_progress_reward": terminal_radial_reward_audit,
            "terminal_target_velocity_vector_reward": {
                **terminal_target_velocity_vector_reward_audit,
                "target_vector_frozen_pre_action": True,
                "estimated_state_only": True,
                "radial_gate_enabled": bool(
                    config.single_uav_terminal_target_velocity_vector_radial_gate_enabled
                ),
                "step_level_contract_audit": terminal_target_velocity_vector_step_audit,
            },
            "scope": {
                "train": "90% normal full-distance + 10% terminal 5-8 m",
                "validation": "normal full-distance only; no curriculum",
                "test": "normal full-distance only; no curriculum",
            },
            "warmup": "not rerun; baseline train-fitted normalizers and potential statistics were loaded",
            "reward_structure_changed": bool(
                float(config.single_uav_terminal_radial_progress_reward_coef) != 0.0
                or float(
                    config.single_uav_terminal_target_velocity_vector_reward_coef
                )
                != 0.0
            ),
            "action_space_changed": False,
            "dynamics_changed": False,
            "ppo_mappo_parameters_changed": False,
            "goal_hold_reward": float(config.single_uav_goal_hold_reward),
            "stable_hold_steps": int(config.single_uav_stable_hold_steps),
        },
    }
    _write_json(scale_root / "manifest.json", _jsonable(scale_manifest))
    _update_root_manifest(output_dir, scale_manifest)

    preliminary_summary = summarize_preliminary(
        output_dir,
        logs_output=logs_dir,
        plots_output=plots_dir,
    )
    # ``summarize_preliminary`` exposes the navigation-analysis paths in its
    # figure contract, but not the nested analysis summary itself.  Re-read
    # the already-generated episode logs so the final fine-tuning report keeps
    # the near-goal speed and hold diagnostics instead of silently dropping
    # those fields.
    navigation_analysis = _run_navigation_analysis(
        scale_root,
        logs_root=logs_dir,
        plots_root=plots_dir,
    )
    test_navigation_summary = (
        navigation_analysis.get("summary", {}).get("test", {})
        if isinstance(navigation_analysis, Mapping)
        else {}
    )
    if not test_navigation_summary:
        test_navigation_summary = _fallback_navigation_summary(test_navigation_rows)
    test_schedule_rows = [row for row in schedule_rows if row["split"] == "test"]
    validation_schedule_rows = [row for row in schedule_rows if row["split"] == "validation"]
    train_schedule_rows = [row for row in schedule_rows if row["split"] == "train"]
    final_checkpoint_sha256 = _sha256(checkpoint_path)
    final_summary: dict[str, Any] = {
        **preliminary_summary,
        "fine_tuning": {
            "label": label,
            "source_checkpoint": {
                "path": str(baseline_checkpoint.resolve()),
                "sha256": baseline_sha256,
                "role": "original_coef025_baseline",
            },
            "output_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": final_checkpoint_sha256,
                "final_policy_hash": final_policy_hash,
            },
            "checkpoint_identity": policy_identity,
            "schedule_requested": {
                "normal_far_distance_fraction": FINE_TUNING_NORMAL_FRACTION,
                "terminal_fraction": FINE_TUNING_TERMINAL_FRACTION,
                "standard_braking_fraction": 0.0,
            },
            "schedule_actual": schedule_counts,
            "terminal_distance_values_m": terminal_distances,
            "target_speed_curve": target_speed_curve,
            "terminal_radial_progress_reward": terminal_radial_reward_audit,
            "terminal_target_velocity_vector_reward": {
                **terminal_target_velocity_vector_reward_audit,
                "target_vector_frozen_pre_action": True,
                "estimated_state_only": True,
                "radial_gate_enabled": bool(
                    config.single_uav_terminal_target_velocity_vector_radial_gate_enabled
                ),
                "step_level_contract_audit": terminal_target_velocity_vector_step_audit,
            },
            "train_schedule_rows": len(train_schedule_rows),
            "validation_test_curriculum_check": {
                "validation_episode_count": len(validation_schedule_rows),
                "test_episode_count": len(test_schedule_rows),
                "validation_all_normal_far_distance": all(
                    row["episode_type"] == "normal_far_distance"
                    and int(row["curriculum_used"]) == 0
                    for row in validation_schedule_rows
                ),
                "test_all_normal_far_distance": all(
                    row["episode_type"] == "normal_far_distance"
                    and int(row["curriculum_used"]) == 0
                    for row in test_schedule_rows
                ),
                "terminal_rows_outside_train": sum(
                    int(row["episode_type"] == "terminal")
                    for row in schedule_rows
                    if row["split"] != "train"
                ),
            },
            "parameter_delta": parameter_delta,
            "test_navigation": test_navigation_summary,
            "baseline_guard": dict(baseline_guard),
            "reward_and_mechanism_changes": {
                "reward_structure_changed": bool(
                    float(config.single_uav_terminal_radial_progress_reward_coef) != 0.0
                    or float(
                        config.single_uav_terminal_target_velocity_vector_reward_coef
                    )
                    != 0.0
                ),
                "target_speed_curve_changed": str(
                    config.single_uav_braking_distance_speed_profile
                )
                != "legacy",
                "target_speed_profile": str(
                    config.single_uav_braking_distance_speed_profile
                ),
                "terminal_radial_progress_reward_coef": float(
                    config.single_uav_terminal_radial_progress_reward_coef
                ),
                "terminal_target_velocity_vector_reward_coef": float(
                    config.single_uav_terminal_target_velocity_vector_reward_coef
                ),
                "terminal_target_velocity_vector_radial_gate_enabled": bool(
                    config.single_uav_terminal_target_velocity_vector_radial_gate_enabled
                ),
                "goal_hold_reward": float(config.single_uav_goal_hold_reward),
                "stable_hold_steps": int(config.single_uav_stable_hold_steps),
                "action_space_changed": False,
                "dynamics_changed": False,
                "ppo_mappo_parameters_changed": False,
                "safety_projection_enabled": bool(config.single_uav_braking_safety_projection_enabled),
            },
        },
    }
    _write_json(output_dir / "summary.json", _jsonable(final_summary))
    validation = validate_preliminary(output_dir)
    final_summary["validate_preliminary"] = validation
    _write_json(output_dir / "summary.json", _jsonable(final_summary))
    return final_summary


def run_navigation_preserving_finetune(
    baseline_checkpoint: str | Path,
    *,
    baseline_verify_output: str | Path,
    baseline_verify_logs: str | Path | None = None,
    baseline_verify_plots: str | Path | None = None,
    smoke_output: str | Path,
    smoke_logs: str | Path | None = None,
    smoke_plots: str | Path | None = None,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    smoke_train_episodes: int = 10,
    smoke_validation_episodes: int = 2,
    smoke_test_episodes: int = 2,
    train_episodes: int = 100,
    validation_episodes: int = 20,
    test_episodes: int = 20,
    speed_profile: str = "legacy",
    terminal_radial_progress_reward_coef: float = 0.0,
    terminal_target_velocity_vector_reward_coef: float = 0.0,
    terminal_target_velocity_vector_radial_gate_enabled: bool = False,
    force: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Verify the baseline, run smoke, then run the finite full fine-tune."""

    checkpoint = Path(baseline_checkpoint)
    baseline = evaluate_baseline_normal_test(
        checkpoint,
        output_dir=baseline_verify_output,
        logs_dir=baseline_verify_logs,
        plots_dir=baseline_verify_plots,
        episodes=20,
        force=force,
        progress=progress,
    )
    if not bool(baseline.get("baseline_reproduction_passed", False)):
        return {
            "status": "baseline_verification_failed",
            "baseline_verification": baseline,
            "fine_tuning_started": False,
        }
    payload, checkpoint_sha256 = _read_checkpoint(checkpoint)
    baseline_guard = _baseline_checkpoint_guard(payload, checkpoint_path=checkpoint)
    smoke_summary = _run_fine_tuning_round(
        payload,
        baseline_checkpoint=checkpoint,
        baseline_guard=baseline_guard,
        baseline_sha256=checkpoint_sha256,
        output_dir=Path(smoke_output),
        logs_dir=(Path(smoke_logs) if smoke_logs is not None else Path("logs/preliminary") / Path(smoke_output).name),
        plots_dir=(Path(smoke_plots) if smoke_plots is not None else Path("plots/preliminary") / Path(smoke_output).name),
        train_episodes=int(smoke_train_episodes),
        validation_episodes=int(smoke_validation_episodes),
        test_episodes=int(smoke_test_episodes),
        force=force,
        progress=progress,
        label="smoke",
        speed_profile=speed_profile,
        terminal_radial_progress_reward_coef=terminal_radial_progress_reward_coef,
        terminal_target_velocity_vector_reward_coef=(
            terminal_target_velocity_vector_reward_coef
        ),
        terminal_target_velocity_vector_radial_gate_enabled=(
            terminal_target_velocity_vector_radial_gate_enabled
        ),
    )
    smoke_validation = smoke_summary.get("validate_preliminary", {})
    smoke_fine_tuning = smoke_summary.get("fine_tuning", {})
    smoke_vector_audit = smoke_fine_tuning.get(
        "terminal_target_velocity_vector_reward", {}
    )
    smoke_vector_step_audit = smoke_vector_audit.get(
        "step_level_contract_audit", {}
    )
    smoke_vector_band = smoke_vector_audit.get("activation_band_m", [])
    smoke_checks = {
        "validate_preliminary_ok": bool(smoke_validation.get("ok", False)),
        "baseline_policy_hash_matches": smoke_fine_tuning.get("checkpoint_identity", {}).get("loaded_policy_hash")
        == _state_dict_hash(payload["policy_state_dict"]),
        "parameters_changed_after_smoke": float(
            smoke_fine_tuning.get("parameter_delta", {}).get("delta_l2", 0.0)
        ) > 0.0,
        "schedule_is_90_10": (
            smoke_fine_tuning.get("schedule_actual", {}).get("normal_far_distance")
            == fine_tuning_schedule_counts(int(smoke_train_episodes))["normal_far_distance"]
            and smoke_fine_tuning.get("schedule_actual", {}).get("terminal")
            == fine_tuning_schedule_counts(int(smoke_train_episodes))["terminal"]
        ),
        "validation_test_curriculum_free": (
            smoke_fine_tuning
            .get("validation_test_curriculum_check", {})
            .get("validation_all_normal_far_distance", False)
            and smoke_fine_tuning
            .get("validation_test_curriculum_check", {})
            .get("test_all_normal_far_distance", False)
        ),
        "terminal_rows_outside_train_zero": smoke_fine_tuning
        .get("validation_test_curriculum_check", {})
        .get("terminal_rows_outside_train", 1)
        == 0,
        "vector_coefficient_matches_requested": math.isclose(
            float(smoke_vector_audit.get("coefficient", float("nan"))),
            float(terminal_target_velocity_vector_reward_coef),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "vector_activation_band_is_1_to_3m": (
            len(smoke_vector_band) == 2
            and math.isclose(float(smoke_vector_band[0]), 1.0, abs_tol=1e-12)
            and math.isclose(float(smoke_vector_band[1]), 3.0, abs_tol=1e-12)
        ),
        "vector_radial_gate_matches_requested": bool(
            smoke_vector_audit.get("radial_gate_enabled", False)
        )
        == bool(terminal_target_velocity_vector_radial_gate_enabled),
        "vector_step_activation_band_verified": bool(
            smoke_vector_step_audit.get("activation_band_ok", False)
        ),
        "vector_reward_nonnegative_and_bounded": bool(
            smoke_vector_step_audit.get("nonnegative_reward_ok", False)
            and smoke_vector_step_audit.get("reward_bound_ok", False)
            and smoke_vector_step_audit.get("signal_bounded_ok", False)
            and smoke_vector_step_audit.get("radial_gate_range_ok", True)
            and (
                not terminal_target_velocity_vector_radial_gate_enabled
                or smoke_vector_step_audit.get(
                    "positive_reward_radial_alignment_ok", False
                )
            )
        ),
        "vector_reward_audit_finite": bool(
            np.isfinite(
                float(smoke_vector_step_audit.get("observed_max_reward", 0.0))
            )
        ),
        "clip_saturation_reasonable": all(
            float(split_audit.get("mean_learning_reward_clip_saturation_fraction", 0.0))
            <= 0.50
            for split_audit in smoke_vector_audit.get("by_split", {}).values()
            if isinstance(split_audit, Mapping)
        ),
    }
    smoke_passed = all(bool(value) for value in smoke_checks.values())
    if not smoke_passed:
        return {
            "status": "smoke_verification_failed",
            "baseline_verification": baseline,
            "smoke": {**smoke_summary, "smoke_checks": smoke_checks},
            "fine_tuning_started": False,
        }
    full_summary = _run_fine_tuning_round(
        payload,
        baseline_checkpoint=checkpoint,
        baseline_guard=baseline_guard,
        baseline_sha256=checkpoint_sha256,
        output_dir=Path(output_dir),
        logs_dir=(Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / Path(output_dir).name),
        plots_dir=(Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / Path(output_dir).name),
        train_episodes=int(train_episodes),
        validation_episodes=int(validation_episodes),
        test_episodes=int(test_episodes),
        force=force,
        progress=progress,
        label="full",
        speed_profile=speed_profile,
        terminal_radial_progress_reward_coef=terminal_radial_progress_reward_coef,
        terminal_target_velocity_vector_reward_coef=(
            terminal_target_velocity_vector_reward_coef
        ),
        terminal_target_velocity_vector_radial_gate_enabled=(
            terminal_target_velocity_vector_radial_gate_enabled
        ),
    )
    full_validation = full_summary.get("validate_preliminary", {})
    return {
        "status": "completed",
        "baseline_verification": baseline,
        "smoke": {**smoke_summary, "smoke_checks": smoke_checks},
        "full_fine_tuning": full_summary,
        "full_validate_preliminary_ok": bool(full_validation.get("ok", False)),
        "fine_tuning_started": True,
    }


__all__ = [
    "FINE_TUNING_CYCLE_LENGTH",
    "FINE_TUNING_NORMAL_FRACTION",
    "FINE_TUNING_TERMINAL_FRACTION",
    "FINE_TUNING_TERMINAL_RANGE_M",
    "evaluate_baseline_normal_test",
    "fine_tuning_episode_kind",
    "fine_tuning_schedule_counts",
    "fine_tuning_terminal_distance_values",
    "build_terminal_radial_progress_reward_audit",
    "build_terminal_target_velocity_vector_reward_audit",
    "build_terminal_target_velocity_vector_step_audit",
    "refresh_terminal_radial_progress_reward_audit",
    "run_navigation_preserving_finetune",
]
