"""Paired independent-flight baseline for the preliminary MAPPO experiment.

The baseline deliberately changes only the controller.  It replays recorded
validation or test scenarios from one three-UAV reference run, disables cooperative UWB
updates, uses fixed equal-share bandwidth, and steers each UAV from its local
EKF estimate.  Episode is the independent replication unit; time samples are
never treated as independent replicates in the paired summary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..simulation.motion_model import MotionConfig, clip_vector_norm
from ..synthetic.episode_generator import _write_csv, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _episode_metrics,
    _make_environment,
    _write_episode_metrics,
)
from .pilot import _EpisodeTraceWriter, _assert_action_feasible, _assert_observation_finite, _seed_everything
from .plotting import _safety_summary


POLICY_SOURCE = "independent_fixed_baseline"


@dataclass(frozen=True)
class IndependentBaselineConfig:
    """Configuration for replaying one frozen reference evaluation split."""

    reference_scale: Path
    output_dir: Path
    split: str = "test"
    controller: str = "independent-fixed"
    cruise_speed_mps: float = 3.0
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        if self.controller != "independent-fixed":
            raise ValueError("controller must be 'independent-fixed'")
        if self.cruise_speed_mps <= 0.0 or not math.isfinite(self.cruise_speed_mps):
            raise ValueError("cruise_speed_mps must be finite and positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def independent_fixed_actions(
    observations: Mapping[str, Mapping[str, Any]],
    motion: MotionConfig,
    *,
    cruise_speed_mps: float = 3.0,
) -> dict[str, dict[str, Any]]:
    """Create truth-free, neighbor-free independent goal-tracking actions."""

    actions: dict[str, dict[str, Any]] = {}
    for uav_id, observation in observations.items():
        estimate = np.asarray(observation["estimate"], dtype=float).reshape(-1)
        goal = np.asarray(observation["goal"], dtype=float).reshape(3)
        if estimate.size < 6 or not np.isfinite(estimate[:6]).all() or not np.isfinite(goal).all():
            raise ValueError(f"non-finite local baseline observation for {uav_id}")

        targets = sorted(str(target) for target in observation.get("potential_targets", ()))
        bandwidth = (
            {target: 1.0 / len(targets) for target in targets}
            if targets
            else {}
        )
        displacement = goal - estimate[:3]
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            desired_velocity = np.zeros(3, dtype=float)
        else:
            desired_speed = min(float(cruise_speed_mps), distance / motion.dt_s)
            desired_velocity = displacement * (desired_speed / distance)
        delta_v = clip_vector_norm(
            desired_velocity - estimate[3:6],
            motion.max_acceleration_mps2 * motion.dt_s,
        )
        actions[uav_id] = {
            "measurement_uav_id": "none",
            "bandwidth": bandwidth,
            "delta_v": delta_v,
        }
    return actions


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    # Writers in the experiment pipeline use UTF-8 with a BOM so that the
    # resulting CSVs open cleanly in spreadsheet applications.  Accept that
    # encoding on read as well; otherwise the first column is silently named
    # ``\ufeff<field>`` and episode/scenario binding fails.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_formal_config(
    manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    force: bool,
    progress: bool,
) -> FormalExperimentConfig:
    raw = dict(manifest.get("config", {}))
    allowed = {item.name for item in fields(FormalExperimentConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    # JSON manifests necessarily serialize path-valued dataclass fields as
    # strings. Restore their type before downstream code calls ``resolve`` or
    # performs path joins. This keeps old manifests and new architecture
    # comparison branches interoperable.
    for key in ("scenario_profile_path", "scenario_path", "scenario_pool_root"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values.update(
        output_dir=output_dir,
        uav_count=int(manifest.get("uav_count", raw.get("uav_count", 0))),
        force=force,
        progress=progress,
    )
    if int(values["uav_count"]) != 3:
        raise ValueError("the first independent baseline must reference a three-UAV scale")
    return FormalExperimentConfig(**values)


def _prepare_output(config: IndependentBaselineConfig, uav_count: int) -> Path:
    scale_root = Path(config.output_dir) / f"scale_{uav_count}"
    if scale_root.exists() and any(scale_root.iterdir()) and not config.force:
        raise FileExistsError(
            f"baseline output is not empty: {scale_root}; use --force to replace it"
        )
    if config.force and scale_root.exists():
        shutil.rmtree(scale_root)
    scale_root.mkdir(parents=True, exist_ok=True)
    return scale_root


def _progress(config: IndependentBaselineConfig, message: str) -> None:
    if config.progress:
        print(f"[baseline] {message}", flush=True)


def _reward_objective_metrics(episode_root: Path) -> dict[str, float]:
    rewards = _read_csv(episode_root / "rewards.csv")
    scenario = _read_json(episode_root / "scenario.json")
    weights = dict(scenario.get("reward_config", {}).get("weights", {}))
    w_unc = float(weights.get("uncertainty", 0.35))

    def mean_column(name: str) -> float:
        values = np.asarray([float(row[name]) for row in rewards], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else 0.0

    joint = mean_column("joint_objective")
    j_loc = w_unc * mean_column("normalized_uncertainty")
    return {"j_loc": j_loc, "j_com": joint - j_loc, "joint_objective": joint}


def _safety_metrics(episode_root: Path) -> dict[str, float]:
    summary, _series, _metadata = _safety_summary(
        _read_csv(episode_root / "ground_truth.csv"),
        _read_csv(episode_root / "rewards.csv"),
        _read_csv(episode_root / "actions.csv"),
        _read_csv(episode_root / "obstacles.csv"),
    )
    return {str(key): float(value) for key, value in summary.items()}


def _run_one_episode(
    *,
    config: IndependentBaselineConfig,
    formal_config: FormalExperimentConfig,
    scale_root: Path,
    source_episode: Mapping[str, Any],
    episode_index: int,
    episode_total: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_id = str(source_episode["episode_id"])
    scenario_id = str(source_episode["scenario_id"])
    seed = int(source_episode["seed"])
    reference_episode_root = Path(config.reference_scale) / config.split / episode_id
    source_scenario = _read_json(reference_episode_root / "scenario.json")
    scenario = source_scenario.get("scenario_generation")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"missing scenario_generation in {reference_episode_root / 'scenario.json'}")

    _seed_everything(seed)
    environment = _make_environment(
        formal_config,
        seed,
        metric_scales=formal_config.metric_scales,
    )
    observations, info = environment.reset(seed=seed, scenario_config=dict(scenario))
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / config.split / episode_id,
        episode_id=episode_id,
        scenario_id=scenario_id,
        split=config.split,
        seed=seed,
        phase=config.split,
        environment=environment,
        normalization_fitted=True,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=POLICY_SOURCE,
    )
    rewards_history: list[float] = []
    effective_edges_seen = 0
    max_neighbor_updates = 0
    _progress(config, f"episode {episode_index}/{episode_total} start seed={seed}")
    started_at = time.perf_counter()
    for step in range(formal_config.max_steps):
        _assert_observation_finite(observations)
        actions = independent_fixed_actions(
            observations,
            environment.config.motion,
            cruise_speed_mps=config.cruise_speed_mps,
        )
        _assert_action_feasible(environment, actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
        rewards_history.extend(float(value) for value in rewards.values())
        effective_edges_seen += len(next_info["central_state"]["effective_edges"])
        max_neighbor_updates = max(
            max_neighbor_updates,
            max(ekf.valid_neighbor_count for ekf in environment.ekfs.values()),
        )
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=actions,
            rewards=rewards,
            info=next_info,
            graph_traces=None,
            policy_source=POLICY_SOURCE,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break

    summary = writer.finalize()
    summary.update(
        seed=seed,
        mean_reward=float(np.mean(rewards_history)) if rewards_history else 0.0,
        effective_edges_seen=effective_edges_seen,
        max_neighbor_updates=max_neighbor_updates,
        update=None,
        source_episode_id=episode_id,
        source_scenario_sha256=_sha256(reference_episode_root / "scenario.json"),
    )
    if max_neighbor_updates != 0:
        raise RuntimeError("independent baseline unexpectedly consumed a neighbor EKF update")
    metrics = _episode_metrics(scale_root / config.split / episode_id, summary)
    metrics.update(_reward_objective_metrics(scale_root / config.split / episode_id))
    metrics.update(_safety_metrics(scale_root / config.split / episode_id))
    _progress(
        config,
        f"episode {episode_index}/{episode_total} complete mean_reward={summary['mean_reward']:.6f} "
        f"elapsed={time.perf_counter() - started_at:.1f}s",
    )
    return summary, metrics


def _reference_metrics(reference_scale: Path, episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_root = reference_scale / str(episode["split"]) / str(episode["episode_id"])
        metrics = _episode_metrics(episode_root, episode)
        metrics.update(_reward_objective_metrics(episode_root))
        metrics.update(_safety_metrics(episode_root))
        rows.append(metrics)
    return rows


def _bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_paired_comparison(
    *,
    config: IndependentBaselineConfig,
    scale_root: Path,
    reference_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    reference_by_seed = {int(row["seed"]): row for row in reference_rows}
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    if reference_by_seed.keys() != baseline_by_seed.keys():
        raise RuntimeError("reference and baseline test seeds do not match exactly")

    metric_names = (
        "mean_reward",
        "position_rmse_m",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
        "collision_rate",
        "safety_projection_count",
        "minimum_clearance_m",
        "j_loc",
        "j_com",
        "joint_objective",
    )
    paired_rows: list[dict[str, Any]] = []
    for seed in sorted(reference_by_seed):
        for metric in metric_names:
            reference = float(reference_by_seed[seed][metric])
            baseline = float(baseline_by_seed[seed][metric])
            difference = baseline - reference
            paired_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "mappo": reference,
                    "independent_fixed_baseline": baseline,
                    "baseline_minus_mappo": difference,
                    "relative_difference": difference / abs(reference) if abs(reference) > 1e-12 else "",
                }
            )
    paired_path = scale_root / "diagnostics" / "paired_metrics.csv"
    _write_csv(
        paired_path,
        (
            "seed",
            "metric",
            "mappo",
            "independent_fixed_baseline",
            "baseline_minus_mappo",
            "relative_difference",
        ),
        paired_rows,
    )

    rng = np.random.default_rng(config.bootstrap_seed)
    summary_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        selected = [row for row in paired_rows if row["metric"] == metric]
        reference = np.asarray([float(row["mappo"]) for row in selected], dtype=float)
        baseline = np.asarray([float(row["independent_fixed_baseline"]) for row in selected], dtype=float)
        differences = baseline - reference
        ci_low, ci_high = _bootstrap_interval(differences, config.bootstrap_samples, rng)
        summary_rows.append(
            {
                "metric": metric,
                "paired_episodes": len(selected),
                "mappo_mean": float(reference.mean()),
                "baseline_mean": float(baseline.mean()),
                "mean_paired_difference": float(differences.mean()),
                "median_paired_difference": float(np.median(differences)),
                "paired_difference_std": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
            }
        )
    summary_path = scale_root / "diagnostics" / "paired_summary.csv"
    _write_csv(
        summary_path,
        (
            "metric",
            "paired_episodes",
            "mappo_mean",
            "baseline_mean",
            "mean_paired_difference",
            "median_paired_difference",
            "paired_difference_std",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
        ),
        summary_rows,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = ("position_rmse_m", "joint_objective", "packet_loss", "aoi_s", "connectivity", "collision_rate")
    labels = {
        "position_rmse_m": "Position RMSE (m)",
        "joint_objective": "Core objective J",
        "packet_loss": "Packet loss",
        "aoi_s": "AoI (s)",
        "connectivity": "Connectivity",
        "collision_rate": "Collision rate",
    }
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, plotted):
        selected = [row for row in paired_rows if row["metric"] == metric]
        for row in selected:
            axis.plot(
                (0, 1),
                (float(row["mappo"]), float(row["independent_fixed_baseline"])),
                color="#94a3b8",
                alpha=0.45,
                linewidth=0.8,
            )
        axis.scatter(
            np.zeros(len(selected)),
            [float(row["mappo"]) for row in selected],
            color="#2563eb",
            s=18,
            zorder=3,
        )
        axis.scatter(
            np.ones(len(selected)),
            [float(row["independent_fixed_baseline"]) for row in selected],
            color="#dc2626",
            s=18,
            zorder=3,
        )
        axis.set_xticks((0, 1), ("MAPPO", "Independent"))
        axis.set_title(labels[metric])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Paired three-UAV test episodes (same scenarios and seeds)")
    plot_path = scale_root / "diagnostics" / "baseline_comparison.png"
    figure.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "paired_metrics": str(paired_path),
        "paired_summary": str(summary_path),
        "comparison_plot": str(plot_path),
    }


def run_independent_baseline(config: IndependentBaselineConfig) -> dict[str, Any]:
    """Replay a reference three-UAV evaluation split under the fixed baseline."""

    reference_scale = Path(config.reference_scale)
    reference_manifest = _read_json(reference_scale / "manifest.json")
    if str(reference_manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(reference_manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    formal_config = _reference_formal_config(
        reference_manifest,
        Path(config.output_dir),
        force=config.force,
        progress=config.progress,
    )
    episodes = [
        episode
        for episode in reference_manifest.get("episodes", [])
        if isinstance(episode, Mapping) and episode.get("split") == config.split
    ]
    if not episodes:
        raise ValueError(f"reference contains no {config.split} episodes")

    scale_root = _prepare_output(config, formal_config.uav_count)
    _copy_calibration_files(scale_root, formal_config)
    source_normalization = reference_scale / "normalization_stats.json"
    if source_normalization.exists():
        shutil.copy2(source_normalization, scale_root / "normalization_stats.json")

    summaries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        summary, row = _run_one_episode(
            config=config,
            formal_config=formal_config,
            scale_root=scale_root,
            source_episode=episode,
            episode_index=index,
            episode_total=len(episodes),
        )
        summaries.append(summary)
        metrics.append(row)
    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", metrics)

    reference_rows = _reference_metrics(reference_scale, episodes)
    comparison = _write_paired_comparison(
        config=config,
        scale_root=scale_root,
        reference_rows=reference_rows,
        baseline_rows=metrics,
    )
    manifest = {
        "dataset_id": "python_multi_uav_independent_fixed_baseline",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "engine": "python_sim",
        "result_status": "engineering_baseline_diagnostic_only",
        "policy_source": POLICY_SOURCE,
        "uav_count": formal_config.uav_count,
        "reference": {
            "scale_root": str(reference_scale),
            "manifest_sha256": _sha256(reference_scale / "manifest.json"),
            "policy_source": "mappo",
            "split": config.split,
        },
        "controller": {
            "name": config.controller,
            "uses_ground_truth": False,
            "uses_gat": False,
            "uses_mappo": False,
            "uses_central_state": False,
            "uses_neighbor_state": False,
            "uses_uwb_cooperative_update": False,
            "measurement_uav_id": "none",
            "bandwidth_policy": "equal_over_potential_edges",
            "trajectory_policy": "local_ekf_estimate_goal_tracking",
            "cruise_speed_mps": config.cruise_speed_mps,
        },
        "paired_design": {
            "independent_unit": "episode",
            "paired_by": ["scenario_id", "seed"],
            "time_steps_are_independent_replicates": False,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "config": {**asdict(formal_config), "output_dir": str(formal_config.output_dir)},
        "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
        "episodes": summaries,
        "comparison": comparison,
    }
    _write_json(scale_root / "manifest.json", manifest)
    root_manifest = {
        "dataset_id": manifest["dataset_id"],
        "schema_version": EPISODE_SCHEMA_VERSION,
        "scales": {
            "3": {
                "uav_count": 3,
                "path": "scale_3",
                "manifest": "scale_3/manifest.json",
                "episode_count": len(summaries),
                "result_status": manifest["result_status"],
            }
        },
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    validation = validate_episode_benchmark(scale_root)
    if not validation["ok"]:
        raise RuntimeError(f"baseline episode validation failed: {validation}")
    report = {
        "output_dir": str(config.output_dir),
        "scale_root": str(scale_root),
        "reference_scale": str(reference_scale),
        "paired_episodes": len(summaries),
        "schema_version": EPISODE_SCHEMA_VERSION,
        "policy_source": POLICY_SOURCE,
        "comparison": comparison,
        "validation_ok": True,
    }
    _write_json(scale_root / "checks" / "integration_summary.json", report)
    return report


__all__ = [
    "IndependentBaselineConfig",
    "POLICY_SOURCE",
    "independent_fixed_actions",
    "run_independent_baseline",
]
