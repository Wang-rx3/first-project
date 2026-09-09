"""Equal-budget validation intervention for cooperative-neighbor selection.

The intervention replays the frozen Proposed policy on the same validation
scenarios and preserves the recorded ``none``/non-``none`` decision for every
UAV and window.  Only the selected neighbor is changed for the two comparator
treatments.  One comparator rotates through actor-visible candidates and the
other chooses the lowest train-fitted actor-visible quality risk.  This
separates measurement density from neighbor-selection quality; it does not
retrain MAPPO or establish policy superiority.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _jsonable, _write_json
from .bandwidth_intervention import _max_numeric_difference, load_recorded_actions
from .baseline import _read_json, _reference_formal_config, _sha256
from .cooperative_update_quality_audit import _quality_event_row
from .formal import _copy_calibration_files, _episode_metrics, _make_environment
from .localization_coupling_intervention import (
    _NEES_95_LOWER,
    _NEES_95_UPPER,
    _bootstrap_interval,
    _mean,
    _position_diagnostics,
    _quantile,
    _write_table,
)
from .measurement_density_intervention import _bandwidth_difference, _gate_row, actor_visible_measurement_candidates
from .pilot import _EpisodeTraceWriter, _assert_action_feasible, _assert_observation_finite, _seed_everything
from .quality_aware_measurement_intervention import fit_training_quality_thresholds, quality_candidate_scores


TREATMENTS = ("recorded_mappo", "equal_budget_round_robin", "equal_budget_quality_aware")
POLICY_PREFIX = "equal_budget_measurement_intervention"
_NUMERIC_TOLERANCE = 2e-6
_PAYLOAD_BYTES = 512


@dataclass(frozen=True)
class EqualBudgetMeasurementInterventionConfig:
    reference_scale: Path
    output_dir: Path
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("equal-budget measurement intervention is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: EqualBudgetMeasurementInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[equal-budget-measurement] {message}", flush=True)


def _prepare_outputs(config: EqualBudgetMeasurementInterventionConfig) -> tuple[Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else root.parent / "logs"
    plots = Path(config.plots_dir) if config.plots_dir is not None else root.parent / "plots"
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"intervention output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _copy_action(recorded: Mapping[str, Any], source: str) -> dict[str, Any]:
    return {
        "measurement_uav_id": str(source),
        "bandwidth": {
            str(target): float(value)
            for target, value in dict(recorded.get("bandwidth", {})).items()
        },
        "delta_v": np.asarray(recorded["delta_v"], dtype=float).reshape(3),
    }


def _choose_equal_budget_actions(
    recorded_actions: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    uav_ids: Sequence[str],
    treatment: str,
    thresholds: Mapping[str, Any],
    cursors: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if treatment not in TREATMENTS:
        raise ValueError(f"unknown equal-budget treatment: {treatment}")
    actions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for uav_id in uav_ids:
        recorded = recorded_actions[uav_id]
        recorded_source = str(recorded.get("measurement_uav_id", "none"))
        candidates = actor_visible_measurement_candidates(observations[uav_id], uav_ids)
        selected = recorded_source
        candidate_scores: list[dict[str, Any]] = []
        if treatment == "recorded_mappo":
            selected = recorded_source
        elif recorded_source == "none":
            # Equal-budget intervention must retain every recorded none.
            selected = "none"
        elif not candidates:
            # This is intentionally recorded as a budget-preservation failure.
            selected = "none"
        elif treatment == "equal_budget_round_robin":
            cursor = int(cursors.get(uav_id, 0))
            selected = candidates[cursor % len(candidates)]
            cursors[uav_id] = cursor + 1
            if len(candidates) > 1 and selected == recorded_source:
                selected = candidates[(cursor + 1) % len(candidates)]
        else:
            candidate_scores = quality_candidate_scores(observations[uav_id], uav_ids, thresholds)
            # Rank every currently visible candidate, not only eligible rows,
            # so that a recorded non-none decision remains non-none.
            selected = str(candidate_scores[0]["source_uav_id"]) if candidate_scores else "none"

        selected_non_none = selected != "none"
        budget_preserved = int(selected_non_none == (recorded_source != "none"))
        selected_score = next(
            (float(row["quality_risk_score"]) for row in candidate_scores if row["source_uav_id"] == selected),
            -1.0,
        )
        actions[uav_id] = _copy_action(recorded, selected)
        metadata[uav_id] = {
            "candidate_count": len(candidates),
            "candidate_uav_ids": candidates,
            "candidate_available": bool(candidates),
            "recorded_measurement_uav_id": recorded_source,
            "selected_measurement_uav_id": selected,
            "recorded_non_none": int(recorded_source != "none"),
            "selected_non_none": int(selected_non_none),
            "budget_preserved": budget_preserved,
            "selection_changed": int(selected != recorded_source),
            "selected_quality_risk_score": selected_score,
            "candidate_quality_json": json.dumps(_jsonable(candidate_scores), sort_keys=True),
            "actor_visible_only": True,
            "uses_physical_los": False,
            "uses_truth_error": False,
            "uses_future_information": False,
        }
    return actions, metadata


def _run_episode(
    *,
    config: EqualBudgetMeasurementInterventionConfig,
    formal_config: Any,
    thresholds: Mapping[str, Any],
    treatment: str,
    scale_root: Path,
    source_episode: Mapping[str, Any],
    episode_index: int,
    episode_total: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_id = str(source_episode["episode_id"])
    scenario_id = str(source_episode["scenario_id"])
    seed = int(source_episode["seed"])
    source_root = Path(config.reference_scale) / config.split / episode_id
    scenario_payload = _read_json(source_root / "scenario.json")
    scenario = scenario_payload.get("scenario_generation")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"missing scenario_generation in {source_root / 'scenario.json'}")
    recorded_windows = load_recorded_actions(source_root)
    _seed_everything(seed)
    environment = _make_environment(
        formal_config,
        seed,
        metric_scales=formal_config.metric_scales,
        cooperative_ekf_enabled=True,
    )
    observations, _info = environment.reset(seed=seed, scenario_config=dict(scenario))
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
        policy_source=f"{POLICY_PREFIX}_{treatment}",
    )
    cursors = {uav_id: 0 for uav_id in environment.uav_ids}
    windows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    errors: list[float] = []
    traces: list[float] = []
    nees_values: list[float] = []
    rewards_history: list[float] = []
    _progress(config, f"{treatment} episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()

    for step, recorded in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        actions, metadata = _choose_equal_budget_actions(
            recorded, observations, environment.uav_ids, treatment, thresholds, cursors
        )
        _assert_action_feasible(environment, actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
        timestamp = (step + 1) * environment.config.motion.dt_s
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=actions,
            rewards=rewards,
            info=next_info,
            graph_traces=None,
            policy_source=f"{POLICY_PREFIX}_{treatment}",
            action_start=step * environment.config.motion.dt_s,
            timestamp=timestamp,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        rewards_history.extend(float(value) for value in rewards.values())
        current_errors, current_traces, current_nees = _position_diagnostics(environment)
        errors.extend(current_errors)
        traces.extend(current_traces)
        nees_values.extend(current_nees)
        gate_by_target = {
            str(event["target_uav_id"]): event
            for event in next_info.get("measurement_gate_events", [])
        }
        components = next_info["objective_components"]
        for uav_id in environment.uav_ids:
            row = _gate_row(
                episode_id=episode_id,
                scenario_id=scenario_id,
                seed=seed,
                split=config.split,
                treatment=treatment,
                step=step,
                timestamp=timestamp,
                uav_id=uav_id,
                metadata=metadata[uav_id],
                gate_event=gate_by_target.get(uav_id),
                environment=environment,
                components=components,
            )
            row.update(
                recorded_non_none=int(metadata[uav_id]["recorded_non_none"]),
                budget_preserved=int(metadata[uav_id]["budget_preserved"]),
                selected_quality_risk_score=float(metadata[uav_id]["selected_quality_risk_score"]),
                candidate_quality_json=str(metadata[uav_id]["candidate_quality_json"]),
                actor_visible_only=int(bool(metadata[uav_id]["actor_visible_only"])),
            )
            windows.append(row)
        for event_index, event in enumerate(next_info.get("relative_range_update_events", [])):
            events.append(
                _quality_event_row(
                    event=event,
                    treatment=treatment,
                    episode_id=episode_id,
                    scenario_id=scenario_id,
                    seed=seed,
                    step=step,
                    event_index=event_index,
                )
            )
        observations = next_observations
        if terminated or truncated:
            break

    summary = writer.finalize()
    summary.update(
        seed=seed,
        mean_reward=_mean(rewards_history),
        source_episode_id=episode_id,
        source_scenario_sha256=_sha256(source_root / "scenario.json"),
        treatment=treatment,
    )
    metrics = _episode_metrics(scale_root / config.split / episode_id, summary)
    accepted = [row for row in events if int(row["accepted"])]
    counts = {label: sum(row["effect_class"] == label for row in accepted) for label in ("beneficial", "harmful", "approximately_neutral")}
    treatment_windows = [row for row in windows if row["treatment"] == treatment]
    recorded_count = sum(int(row["recorded_non_none"]) for row in treatment_windows)
    selected_count = sum(int(row["selected_non_none"]) for row in treatment_windows)
    metrics.update(
        treatment=treatment,
        selected_non_none_count=selected_count,
        recorded_non_none_count=recorded_count,
        budget_mismatch_count=sum(1 - int(row["budget_preserved"]) for row in treatment_windows),
        selection_changed_count=sum(int(row["selection_changed"]) for row in treatment_windows),
        selection_changed_fraction=_mean(int(row["selection_changed"]) for row in treatment_windows),
        measurement_bytes=selected_count * _PAYLOAD_BYTES,
        accepted_update_count=len(accepted),
        ekf_updates_accepted=len(accepted),
        beneficial_fraction=counts["beneficial"] / max(1, len(accepted)),
        harmful_fraction=counts["harmful"] / max(1, len(accepted)),
        neutral_fraction=counts["approximately_neutral"] / max(1, len(accepted)),
        position_error_p95_m=_quantile(errors, 0.95),
        mean_position_covariance_trace_m2=_mean(traces),
        mean_position_nees=_mean(nees_values),
        nees_95_coverage=_mean(_NEES_95_LOWER <= value <= _NEES_95_UPPER for value in nees_values),
    )
    _progress(
        config,
        f"{treatment} episode {episode_index}/{episode_total} complete "
        f"selected={selected_count} updates={len(accepted)} "
        f"rmse={metrics['position_rmse_norm_m']:.6f} elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, windows, events


def _paired_contrasts(rows: Sequence[Mapping[str, Any]], config: EqualBudgetMeasurementInterventionConfig) -> list[dict[str, Any]]:
    metrics = (
        "selected_non_none_count", "selection_changed_fraction", "accepted_update_count",
        "beneficial_fraction", "harmful_fraction", "position_rmse_norm_m",
        "position_p95_norm_m", "j_loc", "j_com", "measurement_bytes",
    )
    by_key = {(str(row["episode_id"]), str(row["treatment"])): row for row in rows}
    episodes = sorted({str(row["episode_id"]) for row in rows})
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for treatment in ("equal_budget_round_robin", "equal_budget_quality_aware"):
        for metric in metrics:
            factual = np.asarray([float(by_key[(episode, "recorded_mappo")][metric]) for episode in episodes], dtype=float)
            comparator = np.asarray([float(by_key[(episode, treatment)][metric]) for episode in episodes], dtype=float)
            difference = comparator - factual
            low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
            output.append({
                "contrast": f"{treatment}_minus_recorded_mappo",
                "metric": metric,
                "paired_episodes": len(episodes),
                "recorded_mappo_mean": float(factual.mean()),
                "comparator_mean": float(comparator.mean()),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "paired_difference_std": float(difference.std(ddof=1)) if difference.size > 1 else 0.0,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            })
    return output


def _invariance_checks(
    config: EqualBudgetMeasurementInterventionConfig,
    roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "recorded_reproduces_reference_max_abs_difference": 0.0,
        "cross_treatment_ground_truth_max_abs_difference": 0.0,
        "cross_treatment_obstacle_max_abs_difference": 0.0,
        "cross_treatment_delta_v_max_abs_difference": 0.0,
        "cross_treatment_bandwidth_max_abs_difference": 0.0,
    }
    budget_mismatch = 0
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        reference = Path(config.reference_scale) / config.split / episode_id
        factual = roots["recorded_mappo"] / config.split / episode_id
        checks["recorded_reproduces_reference_max_abs_difference"] = max(
            checks["recorded_reproduces_reference_max_abs_difference"],
            _max_numeric_difference(
                factual / "estimates.csv", reference / "estimates.csv",
                key_columns=("timestamp", "uav_id"), value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
            _max_numeric_difference(
                factual / "rewards.csv", reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"), value_columns=("reward_total", "joint_objective", "normalized_uncertainty"),
            ),
        )
        for treatment in ("equal_budget_round_robin", "equal_budget_quality_aware"):
            comparator = roots[treatment] / config.split / episode_id
            checks["cross_treatment_ground_truth_max_abs_difference"] = max(
                checks["cross_treatment_ground_truth_max_abs_difference"],
                _max_numeric_difference(factual / "ground_truth.csv", comparator / "ground_truth.csv", key_columns=("timestamp", "uav_id"), value_columns=("x", "y", "z", "vx", "vy", "vz")),
            )
            checks["cross_treatment_obstacle_max_abs_difference"] = max(
                checks["cross_treatment_obstacle_max_abs_difference"],
                _max_numeric_difference(factual / "obstacles.csv", comparator / "obstacles.csv", key_columns=("timestamp", "obstacle_id"), value_columns=("x", "y", "z", "vx", "vy", "vz", "radius")),
            )
            checks["cross_treatment_delta_v_max_abs_difference"] = max(
                checks["cross_treatment_delta_v_max_abs_difference"],
                _max_numeric_difference(factual / "actions.csv", comparator / "actions.csv", key_columns=("timestamp", "uav_id"), value_columns=("delta_vx", "delta_vy", "delta_vz")),
            )
            checks["cross_treatment_bandwidth_max_abs_difference"] = max(
                checks["cross_treatment_bandwidth_max_abs_difference"],
                _bandwidth_difference(factual / "actions.csv", comparator / "actions.csv"),
            )
    return checks


def _plot(rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]], plots_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "recorded_mappo": "Recorded MAPPO",
        "equal_budget_round_robin": "Equal-budget round-robin",
        "equal_budget_quality_aware": "Equal-budget quality-aware",
    }
    colors = {"recorded_mappo": "#2563eb", "equal_budget_round_robin": "#dc2626", "equal_budget_quality_aware": "#16a34a"}
    metrics = (
        ("selected_non_none_count", "Selected measurements / episode"),
        ("selection_changed_fraction", "Neighbor replacement fraction"),
        ("accepted_update_count", "Accepted cooperative EKF updates"),
        ("position_rmse_norm_m", "Position RMSE (m)"),
        ("position_p95_norm_m", "Position P95 (m)"),
        ("j_com", r"$J_{com}$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for treatment in TREATMENTS:
            values = np.asarray([float(row[metric]) for row in rows if row["treatment"] == treatment], dtype=float)
            x = np.full(values.size, list(TREATMENTS).index(treatment), dtype=float)
            jitter = np.linspace(-0.08, 0.08, values.size) if values.size else np.asarray([])
            axis.scatter(x + jitter, values, color=colors[treatment], alpha=0.55, s=22, label=labels[treatment])
            if values.size:
                axis.scatter([list(TREATMENTS).index(treatment)], [float(values.mean())], color=colors[treatment], s=70, edgecolor="black", linewidth=0.5, zorder=4)
        axis.set_title(title)
        axis.set_xticks(range(len(TREATMENTS)), [labels[item] for item in TREATMENTS], rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Paired validation: equal measurement budget, neighbor selection only")
    target = plots_dir / "equal_budget_measurement_intervention.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_equal_budget_measurement_intervention(config: EqualBudgetMeasurementInterventionConfig) -> dict[str, Any]:
    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    if int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("the equal-budget intervention requires three UAVs")
    episodes = [item for item in manifest.get("episodes", []) if isinstance(item, Mapping) and item.get("split") == config.split]
    if len(episodes) != 20:
        raise ValueError(f"the intervention requires exactly 20 validation episodes, found {len(episodes)}")

    logs_dir, plots_dir = _prepare_outputs(config)
    formal_config = _reference_formal_config(manifest, Path(config.output_dir), force=False, progress=config.progress)
    thresholds = fit_training_quality_thresholds(reference_scale)
    _write_json(logs_dir / "quality_thresholds_train_only.json", _jsonable(thresholds))
    roots: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    treatment_manifests: dict[str, Any] = {}
    for treatment in TREATMENTS:
        treatment_root = Path(config.output_dir) / "treatments" / treatment
        scale_root = treatment_root / "scale_3"
        scale_root.mkdir(parents=True, exist_ok=True)
        roots[treatment] = scale_root
        treatment_config = _reference_formal_config(manifest, treatment_root, force=False, progress=config.progress)
        _copy_calibration_files(scale_root, treatment_config)
        normalization = reference_scale / "normalization_stats.json"
        if normalization.exists():
            shutil.copy2(normalization, scale_root / "normalization_stats.json")
        summaries: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metrics, episode_windows, episode_events = _run_episode(
                config=config,
                formal_config=formal_config,
                thresholds=thresholds,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
            )
            summaries.append(summary)
            rows.append(metrics)
            windows.extend(episode_windows)
            events.extend(episode_events)
        treatment_manifest = {
            "dataset_id": f"python_equal_budget_measurement_{treatment}",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "engine": "python_sim",
            "result_status": "engineering_mechanism_diagnostic_only",
            "policy_source": f"{POLICY_PREFIX}_{treatment}",
            "uav_count": 3,
            "reference": {"scale_root": str(reference_scale), "manifest_sha256": _sha256(reference_scale / "manifest.json"), "split": config.split},
            "intervention": {
                "changed_action_field_only": "measurement_uav_id",
                "recorded_none_pattern_preserved": True,
                "equal_budget": True,
                "quality_selection_uses_train_only_thresholds": treatment == "equal_budget_quality_aware",
                "uses_future_information": False,
                "uses_test_data": False,
                "retrained_mappo": False,
            },
            "config": _jsonable({**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)}),
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", _jsonable(treatment_manifest))
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {"scale_root": str(scale_root), "manifest": str(scale_root / "manifest.json"), "validation_ok": True}

    contrasts = _paired_contrasts(rows, config)
    invariants = _invariance_checks(config, roots, episodes)
    budget_mismatches = int(sum(int(row["budget_mismatch_count"]) for row in rows))
    finite = all(np.isfinite(float(row[key])) for row in rows for key in ("position_rmse_norm_m", "position_p95_norm_m", "j_loc", "j_com", "accepted_update_count"))
    pairing_ok = all(float(value) <= _NUMERIC_TOLERANCE for value in invariants.values())
    budget_ok = budget_mismatches == 0
    contrast_map = {(str(row["contrast"]), str(row["metric"])): row for row in contrasts}
    decisions: dict[str, str] = {}
    for treatment in ("equal_budget_round_robin", "equal_budget_quality_aware"):
        item = contrast_map[(f"{treatment}_minus_recorded_mappo", "position_rmse_norm_m")]
        low, high = float(item["bootstrap_95_ci_low"]), float(item["bootstrap_95_ci_high"])
        decisions[treatment] = (
            "equal_budget_selection_improves_rmse" if high < 0.0 else
            "equal_budget_selection_worsens_rmse" if low > 0.0 else
            "equal_budget_selection_effect_inconclusive"
        )
    audit = {
        "experiment": "equal_budget_cooperative_neighbor_selection_intervention",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "paired_episode_count": len(episodes),
        "time_steps_used_as_independent_samples": False,
        "test_data_used": False,
        "quality_threshold_source_split": thresholds["source_split"],
        "checks": invariants | {"budget_mismatch_count": budget_mismatches, "budget_preservation_ok": budget_ok},
        "deterministic_pairing_ok": pairing_ok,
        "equal_budget_ok": budget_ok,
        "all_localization_metrics_finite": finite,
        "decisions": decisions,
        "interpretation_boundary": "This intervention keeps the recorded measurement density and changes only the actor-visible neighbor choice; it does not retrain MAPPO or establish policy superiority.",
        "overall_ok": bool(pairing_ok and budget_ok and finite),
    }
    _write_table(logs_dir / "window_metrics.csv", windows)
    _write_table(logs_dir / "episode_metrics.csv", rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "update_quality_events.csv", events)
    _write_json(logs_dir / "audit_report.json", _jsonable(audit))
    plot_path = _plot(rows, contrasts, plots_dir)
    root_manifest = {
        "dataset_id": "python_equal_budget_measurement_intervention",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_mechanism_diagnostic_only",
        "reference": {"scale_root": str(reference_scale), "manifest_sha256": _sha256(reference_scale / "manifest.json"), "split": config.split, "episode_count": len(episodes)},
        "paired_design": {"independent_unit": "episode", "paired_by": ["episode_id", "scenario_id", "seed"], "time_steps_are_independent_replicates": False, "common_scene_replay": True, "changed_field_only": "measurement_uav_id", "recorded_none_pattern_preserved": True, "quality_threshold_source_split": "train", "test_data_used": False, "bootstrap_samples": config.bootstrap_samples, "bootstrap_seed": config.bootstrap_seed},
        "treatments": treatment_manifests,
        "logs": {"episode_metrics": str(logs_dir / "episode_metrics.csv"), "window_metrics": str(logs_dir / "window_metrics.csv"), "paired_contrasts": str(logs_dir / "paired_contrasts.csv"), "update_quality_events": str(logs_dir / "update_quality_events.csv"), "audit_report": str(logs_dir / "audit_report.json")},
        "plots": {"equal_budget_measurement_intervention": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", _jsonable(root_manifest))
    if not audit["overall_ok"]:
        raise RuntimeError(f"equal-budget measurement audit failed: {audit['checks']}")
    return {"output_dir": str(config.output_dir), "reference_scale": str(reference_scale), "paired_episodes": len(episodes), "treatments": list(TREATMENTS), "logs_dir": str(logs_dir), "plots_dir": str(plots_dir), "audit_ok": True, "decisions": decisions, "plot": str(plot_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the validation-only equal-budget neighbor-selection intervention")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs-output", type=Path, required=True)
    parser.add_argument("--plots-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_equal_budget_measurement_intervention(
        EqualBudgetMeasurementInterventionConfig(
            reference_scale=args.reference,
            output_dir=args.output,
            logs_dir=args.logs_output,
            plots_dir=args.plots_output,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            force=args.force,
            progress=not args.quiet,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EqualBudgetMeasurementInterventionConfig",
    "TREATMENTS",
    "run_equal_budget_measurement_intervention",
]
