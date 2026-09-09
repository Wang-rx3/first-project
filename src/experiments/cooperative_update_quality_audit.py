"""Paired, read-only quality audit for cooperative relative-range EKF updates.

The audit replays the frozen MAPPO measurement choices and the actor-visible
round-robin intervention with common random numbers.  Ground-truth error is
used only after each update as an offline diagnostic label; it is never fed
to the environment, filter, actor, critic, or reward.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_csv, _write_json
from .baseline import _read_json, _reference_formal_config, _sha256
from .ekf_update_strength_audit import _event_row as _strength_event_row
from .formal import _copy_calibration_files, _episode_metrics, _make_environment
from .localization_coupling_intervention import (
    _NEES_95_LOWER,
    _NEES_95_UPPER,
    _bootstrap_interval,
    _mean,
    _position_diagnostics,
    _quantile,
)
from .measurement_density_intervention import (
    POLICY_PREFIX,
    TREATMENTS,
    MeasurementDensityInterventionConfig,
    _audit as _density_audit,
    _paired_contrasts as _density_paired_contrasts,
    intervene_measurement_selection,
)
from .bandwidth_intervention import load_recorded_actions
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)


POLICY_SOURCE = "cooperative_update_quality_audit"
PRACTICAL_EVENT_THRESHOLD_M = 0.001
_FORMULA_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CooperativeUpdateQualityAuditConfig:
    """Configuration for the validation-only paired update-quality audit."""

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
            raise ValueError("cooperative update-quality audit is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: CooperativeUpdateQualityAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[cooperative-update-quality] {message}", flush=True)


def _default_logs_dir(reference_scale: Path) -> Path:
    return Path("logs/preliminary") / reference_scale.parent.name / "cooperative_update_quality"


def _default_plots_dir(reference_scale: Path) -> Path:
    return Path("plots/preliminary") / reference_scale.parent.name / "cooperative_update_quality"


def _prepare_output(
    config: CooperativeUpdateQualityAuditConfig,
) -> tuple[Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir else _default_logs_dir(Path(config.reference_scale))
    plots = Path(config.plots_dir) if config.plots_dir else _default_plots_dir(Path(config.reference_scale))
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"audit output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    _write_csv(path, tuple(rows[0].keys()), [dict(row) for row in rows])


def classify_update_effect(error_change_m: float) -> str:
    """Classify an update using the fixed one-millimetre practical threshold."""

    if error_change_m < -PRACTICAL_EVENT_THRESHOLD_M:
        return "beneficial"
    if error_change_m > PRACTICAL_EVENT_THRESHOLD_M:
        return "harmful"
    return "approximately_neutral"


def _quality_event_row(
    *,
    event: Mapping[str, Any],
    treatment: str,
    episode_id: str,
    scenario_id: str,
    seed: int,
    step: int,
    event_index: int,
) -> dict[str, Any]:
    row = _strength_event_row(
        event=event,
        episode_id=episode_id,
        scenario_id=scenario_id,
        seed=seed,
        step=step,
        event_index=event_index,
    )
    change = float(row["true_error_change_after_minus_before_m"])
    effective = max(1e-12, float(row["effective_measurement_variance_m2"]))
    row.update(
        treatment=treatment,
        effect_class=classify_update_effect(change),
        absolute_error_change_m=abs(change),
        unknown_cross_bound_fraction=float(row["unknown_cross_bound_m2"]) / effective,
        truth_used_offline_only=1,
    )
    return row


def _run_episode(
    *,
    config: CooperativeUpdateQualityAuditConfig,
    formal_config: Any,
    treatment: str,
    scale_root: Path,
    source_episode: Mapping[str, Any],
    episode_index: int,
    episode_total: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    episode_id = str(source_episode["episode_id"])
    scenario_id = str(source_episode["scenario_id"])
    seed = int(source_episode["seed"])
    source_root = Path(config.reference_scale) / config.split / episode_id
    payload = _read_json(source_root / "scenario.json")
    scenario = payload.get("scenario_generation")
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
        policy_source=f"{POLICY_SOURCE}_{treatment}",
    )
    cursors = {uav_id: 0 for uav_id in environment.uav_ids}
    events: list[dict[str, Any]] = []
    errors: list[float] = []
    traces: list[float] = []
    nees_values: list[float] = []
    rewards_history: list[float] = []
    selection_rows = 0
    candidate_rows = 0
    _progress(config, f"{treatment} episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()

    for step, recorded in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        actions, metadata = intervene_measurement_selection(
            recorded, observations, environment.uav_ids, treatment, cursors
        )
        _assert_action_feasible(environment, actions)
        candidate_rows += sum(int(item["candidate_available"]) for item in metadata.values())
        selection_rows += sum(
            int(item["candidate_available"] and item["selected_non_none"])
            for item in metadata.values()
        )
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
            policy_source=f"{POLICY_SOURCE}_{treatment}",
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
        audit_event_count=len(events),
    )
    metrics = _episode_metrics(scale_root / config.split / episode_id, summary)
    accepted = [row for row in events if int(row["accepted"])]
    changes = [float(row["true_error_change_after_minus_before_m"]) for row in accepted]
    counts = {
        label: sum(str(row["effect_class"]) == label for row in accepted)
        for label in ("beneficial", "harmful", "approximately_neutral")
    }
    total = max(1, len(accepted))
    metrics.update(
        treatment=treatment,
        update_event_count=len(events),
        accepted_update_count=len(accepted),
        ekf_updates_accepted=len(accepted),
        beneficial_update_count=counts["beneficial"],
        harmful_update_count=counts["harmful"],
        neutral_update_count=counts["approximately_neutral"],
        beneficial_fraction=counts["beneficial"] / total,
        harmful_fraction=counts["harmful"] / total,
        neutral_fraction=counts["approximately_neutral"] / total,
        mean_true_error_change_m=_mean(changes),
        cumulative_true_error_change_m=float(sum(changes)),
        position_error_p95_m=_quantile(errors, 0.95),
        mean_position_covariance_trace_m2=_mean(traces),
        mean_position_nees=_mean(nees_values),
        nees_95_coverage=_mean(_NEES_95_LOWER <= value <= _NEES_95_UPPER for value in nees_values),
        selection_fraction_given_candidate=selection_rows / max(1, candidate_rows),
    )
    _progress(
        config,
        f"{treatment} episode {episode_index}/{episode_total} complete "
        f"updates={len(accepted)} beneficial={metrics['beneficial_fraction']:.3f} "
        f"harmful={metrics['harmful_fraction']:.3f} elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, events


def _quantile_cutpoints(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in np.quantile(array, (0.25, 0.50, 0.75)))


def _quartile_label(value: float, cuts: tuple[float, float, float]) -> str:
    if value <= cuts[0]:
        return "Q1"
    if value <= cuts[1]:
        return "Q2"
    if value <= cuts[2]:
        return "Q3"
    return "Q4"


def _nis_label(value: float) -> str:
    if value <= 1.0:
        return "[0,1]"
    if value <= 3.0:
        return "(1,3]"
    if value <= 7.815:
        return "(3,7.815]"
    return ">7.815"


def _age_label(value: float) -> str:
    if value <= 0.2:
        return "<=0.2s"
    if value <= 0.5:
        return "(0.2,0.5]s"
    return ">0.5s"


def _quality_strata(
    events: Sequence[Mapping[str, Any]],
    config: CooperativeUpdateQualityAuditConfig,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    accepted = [row for row in events if int(row["accepted"])]
    round_robin = [row for row in accepted if row["treatment"] == "round_robin_candidate"]
    fields = {
        "neighbor_direction_variance": "neighbor_direction_variance_m2",
        "kalman_gain": "kalman_gain_norm",
        "position_correction": "position_correction_norm_m",
        "unknown_cross_fraction": "unknown_cross_bound_fraction",
    }
    cutpoints = {
        name: list(_quantile_cutpoints(float(row[field]) for row in round_robin))
        for name, field in fields.items()
    }

    classifiers: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "los_status": lambda row: "LOS" if int(row["los"]) else "NLOS",
        "nis": lambda row: _nis_label(float(row["nis"])),
        "message_age": lambda row: _age_label(float(row["message_age_s"])),
    }
    for name, field in fields.items():
        cuts = tuple(cutpoints[name])
        classifiers[name] = lambda row, field=field, cuts=cuts: _quartile_label(float(row[field]), cuts)

    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        treatment_rows = [row for row in accepted if row["treatment"] == treatment]
        for factor, classifier in classifiers.items():
            labels = sorted({classifier(row) for row in treatment_rows})
            for label in labels:
                selected = [row for row in treatment_rows if classifier(row) == label]
                by_episode: dict[int, list[Mapping[str, Any]]] = {}
                for row in selected:
                    by_episode.setdefault(int(row["seed"]), []).append(row)
                episode_mean_changes = np.asarray(
                    [
                        _mean(float(row["true_error_change_after_minus_before_m"]) for row in rows)
                        for rows in by_episode.values()
                    ],
                    dtype=float,
                )
                low, high = _bootstrap_interval(
                    episode_mean_changes,
                    config.bootstrap_samples,
                    rng,
                )
                output.append(
                    {
                        "treatment": treatment,
                        "factor": factor,
                        "stratum": label,
                        "event_count": len(selected),
                        "episode_count": len(by_episode),
                        "mean_episode_error_change_m": float(episode_mean_changes.mean()),
                        "bootstrap_95_ci_low_m": low,
                        "bootstrap_95_ci_high_m": high,
                        "pooled_beneficial_fraction": _mean(
                            row["effect_class"] == "beneficial" for row in selected
                        ),
                        "pooled_harmful_fraction": _mean(
                            row["effect_class"] == "harmful" for row in selected
                        ),
                        "pooled_neutral_fraction": _mean(
                            row["effect_class"] == "approximately_neutral" for row in selected
                        ),
                        "mean_nis": _mean(float(row["nis"]) for row in selected),
                        "mean_message_age_s": _mean(float(row["message_age_s"]) for row in selected),
                        "mean_kalman_gain_norm": _mean(float(row["kalman_gain_norm"]) for row in selected),
                        "mean_position_correction_m": _mean(
                            float(row["position_correction_norm_m"]) for row in selected
                        ),
                        "event_rows_are_independent_replicates": 0,
                    }
                )
    return output, cutpoints


def _paired_contrasts(
    episode_rows: Sequence[Mapping[str, Any]],
    config: CooperativeUpdateQualityAuditConfig,
) -> list[dict[str, Any]]:
    metrics = (
        "accepted_update_count",
        "beneficial_fraction",
        "harmful_fraction",
        "neutral_fraction",
        "mean_true_error_change_m",
        "cumulative_true_error_change_m",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
    )
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in episode_rows}
    seeds = sorted({int(row["seed"]) for row in episode_rows})
    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    for metric in metrics:
        factual = np.asarray([float(by_key[(seed, "recorded_mappo")][metric]) for seed in seeds])
        intervention = np.asarray(
            [float(by_key[(seed, "round_robin_candidate")][metric]) for seed in seeds]
        )
        difference = intervention - factual
        low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
        output.append(
            {
                "contrast": "round_robin_candidate_minus_recorded_mappo",
                "metric": metric,
                "paired_episodes": len(seeds),
                "recorded_mappo_mean": float(factual.mean()),
                "round_robin_candidate_mean": float(intervention.mean()),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "paired_difference_std": float(difference.std(ddof=1)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            }
        )
    return output


def _audit(
    *,
    config: CooperativeUpdateQualityAuditConfig,
    treatment_roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    strata_rows: Sequence[Mapping[str, Any]],
    cutpoints: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    density_config = MeasurementDensityInterventionConfig(
        reference_scale=config.reference_scale,
        output_dir=config.output_dir,
        split=config.split,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        progress=False,
    )
    density_contrasts = _density_paired_contrasts(episode_rows, density_config)
    invariance = _density_audit(
        density_config,
        treatment_roots,
        episodes,
        episode_rows,
        density_contrasts,
    )
    accepted = [row for row in event_rows if int(row["accepted"])]
    counts = {
        treatment: sum(row["treatment"] == treatment for row in accepted)
        for treatment in TREATMENTS
    }
    finite_keys = (
        "true_error_change_after_minus_before_m", "innovation_m", "nis",
        "measurement_variance_m2", "neighbor_direction_variance_m2",
        "unknown_cross_bound_m2", "kalman_gain_norm", "position_correction_norm_m",
        "position_covariance_trace_before_m2", "position_covariance_trace_after_linear_m2",
        "position_covariance_trace_after_final_m2", "j_loc_before", "j_loc_after",
    )
    finite = all(np.isfinite(float(row[key])) for row in event_rows for key in finite_keys)
    formula_residual = max(
        (
            max(
                float(row["effective_variance_formula_residual"]),
                float(row["innovation_variance_formula_residual"]),
                float(row["state_correction_formula_residual"]),
            )
            for row in event_rows
        ),
        default=float("inf"),
    )
    effect_closure = max(
        (
            abs(
                float(row["true_error_change_after_minus_before_m"])
                - (
                    float(row["true_position_error_after_m"])
                    - float(row["true_position_error_before_m"])
                )
            )
            for row in event_rows
        ),
        default=float("inf"),
    )
    rr = [row for row in accepted if row["treatment"] == "round_robin_candidate"]
    rr_beneficial = _mean(row["effect_class"] == "beneficial" for row in rr)
    rr_harmful = _mean(row["effect_class"] == "harmful" for row in rr)
    rr_neutral = _mean(row["effect_class"] == "approximately_neutral" for row in rr)
    rr_mean_change = _mean(float(row["true_error_change_after_minus_before_m"]) for row in rr)
    rr_median_correction = _quantile(
        (float(row["position_correction_norm_m"]) for row in rr), 0.5
    )
    rr_median_cross = _quantile(
        (float(row["unknown_cross_bound_fraction"]) for row in rr), 0.5
    )
    risk_rows = [
        row for row in strata_rows
        if row["treatment"] == "round_robin_candidate"
        and int(row["episode_count"]) >= 5
        and float(row["mean_episode_error_change_m"]) > PRACTICAL_EVENT_THRESHOLD_M
        and float(row["bootstrap_95_ci_low_m"]) > 0.0
    ]
    cancellations = bool(
        rr_beneficial >= 0.25
        and rr_harmful >= 0.25
        and abs(rr_mean_change) <= PRACTICAL_EVENT_THRESHOLD_M
    )
    corrections_small = rr_median_correction < 0.01
    cross_dominant = rr_median_cross >= 0.50
    mostly_beneficial = rr_beneficial > rr_harmful and rr_mean_change < -PRACTICAL_EVENT_THRESHOLD_M
    if risk_rows:
        decision = "observable_quality_strata_predict_harm_design_quality_aware_mask"
    elif corrections_small and cross_dominant:
        decision = "updates_have_small_gain_and_cross_bound_dominates_review_covariance_model"
    elif cancellations:
        decision = "beneficial_and_harmful_updates_cancel_compare_quality_aware_selection"
    elif mostly_beneficial:
        decision = "single_updates_beneficial_but_endpoint_gain_small_audit_persistence"
    else:
        decision = "update_quality_inconclusive_retain_diagnostics_before_filter_change"

    counts_ok = counts == {"recorded_mappo": 50, "round_robin_candidate": 1287}
    overall = bool(
        invariance["overall_ok"]
        and counts_ok
        and finite
        and formula_residual <= _FORMULA_TOLERANCE
        and effect_closure <= _FORMULA_TOLERANCE
    )
    return {
        "experiment": "paired_cooperative_ekf_single_update_quality_audit",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "paired_episode_count": len(episodes),
        "update_events_used_as_independent_samples": False,
        "test_data_used": False,
        "truth_used_by_filter_actor_critic_or_reward": False,
        "checks": {
            "accepted_update_counts": counts,
            "expected_accepted_update_counts": {"recorded_mappo": 50, "round_robin_candidate": 1287},
            "event_count_reproduction_ok": counts_ok,
            "round_robin_beneficial_fraction": rr_beneficial,
            "round_robin_harmful_fraction": rr_harmful,
            "round_robin_neutral_fraction": rr_neutral,
            "round_robin_mean_error_change_m": rr_mean_change,
            "round_robin_median_position_correction_m": rr_median_correction,
            "round_robin_median_unknown_cross_fraction": rr_median_cross,
            "risk_strata": [
                {"factor": row["factor"], "stratum": row["stratum"]}
                for row in risk_rows
            ],
            "max_formula_residual": formula_residual,
            "max_true_error_change_closure_residual": effect_closure,
            "density_invariance_checks": invariance["checks"],
        },
        "quality_flags": {
            "observable_risk_strata_detected": bool(risk_rows),
            "position_corrections_small": corrections_small,
            "unknown_cross_bound_dominant": cross_dominant,
            "beneficial_harmful_cancellation": cancellations,
            "single_updates_mostly_beneficial": mostly_beneficial,
            "j_loc_insensitivity_expected_from_covariance_restoration": True,
        },
        "stratification_cutpoints": cutpoints,
        "deterministic_pairing_ok": invariance["deterministic_pairing_ok"],
        "formula_closure_ok": formula_residual <= _FORMULA_TOLERANCE,
        "all_metrics_finite": finite,
        "decision_gate": decision,
        "overall_ok": overall,
        "interpretation_boundary": (
            "Truth error labels are offline diagnostics only. Update events are nested within "
            "20 paired episodes and are not independent replicates."
        ),
    }


def _plot(
    event_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"recorded_mappo": "#2563eb", "round_robin_candidate": "#dc2626"}
    labels = {"recorded_mappo": "Recorded MAPPO", "round_robin_candidate": "Round-robin"}
    accepted = [row for row in event_rows if int(row["accepted"])]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

    classes = ("beneficial", "approximately_neutral", "harmful")
    x = np.arange(len(classes))
    width = 0.36
    for index, treatment in enumerate(TREATMENTS):
        rows = [row for row in accepted if row["treatment"] == treatment]
        values = [_mean(row["effect_class"] == item for row in rows) for item in classes]
        axes[0, 0].bar(x + (index - 0.5) * width, values, width, color=colors[treatment], label=labels[treatment])
    axes[0, 0].set_xticks(x, ("Beneficial", "Neutral", "Harmful"))
    axes[0, 0].set(title="Per-update effect class", ylabel="Fraction")
    axes[0, 0].legend()

    rr = [row for row in accepted if row["treatment"] == "round_robin_candidate"]
    los_groups = [
        [float(row["true_error_change_after_minus_before_m"]) for row in rr if int(row["los"]) == flag]
        for flag in (1, 0)
    ]
    axes[0, 1].boxplot(los_groups, labels=("LOS", "NLOS"), showfliers=False)
    axes[0, 1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0, 1].set(title="Round-robin update effect by propagation", ylabel=r"$e_{after}-e_{before}$ (m)")

    changes = [float(row["true_error_change_after_minus_before_m"]) for row in rr]
    axes[0, 2].scatter([float(row["nis"]) for row in rr], changes, s=11, alpha=0.35, color="#7c3aed")
    axes[0, 2].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0, 2].set(title="NIS and update effect", xlabel="NIS", ylabel=r"$\Delta e$ (m)")

    axes[0, 3].scatter(
        [float(row["position_correction_norm_m"]) for row in rr],
        changes,
        s=11,
        alpha=0.35,
        color="#4f46e5",
    )
    axes[0, 3].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0, 3].set(
        title="Correction magnitude and update effect",
        xlabel="Position correction (m)",
        ylabel=r"$\Delta e$ (m)",
    )

    axes[1, 0].scatter([float(row["message_age_s"]) for row in rr], changes, s=11, alpha=0.35, color="#0891b2")
    axes[1, 0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1, 0].set(title="Message age and update effect", xlabel="Message age (s)", ylabel=r"$\Delta e$ (m)")

    axes[1, 1].scatter(
        [float(row["unknown_cross_bound_fraction"]) for row in rr],
        [float(row["kalman_gain_norm"]) for row in rr],
        s=11,
        alpha=0.35,
        color="#ea580c",
    )
    axes[1, 1].set(title="Cross-correlation bound and gain", xlabel="Cross-bound / effective variance", ylabel="Kalman gain norm")

    before = [float(row["true_position_error_before_m"]) for row in rr]
    after = [float(row["true_position_error_after_m"]) for row in rr]
    axes[1, 2].scatter(before, after, s=11, alpha=0.3, color="#16a34a")
    limit = max(before + after, default=1.0)
    axes[1, 2].plot((0.0, limit), (0.0, limit), linestyle="--", color="#111827", linewidth=0.9)
    axes[1, 2].set(title="Error before and after update", xlabel="Before (m)", ylabel="After (m)")

    covariance_groups = (
        [float(row["position_covariance_trace_before_m2"]) for row in rr],
        [float(row["position_covariance_trace_after_linear_m2"]) for row in rr],
        [float(row["position_covariance_trace_after_final_m2"]) for row in rr],
    )
    axes[1, 3].boxplot(
        covariance_groups,
        labels=("Before", "Linear", "Final"),
        showfliers=False,
    )
    axes[1, 3].set(
        title="Linear update and conservative restoration",
        ylabel=r"Position covariance trace ($m^2$)",
    )

    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    fig.suptitle("Cooperative EKF single-update quality audit; truth used offline only")
    target = plots_dir / "cooperative_update_quality.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_cooperative_update_quality_audit(
    config: CooperativeUpdateQualityAuditConfig,
) -> dict[str, Any]:
    """Run the paired validation-only cooperative update-quality audit."""

    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    if int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("the first update-quality audit requires three UAVs")
    episodes = [
        item for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and item.get("split") == config.split
    ]
    if len(episodes) != 20:
        raise ValueError(f"the frozen audit requires exactly 20 validation episodes, found {len(episodes)}")

    logs_dir, plots_dir = _prepare_output(config)
    formal_config = _reference_formal_config(manifest, Path(config.output_dir), force=False, progress=config.progress)
    treatment_roots: dict[str, Path] = {}
    treatment_manifests: dict[str, Any] = {}
    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        treatment_root = Path(config.output_dir) / "treatments" / treatment
        scale_root = treatment_root / "scale_3"
        scale_root.mkdir(parents=True, exist_ok=True)
        treatment_roots[treatment] = scale_root
        treatment_config = _reference_formal_config(manifest, treatment_root, force=False, progress=config.progress)
        _copy_calibration_files(scale_root, treatment_config)
        normalization = reference_scale / "normalization_stats.json"
        if normalization.exists():
            shutil.copy2(normalization, scale_root / "normalization_stats.json")
        summaries: list[dict[str, Any]] = []
        treatment_metrics: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metrics, events = _run_episode(
                config=config,
                formal_config=formal_config,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
            )
            summaries.append(summary)
            treatment_metrics.append(metrics)
            event_rows.extend(events)
        episode_rows.extend(treatment_metrics)
        treatment_manifest = {
            "dataset_id": f"python_cooperative_update_quality_{treatment}",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "engine": "python_sim",
            "result_status": "engineering_mechanism_diagnostic_only",
            "policy_source": f"{POLICY_SOURCE}_{treatment}",
            "uav_count": 3,
            "reference": {
                "scale_root": str(reference_scale),
                "manifest_sha256": _sha256(reference_scale / "manifest.json"),
                "split": config.split,
            },
            "audit": {
                "treatment": treatment,
                "filter_or_reward_changed": False,
                "truth_used_offline_only": True,
                "round_robin_uses_actor_visible_candidates_only": treatment == "round_robin_candidate",
                "uses_test_data": False,
            },
            "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", treatment_manifest)
        _write_json(treatment_root / "manifest.json", {
            "dataset_id": treatment_manifest["dataset_id"],
            "schema_version": EPISODE_SCHEMA_VERSION,
            "scales": {"3": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json", "episode_count": len(summaries)}},
        })
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {
            "scale_root": str(scale_root),
            "manifest": str(scale_root / "manifest.json"),
            "validation_ok": True,
        }

    strata_rows, cutpoints = _quality_strata(event_rows, config)
    contrast_rows = _paired_contrasts(episode_rows, config)
    audit = _audit(
        config=config,
        treatment_roots=treatment_roots,
        episodes=episodes,
        episode_rows=episode_rows,
        event_rows=event_rows,
        strata_rows=strata_rows,
        cutpoints=cutpoints,
    )
    _write_table(logs_dir / "update_events.csv", event_rows)
    _write_table(logs_dir / "episode_summary.csv", episode_rows)
    _write_table(logs_dir / "quality_strata.csv", strata_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrast_rows)
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(event_rows, episode_rows, plots_dir)
    root_manifest = {
        "dataset_id": "python_cooperative_update_quality_audit_seed2026",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "engineering_mechanism_diagnostic_only",
        "reference": {
            "scale_root": str(reference_scale),
            "manifest_sha256": _sha256(reference_scale / "manifest.json"),
            "split": config.split,
            "episode_count": len(episodes),
        },
        "paired_design": {
            "independent_unit": "episode",
            "paired_by": ["scenario_id", "seed"],
            "update_events_are_independent_replicates": False,
            "common_random_numbers": True,
            "truth_used_offline_only": True,
            "filter_reward_or_policy_changed": False,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "test_data_used": False,
        },
        "treatments": treatment_manifests,
        "logs": {
            "update_events": str(logs_dir / "update_events.csv"),
            "episode_summary": str(logs_dir / "episode_summary.csv"),
            "quality_strata": str(logs_dir / "quality_strata.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"cooperative_update_quality": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"cooperative update-quality audit failed: {audit['checks']}")
    return {
        "output_dir": str(config.output_dir),
        "reference_scale": str(reference_scale),
        "episode_count": len(episodes),
        "accepted_update_counts": audit["checks"]["accepted_update_counts"],
        "schema_version": EPISODE_SCHEMA_VERSION,
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "decision_gate": audit["decision_gate"],
        "quality_flags": audit["quality_flags"],
        "audit_ok": True,
        "plot": str(plot_path),
    }


__all__ = [
    "CooperativeUpdateQualityAuditConfig",
    "PRACTICAL_EVENT_THRESHOLD_M",
    "classify_update_effect",
    "run_cooperative_update_quality_audit",
]
