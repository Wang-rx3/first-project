"""Paired intervention for train-fitted soft measurement-quality scoring.

The experiment preserves the frozen MAPPO trajectory, bandwidth, channel and
sensor random streams.  It changes only ``measurement_uav_id`` and compares
recorded MAPPO, round-robin, the existing hard quality gate, and a continuous
soft risk score.  Physical LOS/NLOS and true error remain offline labels.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_json
from .bandwidth_intervention import _max_numeric_difference
from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .formal import _copy_calibration_files
from .localization_coupling_intervention import _bootstrap_interval, _mean, _write_table
from .measurement_density_intervention import _NUMERIC_TOLERANCE, _bandwidth_difference
from .quality_aware_measurement_intervention import (
    _HIGH_IS_GOOD,
    _LOW_IS_GOOD,
    _RISK_FIELDS,
    _neighbor_covariance_trace,
    _risk,
    _run_episode,
    fit_training_quality_thresholds,
    intervene_quality_measurement_selection,
    quality_candidate_scores,
)


TREATMENTS = (
    "recorded_mappo",
    "round_robin_candidate",
    "quality_aware_candidate",
    "soft_quality_candidate",
)
POLICY_PREFIX = "soft_quality_measurement_intervention"
SOFT_TREATMENT = "soft_quality_candidate"


@dataclass(frozen=True)
class SoftQualityMeasurementInterventionConfig:
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
            raise ValueError("soft quality measurement intervention is validation-only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _default_logs_dir(reference_scale: Path) -> Path:
    return Path("logs/preliminary") / reference_scale.parent.name / "soft_quality_measurement"


def _default_plots_dir(reference_scale: Path) -> Path:
    return Path("plots/preliminary") / reference_scale.parent.name / "soft_quality_measurement"


def _prepare_output(config: SoftQualityMeasurementInterventionConfig) -> tuple[Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir else _default_logs_dir(Path(config.reference_scale))
    plots = Path(config.plots_dir) if config.plots_dir else _default_plots_dir(Path(config.reference_scale))
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"intervention output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _training_covariance_map(episode_root: Path) -> dict[tuple[float, str], float]:
    output: dict[tuple[float, str], float] = {}
    for row in _read_csv(episode_root / "estimates.csv"):
        covariance = np.asarray(json.loads(row["covariance_flat_json"]), dtype=float)
        side = int(round(np.sqrt(covariance.size)))
        covariance = covariance.reshape(side, side)
        output[(round(float(row["timestamp"]), 9), str(row["uav_id"]))] = float(
            np.trace(covariance[:3, :3])
        )
    return output


def _score_from_values(values: Mapping[str, float], thresholds: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    components = {field: _risk(float(values[field]), field, thresholds) for field in _RISK_FIELDS}
    weights = thresholds["soft_score"]["weights"]
    score = float(sum(float(weights[field]) * components[field] for field in _RISK_FIELDS))
    return score, components


def fit_training_soft_quality_model(reference_scale: Path) -> dict[str, Any]:
    """Fit continuous normalization and the none threshold from train only."""

    thresholds = fit_training_quality_thresholds(reference_scale)
    equal_weight = 1.0 / len(_RISK_FIELDS)
    thresholds["soft_score"] = {
        "weights": {field: equal_weight for field in _RISK_FIELDS},
        "weight_source": "predeclared_equal_weights",
        "missing_neighbor_covariance_risk": 1.0,
        "none_threshold_quantile": 0.90,
    }
    train_root = Path(reference_scale) / "train"
    scores: list[float] = []
    connected_rows = 0
    for episode_root in sorted(path for path in train_root.glob("episode_*") if path.is_dir()):
        covariance_by_key = _training_covariance_map(episode_root)
        for row in _read_csv(episode_root / "links.csv"):
            if int(float(row["candidate_edge"])) != 1 or str(row["link_state"]) != "connected":
                continue
            connected_rows += 1
            source = str(row["src_uav_id"])
            timestamp = round(float(row["timestamp"]), 9)
            covariance = covariance_by_key.get((timestamp, source))
            values = {
                "rssi_dbm": float(row["rssi"]),
                "snr_db": float(row["snr"]),
                "packet_loss": float(row["packet_loss"]),
                "delay_s": float(row["delay"]),
                "aoi_s": float(row["aoi"]),
                "queue_length_packets": float(row["queue_length"]),
                "neighbor_covariance_trace_m2": (
                    float(covariance)
                    if covariance is not None and np.isfinite(covariance)
                    else float(thresholds["quantiles"]["neighbor_covariance_trace_m2"]["q75"])
                ),
            }
            score, _components = _score_from_values(values, thresholds)
            scores.append(score)
    if connected_rows == 0 or not scores or not np.all(np.isfinite(scores)):
        raise ValueError("training split has no finite connected-candidate soft scores")
    score_array = np.asarray(scores, dtype=float)
    none_threshold = float(np.quantile(score_array, 0.90))
    thresholds["soft_score"].update(
        connected_candidate_row_count=connected_rows,
        training_score_q25=float(np.quantile(score_array, 0.25)),
        training_score_q50=float(np.quantile(score_array, 0.50)),
        training_score_q75=float(np.quantile(score_array, 0.75)),
        training_score_q90=none_threshold,
        none_threshold=none_threshold,
        selection_rule=(
            "select the connected actor-visible candidate with minimum continuous risk; "
            "select none only when the minimum exceeds the frozen train q90 threshold"
        ),
    )
    thresholds["eligibility_soft"] = {"link_state": "connected only; no per-metric hard gate"}
    thresholds["validation_used_for_fitting"] = False
    thresholds["test_used_for_fitting"] = False
    return thresholds


def soft_quality_candidate_scores(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Score actor-visible candidates without physical LOS, truth, or future state."""

    hard_rows = quality_candidate_scores(observation, uav_ids, thresholds)
    output: list[dict[str, Any]] = []
    for row in hard_rows:
        values = {field: float(row[field]) for field in _HIGH_IS_GOOD + _LOW_IS_GOOD}
        covariance_available = bool(row["neighbor_covariance_available"])
        values["neighbor_covariance_trace_m2"] = (
            float(row["neighbor_covariance_trace_m2"])
            if covariance_available
            else float(thresholds["quantiles"]["neighbor_covariance_trace_m2"]["q75"])
        )
        score, components = _score_from_values(values, thresholds)
        connected = str(row["link_state"]) == "connected"
        output.append(
            {
                **row,
                "eligible": connected,
                "quality_risk_score": score,
                "risk_components": components,
                "hard_gate_eligible": bool(row["eligible"]),
                "actor_visible_only": True,
            }
        )
    return sorted(output, key=lambda row: (float(row["quality_risk_score"]), str(row["source_uav_id"])))


def intervene_soft_quality_measurement_selection(
    recorded_actions: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    uav_ids: Sequence[str],
    treatment: str,
    thresholds: Mapping[str, Any],
    round_robin_cursors: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if treatment != SOFT_TREATMENT:
        return intervene_quality_measurement_selection(
            recorded_actions, observations, uav_ids, treatment, thresholds, round_robin_cursors
        )
    none_threshold = float(thresholds["soft_score"]["none_threshold"])
    actions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for uav_id in uav_ids:
        recorded = recorded_actions[uav_id]
        scores = soft_quality_candidate_scores(observations[uav_id], uav_ids, thresholds)
        connected = [row for row in scores if bool(row["eligible"])]
        best = connected[0] if connected else None
        selected = (
            str(best["source_uav_id"])
            if best is not None and float(best["quality_risk_score"]) <= none_threshold
            else "none"
        )
        recorded_source = str(recorded.get("measurement_uav_id", "none"))
        actions[uav_id] = {
            "measurement_uav_id": selected,
            "bandwidth": {
                str(key): float(value) for key, value in dict(recorded.get("bandwidth", {})).items()
            },
            "delta_v": np.asarray(recorded["delta_v"], dtype=float).reshape(3),
        }
        metadata[uav_id] = {
            "candidate_count": len(scores),
            "candidate_uav_ids": [row["source_uav_id"] for row in scores],
            "candidate_available": bool(scores),
            "eligible_candidate_count": len(connected),
            "recorded_measurement_uav_id": recorded_source,
            "selected_measurement_uav_id": selected,
            "selected_non_none": selected != "none",
            "selection_changed": selected != recorded_source,
            "selected_quality_risk_score": (
                float(best["quality_risk_score"]) if selected != "none" and best is not None else -1.0
            ),
            "candidate_quality_json": json.dumps(scores, sort_keys=True),
            "actor_visible_only": True,
            "uses_physical_los": False,
            "uses_truth_error": False,
            "uses_future_information": False,
        }
    return actions, metadata


def _paired_contrasts(
    rows: Sequence[Mapping[str, Any]], config: SoftQualityMeasurementInterventionConfig
) -> list[dict[str, Any]]:
    metrics = (
        "selection_fraction_given_candidate",
        "accepted_update_count",
        "beneficial_fraction",
        "harmful_fraction",
        "neutral_fraction",
        "nlos_update_fraction",
        "stale_update_fraction",
        "position_rmse_m",
        "position_error_p95_m",
        "j_loc",
        "mean_position_covariance_trace_m2",
        "mean_position_nees",
        "nees_95_coverage",
    )
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    for comparator in TREATMENTS[:-1]:
        for metric in metrics:
            baseline = np.asarray([float(by_key[(seed, comparator)][metric]) for seed in seeds])
            soft = np.asarray([float(by_key[(seed, SOFT_TREATMENT)][metric]) for seed in seeds])
            difference = soft - baseline
            low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
            output.append(
                {
                    "contrast": f"{SOFT_TREATMENT}_minus_{comparator}",
                    "metric": metric,
                    "paired_episodes": len(seeds),
                    "comparator_mean": float(baseline.mean()),
                    "soft_quality_candidate_mean": float(soft.mean()),
                    "mean_paired_difference": float(difference.mean()),
                    "median_paired_difference": float(np.median(difference)),
                    "paired_difference_std": float(difference.std(ddof=1)),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_excludes_zero": int(low > 0.0 or high < 0.0),
                }
            )
    return output


def _invariance_checks(
    config: SoftQualityMeasurementInterventionConfig,
    roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    checks = {
        "recorded_reproduces_reference_max_abs_difference": 0.0,
        "cross_treatment_ground_truth_max_abs_difference": 0.0,
        "cross_treatment_obstacle_max_abs_difference": 0.0,
        "cross_treatment_link_kpi_max_abs_difference": 0.0,
        "cross_treatment_packet_event_max_abs_difference": 0.0,
        "cross_treatment_delta_v_max_abs_difference": 0.0,
        "cross_treatment_bandwidth_max_abs_difference": 0.0,
    }
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        reference = Path(config.reference_scale) / config.split / episode_id
        factual = roots["recorded_mappo"] / config.split / episode_id
        checks["recorded_reproduces_reference_max_abs_difference"] = max(
            checks["recorded_reproduces_reference_max_abs_difference"],
            _max_numeric_difference(
                factual / "estimates.csv",
                reference / "estimates.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
            _max_numeric_difference(
                factual / "rewards.csv",
                reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("reward_total", "joint_objective", "normalized_uncertainty"),
            ),
        )
        for treatment in TREATMENTS[1:]:
            comparator = roots[treatment] / config.split / episode_id
            pairs = (
                ("cross_treatment_ground_truth_max_abs_difference", "ground_truth.csv", ("timestamp", "uav_id"), ("x", "y", "z", "vx", "vy", "vz")),
                ("cross_treatment_obstacle_max_abs_difference", "obstacles.csv", ("timestamp", "obstacle_id"), ("x", "y", "z", "vx", "vy", "vz", "radius")),
                ("cross_treatment_link_kpi_max_abs_difference", "links.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length")),
                ("cross_treatment_packet_event_max_abs_difference", "packet_events.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("success", "delay", "aoi_after")),
                ("cross_treatment_delta_v_max_abs_difference", "actions.csv", ("timestamp", "uav_id"), ("delta_vx", "delta_vy", "delta_vz")),
            )
            for name, filename, keys, values in pairs:
                checks[name] = max(
                    checks[name],
                    _max_numeric_difference(
                        factual / filename,
                        comparator / filename,
                        key_columns=keys,
                        value_columns=values,
                    ),
                )
            checks["cross_treatment_bandwidth_max_abs_difference"] = max(
                checks["cross_treatment_bandwidth_max_abs_difference"],
                _bandwidth_difference(factual / "actions.csv", comparator / "actions.csv"),
            )
    return checks


def _audit(
    config: SoftQualityMeasurementInterventionConfig,
    roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    invariants = _invariance_checks(config, roots, episodes)
    contrast_map = {(str(row["contrast"]), str(row["metric"])): row for row in contrasts}
    round_prefix = f"{SOFT_TREATMENT}_minus_round_robin_candidate"
    hard_prefix = f"{SOFT_TREATMENT}_minus_quality_aware_candidate"
    harmful = contrast_map[(round_prefix, "harmful_fraction")]
    rmse = contrast_map[(round_prefix, "position_rmse_m")]
    soft_updates = float(contrast_map[(round_prefix, "accepted_update_count")]["soft_quality_candidate_mean"])
    round_updates = float(contrast_map[(round_prefix, "accepted_update_count")]["comparator_mean"])
    hard_updates = float(contrast_map[(hard_prefix, "accepted_update_count")]["comparator_mean"])
    harmful_reduced = float(harmful["mean_paired_difference"]) < 0.0 and float(harmful["bootstrap_95_ci_high"]) < 0.0
    rmse_reduced = float(rmse["mean_paired_difference"]) < 0.0 and float(rmse["bootstrap_95_ci_high"]) < 0.0
    retains_more_than_hard = soft_updates > hard_updates
    if retains_more_than_hard and harmful_reduced and rmse_reduced:
        decision = "soft_quality_effective_freeze_then_add_random_seeds"
    elif not retains_more_than_hard:
        decision = "soft_none_threshold_still_too_strict_reassess_on_train_only"
    elif harmful_reduced and not rmse_reduced:
        decision = "soft_quality_reduces_harm_rmse_inconclusive_add_random_seeds"
    elif float(harmful["mean_paired_difference"]) >= 0.0:
        decision = "soft_visible_risk_not_discriminative_reaudit_score_components"
    else:
        decision = "soft_quality_direction_promising_but_uncertain_add_random_seeds"
    finite = all(
        np.isfinite(float(row[key]))
        for row in episode_rows
        for key in ("position_rmse_m", "position_error_p95_m", "j_loc", "mean_position_nees")
    )
    leakage_free = all(
        int(row["physical_los_used_for_selection"]) == 0
        and int(row["truth_error_used_for_selection"]) == 0
        and int(row["future_information_used_for_selection"]) == 0
        for row in window_rows
        if row["treatment"] == SOFT_TREATMENT
    )
    pairing_ok = all(value <= _NUMERIC_TOLERANCE for value in invariants.values())
    train_only = (
        thresholds.get("source_split") == "train"
        and not thresholds.get("validation_used_for_fitting")
        and not thresholds.get("test_used_for_fitting")
    )
    return {
        "experiment": "paired_actor_visible_soft_quality_measurement_intervention",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "paired_episode_count": len(episodes),
        "event_rows_used_as_independent_samples": False,
        "test_data_used": False,
        "checks": {
            **invariants,
            "hard_quality_accepted_updates_mean": hard_updates,
            "soft_quality_accepted_updates_mean": soft_updates,
            "round_robin_accepted_updates_mean": round_updates,
            "soft_to_hard_update_ratio": soft_updates / max(hard_updates, 1e-12),
            "soft_to_round_robin_update_ratio": soft_updates / max(round_updates, 1e-12),
            "soft_minus_round_robin_harmful_fraction": float(harmful["mean_paired_difference"]),
            "harmful_fraction_bootstrap_95_ci": [float(harmful["bootstrap_95_ci_low"]), float(harmful["bootstrap_95_ci_high"])],
            "soft_minus_round_robin_position_rmse_m": float(rmse["mean_paired_difference"]),
            "position_rmse_bootstrap_95_ci_m": [float(rmse["bootstrap_95_ci_low"]), float(rmse["bootstrap_95_ci_high"])],
            "none_threshold": float(thresholds["soft_score"]["none_threshold"]),
        },
        "score_parameters_fitted_from_train_only": bool(train_only),
        "selection_information_leakage_free": bool(leakage_free),
        "deterministic_pairing_ok": bool(pairing_ok),
        "all_metrics_finite": bool(finite),
        "decision_gate": decision,
        "overall_ok": bool(train_only and leakage_free and pairing_ok and finite),
        "interpretation_boundary": (
            "The equal-weight score and train-q90 none threshold were frozen before validation replay. "
            "Physical LOS/NLOS and true error are offline labels only; test remains sealed."
        ),
    }


def _plot(
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "recorded_mappo": "#2563eb",
        "round_robin_candidate": "#dc2626",
        "quality_aware_candidate": "#f59e0b",
        "soft_quality_candidate": "#16a34a",
    }
    labels = {
        "recorded_mappo": "Recorded MAPPO",
        "round_robin_candidate": "Round-robin",
        "quality_aware_candidate": "Hard quality",
        "soft_quality_candidate": "Soft quality",
    }
    short_labels = ["MAPPO", "Round-robin", "Hard", "Soft"]
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in episode_rows}
    seeds = sorted({int(row["seed"]) for row in episode_rows})
    panels = (
        ("selection_fraction_given_candidate", "Non-none selection fraction"),
        ("accepted_update_count", "Accepted cooperative updates"),
        ("harmful_fraction", "Harmful update fraction"),
        ("position_rmse_m", "Position RMSE (m)"),
        ("position_error_p95_m", "Position error P95 (m)"),
        ("mean_position_nees", "Mean position NEES"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.ravel()[:6], panels):
        for seed in seeds:
            axis.plot(
                range(len(TREATMENTS)),
                [float(by_key[(seed, treatment)][metric]) for treatment in TREATMENTS],
                color="#94a3b8",
                alpha=0.30,
                linewidth=0.75,
            )
        means = [
            _mean(float(row[metric]) for row in episode_rows if row["treatment"] == treatment)
            for treatment in TREATMENTS
        ]
        axis.scatter(range(len(TREATMENTS)), means, s=55, c=[colors[t] for t in TREATMENTS], zorder=3)
        axis.set_xticks(range(len(TREATMENTS)), short_labels, rotation=15)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    accepted = [row for row in event_rows if int(row["accepted"])]
    axis = axes.ravel()[6]
    x = np.arange(3)
    width = 0.19
    for offset, treatment in enumerate(TREATMENTS):
        selected = [row for row in accepted if row["treatment"] == treatment]
        fractions = [
            _mean(row["effect_class"] == category for row in selected)
            for category in ("beneficial", "harmful", "approximately_neutral")
        ]
        axis.bar(x + (offset - 1.5) * width, fractions, width, color=colors[treatment], label=labels[treatment])
    axis.set_xticks(x, ["Beneficial", "Harmful", "Neutral"])
    axis.set_ylabel("Fraction of accepted updates")
    axis.set_title("Offline update-effect labels")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)
    axis = axes.ravel()[7]
    for treatment in TREATMENTS:
        selected = [row for row in accepted if row["treatment"] == treatment]
        axis.scatter(
            [float(row["message_age_s"]) for row in selected],
            [float(row["true_error_change_after_minus_before_m"]) for row in selected],
            s=9,
            alpha=0.28,
            color=colors[treatment],
            label=labels[treatment],
            rasterized=True,
        )
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.axvline(0.5, color="#f59e0b", linestyle="--", linewidth=1)
    axis.set_xlabel("Message age (s), offline diagnostic")
    axis.set_ylabel("Error change after-before (m)")
    axis.set_title("Freshness and update effect")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    target = plots_dir / "soft_quality_measurement.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def run_soft_quality_measurement_intervention(
    config: SoftQualityMeasurementInterventionConfig,
) -> dict[str, Any]:
    reference_scale = Path(config.reference_scale)
    manifest = _read_json(reference_scale / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(manifest.get("policy_source")) != "mappo" or int(manifest.get("uav_count", 0)) != 3:
        raise ValueError("reference must be the frozen three-UAV MAPPO run")
    episodes = [
        item
        for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and item.get("split") == config.split
    ]
    if len(episodes) != 20:
        raise ValueError(f"the frozen intervention requires exactly 20 validation episodes, found {len(episodes)}")
    logs_dir, plots_dir = _prepare_output(config)
    thresholds = fit_training_soft_quality_model(reference_scale)
    _write_json(logs_dir / "soft_quality_model.json", thresholds)
    formal_config = _reference_formal_config(
        manifest, Path(config.output_dir), force=False, progress=config.progress
    )
    roots: dict[str, Path] = {}
    treatment_manifests: dict[str, Any] = {}
    episode_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        treatment_root = Path(config.output_dir) / "treatments" / treatment
        scale_root = treatment_root / "scale_3"
        scale_root.mkdir(parents=True, exist_ok=True)
        roots[treatment] = scale_root
        treatment_config = _reference_formal_config(
            manifest, treatment_root, force=False, progress=config.progress
        )
        _copy_calibration_files(scale_root, treatment_config)
        normalization = reference_scale / "normalization_stats.json"
        if normalization.exists():
            shutil.copy2(normalization, scale_root / "normalization_stats.json")
        summaries: list[dict[str, Any]] = []
        metrics_rows: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metrics, windows, events = _run_episode(
                config=config,
                formal_config=formal_config,
                thresholds=thresholds,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
                selector=intervene_soft_quality_measurement_selection,
                policy_prefix=POLICY_PREFIX,
            )
            summaries.append(summary)
            metrics_rows.append(metrics)
            window_rows.extend(windows)
            event_rows.extend(events)
        episode_rows.extend(metrics_rows)
        treatment_manifest = {
            "dataset_id": f"python_soft_quality_measurement_{treatment}",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "engine": "python_sim",
            "result_status": "engineering_mechanism_diagnostic_only",
            "policy_source": f"{POLICY_PREFIX}_{treatment}",
            "uav_count": 3,
            "reference": {
                "scale_root": str(reference_scale),
                "manifest_sha256": _sha256(reference_scale / "manifest.json"),
                "split": config.split,
            },
            "intervention": {
                "treatment": treatment,
                "changed_action_field_only": "measurement_uav_id",
                "soft_score_source": "train",
                "actor_visible_only": treatment == SOFT_TREATMENT,
                "physical_los_or_truth_used_for_selection": False,
                "uses_future_information": False,
                "uses_test_data": False,
                "retrained_mappo": False,
            },
            "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", treatment_manifest)
        _write_json(
            treatment_root / "manifest.json",
            {
                "dataset_id": treatment_manifest["dataset_id"],
                "schema_version": EPISODE_SCHEMA_VERSION,
                "scales": {
                    "3": {
                        "uav_count": 3,
                        "path": "scale_3",
                        "manifest": "scale_3/manifest.json",
                        "episode_count": len(summaries),
                    }
                },
            },
        )
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {"scale_root": str(scale_root), "validation_ok": True}
    contrasts = _paired_contrasts(episode_rows, config)
    audit = _audit(config, roots, episodes, episode_rows, window_rows, contrasts, thresholds)
    _write_table(logs_dir / "window_metrics.csv", window_rows)
    _write_table(logs_dir / "update_events.csv", event_rows)
    _write_table(logs_dir / "episode_metrics.csv", episode_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(episode_rows, event_rows, plots_dir)
    root_manifest = {
        "dataset_id": "python_soft_quality_measurement_seed2026",
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
            "common_random_numbers": True,
            "event_rows_are_independent_replicates": False,
            "score_parameter_source": "train",
            "validation_used_for_score_fitting": False,
            "test_data_used": False,
            "changed_field_only": "measurement_uav_id",
        },
        "soft_quality_model": thresholds,
        "treatments": treatment_manifests,
        "logs": {
            "soft_quality_model": str(logs_dir / "soft_quality_model.json"),
            "window_metrics": str(logs_dir / "window_metrics.csv"),
            "update_events": str(logs_dir / "update_events.csv"),
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"soft_quality_measurement": str(plot_path)},
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(f"soft quality measurement audit failed: {audit['checks']}")
    return {
        "output_dir": str(config.output_dir),
        "reference_scale": str(reference_scale),
        "paired_episodes": len(episodes),
        "treatments": list(TREATMENTS),
        "schema_version": EPISODE_SCHEMA_VERSION,
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "decision_gate": audit["decision_gate"],
        "audit_ok": True,
        "plot": str(plot_path),
    }


__all__ = [
    "SoftQualityMeasurementInterventionConfig",
    "TREATMENTS",
    "fit_training_soft_quality_model",
    "soft_quality_candidate_scores",
    "intervene_soft_quality_measurement_selection",
    "run_soft_quality_measurement_intervention",
]
