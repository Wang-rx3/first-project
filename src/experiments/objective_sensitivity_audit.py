"""Read-only audit of objective, localization, communication, and reward closure.

This module deliberately consumes only the frozen validation outputs of the
five-seed quality-aware MAPPO experiment.  It does not replay the simulator,
fit a policy, read the test split, or use ground-truth error to construct an
objective or an actor input.  Ground-truth positions are used only to produce
offline diagnostic RMSE/P95 labels.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..simulation.ekf import EKFConfig


TREATMENTS = ("baseline_mappo", "quality_aware_mappo")
SEED_RE = re.compile(r"^seed_(\d+)$")

# These are diagnostic variables, not additional manuscript symbols.  The
# core objective remains J = J_loc + J_com; the other fields explain how the
# logged objective and reward respond to the two branches.
AUDIT_METRICS = (
    "position_rmse_m",
    "position_error_p95_m",
    "covariance_position_trace",
    "normalized_uncertainty",
    "j_loc",
    "j_com",
    "joint_objective",
    "mean_reward",
    "collision_penalty",
    "energy_penalty",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "connectivity",
    "accepted_update_count",
    "beneficial_fraction",
    "harmful_fraction",
    "neutral_fraction",
    "nlos_update_fraction",
    "stale_update_fraction",
    "none_selection_rate",
    "measurement_count",
    "mean_selected_quality_risk",
)

COMMUNICATION_COMPONENTS = (
    "j_com_loss",
    "j_com_aoi",
    "j_com_delay",
    "j_com_connectivity",
)

_LOWER_IS_BETTER = {
    "position_rmse_m",
    "position_error_p95_m",
    "covariance_position_trace",
    "normalized_uncertainty",
    "j_loc",
    "j_com",
    "joint_objective",
    "collision_penalty",
    "energy_penalty",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "harmful_fraction",
    "nlos_update_fraction",
    "stale_update_fraction",
    "none_selection_rate",
    "mean_selected_quality_risk",
}

_HIGHER_IS_BETTER = {
    "mean_reward",
    "connectivity",
    "beneficial_fraction",
}


@dataclass(frozen=True)
class ObjectiveSensitivityAuditConfig:
    """Configuration for the validation-only objective sensitivity audit."""

    reference_root: Path
    logs_output: Path | None = None
    plots_output: Path | None = None
    split: str = "validation"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _mean(values: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _position_errors(episode_root: Path) -> list[float]:
    truth_rows = _read_csv(episode_root / "ground_truth.csv")
    estimate_rows = _read_csv(episode_root / "estimates.csv")
    truth: dict[tuple[str, float], np.ndarray] = {}
    for row in truth_rows:
        key = (str(row.get("uav_id", "")), round(_number(row.get("timestamp")), 6))
        position = np.asarray(
            [_number(row.get(axis)) for axis in ("x", "y", "z")], dtype=float
        )
        if np.isfinite(position).all():
            truth[key] = position
    errors: list[float] = []
    for row in estimate_rows:
        key = (str(row.get("uav_id", "")), round(_number(row.get("timestamp")), 6))
        estimate = np.asarray(
            [_number(row.get(axis)) for axis in ("x", "y", "z")], dtype=float
        )
        if key in truth and np.isfinite(estimate).all():
            errors.append(float(np.linalg.norm(estimate - truth[key])))
    return [error for error in errors if math.isfinite(error)]


def _position_diagnostics(episode_root: Path) -> tuple[float, float]:
    errors = _position_errors(episode_root)
    if not errors:
        return float("nan"), float("nan")
    values = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(values**2))), float(np.quantile(values, 0.95))


def _event_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    accepted = [row for row in rows if _as_bool(row.get("accepted", False))]
    deltas = np.asarray(
        [
            _number(row.get("true_position_error_after_m"))
            - _number(row.get("true_position_error_before_m"))
            for row in accepted
        ],
        dtype=float,
    )
    deltas = deltas[np.isfinite(deltas)]
    beneficial = float(np.sum(deltas < -0.001))
    harmful = float(np.sum(deltas > 0.001))
    neutral = float(max(0, deltas.size - int(beneficial) - int(harmful)))
    denominator = float(max(1, deltas.size))
    stale_limit = float(EKFConfig().stale_message_limit_s)
    return {
        "accepted_update_count": float(deltas.size),
        "beneficial_fraction": beneficial / denominator,
        "harmful_fraction": harmful / denominator,
        "neutral_fraction": neutral / denominator,
        "nlos_update_fraction": _mean(
            [1.0 - float(_as_bool(row.get("los", True))) for row in accepted], 0.0
        ),
        "stale_update_fraction": _mean(
            [
                1.0
                if _number(row.get("message_age_s"), 0.0) > stale_limit
                else 0.0
                for row in accepted
            ],
            0.0,
        ),
    }


def _quality_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "none_selection_rate": 0.0,
            "measurement_count": 0.0,
            "mean_selected_quality_risk": -1.0,
        }
    selected = [
        row for row in rows if str(row.get("measurement_uav_id", "none")) != "none"
    ]
    risks = [
        _number(row.get("selected_quality_risk"))
        for row in selected
        if math.isfinite(_number(row.get("selected_quality_risk")))
    ]
    return {
        "none_selection_rate": 1.0 - float(len(selected) / len(rows)),
        "measurement_count": float(len(selected)),
        "mean_selected_quality_risk": _mean(risks, -1.0),
    }


def _reward_and_objective_metrics(
    episode_root: Path,
    metrics_row: Mapping[str, Any],
) -> dict[str, float]:
    rewards = _read_csv(episode_root / "rewards.csv")
    scenario = _read_json(episode_root / "scenario.json")
    reward_config = dict(scenario.get("reward_config", {}))
    weights = dict(reward_config.get("weights", {}))
    w_unc = _number(weights.get("uncertainty", weights.get("unc", 0.35)), 0.35)
    w_loss = _number(weights.get("loss", 0.0), 0.0)
    w_aoi = _number(weights.get("aoi", 0.0), 0.0)
    w_delay = _number(weights.get("delay", 0.0), 0.0)
    w_net = _number(
        weights.get("network", weights.get("connectivity", weights.get("net", 0.0))),
        0.0,
    )

    calculated_jloc: list[float] = []
    calculated_jcom: list[float] = []
    calculated_joint: list[float] = []
    logged_joint: list[float] = []
    reward_values: list[float] = []
    collision_penalties: list[float] = []
    energy_penalties: list[float] = []
    objective_errors: list[float] = []
    reward_errors: list[float] = []
    component_sums = {name: [] for name in COMMUNICATION_COMPONENTS}

    for row in rewards:
        uncertainty = _number(row.get("normalized_uncertainty"), 0.0)
        loss = _number(row.get("normalized_packet_loss"), 0.0)
        aoi = _number(row.get("normalized_aoi"), 0.0)
        delay = _number(row.get("normalized_delay"), 0.0)
        connectivity = _number(row.get("normalized_connectivity"), 0.0)
        jloc = w_unc * uncertainty
        components = {
            "j_com_loss": w_loss * loss,
            "j_com_aoi": w_aoi * aoi,
            "j_com_delay": w_delay * delay,
            "j_com_connectivity": -w_net * connectivity,
        }
        jcom = sum(components.values())
        calculated = jloc + jcom
        logged = _number(row.get("joint_objective"), float("nan"))
        reward = _number(row.get("reward_total"), float("nan"))
        collision = max(0.0, -_number(row.get("reward_collision"), 0.0))
        energy = max(0.0, -_number(row.get("reward_energy"), 0.0))
        calculated_reward = -(logged + collision + energy)
        calculated_jloc.append(jloc)
        calculated_jcom.append(jcom)
        calculated_joint.append(calculated)
        logged_joint.append(logged)
        reward_values.append(reward)
        collision_penalties.append(collision)
        energy_penalties.append(energy)
        objective_errors.append(abs(logged - calculated))
        reward_errors.append(abs(reward - calculated_reward))
        for name, value in components.items():
            component_sums[name].append(value)

    logged_jloc = _number(metrics_row.get("j_loc"), _mean(calculated_jloc))
    logged_jcom = _number(metrics_row.get("j_com"), _mean(logged_joint) - logged_jloc)
    logged_joint_value = _number(metrics_row.get("joint_objective"), _mean(logged_joint))
    return {
        "normalized_uncertainty": _mean(
            [_number(row.get("normalized_uncertainty"), 0.0) for row in rewards], 0.0
        ),
        "j_loc": logged_jloc,
        "j_com": logged_jcom,
        "joint_objective": logged_joint_value,
        "mean_reward": _mean(reward_values, _number(metrics_row.get("mean_reward"))),
        "collision_penalty": _mean(collision_penalties, 0.0),
        "energy_penalty": _mean(energy_penalties, 0.0),
        "j_com_loss": _mean(component_sums["j_com_loss"], 0.0),
        "j_com_aoi": _mean(component_sums["j_com_aoi"], 0.0),
        "j_com_delay": _mean(component_sums["j_com_delay"], 0.0),
        "j_com_connectivity": _mean(component_sums["j_com_connectivity"], 0.0),
        "objective_identity_max_abs_error": max(objective_errors, default=float("nan")),
        "reward_identity_max_abs_error": max(reward_errors, default=float("nan")),
        "objective_formula_mean_abs_error": abs(
            _mean(calculated_jloc) + _mean(calculated_jcom) - logged_joint_value
        ),
    }


def _episode_row(
    branch_scale: Path,
    treatment: str,
    base_seed: int,
    metrics_row: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    split = str(metrics_row.get("split", ""))
    episode_id = str(metrics_row.get("episode_id", ""))
    episode_root = branch_scale / split / episode_id
    if not episode_root.is_dir():
        raise FileNotFoundError(f"missing episode directory: {episode_root}")
    position_rmse, position_p95 = _position_diagnostics(episode_root)
    reward_metrics = _reward_and_objective_metrics(episode_root, metrics_row)
    event_metrics = _event_summary(events)
    quality_metrics = _quality_summary(quality_rows)
    row: dict[str, Any] = {
        "base_seed": int(base_seed),
        "treatment": treatment,
        "split": split,
        "episode_id": episode_id,
        "scenario_id": str(metrics_row.get("scenario_id", "")),
        "uav_count": int(_number(metrics_row.get("uav_count"), 0.0)),
        "position_rmse_m": position_rmse,
        "position_error_p95_m": position_p95,
        "covariance_position_trace": _number(
            metrics_row.get("covariance_position_trace"), float("nan")
        ),
        "packet_loss": _number(metrics_row.get("packet_loss"), float("nan")),
        "delay_s": _number(metrics_row.get("delay_s"), float("nan")),
        "aoi_s": _number(metrics_row.get("aoi_s"), float("nan")),
        "connectivity": _number(metrics_row.get("connectivity"), float("nan")),
        "energy_proxy": _number(metrics_row.get("energy_proxy"), float("nan")),
        "collision_rate": _number(metrics_row.get("collision_rate"), float("nan")),
        "termination_reason": str(metrics_row.get("termination_reason", "")),
        **reward_metrics,
        **event_metrics,
        **quality_metrics,
    }
    for name in AUDIT_METRICS:
        if not math.isfinite(_number(row.get(name))):
            raise ValueError(
                f"non-finite {name} in {treatment}/{split}/{episode_id}"
            )
    return row


def _load_episode_rows(
    reference_root: Path,
    seeds: Sequence[int],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = reference_root / f"seed_{seed}"
        for treatment in TREATMENTS:
            branch_scale = seed_root / treatment / "scale_3"
            diagnostics = branch_scale / "diagnostics"
            metrics_rows = _read_csv(diagnostics / "episode_metrics.csv")
            events = _read_jsonl(diagnostics / "cooperative_updates.jsonl")
            quality = _read_jsonl(diagnostics / "measurement_quality.jsonl")
            selected_metrics = [row for row in metrics_rows if str(row.get("split")) == split]
            if not selected_metrics:
                raise ValueError(f"no {split} rows in {diagnostics / 'episode_metrics.csv'}")
            for metrics_row in selected_metrics:
                episode_id = str(metrics_row.get("episode_id", ""))
                event_rows = [
                    row
                    for row in events
                    if str(row.get("split", "")) == split
                    and str(row.get("episode_id", "")) == episode_id
                ]
                quality_rows = [
                    row
                    for row in quality
                    if str(row.get("split", "")) == split
                    and str(row.get("episode_id", "")) == episode_id
                ]
                rows.append(
                    _episode_row(
                        branch_scale,
                        treatment,
                        seed,
                        metrics_row,
                        event_rows,
                        quality_rows,
                    )
                )
    return rows


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["base_seed"]), str(row["treatment"])), []).append(row)
    return grouped


def _seed_sensitivity(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    grouped = _group_rows(rows)
    output: list[dict[str, Any]] = []
    for seed in seeds:
        baseline = grouped[(int(seed), "baseline_mappo")]
        quality = grouped[(int(seed), "quality_aware_mappo")]
        baseline_by_id = {str(row["episode_id"]): row for row in baseline}
        quality_by_id = {str(row["episode_id"]): row for row in quality}
        if baseline_by_id.keys() != quality_by_id.keys():
            raise ValueError(f"unpaired validation episodes for seed {seed}")
        result: dict[str, Any] = {
            "base_seed": int(seed),
            "baseline_episode_count": len(baseline),
            "quality_aware_episode_count": len(quality),
        }
        for metric in AUDIT_METRICS:
            baseline_value = _mean([_number(row.get(metric)) for row in baseline])
            quality_value = _mean([_number(row.get(metric)) for row in quality])
            result[f"baseline_{metric}"] = baseline_value
            result[f"quality_aware_{metric}"] = quality_value
            result[f"delta_{metric}"] = quality_value - baseline_value
        output.append(result)
    return output


def _bootstrap_interval(
    values: Sequence[float], samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    if array.size == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _seed_metric_summary(
    seed_rows: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(bootstrap_seed)
    summary: dict[str, dict[str, Any]] = {}
    for metric in AUDIT_METRICS:
        values = np.asarray([_number(row[f"delta_{metric}"]) for row in seed_rows], dtype=float)
        ci_low, ci_high = _bootstrap_interval(values, bootstrap_samples, rng)
        baseline = np.asarray(
            [_number(row[f"baseline_{metric}"]) for row in seed_rows], dtype=float
        )
        summary[metric] = {
            "baseline_mean": float(np.mean(baseline)),
            "quality_aware_mean": float(
                np.mean([_number(row[f"quality_aware_{metric}"]) for row in seed_rows])
            ),
            "mean_delta_quality_minus_baseline": float(np.mean(values)),
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "improvement_direction": (
                "lower" if metric in _LOWER_IS_BETTER
                else "higher" if metric in _HIGHER_IS_BETTER
                else "diagnostic_only"
            ),
            "improved_seed_count": (
                int(np.sum(values < 0.0))
                if metric in _LOWER_IS_BETTER
                else int(np.sum(values > 0.0))
                if metric in _HIGHER_IS_BETTER
                else None
            ),
            "seed_count": int(values.size),
        }
    return summary


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _correlation_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    episode_pairs = (
        ("position_rmse_m", "j_loc"),
        ("position_rmse_m", "covariance_position_trace"),
        ("position_rmse_m", "j_com"),
        ("position_rmse_m", "normalized_uncertainty"),
        ("j_com", "accepted_update_count"),
    )
    seed_pairs = (
        ("position_rmse_m", "j_loc"),
        ("position_rmse_m", "covariance_position_trace"),
        ("position_rmse_m", "j_com"),
        ("j_loc", "j_com"),
        ("accepted_update_count", "j_com"),
    )
    for scope, rows, pairs, suffix in (
        ("episode_level_descriptive", episode_rows, episode_pairs, "raw"),
        ("seed_level_paired_delta", seed_rows, seed_pairs, "delta"),
    ):
        for x_metric, y_metric in pairs:
            if suffix == "raw":
                x_values = [_number(row.get(x_metric)) for row in rows]
                y_values = [_number(row.get(y_metric)) for row in rows]
            else:
                x_values = [_number(row.get(f"delta_{x_metric}")) for row in rows]
                y_values = [_number(row.get(f"delta_{y_metric}")) for row in rows]
            correlation = _pearson(x_values, y_values)
            output.append(
                {
                    "scope": scope,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "sample_count": sum(
                        math.isfinite(x) and math.isfinite(y)
                        for x, y in zip(x_values, y_values)
                    ),
                    "correlation": "" if correlation is None else correlation,
                    "inference_unit": (
                        "descriptive episode rows; not independent"
                        if scope == "episode_level_descriptive"
                        else "base training seed"
                    ),
                }
            )
    return output


def _decision(
    metrics: Mapping[str, Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rmse = metrics["position_rmse_m"]
    jloc = metrics["j_loc"]
    jcom = metrics["j_com"]
    reward = metrics["mean_reward"]
    baseline_jloc = abs(float(jloc["baseline_mean"]))
    jloc_tolerance = max(1e-6, 0.01 * baseline_jloc)
    required = int(math.ceil(0.8 * len(seed_rows)))
    rmse_improved = (
        float(rmse["mean_delta_quality_minus_baseline"]) < 0.0
        and int(rmse["improved_seed_count"]) >= required
    )
    jloc_practically_insensitive = abs(
        float(jloc["mean_delta_quality_minus_baseline"])
    ) <= jloc_tolerance
    jloc_improved = (
        float(jloc["mean_delta_quality_minus_baseline"]) < -jloc_tolerance
        and int(jloc["improved_seed_count"]) >= required
    )
    jcom_worsened = float(jcom["mean_delta_quality_minus_baseline"]) > 0.0 and float(
        jcom["bootstrap_95_ci_low"]
    ) > 0.0
    reward_worsened = float(reward["mean_delta_quality_minus_baseline"]) < 0.0 and float(
        reward["bootstrap_95_ci_high"]
    ) < 0.0
    all_aligned = (
        float(rmse["mean_delta_quality_minus_baseline"]) < 0.0
        and float(jloc["mean_delta_quality_minus_baseline"]) < 0.0
        and float(jcom["mean_delta_quality_minus_baseline"]) < 0.0
        and float(reward["mean_delta_quality_minus_baseline"]) > 0.0
    )
    ci_supports_all_aligned = all(
        (
            float(metrics[name]["bootstrap_95_ci_high"]) < 0.0
            if name != "mean_reward"
            else float(metrics[name]["bootstrap_95_ci_low"]) > 0.0
        )
        for name in ("position_rmse_m", "j_loc", "j_com", "mean_reward")
    )
    if rmse_improved and jloc_practically_insensitive:
        classification = "objective_break_jloc_insensitive_to_rmse"
        recommendation = (
            "Design an execution-only uncertainty proxy that responds to localization "
            "mean performance before changing the reward or EKF. Also report the observed "
            "communication-cost tradeoff separately."
        )
    elif jloc_improved and (jcom_worsened or reward_worsened):
        classification = "communication_cost_tradeoff"
        recommendation = (
            "Design a communication-cost-controlled measurement selector; keep the current "
            "objective and filter fixed while comparing the added communication burden."
        )
    elif all_aligned and not ci_supports_all_aligned:
        classification = "statistical_evidence_insufficient"
        recommendation = (
            "Freeze the current code and quality rule, then add pre-registered independent "
            "seeds without tuning on validation or test."
        )
    else:
        classification = "mixed_objective_response_needs_review"
        recommendation = (
            "Keep the implementation frozen and inspect the logged objective components and "
            "communication burden before selecting the next controlled intervention."
        )
    return {
        "classification": classification,
        "required_improved_seed_count": required,
        "rmse_improved_by_seed_gate": rmse_improved,
        "jloc_practically_insensitive": jloc_practically_insensitive,
        "jloc_practical_tolerance": jloc_tolerance,
        "jloc_improved_by_practical_gate": jloc_improved,
        "jcom_worsened_with_ci_support": jcom_worsened,
        "reward_worsened_with_ci_support": reward_worsened,
        "all_core_metrics_aligned": all_aligned,
        "all_aligned_ci_supported": ci_supports_all_aligned,
        "recommendation": recommendation,
    }


def _prepare_outputs(config: ObjectiveSensitivityAuditConfig) -> tuple[Path, Path]:
    run_name = Path(config.reference_root).name
    logs_dir = Path(config.logs_output) if config.logs_output is not None else Path("logs/preliminary") / run_name / "objective_sensitivity"
    plots_dir = Path(config.plots_output) if config.plots_output is not None else Path("plots/preliminary") / run_name / "objective_sensitivity"
    for path in (logs_dir, plots_dir):
        if path.exists() and any(path.iterdir()):
            if not config.force:
                raise FileExistsError(
                    f"objective sensitivity output is not empty: {path}; use --force to replace this audit"
                )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return logs_dir, plots_dir


def _plot_audit(
    seed_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline_mappo": "#2563eb", "quality_aware_mappo": "#16a34a"}
    labels = {"baseline_mappo": "Baseline MAPPO", "quality_aware_mappo": "Quality-aware MAPPO"}
    seeds = [int(row["base_seed"]) for row in seed_rows]
    x = np.arange(len(seeds), dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # (a) Position RMSE, paired by training seed.
    axis = axes[0, 0]
    for treatment, color in colors.items():
        prefix = "baseline" if treatment == "baseline_mappo" else "quality_aware"
        values = [_number(row[f"{prefix}_position_rmse_m"]) for row in seed_rows]
        axis.plot(x, values, marker="o", color=color, label=labels[treatment])
    axis.set_title("Position RMSE by seed")
    axis.set_ylabel("RMSE (m)")
    axis.set_xticks(x, [str(seed) for seed in seeds], rotation=35)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    # (b) J_loc, with the practical sensitivity scale visible.
    axis = axes[0, 1]
    for treatment, color in colors.items():
        prefix = "baseline" if treatment == "baseline_mappo" else "quality_aware"
        values = [_number(row[f"{prefix}_j_loc"]) for row in seed_rows]
        axis.plot(x, values, marker="o", color=color, label=labels[treatment])
    axis.set_title(r"$J_{\mathrm{loc}}$ by seed")
    axis.set_ylabel(r"$J_{\mathrm{loc}}$")
    axis.set_xticks(x, [str(seed) for seed in seeds], rotation=35)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    # (c) Objective decomposition, aggregated over seeds.
    axis = axes[0, 2]
    categories = [r"$J_{\mathrm{loc}}$", r"$J_{\mathrm{com}}$", r"$J_t$"]
    metric_names = ["j_loc", "j_com", "joint_objective"]
    category_x = np.arange(len(categories), dtype=float)
    width = 0.36
    for offset, treatment, color in [(-width / 2, "baseline_mappo", colors["baseline_mappo"]), (width / 2, "quality_aware_mappo", colors["quality_aware_mappo"])]:
        values = [float(metrics[name]["baseline_mean"] if treatment == "baseline_mappo" else metrics[name]["quality_aware_mean"]) for name in metric_names]
        axis.bar(category_x + offset, values, width=width, color=color, label=labels[treatment])
    axis.set_title("Core objective decomposition")
    axis.set_xticks(category_x, categories)
    axis.set_ylabel("value")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    # (d) Accepted updates versus communication cost, descriptive at seed level.
    axis = axes[1, 0]
    for treatment, color in colors.items():
        prefix = "baseline" if treatment == "baseline_mappo" else "quality_aware"
        x_values = [_number(row[f"{prefix}_accepted_update_count"]) for row in seed_rows]
        y_values = [_number(row[f"{prefix}_j_com"]) for row in seed_rows]
        axis.scatter(x_values, y_values, color=color, label=labels[treatment], s=42)
    axis.set_title(r"Communication cost vs. accepted updates")
    axis.set_xlabel("accepted cooperative updates / episode")
    axis.set_ylabel(r"$J_{\mathrm{com}}$")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    # (e) RMSE versus covariance trace; episode rows are diagnostic only.
    axis = axes[1, 1]
    for treatment, color in colors.items():
        selected = [row for row in episode_rows if row["treatment"] == treatment]
        axis.scatter(
            [_number(row["covariance_position_trace"]) for row in selected],
            [_number(row["position_rmse_m"]) for row in selected],
            color=color,
            alpha=0.45,
            s=16,
            label=labels[treatment],
        )
    axis.set_title("RMSE versus covariance trace")
    axis.set_xlabel("position covariance trace (m²)")
    axis.set_ylabel("position RMSE (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    # (f) Seed-level paired bootstrap intervals.  All intervals are computed
    # from five seed means, never from individual windows or EKF events.
    axis = axes[1, 2]
    effect_metrics = ("position_rmse_m", "j_loc", "j_com", "mean_reward")
    effect_labels = ("RMSE", r"$J_{\mathrm{loc}}$", r"$J_{\mathrm{com}}$", "team reward")
    positions = np.arange(len(effect_metrics), dtype=float)
    deltas = [float(metrics[name]["mean_delta_quality_minus_baseline"]) for name in effect_metrics]
    lows = [float(metrics[name]["bootstrap_95_ci_low"]) for name in effect_metrics]
    highs = [float(metrics[name]["bootstrap_95_ci_high"]) for name in effect_metrics]
    lower_errors = np.asarray(deltas) - np.asarray(lows)
    upper_errors = np.asarray(highs) - np.asarray(deltas)
    axis.errorbar(
        positions,
        deltas,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        color="#111827",
        ecolor="#111827",
        capsize=4,
    )
    axis.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1)
    axis.set_title("Quality-aware minus baseline")
    axis.set_xticks(positions, effect_labels)
    axis.set_ylabel("paired difference (validation)")
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Objective sensitivity audit (validation only; diagnostic)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_objective_sensitivity_audit(
    config: ObjectiveSensitivityAuditConfig,
) -> dict[str, Any]:
    """Run the validation-only objective sensitivity audit."""

    if config.split != "validation":
        raise ValueError("objective sensitivity audit is validation-only; test is sealed")
    if int(config.bootstrap_samples) < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    reference_root = Path(config.reference_root)
    if not reference_root.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference_root}")

    seed_dirs: list[tuple[int, Path]] = []
    for path in reference_root.iterdir():
        match = SEED_RE.match(path.name)
        if path.is_dir() and match:
            seed_dirs.append((int(match.group(1)), path))
    seed_dirs.sort()
    if not seed_dirs:
        raise ValueError(f"no seed_* directories found below {reference_root}")
    seeds = [seed for seed, _ in seed_dirs]
    for seed, seed_root in seed_dirs:
        for treatment in TREATMENTS:
            branch_scale = seed_root / treatment / "scale_3"
            if not (branch_scale / "diagnostics" / "episode_metrics.csv").is_file():
                raise FileNotFoundError(f"missing diagnostics for {seed}/{treatment}")
            if not (branch_scale / config.split).is_dir():
                raise FileNotFoundError(f"missing validation split for {seed}/{treatment}")

    # This loader constructs paths only under ``validation``.  It never calls
    # a test reader and never writes below the frozen reference root.
    episode_rows = _load_episode_rows(reference_root, seeds, config.split)
    seed_rows = _seed_sensitivity(episode_rows, seeds)
    metrics = _seed_metric_summary(seed_rows, config.bootstrap_samples, config.bootstrap_seed)
    correlation_rows = _correlation_rows(episode_rows, seed_rows)
    decision = _decision(metrics, seed_rows)

    all_finite = all(
        math.isfinite(_number(row.get(metric)))
        for row in episode_rows
        for metric in AUDIT_METRICS
    )
    objective_closure = max(
        _number(row.get("objective_identity_max_abs_error")) for row in episode_rows
    )
    reward_closure = max(
        _number(row.get("reward_identity_max_abs_error")) for row in episode_rows
    )
    formula_closure = max(
        _number(row.get("objective_formula_mean_abs_error")) for row in episode_rows
    )
    paired_counts_ok = all(
        int(row["baseline_episode_count"]) == int(row["quality_aware_episode_count"])
        for row in seed_rows
    )
    closure_ok = max(objective_closure, reward_closure, formula_closure) <= 1e-6
    logs_dir, plots_dir = _prepare_outputs(config)

    episode_path = logs_dir / "episode_sensitivity.csv"
    seed_path = logs_dir / "seed_sensitivity.csv"
    correlation_path = logs_dir / "metric_correlations.csv"
    plot_path = plots_dir / "objective_sensitivity.png"
    _write_csv(episode_path, episode_rows)
    _write_csv(seed_path, seed_rows)
    _write_csv(correlation_path, correlation_rows)
    _plot_audit(seed_rows, episode_rows, metrics, plot_path)

    report: dict[str, Any] = {
        "audit": "objective_sensitivity",
        "status": "read_only_validation_diagnostic",
        "reference_root": str(reference_root),
        "split": config.split,
        "test_data_used": False,
        "test_data_read": False,
        "source_writes": 0,
        "base_seeds": seeds,
        "seed_count": len(seeds),
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent": True,
        "event_rows_are_not_independent": True,
        "episode_count_by_seed": {
            str(row["base_seed"]): int(row["baseline_episode_count"])
            for row in seed_rows
        },
        "forbidden_objective_inputs": [
            "true_position",
            "true_position_error",
            "physical_los_nlos",
            "future_state",
            "future_packet_outcome",
        ],
        "offline_diagnostic_labels": [
            "position_rmse_m",
            "position_error_p95_m",
        ],
        "closure": {
            "J_equals_J_loc_plus_J_com_max_abs_error": float(objective_closure),
            "reward_formula_max_abs_error": float(reward_closure),
            "component_formula_max_abs_error": float(formula_closure),
            "identity_tolerance": 1e-6,
        },
        "checks": {
            "validation_only": config.split == "validation",
            "test_not_read": True,
            "source_untouched_by_audit": True,
            "all_episode_metrics_finite": all_finite,
            "paired_episode_counts_equal": paired_counts_ok,
            "objective_and_reward_closure": closure_ok,
            "seed_level_inference_used": True,
        },
        "seed_metrics": metrics,
        "decision": decision,
        "decision_gate": decision["classification"],
        "audit_ok": bool(
            all_finite
            and paired_counts_ok
            and closure_ok
            and config.split == "validation"
        ),
        "decision_supported": False,
        "logs": {
            "episode_sensitivity": str(episode_path),
            "seed_sensitivity": str(seed_path),
            "metric_correlations": str(correlation_path),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": {"objective_sensitivity": str(plot_path)},
        "interpretation_boundary": (
            "This is a validation-only diagnostic audit. Ground-truth error is used only "
            "for offline RMSE/P95 labels; it is not used to construct J, the reward, or "
            "any actor input. Test data remain sealed."
        ),
    }
    report_path = logs_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "ObjectiveSensitivityAuditConfig",
    "run_objective_sensitivity_audit",
]
