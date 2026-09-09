"""Final read-only audit and figures for the communication--localization mainline.

This module consumes a completed multi-seed communication-constrained MAPPO
batch.  It deliberately reads only training diagnostics and validation
episodes.  Test episodes are sealed and are never opened by this audit.

The output is a reproducible evidence layer rather than a new experiment:
canonical vector-norm localization metrics, actual measurement/packet load,
EKF update-quality summaries, seed-level paired contrasts, and manuscript-ready
training/validation figures are generated without changing source artefacts.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ekf_update_persistence_audit import (
    EKFUpdatePersistenceAuditConfig,
    run_ekf_update_persistence_audit,
)
from .metric_consistency_audit import (
    MetricConsistencyAuditConfig,
    run_metric_consistency_audit,
)
from .metrics import position_error_metrics


DEFAULT_SEEDS = (27281, 28290, 29309, 30318, 31327)
BASELINE_BRANCH = "baseline_mappo"
PROXY_BRANCH = "proxy_guided_adaptive_budgeted_quality_aware_mappo"
PROPOSED_BRANCH = "communication_constrained_proxy_guided_mappo"
DEFAULT_BRANCHES = (BASELINE_BRANCH, PROXY_BRANCH, PROPOSED_BRANCH)

BRANCH_LABELS = {
    # The internal key remains ``baseline_mappo`` for checkpoint and result
    # compatibility, but the current paper identifies this integrated method
    # as the proposed Local-EKF-GAT-MAPPO framework.
    BASELINE_BRANCH: "Local-EKF-GAT-MAPPO (Proposed Method)",
    PROXY_BRANCH: "Proxy-guided ablation",
    PROPOSED_BRANCH: "Deprecated communication-constrained branch",
}
BRANCH_COLORS = {
    BASELINE_BRANCH: "#356AE6",
    PROXY_BRANCH: "#18A558",
    PROPOSED_BRANCH: "#D62728",
}
BRANCH_OFFSETS = {
    BASELINE_BRANCH: -0.16,
    PROXY_BRANCH: 0.0,
    PROPOSED_BRANCH: 0.16,
}

TRAINING_METRICS = (
    "mean_reward",
    "loss",
    "actor_loss",
    "critic_loss",
    "entropy",
    "clip_fraction",
)

SEED_METRICS = (
    "position_rmse_norm_m",
    "position_p95_norm_m",
    "position_rmse_component_m",
    "j_loc",
    "j_com",
    "mean_reward",
    "accepted_update_count",
    "accepted_updates_per_window",
    "update_coverage_any",
    "accepted_update_rate",
    "beneficial_fraction",
    "neutral_fraction",
    "harmful_fraction",
    "beneficial_per_window",
    "harmful_per_window",
    "measurement_count",
    "measurement_overhead_packets",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "connectivity",
    "energy_proxy",
    "actual_communication_cost",
    "expected_communication_cost",
    "actual_cost_excess",
    "expected_cost_excess",
    "j_loc_saturation_fraction",
)

LOWER_IS_BETTER = {
    "position_rmse_norm_m",
    "position_p95_norm_m",
    "position_rmse_component_m",
    "j_loc",
    "j_com",
    "harmful_fraction",
    "harmful_per_window",
    "measurement_count",
    "measurement_overhead_packets",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "energy_proxy",
    "actual_communication_cost",
    "expected_communication_cost",
    "actual_cost_excess",
    "expected_cost_excess",
    "j_loc_saturation_fraction",
}
HIGHER_IS_BETTER = {
    "mean_reward",
    "beneficial_fraction",
    "beneficial_per_window",
    "connectivity",
    "update_coverage_any",
    "accepted_update_rate",
    "accepted_updates_per_window",
}


@dataclass(frozen=True)
class FinalMainlineAuditConfig:
    """Configuration for a validation-only final mainline audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/"
        "final_mainline_audit"
    )
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    expected_episodes: int = 20
    expected_training_updates: int = 100
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True
    figure_profile: str = "legacy"

    def __post_init__(self) -> None:
        if self.figure_profile not in {"legacy", "mainline-v2"}:
            raise ValueError("unknown figure profile")
        if self.split != "validation":
            raise ValueError("final mainline audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be non-empty and unique")
        if BASELINE_BRANCH not in self.branches or PROPOSED_BRANCH not in self.branches:
            raise ValueError("branches must include baseline_mappo and the proposed branch")
        if self.expected_episodes < 1 or self.expected_training_updates < 1:
            raise ValueError("expected episode/update counts must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
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


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value, float(default))
    return int(round(number)) if math.isfinite(number) else int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _finite(value: Any, default: float = float("nan")) -> float:
    return _number(value, default)


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = np.asarray([_number(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float(default)


def _std(values: Iterable[Any], default: float = float("nan")) -> float:
    numbers = np.asarray([_number(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.std(numbers, ddof=1)) if numbers.size > 1 else float(default)


def _quantile(values: Iterable[Any], quantile: float, default: float = float("nan")) -> float:
    numbers = np.asarray([_number(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.quantile(numbers, quantile)) if numbers.size else float(default)


def _bootstrap_ci(values: Sequence[Any], samples: int, seed: int) -> tuple[float, float]:
    numbers = np.asarray([_number(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        return float("nan"), float("nan")
    if numbers.size == 1:
        return float(numbers[0]), float(numbers[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, numbers.size, size=(int(samples), numbers.size))
    means = numbers[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _prepare_outputs(config: FinalMainlineAuditConfig) -> tuple[Path, Path]:
    logs = Path(config.output_dir)
    plots = Path(config.plots_dir) if config.plots_dir is not None else logs.parent / "final_mainline_figures"
    if config.force:
        for path in (logs, plots):
            if path.exists():
                shutil.rmtree(path)
    elif any(path.exists() and any(path.iterdir()) for path in (logs, plots)):
        raise FileExistsError("final output is not empty; use --force")
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _progress(config: FinalMainlineAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[final-mainline] {message}", flush=True)


def _scale_dir(reference_root: Path, seed: int, branch: str) -> Path:
    path = Path(reference_root) / f"seed_{int(seed)}" / str(branch) / "scale_3"
    if not path.is_dir():
        raise FileNotFoundError(f"missing scale_3 directory: {path}")
    return path


def _validation_episode_dirs(scale: Path) -> list[Path]:
    validation = scale / "validation"
    if not validation.is_dir():
        raise FileNotFoundError(f"missing validation directory: {validation}")
    return sorted(
        (path for path in validation.iterdir() if path.is_dir() and path.name.startswith("episode_")),
        key=lambda path: path.name,
    )


def _validate_layout(config: FinalMainlineAuditConfig) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    complete = True
    for seed in config.seeds:
        for branch in config.branches:
            scale = _scale_dir(config.reference_root, seed, branch)
            manifest_path = scale / "manifest.json"
            manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
            episodes = _validation_episode_dirs(scale)
            metrics = _read_csv(scale / "diagnostics" / "validation_episode_metrics.csv")
            required_files = ("ground_truth.csv", "estimates.csv", "actions.csv", "rewards.csv")
            required_ok = all(
                all((episode / filename).is_file() for filename in required_files)
                for episode in episodes
            )
            row = {
                "seed": int(seed),
                "branch": branch,
                "schema_version": str(manifest.get("schema_version", "")),
                "validation_episode_count": len(episodes),
                "validation_metric_count": len(metrics),
                "required_validation_files": int(required_ok),
                "schema_ok": int(manifest.get("schema_version") == "episode-v1.3"),
            }
            cells[f"{seed}::{branch}"] = row
            complete = complete and len(episodes) == config.expected_episodes
            complete = complete and len(metrics) == config.expected_episodes
            complete = complete and required_ok
            complete = complete and manifest.get("schema_version") == "episode-v1.3"
    return {
        "complete": bool(complete),
        "cells": cells,
        "expected_cells": len(config.seeds) * len(config.branches),
    }


def _event_rows(scale: Path, episode_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _read_jsonl(scale / "diagnostics" / "validation_cooperative_updates.jsonl")
        if str(row.get("episode_id", "")) == str(episode_id)
        and str(row.get("split", "validation")) == "validation"
    ]


def _measurement_counts(episode: Path) -> dict[str, Any]:
    actions = _read_csv(episode / "actions.csv")
    overhead = _read_csv(episode / "measurement_overhead.csv")
    selected_actions = [
        row for row in actions if str(row.get("measurement_uav_id", "none")) != "none"
    ]
    invalid_actions = sum(_integer(row.get("invalid_measurement_action")) != 0 for row in selected_actions)
    transition_invalidated = sum(
        _integer(row.get("measurement_transition_invalidated")) != 0 for row in selected_actions
    )
    overhead_events = [
        row for row in overhead if _integer(row.get("selected_non_none")) == 1
    ]
    invariant_failures = 0
    for row in overhead_events:
        invariant_failures += int(_integer(row.get("payload_size_bytes")) != 512)
        invariant_failures += int(abs(_number(row.get("load_packets_requested"), 0.0) - 1.0) > 1e-9)
        invariant_failures += int(abs(_number(row.get("load_packets_applied"), 0.0) - 1.0) > 1e-9)
    return {
        "selected_non_none_actions": int(len(selected_actions)),
        "invalid_measurement_action_count": int(invalid_actions),
        "measurement_transition_invalidated_count": int(transition_invalidated),
        "overhead_event_count": int(len(overhead_events)),
        "overhead_packets_applied": _mean(
            (row.get("load_packets_applied") for row in overhead_events), 0.0
        ) * len(overhead_events),
        "overhead_invariant_failure_count": int(invariant_failures),
        "action_overhead_count_mismatch": int(len(selected_actions) != len(overhead_events)),
        "payload_size_bytes_min": _quantile(
            (row.get("payload_size_bytes") for row in overhead_events), 0.0, 512.0
        ),
        "payload_size_bytes_max": _quantile(
            (row.get("payload_size_bytes") for row in overhead_events), 1.0, 512.0
        ),
    }


def _episode_rows(
    config: FinalMainlineAuditConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    invariant_failures = 0
    for seed in config.seeds:
        for branch in config.branches:
            scale = _scale_dir(config.reference_root, seed, branch)
            metric_rows = {
                str(row.get("episode_id", "")): row
                for row in _read_csv(scale / "diagnostics" / "validation_episode_metrics.csv")
                if row.get("episode_id")
            }
            for index, episode in enumerate(_validation_episode_dirs(scale), start=1):
                episode_id = episode.name
                source = metric_rows.get(episode_id)
                if source is None:
                    raise ValueError(f"missing validation metrics for {seed}/{branch}/{episode_id}")
                position = position_error_metrics(
                    _read_csv(episode / "ground_truth.csv"),
                    _read_csv(episode / "estimates.csv"),
                )
                events = _event_rows(scale, episode_id)
                accepted = [row for row in events if _bool(row.get("accepted"))]
                counts = _measurement_counts(episode)
                invariant_failures += int(counts["overhead_invariant_failure_count"])
                invariant_failures += int(counts["action_overhead_count_mismatch"])
                accepted_count = len(accepted)
                reward_rows = _read_csv(episode / "rewards.csv")
                uav_count = max(1, _integer(source.get("uav_count"), 3))
                decision_windows = max(1.0, len(reward_rows) / uav_count)
                no_update = accepted_count == 0
                row: dict[str, Any] = {
                    "base_seed": int(seed),
                    "branch": branch,
                    "split": "validation",
                    "episode_id": episode_id,
                    "scenario_id": str(source.get("scenario_id", "")),
                    "position_rmse_norm_m": position.get("rmse_norm_m"),
                    "position_p95_norm_m": position.get("p95_norm_m"),
                    "position_rmse_component_m": _finite(source.get("position_rmse_m")),
                    "position_error_sample_count": position.get("sample_count", 0.0),
                    "j_loc": _finite(source.get("j_loc")),
                    "j_com": _finite(source.get("j_com")),
                    "joint_objective": _finite(source.get("joint_objective")),
                    "mean_reward": _finite(source.get("mean_reward")),
                    "accepted_update_count": float(accepted_count),
                    "accepted_updates_per_window": accepted_count / decision_windows,
                    "accepted_update_rate": accepted_count / max(1.0, decision_windows * uav_count),
                    "update_coverage_any": float(not no_update),
                    "no_update_episode": float(no_update),
                    "update_quality_available": float(not no_update),
                    "beneficial_fraction": "" if no_update else _finite(source.get("beneficial_fraction")),
                    "neutral_fraction": "" if no_update else _finite(source.get("neutral_fraction")),
                    "harmful_fraction": "" if no_update else _finite(source.get("harmful_fraction")),
                    "beneficial_per_window": "" if no_update else _finite(source.get("beneficial_per_window")),
                    "harmful_per_window": "" if no_update else _finite(source.get("harmful_per_window")),
                    "packet_loss": _finite(source.get("packet_loss")),
                    "delay_s": _finite(source.get("delay_s")),
                    "aoi_s": _finite(source.get("aoi_s")),
                    "connectivity": _finite(source.get("connectivity")),
                    "energy_proxy": _finite(source.get("energy_proxy")),
                    # ``validation_episode_metrics.csv`` does not carry the
                    # measurement-count fields in this batch.  Recompute them
                    # from the episode action/overhead logs so the final audit
                    # uses the same invariant that it checks below.
                    "measurement_count": float(counts["selected_non_none_actions"]),
                    "measurement_overhead_packets": float(counts["overhead_packets_applied"]),
                    "measurement_action_count_recomputed": counts["selected_non_none_actions"],
                    "measurement_overhead_event_count_recomputed": counts["overhead_event_count"],
                    "measurement_overhead_packets_recomputed": counts["overhead_packets_applied"],
                    "invalid_measurement_action_count": counts["invalid_measurement_action_count"],
                    "measurement_transition_invalidated_count": counts["measurement_transition_invalidated_count"],
                    "overhead_invariant_failure_count": counts["overhead_invariant_failure_count"],
                    "action_overhead_count_mismatch": counts["action_overhead_count_mismatch"],
                    "payload_size_bytes_min": counts["payload_size_bytes_min"],
                    "payload_size_bytes_max": counts["payload_size_bytes_max"],
                    "j_loc_saturation_fraction": _finite(source.get("j_loc_saturation_fraction")),
                    "termination_reason": str(source.get("termination_reason", "")),
                    "collision_occurred": int(_bool(source.get("collision_occurred"))),
                }
                output.append(row)
                if config.progress and (index == 1 or index == config.expected_episodes or index % 5 == 0):
                    _progress(config, f"{branch} seed={seed}: validation episode {index}/{config.expected_episodes}")
    return output, {
        "overhead_invariant_failure_count": int(invariant_failures),
        "validation_episode_rows": len(output),
    }


def _source_cost_rows(config: FinalMainlineAuditConfig) -> list[dict[str, Any]]:
    source = Path("logs/preliminary") / Path(config.reference_root).name / "validation_communication_cost.csv"
    if not source.is_file():
        raise FileNotFoundError(f"missing validation communication cost log: {source}")
    rows = _read_csv(source)
    selected = [
        row for row in rows
        if str(row.get("split", "")) == "validation"
        and _integer(row.get("base_seed")) in {int(seed) for seed in config.seeds}
        and str(row.get("treatment", "")) in set(config.branches)
    ]
    expected = len(config.seeds) * len(config.branches) * 3
    if len(selected) != expected:
        raise ValueError(f"expected {expected} validation communication-cost rows, found {len(selected)}")
    return [dict(row) for row in selected]


def _cost_seed_rows(cost_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cost_rows:
        groups[(_integer(row.get("base_seed")), str(row.get("treatment", "")))].append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch), rows in sorted(groups.items()):
        expected = _mean(row.get("expected_communication_cost") for row in rows)
        actual = _mean(row.get("actual_communication_cost") for row in rows)
        target = _mean(row.get("target") for row in rows)
        output.append({
            "base_seed": seed,
            "branch": branch,
            "cost_uav_count": len(rows),
            "communication_cost_target_mean": target,
            "expected_communication_cost": expected,
            "actual_communication_cost": actual,
            "expected_cost_excess": expected - target if math.isfinite(expected) and math.isfinite(target) else float("nan"),
            "actual_cost_excess": actual - target if math.isfinite(actual) and math.isfinite(target) else float("nan"),
            "within_expected_target_plus_tolerance": int(all(_integer(row.get("within_expected_target_plus_tolerance")) == 1 for row in rows)),
            "within_actual_target_plus_tolerance": int(all(_integer(row.get("within_actual_target_plus_tolerance")) == 1 for row in rows)),
        })
    return output


def _seed_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    cost_seed_rows: Sequence[Mapping[str, Any]],
    config: FinalMainlineAuditConfig,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        groups[(_integer(row.get("base_seed")), str(row.get("branch", "")))].append(row)
    costs = {(_integer(row.get("base_seed")), str(row.get("branch", ""))): row for row in cost_seed_rows}
    output: list[dict[str, Any]] = []
    for seed in config.seeds:
        for branch in config.branches:
            rows = groups[(int(seed), branch)]
            cost = costs.get((int(seed), branch), {})
            result: dict[str, Any] = {
                "base_seed": int(seed),
                "branch": branch,
                "split": "validation",
                "episode_count": len(rows),
                "eventful_episode_count": int(sum(_number(row.get("update_coverage_any"), 0.0) > 0.5 for row in rows)),
                "no_update_episode_count": int(sum(_number(row.get("no_update_episode"), 0.0) > 0.5 for row in rows)),
                "quality_available_episode_count": int(sum(_number(row.get("update_quality_available"), 0.0) > 0.5 for row in rows)),
                "invalid_measurement_action_count": int(sum(_integer(row.get("invalid_measurement_action_count")) for row in rows)),
                "measurement_transition_invalidated_count": int(sum(_integer(row.get("measurement_transition_invalidated_count")) for row in rows)),
                "overhead_invariant_failure_count": int(sum(_integer(row.get("overhead_invariant_failure_count")) for row in rows)),
                "action_overhead_count_mismatch_count": int(sum(_integer(row.get("action_overhead_count_mismatch")) for row in rows)),
            }
            for metric in SEED_METRICS:
                result[metric] = _mean(row.get(metric) for row in rows)
            result.update({
                "expected_communication_cost": cost.get("expected_communication_cost", float("nan")),
                "actual_communication_cost": cost.get("actual_communication_cost", float("nan")),
                "expected_cost_excess": cost.get("expected_cost_excess", float("nan")),
                "actual_cost_excess": cost.get("actual_cost_excess", float("nan")),
                "communication_cost_target_mean": cost.get("communication_cost_target_mean", float("nan")),
                "within_expected_target_plus_tolerance": cost.get("within_expected_target_plus_tolerance", 0),
                "within_actual_target_plus_tolerance": cost.get("within_actual_target_plus_tolerance", 0),
            })
            output.append(result)
    return output


def _training_rows(config: FinalMainlineAuditConfig) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in config.seeds:
        for branch in config.branches:
            scale = _scale_dir(config.reference_root, seed, branch)
            rows = _read_jsonl(scale / "diagnostics" / "training_metrics.jsonl")
            rows = [row for row in rows if str(row.get("phase", "train")) == "train"]
            for row in rows:
                output.append({
                    "base_seed": int(seed),
                    "branch": branch,
                    "update": _integer(row.get("update")),
                    "phase": "train",
                    **{metric: _finite(row.get(metric)) for metric in TRAINING_METRICS},
                    "collision_occurred": int(_bool(row.get("collision_occurred"))),
                })
    return output


def _training_aggregate(rows: Sequence[Mapping[str, Any]], metric: str, config: FinalMainlineAuditConfig) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("branch", "")), _integer(row.get("update")))].append(row)
    output: list[dict[str, Any]] = []
    for (branch, update), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        values = [_number(row.get(metric)) for row in group]
        lower, upper = _bootstrap_ci(values, config.bootstrap_samples, config.bootstrap_seed + update)
        output.append({
            "branch": branch,
            "update": update,
            "seed_count": int(sum(math.isfinite(value) for value in values)),
            "mean": _mean(values),
            "std": _std(values),
            "ci95_lower": lower,
            "ci95_upper": upper,
        })
    return output


def _paired_contrasts(seed_rows: Sequence[Mapping[str, Any]], config: FinalMainlineAuditConfig) -> list[dict[str, Any]]:
    index = {(_integer(row.get("base_seed")), str(row.get("branch"))): row for row in seed_rows}
    comparisons = [(PROPOSED_BRANCH, BASELINE_BRANCH)]
    if PROXY_BRANCH in config.branches:
        comparisons.append((PROXY_BRANCH, BASELINE_BRANCH))
    metrics = (
        "position_rmse_norm_m",
        "position_p95_norm_m",
        "j_loc",
        "j_com",
        "mean_reward",
        "harmful_fraction",
        "beneficial_fraction",
        "accepted_updates_per_window",
        "update_coverage_any",
        "measurement_count",
        "measurement_overhead_packets",
        "actual_communication_cost",
        "expected_communication_cost",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
    )
    output: list[dict[str, Any]] = []
    for treatment, control in comparisons:
        for metric_index, metric in enumerate(metrics):
            diffs = np.asarray([
                _number(index[(int(seed), treatment)].get(metric))
                - _number(index[(int(seed), control)].get(metric))
                for seed in config.seeds
            ], dtype=float)
            finite = diffs[np.isfinite(diffs)]
            lower, upper = _bootstrap_ci(finite, config.bootstrap_samples, config.bootstrap_seed + metric_index)
            direction = "lower_better" if metric in LOWER_IS_BETTER else "higher_better" if metric in HIGHER_IS_BETTER else "descriptive"
            if direction == "lower_better":
                improved = int(np.sum(finite < 0.0))
            elif direction == "higher_better":
                improved = int(np.sum(finite > 0.0))
            else:
                improved = ""
            output.append({
                "comparison": f"{treatment}_minus_{control}",
                "treatment": treatment,
                "control": control,
                "metric": metric,
                "direction": direction,
                "seed_count": int(finite.size),
                "mean_difference": _mean(finite),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "improved_seed_count": improved,
            })
    return output


def _persistence_seed_map(persistence_logs: Path) -> dict[tuple[int, str], dict[str, Any]]:
    path = persistence_logs / "seed_persistence_summary.csv"
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    return {(_integer(row.get("base_seed")), str(row.get("branch", ""))): dict(row) for row in rows}


def _merge_persistence(seed_rows: list[dict[str, Any]], persistence_logs: Path) -> None:
    index = _persistence_seed_map(persistence_logs)
    for row in seed_rows:
        source = index.get((_integer(row.get("base_seed")), str(row.get("branch", ""))), {})
        for key, value in source.items():
            if key in {"base_seed", "branch"}:
                continue
            current = row.get(key)
            # Episode-derived rows may contain a placeholder NaN for
            # conditional EKF quality metrics.  Fill those fields from the
            # already-generated validation-only persistence summary, while
            # retaining finite values independently recomputed above.
            if key not in row or not math.isfinite(_number(current)):
                row[key] = _finite(value)


def _plot_seed_metric(seed_rows: Sequence[Mapping[str, Any]], metric: str, ylabel: str, title: str, path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = list(config.branches)
    positions = np.arange(len(branches), dtype=float)
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for branch_index, branch in enumerate(branches):
        selected = [row for row in seed_rows if str(row.get("branch")) == branch]
        values = [_number(row.get(metric)) for row in selected]
        values = [value for value in values if math.isfinite(value)]
        if values:
            lower, upper = _bootstrap_ci(values, config.bootstrap_samples, config.bootstrap_seed + branch_index)
            mean = float(np.mean(values))
            axis.errorbar(
                [positions[branch_index]], [mean],
                yerr=[[mean - lower], [upper - mean]],
                color=BRANCH_COLORS.get(branch, "#4b5563"), marker="o", markersize=7,
                capsize=4, linewidth=2, label=BRANCH_LABELS.get(branch, branch),
            )
            axis.scatter(
                np.full(len(values), positions[branch_index]) + np.linspace(-0.07, 0.07, len(values)),
                values, color=BRANCH_COLORS.get(branch, "#4b5563"), alpha=0.45, s=28, zorder=3,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([BRANCH_LABELS.get(branch, branch) for branch in branches], rotation=12, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_training(training_rows: Sequence[Mapping[str, Any]], path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, metric, title, ylabel in (
        (axes[0], "mean_reward", "Training reward", "Mean team reward"),
        (axes[1], "loss", "MAPPO total loss", "Loss"),
    ):
        for branch in config.branches:
            selected = [row for row in training_rows if str(row.get("branch")) == branch]
            by_update: dict[int, list[float]] = defaultdict(list)
            for row in selected:
                value = _number(row.get(metric))
                if math.isfinite(value):
                    by_update[_integer(row.get("update"))].append(value)
            updates = sorted(by_update)
            if not updates:
                continue
            means = np.asarray([np.mean(by_update[update]) for update in updates], dtype=float)
            lowers: list[float] = []
            uppers: list[float] = []
            for update in updates:
                lower, upper = _bootstrap_ci(
                    by_update[update], config.bootstrap_samples,
                    config.bootstrap_seed + update,
                )
                lowers.append(lower)
                uppers.append(upper)
            color = BRANCH_COLORS.get(branch, "#4b5563")
            label = BRANCH_LABELS.get(branch, branch)
            axis.plot(updates, means, color=color, linewidth=2, label=label)
            axis.fill_between(updates, lowers, uppers, color=color, alpha=0.14)
        axis.set_xlabel("Update")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_training_components(training_rows: Sequence[Mapping[str, Any]], path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric, title, ylabel in zip(
        axes.flat,
        ("actor_loss", "critic_loss", "entropy", "clip_fraction"),
        ("Actor loss", "Critic loss", "Entropy", "Clip fraction"),
        ("Loss", "Loss", "Entropy", "Fraction"),
    ):
        for branch in config.branches:
            by_update: dict[int, list[float]] = defaultdict(list)
            for row in training_rows:
                if str(row.get("branch")) != branch:
                    continue
                value = _number(row.get(metric))
                if math.isfinite(value):
                    by_update[_integer(row.get("update"))].append(value)
            updates = sorted(by_update)
            if not updates:
                continue
            means = [float(np.mean(by_update[update])) for update in updates]
            color = BRANCH_COLORS.get(branch, "#4b5563")
            axis.plot(updates, means, color=color, linewidth=1.8, label=BRANCH_LABELS.get(branch, branch))
        axis.set_title(title)
        axis.set_xlabel("Update")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_multi_seed_metrics(seed_rows: Sequence[Mapping[str, Any]], path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("position_rmse_norm_m", "Position RMSE (m)", "Validation localization accuracy"),
        ("position_p95_norm_m", "Position P95 (m)", "Validation tail error"),
        ("update_coverage_any", "Episode coverage", "Episodes with accepted cooperative update"),
        ("j_loc", r"$J_{\mathrm{loc}}$", "Covariance uncertainty objective"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    branches = list(config.branches)
    positions = np.arange(len(branches), dtype=float)
    for axis, (metric, ylabel, title) in zip(axes.flat, panels):
        for branch_index, branch in enumerate(branches):
            values = [
                _number(row.get(metric))
                for row in seed_rows if str(row.get("branch")) == branch
            ]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            lower, upper = _bootstrap_ci(values, config.bootstrap_samples, config.bootstrap_seed + branch_index)
            mean = float(np.mean(values))
            color = BRANCH_COLORS.get(branch, "#4b5563")
            axis.errorbar(
                [positions[branch_index]], [mean], yerr=[[mean - lower], [upper - mean]],
                color=color, marker="o", capsize=4, linewidth=2,
            )
            axis.scatter(
                np.full(len(values), positions[branch_index]) + np.linspace(-0.07, 0.07, len(values)),
                values, color=color, alpha=0.45, s=28,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([BRANCH_LABELS.get(branch, branch) for branch in branches], rotation=12, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=BRANCH_COLORS.get(branch, "#4b5563"), marker="o", label=BRANCH_LABELS.get(branch, branch)) for branch in branches]
    figure.legend(handles=handles, loc="upper center", ncol=len(branches), fontsize=8)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_communication(seed_rows: Sequence[Mapping[str, Any]], path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("measurement_count", "Measurements / episode"),
        ("measurement_overhead_packets", "512-byte packets / episode"),
        ("actual_communication_cost", "Actual normalized cost"),
        ("j_com", r"$J_{\mathrm{com}}$"),
        ("packet_loss", "Packet loss"),
        ("delay_s", "Delay (s)"),
        ("aoi_s", "AoI (s)"),
        ("connectivity", "Connectivity"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    branches = list(config.branches)
    positions = np.arange(len(branches), dtype=float)
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for branch_index, branch in enumerate(branches):
            values = [_number(row.get(metric)) for row in seed_rows if str(row.get("branch")) == branch]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            lower, upper = _bootstrap_ci(values, config.bootstrap_samples, config.bootstrap_seed + branch_index + 100)
            mean = float(np.mean(values))
            color = BRANCH_COLORS.get(branch, "#4b5563")
            axis.errorbar([positions[branch_index]], [mean], yerr=[[mean - lower], [upper - mean]], color=color, marker="o", capsize=4, linewidth=2)
            axis.scatter(np.full(len(values), positions[branch_index]) + np.linspace(-0.07, 0.07, len(values)), values, color=color, alpha=0.45, s=25)
        axis.set_xticks(positions)
        axis.set_xticklabels([BRANCH_LABELS.get(branch, branch) for branch in branches], rotation=12, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=BRANCH_COLORS.get(branch, "#4b5563"), marker="o", label=BRANCH_LABELS.get(branch, branch)) for branch in branches]
    figure.legend(handles=handles, loc="upper center", ncol=len(branches), fontsize=8)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_update_quality(seed_rows: Sequence[Mapping[str, Any]], path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("beneficial_fraction", "Beneficial update fraction"),
        ("harmful_fraction", "Harmful update fraction"),
        ("accepted_updates_per_window", "Accepted updates / window"),
        ("update_coverage_any", "Update coverage"),
        ("j_loc", r"$J_{\mathrm{loc}}$"),
        ("h3_benefit_retained_fraction", "3-window benefit retention"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    branches = list(config.branches)
    positions = np.arange(len(branches), dtype=float)
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for branch_index, branch in enumerate(branches):
            values = [_number(row.get(metric)) for row in seed_rows if str(row.get("branch")) == branch]
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            lower, upper = _bootstrap_ci(values, config.bootstrap_samples, config.bootstrap_seed + branch_index + 200)
            mean = float(np.mean(values))
            color = BRANCH_COLORS.get(branch, "#4b5563")
            axis.errorbar([positions[branch_index]], [mean], yerr=[[mean - lower], [upper - mean]], color=color, marker="o", capsize=4, linewidth=2)
            axis.scatter(np.full(len(values), positions[branch_index]) + np.linspace(-0.07, 0.07, len(values)), values, color=color, alpha=0.45, s=25)
        axis.set_xticks(positions)
        axis.set_xticklabels([BRANCH_LABELS.get(branch, branch) for branch in branches], rotation=12, ha="right", fontsize=7)
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=BRANCH_COLORS.get(branch, "#4b5563"), marker="o", label=BRANCH_LABELS.get(branch, branch)) for branch in branches]
    figure.legend(handles=handles, loc="upper center", ncol=len(branches), fontsize=8)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_tradeoff(seed_rows: Sequence[Mapping[str, Any]], path: Path, config: FinalMainlineAuditConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    index = {(_integer(row.get("base_seed")), str(row.get("branch"))): row for row in seed_rows}
    for seed_index, seed in enumerate(config.seeds):
        base = index.get((int(seed), BASELINE_BRANCH))
        proposed = index.get((int(seed), PROPOSED_BRANCH))
        if base is not None and proposed is not None:
            x0, y0 = _number(base.get("actual_communication_cost")), _number(base.get("position_rmse_norm_m"))
            x1, y1 = _number(proposed.get("actual_communication_cost")), _number(proposed.get("position_rmse_norm_m"))
            if all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                axis.plot([x0, x1], [y0, y1], color="#9ca3af", linewidth=0.9, alpha=0.7)
                axis.text(x0, y0, str(seed), fontsize=7, color=BRANCH_COLORS[BASELINE_BRANCH], ha="right", va="bottom")
                axis.text(x1, y1, str(seed), fontsize=7, color=BRANCH_COLORS[PROPOSED_BRANCH], ha="left", va="top")
    for branch in config.branches:
        x = [_number(row.get("actual_communication_cost")) for row in seed_rows if str(row.get("branch")) == branch]
        y = [_number(row.get("position_rmse_norm_m")) for row in seed_rows if str(row.get("branch")) == branch]
        finite = [(left, right) for left, right in zip(x, y) if math.isfinite(left) and math.isfinite(right)]
        if finite:
            axis.scatter([item[0] for item in finite], [item[1] for item in finite], s=58, color=BRANCH_COLORS.get(branch, "#4b5563"), label=BRANCH_LABELS.get(branch, branch))
    axis.set_xlabel("Actual normalized communication cost")
    axis.set_ylabel("Validation position RMSE (m; vector norm)")
    axis.set_title("Communication–localization trade-off")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _old_report(reference_root: Path) -> dict[str, Any]:
    path = Path("logs/preliminary") / Path(reference_root).name / "audit_report.json"
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError, TypeError):
        return {}


def _contrast(index: Mapping[tuple[str, str], Mapping[str, Any]], comparison: str, metric: str) -> Mapping[str, Any]:
    return index.get((comparison, metric), {})


def run_final_mainline_audit(config: FinalMainlineAuditConfig) -> dict[str, Any]:
    """Generate the final validation-only mainline audit and figures."""

    if config.figure_profile == "mainline-v2":
        from .mainline_figures_v2 import run_mainline_figures_v2
        return run_mainline_figures_v2(config)

    reference = Path(config.reference_root).resolve()
    if not reference.is_dir():
        raise FileNotFoundError(f"reference root not found: {reference}")
    logs, plots = _prepare_outputs(config)
    layout = _validate_layout(config)

    metric_logs = logs / "metric_consistency_audit"
    metric_plots = plots / "metric_consistency_audit"
    metric_report = run_metric_consistency_audit(
        MetricConsistencyAuditConfig(
            reference_root=reference,
            output_dir=metric_logs,
            plots_dir=metric_plots,
            seeds=config.seeds,
            branches=config.branches,
            split="validation",
            expected_episodes=config.expected_episodes,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
            force=True,
            progress=config.progress,
        )
    )
    persistence_logs = logs / "ekf_update_persistence_audit"
    persistence_plots = plots / "ekf_update_persistence_audit"
    persistence_report = run_ekf_update_persistence_audit(
        EKFUpdatePersistenceAuditConfig(
            reference_root=reference,
            output_dir=persistence_logs,
            plots_dir=persistence_plots,
            seeds=config.seeds,
            branches=config.branches,
            split="validation",
            horizons=(1, 2, 3, 5),
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
            force=True,
            progress=config.progress,
        )
    )

    episode_rows, episode_checks = _episode_rows(config)
    cost_rows = _source_cost_rows(config)
    cost_seed_rows = _cost_seed_rows(cost_rows)
    seed_rows = _seed_rows(episode_rows, cost_seed_rows, config)
    _merge_persistence(seed_rows, persistence_logs)
    training_rows = _training_rows(config)
    training_aggregates = {
        metric: _training_aggregate(training_rows, metric, config)
        for metric in TRAINING_METRICS
    }
    contrasts = _paired_contrasts(seed_rows, config)

    _write_csv(logs / "episode_metrics.csv", episode_rows)
    _write_csv(logs / "communication_cost_by_seed.csv", cost_seed_rows)
    _write_csv(logs / "validation_communication_cost.csv", cost_rows)
    _write_csv(logs / "seed_level_metrics.csv", seed_rows)
    _write_csv(logs / "paired_contrasts.csv", contrasts)
    _write_csv(logs / "training_metrics.csv", training_rows)
    for metric, rows in training_aggregates.items():
        _write_csv(logs / f"training_{metric}_aggregate.csv", rows)

    figure_paths = {
        "training_reward_and_mappo_loss": plots / "training_reward_and_mappo_loss.png",
        "training_loss_components": plots / "training_loss_components.png",
        "localization_performance": plots / "localization_performance.png",
        "communication_cost": plots / "communication_cost.png",
        "ekf_update_quality": plots / "ekf_update_quality.png",
        "communication_localization_tradeoff": plots / "communication_localization_tradeoff.png",
    }
    _plot_training(training_rows, figure_paths["training_reward_and_mappo_loss"], config)
    _plot_training_components(training_rows, figure_paths["training_loss_components"], config)
    _plot_multi_seed_metrics(seed_rows, figure_paths["localization_performance"], config)
    _plot_communication(seed_rows, figure_paths["communication_cost"], config)
    _plot_update_quality(seed_rows, figure_paths["ekf_update_quality"], config)
    _plot_tradeoff(seed_rows, figure_paths["communication_localization_tradeoff"], config)

    contrast_index = {
        (str(row.get("comparison")), str(row.get("metric"))): row
        for row in contrasts
    }
    main_comparison = f"{PROPOSED_BRANCH}_minus_{BASELINE_BRANCH}"
    rmse = _contrast(contrast_index, main_comparison, "position_rmse_norm_m")
    actual_cost = _contrast(contrast_index, main_comparison, "actual_communication_cost")
    j_com = _contrast(contrast_index, main_comparison, "j_com")
    harmful = _contrast(contrast_index, main_comparison, "harmful_fraction")
    reward = _contrast(contrast_index, main_comparison, "mean_reward")
    proposed_rows = [row for row in seed_rows if str(row.get("branch")) == PROPOSED_BRANCH]
    target_count = sum(_integer(row.get("within_actual_target_plus_tolerance")) == 1 for row in proposed_rows)
    all_finite = all(
        math.isfinite(_number(row.get(metric)))
        for row in seed_rows
        for metric in (
            "position_rmse_norm_m", "position_p95_norm_m", "j_loc", "j_com",
            "mean_reward", "measurement_count", "measurement_overhead_packets",
            "packet_loss", "delay_s", "aoi_s", "connectivity", "actual_communication_cost",
        )
    )
    training_cells = {
        f"{seed}::{branch}": sum(
            _integer(row.get("base_seed")) == int(seed) and str(row.get("branch")) == branch
            for row in training_rows
        )
        for seed in config.seeds
        for branch in config.branches
    }
    training_complete = all(value == config.expected_training_updates for value in training_cells.values())
    old_report = _old_report(reference)
    old_checks = old_report.get("checks", {}) if isinstance(old_report, Mapping) else {}
    episode_v1_3 = bool(old_checks.get("episode_v1_3_all_branches", False))
    if not episode_v1_3:
        episode_v1_3 = bool(old_report.get("audit_ok", False)) and layout["complete"]

    rmse_ok = (
        _integer(rmse.get("improved_seed_count"), 0) >= 4
        and _number(rmse.get("ci95_upper")) < 0.0
    )
    cost_ok = _number(actual_cost.get("ci95_upper")) <= 0.0
    jcom_ok = _number(j_com.get("ci95_upper")) <= 0.005
    harmful_ok = _number(harmful.get("ci95_upper")) <= 0.02
    reward_ok = _number(reward.get("mean_difference")) >= -0.005
    decision_supported = bool(
        layout["complete"]
        and episode_v1_3
        and all_finite
        and training_complete
        and episode_checks["overhead_invariant_failure_count"] == 0
        and target_count >= 4
        and rmse_ok
        and cost_ok
        and jcom_ok
        and harmful_ok
        and reward_ok
    )
    audit_ok = bool(
        layout["complete"]
        and episode_v1_3
        and all_finite
        and training_complete
        and episode_checks["overhead_invariant_failure_count"] == 0
        and all(_integer(row.get("invalid_measurement_action_count")) == 0 for row in episode_rows)
        and all(_integer(row.get("action_overhead_count_mismatch")) == 0 for row in episode_rows)
    )
    if not layout["complete"] or not episode_v1_3 or not all_finite:
        primary_reason = "structural_or_metric_audit_failure"
    elif not rmse_ok:
        primary_reason = "localization_not_stable"
    elif not cost_ok or not jcom_ok:
        primary_reason = "communication_localization_tradeoff"
    elif not harmful_ok:
        primary_reason = "harmful_update_quality_not_improved"
    elif not reward_ok:
        primary_reason = "team_reward_non_inferiority_not_met"
    else:
        primary_reason = "mainline_criteria_supported"

    report = {
        "audit": "final_communication_localization_mainline",
        "status": "read_only_validation_diagnostic",
        "audit_ok": audit_ok,
        "reference_root": str(reference),
        "source_script": str(Path(__file__).resolve()),
        "seed_set": [int(seed) for seed in config.seeds],
        "branches": list(config.branches),
        "analysis_split": "validation",
        "test_data_read": False,
        "test_data_used": False,
        "independent_unit": "training seed after episode-level reduction",
        "episode_rows_are_not_independent": True,
        "event_rows_are_not_independent": True,
        "metric_definitions": {
            "position_rmse_norm_m": "RMSE of Euclidean position error over UAV-timestamp samples",
            "position_p95_norm_m": "95th percentile of Euclidean position error",
            "position_rmse_component_m": "legacy coordinate-wise RMSE retained for traceability",
            "j_loc": "covariance-based localization uncertainty objective, not RMSE",
            "j_com": "communication objective from the frozen environment reward definition",
            "actual_communication_cost": "mean selected-measurement communication cost from validation cost logs",
        },
        "data_contract": {
            "source_data": str(reference),
            "training_source": "diagnostics/training_metrics.jsonl, phase=train",
            "validation_source": "validation episode CSVs and validation diagnostics",
            "test_source": "sealed; not opened by this audit",
            "episode_horizon_steps": 50,
            "validation_episodes_per_seed_branch": config.expected_episodes,
            "training_updates_per_seed_branch": config.expected_training_updates,
            "payload_size_bytes": 512,
            "transformations": [
                "recompute vector-norm RMSE/P95 from validation ground_truth.csv and estimates.csv",
                "average episode metrics within seed and branch",
                "bootstrap paired seed means with the configured seed",
            ],
        },
        "checks": {
            "audit_ok_structural": bool(audit_ok),
            "layout_complete": bool(layout["complete"]),
            "episode_v1_3_all_branches": bool(episode_v1_3),
            "metric_consistency_audit_ok": bool(metric_report.get("audit_ok", False)),
            "ekf_persistence_audit_ok": bool(persistence_report.get("audit_ok", False)),
            "training_updates_complete": bool(training_complete),
            "all_required_metrics_finite": bool(all_finite),
            "measurement_payload_invariants": bool(episode_checks["overhead_invariant_failure_count"] == 0),
            "invalid_measurement_action_count": int(sum(_integer(row.get("invalid_measurement_action_count")) for row in episode_rows)),
            "action_overhead_count_mismatch_count": int(sum(_integer(row.get("action_overhead_count_mismatch")) for row in episode_rows)),
            "actual_target_plus_tolerance_seed_count": int(target_count),
            "source_untouched_by_audit": True,
            "trainer_update_called": False,
            "optimizer_step_called": False,
            "test_not_read": True,
            "seed_level_inference_used": True,
        },
        "primary_comparison": {
            "comparison": main_comparison,
            "rmse": dict(rmse),
            "actual_communication_cost": dict(actual_cost),
            "j_com": dict(j_com),
            "harmful_fraction": dict(harmful),
            "mean_reward": dict(reward),
            "criteria": {
                "rmse_supported": bool(rmse_ok),
                "actual_communication_cost_non_inferior": bool(cost_ok),
                "j_com_non_inferior": bool(jcom_ok),
                "harmful_fraction_non_inferior": bool(harmful_ok),
                "mean_reward_non_inferior": bool(reward_ok),
            },
        },
        "decision": {
            "primary_reason": primary_reason,
            "decision_supported": decision_supported,
            "text": (
                "The proposed branch meets the preregistered mainline criteria."
                if decision_supported
                else "The proposed branch does not meet all preregistered mainline criteria; report the supported mechanism evidence and the failed gate explicitly."
            ),
        },
        "outputs": {
            "episode_metrics": str(logs / "episode_metrics.csv"),
            "seed_level_metrics": str(logs / "seed_level_metrics.csv"),
            "paired_contrasts": str(logs / "paired_contrasts.csv"),
            "validation_communication_cost": str(logs / "validation_communication_cost.csv"),
            "metric_consistency_report": str(metric_logs / "audit_report.json"),
            "ekf_persistence_report": str(persistence_logs / "audit_report.json"),
            "figures": {name: str(path) for name, path in figure_paths.items()},
        },
        "unsupported_claims": [
            "A training loss decrease alone does not prove localization improvement.",
            "J_loc is not a substitute for state-mean RMSE.",
            "A five-seed result is not evidence for every UAV scale or every scenario.",
            "Safety and dynamic-obstacle results are outside this communication-localization mainline audit.",
        ],
    }
    (logs / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return report


__all__ = [
    "BASELINE_BRANCH",
    "DEFAULT_BRANCHES",
    "DEFAULT_SEEDS",
    "FinalMainlineAuditConfig",
    "PROPOSED_BRANCH",
    "PROXY_BRANCH",
    "run_final_mainline_audit",
]
