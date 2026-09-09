"""Read-only attribution audit for proxy-guided cooperative EKF updates.

The audit replays frozen validation checkpoints and joins three sources of
evidence without feeding offline labels back into the policy:

* actor-visible candidate proxy and communication features;
* deterministic actor/teacher action distributions;
* accepted cooperative EKF update diagnostics, whose truth fields are marked
  as validation-only labels.

Windows/episode folders in the reference root are never modified.  The test
split is intentionally not enumerated or opened.  Window and update rows are
diagnostic only; all confidence intervals are computed after reducing to one
mean per training seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _jsonable
from .policy_mechanism_audit import (
    DEFAULT_SEEDS,
    _array,
    _bootstrap_ci,
    _csv_value,
    _event_int,
    _finite_trace_value,
    _float,
    _load_proxy_model,
    _read_csv,
    _read_json,
    _read_jsonl,
    _replay_branch,
    _safe_mean,
    _teacher_probabilities,
    _write_csv,
    _manifest_for,
)


DEFAULT_BRANCHES = (
    "baseline_mappo",
    "adaptive_budgeted_quality_aware_mappo",
    "proxy_guided_adaptive_budgeted_quality_aware_mappo",
)
LOWER_IS_BETTER = {
    "position_rmse_m",
    "position_error_p95_m",
    "j_loc",
    "j_com",
    "harmful_fraction",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "measurement_count",
    "accepted_update_count",
    "error_change_m",
}
METRICS = (
    "position_rmse_m",
    "position_error_p95_m",
    "j_loc",
    "j_com",
    "mean_reward",
    "accepted_update_count",
    "beneficial_fraction",
    "harmful_fraction",
    "neutral_fraction",
    "measurement_count",
    "measurement_overhead_packets",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "connectivity",
    "teacher_agreement",
    "selected_proxy_utility",
    "linear_cov_reduction_m2",
    "error_change_m",
)
_MANAGED_LOG_FILES = (
    "replay_policy_trace.jsonl",
    "proxy_parameters.json",
    "candidate_proxy_events.csv",
    "accepted_update_quality.csv",
    "teacher_alignment.csv",
    "communication_tradeoff.csv",
    "seed_episode_summary.csv",
    "paired_contrasts.csv",
    "audit_report.json",
)
_MANAGED_PLOT_FILES = ("proxy_update_quality_audit.png",)


@dataclass(frozen=True)
class ProxyUpdateQualityAuditConfig:
    """Configuration for the frozen-checkpoint validation-only audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "proxy_guided_adaptive_quality_mappo_seedset_7071_11127/"
        "proxy_update_quality_audit"
    )
    plots_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    age_variance_coefficient: float = 1.0
    communication_cost_weight: float = 1.0
    temperature: float = 0.25
    device: str = "cpu"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("proxy update-quality audit is validation-only")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be a non-empty tuple of unique branches")
        unknown = sorted(set(self.branches) - set(DEFAULT_BRANCHES))
        if unknown:
            raise ValueError("unsupported branch(es): " + ", ".join(unknown))
        if self.device != "cpu":
            raise ValueError("the deterministic audit supports device='cpu' only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not math.isfinite(float(self.age_variance_coefficient)) or self.age_variance_coefficient < 0.0:
            raise ValueError("age_variance_coefficient must be finite and non-negative")
        if not math.isfinite(float(self.communication_cost_weight)) or self.communication_cost_weight < 0.0:
            raise ValueError("communication_cost_weight must be finite and non-negative")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")


def _prepare_output(path: Path, managed: Sequence[str], *, force: bool) -> None:
    """Prepare only the explicitly managed files; never delete a directory tree."""

    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"audit output is not empty: {path}; use --force")
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for name in managed:
            target = path / name
            if target.exists() and target.is_file():
                target.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _uav_index(value: Any) -> int:
    text = str(value)
    suffix = text.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 10**9


def _episode_number(value: Any) -> int:
    text = str(value)
    try:
        return int(text.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _safe_array(value: Any, length: int, fill: float = 0.0) -> np.ndarray:
    result = np.full(length, float(fill), dtype=float)
    values = _array(value, dtype=float)
    result[: min(length, values.size)] = values[:length]
    result[~np.isfinite(result)] = float(fill)
    return result


_COMPACT_TRACE_FIELDS = (
    "base_seed",
    "episode_seed",
    "split",
    "phase",
    "branch",
    "episode_id",
    "step",
    "timestamp",
    "uav_id",
    "candidate_uav_ids",
    "measurement_proxy_gain",
    "measurement_proxy_cost",
    "measurement_proxy_valid",
    "measurement_proxy_utility",
    "measurement_probabilities",
    "proxy_teacher_probabilities",
    "action_mask",
    "proxy_features_in_actor",
    "submitted_measurement_uav_id",
    "selected_measurement_uav_id",
    "overhead_event_present",
    "accepted_update_count_step",
)


def _compact_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only replay fields needed after candidate/teacher expansion.

    ``_replay_branch`` emits nested diagnostics for reproducibility.  Those
    rows are written to ``replay_policy_trace.jsonl`` immediately, while the
    in-memory aggregation uses this compact projection to avoid duplicating
    large candidate-link payloads across all 15 replays.
    """

    compact: dict[str, Any] = {}
    for key in _COMPACT_TRACE_FIELDS:
        if key in row:
            value = row[key]
            if isinstance(value, (list, tuple, np.ndarray)):
                compact[key] = list(value)
            elif isinstance(value, Mapping):
                compact[key] = dict(value)
            else:
                compact[key] = value
    return compact


def _candidate_rows(trace_rows: Sequence[Mapping[str, Any]], *, weight: float) -> list[dict[str, Any]]:
    """Expand one policy trace row into candidate-level proxy diagnostics."""

    output: list[dict[str, Any]] = []
    for row in trace_rows:
        candidate_ids = [str(item) for item in row.get("candidate_uav_ids", [])]
        if not candidate_ids:
            continue
        count = max(
            3,
            _array(row.get("measurement_proxy_gain", []), dtype=float).size,
            _array(row.get("measurement_proxy_cost", []), dtype=float).size,
            _array(row.get("measurement_proxy_valid", []), dtype=float).size,
        )
        gains = _safe_array(row.get("measurement_proxy_gain", []), count, 0.0)
        costs = _safe_array(row.get("measurement_proxy_cost", []), count, 1.0)
        valids = _safe_array(row.get("measurement_proxy_valid", []), count, 0.0)
        utilities = gains - float(weight) * costs
        actor_probs = _safe_array(row.get("measurement_probabilities", []), count + 1, 0.0)
        teacher_probs = _safe_array(row.get("proxy_teacher_probabilities", []), count + 1, 0.0)
        link_by_source = {
            str(item.get("source_uav_id", "")): item
            for item in row.get("candidate_link_metrics", [])
            if isinstance(item, Mapping)
        }
        ordered = sorted(
            candidate_ids,
            key=lambda source: (
                -int(valids[_uav_index(source)] > 0.5) if _uav_index(source) < count else 0,
                -float(utilities[_uav_index(source)]) if _uav_index(source) < count else 0.0,
                _uav_index(source),
                source,
            ),
        )
        ranks = {source: rank for rank, source in enumerate(ordered, start=1)}
        actor_choice = str(row.get("submitted_measurement_uav_id", row.get("selected_measurement_uav_id", "none")))
        teacher_choice = str(row.get("proxy_teacher_choice", "none"))
        for source in candidate_ids:
            index = _uav_index(source)
            valid_index = 0 <= index < count
            link = link_by_source.get(source, {})
            output.append(
                {
                    "base_seed": int(row.get("base_seed", 0)),
                    "episode_id": str(row.get("episode_id", "")),
                    "episode_seed": int(_float(row.get("episode_seed"), 0.0)),
                    "split": str(row.get("split", "validation")),
                    "branch": str(row.get("branch", "")),
                    "step": int(_float(row.get("step"), 0.0)),
                    "timestamp": _float(row.get("timestamp"), 0.0),
                    "target_uav_id": str(row.get("uav_id", "")),
                    "source_uav_id": source,
                    "proxy_gain": float(gains[index]) if valid_index else 0.0,
                    "proxy_cost": float(costs[index]) if valid_index else 1.0,
                    "proxy_utility": float(utilities[index]) if valid_index else -1.0,
                    "proxy_valid": int(valid_index and valids[index] > 0.5),
                    "proxy_rank": int(ranks[source]),
                    "actor_probability": float(actor_probs[index]) if valid_index else 0.0,
                    "teacher_probability": float(teacher_probs[index]) if valid_index else 0.0,
                    "candidate_action_mask": int(
                        bool(_array(row.get("action_mask", []), dtype=int)[index])
                        if valid_index and _array(row.get("action_mask", []), dtype=int).size > index
                        else 0
                    ),
                    "selected_by_actor": int(actor_choice == source),
                    "selected_by_teacher": int(teacher_choice == source),
                    "actor_choice": actor_choice,
                    "teacher_choice": teacher_choice,
                    "teacher_in_actor": int(row.get("proxy_features_in_actor", 0) == 1),
                    "teacher_actor_agreement": int(actor_choice == teacher_choice),
                    "rssi_dbm": _float(link.get("rssi_dbm"), -100.0),
                    "snr_db": _float(link.get("snr_db"), 0.0),
                    "packet_loss_pred": _float(link.get("packet_loss_pred"), 1.0),
                    "delay_pred_s": _float(link.get("delay_pred_s"), 0.0),
                    "aoi_pred_s": _float(link.get("aoi_pred_s"), 0.0),
                    "queue_length_pred": _float(link.get("queue_length_pred"), 0.0),
                    "message_age_s": _float(link.get("message_age_s"), 0.0),
                    "link_state": str(link.get("link_state", "unknown")),
                    "truth_used_for_proxy": 0,
                }
            )
    return output


def _teacher_rows(trace_rows: Sequence[Mapping[str, Any]], *, weight: float, temperature: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in trace_rows:
        actor = _array(row.get("measurement_probabilities", []), dtype=float)
        gain = _array(row.get("measurement_proxy_gain", []), dtype=float)
        cost = _array(row.get("measurement_proxy_cost", []), dtype=float)
        valid = _array(row.get("measurement_proxy_valid", []), dtype=float)
        mask = _array(row.get("action_mask", []), dtype=float)
        teacher = _teacher_probabilities(
            gain,
            cost,
            valid,
            mask,
            temperature=temperature,
            communication_cost_weight=weight,
        )
        actor = _safe_array(actor, teacher.size, 0.0)
        actor_sum = float(actor.sum())
        if actor_sum <= 0.0 or not math.isfinite(actor_sum):
            actor[-1] = 1.0
            actor_sum = 1.0
        actor /= actor_sum
        teacher = _safe_array(teacher, teacher.size, 0.0)
        teacher_sum = float(teacher.sum())
        if teacher_sum <= 0.0 or not math.isfinite(teacher_sum):
            teacher[-1] = 1.0
            teacher_sum = 1.0
        teacher /= teacher_sum
        kl = float(np.sum(teacher * (np.log(np.clip(teacher, 1e-12, 1.0)) - np.log(np.clip(actor, 1e-12, 1.0)))))
        selected = str(row.get("submitted_measurement_uav_id", row.get("selected_measurement_uav_id", "none")))
        selected_index = _uav_index(selected)
        teacher_index = int(np.argmax(teacher)) if teacher.size else -1
        teacher_choice = str(row.get("proxy_teacher_choice", "none"))
        selected_proxy_utility = 0.0
        if selected != "none":
            utilities = _safe_array(row.get("measurement_proxy_utility", []), 3, 0.0)
            if 0 <= selected_index < utilities.size:
                selected_proxy_utility = float(utilities[selected_index])
        output.append(
            {
                "base_seed": int(row.get("base_seed", 0)),
                "episode_id": str(row.get("episode_id", "")),
                "branch": str(row.get("branch", "")),
                "step": int(_float(row.get("step"), 0.0)),
                "target_uav_id": str(row.get("uav_id", "")),
                "candidate_count": len(row.get("candidate_uav_ids", [])),
                "proxy_valid_any": int(np.any(valid > 0.5)),
                "proxy_features_in_actor": int(row.get("proxy_features_in_actor", 0)),
                "teacher_in_actor": int(row.get("proxy_features_in_actor", 0) == 1),
                "actor_choice": selected,
                "teacher_choice": teacher_choice,
                "teacher_top_choice_recomputed": int(
                    teacher_index == selected_index if selected_index < teacher.size else teacher_index == teacher.size - 1
                ),
                "teacher_actor_agreement": int(selected == teacher_choice),
                "actor_probability_at_selection": float(actor[selected_index]) if 0 <= selected_index < actor.size else 0.0,
                "teacher_probability_at_selection": float(teacher[selected_index]) if 0 <= selected_index < teacher.size else 0.0,
                "actor_probability_at_teacher": float(actor[teacher_index]) if 0 <= teacher_index < actor.size else 0.0,
                "teacher_probability_at_teacher": float(teacher[teacher_index]) if 0 <= teacher_index < teacher.size else 0.0,
                "teacher_kl_actor": kl,
                "measurement_count": int(selected != "none"),
                "none_probability": float(actor[-1]) if actor.size else 1.0,
                "teacher_none_probability": float(teacher[-1]) if teacher.size else 1.0,
                "selected_proxy_utility": selected_proxy_utility,
            }
        )
    return output


def _accepted_update_rows(
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], int]]:
    candidate_map: dict[tuple[int, str, str, int, str, str], Mapping[str, Any]] = {}
    for row in candidate_rows:
        key = (
            int(row.get("base_seed", 0)),
            str(row.get("branch", "")),
            str(row.get("episode_id", "")),
            int(_float(row.get("step"), 0.0)),
            str(row.get("target_uav_id", "")),
            str(row.get("source_uav_id", "")),
        )
        candidate_map[key] = row

    output: list[dict[str, Any]] = []
    replay_counts: dict[tuple[int, str], int] = {}
    for seed in seeds:
        for branch in branches:
            scale_root, _manifest = _manifest_for(reference_root, int(seed), branch)
            events = _read_jsonl(scale_root / "diagnostics" / "cooperative_updates.jsonl")
            for event in events:
                if str(event.get("split", "")) != "validation" or not _is_true(event.get("accepted", False)):
                    continue
                base_seed = int(seed)
                episode_id = str(event.get("episode_id", ""))
                step = int(_float(event.get("step"), 0.0))
                source = str(event.get("source_uav_id", ""))
                target = str(event.get("target_uav_id", ""))
                before_error = _float(event.get("true_position_error_before_m"), 0.0)
                after_error = _float(event.get("true_position_error_after_m"), 0.0)
                error_change = after_error - before_error
                if error_change < -0.001:
                    effect = "beneficial"
                elif error_change > 0.001:
                    effect = "harmful"
                else:
                    effect = "approximately_neutral"
                cov_before = _float(event.get("position_covariance_trace_before_m2"), 0.0)
                cov_linear = _float(event.get("position_covariance_trace_after_linear_m2"), cov_before)
                cov_final = _float(event.get("position_covariance_trace_after_final_m2"), cov_before)
                effective_variance = _float(event.get("effective_measurement_variance_m2"), 0.0)
                unknown_bound = _float(event.get("unknown_cross_bound_m2"), 0.0)
                denominator = max(abs(effective_variance), 1e-12)
                matched = candidate_map.get((base_seed, branch, episode_id, step, target, source), {})
                proxy_gain = _float(matched.get("proxy_gain"), 0.0)
                proxy_cost = _float(matched.get("proxy_cost"), 1.0)
                proxy_utility = 0.0
                proxy_valid = 0
                proxy_rank = 0
                if matched:
                    proxy_utility = _float(matched.get("proxy_utility"), proxy_gain - proxy_cost)
                    proxy_valid = int(_float(matched.get("proxy_valid"), 0.0) > 0.5)
                    proxy_rank = int(_float(matched.get("proxy_rank"), 0.0))
                replay_counts[(base_seed, branch)] = replay_counts.get((base_seed, branch), 0) + 1
                output.append(
                    {
                        "base_seed": base_seed,
                        "branch": branch,
                        "split": "validation",
                        "episode_id": episode_id,
                        "step": step,
                        "source_uav_id": source,
                        "target_uav_id": target,
                        "los_label_offline": int(_is_true(event.get("los", False))),
                        "truth_used_offline_only": 1,
                        "message_age_s": _float(event.get("message_age_s"), 0.0),
                        "message_delay_s": _float(event.get("message_delay_s"), 0.0),
                        "nis": _float(event.get("nis"), 0.0),
                        "measurement_variance_m2": _float(event.get("measurement_variance_m2"), 0.0),
                        "effective_measurement_variance_m2": effective_variance,
                        "local_direction_variance_m2": _float(event.get("local_direction_variance_m2"), 0.0),
                        "neighbor_direction_variance_m2": _float(event.get("neighbor_direction_variance_m2"), 0.0),
                        "unknown_cross_bound_m2": unknown_bound,
                        "unknown_cross_bound_fraction": unknown_bound / denominator,
                        "kalman_gain_norm": _float(event.get("kalman_gain_norm"), 0.0),
                        "position_correction_norm_m": _float(event.get("position_correction_norm_m"), 0.0),
                        "covariance_trace_before_m2": cov_before,
                        "covariance_trace_after_linear_m2": cov_linear,
                        "covariance_trace_after_final_m2": cov_final,
                        "linear_cov_reduction_m2": cov_before - cov_linear,
                        "final_cov_reduction_m2": cov_before - cov_final,
                        "j_loc_before": _float(event.get("j_loc_before"), 0.0),
                        "j_loc_after": _float(event.get("j_loc_after"), 0.0),
                        "true_position_error_before_m": before_error,
                        "true_position_error_after_m": after_error,
                        "error_change_m": error_change,
                        "effect_class": effect,
                        "proxy_gain": proxy_gain,
                        "proxy_cost": proxy_cost,
                        "proxy_utility": proxy_utility,
                        "proxy_valid": proxy_valid,
                        "proxy_rank": proxy_rank,
                        "replay_action_match": int(bool(matched)),
                        "update_quality_label_available": int(
                            "true_position_error_before_m" in event and "true_position_error_after_m" in event
                        ),
                    }
                )
    return output, replay_counts


def _trace_group_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return int(row.get("base_seed", 0)), str(row.get("branch", "")), str(row.get("episode_id", ""))


def _episode_metrics(reference_root: Path, seeds: Sequence[int], branches: Sequence[str]) -> list[dict[str, Any]]:
    root_manifest = _read_json(reference_root / "manifest.json")
    candidates: list[Path] = []
    output_path = root_manifest.get("outputs", {}).get("episode_metrics")
    if output_path:
        candidates.append(Path(str(output_path)))
    candidates.append(Path("logs/preliminary") / reference_root.name / "episode_metrics.csv")
    metrics_path = next((path for path in candidates if path.exists()), None)
    if metrics_path is None:
        raise FileNotFoundError(f"cannot locate episode_metrics.csv for {reference_root}")
    selected = _read_csv(metrics_path)
    seed_set = {str(int(seed)) for seed in seeds}
    branch_set = set(branches)
    return [
        row
        for row in selected
        if str(row.get("split")) == "validation"
        and str(row.get("base_seed", row.get("seed", ""))) in seed_set
        and str(row.get("treatment", "")) in branch_set
    ]


def _episode_summaries(
    reference_metrics: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    traces: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    teachers: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    candidates: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    updates: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for row in trace_rows:
        traces.setdefault(_trace_group_key(row), []).append(row)
    for row in teacher_rows:
        teachers.setdefault(_trace_group_key(row), []).append(row)
    for row in candidate_rows:
        candidates.setdefault(_trace_group_key(row), []).append(row)
    for row in update_rows:
        updates.setdefault(_trace_group_key(row), []).append(row)

    output: list[dict[str, Any]] = []
    for source in reference_metrics:
        seed = int(_float(source.get("base_seed", source.get("seed")), 0.0))
        branch = str(source.get("treatment", ""))
        episode_id = str(source.get("episode_id", ""))
        key = (seed, branch, episode_id)
        group = traces.get(key, [])
        teacher_group = teachers.get(key, [])
        candidate_group = candidates.get(key, [])
        update_group = updates.get(key, [])
        selected = [row for row in group if str(row.get("submitted_measurement_uav_id", "none")) != "none"]
        overhead = [row for row in selected if int(_float(row.get("overhead_event_present"), 0.0)) == 1]
        beneficial = sum(str(row.get("effect_class")) == "beneficial" for row in update_group)
        harmful = sum(str(row.get("effect_class")) == "harmful" for row in update_group)
        neutral = sum(str(row.get("effect_class")) == "approximately_neutral" for row in update_group)
        quality = [
            _float(row.get("linear_cov_reduction_m2"), float("nan"))
            for row in update_group
        ]
        error_changes = [_float(row.get("error_change_m"), float("nan")) for row in update_group]
        proxy_updates = [row for row in update_group if int(_float(row.get("proxy_valid"), 0.0)) == 1]
        actor_teacher = [_float(row.get("teacher_actor_agreement"), float("nan")) for row in teacher_group]
        teacher_kl = [_float(row.get("teacher_kl_actor"), float("nan")) for row in teacher_group]
        selected_proxy: list[float] = []
        for row in group:
            selected_id = str(row.get("submitted_measurement_uav_id", "none"))
            if selected_id == "none":
                continue
            utilities = _array(row.get("measurement_proxy_utility", []), dtype=float)
            selected_index = _uav_index(selected_id)
            if 0 <= selected_index < utilities.size and math.isfinite(float(utilities[selected_index])):
                selected_proxy.append(float(utilities[selected_index]))
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "split": "validation",
                "episode_id": episode_id,
                "scenario_id": str(source.get("scenario_id", "")),
                "sample_unit": "episode",
                "mean_reward": _float(source.get("mean_reward"), 0.0),
                "position_rmse_m": _float(source.get("position_rmse_m"), 0.0),
                "position_error_p95_m": _float(source.get("position_error_p95_m"), 0.0),
                "j_loc": _float(source.get("j_loc"), 0.0),
                "j_com": _float(source.get("j_com"), 0.0),
                "joint_objective": _float(source.get("joint_objective"), 0.0),
                "packet_loss": _float(source.get("packet_loss"), 0.0),
                "delay_s": _float(source.get("delay_s"), 0.0),
                "aoi_s": _float(source.get("aoi_s"), 0.0),
                "connectivity": _float(source.get("connectivity"), 0.0),
                "energy_proxy": _float(source.get("energy_proxy"), 0.0),
                "measurement_count": _float(source.get("measurement_count"), len(selected)),
                "measurement_overhead_packets": _float(source.get("measurement_overhead_packets"), len(overhead)),
                "accepted_update_count": float(len(update_group)),
                "reference_accepted_update_count": _float(source.get("accepted_update_count"), 0.0),
                "beneficial_fraction": beneficial / max(1, len(update_group)),
                "harmful_fraction": harmful / max(1, len(update_group)),
                "neutral_fraction": neutral / max(1, len(update_group)),
                "teacher_agreement": _safe_mean(actor_teacher, default=0.0),
                "teacher_kl_actor": _safe_mean(teacher_kl, default=0.0),
                "candidate_count": float(len(candidate_group)),
                "candidate_proxy_valid_fraction": _safe_mean(
                    [_float(row.get("proxy_valid"), 0.0) for row in candidate_group], default=0.0
                ),
                "selected_proxy_utility": _safe_mean(selected_proxy, default=0.0),
                "proxy_update_count": float(len(proxy_updates)),
                "linear_cov_reduction_m2": _safe_mean(quality, default=0.0),
                "error_change_m": _safe_mean(error_changes, default=0.0),
                "nlos_update_fraction": _safe_mean(
                    [_float(row.get("los_label_offline"), 0.0) == 0.0 for row in update_group], default=0.0
                ),
                "stale_update_fraction": _safe_mean(
                    [_float(row.get("message_age_s"), 0.0) > 0.8 for row in update_group], default=0.0
                ),
                "replay_selected_count": float(len(selected)),
                "replay_overhead_event_count": float(len(overhead)),
                "replay_trace_rows": float(len(group)),
            }
        )
    return output


def _mean_by_seed(rows: Sequence[Mapping[str, Any]], *, group_field: str = "branch") -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row.get("base_seed", 0)), str(row.get(group_field, row.get("branch", "")))), []).append(row)
    numeric = set(METRICS) | {
        "position_error_p95_m",
        "joint_objective",
        "packet_loss",
        "delay_s",
        "aoi_s",
        "connectivity",
        "energy_proxy",
        "teacher_kl_actor",
        "candidate_count",
        "candidate_proxy_valid_fraction",
        "proxy_update_count",
        "nlos_update_fraction",
        "stale_update_fraction",
        "replay_selected_count",
        "replay_overhead_event_count",
        "replay_trace_rows",
    }
    output: list[dict[str, Any]] = []
    for (seed, label), group in sorted(groups.items()):
        item: dict[str, Any] = {"base_seed": seed, "branch": label, "episode_count": len(group)}
        for key in sorted(numeric):
            item[key] = _safe_mean([_float(row.get(key), float("nan")) for row in group], default=0.0)
        output.append(item)
    return output


def _communication_tradeoff(episode_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in episode_rows:
        groups.setdefault((int(row.get("base_seed", 0)), str(row.get("branch", ""))), []).append(row)
    for (seed, branch), group in sorted(groups.items()):
        measurement = [_float(row.get("measurement_count"), 0.0) for row in group]
        jcom = [_float(row.get("j_com"), 0.0) for row in group]
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "episode_count": len(group),
                "mean_measurement_count": _safe_mean(measurement),
                "mean_overhead_packets": _safe_mean([_float(row.get("measurement_overhead_packets"), 0.0) for row in group]),
                "mean_queue_proxy_update_count": _safe_mean([_float(row.get("proxy_update_count"), 0.0) for row in group]),
                "mean_packet_loss": _safe_mean([_float(row.get("packet_loss"), 0.0) for row in group]),
                "mean_delay_s": _safe_mean([_float(row.get("delay_s"), 0.0) for row in group]),
                "mean_aoi_s": _safe_mean([_float(row.get("aoi_s"), 0.0) for row in group]),
                "mean_connectivity": _safe_mean([_float(row.get("connectivity"), 0.0) for row in group]),
                "mean_j_com": _safe_mean(jcom),
                "mean_j_loc": _safe_mean([_float(row.get("j_loc"), 0.0) for row in group]),
                "mean_position_rmse_m": _safe_mean([_float(row.get("position_rmse_m"), 0.0) for row in group]),
                "mean_reward": _safe_mean([_float(row.get("mean_reward"), 0.0) for row in group]),
                "measurement_j_com_pearson_r": _pearson(measurement, jcom),
            }
        )
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else 0.0


def _paired_contrasts(seed_rows: Sequence[Mapping[str, Any]], *, seeds: Sequence[int], branches: Sequence[str], config: ProxyUpdateQualityAuditConfig) -> list[dict[str, Any]]:
    index = {(int(row.get("base_seed", 0)), str(row.get("branch", ""))): row for row in seed_rows}
    comparisons = [
        ("adaptive_minus_baseline", "adaptive_budgeted_quality_aware_mappo", "baseline_mappo"),
        ("proxy_minus_baseline", "proxy_guided_adaptive_budgeted_quality_aware_mappo", "baseline_mappo"),
        ("proxy_minus_adaptive", "proxy_guided_adaptive_budgeted_quality_aware_mappo", "adaptive_budgeted_quality_aware_mappo"),
    ]
    output: list[dict[str, Any]] = []
    for label, treatment, control in comparisons:
        if treatment not in branches or control not in branches:
            continue
        for metric in METRICS:
            differences: list[float] = []
            used_seeds: list[int] = []
            for seed in seeds:
                left = index.get((int(seed), treatment))
                right = index.get((int(seed), control))
                if left is None or right is None:
                    continue
                value = _float(left.get(metric), float("nan")) - _float(right.get(metric), float("nan"))
                if math.isfinite(value):
                    differences.append(value)
                    used_seeds.append(int(seed))
            # ``hash`` is randomized between Python processes.  A stable
            # offset keeps bootstrap results reproducible across reruns.
            stable_key = f"{label}:{metric}".encode("utf-8")
            stable_offset = int.from_bytes(hashlib.sha256(stable_key).digest()[:4], "big") % 100_000
            low, high = _bootstrap_ci(
                differences,
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + stable_offset,
            )
            output.append(
                {
                    "comparison": label,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                    "independent_unit": "base_seed",
                    "seed_count": len(differences),
                    "seeds": used_seeds,
                    "mean_difference": _safe_mean(differences, default=0.0),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "improved_seed_count": int(
                        sum(value < 0.0 for value in differences)
                        if metric in LOWER_IS_BETTER
                        else sum(value > 0.0 for value in differences)
                    ),
                }
            )
    return output


def _proxy_alignment(update_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        group = [
            row for row in update_rows
            if int(row.get("base_seed", 0)) == int(seed) and int(_float(row.get("proxy_valid"), 0.0)) == 1
        ]
        utilities = np.asarray([_float(row.get("proxy_utility"), float("nan")) for row in group], dtype=float)
        finite = utilities[np.isfinite(utilities)]
        if finite.size < 4:
            continue
        threshold = float(np.median(finite))
        high = [row for row in group if _float(row.get("proxy_utility"), float("nan")) >= threshold]
        low = [row for row in group if _float(row.get("proxy_utility"), float("nan")) < threshold]
        if len(high) < 2 or len(low) < 2:
            continue
        high_cov = _safe_mean([_float(row.get("linear_cov_reduction_m2"), float("nan")) for row in high], default=0.0)
        low_cov = _safe_mean([_float(row.get("linear_cov_reduction_m2"), float("nan")) for row in low], default=0.0)
        high_harmful = _safe_mean([str(row.get("effect_class")) == "harmful" for row in high], default=0.0)
        low_harmful = _safe_mean([str(row.get("effect_class")) == "harmful" for row in low], default=0.0)
        per_seed.append(
            {
                "base_seed": int(seed),
                "high_minus_low_linear_cov_reduction_m2": high_cov - low_cov,
                "high_minus_low_harmful_fraction": high_harmful - low_harmful,
                "high_count": len(high),
                "low_count": len(low),
            }
        )
    aligned = [
        row for row in per_seed
        if _float(row.get("high_minus_low_linear_cov_reduction_m2"), 0.0) > 0.0
        and _float(row.get("high_minus_low_harmful_fraction"), 0.0) <= 0.0
    ]
    considered = len(per_seed)
    misaligned = considered >= max(3, math.ceil(len(seeds) / 2)) and len(aligned) < math.ceil(0.6 * considered)
    return {
        "per_seed": per_seed,
        "considered_seed_count": considered,
        "aligned_seed_count": len(aligned),
        "misaligned": bool(misaligned),
    }


def _plot(
    plot_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2f6de1",
        "adaptive_budgeted_quality_aware_mappo": "#1b9e5a",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo": "#d92d2d",
    }
    labels = {
        "baseline_mappo": "baseline",
        "adaptive_budgeted_quality_aware_mappo": "adaptive",
        "proxy_guided_adaptive_budgeted_quality_aware_mappo": "proxy-guided",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    for branch in DEFAULT_BRANCHES:
        group = [row for row in update_rows if str(row.get("branch")) == branch]
        if group:
            axes[0, 0].scatter(
                [_float(row.get("proxy_utility"), 0.0) for row in group],
                [_float(row.get("linear_cov_reduction_m2"), 0.0) for row in group],
                s=9,
                alpha=0.3,
                label=labels[branch],
                color=colors[branch],
            )
    axes[0, 0].set_title("Proxy utility vs linear covariance reduction")
    axes[0, 0].set_xlabel("proxy utility")
    axes[0, 0].set_ylabel("linear covariance reduction (m²)")

    branches = list(DEFAULT_BRANCHES)
    x = np.arange(len(branches))
    for offset, branch in enumerate(branches):
        group = [row for row in update_rows if str(row.get("branch")) == branch]
        total = max(1, len(group))
        values = [
            sum(str(row.get("effect_class")) == category for row in group) / total
            for category in ("beneficial", "approximately_neutral", "harmful")
        ]
        axes[0, 1].bar(x[offset] + np.arange(3) * 0.22 - 0.22, values, width=0.2, color=("#2ca25f", "#bdbdbd", "#de2d26"), label=labels[branch] if offset == 0 else None)
        for idx, value in enumerate(values):
            axes[0, 1].text(x[offset] + idx * 0.22 - 0.22, value + 0.01, f"{value:.2f}", ha="center", fontsize=7)
    axes[0, 1].set_title("Accepted update quality")
    axes[0, 1].set_xticks(x, [labels[b] for b in branches])
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("fraction")

    for branch in branches:
        group = [row for row in seed_rows if str(row.get("branch")) == branch]
        if group:
            axes[0, 2].scatter(
                [_float(row.get("accepted_update_count"), 0.0) for row in group],
                [_float(row.get("position_rmse_m"), 0.0) for row in group],
                label=labels[branch], color=colors[branch], s=45,
            )
    axes[0, 2].set_title("Accepted updates vs position RMSE")
    axes[0, 2].set_xlabel("accepted updates / episode")
    axes[0, 2].set_ylabel("RMSE (m)")

    for branch in branches:
        group = [row for row in seed_rows if str(row.get("branch")) == branch]
        if group:
            axes[1, 0].scatter(
                [_float(row.get("measurement_count"), 0.0) for row in group],
                [_float(row.get("j_com"), 0.0) for row in group],
                label=labels[branch], color=colors[branch], s=45,
            )
    axes[1, 0].set_title("Measurement count vs communication cost")
    axes[1, 0].set_xlabel("measurements / episode")
    axes[1, 0].set_ylabel("J_com")

    agreement = []
    for branch in branches:
        group = [row for row in seed_rows if str(row.get("branch")) == branch]
        agreement.append(_safe_mean([_float(row.get("teacher_agreement"), 0.0) for row in group], default=0.0))
    axes[1, 1].bar([labels[b] for b in branches], agreement, color=[colors[b] for b in branches])
    axes[1, 1].set_title("Teacher—actor agreement")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("fraction")

    rmse_contrasts = [
        row for row in contrasts
        if row.get("comparison") == "proxy_minus_baseline" and row.get("metric") == "position_rmse_m"
    ]
    jcom_contrasts = [
        row for row in contrasts
        if row.get("comparison") == "proxy_minus_baseline" and row.get("metric") == "j_com"
    ]
    names = ["RMSE", "J_com"]
    means = [
        _float(rmse_contrasts[0].get("mean_difference"), 0.0) if rmse_contrasts else 0.0,
        _float(jcom_contrasts[0].get("mean_difference"), 0.0) if jcom_contrasts else 0.0,
    ]
    lows = [
        _float(rmse_contrasts[0].get("bootstrap_95_ci_low"), 0.0) if rmse_contrasts else 0.0,
        _float(jcom_contrasts[0].get("bootstrap_95_ci_low"), 0.0) if jcom_contrasts else 0.0,
    ]
    highs = [
        _float(rmse_contrasts[0].get("bootstrap_95_ci_high"), 0.0) if rmse_contrasts else 0.0,
        _float(jcom_contrasts[0].get("bootstrap_95_ci_high"), 0.0) if jcom_contrasts else 0.0,
    ]
    axes[1, 2].errorbar(names, means, yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)], fmt="o", capsize=5, color="#7a0177")
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_title("Proxy-guided − baseline seed contrasts")
    axes[1, 2].set_ylabel("difference with 95% CI")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        handles, _legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    fig.suptitle("Proxy-guided update-quality audit — validation diagnostic only", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def _main_decision(
    *,
    alignment: Mapping[str, Any],
    teacher_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> tuple[str, dict[str, Any]]:
    alignment_misaligned = bool(alignment.get("misaligned", False))
    teacher_by_seed: dict[int, float] = {}
    for seed in seeds:
        group = [row for row in teacher_rows if int(row.get("base_seed", 0)) == int(seed)]
        teacher_by_seed[int(seed)] = _safe_mean([_float(row.get("teacher_actor_agreement"), float("nan")) for row in group], default=0.0)
    teacher_consistent = sum(value >= 0.5 for value in teacher_by_seed.values())
    teacher_failure = len(teacher_by_seed) >= 3 and teacher_consistent < math.ceil(0.6 * len(teacher_by_seed))

    def _contrast(comparison: str, metric: str) -> Mapping[str, Any] | None:
        return next((row for row in contrasts if row.get("comparison") == comparison and row.get("metric") == metric), None)

    rmse = _contrast("proxy_minus_baseline", "position_rmse_m")
    jcom = _contrast("proxy_minus_baseline", "j_com")
    reward = _contrast("proxy_minus_baseline", "mean_reward")
    measurements = _contrast("proxy_minus_baseline", "measurement_count")
    communication_dominant = bool(
        rmse is not None
        and jcom is not None
        and measurements is not None
        and _float(jcom.get("mean_difference"), 0.0) > 0.0
        and _float(measurements.get("mean_difference"), 0.0) > 0.0
        and (
            _float(reward.get("mean_difference"), 0.0) < 0.0 if reward is not None else True
        )
    )
    ekf_quality_issue = bool(
        not communication_dominant
        and rmse is not None
        and _float(rmse.get("mean_difference"), 0.0) >= 0.0
    )
    if alignment_misaligned:
        cause = "proxy_misaligned_with_update_quality"
    elif teacher_failure:
        cause = "teacher_alignment_failure"
    elif communication_dominant:
        cause = "communication_cost_dominant"
    elif ekf_quality_issue:
        cause = "ekf_update_effective_quality_issue"
    else:
        cause = "insufficient_evidence"
    details = {
        "proxy_alignment": alignment,
        "teacher_agreement_by_seed": teacher_by_seed,
        "teacher_failure": teacher_failure,
        "communication_dominant": communication_dominant,
        "ekf_quality_issue": ekf_quality_issue,
        "priority_order": [
            "proxy_misaligned_with_update_quality",
            "teacher_alignment_failure",
            "communication_cost_dominant",
            "ekf_update_effective_quality_issue",
            "insufficient_evidence",
        ],
    }
    return cause, details


def run_proxy_guided_update_quality_audit(config: ProxyUpdateQualityAuditConfig) -> dict[str, Any]:
    """Replay and attribute frozen proxy-guided validation updates."""

    reference_root = Path(config.reference_root).resolve()
    output_dir = Path(config.output_dir).resolve()
    plots_dir = Path(config.plots_dir).resolve() if config.plots_dir else (
        Path("plots/preliminary") / reference_root.name / "proxy_update_quality_audit"
    ).resolve()
    _prepare_output(output_dir, _MANAGED_LOG_FILES, force=config.force)
    _prepare_output(plots_dir, _MANAGED_PLOT_FILES, force=config.force)
    trace_path = output_dir / "replay_policy_trace.jsonl"
    if trace_path.exists() and config.force:
        trace_path.unlink()

    checkpoint_paths = [
        _manifest_for(reference_root, int(seed), branch)[0] / "checkpoints" / "final.pt"
        for seed in config.seeds for branch in config.branches
    ]
    checkpoint_hash_before = {str(path): _sha256(path) for path in checkpoint_paths}
    trace_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    proxy_parameters: dict[str, Any] = {}
    for seed in config.seeds:
        # The proxy is fitted from baseline train observations and is shared
        # by all replay branches for this seed.  The legacy branches do not
        # necessarily contain their own calibration copy.
        proxy_scale_root, _proxy_manifest = _manifest_for(
            reference_root,
            int(seed),
            "proxy_guided_adaptive_budgeted_quality_aware_mappo",
        )
        proxy_model = _load_proxy_model(proxy_scale_root)
        proxy_parameters[str(seed)] = proxy_model
        for branch in config.branches:
            if config.progress:
                print(f"[proxy-update-audit] start seed={seed} branch={branch}", flush=True)
            replay = _replay_branch(
                reference_root=reference_root,
                base_seed=int(seed),
                branch=branch,
                split=config.split,
                trace_path=trace_path,
                progress=config.progress,
                diagnostic_proxy_model=proxy_model,
            )
            full_trace_rows = replay["rows"]
            # Expand and aggregate while this branch is still isolated, then
            # release the nested link/state payloads before replaying the next
            # branch.  The complete rows remain available on disk in the
            # replay trace JSONL.
            candidate_rows.extend(
                _candidate_rows(full_trace_rows, weight=config.communication_cost_weight)
            )
            teacher_rows.extend(
                _teacher_rows(
                    full_trace_rows,
                    weight=config.communication_cost_weight,
                    temperature=config.temperature,
                )
            )
            trace_rows.extend(_compact_trace_row(row) for row in full_trace_rows)
            replay_checks.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "action_mismatches": int(replay["action_mismatches"]),
                    "reference_action_rows": int(replay["reference_action_rows"]),
                    "overhead_mismatches": int(replay["overhead_mismatches"]),
                    "reference_overhead_rows": int(replay["reference_overhead_rows"]),
                    "replay_overhead_rows": int(replay["replay_overhead_rows"]),
                    "quality_model_source": str(replay["quality_model"].get("source_split", "")),
                    "proxy_model_source": str(proxy_model.get("source_split", "")),
                    "proxy_validation_used_for_fitting": bool(proxy_model.get("validation_used_for_fitting", False)),
                    "proxy_test_used_for_fitting": bool(proxy_model.get("test_used_for_fitting", False)),
                    "measurement_budget_enabled": bool(replay["config"].measurement_budget_enabled),
                }
            )
            del full_trace_rows
            del replay
            gc.collect()

    update_rows, replay_update_counts = _accepted_update_rows(
        reference_root,
        config.seeds,
        config.branches,
        candidate_rows,
    )
    reference_metrics = _episode_metrics(reference_root, config.seeds, config.branches)
    episode_rows = _episode_summaries(
        reference_metrics,
        trace_rows,
        teacher_rows,
        candidate_rows,
        update_rows,
    )
    seed_rows = _mean_by_seed(episode_rows)
    contrasts = _paired_contrasts(
        seed_rows,
        seeds=config.seeds,
        branches=config.branches,
        config=config,
    )
    communication_rows = _communication_tradeoff(episode_rows)
    checkpoint_hash_after = {str(path): _sha256(path) for path in checkpoint_paths}
    alignment = _proxy_alignment(update_rows, config.seeds)
    decision, decision_details = _main_decision(
        alignment=alignment,
        teacher_rows=teacher_rows,
        seed_rows=seed_rows,
        contrasts=contrasts,
        seeds=config.seeds,
    )

    episode_keys = {
        (int(row.get("base_seed", 0)), str(row.get("branch", "")), str(row.get("episode_id", "")))
        for row in episode_rows
    }
    expected_episode_keys = {
        (int(seed), branch, f"episode_{400000 + index:06d}")
        for seed in config.seeds
        for branch in config.branches
        for index in range(1, 21)
    }
    replay_update_match = True
    for seed in config.seeds:
        for branch in config.branches:
            replay_count = sum(
                int(_float(row.get("accepted_update_count_step"), 0.0))
                for row in trace_rows
                if int(row.get("base_seed", 0)) == int(seed) and str(row.get("branch")) == branch
            )
            if replay_count != int(replay_update_counts.get((int(seed), branch), 0)):
                replay_update_match = False
    expected_trace_keys = {(int(seed), branch) for seed in config.seeds for branch in config.branches}
    actual_trace_keys = {(int(row.get("base_seed", 0)), str(row.get("branch", ""))) for row in trace_rows}
    reference_complete = len(reference_metrics) == len(expected_episode_keys) and episode_keys == expected_episode_keys
    replay_actions_exact = bool(replay_checks) and all(
        int(row["action_mismatches"]) == 0 and int(row["reference_action_rows"]) > 0
        for row in replay_checks
    )
    replay_overhead_exact = bool(replay_checks) and all(
        int(row["overhead_mismatches"]) == 0
        and int(row["reference_overhead_rows"]) == int(row["replay_overhead_rows"])
        for row in replay_checks
    )
    proxy_train_only = bool(replay_checks) and all(
        row["proxy_model_source"] == "train"
        and not row["proxy_validation_used_for_fitting"]
        and not row["proxy_test_used_for_fitting"]
        for row in replay_checks
    )
    quality_warmup_only = bool(replay_checks) and all(row["quality_model_source"] == "warmup" for row in replay_checks)
    validation_only = bool(trace_rows) and all(str(row.get("split")) == "validation" for row in trace_rows)
    candidate_validation_only = bool(candidate_rows) and all(str(row.get("split", "validation")) == "validation" for row in candidate_rows)
    update_validation_only = bool(update_rows) and all(str(row.get("split")) == "validation" for row in update_rows)
    forbidden_candidate_keys = sorted(
        {
            str(key)
            for row in candidate_rows
            for key in row
            if any(token in str(key).lower() for token in ("true_position", "true_error", "true_range", "los_nlos", "future_state", "post_update_nis"))
        }
    )
    candidate_finite = all(_finite_trace_value(row) for row in candidate_rows)
    teacher_finite = all(_finite_trace_value(row) for row in teacher_rows)
    update_finite = all(_finite_trace_value(row) for row in update_rows)
    trace_finite = all(_finite_trace_value(row) for row in trace_rows)
    recomputed_teacher_match = True
    for row in trace_rows:
        expected_teacher = _teacher_probabilities(
            _array(row.get("measurement_proxy_gain", []), dtype=float),
            _array(row.get("measurement_proxy_cost", []), dtype=float),
            _array(row.get("measurement_proxy_valid", []), dtype=float),
            _array(row.get("action_mask", []), dtype=float),
            temperature=config.temperature,
            communication_cost_weight=config.communication_cost_weight,
        )
        actual_teacher = _safe_array(row.get("proxy_teacher_probabilities", []), expected_teacher.size, 0.0)
        if expected_teacher.size != actual_teacher.size or not np.allclose(expected_teacher, actual_teacher, atol=1e-6, rtol=1e-6):
            recomputed_teacher_match = False
            break
    hash_unchanged = checkpoint_hash_before == checkpoint_hash_after
    audit_ok = bool(
        reference_complete
        and actual_trace_keys == expected_trace_keys
        and replay_actions_exact
        and replay_overhead_exact
        and replay_update_match
        and proxy_train_only
        and quality_warmup_only
        and validation_only
        and candidate_validation_only
        and update_validation_only
        and not forbidden_candidate_keys
        and trace_finite
        and candidate_finite
        and teacher_finite
        and update_finite
        and recomputed_teacher_match
        and hash_unchanged
        and bool(update_rows)
    )

    candidate_columns = sorted({key for row in candidate_rows for key in row})
    update_columns = sorted({key for row in update_rows for key in row})
    teacher_columns = sorted({key for row in teacher_rows for key in row})
    _write_csv(output_dir / "candidate_proxy_events.csv", candidate_columns or ["base_seed"], candidate_rows)
    _write_csv(output_dir / "accepted_update_quality.csv", update_columns or ["base_seed"], update_rows)
    _write_csv(output_dir / "teacher_alignment.csv", teacher_columns or ["base_seed"], teacher_rows)
    communication_columns = sorted({key for row in communication_rows for key in row})
    _write_csv(output_dir / "communication_tradeoff.csv", communication_columns or ["base_seed"], communication_rows)
    episode_columns = sorted({key for row in episode_rows for key in row})
    _write_csv(output_dir / "seed_episode_summary.csv", episode_columns or ["base_seed"], episode_rows)
    contrast_columns = sorted({key for row in contrasts for key in row})
    _write_csv(output_dir / "paired_contrasts.csv", contrast_columns or ["comparison"], contrasts)
    (output_dir / "proxy_parameters.json").write_text(
        json.dumps(_jsonable(proxy_parameters), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit_report: dict[str, Any] = {
        "audit": "proxy_guided_update_quality_attribution",
        "schema_version": "proxy-update-quality-audit-v1",
        "reference_root": str(reference_root),
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "split": config.split,
        "branches": list(config.branches),
        "seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_unit_before_seed_aggregation": True,
        "test_data_used": False,
        "test_access_count": 0,
        "trainer_update_called": False,
        "optimizer_step_called": False,
        "checkpoint_hash_unchanged": hash_unchanged,
        "checkpoint_hashes_before": checkpoint_hash_before,
        "checkpoint_hashes_after": checkpoint_hash_after,
        "checks": {
            "reference_episode_complete": reference_complete,
            "branch_seed_trace_complete": actual_trace_keys == expected_trace_keys,
            "replay_actions_exact": replay_actions_exact,
            "replay_queue_and_overhead_exact": replay_overhead_exact,
            "accepted_update_count_matches_replay": replay_update_match,
            "proxy_models_train_only": proxy_train_only,
            "quality_models_warmup_only": quality_warmup_only,
            "all_validation_only": validation_only and candidate_validation_only and update_validation_only,
            "candidate_proxy_fields_finite": candidate_finite,
            "teacher_fields_finite": teacher_finite,
            "accepted_update_fields_finite": update_finite,
            "trace_fields_finite": trace_finite,
            "candidate_proxy_forbidden_inputs": forbidden_candidate_keys,
            "teacher_recomputation_exact": recomputed_teacher_match,
            "no_parameter_update": True,
            "original_checkpoints_unchanged": hash_unchanged,
        },
        "replay_checks": replay_checks,
        "counts": {
            "trace_rows": len(trace_rows),
            "candidate_proxy_rows": len(candidate_rows),
            "teacher_alignment_rows": len(teacher_rows),
            "accepted_update_rows": len(update_rows),
            "episode_rows": len(episode_rows),
            "seed_rows": len(seed_rows),
            "paired_contrast_rows": len(contrasts),
        },
        "decision": decision,
        "decision_details": decision_details,
        "interpretation_boundary": (
            "Accepted update truth fields and LOS labels are validation-only diagnostic labels; "
            "they are never used in proxy computation, teacher construction, action selection, or training."
        ),
        "audit_ok": audit_ok,
        "decision_supported": False,
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(_jsonable(audit_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(
        plots_dir / "proxy_update_quality_audit.png",
        candidate_rows,
        update_rows,
        episode_rows,
        seed_rows,
        contrasts,
    )
    return {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "audit_report": str(output_dir / "audit_report.json"),
        "plot": str(plots_dir / "proxy_update_quality_audit.png"),
        "decision": decision,
        "audit_ok": audit_ok,
        "decision_supported": False,
        "trace_rows": len(trace_rows),
        "accepted_update_rows": len(update_rows),
    }


__all__ = ["ProxyUpdateQualityAuditConfig", "run_proxy_guided_update_quality_audit"]
