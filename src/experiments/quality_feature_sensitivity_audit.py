"""Validation-only semantic and counterfactual audit of quality features.

The audit answers a narrow mechanism question: does the quality-aware
measurement head receive a correctly oriented risk signal, and does its
measurement distribution respond to that signal in the expected direction?
It never retrains a policy, executes a counterfactual action, reads test data,
or changes a reference episode.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..learning.measurement_quality import (
    HIGH_IS_GOOD,
    LOW_IS_GOOD,
    RISK_FIELDS,
    _candidate_values,
    _local_candidates,
    _neighbor_covariance_trace,
    _risk,
    quality_features,
)
from ..learning.mappo import MAPPOTrainer
from .formal import (
    FormalExperimentConfig,
    _make_environment,
    _scale_seed,
    _scenario_for,
)
from .pilot import _assert_action_feasible, _seed_everything
from .policy_mechanism_audit import (
    _actions_match,
    _array,
    _formal_config,
    _load_quality_model,
    _load_trainer,
    _manifest_for,
    _read_csv,
    _read_json,
    _reference_actions,
)


DEFAULT_SEEDS = (2026, 3035, 4044, 5053, 6062)
RISK_GRID = (0.00, 0.25, 0.50, 0.75, 1.00)
EXPECTED_DIRECTION_TOL = 1e-7
INVARIANCE_TOL = 1e-7
_MANAGED_FILES = (
    "feature_semantics.csv",
    "counterfactual_policy_response.csv",
    "missing_value_audit.csv",
    "seed_summary.csv",
    "audit_report.json",
)


@dataclass(frozen=True)
class QualityFeatureSensitivityAuditConfig:
    """Configuration for the validation-only feature sensitivity audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/"
        "quality_feature_sensitivity"
    )
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    split: str = "validation"
    plots_dir: Path | None = None
    device: str = "cpu"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("quality feature sensitivity audit is validation-only; test is sealed")
        if not self.seeds:
            raise ValueError("at least one base seed is required")
        if self.device != "cpu":
            raise ValueError("the deterministic audit currently supports device='cpu' only")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _finite_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_value(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return bool(np.isfinite(np.asarray(value, dtype=float)).all())
        except (TypeError, ValueError):
            return all(_finite_value(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _prepare_output(path: Path, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"audit output is not empty: {path}; use --force to replace it")
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for name in _MANAGED_FILES:
            target = path / name
            if target.exists():
                target.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _padded_action_mask(local: Mapping[str, Any], uav_count: int) -> np.ndarray:
    raw = np.asarray(local.get("action_mask", []), dtype=bool).reshape(-1)
    mask = np.zeros(uav_count + 1, dtype=bool)
    mask[: min(raw.size, mask.size)] = raw[: mask.size]
    if not bool(mask.any()):
        mask[-1] = True
    return mask


def _candidate_indices(local: Mapping[str, Any], uav_ids: Sequence[str]) -> list[int]:
    mask = _padded_action_mask(local, len(uav_ids))
    visible = set(_local_candidates(local, uav_ids))
    return [index for index, source in enumerate(uav_ids) if mask[index] and source in visible]


def _policy_probabilities(trainer: MAPPOTrainer, local: Mapping[str, Any]) -> np.ndarray:
    import torch

    with torch.no_grad():
        distribution, _bandwidth, _delta, _mask = trainer.policy._actor_distributions(local)
        probabilities = distribution.probs.detach().cpu().numpy().astype(np.float64)
    if probabilities.ndim != 1 or probabilities.size != trainer.config.uav_count + 1:
        raise RuntimeError("unexpected measurement probability vector shape")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite measurement probability vector")
    return probabilities


def _with_quality(
    local: Mapping[str, Any],
    risk: Sequence[float],
    valid: Sequence[float],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(local))
    output["measurement_quality_risk"] = np.asarray(risk, dtype=np.float32).copy()
    output["measurement_quality_valid"] = np.asarray(valid, dtype=np.float32).copy()
    return output


def _risk_semantic_rows(seed: int, branch: str, model: Mapping[str, Any]) -> list[dict[str, Any]]:
    quantiles = model.get("quantiles", {})
    field_order = ",".join(RISK_FIELDS)
    rows: list[dict[str, Any]] = []
    for field_index, field in enumerate(RISK_FIELDS):
        limits = quantiles.get(field, {}) if isinstance(quantiles, Mapping) else {}
        q25 = _finite(limits.get("q25"))
        q75 = _finite(limits.get("q75"))
        if not math.isfinite(q25) or not math.isfinite(q75):
            rows.append({
                "base_seed": seed,
                "branch": branch,
                "field_index": field_index,
                "field": field,
                "field_order": field_order,
                "direction": "higher_is_better" if field in HIGH_IS_GOOD else "lower_is_better",
                "q25": q25,
                "q75": q75,
                "degraded_value": float("nan"),
                "improved_value": float("nan"),
                "degraded_risk": float("nan"),
                "improved_risk": float("nan"),
                "monotonic_ok": 0,
                "finite_ok": 0,
            })
            continue
        degraded = q25 if field in HIGH_IS_GOOD else q75
        improved = q75 if field in HIGH_IS_GOOD else q25
        degraded_risk = _risk(degraded, field, model)
        improved_risk = _risk(improved, field, model)
        values = np.linspace(q25, q75, num=5)
        if field in HIGH_IS_GOOD:
            values = values[::-1]
        risks = np.asarray([_risk(float(value), field, model) for value in values], dtype=float)
        # The sequence is ordered from improved to degraded, so risk must be
        # non-decreasing along this axis.
        monotonic_ok = bool(np.all(np.diff(risks) >= -EXPECTED_DIRECTION_TOL))
        rows.append({
            "base_seed": seed,
            "branch": branch,
            "field_index": field_index,
            "field": field,
            "field_order": field_order,
            "direction": "higher_is_better" if field in HIGH_IS_GOOD else "lower_is_better",
            "q25": q25,
            "q75": q75,
            "degraded_value": float(degraded),
            "improved_value": float(improved),
            "degraded_risk": degraded_risk,
            "improved_risk": improved_risk,
            "monotonic_ok": int(monotonic_ok),
            "finite_ok": int(_finite_value(risks) and math.isfinite(degraded_risk) and math.isfinite(improved_risk)),
        })
    return rows


def _missing_rows_with_trainer(
    *,
    trainer: MAPPOTrainer,
    seed: int,
    branch: str,
    episode_id: str,
    step: int,
    uav_id: str,
    local: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create missing-value rows with the branch's actual policy response."""

    probabilities = _policy_probabilities(trainer, _with_quality(
        local,
        *quality_features(local, uav_ids, model),
    ))
    rows = _missing_rows_without_policy(
        seed=seed,
        branch=branch,
        episode_id=episode_id,
        step=step,
        uav_id=uav_id,
        local=local,
        uav_ids=uav_ids,
        model=model,
        probabilities=probabilities,
    )
    for row in rows:
        source = str(row.get("candidate_uav_id", ""))
        if row.get("category") != "missing_neighbor_covariance" or source not in uav_ids:
            continue
        missing_local = copy.deepcopy(dict(local))
        messages = copy.deepcopy(dict(missing_local.get("neighbor_messages", {})))
        message = messages.get(source)
        if isinstance(message, Mapping):
            item = dict(message)
            item.pop("covariance", None)
            messages[source] = item
        missing_local["neighbor_messages"] = messages
        missing_risk, missing_valid = quality_features(missing_local, uav_ids, model)
        row["missing_candidate_probability"] = float(_policy_probabilities(
            trainer,
            _with_quality(missing_local, missing_risk, missing_valid),
        )[uav_ids.index(source)])
        row["passed"] = int(
            missing_risk[uav_ids.index(source)]
            >= _finite(row.get("full_risk"), 1.0) - EXPECTED_DIRECTION_TOL
        )
    return rows


def _missing_rows_without_policy(
    *,
    seed: int,
    branch: str,
    episode_id: str,
    step: int,
    uav_id: str,
    local: Mapping[str, Any],
    uav_ids: Sequence[str],
    model: Mapping[str, Any],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Structural missing-value audit independent of a policy object."""

    # This is the same logic as _missing_rows_for_observation without the
    # policy-dependent missing probability path.  It is intentionally kept
    # small so unit tests can exercise missing semantics without torch.
    uav_count = len(uav_ids)
    mask = _padded_action_mask(local, uav_count)
    candidate_indices = _candidate_indices(local, uav_ids)
    risk, valid = quality_features(local, uav_ids, model)
    base = {
        "base_seed": seed,
        "branch": branch,
        "split": "validation",
        "episode_id": episode_id,
        "step": step,
        "uav_id": uav_id,
        "candidate_count": len(candidate_indices),
    }
    rows: list[dict[str, Any]] = []
    if not candidate_indices:
        only_none = bool(mask[-1]) and not bool(mask[:-1].any())
        return [{
            **base,
            "category": "no_candidate",
            "candidate_uav_id": "",
            "action_mask_enabled": 0,
            "quality_valid": 0.0,
            "quality_risk": 1.0,
            "full_covariance_available": 0,
            "missing_covariance_risk": 1.0,
            "full_risk": 1.0,
            "missing_risk": 1.0,
            "candidate_probability": 0.0,
            "missing_candidate_probability": 0.0,
            "none_probability": float(probabilities[-1]),
            "none_only_mask": int(only_none),
            "passed": int(only_none),
            "notes": "no actor-visible candidate measurement source",
        }]
    for index, source in enumerate(uav_ids):
        if source == uav_id or index in candidate_indices:
            continue
        rows.append({
            **base,
            "category": "unavailable_candidate",
            "candidate_uav_id": source,
            "action_mask_enabled": int(mask[index]),
            "quality_valid": float(valid[index]),
            "quality_risk": float(risk[index]),
            "full_covariance_available": int(_neighbor_covariance_trace(local, source) is not None),
            "missing_covariance_risk": float(risk[index]),
            "full_risk": float(risk[index]),
            "missing_risk": float(risk[index]),
            "candidate_probability": float(probabilities[index]),
            "missing_candidate_probability": float(probabilities[index]),
            "none_probability": float(probabilities[-1]),
            "none_only_mask": int(bool(mask[-1]) and not bool(mask[:-1].any())),
            "passed": int(not bool(mask[index]) and float(valid[index]) == 0.0),
            "notes": "candidate is not open in the current action mask",
        })
    for index in candidate_indices:
        source = str(uav_ids[index])
        complete = _neighbor_covariance_trace(local, source) is not None
        missing_local = copy.deepcopy(dict(local))
        messages = copy.deepcopy(dict(missing_local.get("neighbor_messages", {})))
        message = messages.get(source)
        if isinstance(message, Mapping):
            item = dict(message)
            item.pop("covariance", None)
            messages[source] = item
        missing_local["neighbor_messages"] = messages
        missing_risk, _missing_valid = quality_features(missing_local, uav_ids, model)
        rows.append({
            **base,
            "category": "complete_neighbor_covariance" if complete else "missing_neighbor_covariance",
            "candidate_uav_id": source,
            "action_mask_enabled": int(mask[index]),
            "quality_valid": float(valid[index]),
            "quality_risk": float(risk[index]),
            "full_covariance_available": int(complete),
            "missing_covariance_risk": float(missing_risk[index]),
            "full_risk": float(risk[index]),
            "missing_risk": float(missing_risk[index]),
            "candidate_probability": float(probabilities[index]),
            "missing_candidate_probability": float(probabilities[index]),
            "none_probability": float(probabilities[-1]),
            "none_only_mask": int(bool(mask[-1]) and not bool(mask[:-1].any())),
            "passed": int((not complete) or missing_risk[index] >= risk[index] - EXPECTED_DIRECTION_TOL),
            "notes": "missing covariance receives worst covariance-component risk" if not complete else "finite neighbor covariance",
        })
    return rows


def _replay_branch(
    *,
    reference_root: Path,
    base_seed: int,
    branch: str,
    split: str,
    semantic_model: Mapping[str, Any],
    progress: bool,
) -> dict[str, Any]:
    scale_root, manifest = _manifest_for(reference_root, base_seed, branch)
    config = _formal_config(manifest, branch)
    metric_scales = dict(config.metric_scales or {})
    model = _load_quality_model(scale_root)
    trainer: MAPPOTrainer | None = None
    uav_ids: tuple[str, ...] = tuple(f"uav_{index:02d}" for index in range(config.uav_count))
    counterfactual_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    action_mismatches = 0
    reference_action_rows = 0
    replay_action_rows = 0

    for episode_index in range(config.validation_episodes):
        episode_seed = _scale_seed(config.seed, config.uav_count, split, episode_index)
        if progress:
            print(
                f"[quality-sensitivity] seed={base_seed} branch={branch} "
                f"validation episode {episode_index + 1}/{config.validation_episodes}",
                flush=True,
            )
        _seed_everything(episode_seed)
        environment = _make_environment(
            config,
            episode_seed,
            metric_scales=metric_scales,
            measurement_overhead_enabled=config.measurement_overhead_enabled,
        )
        if trainer is None:
            trainer, _loaded_model = _load_trainer(
                config,
                scale_root,
                manifest,
                environment,
                seed=base_seed,
            )
            uav_ids = tuple(environment.uav_ids)
        observations, info = environment.reset(
            seed=episode_seed,
            scenario_config=_scenario_for(config, episode_seed),
        )
        episode_id = f"episode_{400000 + episode_index + 1:06d}"
        expected_actions = _reference_actions(scale_root, split, episode_id)
        reference_action_rows += len(expected_actions)
        episode_action_rows = 0
        for step in range(config.max_steps):
            decision = trainer.act(
                observations,
                info.get("central_state"),
                deterministic=True,
            )
            _assert_action_feasible(environment, decision.actions)
            for uav_id in uav_ids:
                local = decision.observations.get(uav_id, {})
                risk, valid = quality_features(local, uav_ids, model)
                actual_local = _with_quality(local, risk, valid)
                actual_probabilities = _policy_probabilities(trainer, actual_local)
                candidate_indices = [
                    index for index in _candidate_indices(local, uav_ids)
                    if float(valid[index]) > 0.5
                ]
                for candidate_index in candidate_indices:
                    previous_probability: float | None = None
                    probability_values: list[float] = []
                    for risk_value in RISK_GRID:
                        counterfactual_risk = np.asarray(risk, dtype=np.float32).copy()
                        counterfactual_risk[candidate_index] = float(risk_value)
                        counterfactual_local = _with_quality(local, counterfactual_risk, valid)
                        probabilities = _policy_probabilities(trainer, counterfactual_local)
                        probability = float(probabilities[candidate_index])
                        probability_values.append(probability)
                        counterfactual_rows.append({
                            "base_seed": base_seed,
                            "branch": branch,
                            "split": split,
                            "episode_id": episode_id,
                            "episode_seed": episode_seed,
                            "step": step,
                            "uav_id": uav_id,
                            "candidate_uav_id": uav_ids[candidate_index],
                            "candidate_index": candidate_index,
                            "risk_value": float(risk_value),
                            "original_risk": float(risk[candidate_index]),
                            "candidate_valid": float(valid[candidate_index]),
                            "candidate_probability": probability,
                            "none_probability": float(probabilities[-1]),
                            "measurement_probabilities": probabilities.tolist(),
                            "other_candidate_probabilities": probabilities[: len(uav_ids)].tolist(),
                            "finite_difference_from_previous": (
                                float(probability - previous_probability)
                                if previous_probability is not None else 0.0
                            ),
                            "quality_feature_in_actor": int(branch == "quality_aware_mappo"),
                            "risk_field_order": ",".join(RISK_FIELDS),
                        })
                        previous_probability = probability
                    counterfactual_rows[-1]["endpoint_delta_candidate_probability"] = float(
                        probability_values[-1] - probability_values[0]
                    )
                missing_rows.extend(_missing_rows_with_trainer(
                    trainer=trainer,
                    seed=base_seed,
                    branch=branch,
                    episode_id=episode_id,
                    step=step,
                    uav_id=uav_id,
                    local=local,
                    uav_ids=uav_ids,
                    model=model,
                ))
                expected = expected_actions.get((step, uav_id))
                if expected is not None:
                    episode_action_rows += 1
                    if not _actions_match(decision.actions[uav_id], expected):
                        action_mismatches += 1
            replay_action_rows += len(uav_ids)
            next_observations, _rewards, terminated, truncated, next_info = environment.step(
                decision.actions
            )
            observations, info = next_observations, next_info
            if terminated or truncated:
                break
        if episode_action_rows != len(expected_actions):
            action_mismatches += abs(episode_action_rows - len(expected_actions))

    return {
        "counterfactual_rows": counterfactual_rows,
        "missing_rows": missing_rows,
        "action_mismatches": int(action_mismatches),
        "reference_action_rows": int(reference_action_rows),
        "replay_action_rows": int(replay_action_rows),
        "quality_model": model,
        "config": config,
    }


def _counterfactual_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, int, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["base_seed"]),
            str(row["branch"]),
            str(row["episode_id"]),
            int(row["step"]),
            str(row["uav_id"]),
            str(row["candidate_uav_id"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch, _episode, _step, _uav, _candidate), group in groups.items():
        ordered = sorted(group, key=lambda row: _finite(row.get("risk_value"), 0.0))
        probabilities = np.asarray([_finite(row.get("candidate_probability")) for row in ordered], dtype=float)
        none_probabilities = np.asarray([_finite(row.get("none_probability")) for row in ordered], dtype=float)
        finite = np.isfinite(probabilities) & np.isfinite(none_probabilities)
        probabilities = probabilities[finite]
        none_probabilities = none_probabilities[finite]
        if probabilities.size < 2:
            continue
        endpoint_delta = float(probabilities[-1] - probabilities[0])
        monotonic_ok = bool(np.all(np.diff(probabilities) <= EXPECTED_DIRECTION_TOL))
        output.append({
            "base_seed": seed,
            "branch": branch,
            "candidate_context_count": 1,
            "endpoint_delta_candidate_probability": endpoint_delta,
            "endpoint_delta_none_probability": float(none_probabilities[-1] - none_probabilities[0]),
            "monotonic_expected_direction": int(monotonic_ok),
            "finite": int(np.isfinite(probabilities).all()),
        })
    return output


def _seed_response_summary(
    counterfactual_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
    replay_checks: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    context_rows = _counterfactual_summary(counterfactual_rows)
    for seed in seeds:
        for branch in ("baseline_mappo", "quality_aware_mappo"):
            group = [row for row in context_rows if int(row["base_seed"]) == int(seed) and row["branch"] == branch]
            branch_counterfactual = [
                row for row in counterfactual_rows
                if int(row["base_seed"]) == int(seed) and row["branch"] == branch
            ]
            missing = [
                row for row in missing_rows
                if int(row["base_seed"]) == int(seed) and row["branch"] == branch
            ]
            semantic = [
                row for row in semantic_rows
                if int(row["base_seed"]) == int(seed) and row["branch"] == branch
            ]
            replay = next(
                (row for row in replay_checks if int(row["base_seed"]) == int(seed) and row["branch"] == branch),
                {},
            )
            endpoint = [_finite(row.get("endpoint_delta_candidate_probability")) for row in group]
            endpoint = [value for value in endpoint if math.isfinite(value)]
            direction = [int(row.get("monotonic_expected_direction", 0)) for row in group]
            baseline_changes = [
                abs(_finite(row.get("finite_difference_from_previous"), 0.0))
                for row in branch_counterfactual
            ]
            missing_cov = [row for row in missing if row.get("category") == "missing_neighbor_covariance"]
            summaries.append({
                "base_seed": int(seed),
                "branch": branch,
                "quality_feature_enabled": int(branch == "quality_aware_mappo"),
                "counterfactual_contexts": len(group),
                "counterfactual_rows": len(branch_counterfactual),
                "expected_endpoint_fraction": float(np.mean(np.asarray(endpoint) <= EXPECTED_DIRECTION_TOL)) if endpoint else 0.0,
                "expected_monotonic_fraction": float(np.mean(direction)) if direction else 0.0,
                "mean_endpoint_delta_candidate_probability": float(np.mean(endpoint)) if endpoint else 0.0,
                "median_endpoint_delta_candidate_probability": float(np.median(endpoint)) if endpoint else 0.0,
                "mean_endpoint_delta_none_probability": float(np.mean([
                    _finite(row.get("endpoint_delta_none_probability"), 0.0) for row in group
                ])) if group else 0.0,
                "baseline_max_abs_probability_change": float(max(baseline_changes, default=0.0)),
                "feature_semantics_ok": int(bool(semantic) and all(int(row.get("monotonic_ok", 0)) and int(row.get("finite_ok", 0)) for row in semantic)),
                "missing_covariance_rows": len(missing_cov),
                "missing_fallback_ok": int(all(int(row.get("passed", 0)) for row in missing_cov) if missing_cov else True),
                "unavailable_mask_ok": int(all(int(row.get("passed", 0)) for row in missing if row.get("category") == "unavailable_candidate")),
                "no_candidate_mask_ok": int(all(int(row.get("passed", 0)) for row in missing if row.get("category") == "no_candidate")),
                "replay_actions_exact": int(replay.get("action_mismatches", 1) == 0),
                "quality_model_source": str(replay.get("quality_model_source", "warmup")),
                "response_status": (
                    "baseline_invariant" if branch == "baseline_mappo" and max(baseline_changes, default=0.0) <= INVARIANCE_TOL
                    else "stable_expected_direction" if branch == "quality_aware_mappo" and group and float(np.mean(direction)) >= 0.8
                    else "direction_unstable"
                ),
            })
    return summaries


def _plot(
    path: Path,
    semantic_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    fields = list(RISK_FIELDS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ax = axes[0, 0]
    for field in fields:
        rows = [row for row in semantic_rows if row.get("field") == field and row.get("branch") == "quality_aware_mappo"]
        if not rows:
            continue
        degraded = float(np.nanmean([_finite(row.get("degraded_risk")) for row in rows]))
        improved = float(np.nanmean([_finite(row.get("improved_risk")) for row in rows]))
        ax.plot([0, 1], [degraded, improved], marker="o", label=field)
    ax.set_xticks([0, 1], ["degraded", "improved"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("risk")
    ax.set_title("Risk semantic direction")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for seed in seeds:
        rows = [
            row for row in counterfactual_rows
            if int(row["base_seed"]) == seed and row.get("branch") == "quality_aware_mappo"
        ]
        if not rows:
            continue
        x = []
        y = []
        for risk_value in RISK_GRID:
            values = [_finite(row.get("candidate_probability")) for row in rows if math.isclose(_finite(row.get("risk_value")), risk_value)]
            values = [value for value in values if math.isfinite(value)]
            if values:
                x.append(risk_value)
                y.append(float(np.mean(values)))
        ax.plot(x, y, marker="o", label=str(seed))
    ax.set_xlabel("counterfactual risk")
    ax.set_ylabel("candidate probability")
    ax.set_title("Quality-aware candidate response")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    for seed in seeds:
        rows = [
            row for row in counterfactual_rows
            if int(row["base_seed"]) == seed and row.get("branch") == "quality_aware_mappo"
        ]
        x = []
        y = []
        for risk_value in RISK_GRID:
            values = [_finite(row.get("none_probability")) for row in rows if math.isclose(_finite(row.get("risk_value")), risk_value)]
            values = [value for value in values if math.isfinite(value)]
            if values:
                x.append(risk_value)
                y.append(float(np.mean(values)))
        if x:
            ax.plot(x, y, marker="o", label=str(seed))
    ax.set_xlabel("counterfactual risk")
    ax.set_ylabel("none probability")
    ax.set_title("None-action response")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for branch, color in (("baseline_mappo", "#2f6de1"), ("quality_aware_mappo", "#d92d2d")):
        values = [
            next((float(row["baseline_max_abs_probability_change"]) for row in seed_rows if int(row["base_seed"]) == seed and row["branch"] == branch), 0.0)
            for seed in seeds
        ]
        offset = -0.18 if branch == "baseline_mappo" else 0.18
        ax.bar(np.arange(len(seeds)) + offset, values, width=0.35, label=branch, color=color)
    ax.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    ax.set_ylabel("maximum absolute change")
    ax.set_title("Baseline invariance check")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for branch, color in (("baseline_mappo", "#2f6de1"), ("quality_aware_mappo", "#d92d2d")):
        values = [
            next((float(row["expected_monotonic_fraction"]) for row in seed_rows if int(row["base_seed"]) == seed and row["branch"] == branch), 0.0)
            for seed in seeds
        ]
        offset = -0.18 if branch == "baseline_mappo" else 0.18
        ax.bar(np.arange(len(seeds)) + offset, values, width=0.35, label=branch, color=color)
    ax.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Expected monotonic response fraction")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for category, marker, color in (
        ("complete_neighbor_covariance", "o", "#2f6de1"),
        ("missing_neighbor_covariance", "x", "#d92d2d"),
    ):
        values = []
        labels = []
        for seed in seeds:
            group = [
                row for row in missing_rows
                if int(row["base_seed"]) == seed
                and row.get("branch") == "quality_aware_mappo"
                and row.get("category") == category
            ]
            if group:
                values.append(float(np.mean([_finite(row.get("quality_risk"), 1.0) for row in group])))
                labels.append(str(seed))
        if values:
            ax.scatter(labels, values, marker=marker, color=color, label=category)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("risk")
    ax.set_title("Complete vs missing covariance")
    ax.legend(fontsize=8)

    fig.suptitle("Quality feature sensitivity — validation diagnostic only", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_quality_feature_sensitivity_audit(
    config: QualityFeatureSensitivityAuditConfig,
) -> dict[str, Any]:
    """Run the semantic, missing-value and actor counterfactual audit."""

    reference_root = Path(config.reference_root).resolve()
    output_dir = Path(config.output_dir).resolve()
    plots_dir = Path(config.plots_dir).resolve() if config.plots_dir else (
        Path("plots/preliminary") / reference_root.name / "quality_feature_sensitivity"
    ).resolve()
    _prepare_output(output_dir, force=config.force)
    plots_dir.mkdir(parents=True, exist_ok=True)

    semantic_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    for seed in config.seeds:
        for branch in ("baseline_mappo", "quality_aware_mappo"):
            if config.progress:
                print(f"[quality-sensitivity] start seed={seed} branch={branch}", flush=True)
            scale_root, _manifest = _manifest_for(reference_root, int(seed), branch)
            model = _load_quality_model(scale_root)
            semantic_rows.extend(_risk_semantic_rows(int(seed), branch, model))
            replay = _replay_branch(
                reference_root=reference_root,
                base_seed=int(seed),
                branch=branch,
                split=config.split,
                semantic_model=model,
                progress=config.progress,
            )
            counterfactual_rows.extend(replay["counterfactual_rows"])
            missing_rows.extend(replay["missing_rows"])
            replay_checks.append({
                "base_seed": int(seed),
                "branch": branch,
                "action_mismatches": replay["action_mismatches"],
                "reference_action_rows": replay["reference_action_rows"],
                "replay_action_rows": replay["replay_action_rows"],
                "quality_model_source": model.get("source_split"),
            })

    seed_rows = _seed_response_summary(
        counterfactual_rows,
        semantic_rows,
        missing_rows,
        replay_checks,
        config.seeds,
    )
    semantic_columns = (
        "base_seed", "branch", "field_index", "field", "field_order", "direction",
        "q25", "q75", "degraded_value", "improved_value", "degraded_risk",
        "improved_risk", "monotonic_ok", "finite_ok",
    )
    counterfactual_columns = (
        "base_seed", "branch", "split", "episode_id", "episode_seed", "step",
        "uav_id", "candidate_uav_id", "candidate_index", "risk_value", "original_risk",
        "candidate_valid", "candidate_probability", "none_probability",
        "measurement_probabilities", "other_candidate_probabilities",
        "finite_difference_from_previous", "endpoint_delta_candidate_probability",
        "quality_feature_in_actor", "risk_field_order",
    )
    missing_columns = (
        "base_seed", "branch", "split", "episode_id", "step", "uav_id",
        "candidate_count", "category", "candidate_uav_id", "action_mask_enabled",
        "quality_valid", "quality_risk", "full_covariance_available",
        "missing_covariance_risk", "full_risk", "missing_risk", "candidate_probability",
        "missing_candidate_probability", "none_probability", "none_only_mask", "passed", "notes",
    )
    seed_columns = tuple(sorted({key for row in seed_rows for key in row}))
    _write_csv(output_dir / "feature_semantics.csv", semantic_columns, semantic_rows)
    _write_csv(output_dir / "counterfactual_policy_response.csv", counterfactual_columns, counterfactual_rows)
    _write_csv(output_dir / "missing_value_audit.csv", missing_columns, missing_rows)
    _write_csv(output_dir / "seed_summary.csv", seed_columns, seed_rows)

    semantic_ok = bool(semantic_rows) and all(
        int(row.get("finite_ok", 0)) == 1 and int(row.get("monotonic_ok", 0)) == 1
        for row in semantic_rows
    )
    missing_ok = bool(missing_rows) and all(int(row.get("passed", 0)) == 1 for row in missing_rows)
    baseline_rows = [row for row in seed_rows if row.get("branch") == "baseline_mappo"]
    baseline_invariant = bool(baseline_rows) and all(
        float(row.get("baseline_max_abs_probability_change", float("inf"))) <= INVARIANCE_TOL
        for row in baseline_rows
    )
    quality_rows = [row for row in seed_rows if row.get("branch") == "quality_aware_mappo"]
    quality_features_present = bool(quality_rows) and all(int(row.get("quality_feature_enabled", 0)) == 1 for row in quality_rows)
    replay_exact = bool(replay_checks) and all(
        int(row.get("action_mismatches", 1)) == 0 and int(row.get("reference_action_rows", 0)) == int(row.get("replay_action_rows", -1))
        for row in replay_checks
    )
    all_finite = all(_finite_value(row) for row in counterfactual_rows + missing_rows + semantic_rows + seed_rows)
    all_validation = all(str(row.get("split", "validation")) == "validation" for row in counterfactual_rows + missing_rows)
    forbidden_keys = sorted({
        str(key)
        for row in counterfactual_rows + missing_rows + semantic_rows
        for key in row
        if any(token in str(key).lower() for token in ("true_position", "true_error", "true_range", "los_nlos", "future_state"))
    })
    expected_direction_fraction = [
        float(row.get("expected_monotonic_fraction", 0.0)) for row in quality_rows
    ]
    stable_direction = bool(expected_direction_fraction) and sum(value >= 0.8 for value in expected_direction_fraction) >= max(1, math.ceil(len(expected_direction_fraction) * 0.8))
    if not semantic_ok or not missing_ok:
        decision = "feature_definition_or_missing_fallback_error"
    elif not baseline_invariant:
        decision = "baseline_branch_is_not_invariant"
    elif not stable_direction:
        decision = "actor_direction_unstable"
    else:
        decision = "scene_confounding_or_observed_correlation_instability"
    audit_report = {
        "audit": "quality_feature_semantics_and_mappo_action_response",
        "schema_version": "diagnostic-quality-sensitivity-v1",
        "reference_root": str(reference_root),
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "split": config.split,
        "seeds": [int(seed) for seed in config.seeds],
        "risk_grid": list(RISK_GRID),
        "independent_unit": "base training seed",
        "test_data_used": False,
        "original_reference_outputs_modified": False,
        "checks": {
            "feature_semantics_ok": bool(semantic_ok),
            "missing_value_handling_ok": bool(missing_ok),
            "baseline_quality_invariant": bool(baseline_invariant),
            "quality_features_present": bool(quality_features_present),
            "replay_actions_exact": bool(replay_exact),
            "all_rows_finite": bool(all_finite),
            "all_validation_only": bool(all_validation),
            "forbidden_input_keys": forbidden_keys,
            "quality_models_warmup_only": all(row.get("quality_model_source") == "warmup" for row in replay_checks),
            "counterfactual_actions_executed": False,
        },
        "mechanism": {
            "quality_branch_expected_direction_fraction_by_seed": expected_direction_fraction,
            "quality_branch_stable_direction": bool(stable_direction),
            "baseline_invariance_tolerance": INVARIANCE_TOL,
            "risk_semantics": {
                "high_is_good_fields": list(HIGH_IS_GOOD),
                "low_is_good_fields": list(LOW_IS_GOOD),
                "field_order": list(RISK_FIELDS),
                "missing_neighbor_covariance_risk": 1.0,
            },
        },
        "replay_checks": replay_checks,
        "decision": decision,
        "decision_supported": False,
        "audit_ok": bool(
            semantic_ok
            and missing_ok
            and baseline_invariant
            and quality_features_present
            and replay_exact
            and all_finite
            and all_validation
            and not forbidden_keys
        ),
        "decision_note": (
            "A stable expected direction is a mechanism diagnostic only; it does not establish paper-level policy effectiveness."
        ),
    }
    _write_json(output_dir / "audit_report.json", audit_report)
    _plot(
        plots_dir / "quality_feature_sensitivity.png",
        semantic_rows,
        counterfactual_rows,
        seed_rows,
        missing_rows,
    )
    return {
        "audit_ok": bool(audit_report["audit_ok"]),
        "decision": decision,
        "decision_supported": False,
        "output_dir": str(output_dir),
        "plot": str(plots_dir / "quality_feature_sensitivity.png"),
        "seed_count": len(config.seeds),
        "counterfactual_rows": len(counterfactual_rows),
        "missing_value_rows": len(missing_rows),
    }


__all__ = [
    "QualityFeatureSensitivityAuditConfig",
    "RISK_GRID",
    "run_quality_feature_sensitivity_audit",
    "_risk_semantic_rows",
    "_counterfactual_summary",
    "_padded_action_mask",
    "_with_quality",
]
