"""Read-only audit of expected versus sampled measurement rates.

The rate-controlled experiment constrains the differentiable probability of a
non-``none`` measurement, while the environment records a discrete sampled
action.  This module audits that boundary without replaying an environment or
changing any stored result.  Training actions are used only to recompute the
per-seed targets; validation actions and actor-visible probability logs are
used for the primary diagnosis.  The test split is deliberately never
enumerated or opened.

The independent unit for interpretation is the base training seed.  Episode
and UAV rows are retained for mechanism diagnostics, but are not treated as
independent replicates in the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .measurement_rate_constraint_audit import (
    BRANCHES,
    DEFAULT_SEEDS,
    RATE_BRANCH,
    _bootstrap_mean_ci,
    _episode_dirs,
    _fit_targets,
    _float,
    _int,
    _is_selected,
    _logged_target_status,
    _manifest_config,
    _quality_rows,
    _read_csv,
    _scale_root,
    _training_constraint_rows,
    _uav_ids,
    _write_csv,
    _write_json,
)


PLOT_FILE = "measurement_rate_sampling_audit.png"
MANAGED_FILES = (
    "expected_vs_sampled.csv",
    "target_recomputation.csv",
    "mask_budget_audit.csv",
    "training_constraint_audit.csv",
    "seed_summary.csv",
    "audit_report.json",
)


@dataclass(frozen=True)
class MeasurementRateSamplingAuditConfig:
    """Configuration for the validation-only sampling audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/"
        "rate_sampling_audit"
    )
    plots_dir: Path | None = None
    source_logs_dir: Path | None = None
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    uav_count: int = 3
    split: str = "validation"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("measurement-rate sampling audit is validation-only")
        if self.uav_count < 2:
            raise ValueError("uav_count must be at least 2")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: MeasurementRateSamplingAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[measurement-rate-sampling-audit] {message}", flush=True)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _prepare_output(path: Path, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"audit output is not empty: {path}; use --force")
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for name in MANAGED_FILES:
            target = path / name
            if target.exists():
                target.unlink()


def _parse_measurement_mask(value: Any, uav_count: int) -> tuple[list[bool], bool]:
    """Return the non-``none`` mask and whether the field parsed."""

    if isinstance(value, Mapping):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return [False] * uav_count, False
    if not isinstance(payload, Mapping):
        return [False] * uav_count, False
    values = payload.get("measurement")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [False] * uav_count, False
    return [bool(item) for item in list(values)[:uav_count]] + [False] * max(
        0, uav_count - len(list(values)[:uav_count])
    ), True


def _selected_index(value: Any, uav_count: int) -> int | None:
    text = str(value if value is not None else "none").strip().lower()
    if not text or text == "none":
        return None
    prefix = "uav_"
    if not text.startswith(prefix):
        return None
    try:
        index = int(text[len(prefix) :])
    except ValueError:
        return None
    return index if 0 <= index < uav_count else None


def _raw_measurement_id(value: Any) -> tuple[str, bool]:
    """Read the actor action before environment-side transition handling."""

    if isinstance(value, Mapping):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return "none", False
    if not isinstance(payload, Mapping):
        return "none", False
    measurement = str(payload.get("measurement_uav_id", "none")).strip().lower() or "none"
    return measurement, True


def _group_action_rows(scale_root: Path, split: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Read only one requested split and group action rows by episode/UAV."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for episode_root in _episode_dirs(scale_root, split):
        path = episode_root / "actions.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing actions.csv: {path}")
        for ordinal, row in enumerate(_read_csv(path)):
            item: dict[str, Any] = dict(row)
            item.setdefault("episode_id", episode_root.name)
            item["_episode_root"] = episode_root
            item["_row_ordinal"] = ordinal
            if str(item.get("split", split)) != split:
                raise ValueError(f"action split mismatch in {path}: {item.get('split')}")
            key = (str(item.get("episode_id", episode_root.name)), str(item.get("uav_id", "")))
            groups.setdefault(key, []).append(item)
    for rows in groups.values():
        rows.sort(key=lambda row: (_float(row.get("timestamp"), 0.0), int(row.get("_row_ordinal", 0))))
    return groups


def _group_quality_rows(scale_root: Path, split: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group actor-visible probability rows without opening another split."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ordinal, row in enumerate(_quality_rows(scale_root, split)):
        item = dict(row)
        item["_row_ordinal"] = ordinal
        key = (str(item.get("episode_id", "")), str(item.get("uav_id", "")))
        groups.setdefault(key, []).append(item)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                _int(row.get("step"), 0),
                _float(row.get("timestamp"), 0.0),
                int(row.get("_row_ordinal", 0)),
            )
        )
    return groups


def _group_budget_rows(scale_root: Path, split: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Read optional budget rows for one split, aligned by decision order."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for episode_root in _episode_dirs(scale_root, split):
        path = episode_root / "measurement_budget.csv"
        if not path.exists():
            continue
        for ordinal, row in enumerate(_read_csv(path)):
            item = dict(row)
            item["_row_ordinal"] = ordinal
            key = (str(item.get("episode_id", episode_root.name)), str(item.get("uav_id", "")))
            groups.setdefault(key, []).append(item)
    for rows in groups.values():
        rows.sort(key=lambda row: (_int(row.get("step"), 0), int(row.get("_row_ordinal", 0))))
    return groups


def _fit_raw_targets(
    reference_root: Path,
    seeds: Sequence[int],
    uav_count: int,
) -> tuple[dict[tuple[int, str], float], list[dict[str, Any]], int]:
    """Fit actor-action targets from ``action_raw`` in baseline train only.

    The existing constraint target is retained separately because it is based
    on the applied ``measurement_uav_id`` column.  This second target makes
    the actor-sampling audit explicit about the small environment-side gap.
    """

    targets: dict[tuple[int, str], float] = {}
    output: list[dict[str, Any]] = []
    missing_raw = 0
    ids = _uav_ids(uav_count)
    for seed in seeds:
        groups = _group_action_rows(_scale_root(reference_root, int(seed), "baseline_mappo"), "train")
        for uav_id in ids:
            rows = [
                item
                for (episode_id, row_uav), group in groups.items()
                if row_uav == uav_id
                for item in group
            ]
            rows.sort(
                key=lambda row: (
                    str(row.get("episode_id", "")),
                    _float(row.get("timestamp"), 0.0),
                    int(row.get("_row_ordinal", 0)),
                )
            )
            selected = 0
            valid = 0
            for row in rows:
                raw_id, ok = _raw_measurement_id(row.get("action_raw", ""))
                if not ok:
                    missing_raw += 1
                    continue
                valid += 1
                selected += int(raw_id != "none")
            target = float(selected / valid) if valid else float("nan")
            targets[(int(seed), uav_id)] = target
            output.append(
                {
                    "base_seed": int(seed),
                    "uav_id": uav_id,
                    "raw_baseline_train_selected_measurements": int(selected),
                    "raw_baseline_train_decision_windows": int(valid),
                    "raw_baseline_train_measurement_rate": target,
                    "raw_target_source_split": "train",
                    "raw_target_source_branch": "baseline_mappo",
                }
            )
    return targets, output, missing_raw


def _probability_from_quality(row: Mapping[str, Any] | None) -> tuple[float, float, bool]:
    """Return (expected non-none probability, none probability, valid)."""

    if row is None:
        return float("nan"), float("nan"), False
    none_probability = _float(row.get("none_probability"), float("nan"))
    expected = 1.0 - none_probability if _finite_number(none_probability) else float("nan")
    valid = _finite_number(expected) and 0.0 <= expected <= 1.0
    return (float(expected) if valid else float("nan"), float(none_probability) if valid else float("nan"), valid)


def _episode_rows(
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    targets: Mapping[tuple[int, str], float],
    raw_targets: Mapping[tuple[int, str], float],
    uav_count: int,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build one diagnostic row per episode/UAV without replaying the environment."""

    expected_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "quality_alignment_mismatch_count": 0,
        "quality_probability_invalid_count": 0,
        "quality_probability_missing_count": 0,
        "action_mask_unparseable_count": 0,
        "selected_action_mask_invalid_count": 0,
        "logged_invalid_measurement_action_count": 0,
        "budget_over_limit_count": 0,
        "budget_selected_when_exhausted_count": 0,
        "raw_action_missing_count": 0,
        "reference_roots_complete": True,
        "validation_episode_counts": {},
        "action_selection_mode": (
            "stochastic_sample" if split == "train" else "deterministic_argmax"
        ),
    }
    ids = _uav_ids(uav_count)

    for seed in seeds:
        for branch in branches:
            scale_root = _scale_root(reference_root, int(seed), branch)
            if not scale_root.is_dir():
                meta["reference_roots_complete"] = False
                raise FileNotFoundError(f"missing seed/branch scale root: {scale_root}")
            branch_config = _manifest_config(scale_root)
            action_groups = _group_action_rows(scale_root, split)
            quality_groups = _group_quality_rows(scale_root, split)
            budget_groups = _group_budget_rows(scale_root, split)
            episode_ids = sorted({episode_id for episode_id, _ in action_groups})
            meta["validation_episode_counts"][f"{seed}/{branch}"] = len(episode_ids)

            for episode_id in episode_ids:
                for uav_id in ids:
                    rows = action_groups.get((episode_id, uav_id), [])
                    if not rows:
                        raise ValueError(
                            f"missing action rows for seed={seed}, branch={branch}, "
                            f"episode={episode_id}, uav={uav_id}"
                        )
                    quality_rows = quality_groups.get((episode_id, uav_id), [])
                    budget_rows = budget_groups.get((episode_id, uav_id), [])
                    observations: list[dict[str, Any]] = []
                    for index, action in enumerate(rows):
                        quality = quality_rows[index] if index < len(quality_rows) else None
                        budget = budget_rows[index] if index < len(budget_rows) else None
                        expected, none_probability, probability_valid = _probability_from_quality(
                            quality
                        )
                        if quality is None:
                            meta["quality_probability_missing_count"] += 1
                        elif not probability_valid:
                            meta["quality_probability_invalid_count"] += 1

                        raw_id, raw_parseable = _raw_measurement_id(
                            action.get("action_raw", "")
                        )
                        applied_id = (
                            str(action.get("measurement_uav_id", "none")).strip().lower()
                            or "none"
                        )
                        if not raw_parseable:
                            meta["raw_action_missing_count"] += 1
                        quality_id = (
                            str(quality.get("measurement_uav_id", "none")).strip().lower()
                            if quality is not None
                            else "none"
                        )
                        if quality is not None and raw_parseable and quality_id != raw_id:
                            meta["quality_alignment_mismatch_count"] += 1

                        mask, mask_parseable = _parse_measurement_mask(
                            action.get("action_mask_json", ""), uav_count
                        )
                        selected_raw = bool(raw_parseable and raw_id != "none")
                        selected_applied = bool(applied_id != "none")
                        selected_index = _selected_index(raw_id, uav_count)
                        candidate = bool(any(mask)) if mask_parseable else False
                        selected_mask_invalid = bool(
                            selected_raw
                            and (
                                not mask_parseable
                                or selected_index is None
                                or not mask[selected_index]
                            )
                        )
                        if not mask_parseable:
                            meta["action_mask_unparseable_count"] += 1
                        if selected_mask_invalid:
                            meta["selected_action_mask_invalid_count"] += 1

                        logged_invalid = _int(action.get("invalid_measurement_action"), 0)
                        meta["logged_invalid_measurement_action_count"] += int(
                            logged_invalid != 0
                        )
                        if budget is not None:
                            budget_limit = _int(budget.get("budget_limit"), -1)
                            budget_remaining = _float(
                                budget.get("budget_remaining_before"), float("nan")
                            )
                            budget_used = _int(budget.get("budget_used"), 0)
                            budget_available = int(
                                budget_limit >= 0 and _finite_number(budget_remaining)
                            )
                        else:
                            budget_map = branch_config.get(
                                "measurement_budget_by_uav", {}
                            )
                            if not isinstance(budget_map, Mapping):
                                budget_map = {}
                            budget_limit = _int(budget_map.get(uav_id), -1)
                            ratio = (
                                _float(
                                    quality.get(
                                        "measurement_budget_remaining_ratio"
                                    ),
                                    float("nan"),
                                )
                                if quality is not None
                                else float("nan")
                            )
                            budget_available = int(
                                bool(branch_config.get("measurement_budget_enabled", False))
                                and budget_limit >= 0
                                and _finite_number(ratio)
                            )
                            budget_remaining = (
                                float(max(0.0, min(1.0, ratio)) * budget_limit)
                                if budget_available
                                else float("nan")
                            )
                            budget_used = (
                                int(round(max(0.0, budget_limit - budget_remaining)))
                                if budget_available
                                else 0
                            )
                        budget_exhausted = bool(
                            budget_available
                            and _finite_number(budget_remaining)
                            and budget_remaining <= 1e-12
                        )
                        budget_over_limit = bool(
                            budget_available
                            and budget_limit >= 0
                            and budget_used > budget_limit
                        )
                        if budget_over_limit:
                            meta["budget_over_limit_count"] += 1
                        if budget_exhausted and selected_raw:
                            meta["budget_selected_when_exhausted_count"] += 1

                        transition_invalidated = int(
                            _int(action.get("measurement_transition_invalidated"), 0) != 0
                            or (selected_raw and not selected_applied)
                        )
                        observations.append(
                            {
                                "selected_raw": int(selected_raw),
                                "selected_applied": int(selected_applied),
                                "candidate": int(candidate),
                                "mask_parseable": int(mask_parseable),
                                "selected_mask_invalid": int(selected_mask_invalid),
                                "logged_invalid": int(logged_invalid != 0),
                                "expected": expected,
                                "none_probability": none_probability,
                                "probability_valid": int(probability_valid),
                                "budget_available": int(budget_available),
                                "budget_limit": int(budget_limit),
                                "budget_remaining_before": budget_remaining,
                                "budget_used": int(budget_used),
                                "budget_exhausted": int(budget_exhausted),
                                "budget_over_limit": int(budget_over_limit),
                                "quality_present": int(quality is not None),
                                "raw_parseable": int(raw_parseable),
                                "transition_invalidated": transition_invalidated,
                            }
                        )

                    n_windows = len(observations)
                    eligible = [item for item in observations if item["candidate"] == 1]
                    valid_all = [
                        item for item in observations if item["probability_valid"] == 1
                    ]
                    valid_eligible = [
                        item for item in eligible if item["probability_valid"] == 1
                    ]
                    raw_selected_count = int(
                        sum(item["selected_raw"] for item in observations)
                    )
                    applied_selected_count = int(
                        sum(item["selected_applied"] for item in observations)
                    )
                    raw_selected_eligible = int(
                        sum(item["selected_raw"] for item in eligible)
                    )
                    applied_selected_eligible = int(
                        sum(item["selected_applied"] for item in eligible)
                    )
                    expected_all = (
                        float(np.mean([item["expected"] for item in valid_all]))
                        if valid_all
                        else float("nan")
                    )
                    expected_eligible = (
                        float(np.mean([item["expected"] for item in valid_eligible]))
                        if valid_eligible
                        else float("nan")
                    )
                    raw_rate_all = (
                        float(raw_selected_count / n_windows)
                        if n_windows
                        else float("nan")
                    )
                    applied_rate_all = (
                        float(applied_selected_count / n_windows)
                        if n_windows
                        else float("nan")
                    )
                    raw_rate_eligible = (
                        float(raw_selected_eligible / len(eligible))
                        if eligible
                        else float("nan")
                    )
                    applied_rate_eligible = (
                        float(applied_selected_eligible / len(eligible))
                        if eligible
                        else float("nan")
                    )
                    variance_num_all = float(
                        sum(
                            item["expected"] * (1.0 - item["expected"])
                            for item in valid_all
                        )
                    )
                    variance_num_eligible = float(
                        sum(
                            item["expected"] * (1.0 - item["expected"])
                            for item in valid_eligible
                        )
                    )
                    se_all = (
                        float(math.sqrt(max(variance_num_all, 0.0)) / len(valid_all))
                        if valid_all
                        else float("nan")
                    )
                    se_eligible = (
                        float(
                            math.sqrt(max(variance_num_eligible, 0.0))
                            / len(valid_eligible)
                        )
                        if valid_eligible
                        else float("nan")
                    )
                    raw_target = _float(
                        raw_targets.get((int(seed), uav_id)), float("nan")
                    )
                    constraint_target = _float(
                        targets.get((int(seed), uav_id)), float("nan")
                    )
                    raw_minus_expected = (
                        raw_rate_all - expected_all
                        if _finite_number(expected_all)
                        else float("nan")
                    )
                    expected_minus_raw_target = (
                        expected_all - raw_target
                        if _finite_number(expected_all) and _finite_number(raw_target)
                        else float("nan")
                    )
                    expected_minus_constraint_target = (
                        expected_all - constraint_target
                        if _finite_number(expected_all)
                        and _finite_number(constraint_target)
                        else float("nan")
                    )
                    raw_minus_target = (
                        raw_rate_all - raw_target
                        if _finite_number(raw_target)
                        else float("nan")
                    )
                    applied_minus_target = (
                        applied_rate_all - constraint_target
                        if _finite_number(constraint_target)
                        else float("nan")
                    )
                    blocked = sum(
                        int(
                            item["budget_exhausted"]
                            or (item["mask_parseable"] and not item["candidate"])
                        )
                        for item in observations
                    )
                    gate_row = {
                        "base_seed": int(seed),
                        "branch": branch,
                        "split": split,
                        "episode_id": episode_id,
                        "uav_id": uav_id,
                        "decision_windows": n_windows,
                        "eligible_candidate_windows": len(eligible),
                        "candidate_mask_blocked_windows": sum(
                            int(item["mask_parseable"] and not item["candidate"])
                            for item in observations
                        ),
                        "mask_unparseable_windows": sum(
                            int(not item["mask_parseable"]) for item in observations
                        ),
                        "budget_available": int(
                            any(item["budget_available"] for item in observations)
                        ),
                        "budget_limit": max(
                            (int(item["budget_limit"]) for item in observations),
                            default=-1,
                        ),
                        "budget_exhausted_windows": sum(
                            int(item["budget_exhausted"]) for item in observations
                        ),
                        "none_due_budget_windows": sum(
                            int(item["budget_exhausted"] and not item["selected_raw"])
                            for item in observations
                        ),
                        "raw_selected_measurements": raw_selected_count,
                        "applied_selected_measurements": applied_selected_count,
                        "raw_selected_eligible_measurements": raw_selected_eligible,
                        "applied_selected_eligible_measurements": applied_selected_eligible,
                        "raw_selected_none": n_windows - raw_selected_count,
                        "applied_selected_none": n_windows - applied_selected_count,
                        "selected_action_mask_invalid": sum(
                            int(item["selected_mask_invalid"]) for item in observations
                        ),
                        "logged_invalid_measurement_action": sum(
                            int(item["logged_invalid"]) for item in observations
                        ),
                        "budget_over_limit": sum(
                            int(item["budget_over_limit"]) for item in observations
                        ),
                        "budget_selected_when_exhausted": sum(
                            int(item["budget_exhausted"] and item["selected_raw"])
                            for item in observations
                        ),
                        "mask_or_budget_blocked_windows": blocked,
                        "expected_probability_valid_windows": len(valid_all),
                        "expected_probability_valid_eligible_windows": len(valid_eligible),
                        "expected_probability_missing_or_invalid_windows": n_windows
                        - len(valid_all),
                        "quality_rows": sum(
                            int(item["quality_present"]) for item in observations
                        ),
                        "raw_action_missing": sum(
                            int(not item["raw_parseable"]) for item in observations
                        ),
                        "transition_invalidated_windows": sum(
                            int(item["transition_invalidated"]) for item in observations
                        ),
                        "quality_alignment_mismatch": sum(
                            int(
                                index < len(quality_rows)
                                and _raw_measurement_id(
                                    rows[index].get("action_raw", "")
                                )[0]
                                != str(
                                    quality_rows[index].get("measurement_uav_id", "none")
                                ).strip().lower()
                            )
                            for index in range(n_windows)
                        ),
                        "target_measurement_rate": constraint_target,
                        "sampling_target_measurement_rate": raw_target,
                        "source_actions": str(scale_root / split),
                        "source_probability_log": str(
                            scale_root / "diagnostics" / "measurement_quality.jsonl"
                        ),
                    }
                    expected_rows.append(
                        {
                            **gate_row,
                            "action_selection_mode": (
                                "stochastic_sample"
                                if split == "train"
                                else "deterministic_argmax"
                            ),
                            "raw_sampled_rate_all_windows": raw_rate_all,
                            "raw_sampled_rate_eligible_windows": raw_rate_eligible,
                            "applied_measurement_rate_all_windows": applied_rate_all,
                            "applied_measurement_rate_eligible_windows": applied_rate_eligible,
                            "expected_non_none_rate_all_windows": expected_all,
                            "expected_non_none_rate_eligible_windows": expected_eligible,
                            "raw_sampled_minus_expected_all_windows": raw_minus_expected,
                            "expected_minus_target": expected_minus_raw_target,
                            "expected_minus_constraint_target": expected_minus_constraint_target,
                            "raw_sampled_minus_target": raw_minus_target,
                            "applied_minus_constraint_target": applied_minus_target,
                            "raw_applied_rate_gap": raw_rate_all - applied_rate_all,
                            "conditional_sampling_se_all_windows": se_all,
                            "conditional_sampling_se_eligible_windows": se_eligible,
                            "conditional_sampling_z_all_windows": (
                                raw_minus_expected / se_all
                                if _finite_number(raw_minus_expected)
                                and _finite_number(se_all)
                                and se_all > 1e-12
                                else float("nan")
                            ),
                            "sampling_variance_numerator_all_windows": variance_num_all,
                            "sampling_variance_numerator_eligible_windows": variance_num_eligible,
                            "sampling_reference_applicable": int(split == "train"),
                            "sampling_gap_within_1_96_se": int(
                                split == "train"
                                and _finite_number(raw_minus_expected)
                                and _finite_number(se_all)
                                and abs(raw_minus_expected)
                                <= max(0.05, 1.96 * se_all)
                            ),
                            "quality_feature_enabled": int(
                                branch in {"budgeted_quality_aware_mappo", RATE_BRANCH}
                            ),
                            "measurement_budget_feature_enabled": int(
                                branch in {"budgeted_quality_aware_mappo", RATE_BRANCH}
                            ),
                            "measurement_rate_constraint_enabled": int(branch == RATE_BRANCH),
                        }
                    )
                    gate_rows.append(gate_row)
    return expected_rows, gate_rows, meta


def _seed_summary(
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    uav_count: int,
) -> list[dict[str, Any]]:
    """Aggregate validation rows to one row per seed and branch.

    The conditional sampling reference is computed from the pooled Bernoulli
    variance numerator, not by summing episode-level standard errors.  It is
    deliberately marked as not applicable for validation because validation
    actions are deterministic argmax decisions.
    """

    output: list[dict[str, Any]] = []
    keys = sorted(
        {(int(row["base_seed"]), str(row["branch"])) for row in expected_rows}
    )
    for seed, branch in keys:
        rows = [
            row
            for row in expected_rows
            if int(row["base_seed"]) == seed and str(row["branch"]) == branch
        ]
        decisions = sum(_int(row.get("decision_windows"), 0) for row in rows)
        eligible = sum(_int(row.get("eligible_candidate_windows"), 0) for row in rows)
        raw_selected = sum(_int(row.get("raw_selected_measurements"), 0) for row in rows)
        applied_selected = sum(
            _int(row.get("applied_selected_measurements"), 0) for row in rows
        )
        raw_selected_eligible = sum(
            _int(row.get("raw_selected_eligible_measurements"), 0) for row in rows
        )
        applied_selected_eligible = sum(
            _int(row.get("applied_selected_eligible_measurements"), 0)
            for row in rows
        )
        valid_probability = sum(
            _int(row.get("expected_probability_valid_windows"), 0)
            for row in rows
        )
        valid_probability_eligible = sum(
            _int(row.get("expected_probability_valid_eligible_windows"), 0)
            for row in rows
        )
        expected_sum = sum(
            _float(row.get("expected_non_none_rate_all_windows"), 0.0)
            * _int(row.get("expected_probability_valid_windows"), 0)
            for row in rows
            if _finite_number(row.get("expected_non_none_rate_all_windows"))
        )
        expected_eligible_sum = sum(
            _float(row.get("expected_non_none_rate_eligible_windows"), 0.0)
            * _int(row.get("expected_probability_valid_eligible_windows"), 0)
            for row in rows
            if _finite_number(row.get("expected_non_none_rate_eligible_windows"))
        )
        raw_target_sum = sum(
            _float(row.get("sampling_target_measurement_rate"), 0.0)
            * _int(row.get("decision_windows"), 0)
            for row in rows
            if _finite_number(row.get("sampling_target_measurement_rate"))
        )
        constraint_target_sum = sum(
            _float(row.get("target_measurement_rate"), 0.0)
            * _int(row.get("decision_windows"), 0)
            for row in rows
            if _finite_number(row.get("target_measurement_rate"))
        )
        variance_numerator = sum(
            _float(row.get("sampling_variance_numerator_all_windows"), 0.0)
            for row in rows
        )
        expected_all = (
            float(expected_sum / valid_probability)
            if valid_probability
            else float("nan")
        )
        expected_eligible = (
            float(expected_eligible_sum / valid_probability_eligible)
            if valid_probability_eligible
            else float("nan")
        )
        raw_target = float(raw_target_sum / decisions) if decisions else float("nan")
        constraint_target = (
            float(constraint_target_sum / decisions) if decisions else float("nan")
        )
        raw_rate = float(raw_selected / decisions) if decisions else float("nan")
        applied_rate = (
            float(applied_selected / decisions) if decisions else float("nan")
        )
        raw_rate_eligible = (
            float(raw_selected_eligible / eligible) if eligible else float("nan")
        )
        applied_rate_eligible = (
            float(applied_selected_eligible / eligible) if eligible else float("nan")
        )
        raw_minus_expected = (
            raw_rate - expected_all if _finite_number(expected_all) else float("nan")
        )
        expected_minus_target = (
            expected_all - raw_target
            if _finite_number(expected_all) and _finite_number(raw_target)
            else float("nan")
        )
        expected_minus_constraint = (
            expected_all - constraint_target
            if _finite_number(expected_all) and _finite_number(constraint_target)
            else float("nan")
        )
        raw_minus_target = (
            raw_rate - raw_target
            if _finite_number(raw_target)
            else float("nan")
        )
        applied_minus_constraint = (
            applied_rate - constraint_target
            if _finite_number(constraint_target)
            else float("nan")
        )
        reference_se = (
            float(math.sqrt(max(variance_numerator, 0.0)) / valid_probability)
            if valid_probability
            else float("nan")
        )
        transition_total = sum(
            _int(row.get("transition_invalidated_windows"), 0) for row in rows
        )
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "split": "validation",
                "uav_count": int(uav_count),
                "episode_count": len({str(row.get("episode_id", "")) for row in rows}),
                "action_selection_mode": str(
                    rows[0].get("action_selection_mode", "deterministic_argmax")
                )
                if rows
                else "deterministic_argmax",
                "sampling_reference_applicable": int(
                    any(_int(row.get("sampling_reference_applicable"), 0) for row in rows)
                ),
                "decision_windows": decisions,
                "eligible_candidate_windows": eligible,
                "expected_probability_valid_windows": valid_probability,
                "expected_probability_valid_eligible_windows": valid_probability_eligible,
                "raw_selected_measurements": raw_selected,
                "applied_selected_measurements": applied_selected,
                "raw_selected_eligible_measurements": raw_selected_eligible,
                "applied_selected_eligible_measurements": applied_selected_eligible,
                "selected_measurements": raw_selected,
                "selected_none": decisions - raw_selected,
                "raw_sampled_rate_all_windows": raw_rate,
                "raw_sampled_rate_eligible_windows": raw_rate_eligible,
                "applied_measurement_rate_all_windows": applied_rate,
                "applied_measurement_rate_eligible_windows": applied_rate_eligible,
                "actual_sampled_rate_all_windows": raw_rate,
                "actual_sampled_rate_eligible_windows": raw_rate_eligible,
                "expected_non_none_rate_all_windows": expected_all,
                "expected_non_none_rate_eligible_windows": expected_eligible,
                "sampling_target_measurement_rate": raw_target,
                "target_measurement_rate": constraint_target,
                "raw_sampled_minus_expected": raw_minus_expected,
                "actual_minus_expected": raw_minus_expected,
                "raw_sampled_minus_target": raw_minus_target,
                "actual_minus_target": raw_minus_target,
                "expected_minus_target": expected_minus_target,
                "expected_minus_constraint_target": expected_minus_constraint,
                "applied_minus_constraint_target": applied_minus_constraint,
                "raw_applied_rate_gap": raw_rate - applied_rate,
                "absolute_actual_minus_expected": (
                    abs(raw_minus_expected)
                    if _finite_number(raw_minus_expected)
                    else float("nan")
                ),
                "absolute_raw_sampled_minus_target": (
                    abs(raw_minus_target)
                    if _finite_number(raw_minus_target)
                    else float("nan")
                ),
                "absolute_applied_minus_constraint_target": (
                    abs(applied_minus_constraint)
                    if _finite_number(applied_minus_constraint)
                    else float("nan")
                ),
                "absolute_expected_minus_target": (
                    abs(expected_minus_target)
                    if _finite_number(expected_minus_target)
                    else float("nan")
                ),
                "conditional_sampling_reference_se": reference_se,
                "sampling_gap_within_reference": 0,
                "candidate_mask_blocked_fraction": (
                    sum(
                        _int(row.get("candidate_mask_blocked_windows"), 0)
                        for row in rows
                    )
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "mask_unparseable_fraction": (
                    sum(_int(row.get("mask_unparseable_windows"), 0) for row in rows)
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "budget_exhausted_fraction": (
                    sum(_int(row.get("budget_exhausted_windows"), 0) for row in rows)
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "none_due_budget_fraction": (
                    sum(_int(row.get("none_due_budget_windows"), 0) for row in rows)
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "mask_or_budget_blocked_fraction": (
                    sum(
                        _int(row.get("mask_or_budget_blocked_windows"), 0)
                        for row in rows
                    )
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "transition_invalidated_windows": transition_total,
                "transition_invalidated_fraction": (
                    transition_total / decisions if decisions else float("nan")
                ),
                "selected_action_mask_invalid": sum(
                    _int(row.get("selected_action_mask_invalid"), 0) for row in rows
                ),
                "logged_invalid_measurement_action": sum(
                    _int(row.get("logged_invalid_measurement_action"), 0)
                    for row in rows
                ),
                "budget_over_limit": sum(
                    _int(row.get("budget_over_limit"), 0) for row in rows
                ),
                "budget_selected_when_exhausted": sum(
                    _int(row.get("budget_selected_when_exhausted"), 0)
                    for row in rows
                ),
                "quality_alignment_mismatch": sum(
                    _int(row.get("quality_alignment_mismatch"), 0) for row in rows
                ),
                "raw_action_missing": sum(
                    _int(row.get("raw_action_missing"), 0) for row in rows
                ),
                "expected_probability_missing_or_invalid": sum(
                    _int(
                        row.get(
                            "expected_probability_missing_or_invalid_windows", 0
                        ),
                        0,
                    )
                    for row in rows
                ),
                "quality_feature_enabled": int(
                    branch in {"budgeted_quality_aware_mappo", RATE_BRANCH}
                ),
                "measurement_budget_feature_enabled": int(
                    branch in {"budgeted_quality_aware_mappo", RATE_BRANCH}
                ),
                "measurement_rate_constraint_enabled": int(branch == RATE_BRANCH),
            }
        )
    return output


def _seed_summary(
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    uav_count: int,
) -> list[dict[str, Any]]:
    """Aggregate validation rows with a pooled probability variance."""

    output: list[dict[str, Any]] = []
    keys = sorted(
        {(int(row["base_seed"]), str(row["branch"])) for row in expected_rows}
    )
    for seed, branch in keys:
        rows = [
            row
            for row in expected_rows
            if int(row["base_seed"]) == seed and str(row["branch"]) == branch
        ]
        decisions = sum(_int(row.get("decision_windows"), 0) for row in rows)
        eligible = sum(_int(row.get("eligible_candidate_windows"), 0) for row in rows)
        raw_selected = sum(_int(row.get("raw_selected_measurements"), 0) for row in rows)
        applied_selected = sum(
            _int(row.get("applied_selected_measurements"), 0) for row in rows
        )
        raw_selected_eligible = sum(
            _int(row.get("raw_selected_eligible_measurements"), 0)
            for row in rows
        )
        applied_selected_eligible = sum(
            _int(row.get("applied_selected_eligible_measurements"), 0)
            for row in rows
        )
        valid_probability = sum(
            _int(row.get("expected_probability_valid_windows"), 0)
            for row in rows
        )
        valid_probability_eligible = sum(
            _int(row.get("expected_probability_valid_eligible_windows"), 0)
            for row in rows
        )
        expected_sum = sum(
            _float(row.get("expected_non_none_rate_all_windows"), 0.0)
            * _int(row.get("expected_probability_valid_windows"), 0)
            for row in rows
            if _finite_number(row.get("expected_non_none_rate_all_windows"))
        )
        expected_eligible_sum = sum(
            _float(row.get("expected_non_none_rate_eligible_windows"), 0.0)
            * _int(row.get("expected_probability_valid_eligible_windows"), 0)
            for row in rows
            if _finite_number(row.get("expected_non_none_rate_eligible_windows"))
        )
        raw_target_sum = sum(
            _float(row.get("sampling_target_measurement_rate"), 0.0)
            * _int(row.get("decision_windows"), 0)
            for row in rows
            if _finite_number(row.get("sampling_target_measurement_rate"))
        )
        constraint_target_sum = sum(
            _float(row.get("target_measurement_rate"), 0.0)
            * _int(row.get("decision_windows"), 0)
            for row in rows
            if _finite_number(row.get("target_measurement_rate"))
        )
        variance_numerator = sum(
            _float(row.get("sampling_variance_numerator_all_windows"), 0.0)
            for row in rows
        )
        expected_all = (
            float(expected_sum / valid_probability)
            if valid_probability
            else float("nan")
        )
        expected_eligible = (
            float(expected_eligible_sum / valid_probability_eligible)
            if valid_probability_eligible
            else float("nan")
        )
        raw_target = float(raw_target_sum / decisions) if decisions else float("nan")
        constraint_target = (
            float(constraint_target_sum / decisions) if decisions else float("nan")
        )
        raw_rate = float(raw_selected / decisions) if decisions else float("nan")
        applied_rate = (
            float(applied_selected / decisions) if decisions else float("nan")
        )
        raw_rate_eligible = (
            float(raw_selected_eligible / eligible) if eligible else float("nan")
        )
        applied_rate_eligible = (
            float(applied_selected_eligible / eligible) if eligible else float("nan")
        )
        raw_minus_expected = (
            raw_rate - expected_all if _finite_number(expected_all) else float("nan")
        )
        expected_minus_target = (
            expected_all - raw_target
            if _finite_number(expected_all) and _finite_number(raw_target)
            else float("nan")
        )
        expected_minus_constraint = (
            expected_all - constraint_target
            if _finite_number(expected_all) and _finite_number(constraint_target)
            else float("nan")
        )
        raw_minus_target = (
            raw_rate - raw_target
            if _finite_number(raw_target)
            else float("nan")
        )
        applied_minus_constraint = (
            applied_rate - constraint_target
            if _finite_number(constraint_target)
            else float("nan")
        )
        reference_se = (
            float(math.sqrt(max(variance_numerator, 0.0)) / valid_probability)
            if valid_probability
            else float("nan")
        )
        transition_total = sum(
            _int(row.get("transition_invalidated_windows"), 0)
            for row in rows
        )
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "split": "validation",
                "uav_count": int(uav_count),
                "episode_count": len({str(row.get("episode_id", "")) for row in rows}),
                "action_selection_mode": (
                    str(rows[0].get("action_selection_mode", "deterministic_argmax"))
                    if rows
                    else "deterministic_argmax"
                ),
                "sampling_reference_applicable": int(
                    any(
                        _int(row.get("sampling_reference_applicable"), 0)
                        for row in rows
                    )
                ),
                "decision_windows": decisions,
                "eligible_candidate_windows": eligible,
                "expected_probability_valid_windows": valid_probability,
                "expected_probability_valid_eligible_windows": valid_probability_eligible,
                "raw_selected_measurements": raw_selected,
                "applied_selected_measurements": applied_selected,
                "raw_selected_eligible_measurements": raw_selected_eligible,
                "applied_selected_eligible_measurements": applied_selected_eligible,
                "selected_measurements": raw_selected,
                "selected_none": decisions - raw_selected,
                "raw_sampled_rate_all_windows": raw_rate,
                "raw_sampled_rate_eligible_windows": raw_rate_eligible,
                "applied_measurement_rate_all_windows": applied_rate,
                "applied_measurement_rate_eligible_windows": applied_rate_eligible,
                "actual_sampled_rate_all_windows": raw_rate,
                "actual_sampled_rate_eligible_windows": raw_rate_eligible,
                "expected_non_none_rate_all_windows": expected_all,
                "expected_non_none_rate_eligible_windows": expected_eligible,
                "sampling_target_measurement_rate": raw_target,
                "target_measurement_rate": constraint_target,
                "raw_sampled_minus_expected": raw_minus_expected,
                "actual_minus_expected": raw_minus_expected,
                "raw_sampled_minus_target": raw_minus_target,
                "actual_minus_target": raw_minus_target,
                "expected_minus_target": expected_minus_target,
                "expected_minus_constraint_target": expected_minus_constraint,
                "applied_minus_constraint_target": applied_minus_constraint,
                "raw_applied_rate_gap": raw_rate - applied_rate,
                "absolute_actual_minus_expected": (
                    abs(raw_minus_expected)
                    if _finite_number(raw_minus_expected)
                    else float("nan")
                ),
                "absolute_raw_sampled_minus_target": (
                    abs(raw_minus_target)
                    if _finite_number(raw_minus_target)
                    else float("nan")
                ),
                "absolute_applied_minus_constraint_target": (
                    abs(applied_minus_constraint)
                    if _finite_number(applied_minus_constraint)
                    else float("nan")
                ),
                "absolute_expected_minus_target": (
                    abs(expected_minus_target)
                    if _finite_number(expected_minus_target)
                    else float("nan")
                ),
                "conditional_sampling_reference_se": reference_se,
                "sampling_gap_within_reference": 0,
                "candidate_mask_blocked_fraction": (
                    sum(
                        _int(row.get("candidate_mask_blocked_windows"), 0)
                        for row in rows
                    )
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "mask_unparseable_fraction": (
                    sum(_int(row.get("mask_unparseable_windows"), 0) for row in rows)
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "budget_exhausted_fraction": (
                    sum(_int(row.get("budget_exhausted_windows"), 0) for row in rows)
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "none_due_budget_fraction": (
                    sum(_int(row.get("none_due_budget_windows"), 0) for row in rows)
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "mask_or_budget_blocked_fraction": (
                    sum(
                        _int(row.get("mask_or_budget_blocked_windows"), 0)
                        for row in rows
                    )
                    / decisions
                    if decisions
                    else float("nan")
                ),
                "transition_invalidated_windows": transition_total,
                "transition_invalidated_fraction": (
                    transition_total / decisions if decisions else float("nan")
                ),
                "selected_action_mask_invalid": sum(
                    _int(row.get("selected_action_mask_invalid"), 0)
                    for row in rows
                ),
                "logged_invalid_measurement_action": sum(
                    _int(row.get("logged_invalid_measurement_action"), 0)
                    for row in rows
                ),
                "budget_over_limit": sum(
                    _int(row.get("budget_over_limit"), 0) for row in rows
                ),
                "budget_selected_when_exhausted": sum(
                    _int(row.get("budget_selected_when_exhausted"), 0)
                    for row in rows
                ),
                "quality_alignment_mismatch": sum(
                    _int(row.get("quality_alignment_mismatch"), 0)
                    for row in rows
                ),
                "raw_action_missing": sum(
                    _int(row.get("raw_action_missing"), 0) for row in rows
                ),
                "expected_probability_missing_or_invalid": sum(
                    _int(
                        row.get(
                            "expected_probability_missing_or_invalid_windows", 0
                        ),
                        0,
                    )
                    for row in rows
                ),
                "quality_feature_enabled": int(
                    branch in {"budgeted_quality_aware_mappo", RATE_BRANCH}
                ),
                "measurement_budget_feature_enabled": int(
                    branch in {"budgeted_quality_aware_mappo", RATE_BRANCH}
                ),
                "measurement_rate_constraint_enabled": int(branch == RATE_BRANCH),
            }
        )
    return output


def _branch_isolation(
    reference_root: Path,
    source_logs: Path,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check branch manifests and the rate-constraint training log."""

    rows: list[dict[str, Any]] = []
    rate_log = _read_csv(source_logs / "training_rate_constraint.csv")
    mismatches: list[str] = []
    quality_mismatches: list[str] = []
    for seed in seeds:
        for branch in BRANCHES:
            scale_root = _scale_root(reference_root, seed, branch)
            config = _manifest_config(scale_root)
            validation_quality = _quality_rows(scale_root, "validation")
            quality_branch_values = sorted(
                {str(row.get("quality_actor_branch", "")) for row in validation_quality}
            )
            quality_values = sorted(
                {
                    _int(row.get("quality_feature_enabled"), 0)
                    for row in validation_quality
                    if "quality_feature_enabled" in row
                }
            )
            budget_values = sorted(
                {
                    _int(row.get("measurement_budget_feature_enabled"), 0)
                    for row in validation_quality
                    if "measurement_budget_feature_enabled" in row
                }
            )
            constraint_values = sorted(
                {
                    _int(row.get("measurement_rate_constraint_enabled"), 0)
                    for row in validation_quality
                    if "measurement_rate_constraint_enabled" in row
                }
            )
            expected_quality = int(branch in {"budgeted_quality_aware_mappo", RATE_BRANCH})
            expected_budget = expected_quality
            expected_constraint = int(branch == RATE_BRANCH)
            actual = {
                "mappo_variant": str(config.get("mappo_variant", "")),
                "quality": int(float(config.get("measurement_quality_logit_penalty", 0.0)) > 0.0),
                "budget": int(bool(config.get("measurement_budget_enabled", False))),
                "constraint": int(bool(config.get("measurement_rate_constraint_enabled", False))),
            }
            ok = bool(
                actual["mappo_variant"] == branch
                and actual["quality"] == expected_quality
                and actual["budget"] == expected_budget
                and actual["constraint"] == expected_constraint
            )
            if not ok:
                mismatches.append(f"{seed}/{branch}")
            log_flags_match = (
                (not quality_values or expected_quality in quality_values)
                and (not budget_values or expected_budget in budget_values)
                and (not constraint_values or expected_constraint in constraint_values)
                and (not quality_branch_values or branch in quality_branch_values)
            )
            if not log_flags_match:
                quality_mismatches.append(f"{seed}/{branch}")
            rows.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "mappo_variant": actual["mappo_variant"],
                    "quality_feature_expected": expected_quality,
                    "measurement_budget_expected": expected_budget,
                    "measurement_rate_constraint_expected": expected_constraint,
                    "quality_config_value": actual["quality"],
                    "budget_config_value": actual["budget"],
                    "constraint_config_value": actual["constraint"],
                    "manifest_isolation_ok": int(ok),
                    "quality_log_branch_values": quality_branch_values,
                    "quality_log_quality_flag_values": quality_values,
                    "quality_log_budget_flag_values": budget_values,
                    "quality_log_constraint_flag_values": constraint_values,
                    "quality_log_flags_match_config": int(log_flags_match),
                    "rate_training_log_rows": sum(
                        1
                        for item in rate_log
                        if _int(item.get("base_seed"), -1) == int(seed)
                        and str(item.get("treatment", "")) == RATE_BRANCH
                    ),
                    "source_manifest": str(_scale_root(reference_root, seed, branch) / "manifest.json"),
                }
            )
    return rows, {
        "all_manifest_isolation_ok": not mismatches,
        "manifest_mismatches": mismatches,
        "all_quality_log_flags_match": not quality_mismatches,
        "quality_log_flag_mismatches": quality_mismatches,
        "rate_training_log_exists": bool((source_logs / "training_rate_constraint.csv").exists()),
    }


def _numeric_rows_finite(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bool:
    return all(_finite_number(row.get(field)) for row in rows for field in fields)


def _decision_report(
    config: MeasurementRateSamplingAuditConfig,
    targets: Mapping[tuple[int, str], float],
    target_rows: Sequence[Mapping[str, Any]],
    target_status: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    training_meta: Mapping[str, Any],
    source_logs: Path,
    branch_rows: Sequence[Mapping[str, Any]],
    branch_meta: Mapping[str, Any],
    data_meta: Mapping[str, Any],
) -> dict[str, Any]:
    rate_seed_rows = [row for row in seed_rows if str(row.get("branch")) == RATE_BRANCH]
    expected_close = [
        _finite_number(row.get("absolute_expected_minus_target"))
        and float(row["absolute_expected_minus_target"]) <= 0.10
        for row in rate_seed_rows
    ]
    actual_close = [
        _finite_number(row.get("absolute_actual_minus_target"))
        and float(row["absolute_actual_minus_target"]) <= 0.10
        for row in rate_seed_rows
    ]
    sampling_compatible = [bool(_int(row.get("sampling_gap_within_reference"), 0)) for row in rate_seed_rows]
    expected_close_fraction = float(np.mean(expected_close)) if expected_close else 0.0
    actual_close_fraction = float(np.mean(actual_close)) if actual_close else 0.0
    sampling_compatible_fraction = float(np.mean(sampling_compatible)) if sampling_compatible else 0.0
    blocker_fraction = [_float(row.get("mask_or_budget_blocked_fraction"), float("nan")) for row in rate_seed_rows]
    blocker_high = bool(blocker_fraction) and float(np.mean([value >= 0.20 for value in blocker_fraction])) >= 0.5
    actual_expected_gap_large = [
        _finite_number(row.get("absolute_actual_minus_expected"))
        and float(row["absolute_actual_minus_expected"]) > max(
            0.05,
            1.96 * _float(row.get("conditional_sampling_reference_se"), 0.0),
        )
        for row in rate_seed_rows
    ]
    actual_not_close = [not value for value in actual_close]
    alignment_mismatch = int(data_meta.get("quality_alignment_mismatch_count", 0)) > 0
    invalid_probability = int(data_meta.get("quality_probability_invalid_count", 0)) > 0
    invalid_mask = int(data_meta.get("selected_action_mask_invalid_count", 0)) > 0
    invalid_logged = int(data_meta.get("logged_invalid_measurement_action_count", 0)) > 0
    budget_violations = int(data_meta.get("budget_over_limit_count", 0)) + int(
        data_meta.get("budget_selected_when_exhausted_count", 0)
    )
    structural_ok = bool(
        bool(targets)
        and len(target_rows) == len(config.seeds) * config.uav_count
        and bool(target_status.get("complete_seed_bound_match", False))
        and bool(data_meta.get("reference_roots_complete", False))
        and bool(branch_meta.get("all_manifest_isolation_ok", False))
        and bool(training_meta.get("enabled_row_count", 0) > 0)
        and bool(training_meta.get("positive_logged_loss_row_count", 0) > 0)
        and len(seed_rows) == len(config.seeds) * len(BRANCHES)
    )

    if not structural_ok or alignment_mismatch or invalid_probability or invalid_mask or invalid_logged:
        primary_reason = "probability_logging_sampling_mismatch" if (alignment_mismatch or invalid_probability) else "insufficient_evidence"
        detail = (
            "The persisted action and actor-visible probability streams do not align exactly, or contain invalid probability/mask records."
            if primary_reason == "probability_logging_sampling_mismatch"
            else "Required train/validation provenance, branch isolation, or finite diagnostics are incomplete."
        )
        next_action = "Fix only the logging/sampling or action-mask interface, then rerun a seed=2026 Smoke."
    elif blocker_high:
        primary_reason = "budget_or_mask_dominates"
        detail = "Candidate masks or budget exhaustion block a substantial fraction of rate-constraint decision windows."
        next_action = "Audit only the rate denominator and budget/mask interaction; do not change the coefficient yet."
    elif expected_close_fraction < 0.8:
        primary_reason = "expected_rate_not_at_target"
        detail = "The differentiable expected non-none rate is not near the train-derived target for enough independent seeds."
        next_action = "Inspect only the per-UAV constraint aggregation, train/validation distribution shift, and gradient path before retuning."
    elif any(actual_not_close) and any(actual_expected_gap_large):
        primary_reason = "sampled_expected_rate_gap"
        detail = "Expected probabilities are near target, but sampled actions depart beyond the conditional finite-sampling reference."
        next_action = "Inspect only the discrete action sampling, probability logging, and rollout aggregation path; do not increase the coefficient."
    elif sampling_compatible_fraction >= 0.8:
        primary_reason = "insufficient_evidence"
        detail = "Expected and sampled rates are compatible with the conditional finite-sampling reference; no implementation failure is established."
        next_action = "Freeze the coefficient and add pre-registered seeds or episodes; do not retune from this audit."
    else:
        primary_reason = "insufficient_evidence"
        detail = "The available logs do not distinguish a specific rate-control failure mode at the seed level."
        next_action = "Add only the missing per-episode probability/sampling diagnostics, then repeat the seed-level audit."

    rate_actual_deltas = [
        _float(row.get("actual_minus_target"), float("nan")) for row in rate_seed_rows
    ]
    rate_expected_deltas = [
        _float(row.get("expected_minus_target"), float("nan")) for row in rate_seed_rows
    ]
    return {
        "experiment": "measurement_rate_sampling_audit",
        "evidence_status": "validation_only_mechanism_diagnostic",
        "reference_root": str(config.reference_root),
        "source_logs_dir": str(source_logs),
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": list(BRANCHES),
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_read": False,
        "test_data_used_for_selection": False,
        "retraining_performed": False,
        "original_results_modified": False,
        "fit_sources": {
            "measurement_rate_targets": "baseline_mappo/train actions only",
            "training_constraint_diagnosis": "rate branch train logs and train actor-visible probability logs",
            "primary_evaluation": "validation actions and actor-visible probability logs only",
        },
        "fixed_controls": {
            "uav_count": int(config.uav_count),
            "split": config.split,
            "bootstrap_samples": int(config.bootstrap_samples),
            "bootstrap_seed": int(config.bootstrap_seed),
            "measurement_rate_constraint_coef": 0.5,
            "measurement_rate_tolerance": 0.05,
        },
        "checks": {
            "reference_roots_complete": bool(data_meta.get("reference_roots_complete", False)),
            "target_recomputation_complete": len(target_rows) == len(config.seeds) * config.uav_count,
            "target_logged_seed_bound_match": bool(target_status.get("complete_seed_bound_match", False)),
            "branch_isolation": bool(branch_meta.get("all_manifest_isolation_ok", False)),
            "rate_constraint_training_rows_present": bool(training_meta.get("enabled_row_count", 0) > 0),
            "rate_constraint_positive_loss_present": bool(training_meta.get("positive_logged_loss_row_count", 0) > 0),
            "validation_seed_rows_complete": len(seed_rows) == len(config.seeds) * len(BRANCHES),
            "validation_probability_missing_count": int(data_meta.get("quality_probability_missing_count", 0)),
            "validation_probability_invalid_count": int(data_meta.get("quality_probability_invalid_count", 0)),
            "probability_action_alignment_mismatch_count": int(data_meta.get("quality_alignment_mismatch_count", 0)),
            "selected_action_mask_invalid_count": int(data_meta.get("selected_action_mask_invalid_count", 0)),
            "logged_invalid_measurement_action_count": int(data_meta.get("logged_invalid_measurement_action_count", 0)),
            "budget_violations": budget_violations,
            "all_expected_rows_finite": _numeric_rows_finite(
                expected_rows,
                (
                    "decision_windows",
                    "eligible_candidate_windows",
                    "actual_sampled_rate_all_windows",
                    "expected_non_none_rate_all_windows",
                    "conditional_sampling_se_all_windows",
                ),
            ),
            "all_seed_rows_finite": _numeric_rows_finite(
                seed_rows,
                (
                    "decision_windows",
                    "selected_measurements",
                    "actual_sampled_rate_all_windows",
                    "expected_non_none_rate_all_windows",
                    "target_measurement_rate",
                ),
            ),
            "test_split_read": False,
        },
        "evidence": {
            "expected_close_seed_fraction": expected_close_fraction,
            "actual_close_seed_fraction": actual_close_fraction,
            "sampling_compatible_seed_fraction": sampling_compatible_fraction,
            "rate_seed_rows": list(rate_seed_rows),
            "rate_actual_minus_target_bootstrap_95_ci": dict(
                zip(("low", "high"), _bootstrap_mean_ci(rate_actual_deltas, config.bootstrap_samples, config.bootstrap_seed))
            ),
            "rate_expected_minus_target_bootstrap_95_ci": dict(
                zip(("low", "high"), _bootstrap_mean_ci(rate_expected_deltas, config.bootstrap_samples, config.bootstrap_seed + 1))
            ),
            "training_constraint": dict(training_meta),
            "data_meta": dict(data_meta),
        },
        "decision_gate": primary_reason,
        "primary_reason": primary_reason,
        "primary_reason_detail": detail,
        "next_action": next_action,
        "audit_ok": structural_ok and not invalid_probability and not invalid_mask and not invalid_logged,
    }


def _plot(seed_rows: Sequence[Mapping[str, Any]], plots_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2563eb",
        "budgeted_quality_aware_mappo": "#dc2626",
        RATE_BRANCH: "#16a34a",
    }
    labels = {
        "baseline_mappo": "Baseline MAPPO",
        "budgeted_quality_aware_mappo": "Budgeted quality-aware",
        RATE_BRANCH: "Rate-controlled quality-aware",
    }
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for branch in BRANCHES:
        rows = sorted(
            [row for row in seed_rows if str(row.get("branch")) == branch],
            key=lambda row: int(row["base_seed"]),
        )
        x = [int(row["base_seed"]) for row in rows]
        axes[0, 0].plot(
            x,
            [_float(row.get("actual_sampled_rate_all_windows"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=f"{labels[branch]} actual",
        )
        axes[0, 0].plot(
            x,
            [_float(row.get("expected_non_none_rate_all_windows"), float("nan")) for row in rows],
            marker="x",
            linestyle="--",
            alpha=0.7,
            color=colors[branch],
            label=f"{labels[branch]} expected",
        )
        axes[0, 1].plot(
            x,
            [_float(row.get("actual_minus_expected"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=labels[branch],
        )
        axes[1, 0].plot(
            x,
            [_float(row.get("mask_or_budget_blocked_fraction"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=labels[branch],
        )
        axes[1, 1].plot(
            x,
            [_float(row.get("absolute_actual_minus_target"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=labels[branch],
        )
    axes[0, 0].set_title("Validation actual vs expected rate")
    axes[0, 0].set_ylabel("Non-none rate")
    axes[0, 1].set_title("Actual minus expected rate")
    axes[0, 1].set_ylabel("Rate difference")
    axes[1, 0].set_title("Candidate-mask or budget blocking")
    axes[1, 0].set_ylabel("Blocked fraction")
    axes[1, 1].set_title("Absolute deviation from train target")
    axes[1, 1].set_ylabel("Absolute rate deviation")
    for axis in axes.ravel():
        axis.set_xlabel("Base training seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / PLOT_FILE
    figure.suptitle("Measurement-rate sampling audit (validation; test sealed)")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _decision_report(
    config: MeasurementRateSamplingAuditConfig,
    targets: Mapping[tuple[int, str], float],
    target_rows: Sequence[Mapping[str, Any]],
    target_status: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    training_meta: Mapping[str, Any],
    source_logs: Path,
    branch_rows: Sequence[Mapping[str, Any]],
    branch_meta: Mapping[str, Any],
    data_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Make a structural decision without treating validation as sampling."""

    del gate_rows, training_rows, branch_rows
    rate_seed_rows = [
        row for row in seed_rows if str(row.get("branch")) == RATE_BRANCH
    ]
    expected_close = [
        _finite_number(row.get("absolute_expected_minus_target"))
        and float(row["absolute_expected_minus_target"]) <= 0.10
        for row in rate_seed_rows
    ]
    raw_actual_close = [
        _finite_number(row.get("absolute_raw_sampled_minus_target"))
        and float(row["absolute_raw_sampled_minus_target"]) <= 0.10
        for row in rate_seed_rows
    ]
    expected_close_fraction = float(np.mean(expected_close)) if expected_close else 0.0
    raw_actual_close_fraction = (
        float(np.mean(raw_actual_close)) if raw_actual_close else 0.0
    )
    blocker_values = [
        _float(row.get("mask_or_budget_blocked_fraction"), float("nan"))
        for row in rate_seed_rows
    ]
    blocker_values = [value for value in blocker_values if _finite_number(value)]
    blocker_high = bool(blocker_values) and float(
        np.mean([value >= 0.20 for value in blocker_values])
    ) >= 0.5
    validation_mode = str(
        data_meta.get("action_selection_mode", "deterministic_argmax")
    )
    sampling_reference_applicable = validation_mode == "stochastic_sample"

    alignment_mismatch = int(data_meta.get("quality_alignment_mismatch_count", 0)) > 0
    invalid_probability = int(data_meta.get("quality_probability_invalid_count", 0)) > 0
    missing_probability = int(data_meta.get("quality_probability_missing_count", 0)) > 0
    raw_missing = int(data_meta.get("raw_action_missing_count", 0)) > 0
    invalid_mask = int(data_meta.get("selected_action_mask_invalid_count", 0)) > 0
    invalid_logged = int(data_meta.get("logged_invalid_measurement_action_count", 0)) > 0
    budget_violations = int(data_meta.get("budget_over_limit_count", 0)) + int(
        data_meta.get("budget_selected_when_exhausted_count", 0)
    )
    raw_target_complete = bool(target_rows) and all(
        _finite_number(row.get("raw_baseline_train_measurement_rate"))
        and str(row.get("raw_target_source_split")) == "train"
        and str(row.get("raw_target_source_branch")) == "baseline_mappo"
        for row in target_rows
    )
    finite_expected = _numeric_rows_finite(
        expected_rows,
        (
            "decision_windows",
            "eligible_candidate_windows",
            "raw_sampled_rate_all_windows",
            "applied_measurement_rate_all_windows",
            "expected_non_none_rate_all_windows",
            "conditional_sampling_se_all_windows",
            "sampling_variance_numerator_all_windows",
        ),
    )
    finite_seed = _numeric_rows_finite(
        seed_rows,
        (
            "decision_windows",
            "raw_selected_measurements",
            "applied_selected_measurements",
            "raw_sampled_rate_all_windows",
            "applied_measurement_rate_all_windows",
            "expected_non_none_rate_all_windows",
            "sampling_target_measurement_rate",
            "target_measurement_rate",
        ),
    )
    structural_ok = bool(
        targets
        and raw_target_complete
        and len(target_rows) == len(config.seeds) * config.uav_count
        and bool(target_status.get("complete_seed_bound_match", False))
        and bool(data_meta.get("reference_roots_complete", False))
        and bool(branch_meta.get("all_manifest_isolation_ok", False))
        and bool(branch_meta.get("all_quality_log_flags_match", False))
        and bool(training_meta.get("enabled_row_count", 0) > 0)
        and bool(training_meta.get("positive_logged_loss_row_count", 0) > 0)
        and len(seed_rows) == len(config.seeds) * len(BRANCHES)
        and finite_expected
        and finite_seed
        and not missing_probability
        and not raw_missing
    )

    if alignment_mismatch or invalid_probability or invalid_mask or invalid_logged:
        primary_reason = "probability_logging_sampling_mismatch"
        detail = (
            "The raw actor action, actor-visible probability log, action mask, or "
            "invalid-action flag is inconsistent. Environment-side transition "
            "invalidation is excluded from this failure class."
        )
        next_action = (
            "Fix only probability/action or mask logging, then rerun the seed=2026 "
            "diagnostic Smoke."
        )
    elif not structural_ok:
        primary_reason = "insufficient_evidence"
        detail = (
            "Train/validation provenance, branch isolation, finite fields, or "
            "actor diagnostics are incomplete."
        )
        next_action = (
            "Add only the missing provenance or diagnostic fields; do not retune "
            "the rate coefficient."
        )
    elif blocker_high:
        primary_reason = "budget_or_mask_dominates"
        detail = (
            "Budget exhaustion or candidate masks block a substantial fraction of "
            "rate-constraint decision windows."
        )
        next_action = (
            "Audit only the rate denominator and budget/mask interaction before "
            "changing the coefficient."
        )
    elif expected_close_fraction < 0.80:
        primary_reason = "expected_rate_not_at_target"
        detail = (
            "The differentiable expected non-none probability is not close to the "
            "actor-action target for enough independent seeds."
        )
        next_action = (
            "Inspect only per-UAV constraint aggregation, train/validation shift, "
            "and the gradient path before retuning."
        )
    elif sampling_reference_applicable:
        gap_large = [
            _finite_number(row.get("absolute_actual_minus_expected"))
            and _finite_number(row.get("conditional_sampling_reference_se"))
            and float(row["absolute_actual_minus_expected"])
            > max(
                0.05,
                1.96 * float(row["conditional_sampling_reference_se"]),
            )
            for row in rate_seed_rows
        ]
        if any(gap_large):
            primary_reason = "sampled_expected_rate_gap"
            detail = (
                "Expected probabilities are near target, but stochastic realized "
                "actions depart beyond the conditional finite-sampling reference."
            )
            next_action = (
                "Inspect discrete action sampling, probability logging, and "
                "rollout aggregation only."
            )
        else:
            primary_reason = "insufficient_evidence"
            detail = (
                "Train actions are compatible with the conditional sampling "
                "reference; validation is not a sampling test."
            )
            next_action = (
                "Freeze the coefficient and add preregistered seeds or episodes."
            )
    elif not all(raw_actual_close):
        primary_reason = "insufficient_evidence"
        detail = (
            "Validation uses deterministic argmax actions. A difference between "
            "expected probability and realized action is a policy-mode diagnostic, "
            "not Bernoulli sampling noise."
        )
        next_action = (
            "Run a deterministic-policy/action-mode diagnostic before changing "
            "the rate objective or coefficient."
        )
    else:
        primary_reason = "insufficient_evidence"
        detail = (
            "Expected probability, deterministic validation action, and train target "
            "do not establish a distinct rate-control failure mode."
        )
        next_action = "Freeze the coefficient and add preregistered seeds or episodes."

    transition_total = sum(
        _int(row.get("transition_invalidated_windows"), 0)
        for row in expected_rows
    )
    total_decisions = sum(
        _int(row.get("decision_windows"), 0) for row in expected_rows
    )
    transition_fraction = (
        float(transition_total / total_decisions)
        if total_decisions
        else float("nan")
    )
    raw_actual_deltas = [
        _float(row.get("raw_sampled_minus_target"), float("nan"))
        for row in rate_seed_rows
    ]
    expected_deltas = [
        _float(row.get("expected_minus_target"), float("nan"))
        for row in rate_seed_rows
    ]
    return {
        "experiment": "measurement_rate_sampling_audit",
        "evidence_status": "validation_only_mechanism_diagnostic",
        "reference_root": str(config.reference_root),
        "source_logs_dir": str(source_logs),
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": list(BRANCHES),
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_read": False,
        "test_data_used_for_selection": False,
        "retraining_performed": False,
        "original_results_modified": False,
        "fit_sources": {
            "measurement_rate_targets": (
                "baseline_mappo/train actions.csv applied measurement_uav_id"
            ),
            "actor_sampling_targets": (
                "baseline_mappo/train actions.csv action_raw"
            ),
            "training_constraint_diagnosis": (
                "rate branch train logs and actor-visible probability logs"
            ),
            "primary_evaluation": (
                "validation action_raw and actor-visible probability logs"
            ),
        },
        "action_semantics": {
            "train": "stochastic_sample",
            "validation": "deterministic_argmax",
            "raw_actor_field": "action_raw.measurement_uav_id",
            "applied_environment_field": "measurement_uav_id",
            "transition_invalidation_is_not_logging_mismatch": True,
        },
        "fixed_controls": {
            "uav_count": int(config.uav_count),
            "split": config.split,
            "bootstrap_samples": int(config.bootstrap_samples),
            "bootstrap_seed": int(config.bootstrap_seed),
            "measurement_rate_constraint_coef": 0.5,
            "measurement_rate_tolerance": 0.05,
        },
        "checks": {
            "reference_roots_complete": bool(
                data_meta.get("reference_roots_complete", False)
            ),
            "target_recomputation_complete": len(target_rows)
            == len(config.seeds) * config.uav_count,
            "raw_target_recomputation_complete": raw_target_complete,
            "target_logged_seed_bound_match": bool(
                target_status.get("complete_seed_bound_match", False)
            ),
            "branch_isolation": bool(
                branch_meta.get("all_manifest_isolation_ok", False)
            ),
            "quality_log_flags_isolated": bool(
                branch_meta.get("all_quality_log_flags_match", False)
            ),
            "rate_constraint_training_rows_present": bool(
                training_meta.get("enabled_row_count", 0) > 0
            ),
            "rate_constraint_positive_loss_present": bool(
                training_meta.get("positive_logged_loss_row_count", 0) > 0
            ),
            "validation_seed_rows_complete": len(seed_rows)
            == len(config.seeds) * len(BRANCHES),
            "validation_probability_missing_count": int(
                data_meta.get("quality_probability_missing_count", 0)
            ),
            "validation_probability_invalid_count": int(
                data_meta.get("quality_probability_invalid_count", 0)
            ),
            "probability_action_alignment_mismatch_count": int(
                data_meta.get("quality_alignment_mismatch_count", 0)
            ),
            "raw_action_missing_count": int(
                data_meta.get("raw_action_missing_count", 0)
            ),
            "selected_action_mask_invalid_count": int(
                data_meta.get("selected_action_mask_invalid_count", 0)
            ),
            "logged_invalid_measurement_action_count": int(
                data_meta.get("logged_invalid_measurement_action_count", 0)
            ),
            "budget_violations": budget_violations,
            "all_expected_rows_finite": finite_expected,
            "all_seed_rows_finite": finite_seed,
            "validation_action_selection_mode": validation_mode,
            "sampling_reference_applicable": sampling_reference_applicable,
            "test_split_read": False,
        },
        "evidence": {
            "expected_close_seed_fraction": expected_close_fraction,
            "raw_actual_close_seed_fraction": raw_actual_close_fraction,
            "sampling_compatible_seed_fraction": None
            if not sampling_reference_applicable
            else 0.0,
            "rate_seed_rows": list(rate_seed_rows),
            "transition_invalidated_total": transition_total,
            "transition_invalidated_fraction": transition_fraction,
            "raw_target_rows": [
                row
                for row in target_rows
                if "raw_baseline_train_measurement_rate" in row
            ],
            "training_constraint": dict(training_meta),
            "data_meta": dict(data_meta),
        },
        "bootstrap_95_ci": {
            "raw_sampled_minus_target_mean": dict(
                zip(
                    ("low", "high"),
                    _bootstrap_mean_ci(
                        raw_actual_deltas,
                        config.bootstrap_samples,
                        config.bootstrap_seed,
                    ),
                )
            ),
            "expected_minus_target_mean": dict(
                zip(
                    ("low", "high"),
                    _bootstrap_mean_ci(
                        expected_deltas,
                        config.bootstrap_samples,
                        config.bootstrap_seed + 1,
                    ),
                )
            ),
        },
        "decision_gate": primary_reason,
        "primary_reason": primary_reason,
        "primary_reason_detail": detail,
        "next_action": next_action,
        "audit_ok": bool(
            structural_ok
            and not alignment_mismatch
            and not invalid_probability
            and not invalid_mask
            and not invalid_logged
            and budget_violations == 0
        ),
    }


def _plot(seed_rows: Sequence[Mapping[str, Any]], plots_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_mappo": "#2563eb",
        "budgeted_quality_aware_mappo": "#dc2626",
        RATE_BRANCH: "#16a34a",
    }
    labels = {
        "baseline_mappo": "Baseline MAPPO",
        "budgeted_quality_aware_mappo": "Budgeted quality-aware",
        RATE_BRANCH: "Rate-controlled quality-aware",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for branch in BRANCHES:
        rows = sorted(
            [row for row in seed_rows if str(row.get("branch")) == branch],
            key=lambda row: int(row["base_seed"]),
        )
        x = [int(row["base_seed"]) for row in rows]
        axes[0, 0].plot(
            x,
            [_float(row.get("raw_sampled_rate_all_windows"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=f"{labels[branch]} raw",
        )
        axes[0, 0].plot(
            x,
            [_float(row.get("expected_non_none_rate_all_windows"), float("nan")) for row in rows],
            marker="x",
            linestyle="--",
            color=colors[branch],
            alpha=0.75,
            label=f"{labels[branch]} expected",
        )
        axes[0, 1].plot(
            x,
            [_float(row.get("raw_sampled_minus_expected"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=labels[branch],
        )
        axes[1, 0].plot(
            x,
            [_float(row.get("mask_or_budget_blocked_fraction"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=labels[branch],
        )
        axes[1, 1].plot(
            x,
            [_float(row.get("absolute_expected_minus_target"), float("nan")) for row in rows],
            marker="o",
            color=colors[branch],
            label=labels[branch],
        )
    axes[0, 0].set_title("Validation raw action vs expected probability")
    axes[0, 0].set_ylabel("Non-none rate")
    axes[0, 1].set_title("Raw action minus expected probability")
    axes[0, 1].set_ylabel("Rate difference")
    axes[1, 0].set_title("Candidate-mask or budget blocking")
    axes[1, 0].set_ylabel("Blocked fraction")
    axes[1, 1].set_title("Expected probability deviation from raw target")
    axes[1, 1].set_ylabel("Absolute rate deviation")
    for axis in axes.ravel():
        axis.set_xlabel("Base training seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / PLOT_FILE
    figure.suptitle(
        "Measurement-rate sampling audit (validation; deterministic argmax; test sealed)"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def run_measurement_rate_sampling_audit(
    config: MeasurementRateSamplingAuditConfig,
) -> dict[str, Any]:
    """Run the non-mutating expected-versus-sampled rate audit."""

    _prepare_output(config.output_dir, force=config.force)
    plots_dir = config.plots_dir or (
        Path("plots/preliminary") / config.reference_root.name / "rate_sampling_audit"
    )
    plots_dir = Path(plots_dir)
    source_logs = Path(config.source_logs_dir or (Path("logs/preliminary") / config.reference_root.name))
    _progress(config, f"reading train targets from {config.reference_root}")

    for seed in config.seeds:
        for branch in BRANCHES:
            scale_root = _scale_root(config.reference_root, int(seed), branch)
            if not scale_root.is_dir():
                raise FileNotFoundError(f"missing seed/branch scale root: {scale_root}")

    targets, recomputed_targets, _target_source_counts = _fit_targets(
        config.reference_root, config.seeds, config.uav_count
    )
    raw_targets, raw_target_rows, raw_target_missing = _fit_raw_targets(
        config.reference_root, config.seeds, config.uav_count
    )
    target_rows, target_status = _logged_target_status(
        source_logs, recomputed_targets
    )
    raw_by_key = {
        (int(row["base_seed"]), str(row["uav_id"])): row
        for row in raw_target_rows
    }
    target_rows = [
        {
            **dict(row),
            **dict(
                raw_by_key.get(
                    (int(row["base_seed"]), str(row["uav_id"])),
                    {},
                )
            ),
        }
        for row in target_rows
    ]
    _progress(config, "reading training constraint diagnostics")
    training_rows, training_meta = _training_constraint_rows(
        config.reference_root,
        source_logs,
        config.seeds,
        targets,
        config.uav_count,
    )
    _progress(config, "aligning validation probabilities, actions, masks, and budgets")
    expected_rows, gate_rows, data_meta = _episode_rows(
        config.reference_root,
        config.seeds,
        BRANCHES,
        targets,
        raw_targets,
        config.uav_count,
        config.split,
    )
    data_meta["raw_target_missing_count"] = int(raw_target_missing)
    seed_rows = _seed_summary(expected_rows, uav_count=config.uav_count)
    branch_rows, branch_meta = _branch_isolation(config.reference_root, source_logs, config.seeds)
    report = _decision_report(
        config,
        targets,
        target_rows,
        target_status,
        expected_rows,
        gate_rows,
        seed_rows,
        training_rows,
        training_meta,
        source_logs,
        branch_rows,
        branch_meta,
        data_meta,
    )
    plot = _plot(seed_rows, plots_dir)

    _write_csv(config.output_dir / "expected_vs_sampled.csv", expected_rows)
    _write_csv(config.output_dir / "target_recomputation.csv", target_rows)
    _write_csv(config.output_dir / "mask_budget_audit.csv", gate_rows)
    _write_csv(config.output_dir / "training_constraint_audit.csv", training_rows)
    _write_csv(config.output_dir / "seed_summary.csv", seed_rows)
    _write_json(
        config.output_dir / "audit_report.json",
        {**report, "plot": str(plot), "branch_isolation_rows": branch_rows},
    )
    _progress(config, f"decision={report['decision_gate']} audit_ok={report['audit_ok']}")
    return {
        "output_dir": str(config.output_dir),
        "plots_dir": str(plots_dir),
        "audit_report": str(config.output_dir / "audit_report.json"),
        "plot": str(plot),
        "audit_ok": bool(report["audit_ok"]),
        "decision_gate": str(report["decision_gate"]),
        "primary_reason": str(report["primary_reason"]),
        "test_data_read": False,
    }


__all__ = [
    "DEFAULT_SEEDS",
    "MeasurementRateSamplingAuditConfig",
    "run_measurement_rate_sampling_audit",
]
