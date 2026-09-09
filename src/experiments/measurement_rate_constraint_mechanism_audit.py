"""Read-only audit for measurement-rate aggregation and gradient flow.

The audit is deliberately separate from the earlier target/probability audit.
It uses the already persisted train and validation artifacts to answer three
questions:

* Are the per-seed/per-UAV targets and denominators the ones consumed by the
  rate-controlled branch?
* Do the explicit per-UAV training statistics close to the aggregate loss?
* Does a positive rate penalty have a finite path to the measurement head?

No environment is replayed, no optimizer step is executed, and the test
split is never enumerated.  Windows and episodes are diagnostic rows; the
seed is the independent unit for the decision summary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_SEEDS = (7071, 8080, 9099, 10118, 11127)
BRANCHES = (
    "baseline_mappo",
    "budgeted_quality_aware_mappo",
    "rate_controlled_quality_aware_mappo",
)
RATE_BRANCH = "rate_controlled_quality_aware_mappo"
QUALITY_BRANCHES = {"budgeted_quality_aware_mappo", RATE_BRANCH}
MANAGED_FILES = (
    "per_uav_rate_audit.csv",
    "aggregation_audit.csv",
    "train_validation_shift.csv",
    "gradient_path_audit.csv",
    "seed_summary.csv",
    "audit_report.json",
)
PLOT_FILE = "measurement_rate_constraint_mechanism_audit.png"


@dataclass(frozen=True)
class MeasurementRateConstraintMechanismAuditConfig:
    """Configuration for the validation-only mechanism audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/measurement_rate_controlled_mappo_constraint_trace_seedset_7071_11127/"
        "rate_constraint_mechanism_audit"
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
            raise ValueError("measurement-rate mechanism audit is validation-only")
        if self.uav_count < 2:
            raise ValueError("uav_count must be at least 2")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: MeasurementRateConstraintMechanismAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[measurement-rate-constraint-mechanism-audit] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_csv_with_fields(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(payload, Mapping):
                yield dict(payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    number = _float(value, float(default))
    return int(number) if math.isfinite(number) else int(default)


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                columns.append(key)
                seen.add(key)
    if not columns:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (dict, list, tuple)):
                    output[column] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                elif isinstance(value, (np.integer,)):
                    output[column] = int(value)
                elif isinstance(value, (np.floating,)):
                    number = float(value)
                    output[column] = number if math.isfinite(number) else ""
                elif isinstance(value, float) and not math.isfinite(value):
                    output[column] = ""
                else:
                    output[column] = value
            writer.writerow(output)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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


def _prepare_plot_output(path: Path, *, force: bool) -> None:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if force and (path / PLOT_FILE).exists():
        (path / PLOT_FILE).unlink()


def _uav_ids(uav_count: int) -> list[str]:
    return [f"uav_{index:02d}" for index in range(uav_count)]


def _scale_root(reference_root: Path, seed: int, branch: str) -> Path:
    return reference_root / f"seed_{int(seed)}" / branch / "scale_3"


def _episode_dirs(scale_root: Path, split: str) -> list[Path]:
    split_root = scale_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing {split} directory: {split_root}")
    return sorted(path for path in split_root.iterdir() if path.is_dir())


def _action_rows(scale_root: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for episode_root in _episode_dirs(scale_root, split):
        path = episode_root / "actions.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing actions.csv: {path}")
        for row in _read_csv(path):
            if str(row.get("split", split)) != split:
                raise ValueError(f"action split mismatch in {path}: {row.get('split')}")
            row["_episode_root"] = str(episode_root)
            rows.append(row)
    return rows


def _quality_rows(scale_root: Path, split: str) -> list[dict[str, Any]]:
    path = scale_root / "diagnostics" / "measurement_quality.jsonl"
    return [
        row
        for row in _iter_jsonl(path)
        if str(row.get("split", "")) == split
    ]


def _manifest_config(scale_root: Path) -> dict[str, Any]:
    path = scale_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    payload = _read_json(path)
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"manifest config is not an object: {path}")
    return dict(config)


def _is_selected(value: Any) -> bool:
    text = str(value if value is not None else "none").strip().lower()
    return bool(text) and text not in {"none", "nan", "null"}


def _raw_measurement_id(value: Any) -> tuple[str, bool]:
    if isinstance(value, Mapping):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return "", False
    if not isinstance(payload, Mapping):
        return "", False
    selected = payload.get("measurement_uav_id", "none")
    return str(selected).strip().lower(), True


def _parse_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _parse_action_mask(value: Any, uav_count: int) -> tuple[bool, bool, int]:
    payload = _parse_json_field(value, {})
    if not isinstance(payload, Mapping):
        return False, False, 0
    values = payload.get("measurement")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False, False, 0
    flags = [bool(item) for item in list(values)[:uav_count]]
    return bool(any(flags)), True, int(sum(flags))


def _sorted_uav_rows(rows: Sequence[Mapping[str, Any]], uav_id: str) -> list[Mapping[str, Any]]:
    return sorted(
        [row for row in rows if str(row.get("uav_id", "")) == uav_id],
        key=lambda row: (
            _float(row.get("timestamp"), 0.0),
            _int(row.get("step"), 0),
            str(row.get("measurement_uav_id", "")),
        ),
    )


def _budget_rows(scale_root: Path, split: str) -> dict[str, dict[str, list[dict[str, str]]]]:
    output: dict[str, dict[str, list[dict[str, str]]]] = {}
    for episode_root in _episode_dirs(scale_root, split):
        path = episode_root / "measurement_budget.csv"
        if not path.exists():
            continue
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(path):
            grouped[str(row.get("uav_id", ""))].append(row)
        for uav_id in grouped:
            grouped[uav_id] = sorted(grouped[uav_id], key=lambda row: _int(row.get("step"), 0))
        output[episode_root.name] = dict(grouped)
    return output


def _action_summary(
    scale_root: Path,
    split: str,
    *,
    uav_count: int,
) -> dict[str, dict[str, Any]]:
    rows = _action_rows(scale_root, split)
    budgets = _budget_rows(scale_root, split)
    output: dict[str, dict[str, Any]] = {}
    for uav_id in _uav_ids(uav_count):
        selected_raw = 0
        selected_applied = 0
        raw_missing = 0
        candidate_windows = 0
        candidate_blocked = 0
        mask_unparseable = 0
        invalid_action = 0
        transition_invalidated = 0
        budget_exhausted = 0
        none_due_budget = 0
        budget_over_limit = 0
        budget_consumed = 0
        decision_rows = 0
        eligible_rows = 0
        for row in _sorted_uav_rows(rows, uav_id):
            decision_rows += 1
            applied = _is_selected(row.get("measurement_uav_id", "none"))
            if applied:
                selected_applied += 1
            raw, parseable = _raw_measurement_id(row.get("action_raw", ""))
            if not parseable:
                raw_missing += 1
            if parseable and _is_selected(raw):
                selected_raw += 1
            if _int(row.get("invalid_measurement_action"), 0):
                invalid_action += 1
            if _int(row.get("measurement_transition_invalidated"), 0):
                transition_invalidated += 1
            elif parseable and _is_selected(raw) and not applied:
                transition_invalidated += 1
            has_candidate, mask_parseable, candidate_count = _parse_action_mask(
                row.get("action_mask_json", ""), uav_count
            )
            if not mask_parseable:
                mask_unparseable += 1
            elif has_candidate:
                candidate_windows += 1
                eligible_rows += 1
            else:
                candidate_blocked += 1
            episode_id = str(row.get("episode_id", ""))
            budget_for_episode = budgets.get(episode_id, {})
            budget_for_uav = budget_for_episode.get(uav_id, [])
            step = _int(row.get("step"), decision_rows - 1)
            budget_row = (
                budget_for_uav[step]
                if 0 <= step < len(budget_for_uav)
                else (
                    budget_for_uav[decision_rows - 1]
                    if decision_rows - 1 < len(budget_for_uav)
                    else {}
                )
            )
            remaining = _float(budget_row.get("budget_remaining_before"), float("nan"))
            limit = _int(budget_row.get("budget_limit"), -1)
            if math.isfinite(remaining) and remaining <= 0.0:
                budget_exhausted += 1
                if not applied:
                    none_due_budget += 1
            used = _int(budget_row.get("budget_used"), 0)
            consumed = _int(budget_row.get("budget_consumed"), 0)
            budget_consumed += consumed
            if limit >= 0 and used > limit:
                budget_over_limit += 1
        output[uav_id] = {
            "decision_windows": decision_rows,
            "eligible_candidate_windows": eligible_rows,
            "candidate_windows": candidate_windows,
            "candidate_mask_blocked_windows": candidate_blocked,
            "mask_unparseable_windows": mask_unparseable,
            "selected_raw_measurements": selected_raw,
            "selected_applied_measurements": selected_applied,
            "raw_action_missing": raw_missing,
            "logged_invalid_measurement_action": invalid_action,
            "transition_invalidated_windows": transition_invalidated,
            "budget_exhausted_windows": budget_exhausted,
            "none_due_budget_windows": none_due_budget,
            "budget_over_limit": budget_over_limit,
            "budget_consumed": budget_consumed,
            "raw_rate": (
                float(selected_raw / decision_rows) if decision_rows else float("nan")
            ),
            "applied_rate": (
                float(selected_applied / decision_rows) if decision_rows else float("nan")
            ),
            "candidate_fraction": (
                float(candidate_windows / decision_rows) if decision_rows else float("nan")
            ),
            "candidate_blocked_fraction": (
                float(candidate_blocked / decision_rows) if decision_rows else float("nan")
            ),
            "budget_exhausted_fraction": (
                float(budget_exhausted / decision_rows) if decision_rows else float("nan")
            ),
        }
    return output


def _quality_episode_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    uav_count: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    none_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        uav_id = str(row.get("uav_id", ""))
        episode_id = str(row.get("episode_id", ""))
        if uav_id not in _uav_ids(uav_count) or not episode_id:
            continue
        none_probability = _float(row.get("none_probability"), float("nan"))
        if not math.isfinite(none_probability) or not 0.0 <= none_probability <= 1.0:
            continue
        grouped[(episode_id, uav_id)].append(1.0 - none_probability)
        none_values[(episode_id, uav_id)].append(none_probability)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for key, values in grouped.items():
        output[key] = {
            "expected_mean": float(np.mean(values)),
            "expected_std": float(np.std(values)),
            "none_probability_mean": float(np.mean(none_values[key])),
            "rows": len(values),
        }
    return output


def _quality_aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    uav_count: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        uav_id = str(row.get("uav_id", ""))
        if uav_id not in _uav_ids(uav_count):
            continue
        none_probability = _float(row.get("none_probability"), float("nan"))
        expected = 1.0 - none_probability
        if math.isfinite(expected) and 0.0 <= expected <= 1.0:
            grouped[uav_id].append(expected)
    return {
        uav_id: {
            "expected_mean": float(np.mean(values)) if values else float("nan"),
            "expected_std": float(np.std(values)) if values else float("nan"),
            "expected_rows": len(values),
        }
        for uav_id, values in grouped.items()
    }


def _fit_targets(
    reference_root: Path,
    seeds: Sequence[int],
    *,
    uav_count: int,
) -> tuple[dict[tuple[int, str], float], list[dict[str, Any]], dict[str, Any]]:
    targets: dict[tuple[int, str], float] = {}
    rows: list[dict[str, Any]] = []
    source_meta: dict[str, Any] = {}
    for seed in seeds:
        scale_root = _scale_root(reference_root, seed, "baseline_mappo")
        summaries = _action_summary(scale_root, "train", uav_count=uav_count)
        source_meta[str(seed)] = {}
        for uav_id in _uav_ids(uav_count):
            summary = summaries[uav_id]
            windows = int(summary["decision_windows"])
            if windows <= 0:
                raise ValueError(f"no baseline train windows for seed={seed}, uav={uav_id}")
            applied_target = float(summary["applied_rate"])
            raw_target = float(summary["raw_rate"])
            targets[(int(seed), uav_id)] = applied_target
            rows.append(
                {
                    "base_seed": int(seed),
                    "uav_id": uav_id,
                    "baseline_train_decision_windows": windows,
                    "baseline_train_applied_selected_measurements": int(
                        summary["selected_applied_measurements"]
                    ),
                    "baseline_train_raw_selected_measurements": int(
                        summary["selected_raw_measurements"]
                    ),
                    "constraint_target": applied_target,
                    "actor_target": raw_target,
                    "raw_action_missing": int(summary["raw_action_missing"]),
                    "source_split": "train",
                    "source_branch": "baseline_mappo",
                    "recomputed_from": str(scale_root / "train"),
                }
            )
            source_meta[str(seed)][uav_id] = {
                "decision_windows": windows,
                "applied_selected": int(summary["selected_applied_measurements"]),
                "raw_selected": int(summary["selected_raw_measurements"]),
                "constraint_target": applied_target,
                "actor_target": raw_target,
            }
    return targets, rows, source_meta


def _merge_manifest_targets(
    reference_root: Path,
    target_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    uav_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatches: list[str] = []
    for row in target_rows:
        seed = int(row["base_seed"])
        uav_id = str(row["uav_id"])
        config = _manifest_config(_scale_root(reference_root, seed, RATE_BRANCH))
        manifest_targets = config.get("measurement_rate_target_by_uav")
        manifest_target = (
            _float(manifest_targets.get(uav_id), float("nan"))
            if isinstance(manifest_targets, Mapping)
            else float("nan")
        )
        constraint_target = _float(row.get("constraint_target"), float("nan"))
        available = int(math.isfinite(manifest_target))
        if not available:
            missing.append(f"{seed}/{uav_id}")
        if available and abs(manifest_target - constraint_target) > 1e-12:
            mismatches.append(f"{seed}/{uav_id}")
        output.append(
            {
                **dict(row),
                "logged_constraint_target": manifest_target,
                "constraint_target_available": available,
                "constraint_target_match": int(
                    available and abs(manifest_target - constraint_target) <= 1e-12
                ),
            }
        )
    return output, {
        "missing_manifest_targets": missing,
        "manifest_target_mismatches": mismatches,
        "complete": len(output) == len(seeds) * len(_uav_ids(uav_count))
        and not missing
        and not mismatches,
    }


def _logged_target_status(
    source_logs: Path,
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_logs / "measurement_rate_targets.csv"
    fields, logged = _read_csv_with_fields(path)
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    duplicates: list[str] = []
    if "base_seed" in fields:
        for row in logged:
            key = (_int(row.get("base_seed"), -1), str(row.get("uav_id", "")))
            if key in by_key:
                duplicates.append(f"{key[0]}/{key[1]}")
            by_key[key] = row
    output: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in target_rows:
        key = (int(row["base_seed"]), str(row["uav_id"]))
        logged_row = by_key.get(key)
        if not path.exists():
            status = "missing_file"
            available = 0
            match = 0
        elif "base_seed" not in fields:
            status = "missing_base_seed_column"
            available = 0
            match = 0
        elif logged_row is None:
            status = "missing_seed_uav_row"
            available = 0
            match = 0
        else:
            selected_match = _int(
                logged_row.get("baseline_train_selected_measurements"),
                -1,
            ) == int(row["baseline_train_applied_selected_measurements"])
            windows_match = _int(
                logged_row.get("baseline_train_decision_windows"),
                -1,
            ) == int(row["baseline_train_decision_windows"])
            rate_match = abs(
                _float(logged_row.get("target_measurement_rate"), float("nan"))
                - _float(row.get("constraint_target"), float("nan"))
            ) <= 1e-12
            source_match = (
                str(logged_row.get("source_split", "")) == "train"
                and str(logged_row.get("source_branch", "")) == "baseline_mappo"
            )
            match = int(selected_match and windows_match and rate_match and source_match)
            status = "match" if match else "value_or_source_mismatch"
            available = 1
            if not match:
                mismatches.append(f"{key[0]}/{key[1]}")
        output.append(
            {
                **dict(row),
                "logged_target_file": str(path),
                "logged_target_available": available,
                "logged_target_match": match,
                "logged_target_status": status,
            }
        )
    return output, {
        "file_exists": bool(path.exists()),
        "fields": fields,
        "row_count": len(logged),
        "has_base_seed": "base_seed" in fields,
        "duplicate_seed_uav_keys": duplicates,
        "value_or_source_mismatches": mismatches,
        "complete_seed_bound_match": bool(output)
        and all(int(row["logged_target_match"]) == 1 for row in output),
    }


def _branch_isolation(
    reference_root: Path,
    seeds: Sequence[int],
    *,
    uav_count: int,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    all_ok = True
    mismatches: list[str] = []
    for seed in seeds:
        for branch in BRANCHES:
            scale_root = _scale_root(reference_root, seed, branch)
            config = _manifest_config(scale_root)
            quality_rows = _quality_rows(scale_root, split)
            quality_flags = sorted(
                {_int(row.get("quality_feature_enabled"), -1) for row in quality_rows}
            )
            budget_flags = sorted(
                {
                    _int(row.get("measurement_budget_feature_enabled"), -1)
                    for row in quality_rows
                }
            )
            constraint_flags = sorted(
                {
                    _int(row.get("measurement_rate_constraint_enabled"), -1)
                    for row in quality_rows
                }
            )
            expected_quality = int(branch in QUALITY_BRANCHES)
            expected_budget = int(branch in QUALITY_BRANCHES)
            expected_constraint = int(branch == RATE_BRANCH)
            checks = {
                "variant": str(config.get("mappo_variant", "")) == branch,
                "quality": int(
                    bool(_float(config.get("measurement_quality_logit_penalty"), 0.0) > 0.0)
                )
                == expected_quality,
                "budget": int(bool(config.get("measurement_budget_enabled", False)))
                == expected_budget,
                "constraint": int(
                    bool(config.get("measurement_rate_constraint_enabled", False))
                )
                == expected_constraint,
                "quality_log": not quality_flags
                or quality_flags == [expected_quality],
                "budget_log": not budget_flags or budget_flags == [expected_budget],
                "constraint_log": not constraint_flags
                or constraint_flags == [expected_constraint],
            }
            ok = bool(all(checks.values()))
            all_ok = all_ok and ok
            if not ok:
                mismatches.append(f"{seed}/{branch}")
            output.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "split": split,
                    "manifest_ok": int(ok),
                    "mappo_variant": str(config.get("mappo_variant", "")),
                    "quality_feature_enabled": expected_quality,
                    "measurement_budget_enabled": expected_budget,
                    "measurement_rate_constraint_enabled": expected_constraint,
                    "quality_log_flags": quality_flags,
                    "budget_log_flags": budget_flags,
                    "constraint_log_flags": constraint_flags,
                    "quality_rows": len(quality_rows),
                    "source_manifest": str(scale_root / "manifest.json"),
                }
            )
    return output, {
        "all_ok": all_ok,
        "mismatches": mismatches,
    }


def _per_uav_rows(
    reference_root: Path,
    seeds: Sequence[int],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    uav_count: int,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_by_key = {
        (int(row["base_seed"]), str(row["uav_id"])): row for row in target_rows
    }
    per_uav: list[dict[str, Any]] = []
    shift: list[dict[str, Any]] = []
    branch_meta: dict[str, Any] = {}
    for seed in seeds:
        seed_shift_rows: list[dict[str, Any]] = []
        for branch in BRANCHES:
            scale_root = _scale_root(reference_root, seed, branch)
            train_quality = _quality_aggregate(
                _quality_rows(scale_root, "train"), uav_count=uav_count
            )
            validation_quality = _quality_aggregate(
                _quality_rows(scale_root, split), uav_count=uav_count
            )
            train_actions = _action_summary(scale_root, "train", uav_count=uav_count)
            validation_actions = _action_summary(
                scale_root, split, uav_count=uav_count
            )
            branch_meta[f"{seed}/{branch}"] = {
                "train_quality_rows": sum(
                    int(item["expected_rows"]) for item in train_quality.values()
                ),
                "validation_quality_rows": sum(
                    int(item["expected_rows"]) for item in validation_quality.values()
                ),
                "train_action_rows": sum(
                    int(item["decision_windows"]) for item in train_actions.values()
                ),
                "validation_action_rows": sum(
                    int(item["decision_windows"]) for item in validation_actions.values()
                ),
            }
            for uav_id in _uav_ids(uav_count):
                target = target_by_key[(int(seed), uav_id)]
                train_q = train_quality.get(uav_id, {})
                val_q = validation_quality.get(uav_id, {})
                train_a = train_actions[uav_id]
                val_a = validation_actions[uav_id]
                constraint_target = _float(target.get("constraint_target"), float("nan"))
                train_expected = _float(train_q.get("expected_mean"), float("nan"))
                val_expected = _float(val_q.get("expected_mean"), float("nan"))
                per_uav.append(
                    {
                        "base_seed": int(seed),
                        "branch": branch,
                        "uav_id": uav_id,
                        "train_expected_non_none_rate": train_expected,
                        "validation_expected_non_none_rate": val_expected,
                        "constraint_target": constraint_target,
                        "actor_target": _float(target.get("actor_target"), float("nan")),
                        "train_raw_action_rate": train_a["raw_rate"],
                        "train_applied_action_rate": train_a["applied_rate"],
                        "validation_raw_action_rate": val_a["raw_rate"],
                        "validation_applied_action_rate": val_a["applied_rate"],
                        "validation_none_probability_mean": (
                            1.0 - val_expected if math.isfinite(val_expected) else float("nan")
                        ),
                        "train_expected_minus_target": (
                            train_expected - constraint_target
                            if math.isfinite(train_expected) and math.isfinite(constraint_target)
                            else float("nan")
                        ),
                        "validation_expected_minus_target": (
                            val_expected - constraint_target
                            if math.isfinite(val_expected) and math.isfinite(constraint_target)
                            else float("nan")
                        ),
                        "validation_raw_minus_expected": (
                            val_a["raw_rate"] - val_expected
                            if math.isfinite(val_expected)
                            else float("nan")
                        ),
                        "validation_applied_minus_raw": (
                            val_a["applied_rate"] - val_a["raw_rate"]
                        ),
                        "train_decision_windows": int(train_a["decision_windows"]),
                        "validation_decision_windows": int(val_a["decision_windows"]),
                        "validation_eligible_candidate_windows": int(
                            val_a["eligible_candidate_windows"]
                        ),
                        "validation_candidate_blocked_fraction": val_a[
                            "candidate_blocked_fraction"
                        ],
                        "validation_mask_unparseable_windows": int(
                            val_a["mask_unparseable_windows"]
                        ),
                        "validation_budget_exhausted_fraction": val_a[
                            "budget_exhausted_fraction"
                        ],
                        "validation_none_due_budget_windows": int(
                            val_a["none_due_budget_windows"]
                        ),
                        "validation_transition_invalidated_windows": int(
                            val_a["transition_invalidated_windows"]
                        ),
                        "validation_invalid_action_count": int(
                            val_a["logged_invalid_measurement_action"]
                        ),
                        "validation_raw_action_missing": int(val_a["raw_action_missing"]),
                        "quality_rows_train": int(train_q.get("expected_rows", 0)),
                        "quality_rows_validation": int(val_q.get("expected_rows", 0)),
                    }
                )
                if branch == RATE_BRANCH:
                    tolerance = 0.05
                    config = _manifest_config(scale_root)
                    tolerance = _float(
                        config.get("measurement_rate_tolerance"), tolerance
                    )
                    seed_shift_rows.append(
                        {
                            "base_seed": int(seed),
                            "uav_id": uav_id,
                            "train_expected_non_none_rate": train_expected,
                            "validation_expected_non_none_rate": val_expected,
                            "constraint_target": constraint_target,
                            "actor_target": _float(
                                target.get("actor_target"), float("nan")
                            ),
                            "train_expected_minus_target": (
                                train_expected - constraint_target
                                if math.isfinite(train_expected)
                                and math.isfinite(constraint_target)
                                else float("nan")
                            ),
                            "validation_expected_minus_target": (
                                val_expected - constraint_target
                                if math.isfinite(val_expected)
                                and math.isfinite(constraint_target)
                                else float("nan")
                            ),
                            "validation_minus_train_expected": (
                                val_expected - train_expected
                                if math.isfinite(val_expected)
                                and math.isfinite(train_expected)
                                else float("nan")
                            ),
                            "train_close_within_tolerance": int(
                                math.isfinite(train_expected)
                                and math.isfinite(constraint_target)
                                and abs(train_expected - constraint_target) <= tolerance
                            ),
                            "validation_close_within_tolerance": int(
                                math.isfinite(val_expected)
                                and math.isfinite(constraint_target)
                                and abs(val_expected - constraint_target) <= tolerance
                            ),
                            "tolerance": tolerance,
                            "validation_raw_action_rate": val_a["raw_rate"],
                            "validation_applied_action_rate": val_a["applied_rate"],
                            "validation_candidate_blocked_fraction": val_a[
                                "candidate_blocked_fraction"
                            ],
                            "validation_budget_exhausted_fraction": val_a[
                                "budget_exhausted_fraction"
                            ],
                            "validation_decision_windows": int(
                                val_a["decision_windows"]
                            ),
                            "validation_eligible_candidate_windows": int(
                                val_a["eligible_candidate_windows"]
                            ),
                        }
                    )
        shift.extend(seed_shift_rows)
    return per_uav, shift, branch_meta


def _rate_aggregation_source_audit() -> dict[str, Any]:
    """Check the implemented update path without executing a training step.

    The persisted CSV contains per-minibatch aggregate losses and separate
    per-UAV summary means.  Because an incomplete final minibatch can contain
    different numbers of samples for different UAVs, averaging the three
    summary columns is not a valid reconstruction of the aggregate loss.  We
    therefore audit the source-level grouping and use the persisted aggregate
    plus its weighted closure for the data-level check.
    """

    source = Path(__file__).resolve().parents[1] / "learning" / "mappo.py"
    markers = {
        "group_probabilities_by_uav": 'measurement_probability_values.setdefault(str(record["uav_id"]), []).append',
        "mean_probability_within_uav": "mean_probability = torch.stack(probabilities).mean()",
        "mean_penalty_across_active_uavs": "rate_constraint_loss = torch.stack(penalties).mean()",
        "weighted_constraint_added_to_loss": "self.config.measurement_rate_constraint_coef * rate_constraint_loss",
    }
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return {
            "source_file": str(source),
            "source_readable": False,
            "source_error": f"{type(exc).__name__}: {exc}",
            "markers": {name: False for name in markers},
            "source_aggregation_ok": False,
        }
    found = {name: marker in text for name, marker in markers.items()}
    return {
        "source_file": str(source),
        "source_readable": True,
        "source_error": "",
        "markers": found,
        "source_aggregation_ok": bool(all(found.values())),
        "interpretation": (
            "probabilities are grouped by UAV, averaged within each active UAV, "
            "then active-UAV penalties are averaged; per-UAV CSV summary means "
            "are not an exact reconstruction when minibatch coverage differs"
        ),
    }


def _training_aggregation_rows(
    source_logs: Path,
    reference_root: Path,
    seeds: Sequence[int],
    *,
    uav_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_logs / "training_rate_constraint.csv"
    logged = [
        row
        for row in _read_csv(path)
        if str(row.get("phase", "train")) == "train"
        and str(row.get("treatment", "")) == RATE_BRANCH
        and _int(row.get("base_seed"), -1) in set(int(seed) for seed in seeds)
    ]
    output: list[dict[str, Any]] = []
    missing_rows: list[str] = []
    for seed in seeds:
        seed_rows = [
            row for row in logged if _int(row.get("base_seed"), -1) == int(seed)
        ]
        config = _manifest_config(_scale_root(reference_root, seed, RATE_BRANCH))
        coefficient = _float(config.get("measurement_rate_constraint_coef"), float("nan"))
        tolerance = _float(config.get("measurement_rate_tolerance"), float("nan"))
        if not seed_rows:
            missing_rows.append(str(seed))
            output.append(
                {
                    "base_seed": int(seed),
                    "update": "",
                    "episode_id": "",
                    "constraint_enabled": 0,
                    "aggregation_ok": 0,
                    "missing_training_log": 1,
                }
            )
            continue
        for row in seed_rows:
            expected_values: list[float] = []
            target_values: list[float] = []
            upper_values: list[float] = []
            lower_values: list[float] = []
            raw_loss_values: list[float] = []
            active_values: list[float] = []
            fields_complete = True
            for uav_id in _uav_ids(uav_count):
                values = [
                    _float(row.get(f"measurement_rate_expected_{uav_id}"), float("nan")),
                    _float(row.get(f"measurement_rate_target_{uav_id}"), float("nan")),
                    _float(row.get(f"measurement_rate_upper_residual_{uav_id}"), float("nan")),
                    _float(row.get(f"measurement_rate_lower_residual_{uav_id}"), float("nan")),
                    _float(row.get(f"measurement_rate_raw_loss_{uav_id}"), float("nan")),
                    _float(row.get(f"measurement_rate_active_fraction_{uav_id}"), float("nan")),
                ]
                if not all(math.isfinite(value) for value in values):
                    fields_complete = False
                expected_values.append(values[0])
                target_values.append(values[1])
                upper_values.append(values[2])
                lower_values.append(values[3])
                raw_loss_values.append(values[4])
                active_values.append(values[5])
            aggregate_logged = _float(
                row.get("measurement_rate_constraint_loss"), float("nan")
            )
            weighted_logged = _float(
                row.get("measurement_rate_constraint_weighted_loss"), float("nan")
            )
            expected_mean_logged = _float(
                row.get("measurement_rate_expected_mean"), float("nan")
            )
            upper_mean_logged = _float(
                row.get("measurement_rate_upper_residual_mean"), float("nan")
            )
            lower_mean_logged = _float(
                row.get("measurement_rate_lower_residual_mean"), float("nan")
            )
            active_mean_logged = _float(
                row.get("measurement_rate_constraint_active_fraction"), float("nan")
            )
            recomputed_aggregate = (
                float(np.mean(raw_loss_values))
                if fields_complete
                else float("nan")
            )
            recomputed_weighted = (
                coefficient * recomputed_aggregate
                if math.isfinite(coefficient) and math.isfinite(recomputed_aggregate)
                else float("nan")
            )
            logged_weighted_recomputed = (
                coefficient * aggregate_logged
                if math.isfinite(coefficient) and math.isfinite(aggregate_logged)
                else float("nan")
            )
            recomputed_expected_mean = (
                float(np.mean(expected_values)) if fields_complete else float("nan")
            )
            recomputed_upper_mean = (
                float(np.mean(upper_values)) if fields_complete else float("nan")
            )
            recomputed_lower_mean = (
                float(np.mean(lower_values)) if fields_complete else float("nan")
            )
            recomputed_active_mean = (
                float(np.mean(active_values)) if fields_complete else float("nan")
            )
            formula_losses = [
                max(expected - target - tolerance, 0.0) ** 2
                + max(target - tolerance - expected, 0.0) ** 2
                if fields_complete
                else float("nan")
                for expected, target in zip(expected_values, target_values)
            ]
            formula_mean = (
                float(np.mean(formula_losses)) if fields_complete else float("nan")
            )
            target_match = bool(
                fields_complete
                and all(
                    abs(
                        target
                        - _float(
                            (
                                config.get("measurement_rate_target_by_uav")
                                if isinstance(
                                    config.get("measurement_rate_target_by_uav"),
                                    Mapping,
                                )
                                else {}
                            ).get(uav_id),
                            float("nan"),
                        )
                    )
                    <= 1e-12
                    for target, uav_id in zip(target_values, _uav_ids(uav_count))
                )
            )
            gaps = {
                "aggregate_loss_gap": (
                    abs(aggregate_logged - recomputed_aggregate)
                    if math.isfinite(aggregate_logged)
                    and math.isfinite(recomputed_aggregate)
                    else float("nan")
                ),
                "weighted_loss_gap": (
                    abs(weighted_logged - logged_weighted_recomputed)
                    if math.isfinite(weighted_logged)
                    and math.isfinite(logged_weighted_recomputed)
                    else float("nan")
                ),
                "expected_mean_gap": (
                    abs(expected_mean_logged - recomputed_expected_mean)
                    if math.isfinite(expected_mean_logged)
                    and math.isfinite(recomputed_expected_mean)
                    else float("nan")
                ),
                "upper_mean_gap": (
                    abs(upper_mean_logged - recomputed_upper_mean)
                    if math.isfinite(upper_mean_logged)
                    and math.isfinite(recomputed_upper_mean)
                    else float("nan")
                ),
                "lower_mean_gap": (
                    abs(lower_mean_logged - recomputed_lower_mean)
                    if math.isfinite(lower_mean_logged)
                    and math.isfinite(recomputed_lower_mean)
                    else float("nan")
                ),
                "active_mean_gap": (
                    abs(active_mean_logged - recomputed_active_mean)
                    if math.isfinite(active_mean_logged)
                    and math.isfinite(recomputed_active_mean)
                    else float("nan")
                ),
                "formula_mean_gap_from_logged_expected": (
                    abs(aggregate_logged - formula_mean)
                    if math.isfinite(aggregate_logged) and math.isfinite(formula_mean)
                    else float("nan")
                ),
            }
            per_uav_values_valid = bool(
                fields_complete
                and all(value >= -1e-12 for value in raw_loss_values)
                and all(value >= -1e-12 for value in upper_values)
                and all(value >= -1e-12 for value in lower_values)
                and all(-1e-12 <= value <= 1.0 + 1e-12 for value in active_values)
            )
            finite_gaps = [value for value in gaps.values() if math.isfinite(value)]
            aggregation_ok = bool(
                fields_complete
                and _int(row.get("measurement_rate_constraint_enabled"), 0) == 1
                and target_match
                and math.isfinite(aggregate_logged)
                and math.isfinite(weighted_logged)
                and math.isfinite(coefficient)
                and coefficient >= 0.0
                and finite_gaps
                and gaps["weighted_loss_gap"] <= 1e-8
                and per_uav_values_valid
            )
            output.append(
                {
                    "base_seed": int(seed),
                    "update": _int(row.get("update"), 0),
                    "episode_id": str(row.get("episode_id", row.get("episode", ""))),
                    "constraint_enabled": _int(
                        row.get("measurement_rate_constraint_enabled"), 0
                    ),
                    "coefficient": coefficient,
                    "tolerance": tolerance,
                    "samples": _float(row.get("samples"), float("nan")),
                    "logged_constraint_loss": aggregate_logged,
                    "recomputed_constraint_loss": recomputed_aggregate,
                    "logged_weighted_loss": weighted_logged,
                    "logged_weighted_loss_recomputed_from_logged_aggregate": logged_weighted_recomputed,
                    "recomputed_weighted_loss": recomputed_weighted,
                    "logged_expected_mean": expected_mean_logged,
                    "recomputed_expected_mean": recomputed_expected_mean,
                    "logged_upper_residual_mean": upper_mean_logged,
                    "logged_lower_residual_mean": lower_mean_logged,
                    "logged_active_fraction": active_mean_logged,
                    "recomputed_active_fraction": recomputed_active_mean,
                    "formula_mean_from_logged_expected": formula_mean,
                    "target_fields_match_manifest": int(target_match),
                    "per_uav_fields_complete": int(fields_complete),
                    "per_uav_values_valid": int(per_uav_values_valid),
                    "per_uav_summary_reconstruction_is_exact": 0,
                    "per_uav_summary_reconstruction_note": (
                        "not exact in general: aggregate is averaged per minibatch, "
                        "while per-UAV fields are separate means"
                    ),
                    "aggregation_ok": int(aggregation_ok),
                    "missing_training_log": 0,
                    **gaps,
                }
            )
    return output, {
        "source_file": str(path),
        "logged_row_count": len(logged),
        "missing_seed_rows": missing_rows,
        "aggregation_ok_rows": sum(int(row.get("aggregation_ok", 0)) for row in output),
        "complete_rows": len(output) == len(logged) + len(missing_rows),
        "source_audit": _rate_aggregation_source_audit(),
        "data_closure_basis": (
            "logged aggregate loss equals coefficient-weighted loss; per-UAV "
            "summary means are validated for finiteness/non-negativity and are "
            "not naively averaged to reconstruct a minibatch-weighted aggregate"
        ),
    }


def _json_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(value: Any, length: int, fill: float) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        array = np.empty(0, dtype=np.float32)
    output = np.full(length, fill, dtype=np.float32)
    if array.size:
        size = min(length, array.size)
        output[:size] = array[:size]
    output = np.nan_to_num(output, nan=fill, posinf=fill, neginf=fill)
    return output


def _graph_observation(row: Mapping[str, Any], quality_row: Mapping[str, Any], uav_count: int) -> dict[str, Any]:
    def parsed(name: str, default: Any) -> Any:
        return _parse_json_field(row.get(name, ""), default)

    action_mask = parsed("action_mask_json", {"measurement": [True] * (uav_count + 1)})
    mask_values = (
        list(action_mask.get("measurement", []))
        if isinstance(action_mask, Mapping)
        else [True] * (uav_count + 1)
    )
    candidate_mask = np.asarray(
        [bool(item) for item in mask_values[:uav_count]],
        dtype=np.float32,
    )
    quality_penalty = _finite_array(
        _parse_json_field(quality_row.get("measurement_quality_penalty", []), []),
        uav_count,
        0.0,
    )
    coefficient = _float(
        quality_row.get("measurement_quality_logit_penalty"), 1.0
    )
    risk = np.clip(
        quality_penalty / max(coefficient, 1e-8),
        0.0,
        1.0,
    )
    valid = candidate_mask.copy()
    return {
        "node_features": parsed("node_features_json", []),
        "effective_edge_index": parsed("effective_edge_index_json", []),
        "effective_edge_features": parsed("effective_edge_features_json", []),
        "candidate_edge_features": parsed("candidate_edge_features_json", {}),
        "candidate_incoming_edge_features": parsed(
            "candidate_incoming_edge_features_json", {}
        ),
        "potential_edge_features": parsed("potential_edge_features_json", {}),
        "action_mask": mask_values,
        "bandwidth_mask": parsed("bandwidth_mask_json", [True] * uav_count),
        "measurement_quality_risk": risk.tolist(),
        "measurement_quality_valid": valid.tolist(),
        "measurement_budget_remaining_ratio": _float(
            quality_row.get("measurement_budget_remaining_ratio"), 1.0
        ),
    }


def _gradient_norm(parameters: Iterable[Any]) -> float:
    values: list[float] = []
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        number = float(gradient.detach().norm().cpu())
        if math.isfinite(number):
            values.append(number * number)
    return float(math.sqrt(sum(values))) if values else 0.0


def _gradient_probe_for_seed(
    reference_root: Path,
    seed: int,
    *,
    uav_count: int,
) -> dict[str, Any]:
    checkpoint = (
        _scale_root(reference_root, seed, RATE_BRANCH)
        / "checkpoints"
        / "final.pt"
    )
    result: dict[str, Any] = {
        "base_seed": int(seed),
        "checkpoint": str(checkpoint),
        "probe_ok": 0,
        "error": "",
    }
    if not checkpoint.exists():
        result["error"] = "missing_checkpoint"
        return result
    before_hash = _json_hash(checkpoint)
    try:
        import torch

        from ..learning.mappo import MAPPOConfig, MAPPOPolicy

        # Checkpoints in this repository are trusted local experiment
        # artifacts and contain pathlib/configuration objects in addition to
        # tensors.  PyTorch 2.6 changed the default to weights_only=True,
        # which rejects those persisted configuration objects.  The audit
        # never executes an optimizer step; it only restores the policy for a
        # read-only autograd dry-run.
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with older PyTorch versions
            payload = torch.load(checkpoint, map_location="cpu")
        trainer_config = dict(payload["trainer_config"])
        policy_config = MAPPOConfig(**trainer_config)
        policy = MAPPOPolicy(policy_config, uav_ids=_uav_ids(uav_count))
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        policy.eval()
        quality_rows = _quality_rows(
            _scale_root(reference_root, seed, RATE_BRANCH), "train"
        )
        quality_by_key = {
            (
                str(row.get("episode_id", "")),
                _int(row.get("step"), 0),
                str(row.get("uav_id", "")),
            ): row
            for row in quality_rows
        }
        observations: dict[str, list[dict[str, Any]]] = {
            uav_id: [] for uav_id in _uav_ids(uav_count)
        }
        train_root = _scale_root(reference_root, seed, RATE_BRANCH) / "train"
        for episode_root in sorted(path for path in train_root.iterdir() if path.is_dir()):
            graph_path = episode_root / "graph_features.csv"
            if not graph_path.exists():
                continue
            for row in _read_csv(graph_path):
                uav_id = str(row.get("uav_id", ""))
                if uav_id not in observations or len(observations[uav_id]) >= 8:
                    continue
                key = (
                    str(row.get("episode_id", episode_root.name)),
                    _int(row.get("step"), 0),
                    uav_id,
                )
                quality_row = quality_by_key.get(key)
                if quality_row is None:
                    continue
                observations[uav_id].append(
                    _graph_observation(row, quality_row, uav_count)
                )
            if all(len(values) >= 8 for values in observations.values()):
                break
        missing = [uav_id for uav_id, values in observations.items() if not values]
        if missing:
            result["error"] = f"missing_train_graph_observations:{','.join(missing)}"
            return result
        targets = policy_config.measurement_rate_target_by_uav or {}
        tolerance = float(policy_config.measurement_rate_tolerance)
        penalties: list[Any] = []
        expected_by_uav: dict[str, float] = {}
        active_by_uav: dict[str, int] = {}
        for uav_id in _uav_ids(uav_count):
            probabilities = [
                policy.measurement_probability(observation)
                for observation in observations[uav_id]
            ]
            mean_probability = torch.stack(probabilities).mean()
            target = float(targets[uav_id])
            upper = torch.relu(mean_probability - target - tolerance)
            lower = torch.relu(target - tolerance - mean_probability)
            penalty = upper.pow(2) + lower.pow(2)
            penalties.append(penalty)
            expected_by_uav[uav_id] = float(mean_probability.detach().cpu())
            active_by_uav[uav_id] = int(float(penalty.detach().cpu()) > 1e-12)
        loss = torch.stack(penalties).mean()
        policy.zero_grad(set_to_none=True)
        loss.backward()
        weight_gradient = policy.measurement_head.weight.grad
        quality_dim = (
            2 * uav_count if policy_config.use_measurement_quality_features else 0
        )
        actor_dim = int(policy_config.actor_hidden_dim)
        quality_slice_norm = 0.0
        if weight_gradient is not None and quality_dim:
            quality_slice = weight_gradient[:, actor_dim : actor_dim + quality_dim]
            quality_slice_norm = float(quality_slice.detach().norm().cpu())
        parameter_digest_before = hashlib.sha256(
            b"".join(
                parameter.detach().cpu().numpy().tobytes()
                for _, parameter in sorted(policy.named_parameters())
            )
        ).hexdigest()
        result.update(
            {
                "probe_ok": 1,
                "error": "",
                "checkpoint_hash_before": before_hash,
                "checkpoint_hash_after": _json_hash(checkpoint),
                "checkpoint_unchanged": int(before_hash == _json_hash(checkpoint)),
                "parameter_digest_after_backward": parameter_digest_before,
                "loss": float(loss.detach().cpu()),
                "positive_residual": int(float(loss.detach().cpu()) > 1e-12),
                "expected_rate_uav": expected_by_uav,
                "active_uav_count": int(sum(active_by_uav.values())),
                "measurement_head_gradient_norm": _gradient_norm(
                    policy.measurement_head.parameters()
                ),
                "quality_input_gradient_norm": quality_slice_norm,
                "actor_body_gradient_norm": _gradient_norm(policy.actor_body.parameters()),
                "gat_gradient_norm": _gradient_norm(policy.gat.parameters()),
                "all_gradient_fields_finite": all(
                    math.isfinite(_float(value, float("nan")))
                    for value in (
                        loss.detach().cpu(),
                        _gradient_norm(policy.measurement_head.parameters()),
                        quality_slice_norm,
                        _gradient_norm(policy.actor_body.parameters()),
                        _gradient_norm(policy.gat.parameters()),
                    )
                ),
                "observations_per_uav": {
                    uav_id: len(values) for uav_id, values in observations.items()
                },
            }
        )
        result["quality_gradient_nonzero_when_active"] = int(
            not result["positive_residual"]
            or result["quality_input_gradient_norm"] > 1e-12
        )
        result["head_gradient_nonzero_when_active"] = int(
            not result["positive_residual"]
            or result["measurement_head_gradient_norm"] > 1e-12
        )
        result["parameter_digest_before_backward"] = parameter_digest_before
        return result
    except Exception as exc:  # pragma: no cover - exercised by real artifacts
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result.setdefault("checkpoint_hash_before", before_hash)
        result.setdefault("checkpoint_hash_after", _json_hash(checkpoint))
        result.setdefault(
            "checkpoint_unchanged",
            int(result["checkpoint_hash_before"] == result["checkpoint_hash_after"]),
        )


def _mean(values: Iterable[Any]) -> float:
    numbers = np.asarray(
        [_float(value, float("nan")) for value in values], dtype=float
    )
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float("nan")


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=float,
    )
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _seed_summary(
    seeds: Sequence[int],
    per_uav_rows: Sequence[Mapping[str, Any]],
    shift_rows: Sequence[Mapping[str, Any]],
    aggregation_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    isolation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in seeds:
        rate_rows = [
            row
            for row in per_uav_rows
            if int(row["base_seed"]) == int(seed) and row["branch"] == RATE_BRANCH
        ]
        shifts = [
            row for row in shift_rows if int(row["base_seed"]) == int(seed)
        ]
        aggregations = [
            row for row in aggregation_rows if int(row["base_seed"]) == int(seed)
        ]
        gradient = next(
            (row for row in gradient_rows if int(row["base_seed"]) == int(seed)),
            {},
        )
        isolation = [
            row for row in isolation_rows if int(row["base_seed"]) == int(seed)
        ]
        tolerance = _mean(row.get("tolerance") for row in shifts)
        train_close = int(
            bool(shifts)
            and all(_int(row.get("train_close_within_tolerance"), 0) == 1 for row in shifts)
        )
        validation_close = int(
            bool(shifts)
            and all(
                _int(row.get("validation_close_within_tolerance"), 0) == 1
                for row in shifts
            )
        )
        mask_fraction = _mean(
            row.get("validation_candidate_blocked_fraction") for row in shifts
        )
        budget_fraction = _mean(
            row.get("validation_budget_exhausted_fraction") for row in shifts
        )
        expected_train = _mean(
            row.get("train_expected_non_none_rate") for row in shifts
        )
        expected_validation = _mean(
            row.get("validation_expected_non_none_rate") for row in shifts
        )
        target = _mean(row.get("constraint_target") for row in shifts)
        output.append(
            {
                "base_seed": int(seed),
                "uav_count": len(rate_rows),
                "rate_branch_train_expected_mean": expected_train,
                "rate_branch_validation_expected_mean": expected_validation,
                "rate_branch_constraint_target_mean": target,
                "rate_branch_train_abs_gap_mean": _mean(
                    abs(_float(row.get("train_expected_minus_target"), float("nan")))
                    for row in shifts
                ),
                "rate_branch_validation_abs_gap_mean": _mean(
                    abs(_float(row.get("validation_expected_minus_target"), float("nan")))
                    for row in shifts
                ),
                "train_all_uav_close_within_tolerance": train_close,
                "validation_all_uav_close_within_tolerance": validation_close,
                "tolerance": tolerance,
                "validation_candidate_blocked_fraction_mean": mask_fraction,
                "validation_budget_exhausted_fraction_mean": budget_fraction,
                "validation_mask_or_budget_fraction_mean": (
                    mask_fraction + budget_fraction
                    if math.isfinite(mask_fraction) and math.isfinite(budget_fraction)
                    else float("nan")
                ),
                "aggregation_rows": len(aggregations),
                "aggregation_ok_fraction": _mean(
                    row.get("aggregation_ok") for row in aggregations
                ),
                "aggregation_max_loss_gap": max(
                    [
                        _float(row.get("aggregate_loss_gap"), 0.0)
                        for row in aggregations
                        if math.isfinite(_float(row.get("aggregate_loss_gap"), float("nan")))
                    ]
                    or [float("nan")]
                ),
                "gradient_probe_ok": _int(gradient.get("probe_ok"), 0),
                "gradient_positive_residual": _int(
                    gradient.get("positive_residual"), 0
                ),
                "measurement_head_gradient_norm": _float(
                    gradient.get("measurement_head_gradient_norm"), float("nan")
                ),
                "quality_input_gradient_norm": _float(
                    gradient.get("quality_input_gradient_norm"), float("nan")
                ),
                "actor_body_gradient_norm": _float(
                    gradient.get("actor_body_gradient_norm"), float("nan")
                ),
                "gat_gradient_norm": _float(
                    gradient.get("gat_gradient_norm"), float("nan")
                ),
                "checkpoint_unchanged": _int(
                    gradient.get("checkpoint_unchanged"), 0
                ),
                "branch_isolation_ok": int(
                    bool(isolation) and all(_int(row.get("manifest_ok"), 0) == 1 for row in isolation)
                ),
            }
        )
    return output


def _plot(
    seed_rows: Sequence[Mapping[str, Any]],
    shift_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / PLOT_FILE
    seeds = [int(row["base_seed"]) for row in seed_rows]
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    train_expected = [_float(row.get("rate_branch_train_expected_mean"), float("nan")) for row in seed_rows]
    validation_expected = [
        _float(row.get("rate_branch_validation_expected_mean"), float("nan"))
        for row in seed_rows
    ]
    targets = [_float(row.get("rate_branch_constraint_target_mean"), float("nan")) for row in seed_rows]
    axes[0, 0].plot(x, train_expected, "o-", label="train expected")
    axes[0, 0].plot(x, validation_expected, "s-", label="validation expected")
    axes[0, 0].plot(x, targets, "k--", label="constraint target")
    axes[0, 0].set_title("Expected non-none rate vs target")
    axes[0, 0].set_ylabel("rate")
    axes[0, 0].set_xticks(x, [str(seed) for seed in seeds], rotation=30)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    shift_x = np.arange(len(shift_rows))
    if shift_rows:
        axes[0, 1].scatter(
            shift_x,
            [_float(row.get("train_expected_non_none_rate"), float("nan")) for row in shift_rows],
            label="train",
            s=22,
        )
        axes[0, 1].scatter(
            shift_x,
            [_float(row.get("validation_expected_non_none_rate"), float("nan")) for row in shift_rows],
            label="validation",
            s=22,
        )
        axes[0, 1].scatter(
            shift_x,
            [_float(row.get("constraint_target"), float("nan")) for row in shift_rows],
            label="target",
            marker="_",
            s=100,
        )
        axes[0, 1].set_xticks(
            shift_x,
            [f"{row['base_seed']}/{row['uav_id']}" for row in shift_rows],
            rotation=75,
            fontsize=7,
        )
    axes[0, 1].set_title("Per-UAV train/validation shift")
    axes[0, 1].set_ylabel("rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        x,
        [_float(row.get("validation_candidate_blocked_fraction_mean"), float("nan")) for row in seed_rows],
        "o-",
        label="candidate mask blocked",
    )
    axes[1, 0].plot(
        x,
        [_float(row.get("validation_budget_exhausted_fraction_mean"), float("nan")) for row in seed_rows],
        "s-",
        label="budget exhausted",
    )
    axes[1, 0].plot(
        x,
        [_float(row.get("validation_mask_or_budget_fraction_mean"), float("nan")) for row in seed_rows],
        "^-",
        label="sum diagnostic",
    )
    axes[1, 0].set_title("Mask and budget diagnostics")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_xticks(x, [str(seed) for seed in seeds], rotation=30)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    if gradient_rows:
        gx = np.arange(len(gradient_rows))
        axes[1, 1].bar(
            gx - 0.2,
            [_float(row.get("measurement_head_gradient_norm"), 0.0) for row in gradient_rows],
            width=0.2,
            label="measurement head",
        )
        axes[1, 1].bar(
            gx,
            [_float(row.get("quality_input_gradient_norm"), 0.0) for row in gradient_rows],
            width=0.2,
            label="quality input columns",
        )
        axes[1, 1].bar(
            gx + 0.2,
            [_float(row.get("actor_body_gradient_norm"), 0.0) for row in gradient_rows],
            width=0.2,
            label="actor body",
        )
        axes[1, 1].set_xticks(gx, [str(row["base_seed"]) for row in gradient_rows], rotation=30)
    axes[1, 1].set_title("Rate-loss gradient path")
    axes[1, 1].set_ylabel("gradient norm")
    axes[1, 1].set_yscale("symlog", linthresh=1e-8)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Measurement-rate constraint mechanism audit")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _decision_report(
    config: MeasurementRateConstraintMechanismAuditConfig,
    *,
    source_logs: Path,
    target_rows: Sequence[Mapping[str, Any]],
    target_status: Mapping[str, Any],
    manifest_target_status: Mapping[str, Any],
    per_uav_rows: Sequence[Mapping[str, Any]],
    shift_rows: Sequence[Mapping[str, Any]],
    aggregation_rows: Sequence[Mapping[str, Any]],
    aggregation_meta: Mapping[str, Any],
    gradient_rows: Sequence[Mapping[str, Any]],
    isolation_rows: Sequence[Mapping[str, Any]],
    isolation_meta: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    plot_path: Path,
) -> dict[str, Any]:
    target_complete = bool(
        len(target_rows) == len(config.seeds) * config.uav_count
        and manifest_target_status.get("complete", False)
        and target_status.get("complete_seed_bound_match", False)
        and all(_finite(row) for row in target_rows)
    )
    aggregation_complete = bool(
        aggregation_rows
        and not aggregation_meta.get("missing_seed_rows")
        and aggregation_meta.get("source_audit", {}).get("source_aggregation_ok", False)
        and all(_int(row.get("aggregation_ok"), 0) == 1 for row in aggregation_rows)
    )
    gradient_complete = bool(
        len(gradient_rows) == len(config.seeds)
        and all(_int(row.get("probe_ok"), 0) == 1 for row in gradient_rows)
        and all(_int(row.get("checkpoint_unchanged"), 0) == 1 for row in gradient_rows)
        and all(_int(row.get("all_gradient_fields_finite"), 0) == 1 for row in gradient_rows)
    )
    gradient_path_ok = bool(
        gradient_complete
        and all(
            not _int(row.get("positive_residual"), 0)
            or (
                _int(row.get("head_gradient_nonzero_when_active"), 0) == 1
                and _int(row.get("quality_gradient_nonzero_when_active"), 0) == 1
            )
            for row in gradient_rows
        )
    )
    branch_isolation_ok = bool(
        len(isolation_rows) == len(config.seeds) * len(BRANCHES)
        and isolation_meta.get("all_ok", False)
    )
    per_uav_finite = bool(
        per_uav_rows
        and all(
            all(
                math.isfinite(_float(row.get(field), float("nan")))
                for field in (
                    "train_expected_non_none_rate",
                    "validation_expected_non_none_rate",
                    "constraint_target",
                    "validation_raw_action_rate",
                    "validation_applied_action_rate",
                )
            )
            for row in per_uav_rows
        )
    )
    structural_ok = bool(
        target_complete
        and aggregation_complete
        and gradient_complete
        and branch_isolation_ok
        and per_uav_finite
    )

    rate_shift_seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        rows = [row for row in shift_rows if int(row["base_seed"]) == int(seed)]
        if not rows:
            continue
        tolerance = _mean(row.get("tolerance") for row in rows)
        train_gap = _mean(
            abs(_float(row.get("train_expected_minus_target"), float("nan")))
            for row in rows
        )
        validation_gap = _mean(
            abs(_float(row.get("validation_expected_minus_target"), float("nan")))
            for row in rows
        )
        rate_shift_seed_rows.append(
            {
                "base_seed": int(seed),
                "train_abs_gap_mean": train_gap,
                "validation_abs_gap_mean": validation_gap,
                "tolerance": tolerance,
                "train_all_uav_close": int(
                    all(_int(row.get("train_close_within_tolerance"), 0) == 1 for row in rows)
                ),
                "validation_all_uav_close": int(
                    all(
                        _int(row.get("validation_close_within_tolerance"), 0) == 1
                        for row in rows
                    )
                ),
                "validation_mask_budget_fraction": _mean(
                    (
                        _float(row.get("validation_candidate_blocked_fraction"), float("nan"))
                        + _float(row.get("validation_budget_exhausted_fraction"), float("nan"))
                    )
                    for row in rows
                ),
            }
        )
    expected_close_seed_fraction = (
        float(
            np.mean(
                [
                    int(row["validation_all_uav_close"]) == 1
                    for row in rate_shift_seed_rows
                ]
            )
        )
        if rate_shift_seed_rows
        else float("nan")
    )
    train_close_seed_fraction = (
        float(
            np.mean(
                [int(row["train_all_uav_close"]) == 1 for row in rate_shift_seed_rows]
            )
        )
        if rate_shift_seed_rows
        else float("nan")
    )
    mask_budget_values = [
        _float(row.get("validation_mask_budget_fraction"), float("nan"))
        for row in rate_shift_seed_rows
    ]
    mask_budget_values = [value for value in mask_budget_values if math.isfinite(value)]
    mask_budget_dominates = bool(
        mask_budget_values
        and (
            float(np.mean(mask_budget_values)) >= 0.20
            or sum(value >= 0.20 for value in mask_budget_values) >= 3
        )
    )
    train_val_shift = bool(
        rate_shift_seed_rows
        and sum(
            _float(row["validation_abs_gap_mean"], float("nan"))
            - _float(row["train_abs_gap_mean"], float("nan"))
            >= 0.10
            for row in rate_shift_seed_rows
        )
        >= 3
    )
    target_issue = not target_complete
    aggregation_issue = target_complete and not aggregation_complete
    gradient_issue = target_complete and aggregation_complete and not gradient_path_ok
    if target_issue:
        primary_reason = "target_or_denominator_issue"
        next_action = "Only fix target binding or the per-UAV window denominator, then run a seed=2026 Smoke."
    elif aggregation_issue:
        primary_reason = "constraint_aggregation_issue"
        next_action = "Only fix per-UAV aggregation and loss logging; keep coefficient, reward, EKF, and channel unchanged."
    elif gradient_issue:
        primary_reason = "gradient_path_issue"
        next_action = "Only fix the rate-loss connection to the measurement head; do not change the objective weights."
    elif mask_budget_dominates:
        primary_reason = "budget_or_mask_dominates"
        next_action = "Separate eligible-action rate from all-window rate and audit the mask/budget interface before retraining."
    elif train_val_shift:
        primary_reason = "train_validation_shift"
        next_action = "Freeze the current rule and preregister additional seeds or scenarios; do not retune the coefficient."
    elif (
        structural_ok
        and math.isfinite(expected_close_seed_fraction)
        and expected_close_seed_fraction < 0.80
    ):
        primary_reason = "constraint_active_but_insufficient"
        next_action = "Freeze coefficient 0.5 and tolerance 0.05; design a preregistered rate-control experiment instead of blindly increasing the coefficient."
    else:
        primary_reason = "insufficient_evidence"
        next_action = "Add only the missing diagnostic evidence and rerun this audit; do not retrain yet."
    return {
        "experiment": "measurement_rate_constraint_mechanism_audit",
        "evidence_status": "validation_only_mechanism_diagnostic",
        "reference_root": str(config.reference_root),
        "source_logs_dir": str(source_logs),
        "output_dir": str(config.output_dir),
        "plot": str(plot_path),
        "base_seeds": [int(seed) for seed in config.seeds],
        "branches": list(BRANCHES),
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent": True,
        "test_data_read": False,
        "test_data_used_for_selection": False,
        "retraining_performed": False,
        "original_results_modified": False,
        "fixed_controls": {
            "uav_count": int(config.uav_count),
            "split": config.split,
            "measurement_rate_constraint_coef": 0.5,
            "measurement_rate_tolerance": 0.05,
            "bootstrap_samples": int(config.bootstrap_samples),
            "bootstrap_seed": int(config.bootstrap_seed),
        },
        "action_semantics": {
            "constraint_target": "baseline_mappo/train applied measurement_uav_id",
            "actor_target": "baseline_mappo/train action_raw.measurement_uav_id",
            "validation_expected_rate": "actor-visible 1-none_probability",
            "validation_action_mode": "deterministic_argmax",
            "raw_and_applied_actions_reported_separately": True,
        },
        "checks": {
            "target_recomputation_complete": target_complete,
            "logged_target_seed_bound_match": bool(
                target_status.get("complete_seed_bound_match", False)
            ),
            "manifest_target_match": bool(manifest_target_status.get("complete", False)),
            "per_uav_aggregation_closed": aggregation_complete,
            "gradient_probe_complete": gradient_complete,
            "gradient_path_nonzero_when_active": gradient_path_ok,
            "branch_isolation": branch_isolation_ok,
            "all_diagnostic_fields_finite": per_uav_finite
            and all(_finite(row) for row in aggregation_rows if row.get("missing_training_log", 0) == 0),
            "checkpoint_unchanged": bool(
                gradient_rows
                and all(_int(row.get("checkpoint_unchanged"), 0) == 1 for row in gradient_rows)
            ),
            "test_split_read": False,
        },
        "audit_ok": structural_ok,
        "decision_supported": bool(
            structural_ok
            and primary_reason != "insufficient_evidence"
        ),
        "primary_reason": primary_reason,
        "primary_reason_detail": {
            "target_complete": target_complete,
            "aggregation_complete": aggregation_complete,
            "gradient_path_ok": gradient_path_ok,
            "train_close_seed_fraction": train_close_seed_fraction,
            "validation_close_seed_fraction": expected_close_seed_fraction,
            "mask_budget_dominates": mask_budget_dominates,
            "train_validation_shift": train_val_shift,
            "aggregation_meta": dict(aggregation_meta),
        },
        "next_action": next_action,
        "target_status": dict(target_status),
        "manifest_target_status": dict(manifest_target_status),
        "seed_level_summary": rate_shift_seed_rows,
    }


def run_measurement_rate_constraint_mechanism_audit(
    config: MeasurementRateConstraintMechanismAuditConfig,
) -> dict[str, Any]:
    """Run the read-only mechanism audit and write CSV/JSON/PNG outputs."""

    reference_root = Path(config.reference_root)
    source_logs = Path(config.source_logs_dir) if config.source_logs_dir else (
        Path("logs") / "preliminary" / reference_root.name
    )
    output_dir = Path(config.output_dir)
    plots_dir = (
        Path(config.plots_dir)
        if config.plots_dir is not None
        else Path("plots") / "preliminary" / reference_root.name / "rate_constraint_mechanism_audit"
    )
    config = MeasurementRateConstraintMechanismAuditConfig(
        **{
            **config.__dict__,
            "reference_root": reference_root,
            "source_logs_dir": source_logs,
            "output_dir": output_dir,
            "plots_dir": plots_dir,
        }
    )
    _prepare_output(output_dir, force=config.force)
    _prepare_plot_output(plots_dir, force=config.force)
    _progress(config, "recomputing baseline train targets")
    targets, target_rows, source_meta = _fit_targets(
        reference_root,
        config.seeds,
        uav_count=config.uav_count,
    )
    target_rows, manifest_target_status = _merge_manifest_targets(
        reference_root,
        target_rows,
        config.seeds,
        uav_count=config.uav_count,
    )
    target_rows, target_status = _logged_target_status(source_logs, target_rows)
    _progress(config, "auditing branches and train/validation distributions")
    isolation_rows, isolation_meta = _branch_isolation(
        reference_root,
        config.seeds,
        uav_count=config.uav_count,
        split=config.split,
    )
    per_uav_rows, shift_rows, branch_meta = _per_uav_rows(
        reference_root,
        config.seeds,
        target_rows,
        uav_count=config.uav_count,
        split=config.split,
    )
    _progress(config, "checking explicit per-UAV aggregation fields")
    aggregation_rows, aggregation_meta = _training_aggregation_rows(
        source_logs,
        reference_root,
        config.seeds,
        uav_count=config.uav_count,
    )
    _progress(config, "running checkpoint gradient dry-runs without optimizer steps")
    gradient_rows = [
        _gradient_probe_for_seed(
            reference_root,
            seed,
            uav_count=config.uav_count,
        )
        for seed in config.seeds
    ]
    seed_rows = _seed_summary(
        config.seeds,
        per_uav_rows,
        shift_rows,
        aggregation_rows,
        gradient_rows,
        isolation_rows,
    )
    plot_path = _plot(seed_rows, shift_rows, gradient_rows, plots_dir)
    report = _decision_report(
        config,
        source_logs=source_logs,
        target_rows=target_rows,
        target_status=target_status,
        manifest_target_status=manifest_target_status,
        per_uav_rows=per_uav_rows,
        shift_rows=shift_rows,
        aggregation_rows=aggregation_rows,
        aggregation_meta=aggregation_meta,
        gradient_rows=gradient_rows,
        isolation_rows=isolation_rows,
        isolation_meta=isolation_meta,
        seed_rows=seed_rows,
        plot_path=plot_path,
    )
    _write_csv(output_dir / "per_uav_rate_audit.csv", per_uav_rows)
    _write_csv(output_dir / "aggregation_audit.csv", aggregation_rows)
    _write_csv(output_dir / "train_validation_shift.csv", shift_rows)
    _write_csv(output_dir / "gradient_path_audit.csv", gradient_rows)
    _write_csv(output_dir / "seed_summary.csv", seed_rows)
    report["source_meta"] = source_meta
    report["branch_meta"] = branch_meta
    report["isolation_meta"] = dict(isolation_meta)
    _write_json(output_dir / "audit_report.json", report)
    _progress(
        config,
        f"complete: primary_reason={report['primary_reason']} audit_ok={report['audit_ok']}",
    )
    return {
        "audit_ok": bool(report["audit_ok"]),
        "decision_supported": bool(report["decision_supported"]),
        "primary_reason": report["primary_reason"],
        "decision_gate": report["primary_reason"],
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
    }


__all__ = [
    "MeasurementRateConstraintMechanismAuditConfig",
    "run_measurement_rate_constraint_mechanism_audit",
]
