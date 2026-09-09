"""Read-only audit for the train-time measurement-rate constraint.

The rate-controlled experiment already completed its training run.  This
module deliberately does not replay or retrain an environment.  It recomputes
the target rate from the baseline training actions, joins the existing
validation-only policy diagnostics, and reports which part of the rate-control
chain can (and cannot) be established from the persisted evidence.

The audit never enumerates or reads a ``test`` directory.  Training actions
are used only for target recomputation; validation actions and actor-visible
probability logs are used for diagnosis.  Time windows are diagnostic rows,
not independent observations for the reported seed-level conclusions.
"""

from __future__ import annotations

import csv
import json
import math
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
    "target_recomputation.csv",
    "training_constraint_audit.csv",
    "validation_rate_audit.csv",
    "branch_isolation_audit.csv",
    "audit_report.json",
)
PLOT_FILE = "measurement_rate_constraint_audit.png"


@dataclass(frozen=True)
class MeasurementRateConstraintAuditConfig:
    """Configuration for the validation-only rate-constraint audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/"
        "rate_constraint_audit"
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
            raise ValueError("measurement-rate constraint audit is validation-only")
        if self.uav_count < 2:
            raise ValueError("uav_count must be at least 2")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique sequence")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _progress(config: MeasurementRateConstraintAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[measurement-rate-constraint-audit] {message}", flush=True)


def _read_csv_with_fields(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return _read_csv_with_fields(path)[1]


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
        raise ValueError(f"expected a JSON object: {path}")
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
                    output[column] = float(value) if math.isfinite(float(value)) else ""
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
            rows.append(row)
    return rows


def _quality_rows(scale_root: Path, split: str) -> list[dict[str, Any]]:
    path = scale_root / "diagnostics" / "measurement_quality.jsonl"
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        if str(row.get("split", "")) == split:
            rows.append(row)
    return rows


def _uav_ids(uav_count: int) -> list[str]:
    return [f"uav_{index:02d}" for index in range(uav_count)]


def _is_selected(row: Mapping[str, Any]) -> bool:
    value = str(row.get("measurement_uav_id", "none")).strip().lower()
    return bool(value) and value != "none"


def _fit_targets(
    reference_root: Path,
    seeds: Sequence[int],
    uav_count: int,
) -> tuple[dict[tuple[int, str], float], list[dict[str, Any]], dict[str, Any]]:
    targets: dict[tuple[int, str], float] = {}
    output: list[dict[str, Any]] = []
    source_counts: dict[str, Any] = {}
    ids = _uav_ids(uav_count)
    for seed in seeds:
        scale_root = _scale_root(reference_root, seed, "baseline_mappo")
        rows = _action_rows(scale_root, "train")
        selected = {uav_id: 0 for uav_id in ids}
        windows = {uav_id: 0 for uav_id in ids}
        for row in rows:
            uav_id = str(row.get("uav_id", ""))
            if uav_id not in windows:
                continue
            windows[uav_id] += 1
            selected[uav_id] += int(_is_selected(row))
        for uav_id in ids:
            if windows[uav_id] <= 0:
                raise ValueError(f"no baseline train windows for seed={seed}, uav={uav_id}")
            target = float(selected[uav_id] / windows[uav_id])
            targets[(int(seed), uav_id)] = target
            output.append(
                {
                    "base_seed": int(seed),
                    "uav_id": uav_id,
                    "baseline_train_selected_measurements": int(selected[uav_id]),
                    "baseline_train_decision_windows": int(windows[uav_id]),
                    "target_measurement_rate": target,
                    "source_split": "train",
                    "source_branch": "baseline_mappo",
                    "recomputed_from": str(scale_root / "train"),
                }
            )
        source_counts[str(seed)] = {
            uav_id: {
                "selected": int(selected[uav_id]),
                "decision_windows": int(windows[uav_id]),
            }
            for uav_id in ids
        }
    return targets, output, source_counts


def _logged_target_status(
    source_logs: Path,
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_logs / "measurement_rate_targets.csv"
    fields, logged = _read_csv_with_fields(path)
    has_seed = "base_seed" in fields
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    if has_seed:
        for row in logged:
            try:
                key = (_int(row.get("base_seed"), -1), str(row.get("uav_id", "")))
            except (TypeError, ValueError):
                continue
            if key in by_key:
                duplicate_keys.append(f"{key[0]}/{key[1]}")
            by_key[key] = row
    output: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in target_rows:
        seed = int(row["base_seed"])
        uav_id = str(row["uav_id"])
        key = (seed, uav_id)
        logged_row = by_key.get(key)
        if not path.exists():
            status = "missing_file"
            available = 0
            match = 0
        elif not has_seed:
            status = "missing_base_seed_column"
            available = 0
            match = 0
        elif logged_row is None:
            status = "missing_seed_uav_row"
            available = 0
            match = 0
        else:
            selected_match = _int(logged_row.get("baseline_train_selected_measurements"), -1) == int(
                row["baseline_train_selected_measurements"]
            )
            windows_match = _int(logged_row.get("baseline_train_decision_windows"), -1) == int(
                row["baseline_train_decision_windows"]
            )
            rate_match = abs(
                _float(logged_row.get("target_measurement_rate"), float("nan"))
                - _float(row.get("target_measurement_rate"), float("nan"))
            ) <= 1e-12
            source_match = (
                str(logged_row.get("source_split", "")) == "train"
                and str(logged_row.get("source_branch", "")) == "baseline_mappo"
            )
            match = int(selected_match and windows_match and rate_match and source_match)
            status = "match" if match else "value_or_source_mismatch"
            available = 1
            if not match:
                mismatches.append(f"{seed}/{uav_id}")
        output.append(
            {
                **dict(row),
                "logged_target_file": str(path),
                "logged_target_seed_bound": int(has_seed),
                "logged_target_available": int(available),
                "logged_target_match": int(match),
                "logged_target_status": status,
            }
        )
    status_summary = {
        "file_exists": bool(path.exists()),
        "fields": fields,
        "row_count": len(logged),
        "has_base_seed": bool(has_seed),
        "duplicate_seed_uav_keys": duplicate_keys,
        "value_or_source_mismatches": mismatches,
        "complete_seed_bound_match": bool(output) and all(
            int(row["logged_target_match"]) == 1 for row in output
        ),
    }
    return output, status_summary


def _manifest_config(scale_root: Path) -> dict[str, Any]:
    path = scale_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing scale manifest: {path}")
    manifest = _read_json(path)
    config = manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"manifest config is not an object: {path}")
    return dict(config)


def _quality_group(rows: Sequence[Mapping[str, Any]], uav_count: int) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    ids = set(_uav_ids(uav_count))
    for row in rows:
        uav_id = str(row.get("uav_id", ""))
        episode_id = str(row.get("episode_id", ""))
        if uav_id not in ids or not episode_id:
            continue
        none_probability = _float(row.get("none_probability"), float("nan"))
        if not math.isfinite(none_probability):
            continue
        expected = 1.0 - none_probability
        if not 0.0 <= expected <= 1.0:
            continue
        item = groups.setdefault(
            (episode_id, uav_id),
            {
                "values": [],
                "none_values": [],
                "quality_feature_enabled": set(),
                "budget_feature_enabled": set(),
                "constraint_enabled": set(),
                "actor_branches": set(),
                "rows": 0,
            },
        )
        item["values"].append(expected)
        item["none_values"].append(none_probability)
        item["rows"] += 1
        for field, target in (
            ("quality_feature_enabled", "quality_feature_enabled"),
            ("measurement_budget_feature_enabled", "budget_feature_enabled"),
            ("measurement_rate_constraint_enabled", "constraint_enabled"),
        ):
            if field in row:
                item[target].add(_int(row.get(field), 0))
        if "quality_actor_branch" in row:
            item["actor_branches"].add(str(row.get("quality_actor_branch", "")))
    return groups


def _quality_group_summary(groups: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for key, item in groups.items():
        values = [float(value) for value in item.get("values", []) if math.isfinite(float(value))]
        none_values = [float(value) for value in item.get("none_values", []) if math.isfinite(float(value))]
        output[key] = {
            "expected_rate": float(np.mean(values)) if values else float("nan"),
            "none_probability_mean": float(np.mean(none_values)) if none_values else float("nan"),
            "quality_feature_enabled_values": sorted(int(value) for value in item.get("quality_feature_enabled", set())),
            "budget_feature_enabled_values": sorted(int(value) for value in item.get("budget_feature_enabled", set())),
            "constraint_enabled_values": sorted(int(value) for value in item.get("constraint_enabled", set())),
            "actor_branch_values": sorted(str(value) for value in item.get("actor_branches", set())),
            "quality_rows": int(item.get("rows", 0)),
        }
    return output


def _parse_action_mask(value: Any, uav_count: int) -> tuple[bool, bool]:
    """Return (has_non_none_candidate, parseable)."""

    if isinstance(value, Mapping):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False, False
    if not isinstance(payload, Mapping):
        return False, False
    values = payload.get("measurement")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False, False
    return any(bool(item) for item in list(values)[:uav_count]), True


def _budget_alignment(
    scale_root: Path,
    action_rows: Sequence[Mapping[str, Any]],
    split: str,
    uav_count: int,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Align budget rows to actions without reading any other split."""

    path_by_episode: dict[str, list[dict[str, str]]] = {}
    for episode_root in _episode_dirs(scale_root, split):
        path = episode_root / "measurement_budget.csv"
        if path.exists():
            path_by_episode[episode_root.name] = _read_csv(path)
    action_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in action_rows:
        action_groups.setdefault((str(row.get("episode_id", "")), str(row.get("uav_id", ""))), []).append(row)
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for (episode_id, uav_id), rows in action_groups.items():
        rows = sorted(rows, key=lambda row: (_float(row.get("timestamp"), 0.0), str(row.get("measurement_uav_id", ""))))
        budget_rows = [
            row for row in path_by_episode.get(episode_id, [])
            if str(row.get("uav_id", "")) == uav_id
        ]
        budget_rows = sorted(budget_rows, key=lambda row: _int(row.get("step"), 0))
        for index, row in enumerate(rows):
            budget = budget_rows[index] if index < len(budget_rows) else {}
            result[(episode_id, uav_id, index)] = {
                "budget_available": int(bool(budget)),
                "budget_limit": _int(budget.get("budget_limit"), -1),
                "budget_remaining_before": _float(budget.get("budget_remaining_before"), float("nan")),
                "budget_consumed": _int(budget.get("budget_consumed"), 0),
                "budget_remaining_after": _float(budget.get("budget_remaining_after"), float("nan")),
                "budget_used": _int(budget.get("budget_used"), 0),
            }
    return result


def _validation_rows(
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    targets: Mapping[tuple[int, str], float],
    uav_count: int,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    branch_meta: dict[str, Any] = {}
    ids = _uav_ids(uav_count)
    for seed in seeds:
        for branch in branches:
            scale_root = _scale_root(reference_root, seed, branch)
            config = _manifest_config(scale_root)
            action_rows = _action_rows(scale_root, split)
            quality = _quality_group_summary(_quality_group(_quality_rows(scale_root, split), uav_count))
            budget = _budget_alignment(scale_root, action_rows, split, uav_count)
            by_uav: dict[str, list[Mapping[str, Any]]] = {uav_id: [] for uav_id in ids}
            for row in action_rows:
                uav_id = str(row.get("uav_id", ""))
                if uav_id in by_uav:
                    by_uav[uav_id].append(row)
            for uav_id in ids:
                rows = sorted(
                    by_uav[uav_id],
                    key=lambda row: (_float(row.get("timestamp"), 0.0), str(row.get("measurement_uav_id", ""))),
                )
                selected_count = sum(int(_is_selected(row)) for row in rows)
                candidate_mask_blocked = 0
                mask_unparseable = 0
                budget_exhausted = 0
                none_due_budget = 0
                budget_limits: list[int] = []
                for index, row in enumerate(rows):
                    has_candidate, parseable = _parse_action_mask(row.get("action_mask_json", ""), uav_count)
                    if parseable and not has_candidate:
                        candidate_mask_blocked += 1
                    elif not parseable:
                        mask_unparseable += 1
                    budget_info = budget.get((str(row.get("episode_id", "")), uav_id, index), {})
                    remaining = _float(budget_info.get("budget_remaining_before"), float("nan"))
                    if math.isfinite(remaining) and remaining <= 0.0:
                        budget_exhausted += 1
                        if not _is_selected(row):
                            none_due_budget += 1
                    limit = _int(budget_info.get("budget_limit"), -1)
                    if limit >= 0:
                        budget_limits.append(limit)
                expected_values = [
                    item["expected_rate"]
                    for (episode_id, quality_uav), item in quality.items()
                    if quality_uav == uav_id and math.isfinite(item["expected_rate"])
                ]
                expected_rate = float(np.mean(expected_values)) if expected_values else float("nan")
                target = float(targets[(int(seed), uav_id)])
                windows = len(rows)
                actual_rate = float(selected_count / windows) if windows else float("nan")
                row_quality = [
                    item for (episode_id, quality_uav), item in quality.items()
                    if quality_uav == uav_id
                ]
                quality_feature_values = sorted({value for item in row_quality for value in item["quality_feature_enabled_values"]})
                budget_feature_values = sorted({value for item in row_quality for value in item["budget_feature_enabled_values"]})
                constraint_feature_values = sorted({value for item in row_quality for value in item["constraint_enabled_values"]})
                actor_branch_values = sorted({value for item in row_quality for value in item["actor_branch_values"]})
                output.append(
                    {
                        "base_seed": int(seed),
                        "treatment": branch,
                        "split": split,
                        "uav_id": uav_id,
                        "decision_windows": int(windows),
                        "selected_measurements": int(selected_count),
                        "actual_sampled_rate": actual_rate,
                        "expected_non_none_rate": expected_rate,
                        "target_measurement_rate": target,
                        "actual_minus_target": actual_rate - target,
                        "expected_minus_target": expected_rate - target,
                        "actual_minus_expected": actual_rate - expected_rate,
                        "none_probability_mean": 1.0 - expected_rate if math.isfinite(expected_rate) else float("nan"),
                        "candidate_mask_blocked_windows": int(candidate_mask_blocked),
                        "candidate_mask_blocked_fraction": float(candidate_mask_blocked / windows) if windows else float("nan"),
                        "mask_unparseable_windows": int(mask_unparseable),
                        "budget_available": int(bool(budget_limits)),
                        "budget_limit": int(round(float(np.median(budget_limits)))) if budget_limits else -1,
                        "budget_exhausted_windows": int(budget_exhausted),
                        "budget_exhausted_fraction": float(budget_exhausted / windows) if windows else float("nan"),
                        "none_due_budget_windows": int(none_due_budget),
                        "none_due_budget_fraction": float(none_due_budget / windows) if windows else float("nan"),
                        "quality_feature_log_values": quality_feature_values,
                        "budget_feature_log_values": budget_feature_values,
                        "constraint_feature_log_values": constraint_feature_values,
                        "quality_actor_branch_log_values": actor_branch_values,
                        "quality_rows": int(sum(item["quality_rows"] for item in row_quality)),
                        "source_actions": str(scale_root / split),
                        "source_probability_log": str(scale_root / "diagnostics" / "measurement_quality.jsonl"),
                    }
                )
            branch_meta[f"{seed}/{branch}"] = {
                "config": config,
                "action_row_count": len(action_rows),
                "quality_row_count": len(_quality_rows(scale_root, split)),
            }
    return output, branch_meta


def _training_constraint_rows(
    reference_root: Path,
    source_logs: Path,
    seeds: Sequence[int],
    targets: Mapping[tuple[int, str], float],
    uav_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join train-only expected probabilities with the persisted rate log."""

    ids = _uav_ids(uav_count)
    logged_rows = [
        row for row in _read_csv(source_logs / "training_rate_constraint.csv")
        if str(row.get("phase", "train")) == "train"
        and str(row.get("treatment", RATE_BRANCH)) == RATE_BRANCH
        and _int(row.get("base_seed"), -1) in set(int(seed) for seed in seeds)
    ]
    output: list[dict[str, Any]] = []
    missing_expected = 0
    expected_count = 0
    positive_loss_count = 0
    enabled_count = 0
    explicit_field_count = 0
    for seed in seeds:
        scale_root = _scale_root(reference_root, seed, RATE_BRANCH)
        config = _manifest_config(scale_root)
        tolerance = _float(config.get("measurement_rate_tolerance"), 0.05)
        coefficient = _float(config.get("measurement_rate_constraint_coef"), 0.0)
        quality_summary = _quality_group_summary(_quality_group(_quality_rows(scale_root, "train"), uav_count))
        seed_logged = [row for row in logged_rows if _int(row.get("base_seed"), -1) == int(seed)]
        if not seed_logged:
            # Preserve a visible evidence row when the aggregate log is absent.
            seed_logged = [
                {
                    "base_seed": seed,
                    "episode_id": episode_id,
                    "update": index + 1,
                    "measurement_rate_constraint_enabled": 1,
                    "measurement_rate_constraint_loss": "",
                }
                for index, (episode_id, _uav_id) in enumerate(sorted(quality_summary))
                if _uav_id == ids[0]
            ]
        for logged in seed_logged:
            episode_id = str(logged.get("episode_id", logged.get("episode", "")))
            update = _int(logged.get("update"), 0)
            enabled = _int(logged.get("measurement_rate_constraint_enabled"), 0)
            logged_loss = _float(logged.get("measurement_rate_constraint_loss"), float("nan"))
            enabled_count += int(enabled == 1)
            positive_loss_count += int(math.isfinite(logged_loss) and logged_loss > 1e-12)
            for uav_id in ids:
                summary = quality_summary.get((episode_id, uav_id), {})
                expected = _float(summary.get("expected_rate"), float("nan"))
                expected_available = int(math.isfinite(expected))
                expected_count += expected_available
                if not expected_available:
                    missing_expected += 1
                target = float(targets[(int(seed), uav_id)])
                upper = max(expected - target - tolerance, 0.0) if expected_available else float("nan")
                lower = max(target - tolerance - expected, 0.0) if expected_available else float("nan")
                raw_loss = upper * upper + lower * lower if expected_available else float("nan")
                weighted_loss = coefficient * raw_loss if expected_available else float("nan")
                explicit_target = _float(logged.get(f"measurement_rate_target_{uav_id}"), float("nan"))
                explicit_expected = _float(logged.get(f"measurement_rate_expected_{uav_id}"), float("nan"))
                explicit_field_count += int(math.isfinite(explicit_target) or math.isfinite(explicit_expected))
                output.append(
                    {
                        "base_seed": int(seed),
                        "treatment": RATE_BRANCH,
                        "phase": "train",
                        "update": int(update),
                        "episode_id": episode_id,
                        "uav_id": uav_id,
                        "target_measurement_rate": target,
                        "expected_non_none_rate": expected,
                        "upper_residual": upper,
                        "lower_residual": lower,
                        "raw_constraint_loss": raw_loss,
                        "weighted_constraint_loss": weighted_loss,
                        "logged_constraint_loss": logged_loss,
                        "logged_weighted_constraint_loss": _float(
                            logged.get("measurement_rate_constraint_weighted_loss"), float("nan")
                        ),
                        "constraint_enabled": int(enabled),
                        "expected_rate_available": expected_available,
                        "explicit_target_logged": int(math.isfinite(explicit_target)),
                        "explicit_expected_logged": int(math.isfinite(explicit_expected)),
                        "source": str(source_logs / "training_rate_constraint.csv"),
                    }
                )
    meta = {
        "logged_row_count": len(logged_rows),
        "expanded_row_count": len(output),
        "enabled_row_count": enabled_count,
        "positive_logged_loss_row_count": positive_loss_count,
        "expected_rate_available_count": expected_count,
        "missing_expected_rate_count": missing_expected,
        "explicit_per_uav_field_count": explicit_field_count,
        "aggregate_log_fields": sorted({key for row in logged_rows for key in row}),
    }
    return output, meta


def _branch_isolation_rows(
    reference_root: Path,
    source_logs: Path,
    seeds: Sequence[int],
    branches: Sequence[str],
    uav_count: int,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    all_manifest_ok = True
    log_mismatches: list[str] = []
    rate_log = _read_csv(source_logs / "training_rate_constraint.csv")
    for seed in seeds:
        for branch in branches:
            scale_root = _scale_root(reference_root, seed, branch)
            config = _manifest_config(scale_root)
            quality_rows = _quality_rows(scale_root, split)
            branch_values = sorted({str(row.get("quality_actor_branch", "")) for row in quality_rows})
            quality_values = sorted({_int(row.get("quality_feature_enabled"), 0) for row in quality_rows if "quality_feature_enabled" in row})
            budget_values = sorted({_int(row.get("measurement_budget_feature_enabled"), 0) for row in quality_rows if "measurement_budget_feature_enabled" in row})
            constraint_values = sorted({_int(row.get("measurement_rate_constraint_enabled"), 0) for row in quality_rows if "measurement_rate_constraint_enabled" in row})
            expected_quality = int(branch in QUALITY_BRANCHES)
            expected_budget = int(branch in QUALITY_BRANCHES)
            expected_constraint = int(branch == RATE_BRANCH)
            config_variant_ok = str(config.get("mappo_variant", "")) == branch
            config_quality_ok = int(bool(config.get("measurement_quality_logit_penalty", 0.0) > 0.0)) == expected_quality
            config_budget_ok = int(bool(config.get("measurement_budget_enabled", False))) == expected_budget
            config_constraint_ok = int(bool(config.get("measurement_rate_constraint_enabled", False))) == expected_constraint
            manifest_ok = bool(config_variant_ok and config_quality_ok and config_budget_ok and config_constraint_ok)
            all_manifest_ok = all_manifest_ok and manifest_ok
            log_flags_match = (
                (not quality_values or expected_quality in quality_values)
                and (not budget_values or expected_budget in budget_values)
                and (not constraint_values or expected_constraint in constraint_values)
            )
            if not log_flags_match:
                log_mismatches.append(f"{seed}/{branch}")
            rate_rows = [
                row for row in rate_log
                if _int(row.get("base_seed"), -1) == int(seed)
                and str(row.get("treatment", "")) == RATE_BRANCH
                and str(row.get("phase", "train")) == "train"
            ]
            output.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "mappo_variant": str(config.get("mappo_variant", "")),
                    "config_quality_feature_enabled": expected_quality,
                    "config_measurement_budget_enabled": int(bool(config.get("measurement_budget_enabled", False))),
                    "config_rate_constraint_enabled": int(bool(config.get("measurement_rate_constraint_enabled", False))),
                    "config_rate_constraint_coef": _float(config.get("measurement_rate_constraint_coef"), 0.0),
                    "config_rate_tolerance": _float(config.get("measurement_rate_tolerance"), 0.05),
                    "config_target_present": int(isinstance(config.get("measurement_rate_target_by_uav"), Mapping)),
                    "manifest_isolation_ok": int(manifest_ok),
                    "quality_log_branch_values": branch_values,
                    "quality_log_quality_flag_values": quality_values,
                    "quality_log_budget_flag_values": budget_values,
                    "quality_log_constraint_flag_values": constraint_values,
                    "quality_log_flags_match_config": int(log_flags_match),
                    "rate_constraint_training_log_rows": len(rate_rows),
                    "source_manifest": str(scale_root / "manifest.json"),
                }
            )
    return output, {
        "all_manifest_isolation_ok": all_manifest_ok,
        "quality_log_flag_mismatches": log_mismatches,
    }


def _mean(values: Iterable[Any]) -> float:
    numbers = np.asarray([_float(value, float("nan")) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(np.mean(numbers)) if numbers.size else float("nan")


def _bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _reason_and_report(
    config: MeasurementRateConstraintAuditConfig,
    targets: Mapping[tuple[int, str], float],
    target_rows: Sequence[Mapping[str, Any]],
    target_status: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    training_meta: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    isolation_rows: Sequence[Mapping[str, Any]],
    isolation_meta: Mapping[str, Any],
    source_logs: Path,
) -> dict[str, Any]:
    rate_validation = [row for row in validation_rows if str(row.get("treatment")) == RATE_BRANCH]
    expected_gap = [abs(_float(row.get("expected_minus_target"), float("nan"))) for row in rate_validation]
    actual_gap = [abs(_float(row.get("actual_minus_target"), float("nan"))) for row in rate_validation]
    blocker = [
        max(
            _float(row.get("budget_exhausted_fraction"), 0.0),
            _float(row.get("candidate_mask_blocked_fraction"), 0.0),
        )
        for row in rate_validation
    ]
    expected_close = bool(expected_gap) and float(np.mean([value <= 0.10 for value in expected_gap])) >= 0.8
    actual_not_close = bool(actual_gap) and float(np.mean([value > 0.10 for value in actual_gap])) >= 0.2
    blocker_high = bool(blocker) and float(np.mean([value >= 0.20 for value in blocker])) >= 0.5
    enabled = int(training_meta.get("enabled_row_count", 0)) > 0
    positive_loss = int(training_meta.get("positive_logged_loss_row_count", 0)) > 0
    expected_available = int(training_meta.get("expected_rate_available_count", 0)) > 0
    rate_config_rows = [row for row in isolation_rows if str(row.get("branch")) == RATE_BRANCH]
    rate_config_ok = bool(rate_config_rows) and all(
        int(row.get("config_rate_constraint_enabled", 0)) == 1
        and int(row.get("config_target_present", 0)) == 1
        for row in rate_config_rows
    )

    # Missing seed binding in the persisted target table is an auditability
    # error even when the target can be recomputed correctly from the source
    # actions.  It is deliberately reported before behavioural hypotheses:
    # the existing artifact cannot prove which target belonged to which seed.
    if not bool(target_status.get("complete_seed_bound_match", False)):
        primary_reason = "target_provenance_error"
        detail = (
            "Targets are recomputable from baseline train actions, but the persisted target table "
            "is not seed-bound (or does not match the recomputation), so the existing formal artifact "
            "is not fully traceable per seed/UAV."
        )
        next_action = (
            "Regenerate only the diagnostic provenance with base_seed and per-UAV target/expected-rate "
            "fields, then run a seed=2026 Smoke before any five-seed rerun."
        )
    elif not enabled or not positive_loss or not expected_available or not rate_config_ok:
        primary_reason = "constraint_inactive"
        detail = (
            "The rate branch is not evidenced as an enabled, non-zero training constraint with a "
            "complete target/expected-rate path."
        )
        next_action = "Fix only the constraint logging or gradient-path interface, then run a seed=2026 Smoke."
    elif expected_close and actual_not_close and blocker_high:
        primary_reason = "budget_or_mask_dominates"
        detail = (
            "Expected policy probability is near the train target, but sampled validation actions are "
            "frequently blocked by budget exhaustion or the non-none action mask."
        )
        next_action = "Audit only the budget/mask interface and its episode reset semantics before changing coefficients."
    elif expected_close and actual_not_close:
        primary_reason = "sampled_expected_rate_gap"
        detail = (
            "Expected non-none probability is near the target while sampled non-none actions are not; "
            "the remaining gap is between policy probabilities and sampled actions."
        )
        next_action = "Add only action-probability versus sampled-action diagnostics and run a seed=2026 Smoke."
    elif enabled and positive_loss and expected_available and actual_not_close:
        primary_reason = "constraint_active_but_insufficient"
        detail = "The constraint is active in the logged training updates but does not control validation rates sufficiently."
        next_action = "Freeze this coefficient/tolerance result and preregister a separate communication-cost constraint; do not blindly increase the coefficient."
    else:
        primary_reason = "insufficient_evidence"
        detail = "The persisted logs do not distinguish the allowed rate-control failure modes."
        next_action = "Add the missing per-UAV rate diagnostics and repeat only the seed=2026 diagnostic Smoke."

    seed_level: list[dict[str, Any]] = []
    for seed in sorted({int(row["base_seed"]) for row in rate_validation}):
        selected = [row for row in rate_validation if int(row["base_seed"]) == seed]
        actual_delta = _mean(row.get("actual_minus_target") for row in selected)
        expected_delta = _mean(row.get("expected_minus_target") for row in selected)
        seed_level.append(
            {
                "base_seed": seed,
                "actual_minus_target_mean": actual_delta,
                "expected_minus_target_mean": expected_delta,
                "actual_absolute_gap_mean": _mean(abs(_float(row.get("actual_minus_target"), float("nan"))) for row in selected),
                "expected_absolute_gap_mean": _mean(abs(_float(row.get("expected_minus_target"), float("nan"))) for row in selected),
            }
        )
    actual_deltas = [row["actual_minus_target_mean"] for row in seed_level]
    expected_deltas = [row["expected_minus_target_mean"] for row in seed_level]
    return {
        "experiment": "measurement_rate_constraint_audit",
        "evidence_status": "validation_only_mechanism_diagnostic",
        "reference_root": str(config.reference_root),
        "source_logs_dir": str(source_logs),
        "base_seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "episode_rows_are_not_independent_across_windows": True,
        "test_data_read": False,
        "test_data_used_for_selection": False,
        "retraining_performed": False,
        "original_results_modified": False,
        "fit_sources": {
            "measurement_rate_targets": "baseline_mappo/train actions only",
            "expected_measurement_rate": "actor-visible measurement_quality.jsonl; train for constraint diagnosis, validation for evaluation",
            "validation_policy": "validation actions and probability logs only",
        },
        "fixed_controls": {
            "uav_count": int(config.uav_count),
            "split": config.split,
            "bootstrap_samples": int(config.bootstrap_samples),
            "bootstrap_seed": int(config.bootstrap_seed),
        },
        "checks": {
            "target_recomputation_complete": len(target_rows) == len(config.seeds) * config.uav_count
            and all(
                str(row.get("source_split")) == "train"
                and str(row.get("source_branch")) == "baseline_mappo"
                and 0.0 <= _float(row.get("target_measurement_rate"), -1.0) <= 1.0
                for row in target_rows
            ),
            "target_logged_seed_bound_match": bool(target_status.get("complete_seed_bound_match", False)),
            "target_logged_fields": target_status.get("fields", []),
            "branch_manifest_isolation": bool(isolation_meta.get("all_manifest_isolation_ok", False)),
            "rate_constraint_logged_enabled": enabled,
            "rate_constraint_logged_positive_loss": positive_loss,
            "train_expected_rate_reconstructable": expected_available,
            "validation_rows_complete": len(validation_rows) == len(config.seeds) * len(BRANCHES) * config.uav_count,
            "validation_fields_finite": all(
                all(
                    math.isfinite(_float(row.get(field), float("nan")))
                    for field in (
                        "decision_windows",
                        "selected_measurements",
                        "actual_sampled_rate",
                        "target_measurement_rate",
                        "actual_minus_target",
                    )
                )
                for row in validation_rows
            ),
            "training_audit_fields_finite": all(
                all(
                    math.isfinite(_float(row.get(field), float("nan")))
                    for field in (
                        "target_measurement_rate",
                        "expected_non_none_rate",
                        "raw_constraint_loss",
                        "weighted_constraint_loss",
                    )
                )
                for row in training_rows
                if int(row.get("expected_rate_available", 0)) == 1
            ),
            "test_split_read": False,
        },
        "evidence": {
            "target_source_counts": {
                str(seed): {
                    uav_id: {
                        "selected": int(next(row["baseline_train_selected_measurements"] for row in target_rows if int(row["base_seed"]) == seed and row["uav_id"] == uav_id)),
                        "decision_windows": int(next(row["baseline_train_decision_windows"] for row in target_rows if int(row["base_seed"]) == seed and row["uav_id"] == uav_id)),
                    }
                    for uav_id in _uav_ids(config.uav_count)
                }
                for seed in config.seeds
            },
            "target_status": dict(target_status),
            "training_constraint": dict(training_meta),
            "quality_log_flag_mismatches": list(isolation_meta.get("quality_log_flag_mismatches", [])),
            "validation_rate_rows": len(validation_rows),
            "rate_validation_expected_absolute_gap_mean": _mean(expected_gap),
            "rate_validation_actual_absolute_gap_mean": _mean(actual_gap),
            "rate_validation_budget_or_mask_blocker_fraction_mean": _mean(blocker),
            "expected_rate_near_target": expected_close,
            "actual_rate_not_near_target": actual_not_close,
            "budget_or_mask_blocker_high": blocker_high,
        },
        "seed_level_validation_summary": seed_level,
        "bootstrap_95_ci": {
            "actual_minus_target_mean": dict(zip(("low", "high"), _bootstrap_mean_ci(actual_deltas, config.bootstrap_samples, config.bootstrap_seed))),
            "expected_minus_target_mean": dict(zip(("low", "high"), _bootstrap_mean_ci(expected_deltas, config.bootstrap_samples, config.bootstrap_seed + 1))),
        },
        "decision_gate": primary_reason,
        "primary_reason": primary_reason,
        "primary_reason_detail": detail,
        "next_action": next_action,
        "non_decisive_observations": {
            "algorithmic_stability_not_rejudged": True,
            "target_values_were_recomputed_from_train": bool(targets),
            "quality_log_flag_mismatch_does_not_change_manifest_config": bool(isolation_meta.get("quality_log_flag_mismatches")),
        },
        "audit_ok": bool(
            len(target_rows) == len(config.seeds) * config.uav_count
            and len(validation_rows) == len(config.seeds) * len(BRANCHES) * config.uav_count
            and bool(isolation_meta.get("all_manifest_isolation_ok", False))
            and all(_finite(row) for row in target_rows)
            and all(_finite(row) for row in validation_rows)
        ),
    }


def _plot(
    validation_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> Path:
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
    seeds = sorted({int(row["base_seed"]) for row in validation_rows})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for branch in BRANCHES:
        rows = [row for row in validation_rows if str(row.get("treatment")) == branch]
        actual = []
        expected = []
        target = []
        deviation = []
        for seed in seeds:
            current = [row for row in rows if int(row["base_seed"]) == seed]
            actual.append(_mean(row.get("actual_sampled_rate") for row in current))
            expected.append(_mean(row.get("expected_non_none_rate") for row in current))
            target.append(_mean(row.get("target_measurement_rate") for row in current))
            deviation.append(_mean(abs(_float(row.get("actual_minus_target"), float("nan"))) for row in current))
        axes[0, 0].plot(seeds, actual, marker="o", color=colors[branch], label=labels[branch])
        axes[0, 0].plot(seeds, expected, marker="x", linestyle="--", color=colors[branch], alpha=0.65)
        axes[0, 1].plot(seeds, deviation, marker="o", color=colors[branch], label=labels[branch])
        axes[1, 0].plot(seeds, target, marker="o", color=colors[branch], label=labels[branch])
    axes[0, 0].set_title("Validation actual (solid) vs expected (dashed) rate")
    axes[0, 0].set_ylabel("Non-none measurement rate")
    axes[0, 1].set_title("Absolute deviation from train target")
    axes[0, 1].set_ylabel("Mean absolute deviation")
    axes[1, 0].set_title("Train-derived target rate")
    axes[1, 0].set_ylabel("Target rate")
    train_losses = []
    for seed in seeds:
        current = [row for row in training_rows if int(row.get("base_seed", -1)) == seed]
        values = [_float(row.get("logged_constraint_loss"), float("nan")) for row in current]
        train_losses.append(_mean(values))
    axes[1, 1].plot(seeds, train_losses, marker="o", color=colors[RATE_BRANCH], label="Logged raw constraint loss")
    axes[1, 1].set_title("Rate-branch training constraint loss")
    axes[1, 1].set_ylabel("Aggregate raw loss")
    for axis in axes.ravel():
        axis.set_xlabel("Base training seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output = plots_dir / PLOT_FILE
    figure.suptitle("Measurement-rate constraint audit (validation; test sealed)")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def run_measurement_rate_constraint_audit(
    config: MeasurementRateConstraintAuditConfig,
) -> dict[str, Any]:
    """Run the non-mutating measurement-rate constraint audit."""

    _prepare_output(config.output_dir, force=config.force)
    plots_dir = config.plots_dir or (
        Path("plots/preliminary") / config.reference_root.name / "rate_constraint_audit"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    source_logs = config.source_logs_dir or (Path("logs/preliminary") / config.reference_root.name)
    source_logs = Path(source_logs)
    _progress(config, f"reading train targets from {config.reference_root}")

    for seed in config.seeds:
        for branch in BRANCHES:
            scale_root = _scale_root(config.reference_root, int(seed), branch)
            if not scale_root.is_dir():
                raise FileNotFoundError(f"missing seed/branch scale root: {scale_root}")

    targets, target_rows, target_counts = _fit_targets(
        config.reference_root,
        config.seeds,
        config.uav_count,
    )
    target_rows, target_status = _logged_target_status(source_logs, target_rows)
    _progress(config, "auditing training constraint diagnostics")
    training_rows, training_meta = _training_constraint_rows(
        config.reference_root,
        source_logs,
        config.seeds,
        targets,
        config.uav_count,
    )
    _progress(config, "auditing validation sampled and expected rates")
    validation_rows, _branch_meta = _validation_rows(
        config.reference_root,
        config.seeds,
        BRANCHES,
        targets,
        config.uav_count,
        config.split,
    )
    isolation_rows, isolation_meta = _branch_isolation_rows(
        config.reference_root,
        source_logs,
        config.seeds,
        BRANCHES,
        config.uav_count,
        config.split,
    )
    report = _reason_and_report(
        config,
        targets,
        target_rows,
        target_status,
        training_rows,
        training_meta,
        validation_rows,
        isolation_rows,
        isolation_meta,
        source_logs,
    )
    plot = _plot(validation_rows, training_rows, plots_dir)

    _write_csv(config.output_dir / "target_recomputation.csv", target_rows)
    _write_csv(config.output_dir / "training_constraint_audit.csv", training_rows)
    _write_csv(config.output_dir / "validation_rate_audit.csv", validation_rows)
    _write_csv(config.output_dir / "branch_isolation_audit.csv", isolation_rows)
    _write_json(config.output_dir / "audit_report.json", {**report, "plot": str(plot)})
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
    "BRANCHES",
    "DEFAULT_SEEDS",
    "MeasurementRateConstraintAuditConfig",
    "RATE_BRANCH",
    "run_measurement_rate_constraint_audit",
]
