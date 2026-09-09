"""Validation-only audit of cooperative UWB and subsequent VO interaction.

This module deliberately does not replay an environment or update a policy.
It reads only the validation split of a completed development comparison and
turns the saved EKF event/diagnostic rows into persistence and next-sensor
evidence.  In particular, it must not be used to fit a scale or select a
checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = (
    "local_only_fixed",
    "random_cooperative",
    "budgeted_greedy_ig",
    "baseline_mappo",
)
METHOD_LABELS = {
    "local_only_fixed": "Local",
    "random_cooperative": "Random",
    "budgeted_greedy_ig": "Greedy IG",
    "baseline_mappo": "Local-EKF-GAT-MAPPO",
}
HORIZONS = (1, 2, 3, 5)
BENEFIT_THRESHOLD_M = 0.001
TOL = 1e-9


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any, *, allow_na: bool = False) -> float:
    if allow_na and (value is None or str(value).strip().upper() in {"", "NA", "N/A", "NAN"}):
        return float("nan")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def _event_class(event: Mapping[str, Any]) -> str:
    if not _truth(event.get("accepted")):
        return "rejected"
    delta = _number(event["true_position_error_after_m"]) - _number(event["true_position_error_before_m"])
    if delta < -BENEFIT_THRESHOLD_M:
        return "beneficial"
    if delta > BENEFIT_THRESHOLD_M:
        return "harmful"
    return "neutral"


def _validation_step(timestamp: Any, dt_s: float = 0.2) -> int:
    """Map a saved post-step timestamp to the zero-based EKF step."""

    return int(round(_number(timestamp) / dt_s)) - 1


def _next_sensor(mapping: Mapping[tuple[str, int], Mapping[str, Any]], target: str, step: int) -> str:
    for future_step in range(step + 1, step + 60):
        row = mapping.get((target, future_step))
        if row is None:
            continue
        if _truth(row.get("vo_update_accepted")):
            return "visual_odometry"
        if _truth(row.get("uwb_update_accepted")):
            return "onboard_uwb"
        if _truth(row.get("gnss_update_accepted")):
            return "gnss"
    return "none_observed"


def _assert_validation_path(path: Path, *, method_root: Path | None = None) -> None:
    lowered = {part.lower() for part in path.parts}
    if "test" in lowered:
        raise RuntimeError(f"VO interaction audit attempted to access test data: {path}")
    if method_root is not None:
        try:
            relative = path.resolve().relative_to(method_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"path is outside method root: {path}") from exc
        if not relative.parts or relative.parts[0].lower() != "validation" and "validation" not in lowered:
            raise RuntimeError(f"raw VO audit path is not validation data: {path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    _assert_validation_path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _assert_validation_path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _raw_episode_dirs(method_root: Path) -> list[Path]:
    validation_root = method_root / "validation"
    _assert_validation_path(validation_root, method_root=method_root)
    episodes = sorted(path for path in validation_root.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no validation episodes under {validation_root}")
    for path in episodes:
        _assert_validation_path(path, method_root=method_root)
    return episodes


def _event_path(method_root: Path) -> Path:
    root_event_path = method_root / "diagnostics" / "validation_cooperative_updates.jsonl"
    if root_event_path.is_file() and root_event_path.stat().st_size:
        return root_event_path
    # Rule methods also keep the validation event log in the validation
    # episode tree.  The current comparison protocol normally uses the root
    # diagnostic path, but this fallback keeps the audit schema explicit.
    episode_paths = sorted(method_root.glob("validation/episode_*/validation_cooperative_updates.jsonl"))
    if episode_paths:
        return episode_paths[0]
    return root_event_path


def _error_mappings(method_root: Path, episodes: Sequence[Path]) -> dict[str, dict[tuple[str, int], dict[str, str]]]:
    mappings: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for episode in episodes:
        path = episode / "ekf_diagnostics.csv"
        _assert_validation_path(path, method_root=method_root)
        mapping: dict[tuple[str, int], dict[str, str]] = {}
        for row in _read_csv(path):
            mapping[(str(row["uav_id"]), _validation_step(row["timestamp"]))] = row
        mappings[episode.name] = mapping
    return mappings


def _analyse_method(method: str, method_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = _raw_episode_dirs(method_root)
    event_path = _event_path(method_root)
    events = _read_jsonl(event_path) if event_path.is_file() and event_path.stat().st_size else []
    mappings = _error_mappings(method_root, episodes)
    accepted = [row for row in events if str(row.get("split", "validation")) == "validation" and _truth(row.get("accepted"))]
    classes = Counter(_event_class(row) for row in accepted)

    next_sensors = [
        _next_sensor(
            mappings.get(str(event["episode_id"]), {}),
            str(event["target_uav_id"]),
            int(event["step"]),
        )
        for event in accepted
    ]
    sensor_counts = Counter(next_sensors)

    retention: dict[str, Any] = {}
    for horizon in HORIZONS:
        observed = 0
        retained = 0
        censored = 0
        for event in accepted:
            if _event_class(event) != "beneficial":
                continue
            mapping = mappings.get(str(event["episode_id"]), {})
            future = mapping.get((str(event["target_uav_id"]), int(event["step"]) + horizon))
            if future is None:
                censored += 1
                continue
            observed += 1
            baseline = _number(event["true_position_error_before_m"])
            retained += int(_number(future["position_error_m"]) < baseline - BENEFIT_THRESHOLD_M)
        retention[f"h{horizon}_beneficial_count"] = classes["beneficial"]
        retention[f"h{horizon}_observed_count"] = observed
        retention[f"h{horizon}_censored_count"] = censored
        retention[f"h{horizon}_retained_fraction"] = retained / observed if observed else float("nan")

    restored = sum(_truth(event.get("covariance_restored")) for event in accepted)
    jloc_deltas = [abs(_number(event["j_loc_after"]) - _number(event["j_loc_before"])) for event in accepted]
    row: dict[str, Any] = {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "validation_episode_count": len(episodes),
        "event_log": str(event_path),
        "accepted_events": len(accepted),
        "beneficial_events": classes["beneficial"],
        "neutral_events": classes["neutral"],
        "harmful_events": classes["harmful"],
        "beneficial_fraction": classes["beneficial"] / len(accepted) if accepted else float("nan"),
        "neutral_fraction": classes["neutral"] / len(accepted) if accepted else float("nan"),
        "harmful_fraction": classes["harmful"] / len(accepted) if accepted else float("nan"),
        "covariance_restored_fraction": restored / len(accepted) if accepted else float("nan"),
        "mean_abs_jloc_delta": _finite_mean(jloc_deltas),
        "next_sensor_observed_count": len(next_sensors),
        "next_visual_odometry_count": sensor_counts["visual_odometry"],
        "next_visual_odometry_fraction": sensor_counts["visual_odometry"] / len(next_sensors) if next_sensors else float("nan"),
        "next_onboard_uwb_fraction": sensor_counts["onboard_uwb"] / len(next_sensors) if next_sensors else float("nan"),
        "next_none_observed_fraction": sensor_counts["none_observed"] / len(next_sensors) if next_sensors else float("nan"),
        **retention,
    }
    if math.isfinite(row["h1_retained_fraction"]) and math.isfinite(row["h5_retained_fraction"]):
        persistence_status = "confirmed" if row["h5_retained_fraction"] < row["h1_retained_fraction"] else "inconclusive"
    else:
        persistence_status = "inconclusive"
    if math.isfinite(row["next_visual_odometry_fraction"]):
        vo_status = "confirmed" if row["next_visual_odometry_fraction"] > 0.9 else "inconclusive"
    else:
        vo_status = "inconclusive"
    jloc_status = (
        "confirmed"
        if math.isfinite(row["covariance_restored_fraction"])
        and row["covariance_restored_fraction"] > 0.99
        and math.isfinite(row["mean_abs_jloc_delta"])
        and row["mean_abs_jloc_delta"] < TOL
        else "inconclusive"
    )
    findings = [
        {
            "method": method,
            "finding": "ekf_benefit_not_persistent",
            "status": persistence_status,
            "evidence": {"h1": row["h1_retained_fraction"], "h5": row["h5_retained_fraction"]},
            "interpretation": "beneficial UWB event improvement fades over later validation windows",
        },
        {
            "method": method,
            "finding": "visual_odometry_overwrites_next_transition",
            "status": vo_status,
            "evidence": {"fraction": row["next_visual_odometry_fraction"], "count": row["next_visual_odometry_count"]},
            "interpretation": "the first subsequent accepted update is overwhelmingly visual odometry",
        },
        {
            "method": method,
            "finding": "jloc_objective_insensitive",
            "status": jloc_status,
            "evidence": {
                "covariance_restored_fraction": row["covariance_restored_fraction"],
                "mean_abs_jloc_delta": row["mean_abs_jloc_delta"],
            },
            "interpretation": "J_loc remains a covariance diagnostic and is not a sensitive proxy for mean-error change",
        },
    ]
    sensor_rows = [
        {
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "next_sensor": sensor,
            "count": count,
            "fraction": count / len(next_sensors) if next_sensors else float("nan"),
        }
        for sensor, count in sorted(sensor_counts.items())
    ]
    return {"summary": row, "findings": findings}, sensor_rows


def _cell(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else "NA"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, default=str)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    return value


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty VO audit table: {path}")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _plot(summary_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [str(row["label"]) for row in summary_rows]
    x = np.arange(len(labels))
    h1 = np.asarray([float(row["h1_retained_fraction"]) if math.isfinite(float(row["h1_retained_fraction"])) else np.nan for row in summary_rows])
    h5 = np.asarray([float(row["h5_retained_fraction"]) if math.isfinite(float(row["h5_retained_fraction"])) else np.nan for row in summary_rows])
    vo = np.asarray([float(row["next_visual_odometry_fraction"]) if math.isfinite(float(row["next_visual_odometry_fraction"])) else np.nan for row in summary_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, h1, width, label="h1")
    axes[0].bar(x + width / 2, h5, width, label="h5")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction retaining mean-error improvement")
    axes[0].set_title("Beneficial UWB-update persistence")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, vo, color="#9467bd")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("First accepted update after UWB")
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run(
    *,
    data_root: Path = Path("data/synthetic/mainline_core_framework_v4_delta_scale_seed42"),
    logs_root: Path = Path("logs/preliminary/mainline_core_framework_v4_delta_scale_seed42"),
    plots_root: Path = Path("plots/preliminary/mainline_core_framework_v4_delta_scale_seed42"),
    output_name: str = "vo_interaction_audit",
) -> dict[str, Any]:
    """Run a read-only validation-only VO interaction audit."""

    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    source_log_root = logs_root / "development" / "loc-progress-v2"
    report_path = source_log_root / "audit_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"completed v2 development report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    method_paths = report.get("method_paths", {})
    missing = [method for method in METHOD_ORDER if method not in method_paths]
    if missing:
        raise ValueError(f"v2 development report is missing methods: {missing}")

    output_logs = source_log_root / output_name
    output_plots = plots_root / "development" / "loc-progress-v2" / output_name
    for path in (output_logs, output_plots):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty VO audit directory: {path}")
        path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    resolved_paths: dict[str, str] = {}
    for method in METHOD_ORDER:
        raw_path = Path(str(method_paths[method]))
        method_root = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
        method_root = method_root.resolve()
        try:
            method_root.relative_to(data_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"method path is outside data root: {method_root}") from exc
        result, method_sensors = _analyse_method(method, method_root)
        summary_rows.append(result["summary"])
        finding_rows.extend(result["findings"])
        sensor_rows.extend(method_sensors)
        resolved_paths[method] = str(method_root)

    _write_rows(output_logs / "vo_interaction_summary.csv", summary_rows)
    _write_rows(output_logs / "vo_next_sensor.csv", sensor_rows)
    _write_rows(output_logs / "vo_interaction_findings.csv", finding_rows)
    figure_path = output_plots / "vo_interaction_persistence.png"
    _plot(summary_rows, figure_path)

    output_report = {
        "schema_version": "vo-interaction-validation-audit-v1",
        "status": "complete",
        "audit_ok": True,
        "source_protocol": report.get("protocol"),
        "source_stage": "development",
        "source_arm": "loc-progress-v2",
        "source_data_root": str(data_root),
        "source_logs_root": str(source_log_root),
        "source_splits_read": ["validation"],
        "test_access_attempts": 0,
        "test_accessed": False,
        "training_updates_executed": False,
        "checkpoint_modified": False,
        "methods": list(METHOD_ORDER),
        "method_paths": resolved_paths,
        "summaries": summary_rows,
        "findings": finding_rows,
        "evidence_limits": [
            "validation-only single-seed evidence; no cross-initialization stability inference",
            "retention is censored when the requested future window is absent",
            "truth-derived position errors classify update quality and are audit labels, not actor inputs",
            "a subsequent VO update is evidence of interaction, not proof that VO overwrote all UWB information",
        ],
    }
    (output_logs / "vo_interaction_report.json").write_text(
        json.dumps(_jsonable(output_report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_logs / "figure_index.md").write_text(
        "# Validation-only VO interaction audit\n\n"
        "Only validation episodes were read; test was not opened and no policy/checkpoint was modified.\n\n"
        "- `vo_interaction_persistence.png/.pdf`: h1/h5 beneficial-update retention and the first accepted sensor after each accepted UWB update.\n"
        "- `vo_interaction_summary.csv`: per-method event counts, retention, next-sensor fractions, and J_loc sensitivity.\n"
        "- `vo_next_sensor.csv`: next accepted sensor count/fraction.\n"
        "- `vo_interaction_findings.csv`: confirmed or inconclusive mechanism findings.\n",
        encoding="utf-8",
    )
    return {
        "audit_ok": True,
        "status": "complete",
        "report": str(output_logs / "vo_interaction_report.json"),
        "summary": str(output_logs / "vo_interaction_summary.csv"),
        "figure": str(figure_path),
        "test_access_attempts": 0,
    }


__all__ = ["run"]
