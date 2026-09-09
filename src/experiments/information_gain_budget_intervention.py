"""Validation-only information-gain and measurement-budget intervention.

This module replays the frozen ``baseline_mappo`` branch of the five-seed
quality-aware MAPPO experiment.  It changes only ``measurement_uav_id`` and
keeps bandwidth, velocity, scenario, sensor randomness, and communication
randomness paired across treatments.  The information-gain score is computed
from execution-time observable estimates, covariances, UWB noise metadata and
link-quality fields; truth labels are used only for offline diagnostics that
are written after the action has been selected.

The independent statistical unit is the base random seed.  Validation
episodes are paired within seed, while windows and EKF events are diagnostic
records rather than independent observations.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_json
from .bandwidth_intervention import _max_numeric_difference, load_recorded_actions
from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .cooperative_update_quality_audit import _quality_event_row
from .execution_uncertainty_proxy_audit import (
    ProxyScales,
    _fit_scales,
    _link_quality,
    _normalized,
    _raw_information_gain,
    _candidate_rows_from_episode,
)
from .formal import _copy_calibration_files
from .localization_coupling_intervention import _bootstrap_interval, _mean, _write_table
from .measurement_density_intervention import actor_visible_measurement_candidates
from .measurement_density_intervention import _bandwidth_difference
from .pilot import _EpisodeTraceWriter, _assert_action_feasible, _assert_observation_finite, _seed_everything
from .quality_aware_measurement_intervention import (
    QualityAwareMeasurementInterventionConfig,
    _run_episode as _run_replay_episode,
)


TREATMENTS = (
    "recorded_mappo",
    "unconstrained_information_gain",
    "budgeted_information_gain",
)
# Kept as the backwards-compatible default for callers that do not provide an
# explicit seed list.  New experiments should pass ``seeds`` explicitly.
EXPECTED_BASE_SEEDS = (2026, 3035, 4044, 5053, 6062)
SEED_RE = re.compile(r"^seed_(\d+)$")
NUMERIC_TOLERANCE = 2e-6
EPS = 1e-9
POLICY_PREFIX = "information_gain_budget_intervention"

OVERHEAD_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "scenario_id",
    "seed",
    "base_seed",
    "split",
    "treatment",
    "step",
    "timestamp",
    "source_uav_id",
    "target_uav_id",
    "selected_non_none",
    "overhead_mode",
    "payload_size_bytes",
    "load_packets_requested",
    "load_packets_applied",
    "potential_edge",
    "candidate_edge",
    "effective_edge",
    "queue_before_packets",
    "queue_with_measurement_packets",
    "queue_next_packets",
    "channel_bandwidth_hz",
    "rate_bps",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "link_state",
    "ekf_behavior_unchanged",
)


@dataclass(frozen=True)
class InformationGainBudgetInterventionConfig:
    """Configuration for the validation-only paired intervention."""

    reference_root: Path
    reference_branch: str = "baseline_mappo"
    output_dir: Path = Path("data/synthetic/python_sim_information_gain_budget_intervention_seed2026")
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    age_variance_coefficient: float = 1.0
    seeds: tuple[int, ...] | None = None
    measurement_overhead_enabled: bool = False
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("information-gain budget intervention is validation-only")
        if self.reference_branch != "baseline_mappo":
            raise ValueError("the intervention must use baseline_mappo as the common reference")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not math.isfinite(self.age_variance_coefficient) or self.age_variance_coefficient < 0.0:
            raise ValueError("age_variance_coefficient must be finite and non-negative")
        if self.seeds is not None:
            normalized = tuple(sorted(int(seed) for seed in self.seeds))
            if len(normalized) != 5 or len(set(normalized)) != 5 or any(seed < 0 for seed in normalized):
                raise ValueError("seeds must contain exactly five unique non-negative integers")
            object.__setattr__(self, "seeds", normalized)


def _progress(config: InformationGainBudgetInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[information-gain-budget] {message}", flush=True)


def _default_logs_dir(reference_root: Path) -> Path:
    return Path("logs/preliminary") / reference_root.name / "information_gain_budget_intervention"


def _default_plots_dir(reference_root: Path) -> Path:
    return Path("plots/preliminary") / reference_root.name / "information_gain_budget_intervention"


def _prepare_outputs(config: InformationGainBudgetInterventionConfig) -> tuple[Path, Path]:
    root = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else _default_logs_dir(Path(config.reference_root))
    plots = Path(config.plots_dir) if config.plots_dir is not None else _default_plots_dir(Path(config.reference_root))
    for directory in (root, logs, plots):
        if directory.exists() and any(directory.iterdir()) and not config.force:
            raise FileExistsError(f"information-gain intervention output is not empty: {directory}; use --force")
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return logs, plots


def _uav_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(value))
    return (int(match.group(1)) if match else 10**9, str(value))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _scale_roots(reference_root: Path, branch: str) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for path in sorted(reference_root.glob("seed_*"), key=lambda item: item.name):
        match = SEED_RE.match(path.name)
        if not match:
            continue
        scale = path / branch / "scale_3"
        if scale.is_dir():
            roots[int(match.group(1))] = scale
    return roots


def _resolve_base_seeds(
    reference_root: Path,
    branch: str,
    requested: Sequence[int] | None,
) -> tuple[tuple[int, ...], dict[int, Path]]:
    """Resolve and validate the five seed branches without a fixed seed list."""

    roots = _scale_roots(reference_root, branch)
    discovered = tuple(sorted(roots))
    seeds = tuple(sorted(int(seed) for seed in requested)) if requested is not None else discovered
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError(f"information-gain intervention requires exactly five unique seeds; found {list(seeds)}")
    if set(roots) != set(seeds):
        raise ValueError(
            f"reference root must contain exactly requested seeds {list(seeds)}; found {list(discovered)}"
        )
    return seeds, roots


def _seedset_label(seeds: Sequence[int]) -> str:
    """Return a stable, human-readable label for an explicit seed set."""

    return "_".join(str(int(seed)) for seed in sorted(seeds))


def _load_manifest(scale_root: Path) -> dict[str, Any]:
    manifest = _read_json(scale_root / "manifest.json")
    if str(manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"{scale_root} must use {EPISODE_SCHEMA_VERSION}")
    if int(manifest.get("uav_count", 0)) != 3:
        raise ValueError(f"{scale_root} is not a three-UAV scale")
    return manifest


def _split_episodes(scale_root: Path, split: str) -> list[dict[str, Any]]:
    manifest = _load_manifest(scale_root)
    episodes = [
        dict(item)
        for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and str(item.get("split")) == split
    ]
    episodes.sort(key=lambda item: str(item.get("episode_id", "")))
    return episodes


def _train_budget(scale_root: Path, base_seed: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    train_root = scale_root / "train"
    episode_roots = sorted(path for path in train_root.glob("episode_*") if path.is_dir())
    if not episode_roots:
        raise ValueError(f"no training episodes under {train_root}")
    by_uav: dict[str, list[int]] = defaultdict(list)
    for episode_root in episode_roots:
        counts: dict[str, int] = defaultdict(int)
        for row in _read_csv(episode_root / "actions.csv"):
            uav_id = str(row.get("uav_id", ""))
            if uav_id and str(row.get("measurement_uav_id", "none")) != "none":
                counts[uav_id] += 1
        for uav_id in sorted(counts, key=_uav_sort_key):
            by_uav[uav_id].append(int(counts[uav_id]))
        # A missing UAV row in a malformed episode must not silently increase
        # its budget; infer the complete set from all action rows.
        all_uavs = sorted({str(row.get("uav_id", "")) for row in _read_csv(episode_root / "actions.csv") if row.get("uav_id")}, key=_uav_sort_key)
        for uav_id in all_uavs:
            if uav_id not in counts:
                by_uav[uav_id].append(0)
    budgets: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for uav_id in sorted(by_uav, key=_uav_sort_key):
        values = np.asarray(by_uav[uav_id], dtype=float)
        mean_count = float(values.mean()) if values.size else 0.0
        budget = int(max(0, round(mean_count)))
        budgets[uav_id] = budget
        rows.append({
            "base_seed": int(base_seed),
            "uav_id": uav_id,
            "source_split": "train",
            "training_episode_count": int(values.size),
            "baseline_measurement_count_mean": mean_count,
            "baseline_measurement_count_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "measurement_budget": budget,
        })
    return budgets, rows


def _fit_seed_proxy(scale_root: Path, base_seed: int, age_variance_coefficient: float) -> tuple[ProxyScales, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for episode_root in sorted((scale_root / "train").glob("episode_*")):
        if episode_root.is_dir():
            rows.extend(
                _candidate_rows_from_episode(
                    episode_root,
                    base_seed=base_seed,
                    treatment="baseline_mappo",
                    split="train",
                    include_offline_label=False,
                    age_variance_coefficient=age_variance_coefficient,
                )
            )
    if not rows:
        raise ValueError(f"training split has no proxy candidate rows under {scale_root}")
    return _fit_scales(rows, age_variance_coefficient), rows


def _safe_link_value(link: Mapping[str, Any], key: str, default: float) -> float:
    value = _number(link.get(key), default)
    return float(value) if math.isfinite(value) else float(default)


def _online_candidate_rows(
    observation: Mapping[str, Any],
    uav_ids: Sequence[str],
    scales: ProxyScales,
    measurement_std_m: float,
    age_variance_coefficient: float,
) -> list[dict[str, Any]]:
    """Build proxy rows from the current actor-visible observation only."""

    local_estimate = np.asarray(observation.get("estimate", ()), dtype=float).reshape(-1)
    local_covariance = np.asarray(observation.get("covariance", ()), dtype=float)
    if local_estimate.size < 3 or local_covariance.ndim != 2 or local_covariance.shape[0] < 3:
        return []
    own_position = local_estimate[:3]
    own_cov_diag = np.diag(local_covariance[:3, :3])
    if not np.isfinite(own_position).all() or not np.isfinite(own_cov_diag).all():
        return []
    candidates = actor_visible_measurement_candidates(observation, uav_ids)
    incoming = observation.get("candidate_incoming_links", {})
    if not isinstance(incoming, Mapping):
        incoming = {}
    messages = observation.get("neighbor_messages", {})
    if not isinstance(messages, Mapping):
        messages = {}
    rows: list[dict[str, Any]] = []
    rssi_mid = 0.5 * (scales.rssi_q05_dbm + scales.rssi_q95_dbm)
    snr_mid = 0.5 * (scales.snr_q05_db + scales.snr_q95_db)
    for source in sorted(candidates, key=_uav_sort_key):
        link = incoming.get(source, {})
        if not isinstance(link, Mapping):
            link = {}
        message = messages.get(source)
        neighbor_cov_diag: np.ndarray | None = None
        u_hat = np.full(3, 1.0 / math.sqrt(3.0), dtype=float)
        geometry_source = "isotropic_no_neighbor_estimate"
        if isinstance(message, Mapping):
            neighbor_estimate = np.asarray(message.get("estimate", ()), dtype=float).reshape(-1)
            neighbor_covariance = np.asarray(message.get("covariance", ()), dtype=float)
            if neighbor_estimate.size >= 3 and neighbor_covariance.ndim == 2 and neighbor_covariance.shape[0] >= 3:
                relative = neighbor_estimate[:3] - own_position
                relative_norm = float(np.linalg.norm(relative))
                candidate_cov_diag = np.diag(neighbor_covariance[:3, :3])
                if relative_norm > EPS and np.isfinite(candidate_cov_diag).all():
                    u_hat = relative / relative_norm
                    neighbor_cov_diag = np.maximum(candidate_cov_diag, 0.0)
                    geometry_source = "received_neighbor_estimate"
        row = {
            "own_position": own_position.copy(),
            "own_cov_diag": np.maximum(own_cov_diag, 0.0),
            "neighbor_cov_diag": neighbor_cov_diag,
            "u_hat": u_hat,
            "measurement_std_m": max(float(measurement_std_m), EPS),
            "message_age_s": max(_safe_link_value(link, "aoi_s", 0.0), 0.0),
            "rssi_dbm": _safe_link_value(link, "rssi_dbm", rssi_mid),
            "snr_db": _safe_link_value(link, "snr_db", snr_mid),
            "packet_loss_pred": np.clip(_safe_link_value(link, "packet_loss", 1.0), 0.0, 1.0),
            "delay_pred_s": max(_safe_link_value(link, "delay_s", scales.delay_q75_s), 0.0),
            "aoi_pred_s": max(_safe_link_value(link, "aoi_s", scales.age_q75_s), 0.0),
            "queue_length_pred": max(_safe_link_value(link, "queue_length_packets", scales.queue_q75_packets), 0.0),
            "link_state": str(link.get("link_state", "unknown")),
            "candidate_edge": _safe_link_value(link, "candidate_edge", 1.0),
            "source_uav_id": str(source),
            "geometry_source": geometry_source,
            "neighbor_covariance_available": int(neighbor_cov_diag is not None),
        }
        raw = _raw_information_gain(row, scales, age_variance_coefficient)
        normalized = _normalized(raw, scales.ig_q05, scales.ig_q95)
        row.update(
            link_quality_q=_link_quality(row, scales),
            expected_information_gain_raw=raw,
            expected_information_gain_norm=normalized,
            proxy_risk=(1.0 - normalized) if math.isfinite(normalized) else float("nan"),
            proxy_valid=int(math.isfinite(raw) and math.isfinite(normalized)),
        )
        rows.append(row)
    valid = [row for row in rows if int(row["proxy_valid"])]
    ordered = sorted(valid, key=lambda row: (-float(row["expected_information_gain_raw"]), _uav_sort_key(str(row["source_uav_id"]))))
    ranks = {str(row["source_uav_id"]): rank for rank, row in enumerate(ordered, start=1)}
    spread = 0.0
    if len(ordered) >= 2:
        values = [float(row["expected_information_gain_raw"]) for row in ordered]
        spread = max(values) - min(values)
    for row in rows:
        row["proxy_rank"] = int(ranks.get(str(row["source_uav_id"]), 0))
        row["window_proxy_spread_raw"] = float(spread)
    return rows


def intervene_information_gain_selection(
    recorded_actions: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    uav_ids: Sequence[str],
    treatment: str,
    context: Mapping[str, Any],
    cursors: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select the highest current proxy score while preserving other actions."""

    if treatment not in TREATMENTS:
        raise ValueError(f"unknown information-gain treatment: {treatment}")
    scales = context["proxy_scales"]
    if not isinstance(scales, ProxyScales):
        raise TypeError("context['proxy_scales'] must be ProxyScales")
    measurement_std_m = float(context["measurement_std_m"])
    age_variance_coefficient = float(context["age_variance_coefficient"])
    budgets = {str(key): int(value) for key, value in dict(context.get("budgets", {})).items()}
    actions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for uav_id in uav_ids:
        recorded = recorded_actions[uav_id]
        candidates = _online_candidate_rows(
            observations[uav_id], uav_ids, scales, measurement_std_m, age_variance_coefficient
        )
        ordered = sorted(
            [row for row in candidates if int(row["proxy_valid"])],
            key=lambda row: (-float(row["expected_information_gain_raw"]), _uav_sort_key(str(row["source_uav_id"]))),
        )
        recorded_source = str(recorded.get("measurement_uav_id", "none"))
        selected = recorded_source
        used_before = int(cursors.get(uav_id, 0)) if treatment == "budgeted_information_gain" else -1
        budget = int(budgets.get(uav_id, -1)) if treatment == "budgeted_information_gain" else -1
        if treatment == "unconstrained_information_gain":
            selected = str(ordered[0]["source_uav_id"]) if ordered else "none"
        elif treatment == "budgeted_information_gain":
            if used_before < budget and ordered:
                selected = str(ordered[0]["source_uav_id"])
                cursors[uav_id] = used_before + 1
            else:
                selected = "none"
        selected_row = next((row for row in ordered if str(row["source_uav_id"]) == selected), None)
        used_after = int(cursors.get(uav_id, used_before)) if treatment == "budgeted_information_gain" else -1
        candidate_payload = {
            "candidates": [_json_safe(row) for row in candidates],
            "selection": {
                "expected_information_gain_raw": float(selected_row["expected_information_gain_raw"]) if selected_row else 0.0,
                "expected_information_gain_norm": float(selected_row["expected_information_gain_norm"]) if selected_row else 0.0,
                "proxy_risk": float(selected_row["proxy_risk"]) if selected_row else 1.0,
                "proxy_rank": int(selected_row["proxy_rank"]) if selected_row else 0,
                "window_proxy_spread_raw": float(candidates[0]["window_proxy_spread_raw"]) if candidates else 0.0,
                "budget": budget,
                "budget_used_before": used_before,
                "budget_used_after": used_after,
                "budget_exhausted": int(treatment == "budgeted_information_gain" and selected == "none" and used_before >= budget),
            },
        }
        actions[uav_id] = {
            "measurement_uav_id": selected,
            "bandwidth": {str(key): float(value) for key, value in dict(recorded.get("bandwidth", {})).items()},
            "delta_v": np.asarray(recorded["delta_v"], dtype=float).reshape(3),
        }
        metadata[uav_id] = {
            "candidate_count": len(candidates),
            "candidate_uav_ids": [str(row["source_uav_id"]) for row in candidates],
            "candidate_available": bool(candidates),
            "eligible_candidate_count": len(ordered),
            "recorded_measurement_uav_id": recorded_source,
            "selected_measurement_uav_id": selected,
            "selected_non_none": selected != "none",
            "selection_changed": selected != recorded_source,
            "selected_quality_risk_score": float(selected_row["proxy_risk"]) if selected_row else 1.0,
            "measurement_budget": budget,
            "budget_used_before": used_before,
            "budget_used_after": used_after,
            "budget_exhausted": int(
                treatment == "budgeted_information_gain" and selected == "none" and used_before >= budget
            ),
            "candidate_quality_json": json.dumps(candidate_payload, sort_keys=True, allow_nan=False),
            "actor_visible_only": True,
            "uses_physical_los": False,
            "uses_truth_error": False,
            "uses_future_information": False,
        }
    return actions, metadata


def _enrich_window_rows(rows: Sequence[Mapping[str, Any]], base_seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        row["base_seed"] = int(base_seed)
        row["episode_seed"] = int(_number(row.get("seed"), -1))
        payload: Mapping[str, Any] = {}
        try:
            decoded = json.loads(str(row.get("candidate_quality_json", "{}")))
            if isinstance(decoded, Mapping):
                payload = decoded
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        selection = payload.get("selection", {}) if isinstance(payload.get("selection", {}), Mapping) else {}
        selected = str(row.get("selected_measurement_uav_id", "none"))
        row.update(
            expected_information_gain_raw=float(selection.get("expected_information_gain_raw", 0.0)),
            expected_information_gain_norm=float(selection.get("expected_information_gain_norm", 0.0)),
            proxy_risk=float(selection.get("proxy_risk", 1.0)),
            proxy_rank=int(selection.get("proxy_rank", 0)),
            window_proxy_spread_raw=float(selection.get("window_proxy_spread_raw", 0.0)),
            measurement_budget=int(selection.get("budget", -1)),
            budget_used_before=int(selection.get("budget_used_before", -1)),
            budget_used_after=int(selection.get("budget_used_after", -1)),
            budget_exhausted=int(selection.get("budget_exhausted", 0)),
            selected_proxy_valid=int(selected != "none" and int(selection.get("proxy_rank", 0)) > 0),
            information_gain_selection_leakage_free=1,
        )
        output.append(row)
    return output


def _episode_metrics_with_intervention(
    metrics: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    base_seed: int,
    treatment: str,
    budgets: Mapping[str, int],
) -> dict[str, Any]:
    row = dict(metrics)
    row.update(
        base_seed=int(base_seed),
        treatment=treatment,
        episode_seed=int(_number(row.get("seed"), -1)),
        measurement_count_total=int(sum(int(r.get("selected_non_none", 0)) for r in windows)),
        measurement_candidate_windows=int(sum(int(r.get("candidate_available", 0)) for r in windows)),
        measurement_budget_total=int(sum(int(value) for value in budgets.values())),
    )
    accepted = [event for event in events if int(_number(event.get("accepted"), 0)) == 1]
    row["accepted_update_count"] = int(len(accepted))
    total = max(1, len(accepted))
    row["beneficial_fraction"] = float(sum(event.get("effect_class") == "beneficial" for event in accepted) / total)
    row["harmful_fraction"] = float(sum(event.get("effect_class") == "harmful" for event in accepted) / total)
    row["neutral_fraction"] = float(sum(event.get("effect_class") == "approximately_neutral" for event in accepted) / total)
    row["nlos_update_fraction"] = float(_mean(1.0 - _number(event.get("los"), 1.0) for event in accepted))
    row["stale_update_fraction"] = float(_mean(_number(event.get("message_age_s"), 0.0) > 0.5 for event in accepted))
    return row


def _seed_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["base_seed"]), str(row["treatment"]))].append(row)
    keys = (
        "mean_reward", "j_loc", "j_com", "joint_objective", "position_rmse_m",
        "position_error_p95_m", "covariance_position_trace", "accepted_update_count",
        "beneficial_fraction", "harmful_fraction", "neutral_fraction", "packet_loss",
        "delay_s", "aoi_s", "connectivity", "energy_proxy", "collision_rate",
        "measurement_count_total", "measurement_budget_total", "measurement_candidate_windows",
    )
    output: list[dict[str, Any]] = []
    for (base_seed, treatment), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "base_seed": base_seed,
            "treatment": treatment,
            "validation_episode_count": len(group),
        }
        for key in keys:
            values = [_number(row.get(key), 0.0) for row in group]
            item[key] = float(np.mean(values)) if values else 0.0
        output.append(item)
    return output


def _paired_contrasts(seed_rows: Sequence[Mapping[str, Any]], config: InformationGainBudgetInterventionConfig) -> list[dict[str, Any]]:
    metrics = (
        "position_rmse_m", "position_error_p95_m", "j_loc", "j_com", "joint_objective",
        "mean_reward", "accepted_update_count", "beneficial_fraction", "harmful_fraction",
        "packet_loss", "delay_s", "aoi_s", "connectivity", "energy_proxy", "measurement_count_total",
    )
    by_key = {(int(row["base_seed"]), str(row["treatment"])): row for row in seed_rows}
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.bootstrap_seed)
    for treatment in TREATMENTS[1:]:
        for metric in metrics:
            baseline = np.asarray([_number(by_key[(seed, "recorded_mappo")].get(metric), 0.0) for seed in seeds], dtype=float)
            comparator = np.asarray([_number(by_key[(seed, treatment)].get(metric), 0.0) for seed in seeds], dtype=float)
            difference = comparator - baseline
            low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
            output.append({
                "contrast": f"{treatment}_minus_recorded_mappo",
                "treatment": treatment,
                "metric": metric,
                "independent_unit": "base_seed",
                "paired_seed_count": len(seeds),
                "comparator_mean": float(baseline.mean()),
                "treatment_mean": float(comparator.mean()),
                "mean_paired_difference": float(difference.mean()),
                "paired_difference_std": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                "bootstrap_95_ci_low": float(low),
                "bootstrap_95_ci_high": float(high),
                "ci_excludes_zero": int(low > 0.0 or high < 0.0),
            })
    return output


def _numeric_difference_for_pair(first: Path, second: Path) -> float:
    values = (
        ("ground_truth.csv", ("timestamp", "uav_id"), ("x", "y", "z", "vx", "vy", "vz")),
        ("obstacles.csv", ("timestamp", "obstacle_id"), ("x", "y", "z", "vx", "vy", "vz", "radius")),
        ("links.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length")),
        ("packet_events.csv", ("timestamp", "src_uav_id", "dst_uav_id"), ("success", "delay", "aoi_after")),
        ("actions.csv", ("timestamp", "uav_id"), ("delta_vx", "delta_vy", "delta_vz")),
    )
    maximum = 0.0
    for filename, keys, columns in values:
        maximum = max(maximum, _max_numeric_difference(first / filename, second / filename, key_columns=keys, value_columns=columns))
    maximum = max(maximum, _bandwidth_difference(first / "actions.csv", second / "actions.csv"))
    return float(maximum)


def _invariance_checks(
    reference_roots: Mapping[int, Path],
    treatment_roots: Mapping[tuple[int, str], Path],
    episodes_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    checks = {
        "recorded_reproduces_reference_max_abs_difference": 0.0,
        "cross_treatment_truth_max_abs_difference": 0.0,
        "cross_treatment_obstacle_max_abs_difference": 0.0,
        "cross_treatment_delta_v_max_abs_difference": 0.0,
        "cross_treatment_bandwidth_max_abs_difference": 0.0,
    }
    for base_seed, episodes in episodes_by_seed.items():
        reference_root = reference_roots[base_seed]
        factual_root = treatment_roots[(base_seed, "recorded_mappo")]
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            reference = reference_root / "validation" / episode_id
            factual = factual_root / "validation" / episode_id
            checks["recorded_reproduces_reference_max_abs_difference"] = max(
                checks["recorded_reproduces_reference_max_abs_difference"],
                _numeric_difference_for_pair(factual, reference),
                _max_numeric_difference(factual / "estimates.csv", reference / "estimates.csv", key_columns=("timestamp", "uav_id"), value_columns=("x", "y", "z", "vx", "vy", "vz")),
                _max_numeric_difference(factual / "rewards.csv", reference / "rewards.csv", key_columns=("timestamp", "uav_id"), value_columns=("reward_total", "joint_objective", "normalized_uncertainty")),
            )
            for treatment in TREATMENTS[1:]:
                candidate = treatment_roots[(base_seed, treatment)] / "validation" / episode_id
                checks["cross_treatment_truth_max_abs_difference"] = max(
                    checks["cross_treatment_truth_max_abs_difference"],
                    _max_numeric_difference(factual / "ground_truth.csv", candidate / "ground_truth.csv", key_columns=("timestamp", "uav_id"), value_columns=("x", "y", "z", "vx", "vy", "vz")),
                )
                checks["cross_treatment_obstacle_max_abs_difference"] = max(
                    checks["cross_treatment_obstacle_max_abs_difference"],
                    _max_numeric_difference(factual / "obstacles.csv", candidate / "obstacles.csv", key_columns=("timestamp", "obstacle_id"), value_columns=("x", "y", "z", "vx", "vy", "vz", "radius")),
                )
                checks["cross_treatment_delta_v_max_abs_difference"] = max(
                    checks["cross_treatment_delta_v_max_abs_difference"],
                    _max_numeric_difference(factual / "actions.csv", candidate / "actions.csv", key_columns=("timestamp", "uav_id"), value_columns=("delta_vx", "delta_vy", "delta_vz")),
                )
                checks["cross_treatment_bandwidth_max_abs_difference"] = max(
                    checks["cross_treatment_bandwidth_max_abs_difference"],
                    _bandwidth_difference(factual / "actions.csv", candidate / "actions.csv"),
                )
    return checks


def _communication_difference_checks(
    treatment_roots: Mapping[tuple[int, str], Path],
    episodes_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    """Measure communication changes without treating them as invariants.

    Measurement overhead is intentionally allowed to change queue-derived
    communication fields.  These values are reported for diagnosis and are
    not included in the common-random-number invariance gate.
    """

    checks = {
        "cross_treatment_link_kpi_max_abs_difference": 0.0,
        "cross_treatment_packet_event_max_abs_difference": 0.0,
    }
    for base_seed, episodes in episodes_by_seed.items():
        factual_root = treatment_roots[(base_seed, "recorded_mappo")]
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            factual = factual_root / "validation" / episode_id
            for treatment in TREATMENTS[1:]:
                candidate = treatment_roots[(base_seed, treatment)] / "validation" / episode_id
                checks["cross_treatment_link_kpi_max_abs_difference"] = max(
                    checks["cross_treatment_link_kpi_max_abs_difference"],
                    _max_numeric_difference(
                        factual / "links.csv",
                        candidate / "links.csv",
                        key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                        value_columns=("rx_power_dbm", "rssi", "rate_bps", "packet_loss", "delay", "aoi", "queue_length"),
                    ),
                )
                checks["cross_treatment_packet_event_max_abs_difference"] = max(
                    checks["cross_treatment_packet_event_max_abs_difference"],
                    _max_numeric_difference(
                        factual / "packet_events.csv",
                        candidate / "packet_events.csv",
                        key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                        value_columns=("success", "delay", "aoi_after"),
                    ),
                )
    return checks


def _measurement_overhead_audit(
    overhead_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Check one 512-byte equivalent load for each selected measurement."""

    numeric_fields = (
        "payload_size_bytes",
        "load_packets_requested",
        "load_packets_applied",
        "queue_before_packets",
        "queue_with_measurement_packets",
        "queue_next_packets",
        "channel_bandwidth_hz",
        "rate_bps",
        "packet_loss",
        "delay_s",
        "aoi_s",
    )
    finite = all(_finite(row.get(field)) for row in overhead_rows for field in numeric_fields)
    row_semantics = all(
        int(_number(row.get("selected_non_none"), 0.0)) == 1
        and int(_number(row.get("payload_size_bytes"), -1.0)) == 512
        and abs(_number(row.get("load_packets_requested"), float("nan")) - 1.0) <= NUMERIC_TOLERANCE
        and _number(row.get("load_packets_applied"), -1.0) >= -NUMERIC_TOLERANCE
        for row in overhead_rows
    )
    selected_by_action: dict[tuple[int, str, str, int, str], int] = defaultdict(int)
    for row in window_rows:
        if int(_number(row.get("selected_non_none"), 0.0)) != 1:
            continue
        selected_by_action[
            (
                int(_number(row.get("base_seed"), -1.0)),
                str(row.get("treatment", "")),
                str(row.get("episode_id", "")),
                int(_number(row.get("step"), -1.0)),
                str(row.get("uav_id", "")),
            )
        ] += 1
    overhead_by_action: dict[tuple[int, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in overhead_rows:
        overhead_by_action[
            (
                int(_number(row.get("base_seed"), -1.0)),
                str(row.get("treatment", "")),
                str(row.get("episode_id", "")),
                int(_number(row.get("step"), -1.0)),
                str(row.get("target_uav_id", "")),
            )
        ].append(row)
    correspondence_residual = 0.0
    action_keys = set(selected_by_action) | set(overhead_by_action)
    for key in action_keys:
        requested = float(selected_by_action.get(key, 0))
        rows = overhead_by_action.get(key, [])
        applied_requested = float(sum(_number(row.get("load_packets_requested"), 0.0) for row in rows))
        correspondence_residual = max(correspondence_residual, abs(requested - applied_requested))
        if requested > 0.0 and len(rows) != int(requested):
            correspondence_residual = max(correspondence_residual, 1.0)
    if not enabled:
        correspondence_ok = len(overhead_rows) == 0
    else:
        correspondence_ok = correspondence_residual <= NUMERIC_TOLERANCE and finite and row_semantics
    return {
        "overhead_enabled": bool(enabled),
        "overhead_row_count": len(overhead_rows),
        "selected_measurement_count": int(sum(selected_by_action.values())),
        "overhead_values_finite": bool(finite),
        "overhead_row_semantics_ok": bool(row_semantics),
        "overhead_selection_correspondence_max_abs_difference": float(correspondence_residual),
        "overhead_correspondence_ok": bool(correspondence_ok),
    }


def _budget_usage(window_rows: Sequence[Mapping[str, Any]], budgets: Mapping[int, Mapping[str, int]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, str], int] = defaultdict(int)
    for row in window_rows:
        if int(row.get("selected_non_none", 0)):
            groups[(int(row["base_seed"]), str(row["treatment"]), str(row["episode_id"]), str(row["uav_id"]))] += 1
    output: list[dict[str, Any]] = []
    for base_seed in sorted(budgets):
        for treatment in TREATMENTS:
            for uav_id, budget in sorted(budgets[base_seed].items(), key=lambda item: _uav_sort_key(item[0])):
                values = [count for (seed, item_treatment, _episode, item_uav), count in groups.items() if seed == base_seed and item_treatment == treatment and item_uav == uav_id]
                max_used = max(values or [0])
                mean_used = float(np.mean(values)) if values else 0.0
                output.append({
                    "base_seed": base_seed,
                    "treatment": treatment,
                    "uav_id": uav_id,
                    "validation_episode_count": len(values),
                    "measurement_budget": int(budget) if treatment == "budgeted_information_gain" else -1,
                    "mean_measurement_count": mean_used,
                    "max_measurement_count": int(max_used),
                    "max_budget_excess": int(max(0, max_used - budget)) if treatment == "budgeted_information_gain" else 0,
                    "mean_budget_usage_fraction": mean_used / max(1, budget) if treatment == "budgeted_information_gain" else -1.0,
                })
    return output


def _plot(seed_rows: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]], budgets: Sequence[Mapping[str, Any]], plots_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "recorded_mappo": "Recorded MAPPO",
        "unconstrained_information_gain": "Unconstrained IG",
        "budgeted_information_gain": "Budgeted IG",
    }
    colors = {
        "recorded_mappo": "#2563eb",
        "unconstrained_information_gain": "#dc2626",
        "budgeted_information_gain": "#16a34a",
    }
    by_key = {(int(row["base_seed"]), str(row["treatment"])): row for row in seed_rows}
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    treatment_x = np.arange(len(TREATMENTS))

    def panel(axis: Any, metric: str, title: str, ylabel: str | None = None) -> None:
        for seed in seeds:
            axis.plot(treatment_x, [float(by_key[(seed, treatment)][metric]) for treatment in TREATMENTS], color="#94a3b8", alpha=0.35, linewidth=0.8)
        means = [float(np.mean([float(by_key[(seed, treatment)][metric]) for seed in seeds])) for treatment in TREATMENTS]
        axis.scatter(treatment_x, means, c=[colors[treatment] for treatment in TREATMENTS], s=60, zorder=3)
        axis.set_xticks(treatment_x, [labels[treatment] for treatment in TREATMENTS], rotation=18, ha="right")
        axis.set_title(title)
        if ylabel:
            axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    panel(axes[0, 0], "measurement_count_total", "Measurements per validation episode", "count")
    budget_total = float(np.mean([float(row["measurement_budget_total"]) for row in seed_rows if row["treatment"] == "recorded_mappo"])) if seed_rows else 0.0
    axes[0, 0].axhline(budget_total, color="#111827", linestyle="--", linewidth=1, label="Baseline train budget")
    axes[0, 0].legend(fontsize=8)
    panel(axes[0, 1], "position_rmse_m", "Position RMSE", "m")
    panel(axes[0, 2], "j_loc", r"$J_{loc}$", "cost")
    panel(axes[1, 0], "j_com", r"$J_{com}$", "cost")
    panel(axes[1, 1], "mean_reward", "Team reward", "reward")
    panel(axes[1, 2], "harmful_fraction", "Harmful accepted-update fraction", "fraction")
    fig.suptitle("Information-gain budget intervention; baseline-MAPPO validation only; lines pair identical seeds")
    target = plots_dir / "information_gain_budget_intervention.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def _audit_report(
    config: InformationGainBudgetInterventionConfig,
    reference_roots: Mapping[int, Path],
    treatment_roots: Mapping[tuple[int, str], Path],
    episodes_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    seed_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    budget_usage: Sequence[Mapping[str, Any]],
    overhead_rows: Sequence[Mapping[str, Any]],
    validation_results: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    invariants = _invariance_checks(reference_roots, treatment_roots, episodes_by_seed)
    communication_changes = _communication_difference_checks(treatment_roots, episodes_by_seed)
    overhead_audit = _measurement_overhead_audit(
        overhead_rows,
        window_rows,
        enabled=config.measurement_overhead_enabled,
    )
    by_contrast = {(str(row["treatment"]), str(row["metric"])): row for row in contrasts}
    finite = all(
        _finite(row.get(key))
        for row in seed_rows
        for key in ("position_rmse_m", "j_loc", "j_com", "mean_reward", "harmful_fraction")
    ) and all(
        _finite(row.get(key))
        for row in window_rows
        for key in ("expected_information_gain_raw", "expected_information_gain_norm", "proxy_risk")
    )
    budget_cap_ok = all(int(row["max_budget_excess"]) == 0 for row in budget_usage)
    validation_ok = all(bool(result.get("ok")) for result in validation_results.values())
    leakage_free = all(
        int(row.get("information_gain_selection_leakage_free", 0)) == 1
        for row in window_rows
    )
    pairing_ok = all(float(value) <= NUMERIC_TOLERANCE for value in invariants.values())
    seeds = sorted({int(row["base_seed"]) for row in seed_rows})
    expected_seeds_ok = tuple(seeds) == tuple(sorted(reference_roots))
    unconstrained_rmse = by_contrast[("unconstrained_information_gain", "position_rmse_m")]
    unconstrained_jcom = by_contrast[("unconstrained_information_gain", "j_com")]
    budgeted_rmse = by_contrast[("budgeted_information_gain", "position_rmse_m")]
    budgeted_jloc = by_contrast[("budgeted_information_gain", "j_loc")]
    budgeted_jcom = by_contrast[("budgeted_information_gain", "j_com")]
    tradeoff_exposed = (
        float(unconstrained_rmse["mean_paired_difference"]) < 0.0
        and float(unconstrained_rmse["bootstrap_95_ci_high"]) < 0.0
        and float(unconstrained_jcom["mean_paired_difference"]) > 0.0
        and float(unconstrained_jcom["bootstrap_95_ci_low"]) > 0.0
    )
    budgeted_effective = (
        (
            float(budgeted_rmse["mean_paired_difference"]) < 0.0
            and float(budgeted_rmse["bootstrap_95_ci_high"]) < 0.0
        )
        or (
            float(budgeted_jloc["mean_paired_difference"]) < 0.0
            and float(budgeted_jloc["bootstrap_95_ci_high"]) < 0.0
        )
    ) and float(budgeted_jcom["bootstrap_95_ci_high"]) <= 0.0
    if tradeoff_exposed:
        decision = "localization_gain_communication_cost_tradeoff_exposed"
    elif budgeted_effective:
        decision = "budgeted_information_gain_preliminarily_effective"
    elif float(unconstrained_rmse["mean_paired_difference"]) < 0.0 or float(budgeted_rmse["mean_paired_difference"]) < 0.0:
        decision = "localization_direction_promising_but_statistically_inconclusive"
    else:
        decision = "information_gain_proxy_not_yet_actionable"
    return {
        "experiment": "budgeted_information_gain_paired_intervention",
        "evidence_status": "validation_only_mechanism_diagnostic",
        "reference_branch": config.reference_branch,
        "independent_unit": "base_seed",
        "paired_episode_unit": "episode_within_base_seed",
        "base_seed_count": len(seeds),
        "base_seeds": seeds,
        "time_windows_are_independent_samples": False,
        "ekf_events_are_independent_samples": False,
        "test_data_used": False,
        "truth_used_for_selection": False,
        "los_nlos_truth_used_for_selection": False,
        "future_information_used_for_selection": False,
        "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
        "communication_kpis_allowed_to_change": bool(config.measurement_overhead_enabled),
        "communication_changes": communication_changes,
        "overhead_audit": overhead_audit,
        "changed_action_field_only": "measurement_uav_id",
        "checks": {
            **invariants,
            "budget_cap_ok": budget_cap_ok,
            "validation_episode_schema_ok": validation_ok,
            "proxy_and_metrics_finite": finite,
            "selection_leakage_free": leakage_free,
            "deterministic_seed_set_ok": expected_seeds_ok,
            "overhead_values_finite": bool(overhead_audit["overhead_values_finite"]),
            "overhead_row_semantics_ok": bool(overhead_audit["overhead_row_semantics_ok"]),
            "overhead_correspondence_ok": bool(overhead_audit["overhead_correspondence_ok"]),
            "tradeoff_exposed": tradeoff_exposed,
            "budgeted_effective": budgeted_effective,
            "unconstrained_rmse_mean_difference": float(unconstrained_rmse["mean_paired_difference"]),
            "unconstrained_rmse_bootstrap_95_ci": [float(unconstrained_rmse["bootstrap_95_ci_low"]), float(unconstrained_rmse["bootstrap_95_ci_high"])],
            "unconstrained_j_com_mean_difference": float(unconstrained_jcom["mean_paired_difference"]),
            "unconstrained_j_com_bootstrap_95_ci": [float(unconstrained_jcom["bootstrap_95_ci_low"]), float(unconstrained_jcom["bootstrap_95_ci_high"])],
            "budgeted_rmse_mean_difference": float(budgeted_rmse["mean_paired_difference"]),
            "budgeted_j_loc_mean_difference": float(budgeted_jloc["mean_paired_difference"]),
            "budgeted_j_com_mean_difference": float(budgeted_jcom["mean_paired_difference"]),
        },
        "decision_gate": decision,
        "overall_ok": bool(
            pairing_ok
            and budget_cap_ok
            and validation_ok
            and finite
            and leakage_free
            and expected_seeds_ok
            and overhead_audit["overhead_correspondence_ok"]
        ),
        "decision_supported": bool(tradeoff_exposed or budgeted_effective),
        "interpretation_boundary": "This is a paired validation mechanism experiment. It does not retrain MAPPO, alter EKF/reward/channel models, or establish final policy superiority.",
    }


def run_information_gain_budget_intervention(config: InformationGainBudgetInterventionConfig) -> dict[str, Any]:
    """Run the five-seed validation-only paired intervention."""

    reference_root = Path(config.reference_root)
    base_seeds, reference_roots = _resolve_base_seeds(
        reference_root,
        config.reference_branch,
        config.seeds,
    )
    episodes_by_seed = {seed: _split_episodes(scale_root, config.split) for seed, scale_root in reference_roots.items()}
    if any(len(episodes) != 20 for episodes in episodes_by_seed.values()):
        counts = {seed: len(episodes) for seed, episodes in episodes_by_seed.items()}
        raise ValueError(f"expected 20 validation episodes per seed, found {counts}")

    logs_dir, plots_dir = _prepare_outputs(config)
    all_window_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []
    all_overhead_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    budgets_by_seed: dict[int, dict[str, int]] = {}
    budget_rows: list[dict[str, Any]] = []
    proxy_parameters: dict[str, Any] = {
        "source_branch": config.reference_branch,
        "train_only": True,
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "age_variance_coefficient": config.age_variance_coefficient,
        "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
        "seed_set": list(base_seeds),
        "seeds": {},
    }
    treatment_roots: dict[tuple[int, str], Path] = {}
    validation_results: dict[tuple[int, str], Mapping[str, Any]] = {}

    for base_seed in base_seeds:
        reference_scale = reference_roots[base_seed]
        reference_manifest = _load_manifest(reference_scale)
        budgets, budget_rows_seed = _train_budget(reference_scale, base_seed)
        budgets_by_seed[base_seed] = budgets
        budget_rows.extend(budget_rows_seed)
        scales, train_proxy_rows = _fit_seed_proxy(reference_scale, base_seed, config.age_variance_coefficient)
        proxy_parameters["seeds"][str(base_seed)] = {
            "scales": scales.as_dict(),
            "training_candidate_row_count": len(train_proxy_rows),
            "measurement_std_m": float(scales.measurement_std_median),
            "measurement_budgets": budgets,
        }
        context = {
            "proxy_scales": scales,
            "budgets": budgets,
            "measurement_std_m": float(scales.measurement_std_median),
            "age_variance_coefficient": config.age_variance_coefficient,
        }
        for treatment in TREATMENTS:
            treatment_root = Path(config.output_dir) / f"seed_{base_seed}" / treatment
            scale_root = treatment_root / "scale_3"
            scale_root.mkdir(parents=True, exist_ok=True)
            treatment_roots[(base_seed, treatment)] = scale_root
            treatment_config = _reference_formal_config(
                reference_manifest,
                treatment_root,
                force=False,
                progress=config.progress,
            )
            _copy_calibration_files(scale_root, treatment_config)
            normalization = reference_scale / "normalization_stats.json"
            if normalization.exists():
                shutil.copy2(normalization, scale_root / "normalization_stats.json")
            replay_config = QualityAwareMeasurementInterventionConfig(
                reference_scale=reference_scale,
                output_dir=treatment_root,
                split=config.split,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
                force=False,
                progress=config.progress,
            )
            episode_rows_seed: list[dict[str, Any]] = []
            summaries: list[dict[str, Any]] = []
            episodes = episodes_by_seed[base_seed]
            for index, source_episode in enumerate(episodes, start=1):
                overhead_rows_episode: list[dict[str, Any]] = []
                summary, metrics, windows, events = _run_replay_episode(
                    config=replay_config,
                    formal_config=treatment_config,
                    thresholds=context,
                    treatment=treatment,
                    scale_root=scale_root,
                    source_episode=source_episode,
                    episode_index=index,
                    episode_total=len(episodes),
                    selector=intervene_information_gain_selection,
                    policy_prefix=POLICY_PREFIX,
                    measurement_overhead_enabled=config.measurement_overhead_enabled,
                    overhead_rows_sink=overhead_rows_episode,
                )
                enriched_windows = _enrich_window_rows(windows, base_seed)
                enriched_events = [dict(event, base_seed=base_seed, episode_seed=int(_number(event.get("seed"), -1))) for event in events]
                enriched_overhead = [dict(row, base_seed=base_seed) for row in overhead_rows_episode]
                enriched_metrics = _episode_metrics_with_intervention(metrics, enriched_windows, enriched_events, base_seed, treatment, budgets)
                episode_rows_seed.append(enriched_metrics)
                summaries.append(summary)
                all_window_rows.extend(enriched_windows)
                all_event_rows.extend(enriched_events)
                all_overhead_rows.extend(enriched_overhead)
            all_episode_rows.extend(episode_rows_seed)
            treatment_manifest = {
                "dataset_id": f"python_information_gain_budget_{base_seed}_{treatment}",
                "schema_version": EPISODE_SCHEMA_VERSION,
                "engine": "python_sim",
                "result_status": "validation_only_mechanism_diagnostic",
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
                    "information_gain_uses_execution_time_fields_only": True,
                    "training_fitted_proxy_scales": True,
                    "measurement_budget_source": "baseline_mappo_train_mean_rounded",
                    "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
                    "measurement_overhead_payload_size_bytes": 512,
                    "uses_future_information": False,
                    "uses_test_data": False,
                    "retrained_mappo": False,
                },
                "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
                "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
                "episodes": summaries,
            }
            _write_json(scale_root / "manifest.json", treatment_manifest)
            validation_results[(base_seed, treatment)] = validate_episode_benchmark(scale_root)
            if not validation_results[(base_seed, treatment)].get("ok", False):
                raise RuntimeError(f"{base_seed}/{treatment} episode validation failed: {validation_results[(base_seed, treatment)]}")
        _progress(config, f"seed {base_seed} complete ({len(TREATMENTS)} treatments, 20 validation episodes)")

    seed_rows = _seed_metrics(all_episode_rows)
    contrasts = _paired_contrasts(seed_rows, config)
    usage_rows = _budget_usage(all_window_rows, budgets_by_seed)
    audit = _audit_report(
        config,
        reference_roots,
        treatment_roots,
        episodes_by_seed,
        seed_rows,
        all_window_rows,
        contrasts,
        usage_rows,
        all_overhead_rows,
        validation_results,
    )
    _write_json(logs_dir / "proxy_parameters.json", proxy_parameters)
    _write_table(logs_dir / "measurement_budgets.csv", budget_rows)
    _write_table(logs_dir / "window_metrics.csv", all_window_rows)
    _write_table(logs_dir / "update_events.csv", all_event_rows)
    _write_table(logs_dir / "episode_metrics.csv", all_episode_rows)
    _write_table(logs_dir / "seed_metrics.csv", seed_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "budget_usage.csv", usage_rows)
    _write_table(
        logs_dir / "measurement_overhead.csv",
        all_overhead_rows or [{column: "" for column in OVERHEAD_COLUMNS}],
    )
    _write_json(logs_dir / "audit_report.json", audit)
    plot_path = _plot(seed_rows, contrasts, budget_rows, plots_dir)

    root_manifest = {
        "dataset_id": f"python_information_gain_budget_intervention_seedset_{_seedset_label(base_seeds)}",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "result_status": "validation_only_mechanism_diagnostic",
        "reference": {
            "root": str(reference_root),
            "branch": config.reference_branch,
            "split": config.split,
            "test_data_used": False,
            "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
            "communication_kpis_allowed_to_change": bool(config.measurement_overhead_enabled),
        },
        "paired_design": {
            "independent_unit": "base_seed",
            "paired_episode_unit": "episode_within_base_seed",
            "base_seeds": list(base_seeds),
            "common_random_numbers": True,
            "changed_field_only": "measurement_uav_id",
            "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "treatments": {
            treatment: {
                "path": f"seed_<base_seed>/{treatment}/scale_3",
                "validation_episode_count_per_seed": 20,
            }
            for treatment in TREATMENTS
        },
        "logs": {name: str(logs_dir / name) for name in (
            "proxy_parameters.json", "measurement_budgets.csv", "window_metrics.csv", "update_events.csv",
            "episode_metrics.csv", "seed_metrics.csv", "paired_contrasts.csv", "budget_usage.csv", "measurement_overhead.csv", "audit_report.json",
        )},
        "plots": {"information_gain_budget_intervention": str(plot_path)},
        "audit_ok": bool(audit["overall_ok"]),
        "decision_supported": bool(audit["decision_supported"]),
        "decision_gate": audit["decision_gate"],
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    return {
        "output_dir": str(config.output_dir),
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "audit_report": str(logs_dir / "audit_report.json"),
        "plot": str(plot_path),
        "audit_ok": bool(audit["overall_ok"]),
        "decision_supported": bool(audit["decision_supported"]),
        "decision_gate": audit["decision_gate"],
    }


__all__ = [
    "InformationGainBudgetInterventionConfig",
    "intervene_information_gain_selection",
    "run_information_gain_budget_intervention",
]
