"""Paired bandwidth-action intervention for the frozen moderate validation set.

The experiment replays every reference episode under common random numbers and
changes only the per-edge bandwidth allocation.  Episodes, rather than decision
windows, are the independent statistical units.  Window rows are retained only
for mechanism diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_FILE_SPECS, EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_csv, _write_json
from .baseline import _read_csv, _read_json, _reference_formal_config, _sha256
from .formal import _copy_calibration_files, _episode_metrics, _make_environment
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _seed_everything,
)


TREATMENTS = ("recorded_mappo", "equal_share", "permuted_share")
POLICY_PREFIX = "bandwidth_intervention"
_ACTION_TOLERANCE = 1e-6
_NUMERIC_TOLERANCE = 2e-6


@dataclass(frozen=True)
class BandwidthInterventionConfig:
    """Configuration for the validation-only controlled intervention."""

    reference_scale: Path
    output_dir: Path
    split: str = "validation"
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    minimum_informative_fraction: float = 0.20
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("bandwidth intervention is validation-only; test data are prohibited")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.0 <= self.minimum_informative_fraction <= 1.0:
            raise ValueError("minimum_informative_fraction must lie in [0, 1]")


def _progress(config: BandwidthInterventionConfig, message: str) -> None:
    if config.progress:
        print(f"[bandwidth-intervention] {message}", flush=True)


def _json_action(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        action = json.loads(str(row["action_raw"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("actions.csv contains an invalid action_raw payload") from exc
    if not isinstance(action, dict):
        raise ValueError("action_raw must decode to an object")
    delta = np.asarray(action.get("delta_v", ()), dtype=float).reshape(-1)
    if delta.size != 3 or not np.isfinite(delta).all():
        raise ValueError("recorded delta_v must contain three finite values")
    action["delta_v"] = delta.tolist()
    action["measurement_uav_id"] = str(action.get("measurement_uav_id", "none"))
    action["bandwidth"] = {
        str(target): float(value)
        for target, value in dict(action.get("bandwidth", {})).items()
    }
    return action


def load_recorded_actions(episode_root: Path) -> list[dict[str, dict[str, Any]]]:
    """Load one joint action per applied decision window."""

    rows = _read_csv(episode_root / "actions.csv")
    grouped: dict[float, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = round(float(row["applied_window_start"]), 9)
        uav_id = str(row["uav_id"])
        if uav_id in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate recorded action for {uav_id} at {key}")
        grouped[key][uav_id] = _json_action(row)
    if not grouped:
        raise ValueError(f"no recorded actions found in {episode_root}")
    return [grouped[key] for key in sorted(grouped)]


def intervene_bandwidth(
    recorded_action: Mapping[str, Any],
    potential_targets: Sequence[str],
    treatment: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an action with only the bandwidth field modified."""

    if treatment not in TREATMENTS:
        raise ValueError(f"unknown bandwidth treatment: {treatment}")
    targets = sorted({str(target) for target in potential_targets})
    factual = {
        target: max(0.0, float(dict(recorded_action.get("bandwidth", {})).get(target, 0.0)))
        for target in targets
    }
    factual_total = min(1.0, sum(factual.values()))
    if treatment == "recorded_mappo":
        bandwidth = {
            str(target): float(value)
            for target, value in dict(recorded_action.get("bandwidth", {})).items()
        }
    elif treatment == "equal_share":
        bandwidth = {target: 1.0 / len(targets) for target in targets} if targets else {}
    else:
        values = [factual[target] for target in targets]
        bandwidth = {target: value for target, value in zip(targets, reversed(values))}

    effective = {target: max(0.0, float(bandwidth.get(target, 0.0))) for target in targets}
    l1_delta = float(sum(abs(effective[target] - factual[target]) for target in targets))
    action = {
        "measurement_uav_id": str(recorded_action.get("measurement_uav_id", "none")),
        "bandwidth": bandwidth,
        "delta_v": np.asarray(recorded_action["delta_v"], dtype=float).reshape(3),
    }
    metadata = {
        "potential_target_count": len(targets),
        "potential_targets": targets,
        "recorded_bandwidth": factual,
        "intervened_bandwidth": effective,
        "recorded_total": factual_total,
        "intervened_total": float(sum(effective.values())),
        "bandwidth_l1_delta": l1_delta,
        "eligible_for_intervention": len(targets) >= 2,
        "effective_intervention": len(targets) >= 2 and l1_delta > _ACTION_TOLERANCE,
    }
    return action, metadata


def _default_logs_dir(reference_scale: Path) -> Path:
    run_name = reference_scale.parent.name
    return Path("logs/preliminary") / run_name / "bandwidth_intervention"


def _default_plots_dir(reference_scale: Path) -> Path:
    run_name = reference_scale.parent.name
    return Path("plots/preliminary") / run_name / "bandwidth_intervention"


def _prepare_output(config: BandwidthInterventionConfig) -> None:
    root = Path(config.output_dir)
    if root.exists() and any(root.iterdir()) and not config.force:
        raise FileExistsError(f"intervention output is not empty: {root}; use --force to replace it")
    if config.force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for directory in (
        Path(config.logs_dir) if config.logs_dir else _default_logs_dir(Path(config.reference_scale)),
        Path(config.plots_dir) if config.plots_dir else _default_plots_dir(Path(config.reference_scale)),
    ):
        if config.force and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float(default)


def _quantile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else float(default)


def _connectivity_from_snapshot(snapshot: Any, uav_ids: Sequence[str]) -> float:
    adjacency = {uav_id: set() for uav_id in uav_ids}
    for source, target in snapshot.effective_edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(uav_ids)
    largest = 0
    while unseen:
        root = unseen.pop()
        component = {root}
        frontier = [root]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        largest = max(largest, len(component))
    return largest / max(1, len(uav_ids))


def _window_link_metrics(snapshot: Any, uav_ids: Sequence[str]) -> dict[str, float]:
    all_links = list(snapshot.metrics.values())
    candidates = [item for item in all_links if item.candidate_edge]
    selected = candidates
    return {
        "potential_edge_count": float(len(snapshot.potential_edges)),
        "candidate_edge_count": float(len(snapshot.candidate_edges)),
        "effective_edge_count": float(len(snapshot.effective_edges)),
        # Physical action-to-rate diagnostics use every potential edge.  The
        # communication objective below still follows the canonical
        # candidate-edge aggregation rule used by the environment.
        "mean_channel_bandwidth_hz": _mean((item.channel_bandwidth_hz for item in all_links)),
        "mean_rate_bps": _mean((item.rate_bps for item in all_links)),
        "packet_loss": _mean((item.packet_loss for item in selected), 1.0),
        "delay_s": _mean((item.delay_s for item in selected), 1.0),
        "aoi_s": _mean((item.aoi_s for item in selected), 60.0),
        "mean_queue_length": _mean((item.queue_length_packets for item in all_links)),
        "connectivity": _connectivity_from_snapshot(snapshot, uav_ids),
    }


def _window_reward_metrics(info: Mapping[str, Any]) -> dict[str, float]:
    global_metrics = info["global_metrics"]
    components = info["objective_components"]
    return {
        "uncertainty": float(global_metrics["uncertainty"]),
        "packet_loss_normalized": float(global_metrics["packet_loss_normalized"]),
        "delay_normalized": float(global_metrics["delay_normalized"]),
        "aoi_normalized": float(global_metrics["aoi_normalized"]),
        "connectivity_normalized": float(global_metrics["connectivity_normalized"]),
        "j_loc": float(components["j_loc"]),
        "j_com": float(components["j_com"]),
        "joint_objective": float(components["joint_objective"]),
    }


def _episode_summary_from_windows(
    base: Mapping[str, Any], window_rows: Sequence[Mapping[str, Any]], treatment: str
) -> dict[str, Any]:
    packet_loss = [float(row["packet_loss"]) for row in window_rows]
    delay = [float(row["delay_s"]) for row in window_rows]
    aoi = [float(row["aoi_s"]) for row in window_rows]
    rate = [float(row["mean_rate_bps"]) for row in window_rows]
    return {
        **dict(base),
        "treatment": treatment,
        "window_count": len(window_rows),
        "informative_window_count": sum(int(row["informative_window"]) for row in window_rows),
        "packet_loss_p95": _quantile(packet_loss, 0.95, 1.0),
        "delay_s_p95": _quantile(delay, 0.95, 1.0),
        "aoi_s_p95": _quantile(aoi, 0.95, 60.0),
        "mean_rate_bps": _mean(rate),
        "mean_queue_length": _mean(float(row["mean_queue_length"]) for row in window_rows),
        "j_com_loss": _mean(float(row["j_com_loss"]) for row in window_rows),
        "j_com_delay": _mean(float(row["j_com_delay"]) for row in window_rows),
        "j_com_aoi": _mean(float(row["j_com_aoi"]) for row in window_rows),
        "j_com_connectivity": _mean(float(row["j_com_connectivity"]) for row in window_rows),
        "loss_clip_fraction": _mean(float(row["loss_clipped"]) for row in window_rows),
        "delay_clip_fraction": _mean(float(row["delay_clipped"]) for row in window_rows),
        "aoi_clip_fraction": _mean(float(row["aoi_clipped"]) for row in window_rows),
    }


def _run_episode(
    *,
    config: BandwidthInterventionConfig,
    formal_config: Any,
    treatment: str,
    scale_root: Path,
    source_episode: Mapping[str, Any],
    episode_index: int,
    episode_total: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_id = str(source_episode["episode_id"])
    scenario_id = str(source_episode["scenario_id"])
    seed = int(source_episode["seed"])
    source_root = Path(config.reference_scale) / config.split / episode_id
    scenario_payload = _read_json(source_root / "scenario.json")
    scenario = scenario_payload.get("scenario_generation")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"missing scenario_generation in {source_root / 'scenario.json'}")
    recorded_windows = load_recorded_actions(source_root)

    _seed_everything(seed)
    environment = _make_environment(formal_config, seed, metric_scales=formal_config.metric_scales)
    observations, info = environment.reset(seed=seed, scenario_config=dict(scenario))
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / config.split / episode_id,
        episode_id=episode_id,
        scenario_id=scenario_id,
        split=config.split,
        seed=seed,
        phase=config.split,
        environment=environment,
        normalization_fitted=True,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=f"{POLICY_PREFIX}_{treatment}",
    )
    _progress(config, f"{treatment} episode {episode_index}/{episode_total} start seed={seed}")
    started = time.perf_counter()
    window_rows: list[dict[str, Any]] = []
    source_window_rows: list[dict[str, Any]] = []
    rewards_history: list[float] = []
    effective_edges_seen = 0
    max_neighbor_updates = 0
    for step, recorded_joint in enumerate(recorded_windows[: formal_config.max_steps]):
        _assert_observation_finite(observations)
        if set(recorded_joint) != set(environment.uav_ids):
            raise ValueError(f"recorded UAV set mismatch in {episode_id} step {step}")
        actions: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for uav_id in environment.uav_ids:
            action, action_meta = intervene_bandwidth(
                recorded_joint[uav_id], observations[uav_id].get("potential_targets", ()), treatment
            )
            actions[uav_id] = action
            metadata[uav_id] = action_meta
        _assert_action_feasible(environment, actions)
        next_observations, rewards, terminated, truncated, next_info = environment.step(actions)
        rewards_history.extend(float(value) for value in rewards.values())
        effective_edges_seen += len(next_info["central_state"]["effective_edges"])
        max_neighbor_updates = max(
            max_neighbor_updates,
            max(ekf.valid_neighbor_count for ekf in environment.ekfs.values()),
        )
        timestamp = (step + 1) * environment.config.motion.dt_s
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=actions,
            rewards=rewards,
            info=next_info,
            graph_traces=None,
            policy_source=f"{POLICY_PREFIX}_{treatment}",
            action_start=step * environment.config.motion.dt_s,
            timestamp=timestamp,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        link = _window_link_metrics(next_info["topology"], environment.uav_ids)
        reward = _window_reward_metrics(next_info)
        weights = environment.config.reward.weights
        global_informative = any(item["effective_intervention"] for item in metadata.values())
        row = {
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "split": config.split,
            "treatment": treatment,
            "step": step,
            "timestamp": timestamp,
            "informative_window": int(global_informative),
            "eligible_source_count": sum(int(item["eligible_for_intervention"]) for item in metadata.values()),
            "effective_source_count": sum(int(item["effective_intervention"]) for item in metadata.values()),
            **link,
            **reward,
        }
        row.update(
            j_com_loss=float(weights["loss"]) * row["packet_loss_normalized"],
            j_com_delay=float(weights["delay"]) * row["delay_normalized"],
            j_com_aoi=float(weights["aoi"]) * row["aoi_normalized"],
            j_com_connectivity=-float(weights["connectivity"]) * row["connectivity_normalized"],
            loss_clipped=int(row["packet_loss_normalized"] >= 1.0 - 1e-12),
            delay_clipped=int(row["delay_normalized"] >= 1.0 - 1e-12),
            aoi_clipped=int(row["aoi_normalized"] >= 1.0 - 1e-12),
        )
        window_rows.append(row)
        for uav_id, item in metadata.items():
            source_window_rows.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "treatment": treatment,
                    "step": step,
                    "timestamp": timestamp,
                    "uav_id": uav_id,
                    "potential_target_count": item["potential_target_count"],
                    "eligible_for_intervention": int(item["eligible_for_intervention"]),
                    "effective_intervention": int(item["effective_intervention"]),
                    "bandwidth_l1_delta": item["bandwidth_l1_delta"],
                    "recorded_total": item["recorded_total"],
                    "intervened_total": item["intervened_total"],
                    "recorded_bandwidth_json": json.dumps(item["recorded_bandwidth"], sort_keys=True),
                    "intervened_bandwidth_json": json.dumps(item["intervened_bandwidth"], sort_keys=True),
                }
            )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break

    summary = writer.finalize()
    summary.update(
        seed=seed,
        mean_reward=_mean(rewards_history),
        effective_edges_seen=effective_edges_seen,
        max_neighbor_updates=max_neighbor_updates,
        source_episode_id=episode_id,
        source_scenario_sha256=_sha256(source_root / "scenario.json"),
        treatment=treatment,
    )
    metrics = _episode_metrics(scale_root / config.split / episode_id, summary)
    metrics = _episode_summary_from_windows(metrics, window_rows, treatment)
    _progress(
        config,
        f"{treatment} episode {episode_index}/{episode_total} complete "
        f"j_com={metrics['j_com']:.6f} elapsed={time.perf_counter() - started:.1f}s",
    )
    return summary, metrics, window_rows, source_window_rows


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    _write_csv(path, tuple(rows[0].keys()), [dict(row) for row in rows])


def _bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_contrasts(
    rows: Sequence[Mapping[str, Any]], config: BandwidthInterventionConfig
) -> list[dict[str, Any]]:
    metrics = (
        "mean_rate_bps",
        "packet_loss",
        "packet_loss_p95",
        "delay_s",
        "delay_s_p95",
        "aoi_s",
        "aoi_s_p95",
        "connectivity",
        "mean_queue_length",
        "j_com_loss",
        "j_com_delay",
        "j_com_aoi",
        "j_com_connectivity",
        "j_com",
        "joint_objective",
    )
    by_key = {(int(row["seed"]), str(row["treatment"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(config.bootstrap_seed)
    output: list[dict[str, Any]] = []
    for comparator in ("equal_share", "permuted_share"):
        for metric in metrics:
            factual = np.asarray([float(by_key[(seed, "recorded_mappo")][metric]) for seed in seeds])
            alternative = np.asarray([float(by_key[(seed, comparator)][metric]) for seed in seeds])
            difference = factual - alternative
            low, high = _bootstrap_interval(difference, config.bootstrap_samples, rng)
            output.append(
                {
                    "contrast": f"recorded_mappo_minus_{comparator}",
                    "metric": metric,
                    "paired_episodes": len(seeds),
                    "recorded_mappo_mean": float(factual.mean()),
                    "comparator_mean": float(alternative.mean()),
                    "mean_paired_difference": float(difference.mean()),
                    "median_paired_difference": float(np.median(difference)),
                    "paired_difference_std": float(difference.std(ddof=1)) if difference.size > 1 else 0.0,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "ci_excludes_zero": int(low > 0.0 or high < 0.0),
                }
            )
    return output


def _max_numeric_difference(
    first: Path,
    second: Path,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
) -> float:
    left = _read_csv(first)
    right = _read_csv(second)
    left_map = {tuple(row[key] for key in key_columns): row for row in left}
    right_map = {tuple(row[key] for key in key_columns): row for row in right}
    if left_map.keys() != right_map.keys():
        return float("inf")
    maximum = 0.0
    for key in left_map:
        for column in value_columns:
            maximum = max(maximum, abs(float(left_map[key][column]) - float(right_map[key][column])))
    return maximum


def _action_invariance(first: Path, second: Path) -> tuple[float, bool]:
    left = _read_csv(first)
    right = _read_csv(second)
    keys = ("timestamp", "uav_id")
    left_map = {tuple(row[key] for key in keys): row for row in left}
    right_map = {tuple(row[key] for key in keys): row for row in right}
    if left_map.keys() != right_map.keys():
        return float("inf"), False
    maximum = 0.0
    measurement_equal = True
    for key in left_map:
        measurement_equal &= (
            str(left_map[key]["measurement_uav_id"])
            == str(right_map[key]["measurement_uav_id"])
        )
        for column in ("delta_vx", "delta_vy", "delta_vz"):
            maximum = max(
                maximum,
                abs(float(left_map[key][column]) - float(right_map[key][column])),
            )
    return maximum, measurement_equal


def _audit_replays(
    config: BandwidthInterventionConfig,
    treatment_roots: Mapping[str, Path],
    episodes: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    source_window_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trajectory_max = 0.0
    obstacle_max = 0.0
    radio_randomness_max = 0.0
    factual_trajectory_max = 0.0
    factual_link_kpi_max = 0.0
    factual_objective_max = 0.0
    non_bandwidth_action_max = 0.0
    measurement_actions_equal = True
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        factual = treatment_roots["recorded_mappo"] / config.split / episode_id
        reference = Path(config.reference_scale) / config.split / episode_id
        factual_trajectory_max = max(
            factual_trajectory_max,
            _max_numeric_difference(
                factual / "ground_truth.csv",
                reference / "ground_truth.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=("x", "y", "z", "vx", "vy", "vz"),
            ),
        )
        factual_link_kpi_max = max(
            factual_link_kpi_max,
            _max_numeric_difference(
                factual / "links.csv",
                reference / "links.csv",
                key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                value_columns=("rate_bps", "packet_loss", "delay", "aoi", "queue_length"),
            ),
        )
        factual_objective_max = max(
            factual_objective_max,
            _max_numeric_difference(
                factual / "rewards.csv",
                reference / "rewards.csv",
                key_columns=("timestamp", "uav_id"),
                value_columns=(
                    "joint_objective",
                    "normalized_packet_loss",
                    "normalized_delay",
                    "normalized_aoi",
                    "normalized_connectivity",
                ),
            ),
        )
        action_difference, action_equal = _action_invariance(
            factual / "actions.csv", reference / "actions.csv"
        )
        non_bandwidth_action_max = max(non_bandwidth_action_max, action_difference)
        measurement_actions_equal &= action_equal
        for treatment in ("equal_share", "permuted_share"):
            other = treatment_roots[treatment] / config.split / episode_id
            trajectory_max = max(
                trajectory_max,
                _max_numeric_difference(
                    factual / "ground_truth.csv",
                    other / "ground_truth.csv",
                    key_columns=("timestamp", "uav_id"),
                    value_columns=("x", "y", "z", "vx", "vy", "vz"),
                ),
            )
            obstacle_max = max(
                obstacle_max,
                _max_numeric_difference(
                    factual / "obstacles.csv",
                    other / "obstacles.csv",
                    key_columns=("timestamp", "obstacle_id"),
                    value_columns=("x", "y", "z", "vx", "vy", "vz", "radius"),
                ),
            )
            radio_randomness_max = max(
                radio_randomness_max,
                _max_numeric_difference(
                    factual / "links.csv",
                    other / "links.csv",
                    key_columns=("timestamp", "src_uav_id", "dst_uav_id"),
                    value_columns=("rx_power_dbm", "rssi"),
                ),
            )
            action_difference, action_equal = _action_invariance(
                factual / "actions.csv", other / "actions.csv"
            )
            non_bandwidth_action_max = max(non_bandwidth_action_max, action_difference)
            measurement_actions_equal &= action_equal

    all_sources = [row for row in source_window_rows if row["treatment"] != "recorded_mappo"]
    eligible_sources = [row for row in all_sources if int(row["eligible_for_intervention"])]
    effective_sources = [row for row in all_sources if int(row["effective_intervention"])]
    intervention_rows = [row for row in episode_rows if row["treatment"] != "recorded_mappo"]
    total_windows = sum(int(row["window_count"]) for row in intervention_rows)
    effective_windows = sum(int(row["informative_window_count"]) for row in intervention_rows)
    window_fraction = effective_windows / max(1, total_windows)
    source_fraction = len(effective_sources) / max(1, len(all_sources))

    contrast_map = {(row["contrast"], row["metric"]): row for row in contrasts}
    raw_metrics = ("mean_rate_bps", "packet_loss_p95", "delay_s_p95", "aoi_s_p95", "connectivity")
    raw_response = any(
        int(contrast_map[(contrast, metric)]["ci_excludes_zero"])
        for contrast in ("recorded_mappo_minus_equal_share", "recorded_mappo_minus_permuted_share")
        for metric in raw_metrics
    )
    j_response = any(
        int(contrast_map[(contrast, "j_com")]["ci_excludes_zero"])
        for contrast in ("recorded_mappo_minus_equal_share", "recorded_mappo_minus_permuted_share")
    )
    recorded_better = all(
        float(contrast_map[(contrast, "j_com")]["bootstrap_95_ci_high"]) < 0.0
        for contrast in ("recorded_mappo_minus_equal_share", "recorded_mappo_minus_permuted_share")
    )
    if not raw_response:
        decision = "inspect_bandwidth_channel_queue_coupling"
    elif not j_response:
        decision = "revise_normalization_and_tail_risk_expression"
    elif recorded_better:
        decision = "bandwidth_chain_normal_proceed_to_communication_localization_coupling"
    else:
        decision = "strategy_not_stably_better_than_simple_bandwidth_allocations"

    checks = {
        "reference_factual_trajectory_max_abs_difference": factual_trajectory_max,
        "reference_factual_link_kpi_max_abs_difference": factual_link_kpi_max,
        "reference_factual_objective_max_abs_difference": factual_objective_max,
        "cross_treatment_trajectory_max_abs_difference": trajectory_max,
        "cross_treatment_obstacle_max_abs_difference": obstacle_max,
        "cross_treatment_rx_rssi_max_abs_difference": radio_randomness_max,
        "non_bandwidth_delta_v_max_abs_difference": non_bandwidth_action_max,
        "measurement_actions_identical": measurement_actions_equal,
        "bandwidth_nonnegative": all(float(row["intervened_total"]) >= -_ACTION_TOLERANCE for row in all_sources),
        "bandwidth_sum_at_most_one": all(float(row["intervened_total"]) <= 1.0 + _ACTION_TOLERANCE for row in all_sources),
        "effective_intervention_window_fraction": window_fraction,
        "effective_intervention_source_window_fraction": source_fraction,
        "eligible_source_windows": len(eligible_sources),
        "effective_source_windows": len(effective_sources),
        "coverage_gate": window_fraction >= config.minimum_informative_fraction,
        "raw_kpi_response_detected": raw_response,
        "j_com_response_detected": j_response,
        "recorded_mappo_better_than_both_simple_allocations_on_j_com": recorded_better,
    }
    deterministic_ok = all(
        value <= _NUMERIC_TOLERANCE
        for value in (
            factual_trajectory_max,
            factual_link_kpi_max,
            factual_objective_max,
            trajectory_max,
            obstacle_max,
            radio_randomness_max,
            non_bandwidth_action_max,
        )
    ) and measurement_actions_equal
    feasibility_ok = bool(checks["bandwidth_nonnegative"] and checks["bandwidth_sum_at_most_one"])
    return {
        "experiment": "paired_bandwidth_action_intervention",
        "evidence_status": "engineering_mechanism_diagnostic_only",
        "independent_unit": "episode",
        "time_steps_used_as_independent_samples": False,
        "test_data_used": False,
        "checks": checks,
        "deterministic_pairing_ok": deterministic_ok,
        "bandwidth_feasibility_ok": feasibility_ok,
        "decision_gate": decision,
        "overall_ok": deterministic_ok and feasibility_ok and bool(checks["coverage_gate"]),
        "interpretation_boundary": (
            "This paired validation intervention identifies the bandwidth mechanism only; "
            "it is not a final MAPPO performance claim and does not test localization coupling."
        ),
    }


def _plot_diagnostics(
    window_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"recorded_mappo": "#2563eb", "equal_share": "#16a34a", "permuted_share": "#dc2626"}
    labels = {"recorded_mappo": "Recorded MAPPO", "equal_share": "Equal share", "permuted_share": "Permuted share"}
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for treatment in TREATMENTS:
        selected = [row for row in window_rows if row["treatment"] == treatment and int(row["informative_window"])]
        axes[0].scatter(
            [float(row["mean_channel_bandwidth_hz"]) for row in selected],
            [float(row["mean_rate_bps"]) for row in selected],
            s=9,
            alpha=0.35,
            color=colors[treatment],
            label=labels[treatment],
        )
        axes[1].scatter(
            [float(row["mean_rate_bps"]) for row in selected],
            [float(row["mean_queue_length"]) for row in selected],
            s=9,
            alpha=0.35,
            color=colors[treatment],
        )
    axes[0].set(xlabel="Mean candidate-edge bandwidth (Hz)", ylabel="Mean candidate-edge rate (bit/s)")
    axes[1].set(xlabel="Mean candidate-edge rate (bit/s)", ylabel="Mean queue length (packets)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    path = plots_dir / "bandwidth_rate_response.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["bandwidth_rate_response"] = str(path)

    def paired_panel(filename: str, metrics: Sequence[tuple[str, str]]) -> None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), constrained_layout=True)
        axes_array = np.atleast_1d(axes)
        seeds = sorted({int(row["seed"]) for row in episode_rows})
        by_key = {(int(row["seed"]), str(row["treatment"])): row for row in episode_rows}
        for axis, (metric, title) in zip(axes_array, metrics):
            for seed in seeds:
                values = [float(by_key[(seed, treatment)][metric]) for treatment in TREATMENTS]
                axis.plot(range(3), values, color="#94a3b8", alpha=0.35, linewidth=0.7)
            means = [np.mean([float(by_key[(seed, treatment)][metric]) for seed in seeds]) for treatment in TREATMENTS]
            axis.scatter(range(3), means, color=[colors[item] for item in TREATMENTS], s=42, zorder=3)
            axis.set_xticks(range(3), ("MAPPO", "Equal", "Permuted"), rotation=15)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle("Paired validation episodes; lines connect identical seeds")
        target = plots_dir / filename
        fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[filename[:-4]] = str(target)

    paired_panel(
        "communication_kpi_comparison.png",
        (("packet_loss_p95", "Packet loss p95"), ("delay_s_p95", "Delay p95 (s)"), ("aoi_s_p95", "AoI p95 (s)"), ("connectivity", "Connectivity")),
    )
    paired_panel(
        "jcom_components.png",
        (("j_com_loss", "Loss term"), ("j_com_delay", "Delay term"), ("j_com_aoi", "AoI term"), ("j_com_connectivity", "Connectivity term"), ("j_com", "J_com")),
    )
    paired_panel(
        "normalization_saturation.png",
        (("loss_clip_fraction", "Loss clipped fraction"), ("delay_clip_fraction", "Delay clipped fraction"), ("aoi_clip_fraction", "AoI clipped fraction")),
    )
    return paths


def run_bandwidth_intervention(config: BandwidthInterventionConfig) -> dict[str, Any]:
    """Run the validation-only paired bandwidth intervention."""

    reference_scale = Path(config.reference_scale)
    reference_manifest = _read_json(reference_scale / "manifest.json")
    if str(reference_manifest.get("schema_version")) != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"reference must use {EPISODE_SCHEMA_VERSION}")
    if str(reference_manifest.get("policy_source")) != "mappo":
        raise ValueError("reference manifest must be a MAPPO run")
    if int(reference_manifest.get("uav_count", 0)) != 3:
        raise ValueError("the first bandwidth intervention is restricted to three UAVs")
    episodes = [
        item
        for item in reference_manifest.get("episodes", [])
        if isinstance(item, Mapping) and item.get("split") == config.split
    ]
    if len(episodes) != 20:
        raise ValueError(f"the frozen intervention requires exactly 20 validation episodes, found {len(episodes)}")

    _prepare_output(config)
    logs_dir = Path(config.logs_dir) if config.logs_dir else _default_logs_dir(reference_scale)
    plots_dir = Path(config.plots_dir) if config.plots_dir else _default_plots_dir(reference_scale)
    formal_config = _reference_formal_config(
        reference_manifest,
        Path(config.output_dir),
        force=False,
        progress=config.progress,
    )
    treatment_roots: dict[str, Path] = {}
    all_window_rows: list[dict[str, Any]] = []
    all_source_window_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    treatment_manifests: dict[str, Any] = {}
    for treatment in TREATMENTS:
        treatment_root = Path(config.output_dir) / "treatments" / treatment
        scale_root = treatment_root / "scale_3"
        scale_root.mkdir(parents=True, exist_ok=True)
        treatment_roots[treatment] = scale_root
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
        summaries: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for index, source_episode in enumerate(episodes, start=1):
            summary, metric, windows, source_windows = _run_episode(
                config=config,
                formal_config=formal_config,
                treatment=treatment,
                scale_root=scale_root,
                source_episode=source_episode,
                episode_index=index,
                episode_total=len(episodes),
            )
            summaries.append(summary)
            metrics.append(metric)
            all_window_rows.extend(windows)
            all_source_window_rows.extend(source_windows)
        all_episode_rows.extend(metrics)
        manifest = {
            "dataset_id": f"python_bandwidth_intervention_{treatment}",
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
                "changed_field": "bandwidth",
                "fixed_fields": ["scenario", "seed", "delta_v", "measurement_uav_id"],
                "uses_test_data": False,
                "retrained_mappo": False,
            },
            "config": {**asdict(treatment_config), "output_dir": str(treatment_config.output_dir)},
            "files_per_episode": list(EPISODE_FILE_SPECS) + ["scenario.json"],
            "episodes": summaries,
        }
        _write_json(scale_root / "manifest.json", manifest)
        _write_json(
            treatment_root / "manifest.json",
            {
                "dataset_id": manifest["dataset_id"],
                "schema_version": EPISODE_SCHEMA_VERSION,
                "scales": {"3": {"uav_count": 3, "path": "scale_3", "manifest": "scale_3/manifest.json", "episode_count": len(summaries)}},
            },
        )
        validation = validate_episode_benchmark(scale_root)
        if not validation["ok"]:
            raise RuntimeError(f"{treatment} episode validation failed: {validation}")
        treatment_manifests[treatment] = {
            "scale_root": str(scale_root),
            "manifest": str(scale_root / "manifest.json"),
            "validation_ok": True,
        }

    contrasts = _paired_contrasts(all_episode_rows, config)
    audit = _audit_replays(
        config,
        treatment_roots,
        episodes,
        all_episode_rows,
        all_source_window_rows,
        contrasts,
    )
    coverage_rows = []
    for treatment in TREATMENTS:
        selected = [row for row in all_source_window_rows if row["treatment"] == treatment]
        eligible = [row for row in selected if int(row["eligible_for_intervention"])]
        effective = [row for row in selected if int(row["effective_intervention"])]
        episode_selected = [row for row in all_episode_rows if row["treatment"] == treatment]
        coverage_rows.append(
            {
                "treatment": treatment,
                "source_windows": len(selected),
                "eligible_source_windows": len(eligible),
                "effective_source_windows": len(effective),
                "effective_source_window_fraction": len(effective) / max(1, len(selected)),
                "global_windows": sum(int(row["window_count"]) for row in episode_selected),
                "effective_global_windows": sum(int(row["informative_window_count"]) for row in episode_selected),
                "effective_global_window_fraction": sum(int(row["informative_window_count"]) for row in episode_selected) / max(1, sum(int(row["window_count"]) for row in episode_selected)),
            }
        )
    _write_table(logs_dir / "window_metrics.csv", all_window_rows)
    _write_table(logs_dir / "source_window_actions.csv", all_source_window_rows)
    _write_table(logs_dir / "episode_metrics.csv", all_episode_rows)
    _write_table(logs_dir / "paired_contrasts.csv", contrasts)
    _write_table(logs_dir / "intervention_coverage.csv", coverage_rows)
    _write_json(logs_dir / "audit_report.json", audit)
    plots = _plot_diagnostics(all_window_rows, all_episode_rows, plots_dir)

    root_manifest = {
        "dataset_id": "python_bandwidth_action_intervention_moderate_validation_seed2026",
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
            "time_steps_are_independent_replicates": False,
            "common_random_numbers": True,
            "changed_field_only": "bandwidth",
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "test_data_used": False,
        },
        "treatments": treatment_manifests,
        "logs": {
            "window_metrics": str(logs_dir / "window_metrics.csv"),
            "source_window_actions": str(logs_dir / "source_window_actions.csv"),
            "episode_metrics": str(logs_dir / "episode_metrics.csv"),
            "paired_contrasts": str(logs_dir / "paired_contrasts.csv"),
            "intervention_coverage": str(logs_dir / "intervention_coverage.csv"),
            "audit_report": str(logs_dir / "audit_report.json"),
        },
        "plots": plots,
        "audit": audit,
    }
    _write_json(Path(config.output_dir) / "manifest.json", root_manifest)
    if not audit["overall_ok"]:
        raise RuntimeError(
            "bandwidth intervention audit did not pass; inspect audit_report.json: "
            f"{audit['checks']}"
        )
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
        "plots": plots,
    }


__all__ = [
    "BandwidthInterventionConfig",
    "TREATMENTS",
    "intervene_bandwidth",
    "load_recorded_actions",
    "run_bandwidth_intervention",
]
