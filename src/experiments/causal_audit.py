"""Window-aligned action-to-KPI audit for paired MAPPO and fixed baselines.

The audit is deliberately diagnostic.  It checks whether actions, physical
link metrics, objective components and rewards form a numerically closed
chain, but it does not interpret observational correlations as treatment
effects.  Episode-level summaries are the independent units used for
uncertainty intervals.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CausalAuditConfig:
    reference_scale: Path
    baseline_scale: Path
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    split: str = "test"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty audit table: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _timestamp(row: Mapping[str, Any]) -> float:
    return round(_number(row, "timestamp"), 9)


def _group_by_timestamp(rows: Iterable[Mapping[str, Any]]) -> dict[float, list[Mapping[str, Any]]]:
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_timestamp(row), []).append(row)
    return grouped


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float(default)


def _episode_directories(scale: Path, split: str) -> dict[int, Path]:
    split_root = scale / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing {split} split: {split_root}")
    episodes: dict[int, Path] = {}
    for episode in sorted(split_root.glob("episode_*")):
        scenario_path = episode / "scenario.json"
        if not scenario_path.exists():
            continue
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        seed = int(scenario["seed"])
        if seed in episodes:
            raise ValueError(f"duplicate seed {seed} under {split_root}")
        episodes[seed] = episode
    if not episodes:
        raise ValueError(f"no episodes found under {split_root}")
    return episodes


def _action_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    delta_norms: list[float] = []
    entropies: list[float] = []
    max_shares: list[float] = []
    measurements = 0
    for row in rows:
        delta_norms.append(
            float(
                np.linalg.norm(
                    [
                        _number(row, "delta_vx"),
                        _number(row, "delta_vy"),
                        _number(row, "delta_vz"),
                    ]
                )
            )
        )
        try:
            allocations = json.loads(str(row.get("bandwidth_allocations_json", "{}")))
        except json.JSONDecodeError:
            allocations = {}
        shares = np.asarray(
            [max(0.0, float(value)) for value in allocations.values()], dtype=float
        )
        if shares.size and shares.sum() > 0.0:
            shares = shares / shares.sum()
            entropy = float(-np.sum(shares * np.log(np.clip(shares, 1e-12, None))))
            if shares.size > 1:
                entropy /= math.log(shares.size)
            else:
                entropy = 0.0
            entropies.append(entropy)
            max_shares.append(float(shares.max()))
        else:
            entropies.append(0.0)
            max_shares.append(0.0)
        measurements += int(str(row.get("measurement_uav_id", "none")) != "none")
    count = max(1, len(rows))
    return {
        "mean_delta_v_norm": _mean(delta_norms),
        "max_delta_v_norm": max(delta_norms, default=0.0),
        "mean_bandwidth_entropy": _mean(entropies),
        "mean_max_bandwidth_share": _mean(max_shares),
        "measurement_selection_fraction": measurements / count,
        "action_clipped_fraction": _mean(_number(row, "action_clipped") for row in rows),
        "safety_projected_fraction": _mean(_number(row, "safety_projected") for row in rows),
    }


def _link_features(
    rows: Sequence[Mapping[str, Any]],
    uav_ids: Sequence[str],
    *,
    worst_case_delay_s: float,
    worst_case_aoi_s: float,
) -> dict[str, float]:
    candidate = [row for row in rows if int(_number(row, "candidate_edge")) == 1]
    potential = [row for row in rows if int(_number(row, "potential_edge")) == 1]
    effective = [row for row in rows if int(_number(row, "effective_edge")) == 1]
    selected = candidate
    adjacency = {uav_id: set() for uav_id in uav_ids}
    for row in effective:
        source = str(row["src_uav_id"])
        target = str(row["dst_uav_id"])
        if source in adjacency and target in adjacency:
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
    return {
        "potential_edge_count": float(len(potential)),
        "candidate_edge_count": float(len(candidate)),
        "effective_edge_count": float(len(effective)),
        "packet_loss": _mean((_number(row, "packet_loss") for row in selected), 1.0),
        "delay_s": _mean(
            (_number(row, "delay") for row in selected), worst_case_delay_s
        ),
        "aoi_s": _mean((_number(row, "aoi") for row in selected), worst_case_aoi_s),
        "connectivity": largest / max(1, len(uav_ids)),
        "nlos_fraction": _mean((1.0 - _number(row, "los") for row in selected), 0.0),
        "mean_queue_length": _mean((_number(row, "queue_length") for row in selected), 0.0),
        "mean_distance_m": _mean((_number(row, "distance") for row in selected), 0.0),
        "mean_rate_bps": _mean((_number(row, "rate_bps") for row in selected), 0.0),
    }


def _reward_features(row: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, float]:
    weights = dict(scenario.get("reward_config", {}).get("weights", {}))
    w_unc = float(weights.get("uncertainty", weights.get("unc", 0.0)))
    w_loss = float(weights.get("loss", 0.0))
    w_aoi = float(weights.get("aoi", 0.0))
    w_delay = float(weights.get("delay", 0.0))
    w_net = float(weights.get("network", weights.get("connectivity", weights.get("net", 0.0))))
    uncertainty = _number(row, "normalized_uncertainty")
    loss = _number(row, "normalized_packet_loss")
    aoi = _number(row, "normalized_aoi")
    delay = _number(row, "normalized_delay")
    connectivity = _number(row, "normalized_connectivity")
    j_loc = w_unc * uncertainty
    j_loss = w_loss * loss
    j_aoi = w_aoi * aoi
    j_delay = w_delay * delay
    j_connectivity = -w_net * connectivity
    j_com = j_loss + j_aoi + j_delay + j_connectivity
    calculated_j = j_loc + j_com
    logged_j = _number(row, "joint_objective")
    collision_penalty = max(0.0, -_number(row, "reward_collision"))
    energy_penalty = max(0.0, -_number(row, "reward_energy"))
    reward = _number(row, "reward_total")
    calculated_reward = -(calculated_j + collision_penalty + energy_penalty)
    return {
        "team_reward": reward,
        "j_loc": j_loc,
        "j_com_loss": j_loss,
        "j_com_aoi": j_aoi,
        "j_com_delay": j_delay,
        "j_com_connectivity": j_connectivity,
        "j_com": j_com,
        "joint_objective": logged_j,
        "collision_penalty": collision_penalty,
        "energy_penalty": energy_penalty,
        "normalized_uncertainty": uncertainty,
        "normalized_packet_loss": loss,
        "normalized_aoi": aoi,
        "normalized_delay": delay,
        "normalized_connectivity": connectivity,
        "objective_identity_abs_error": abs(logged_j - calculated_j),
        "reward_identity_abs_error": abs(reward - calculated_reward),
    }


def _audit_episode(
    controller: str,
    episode: Path,
    *,
    worst_case_delay_s: float,
    worst_case_aoi_s: float,
) -> list[dict[str, Any]]:
    scenario = json.loads((episode / "scenario.json").read_text(encoding="utf-8"))
    actions = _group_by_timestamp(_read_csv(episode / "actions.csv"))
    links = _group_by_timestamp(_read_csv(episode / "links.csv"))
    rewards = _group_by_timestamp(_read_csv(episode / "rewards.csv"))
    # A valid decision window can have no potential/candidate links, in which
    # case links.csv legitimately has no row.  Such windows are precisely the
    # difficult cases and must not disappear from the audit.
    timestamps = sorted(set(actions) & set(rewards))
    if not timestamps:
        raise ValueError(f"no aligned action/link/reward windows in {episode}")
    uav_ids = sorted({str(row["uav_id"]) for rows in actions.values() for row in rows})
    audited: list[dict[str, Any]] = []
    for timestamp in timestamps:
        action = _action_features(actions[timestamp])
        link = _link_features(
            links.get(timestamp, ()),
            uav_ids,
            worst_case_delay_s=worst_case_delay_s,
            worst_case_aoi_s=worst_case_aoi_s,
        )
        reward = _reward_features(rewards[timestamp][0], scenario)
        audited.append(
            {
                "controller": controller,
                "episode_id": scenario["episode_id"],
                "scenario_id": scenario["scenario_id"],
                "seed": int(scenario["seed"]),
                "split": scenario["split"],
                "window_end_s": timestamp,
                **action,
                **link,
                **reward,
            }
        )
    return audited


_EPISODE_MEAN_FIELDS = (
    "mean_delta_v_norm",
    "max_delta_v_norm",
    "mean_bandwidth_entropy",
    "mean_max_bandwidth_share",
    "measurement_selection_fraction",
    "action_clipped_fraction",
    "safety_projected_fraction",
    "potential_edge_count",
    "candidate_edge_count",
    "effective_edge_count",
    "packet_loss",
    "delay_s",
    "aoi_s",
    "connectivity",
    "nlos_fraction",
    "mean_queue_length",
    "mean_distance_m",
    "mean_rate_bps",
    "team_reward",
    "j_loc",
    "j_com_loss",
    "j_com_aoi",
    "j_com_delay",
    "j_com_connectivity",
    "j_com",
    "joint_objective",
    "collision_penalty",
    "energy_penalty",
)


def _episode_summaries(window_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in window_rows:
        grouped.setdefault((str(row["controller"]), int(row["seed"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (controller, seed), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "controller": controller,
            "seed": seed,
            "episode_id": rows[0]["episode_id"],
            "windows": len(rows),
        }
        for field in _EPISODE_MEAN_FIELDS:
            summary[field] = _mean(float(row[field]) for row in rows)
        summary["no_candidate_window_fraction"] = _mean(
            1.0 if float(row["candidate_edge_count"]) <= 0.0 else 0.0 for row in rows
        )
        for field in ("packet_loss", "delay_s", "aoi_s"):
            summary[f"{field}_p95"] = float(
                np.quantile([float(row[field]) for row in rows], 0.95)
            )
        for field in (
            "normalized_uncertainty",
            "normalized_packet_loss",
            "normalized_aoi",
            "normalized_delay",
            "normalized_connectivity",
        ):
            summary[f"{field}_clip_fraction"] = _mean(
                1.0 if float(row[field]) >= 1.0 - 1e-9 else 0.0 for row in rows
            )
        summary["objective_identity_max_abs_error"] = max(
            float(row["objective_identity_abs_error"]) for row in rows
        )
        summary["reward_identity_max_abs_error"] = max(
            float(row["reward_identity_abs_error"]) for row in rows
        )
        summaries.append(summary)
    return summaries


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.size < 3 or np.std(x_array) <= 1e-12 or np.std(y_array) <= 1e-12:
        return None
    value = float(np.corrcoef(x_array, y_array)[0, 1])
    return value if math.isfinite(value) else None


_RELATIONS = (
    ("mean_bandwidth_entropy", "packet_loss"),
    ("mean_bandwidth_entropy", "delay_s"),
    ("mean_bandwidth_entropy", "aoi_s"),
    ("mean_bandwidth_entropy", "connectivity"),
    ("mean_bandwidth_entropy", "j_com"),
    ("mean_max_bandwidth_share", "packet_loss"),
    ("mean_max_bandwidth_share", "j_com"),
    ("mean_delta_v_norm", "mean_distance_m"),
    ("mean_delta_v_norm", "nlos_fraction"),
    ("mean_delta_v_norm", "connectivity"),
    ("mean_delta_v_norm", "j_loc"),
    ("mean_delta_v_norm", "j_com"),
    ("measurement_selection_fraction", "j_loc"),
)


def _episode_relations(window_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in window_rows:
        grouped.setdefault((str(row["controller"]), int(row["seed"])), []).append(row)
    relations: list[dict[str, Any]] = []
    for (controller, seed), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["window_end_s"]))
        for action_field, outcome_field in _RELATIONS:
            correlation = _pearson(
                [float(row[action_field]) for row in ordered],
                [float(row[outcome_field]) for row in ordered],
            )
            if correlation is None:
                continue
            relations.append(
                {
                    "controller": controller,
                    "seed": seed,
                    "action_metric": action_field,
                    "outcome_metric": outcome_field,
                    "within_episode_pearson_r": correlation,
                    "windows": len(ordered),
                }
            )
    return relations


def _bootstrap_mean_interval(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _relation_summary(
    rows: Sequence[Mapping[str, Any]], samples: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["controller"]), str(row["action_metric"]), str(row["outcome_metric"]))
        grouped.setdefault(key, []).append(float(row["within_episode_pearson_r"]))
    summary: list[dict[str, Any]] = []
    for (controller, action_metric, outcome_metric), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        low, high = _bootstrap_mean_interval(array, samples, rng)
        summary.append(
            {
                "controller": controller,
                "action_metric": action_metric,
                "outcome_metric": outcome_metric,
                "episodes_with_variation": int(array.size),
                "mean_within_episode_pearson_r": float(array.mean()),
                "median_within_episode_pearson_r": float(np.median(array)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "interpretation_limit": "diagnostic_association_not_causal_effect",
            }
        )
    return summary


_PAIRED_FIELDS = (
    "team_reward",
    "j_loc",
    "j_com_loss",
    "j_com_aoi",
    "j_com_delay",
    "j_com_connectivity",
    "j_com",
    "joint_objective",
    "packet_loss",
    "packet_loss_p95",
    "delay_s",
    "delay_s_p95",
    "aoi_s",
    "aoi_s_p95",
    "connectivity",
    "no_candidate_window_fraction",
    "nlos_fraction",
    "mean_queue_length",
    "mean_delta_v_norm",
    "mean_bandwidth_entropy",
    "mean_max_bandwidth_share",
    "measurement_selection_fraction",
    "collision_penalty",
    "energy_penalty",
)


def _paired_effects(
    episode_rows: Sequence[Mapping[str, Any]], samples: int, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(str(row["controller"]), int(row["seed"])): row for row in episode_rows}
    mappo_seeds = {seed for controller, seed in by_key if controller == "mappo"}
    baseline_seeds = {seed for controller, seed in by_key if controller == "independent_fixed"}
    if mappo_seeds != baseline_seeds:
        raise ValueError("MAPPO and baseline episode seeds must match exactly")
    paired: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for field in _PAIRED_FIELDS:
        differences: list[float] = []
        for seed in sorted(mappo_seeds):
            mappo = float(by_key[("mappo", seed)][field])
            baseline = float(by_key[("independent_fixed", seed)][field])
            difference = baseline - mappo
            differences.append(difference)
            paired.append(
                {
                    "seed": seed,
                    "metric": field,
                    "mappo": mappo,
                    "independent_fixed": baseline,
                    "baseline_minus_mappo": difference,
                }
            )
        array = np.asarray(differences, dtype=float)
        low, high = _bootstrap_mean_interval(array, samples, rng)
        summary.append(
            {
                "metric": field,
                "paired_episodes": int(array.size),
                "mappo_mean": _mean(float(by_key[("mappo", seed)][field]) for seed in mappo_seeds),
                "baseline_mean": _mean(
                    float(by_key[("independent_fixed", seed)][field]) for seed in baseline_seeds
                ),
                "mean_baseline_minus_mappo": float(array.mean()),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
            }
        )
    return paired, summary


def _plot_time_chain(window_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("packet_loss", "Packet loss"),
        ("delay_s", "Delay (s)"),
        ("aoi_s", "AoI (s)"),
        ("connectivity", "Connectivity"),
        ("j_com", "Communication cost J_com"),
        ("team_reward", "Team reward"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {"mappo": "#2563eb", "independent_fixed": "#dc2626"}
    labels = {"mappo": "MAPPO", "independent_fixed": "Independent fixed"}
    for axis, (metric, title) in zip(axes.flat, metrics):
        for controller in ("mappo", "independent_fixed"):
            selected = [row for row in window_rows if row["controller"] == controller]
            by_time: dict[float, list[float]] = {}
            for row in selected:
                by_time.setdefault(float(row["window_end_s"]), []).append(float(row[metric]))
            times = np.asarray(sorted(by_time), dtype=float)
            means = np.asarray([np.mean(by_time[time]) for time in times], dtype=float)
            errors = np.asarray(
                [
                    1.96 * np.std(by_time[time], ddof=1) / math.sqrt(len(by_time[time]))
                    if len(by_time[time]) > 1
                    else 0.0
                    for time in times
                ],
                dtype=float,
            )
            axis.plot(times, means, color=colors[controller], label=labels[controller])
            axis.fill_between(times, means - errors, means + errors, color=colors[controller], alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Applied window end (s)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Window-aligned action → KPI → objective → reward audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_jcom_components(episode_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = ("j_com_loss", "j_com_aoi", "j_com_delay", "j_com_connectivity", "j_com")
    labels = ("Loss", "AoI", "Delay", "-Connectivity", "J_com")
    controllers = ("mappo", "independent_fixed")
    colors = ("#2563eb", "#dc2626")
    x = np.arange(len(components), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, (controller, color) in enumerate(zip(controllers, colors)):
        selected = [row for row in episode_rows if row["controller"] == controller]
        means = [_mean(float(row[field]) for row in selected) for field in components]
        errors = [
            1.96 * np.std([float(row[field]) for row in selected], ddof=1) / math.sqrt(len(selected))
            if len(selected) > 1
            else 0.0
            for field in components
        ]
        axis.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label="MAPPO" if controller == "mappo" else "Independent fixed",
        )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Episode mean weighted contribution")
    axis.set_title("Communication objective decomposition (95% episode-level CI)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_paired_effects(paired_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("j_com", "packet_loss", "delay_s", "aoi_s", "connectivity", "team_reward")
    labels = {
        "j_com": "J_com",
        "packet_loss": "Packet loss",
        "delay_s": "Delay (s)",
        "aoi_s": "AoI (s)",
        "connectivity": "Connectivity",
        "team_reward": "Team reward",
    }
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        selected = [row for row in paired_rows if row["metric"] == metric]
        for row in selected:
            axis.plot(
                (0, 1),
                (float(row["mappo"]), float(row["independent_fixed"])),
                color="#94a3b8",
                alpha=0.45,
                linewidth=0.8,
            )
        axis.scatter(np.zeros(len(selected)), [float(row["mappo"]) for row in selected], color="#2563eb", s=16)
        axis.scatter(
            np.ones(len(selected)),
            [float(row["independent_fixed"]) for row in selected],
            color="#dc2626",
            s=16,
        )
        axis.set_xticks((0, 1), ("MAPPO", "Independent"))
        axis.set_title(labels[metric])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Paired episode effects (same seeds and scenarios)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_saturation(episode_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (
        "normalized_uncertainty_clip_fraction",
        "normalized_packet_loss_clip_fraction",
        "normalized_aoi_clip_fraction",
        "normalized_delay_clip_fraction",
        "normalized_connectivity_clip_fraction",
    )
    labels = ("Uncertainty", "Packet loss", "AoI", "Delay", "Connectivity")
    x = np.arange(len(fields), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, (controller, color, label) in enumerate(
        (("mappo", "#2563eb", "MAPPO"), ("independent_fixed", "#dc2626", "Independent fixed"))
    ):
        selected = [row for row in episode_rows if row["controller"] == controller]
        means = [_mean(float(row[field]) for row in selected) for field in fields]
        axis.bar(x + (index - 0.5) * width, means, width, color=color, label=label)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of windows clipped at 1")
    axis.set_title("Training-P95 metric normalization saturation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_causal_audit(config: CausalAuditConfig) -> dict[str, Any]:
    """Audit a paired MAPPO/baseline split without changing either run."""

    reference_scale = Path(config.reference_scale)
    baseline_scale = Path(config.baseline_scale)
    reference_episodes = _episode_directories(reference_scale, config.split)
    baseline_episodes = _episode_directories(baseline_scale, config.split)
    if reference_episodes.keys() != baseline_episodes.keys():
        raise ValueError("reference and baseline splits must contain exactly the same seeds")

    reference_manifest = json.loads(
        (reference_scale / "manifest.json").read_text(encoding="utf-8")
    )
    communication = dict(reference_manifest.get("communication_parameters", {}))
    worst_case_delay_s = float(communication.get("max_delay_s", 1.0))
    worst_case_aoi_s = float(communication.get("aoi_cap_s", 60.0))

    run_name = reference_scale.parent.name
    audit_name = "causal_audit" if config.split == "test" else f"causal_audit_{config.split}"
    logs_dir = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / run_name / audit_name
    plots_dir = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / run_name / audit_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    window_rows: list[dict[str, Any]] = []
    for seed in sorted(reference_episodes):
        window_rows.extend(
            _audit_episode(
                "mappo",
                reference_episodes[seed],
                worst_case_delay_s=worst_case_delay_s,
                worst_case_aoi_s=worst_case_aoi_s,
            )
        )
        window_rows.extend(
            _audit_episode(
                "independent_fixed",
                baseline_episodes[seed],
                worst_case_delay_s=worst_case_delay_s,
                worst_case_aoi_s=worst_case_aoi_s,
            )
        )
    episode_rows = _episode_summaries(window_rows)
    relation_rows = _episode_relations(window_rows)
    rng = np.random.default_rng(config.bootstrap_seed)
    relation_summary = _relation_summary(relation_rows, config.bootstrap_samples, rng)
    paired_rows, paired_summary = _paired_effects(
        episode_rows, config.bootstrap_samples, rng
    )

    max_objective_error = max(float(row["objective_identity_abs_error"]) for row in window_rows)
    max_reward_error = max(float(row["reward_identity_abs_error"]) for row in window_rows)
    if max_objective_error > 1e-6 or max_reward_error > 1e-6:
        raise AssertionError(
            "objective/reward chain does not close: "
            f"objective={max_objective_error:.3g}, reward={max_reward_error:.3g}"
        )

    paths = {
        "window_audit": logs_dir / "window_action_kpi_objective_reward.csv",
        "episode_summary": logs_dir / "episode_summary.csv",
        "episode_relations": logs_dir / "episode_action_kpi_relations.csv",
        "relation_summary": logs_dir / "action_kpi_relation_summary.csv",
        "paired_effects": logs_dir / "paired_episode_effects.csv",
        "paired_summary": logs_dir / "paired_episode_summary.csv",
    }
    _write_csv(paths["window_audit"], window_rows)
    _write_csv(paths["episode_summary"], episode_rows)
    if relation_rows:
        _write_csv(paths["episode_relations"], relation_rows)
    if relation_summary:
        _write_csv(paths["relation_summary"], relation_summary)
    _write_csv(paths["paired_effects"], paired_rows)
    _write_csv(paths["paired_summary"], paired_summary)

    plot_paths = {
        "time_chain": plots_dir / "action_kpi_chain.png",
        "jcom_components": plots_dir / "jcom_contributions.png",
        "paired_effects": plots_dir / "paired_episode_effects.png",
        "normalization_saturation": plots_dir / "normalization_saturation.png",
    }
    _plot_time_chain(window_rows, plot_paths["time_chain"])
    _plot_jcom_components(episode_rows, plot_paths["jcom_components"])
    _plot_paired_effects(paired_rows, plot_paths["paired_effects"])
    _plot_saturation(episode_rows, plot_paths["normalization_saturation"])

    report = {
        "status": "diagnostic_association_audit_not_causal_effect_estimate",
        "reference_scale": str(reference_scale),
        "baseline_scale": str(baseline_scale),
        "split": config.split,
        "paired_episodes": len(reference_episodes),
        "window_rows": len(window_rows),
        "objective_identity_max_abs_error": max_objective_error,
        "reward_identity_max_abs_error": max_reward_error,
        "logs": {name: str(path) for name, path in paths.items() if path.exists()},
        "plots": {name: str(path) for name, path in plot_paths.items()},
        "next_inference_boundary": (
            "Use this audit to locate weak links; confirm action effects with pre-registered "
            "controlled action perturbations before making causal claims."
        ),
    }
    (logs_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
