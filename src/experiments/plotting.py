"""Generate the eight paper-oriented main figures from episode logs.

The plotting layer is intentionally downstream of the simulator.  It reads the
canonical episode CSV files and derives analysis tables without changing the
episode contract. Validation and test records remain separate throughout
the aggregation; the test split is used for the main time-series figures.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import candidate_link_series


SCALE_COLORS = {3: "#2563eb", 5: "#16a34a", 7: "#dc2626"}
SPLIT_STYLES = {"validation": "--", "test": "-"}
DEFAULT_WEIGHTS = {
    "uncertainty": 0.35,
    "loss": 0.20,
    "aoi": 0.15,
    "delay": 0.15,
    "connectivity": 0.15,
}
DEFAULT_LAMBDA_COLL = 10.0
DEFAULT_LAMBDA_ENERGY = 0.01


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _timestamp(row: Mapping[str, Any]) -> float:
    return round(_number(row.get("timestamp"), 0.0), 6)


def _group_by_timestamp(rows: Iterable[Mapping[str, Any]]) -> dict[float, list[Mapping[str, Any]]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_timestamp(row)].append(row)
    return dict(sorted(grouped.items()))


def _unique_first(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [group[0] for group in _group_by_timestamp(rows).values() if group]


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    numbers = np.asarray([_number(value) for value in values], dtype=float)
    return numbers[np.isfinite(numbers)]


def _mean(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = _finite_values(values)
    return float(np.mean(numbers)) if numbers.size else float(default)


def _std(values: Iterable[Any]) -> float:
    numbers = _finite_values(values)
    return float(np.std(numbers, ddof=1)) if numbers.size > 1 else 0.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def _scale_directories(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.glob("scale_*")
            if path.is_dir() and path.name[6:] in {"3", "5", "7"}
        ],
        key=lambda path: int(path.name[6:]),
    )


def _reward_parameters(scenario: Mapping[str, Any]) -> dict[str, Any]:
    configured = scenario.get("reward_config", {})
    configured = configured if isinstance(configured, Mapping) else {}
    configured_weights = configured.get("weights", {})
    configured_weights = configured_weights if isinstance(configured_weights, Mapping) else {}
    weights = {
        key: _number(configured_weights.get(key), fallback)
        for key, fallback in DEFAULT_WEIGHTS.items()
    }
    return {
        "weights": weights,
        "lambda_coll": _number(configured.get("lambda_coll"), DEFAULT_LAMBDA_COLL),
        "lambda_energy": _number(configured.get("lambda_energy"), DEFAULT_LAMBDA_ENERGY),
        "source": "scenario.json" if configured else "python RewardConfig defaults",
    }


def _state_maps(
    ground_truth: Sequence[Mapping[str, Any]],
    estimates: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, float], Mapping[str, Any]], dict[tuple[str, float], Mapping[str, Any]]]:
    truth_map = {(str(row.get("uav_id")), _timestamp(row)): row for row in ground_truth}
    estimate_map = {(str(row.get("uav_id")), _timestamp(row)): row for row in estimates}
    return truth_map, estimate_map


def _localization_series(
    ground_truth: Sequence[Mapping[str, Any]],
    estimates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    truth_map, estimate_map = _state_maps(ground_truth, estimates)
    by_time: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {"position": [], "velocity": [], "covariance": []}
    )
    position_errors: list[float] = []
    velocity_errors: list[float] = []
    for key in sorted(set(truth_map) & set(estimate_map)):
        truth = truth_map[key]
        estimate = estimate_map[key]
        position_error = np.asarray(
            [_number(estimate.get(axis)) - _number(truth.get(axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
        velocity_error = np.asarray(
            [_number(estimate.get(axis)) - _number(truth.get(axis)) for axis in ("vx", "vy", "vz")],
            dtype=float,
        )
        if not np.isfinite(position_error).all() or not np.isfinite(velocity_error).all():
            continue
        position_norm = float(np.linalg.norm(position_error))
        velocity_norm = float(np.linalg.norm(velocity_error))
        position_errors.append(position_norm)
        velocity_errors.append(velocity_norm)
        covariance_trace = 0.0
        try:
            covariance = np.asarray(
                json.loads(str(estimate.get("covariance_flat_json", "[]"))),
                dtype=float,
            ).reshape(-1)
            dimension = int(round(math.sqrt(covariance.size)))
            if dimension * dimension == covariance.size and dimension >= 3:
                covariance_trace = float(
                    covariance[0] + covariance[dimension + 1] + covariance[2 * dimension + 2]
                )
        except (TypeError, ValueError, json.JSONDecodeError):
            covariance_trace = 0.0
        entry = by_time[_timestamp(truth)]
        entry["position"].append(position_norm)
        entry["velocity"].append(velocity_norm)
        entry["covariance"].append(covariance_trace)

    series = []
    for timestamp, values in sorted(by_time.items()):
        series.append(
            {
                "timestamp": timestamp,
                "position_rmse_m": float(np.sqrt(np.mean(np.square(values["position"])))) if values["position"] else 0.0,
                "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(values["velocity"])))) if values["velocity"] else 0.0,
                "covariance_position_trace": _mean(values["covariance"]),
            }
        )
    return series, np.asarray(position_errors, dtype=float), np.asarray(velocity_errors, dtype=float)


def _communication_series(links: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    return candidate_link_series(links)


def _reward_series(
    rewards: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> list[dict[str, float]]:
    weights = parameters["weights"]
    grouped = _group_by_timestamp(rewards)
    series: list[dict[str, float]] = []
    for timestamp, rows in grouped.items():
        j = _mean(row.get("joint_objective") for row in rows)
        j_loc = weights["uncertainty"] * _mean(row.get("normalized_uncertainty") for row in rows)
        collision_penalty = _mean(-_number(row.get("reward_collision"), 0.0) for row in rows)
        energy_penalty = _mean(-_number(row.get("reward_energy"), 0.0) for row in rows)
        team_reward = _mean(row.get("reward_total") for row in rows)
        learning_reward = _mean(
            row.get("learning_reward_total") for row in rows
        )
        series.append(
            {
                "timestamp": timestamp,
                "mean_reward": team_reward,
                "learning_reward": learning_reward,
                "j_loc": j_loc,
                "j_com": j - j_loc,
                "joint_objective": j,
                "collision_penalty": collision_penalty,
                "energy_penalty": energy_penalty,
                "energy_normalized": energy_penalty / max(float(parameters["lambda_energy"]), 1e-12),
                "collision": _mean(row.get("collision") for row in rows),
                "connectivity": _mean(row.get("reward_connectivity") for row in rows),
                "localization_progress_score": _mean(
                    row.get("localization_progress_score") for row in rows
                ),
                "goal_progress_score": _mean(
                    row.get("goal_progress_score") for row in rows
                ),
                "goal_progress_reward": _mean(
                    row.get("single_uav_goal_progress_reward") for row in rows
                ),
                "goal_bonus": _mean(
                    row.get("single_uav_goal_bonus") for row in rows
                ),
                "localization_cost": _mean(
                    row.get("single_uav_localization_cost") for row in rows
                ),
                "goal_reached": _mean(
                    row.get("single_uav_goal_reached") for row in rows
                ),
                "goal_in_radius": _mean(
                    row.get("single_uav_goal_in_radius") for row in rows
                ),
                "goal_speed_ok": _mean(
                    row.get("single_uav_goal_speed_ok") for row in rows
                ),
                "goal_stable": _mean(
                    row.get("single_uav_goal_stable") for row in rows
                ),
                "speed_mps": _mean(
                    row.get("single_uav_speed_mps") for row in rows
                ),
                "near_goal_speed_cost": _mean(
                    row.get("single_uav_near_goal_speed_cost") for row in rows
                ),
                "goal_distance": _mean(
                    row.get("current_goal_distance") for row in rows
                ),
            }
        )
    return series


def _gnss_series(gnss: Sequence[Mapping[str, Any]]) -> dict[float, float]:
    values: dict[float, float] = {}
    for timestamp, rows in _group_by_timestamp(gnss).items():
        modes = {str(row.get("mode", "unknown")).lower() for row in rows}
        values[timestamp] = 1.0 if modes & {"denied", "degraded", "invalid", "none"} else 0.0
    return values


def _obstacle_clearance(position: np.ndarray, obstacle: Mapping[str, Any]) -> float:
    geometry = str(obstacle.get("geometry_type", "")).lower()
    center = np.asarray(
        [_number(obstacle.get("x")), _number(obstacle.get("y")), _number(obstacle.get("z"))],
        dtype=float,
    )
    if geometry == "sphere":
        radius = max(0.0, _number(obstacle.get("radius"), 0.0))
        return float(np.linalg.norm(position - center) - radius)
    bounds = np.asarray(
        [
            [_number(obstacle.get("x_min")), _number(obstacle.get("x_max"))],
            [_number(obstacle.get("y_min")), _number(obstacle.get("y_max"))],
            [_number(obstacle.get("z_min")), _number(obstacle.get("z_max"))],
        ],
        dtype=float,
    )
    if not np.isfinite(bounds).all():
        radius = max(0.0, _number(obstacle.get("radius"), 0.0))
        return float(np.linalg.norm(position - center) - radius)
    outside = np.maximum(np.maximum(bounds[:, 0] - position, 0.0), position - bounds[:, 1])
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance
    inside_depth = float(np.min(np.minimum(position - bounds[:, 0], bounds[:, 1] - position)))
    return -inside_depth


def _safety_summary(
    ground_truth: Sequence[Mapping[str, Any]],
    rewards: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    obstacles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, Any]]:
    truth_by_time: dict[float, list[np.ndarray]] = defaultdict(list)
    truth_by_uav: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for row in ground_truth:
        position = np.asarray([_number(row.get(axis)) for axis in ("x", "y", "z")], dtype=float)
        if np.isfinite(position).all():
            timestamp = _timestamp(row)
            truth_by_time[timestamp].append(position)
            truth_by_uav[str(row.get("uav_id"))].append((timestamp, position))

    obstacle_by_time = _group_by_timestamp(obstacles)
    safety_series: list[dict[str, float]] = []
    clearance_samples: list[float] = []
    for timestamp, positions in sorted(truth_by_time.items()):
        clearance = []
        for position in positions:
            for obstacle in obstacle_by_time.get(timestamp, []):
                if int(round(_number(obstacle.get("active"), 1.0))) == 1:
                    value = _obstacle_clearance(position, obstacle)
                    if np.isfinite(value):
                        clearance.append(value)
        minimum_clearance = float(np.min(clearance)) if clearance else 0.0
        clearance_samples.extend(clearance)
        safety_series.append({"timestamp": timestamp, "minimum_clearance_m": minimum_clearance})

    path_lengths = []
    for positions in truth_by_uav.values():
        positions.sort(key=lambda item: item[0])
        if len(positions) >= 2:
            path_lengths.append(float(sum(np.linalg.norm(b - a) for (_, a), (_, b) in zip(positions, positions[1:]))))
        else:
            path_lengths.append(0.0)

    action_norms = []
    for row in actions:
        vector = np.asarray([_number(row.get(f"delta_v{axis}"), 0.0) for axis in ("x", "y", "z")], dtype=float)
        if np.isfinite(vector).all():
            action_norms.append(float(np.linalg.norm(vector)))
    reward_times = _unique_first(rewards)
    safety_projection_count = sum(
        int(round(_number(row.get("safety_projected"), 0.0)))
        for row in actions
        if np.isfinite(_number(row.get("safety_projected"), 0.0))
    )
    boundary_projection_count = sum(
        int(round(_number(row.get("boundary_projected"), 0.0)))
        for row in actions
        if np.isfinite(_number(row.get("boundary_projected"), 0.0))
    )
    summary = {
        "collision_rate": _mean(row.get("collision") for row in reward_times),
        "path_length_m": _mean(path_lengths),
        "minimum_clearance_m": float(np.min(clearance_samples)) if clearance_samples else 0.0,
        "action_delta_norm": _mean(action_norms),
        "safety_projection_count": float(safety_projection_count),
        "boundary_projection_count": float(boundary_projection_count),
    }
    return summary, safety_series, {
        "truth_by_uav": truth_by_uav,
        "obstacles": list(obstacles),
        "collision_times": [
            _timestamp(row) for row in reward_times if _number(row.get("collision"), 0.0) > 0.0
        ],
    }


def _load_episode(episode_path: Path, scale: int, split: str) -> dict[str, Any]:
    scenario = _read_json(episode_path / "scenario.json")
    parameters = _reward_parameters(scenario)
    ground_truth = _read_csv(episode_path / "ground_truth.csv")
    estimates = _read_csv(episode_path / "estimates.csv")
    rewards = _read_csv(episode_path / "rewards.csv")
    links = _read_csv(episode_path / "links.csv")
    actions = _read_csv(episode_path / "actions.csv")
    obstacles = _read_csv(episode_path / "obstacles.csv")
    gnss = _read_csv(episode_path / "gnss.csv")
    ekf_diagnostics = _read_csv(episode_path / "ekf_diagnostics.csv")

    localization, position_errors, velocity_errors = _localization_series(ground_truth, estimates)
    communication = _communication_series(links)
    reward_series = _reward_series(rewards, parameters)
    safety_summary, safety_series, trajectory = _safety_summary(
        ground_truth, rewards, actions, obstacles
    )
    estimate_by_uav: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for row in estimates:
        point = np.asarray([_number(row.get(axis)) for axis in ("x", "y", "z")], dtype=float)
        if np.isfinite(point).all():
            estimate_by_uav[str(row.get("uav_id"))].append((_timestamp(row), point))
    trajectory["estimate_by_uav"] = dict(estimate_by_uav)
    localization_summary = {
        "position_rmse_m": _mean(row["position_rmse_m"] for row in localization),
        "velocity_rmse_mps": _mean(row["velocity_rmse_mps"] for row in localization),
        "covariance_position_trace": _mean(row["covariance_position_trace"] for row in localization),
    }
    communication_summary = {
        "packet_loss": _mean(row["packet_loss"] for row in communication),
        "delay_s": _mean(row["delay_s"] for row in communication),
        "aoi_s": _mean(row["aoi_s"] for row in communication),
    }
    reward_summary = {
        "mean_reward": _mean(row["mean_reward"] for row in reward_series),
        "joint_objective": _mean(row["joint_objective"] for row in reward_series),
        "j_loc": _mean(row["j_loc"] for row in reward_series),
        "j_com": _mean(row["j_com"] for row in reward_series),
        "collision_penalty": _mean(row["collision_penalty"] for row in reward_series),
        "energy_penalty": _mean(row["energy_penalty"] for row in reward_series),
        "energy_normalized": _mean(row["energy_normalized"] for row in reward_series),
        "connectivity": _mean(row["connectivity"] for row in reward_series),
    }
    goals = scenario.get("goals", {})
    return {
        "episode_id": str(scenario.get("episode_id", episode_path.name)),
        "scenario_id": str(scenario.get("scenario_id", "")),
        "seed": int(round(_number(scenario.get("seed"), 0.0))),
        "scale": int(scale),
        "split": str(split),
        "parameters": parameters,
        "reward_summary": {**reward_summary, **communication_summary, **localization_summary, **safety_summary},
        "localization": localization,
        "communication": communication,
        "reward_series": reward_series,
        "safety_series": safety_series,
        "gnss": _gnss_series(gnss),
        "position_errors": position_errors,
        "velocity_errors": velocity_errors,
        "trajectory": trajectory,
        "ekf_diagnostics": ekf_diagnostics,
        "goals": goals if isinstance(goals, Mapping) else {},
    }


def _collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for scale_dir in _scale_directories(root):
        scale = int(scale_dir.name[6:])
        metrics_rows = _read_csv(scale_dir / "diagnostics" / "episode_metrics.csv")
        for row in metrics_rows:
            split = str(row.get("split", ""))
            episode_id = str(row.get("episode_id", ""))
            if split not in {"validation", "test"} or not episode_id:
                continue
            episode_path = scale_dir / split / episode_id
            if episode_path.is_dir():
                episodes.append(_load_episode(episode_path, scale, split))
        training_path = scale_dir / "diagnostics" / "training_metrics.jsonl"
        if training_path.exists():
            with training_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        row["uav_count"] = int(round(_number(row.get("uav_count"), scale)))
                        training.append(row)
    if not episodes:
        raise FileNotFoundError(f"no validation/test episode logs found under {root}")
    return episodes, training


def _episode_summary_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        summary = episode["reward_summary"]
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "seed": episode["seed"],
                "uav_count": episode["scale"],
                "split": episode["split"],
                **{key: float(value) for key, value in summary.items()},
            }
        )
    return rows


def _series_rows(episodes: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if kind == "localization":
            values = episode["localization"]
        elif kind == "communication":
            values = episode["communication"]
        elif kind == "objective":
            values = episode["reward_series"]
        else:
            values = episode["safety_series"]
        for value in values:
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "seed": episode["seed"],
                    "uav_count": episode["scale"],
                    "split": episode["split"],
                    **{key: float(item) for key, item in value.items()},
                }
            )
    return rows


def _plot_series(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    ylabel: str,
    bounds: tuple[float | None, float | None] | None = None,
) -> None:
    for scale in sorted({int(row["uav_count"]) for row in rows}):
        selected = [row for row in rows if int(row["uav_count"]) == scale]
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in selected:
            value = _number(row.get(metric))
            if np.isfinite(value):
                grouped[_number(row.get("timestamp"), 0.0)].append(value)
        timestamps = sorted(grouped)
        if not timestamps:
            continue
        means = np.asarray([float(np.mean(grouped[timestamp])) for timestamp in timestamps], dtype=float)
        deviations = np.asarray([float(np.std(grouped[timestamp])) for timestamp in timestamps], dtype=float)
        color = SCALE_COLORS.get(scale, "#7c3aed")
        lower = means - deviations
        upper = means + deviations
        if bounds is not None:
            lower_bound, upper_bound = bounds
            if lower_bound is not None:
                means = np.maximum(means, lower_bound)
                lower = np.maximum(lower, lower_bound)
                upper = np.maximum(upper, lower_bound)
            if upper_bound is not None:
                means = np.minimum(means, upper_bound)
                lower = np.minimum(lower, upper_bound)
                upper = np.minimum(upper, upper_bound)
        ax.plot(timestamps, means, color=color, label=f"{scale} UAV")
        if np.any(deviations):
            ax.fill_between(timestamps, lower, upper, color=color, alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _plot_training(training: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_reward", "Mean team reward"),
        ("loss", "MAPPO loss"),
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("entropy", "Entropy"),
        ("clip_fraction", "Clip fraction"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for scale in sorted({int(row["uav_count"]) for row in training}):
            selected = [row for row in training if int(row["uav_count"]) == scale]
            selected.sort(key=lambda row: int(_number(row.get("update"), 0.0)))
            x = [int(_number(row.get("update"), index + 1)) for index, row in enumerate(selected)]
            y = [_number(row.get(metric), 0.0) for row in selected]
            axis.plot(x, y, color=SCALE_COLORS.get(scale, "#7c3aed"), label=f"{scale} UAV")
            collision_rows = [
                (x_value, y_value)
                for x_value, y_value, row in zip(x, y, selected)
                if bool(row.get("collision_occurred", False))
            ]
            if collision_rows:
                axis.scatter(
                    [item[0] for item in collision_rows],
                    [item[1] for item in collision_rows],
                    color=SCALE_COLORS.get(scale, "#7c3aed"),
                    marker="x",
                    s=35,
                    zorder=4,
                )
        axis.set_title(label)
        axis.set_xlabel("Update")
        axis.grid(alpha=0.25)
        axis.legend()
    _save(figure, path)
    plt.close(figure)


def _plot_validation_test(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("mean_reward", "Mean reward"),
        ("joint_objective", "Core objective J"),
        ("position_rmse_m", "Position RMSE (m)"),
        ("packet_loss", "Packet loss"),
        ("delay_s", "Delay (s)"),
        ("aoi_s", "AoI (s)"),
        ("connectivity", "Connectivity"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        scales = sorted({int(episode["scale"]) for episode in episodes})
        x = np.arange(len(scales))
        width = 0.36
        for offset, split in enumerate(("validation", "test")):
            means = []
            deviations = []
            for scale in scales:
                values = [
                    episode["reward_summary"].get(metric, 0.0)
                    for episode in episodes
                    if episode["split"] == split and int(episode["scale"]) == scale
                ]
                means.append(_mean(values))
                deviations.append(_std(values))
            axis.bar(
                x + (offset - 0.5) * width,
                means,
                width,
                yerr=deviations,
                capsize=3,
                label=split,
                alpha=0.85,
            )
        axis.set_title(label)
        axis.set_xticks(x, [str(scale) for scale in scales])
        axis.set_xlabel("UAV count")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes.flat[-1].axis("off")
    _save(figure, path)
    plt.close(figure)


def _mean_series(rows: Sequence[Mapping[str, Any]], metric: str, scale: int) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if int(row["uav_count"]) == scale and row["split"] == "test"]
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in selected:
        value = _number(row.get(metric))
        if np.isfinite(value):
            grouped[_number(row.get("timestamp"), 0.0)].append(value)
    timestamps = sorted(grouped)
    return timestamps, [_mean(grouped[timestamp]) for timestamp in timestamps]


def _plot_objective(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _series_rows(episodes, "objective")
    scales = sorted({int(episode["scale"]) for episode in episodes})
    figure, axes = plt.subplots(2, len(scales), figsize=(5 * len(scales), 7), squeeze=False, constrained_layout=True)
    for column, scale in enumerate(scales):
        top = axes[0, column]
        bottom = axes[1, column]
        selected = [row for row in rows if int(row["uav_count"]) == scale and row["split"] == "test"]
        for metric, label, color in (
            ("j_loc", "J_loc", "#2563eb"),
            ("j_com", "J_com", "#16a34a"),
            ("joint_objective", "J", "#dc2626"),
        ):
            timestamps, values = _mean_series(selected, metric, scale)
            top.plot(timestamps, values, label=label, color=color)
        for metric, label, color in (
            ("mean_reward", "r", "#111827"),
            ("joint_objective", "-J", "#dc2626"),
            ("collision_penalty", "-collision penalty", "#f97316"),
            ("energy_penalty", "-energy penalty", "#8b5cf6"),
        ):
            timestamps, values = _mean_series(selected, metric, scale)
            if metric == "joint_objective":
                values = [-value for value in values]
            elif metric in {"collision_penalty", "energy_penalty"}:
                values = [-value for value in values]
            bottom.plot(timestamps, values, label=label, color=color)
        top.set_title(f"{scale} UAV: core objective")
        bottom.set_title(f"{scale} UAV: reward decomposition")
        for axis in (top, bottom):
            axis.set_xlabel("Time (s)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    _save(figure, path)
    plt.close(figure)


def _plot_localization(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _series_rows(episodes, "localization")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, metric, label in (
        (axes[0, 0], "position_rmse_m", "Position RMSE (m)"),
        (axes[0, 1], "velocity_rmse_mps", "Velocity RMSE (m/s)"),
        (axes[1, 0], "covariance_position_trace", "Position covariance trace"),
    ):
        _plot_series(
            axis,
            [row for row in rows if row["split"] == "test"],
            metric,
            ylabel=label,
            bounds=(0.0, None),
        )
        axis.set_title(label)

    gnss_axis = axes[1, 1]
    for scale in sorted({int(episode["scale"]) for episode in episodes}):
        selected = [episode for episode in episodes if int(episode["scale"]) == scale and episode["split"] == "test"]
        grouped: dict[float, list[float]] = defaultdict(list)
        for episode in selected:
            for timestamp, value in episode["gnss"].items():
                grouped[float(timestamp)].append(float(value))
        timestamps = sorted(grouped)
        gnss_axis.plot(
            timestamps,
            [_mean(grouped[timestamp]) for timestamp in timestamps],
            color=SCALE_COLORS.get(scale, "#7c3aed"),
            label=f"{scale} UAV",
        )
    gnss_axis.set_title("GNSS denied/degraded indicator")
    gnss_axis.set_xlabel("Time (s)")
    gnss_axis.set_ylabel("Fraction of UAVs")
    gnss_axis.set_ylim(-0.05, 1.05)
    gnss_axis.grid(alpha=0.25)
    gnss_axis.legend()
    _save(figure, path)
    plt.close(figure)


def _plot_cdf(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, key, label in (
        (axes[0], "position_errors", "Position error (m)"),
        (axes[1], "velocity_errors", "Velocity error (m/s)"),
    ):
        for split in ("validation", "test"):
            for scale in sorted({int(episode["scale"]) for episode in episodes}):
                samples = [
                    np.asarray(episode[key], dtype=float)
                    for episode in episodes
                    if episode["split"] == split and int(episode["scale"]) == scale
                ]
                values = np.concatenate(samples) if samples else np.empty(0, dtype=float)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                values.sort()
                axis.plot(
                    values,
                    np.linspace(0.0, 1.0, values.size, endpoint=True),
                    color=SCALE_COLORS.get(scale, "#7c3aed"),
                    linestyle=SPLIT_STYLES[split],
                    label=f"{scale} UAV {split}",
                )
        axis.set_xlabel(label)
        axis.set_ylabel("CDF")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    _save(figure, path)
    plt.close(figure)


def _plot_communication(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _series_rows(episodes, "communication")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, metric, label in (
        (axes[0, 0], "packet_loss", "Packet loss"),
        (axes[0, 1], "delay_s", "Delay (s)"),
        (axes[1, 0], "aoi_s", "AoI (s)"),
    ):
        bounds = (0.0, 1.0) if metric == "packet_loss" else (0.0, None)
        _plot_series(
            axis,
            [row for row in rows if row["split"] == "test"],
            metric,
            ylabel=label,
            bounds=bounds,
        )
        axis.set_title(label)
    connectivity_axis = axes[1, 1]
    reward_rows = []
    for episode in episodes:
        if episode["split"] != "test":
            continue
        for row in episode["reward_series"]:
            reward_rows.append({"uav_count": episode["scale"], "timestamp": row["timestamp"], "connectivity": row["connectivity"], "split": "test"})
    _plot_series(
        connectivity_axis,
        reward_rows,
        "connectivity",
        ylabel="Largest connected component fraction",
        bounds=(0.0, 1.0),
    )
    connectivity_axis.set_title("Connectivity")
    _save(figure, path)
    plt.close(figure)


def _plot_safety(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("collision_rate", "Collision rate"),
        ("energy_normalized", "Normalized energy"),
        ("minimum_clearance_m", "Minimum clearance (m)"),
        ("path_length_m", "Mean path length (m)"),
        ("action_delta_norm", "Mean velocity increment"),
        ("safety_projection_count", "Safety projection count"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    scales = sorted({int(episode["scale"]) for episode in episodes})
    x = np.arange(len(scales))
    selected_episodes = [episode for episode in episodes if episode["split"] == "test"]
    for axis, (metric, label) in zip(axes.flat, metrics):
        means = []
        deviations = []
        for scale in scales:
            values = [episode["reward_summary"].get(metric, 0.0) for episode in selected_episodes if int(episode["scale"]) == scale]
            means.append(_mean(values))
            deviations.append(_std(values))
        axis.bar(x, means, yerr=deviations, capsize=3, color=[SCALE_COLORS.get(scale, "#7c3aed") for scale in scales])
        axis.set_title(label)
        axis.set_xticks(x, [str(scale) for scale in scales])
        axis.set_xlabel("UAV count")
        axis.grid(axis="y", alpha=0.25)
    _save(figure, path)
    plt.close(figure)


def _draw_box(axis: Any, obstacle: Mapping[str, Any]) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x_min, x_max = _number(obstacle.get("x_min")), _number(obstacle.get("x_max"))
    y_min, y_max = _number(obstacle.get("y_min")), _number(obstacle.get("y_max"))
    z_min, z_max = _number(obstacle.get("z_min")), _number(obstacle.get("z_max"))
    if not np.isfinite([x_min, x_max, y_min, y_max, z_min, z_max]).all():
        return
    vertices = np.asarray(
        [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
        ],
        dtype=float,
    )
    faces = [
        [vertices[index] for index in (0, 1, 2, 3)],
        [vertices[index] for index in (4, 5, 6, 7)],
        [vertices[index] for index in (0, 1, 5, 4)],
        [vertices[index] for index in (2, 3, 7, 6)],
        [vertices[index] for index in (1, 2, 6, 5)],
        [vertices[index] for index in (0, 3, 7, 4)],
    ]
    axis.add_collection3d(Poly3DCollection(faces, alpha=0.12, facecolor="#6b7280", edgecolor="#4b5563"))


def _plot_trajectory(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # register 3D projection

    scales = sorted({int(episode["scale"]) for episode in episodes})
    figure = plt.figure(figsize=(6 * max(1, len(scales)), 6))
    for index, scale in enumerate(scales, start=1):
        axis = figure.add_subplot(1, len(scales), index, projection="3d")
        candidates = [episode for episode in episodes if episode["split"] == "test" and int(episode["scale"]) == scale]
        if not candidates:
            axis.set_title(f"{scale} UAV (no test data)")
            continue
        episode = sorted(candidates, key=lambda item: item["episode_id"])[0]
        trajectory = episode["trajectory"]
        for uav_id, values in sorted(trajectory["truth_by_uav"].items()):
            values = sorted(values, key=lambda item: item[0])
            points = np.asarray([point for _, point in values], dtype=float)
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color=SCALE_COLORS.get(scale, "#7c3aed"), alpha=0.65, label=f"{uav_id} truth")
        for uav_id, values in sorted(trajectory.get("estimate_by_uav", {}).items()):
            values = sorted(values, key=lambda item: item[0])
            points = np.asarray([point for _, point in values], dtype=float)
            if points.size:
                axis.plot(
                    points[:, 0], points[:, 1], points[:, 2],
                    color=SCALE_COLORS.get(scale, "#7c3aed"),
                    linestyle="--",
                    alpha=0.55,
                    label=f"{uav_id} EKF",
                )
        for obstacle in trajectory["obstacles"]:
            if str(obstacle.get("geometry_type", "")).lower() == "box":
                _draw_box(axis, obstacle)
            else:
                center = np.asarray([_number(obstacle.get(axis_name)) for axis_name in ("x", "y", "z")], dtype=float)
                if np.isfinite(center).all():
                    axis.scatter(center[0], center[1], center[2], color="#6b7280", marker="o", alpha=0.35)
        goal_payload = episode.get("goals", {})
        if isinstance(goal_payload, Mapping):
            for goal in goal_payload.values():
                goal = np.asarray(goal, dtype=float).reshape(-1)
                if goal.size >= 3 and np.isfinite(goal[:3]).all():
                    axis.scatter(goal[0], goal[1], goal[2], color="#111827", marker="*", s=55)
        collision_times = set(trajectory["collision_times"])
        if collision_times:
            for values in trajectory["truth_by_uav"].values():
                points = [point for timestamp, point in values if timestamp in collision_times]
                if points:
                    points = np.asarray(points, dtype=float)
                    axis.scatter(points[:, 0], points[:, 1], points[:, 2], color="#dc2626", marker="x", s=35)
        axis.set_title(f"{scale} UAV test trajectory")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.grid(alpha=0.25)
    _save(figure, path)
    plt.close(figure)


def _diagnostic_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for episode in episodes:
        for row in episode.get("ekf_diagnostics", []):
            payload: dict[str, float] = {
                "episode_id": episode["episode_id"],
                "seed": episode["seed"],
                "uav_count": episode["scale"],
                "split": episode["split"],
                "timestamp": _timestamp(row),
            }
            for key in (
                "position_error_m",
                "position_nees",
                "position_covariance_trace_m2",
                "nis_vo",
                "nis_uwb_mean",
                "nis_uwb_max",
                "nis_gnss",
                "nis_vo_accepted",
                "nis_vo_rejected",
                "nis_uwb_accepted_mean",
                "nis_uwb_rejected_mean",
                "nis_gnss_accepted",
                "nis_gnss_rejected",
            ):
                payload[key] = _number(row.get(key), -1.0)
            rows.append(payload)
    return rows


def _plot_ekf_consistency(
    episodes: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Plot v1.3 EKF consistency diagnostics without fabricating v1.2 data."""

    import matplotlib.pyplot as plt

    rows = _diagnostic_rows(episodes)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    if not rows:
        axes.flat[0].text(0.5, 0.5, "No episode-v1.3 EKF diagnostics", ha="center", va="center")
        for axis in axes.flat:
            axis.set_axis_off()
        _save(figure, path)
        plt.close(figure)
        return

    selected = [row for row in rows if row["split"] == "test"] or rows
    scales = sorted({int(row["uav_count"]) for row in selected})
    for scale in scales:
        scale_rows = [row for row in selected if int(row["uav_count"]) == scale]
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in scale_rows:
            if row["position_nees"] >= 0:
                grouped[row["timestamp"]].append(row["position_nees"])
        timestamps = sorted(grouped)
        if timestamps:
            axes[0, 0].plot(
                timestamps,
                [_mean(grouped[timestamp]) for timestamp in timestamps],
                color=SCALE_COLORS.get(scale, "#7c3aed"),
                label=f"{scale} UAV",
            )
    axes[0, 0].axhline(0.216, color="#6b7280", linestyle=":", linewidth=1, label="NEES 95% lower")
    axes[0, 0].axhline(7.815, color="#111827", linestyle="--", linewidth=1, label="NEES 95% upper")
    axes[0, 0].set_title("Position NEES (test)")
    axes[0, 0].set_ylabel("NEES")

    for scale in scales:
        scale_rows = [row for row in selected if int(row["uav_count"]) == scale]
        for metric, label, style in (
            ("nis_vo_accepted", "VO accepted NIS", "-"),
            ("nis_vo_rejected", "VO rejected NIS", "--"),
            ("nis_uwb_accepted_mean", "UWB accepted NIS", "-"),
            ("nis_uwb_rejected_mean", "UWB rejected NIS", "--"),
            ("nis_gnss_accepted", "GNSS accepted NIS", ":"),
            ("nis_gnss_rejected", "GNSS rejected NIS", "-.") ,
        ):
            grouped = defaultdict(list)
            for row in scale_rows:
                if row[metric] >= 0:
                    grouped[row["timestamp"]].append(row[metric])
            timestamps = sorted(grouped)
            if timestamps:
                axes[0, 1].plot(
                    timestamps,
                    [_mean(grouped[timestamp]) for timestamp in timestamps],
                    color=SCALE_COLORS.get(scale, "#7c3aed"),
                    linestyle=style,
                    label=f"{scale} UAV {label}",
                )
    axes[0, 1].set_title("Innovation NIS (test)")
    axes[0, 1].set_ylabel("NIS")

    for scale in scales:
        scale_rows = [row for row in selected if int(row["uav_count"]) == scale]
        for metric, label, style in (
            ("position_error_m", "Position error", "-"),
            ("position_covariance_trace_m2", "Covariance trace", "--"),
        ):
            grouped = defaultdict(list)
            for row in scale_rows:
                if row[metric] >= 0:
                    grouped[row["timestamp"]].append(row[metric])
            timestamps = sorted(grouped)
            if timestamps:
                axes[1, 0].plot(
                    timestamps,
                    [_mean(grouped[timestamp]) for timestamp in timestamps],
                    color=SCALE_COLORS.get(scale, "#7c3aed"),
                    linestyle=style,
                    label=f"{scale} UAV {label}",
                )
    axes[1, 0].set_title("Position error and covariance trace (test)")
    axes[1, 0].set_ylabel("m / m²")

    collision_points = []
    for episode in episodes:
        if episode["split"] != "test":
            continue
        collision_times = set(episode["trajectory"].get("collision_times", []))
        for row in episode.get("ekf_diagnostics", []):
            if _timestamp(row) in collision_times:
                collision_points.append((episode["scale"], _timestamp(row)))
    for scale, timestamp in collision_points:
        axes[1, 1].axvline(
            timestamp,
            color=SCALE_COLORS.get(int(scale), "#7c3aed"),
            alpha=0.35,
            linestyle="--",
        )
    axes[1, 1].text(
        0.03,
        0.92,
        "Dashed lines: collision termination",
        transform=axes[1, 1].transAxes,
        fontsize=9,
    )
    axes[1, 1].set_title("Collision termination markers")
    axes[1, 1].set_xlabel("Time (s)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    _save(figure, path)
    plt.close(figure)


def plot_main_figures(
    input_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    logs_output: str | Path | None = None,
    include_validation: bool = True,
    include_test: bool = True,
) -> dict[str, Any]:
    """Create the eight paper main figures and derived analysis CSV files."""

    root = Path(input_root)
    result_name = root.name
    plot_root = Path(output_dir) if output_dir is not None else Path("plots/preliminary") / result_name / "main_figures"
    log_root = Path(logs_output) if logs_output is not None else Path("logs/preliminary") / result_name / "main_figures"
    episodes, training = _collect(root)
    allowed_splits = set()
    if include_validation:
        allowed_splits.add("validation")
    if include_test:
        allowed_splits.add("test")
    episodes = [episode for episode in episodes if episode["split"] in allowed_splits]
    if not episodes:
        raise ValueError("no requested validation/test episodes are available")

    _write_csv(log_root / "validation_test_summary.csv", _episode_summary_rows(episodes))
    _write_csv(log_root / "objective_metrics.csv", _series_rows(episodes, "objective"))
    _write_csv(log_root / "localization_metrics.csv", _series_rows(episodes, "localization"))
    _write_csv(log_root / "communication_metrics.csv", _series_rows(episodes, "communication"))
    _write_csv(log_root / "safety_energy_metrics.csv", _episode_summary_rows(episodes), columns=[
        "episode_id", "seed", "uav_count", "split", "collision_rate", "energy_normalized",
        "minimum_clearance_m", "path_length_m", "action_delta_norm",
        "safety_projection_count", "boundary_projection_count",
    ])

    figures = {
        "fig01_training_curves": plot_root / "fig01_training_curves.png",
        "fig02_validation_test_comparison": plot_root / "fig02_validation_test_comparison.png",
        "fig03_objective_reward_decomposition": plot_root / "fig03_objective_reward_decomposition.png",
        "fig04_localization_time_series": plot_root / "fig04_localization_time_series.png",
        "fig05_localization_cdf": plot_root / "fig05_localization_cdf.png",
        "fig06_communication_metrics": plot_root / "fig06_communication_metrics.png",
        "fig07_safety_energy": plot_root / "fig07_safety_energy.png",
        "fig08_trajectory_3d": plot_root / "fig08_trajectory_3d.png",
    }
    _plot_training(training, figures["fig01_training_curves"])
    _plot_validation_test(episodes, figures["fig02_validation_test_comparison"])
    _plot_objective(episodes, figures["fig03_objective_reward_decomposition"])
    _plot_localization(episodes, figures["fig04_localization_time_series"])
    _plot_cdf(episodes, figures["fig05_localization_cdf"])
    _plot_communication(episodes, figures["fig06_communication_metrics"])
    _plot_safety(episodes, figures["fig07_safety_energy"])
    _plot_trajectory(episodes, figures["fig08_trajectory_3d"])

    diagnostic_root = plot_root.parent / "diagnostics"
    diagnostic_path = diagnostic_root / "ekf_consistency.png"
    diagnostic_rows = _diagnostic_rows(episodes)
    diagnostic_log_path = log_root.parent / "diagnostics" / "ekf_consistency_metrics.csv"
    _write_csv(diagnostic_log_path, diagnostic_rows)
    _plot_ekf_consistency(episodes, diagnostic_path)

    return {
        "input_root": str(root),
        "plots": {key: str(value) for key, value in figures.items()},
        "diagnostics": {
            "ekf_consistency": str(diagnostic_path),
        },
        "logs": {
            "validation_test_summary": str(log_root / "validation_test_summary.csv"),
            "objective_metrics": str(log_root / "objective_metrics.csv"),
            "localization_metrics": str(log_root / "localization_metrics.csv"),
            "communication_metrics": str(log_root / "communication_metrics.csv"),
            "safety_energy_metrics": str(log_root / "safety_energy_metrics.csv"),
            "ekf_consistency_metrics": str(diagnostic_log_path),
        },
        "episode_count": len(episodes),
        "training_rows": len(training),
        "splits": sorted({episode["split"] for episode in episodes}),
        "scales": sorted({int(episode["scale"]) for episode in episodes}),
        "result_status": "engineering_preliminary_main_figures",
    }


__all__ = ["plot_main_figures"]
