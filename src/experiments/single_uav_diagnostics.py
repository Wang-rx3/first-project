"""Episode-level diagnostic plots for a single-UAV run.

The figures in this module are diagnostic views of one recorded episode.  They
read the canonical episode CSV files and do not rerun the simulator or alter
the policy.  In particular, communication fields remain visible even when a
single-UAV episode has no inter-UAV links; that situation is recorded in the
manifest instead of being treated as a communication-performance result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .plotting import (
    _group_by_timestamp,
    _localization_series,
    _number,
    _read_csv,
    _reward_parameters,
    _reward_series,
    _timestamp,
)


@dataclass(frozen=True)
class SingleUAVDiagnosticsConfig:
    """Inputs and output location for one single-UAV episode diagnostic."""

    episode_path: Path
    output_dir: Path
    force: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read episode scenario JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"episode scenario must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"diagnostic output is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"diagnostic output directory is not empty: {path}; use --force or choose another output"
        )
    path.mkdir(parents=True, exist_ok=True)


def _point_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    axes: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    points: list[np.ndarray] = []
    for row in rows:
        point = np.asarray([_number(row.get(axis)) for axis in axes], dtype=float)
        timestamp = _timestamp(row)
        if np.isfinite(point).all() and np.isfinite(timestamp):
            timestamps.append(timestamp)
            points.append(point)
    if not points:
        return np.asarray([], dtype=float), np.empty((0, 3), dtype=float)
    order = np.argsort(np.asarray(timestamps, dtype=float))
    return (
        np.asarray(timestamps, dtype=float)[order],
        np.asarray(points, dtype=float)[order],
    )


def _first_obstacle_snapshot(
    obstacles: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    grouped = _group_by_timestamp(obstacles)
    if not grouped:
        return []
    first_timestamp = next(iter(grouped))
    return grouped[first_timestamp]


def _draw_xy_obstacles(axis: Any, obstacles: Sequence[Mapping[str, Any]]) -> None:
    from matplotlib.patches import Circle, Rectangle

    for obstacle in _first_obstacle_snapshot(obstacles):
        geometry = str(obstacle.get("geometry_type", "")).lower()
        if geometry == "box":
            x_min = _number(obstacle.get("x_min"))
            x_max = _number(obstacle.get("x_max"))
            y_min = _number(obstacle.get("y_min"))
            y_max = _number(obstacle.get("y_max"))
            if np.isfinite([x_min, x_max, y_min, y_max]).all():
                axis.add_patch(
                    Rectangle(
                        (x_min, y_min),
                        max(0.0, x_max - x_min),
                        max(0.0, y_max - y_min),
                        facecolor="#9ca3af",
                        edgecolor="#4b5563",
                        alpha=0.25,
                        label="Obstacle",
                    )
                )
        elif geometry == "sphere":
            x = _number(obstacle.get("x"))
            y = _number(obstacle.get("y"))
            radius = max(0.0, _number(obstacle.get("radius"), 0.0))
            if np.isfinite([x, y, radius]).all():
                axis.add_patch(
                    Circle(
                        (x, y),
                        radius,
                        facecolor="#f59e0b",
                        edgecolor="#b45309",
                        alpha=0.25,
                        label="Dynamic obstacle",
                    )
                )


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def _plot_localization(
    localization: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
    split: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    timestamps = np.asarray([float(row["timestamp"]) for row in localization], dtype=float)
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    panels = (
        ("position_rmse_m", "Position RMSE (m)", "#2563eb"),
        ("velocity_rmse_mps", "Velocity RMSE (m/s)", "#16a34a"),
        ("covariance_position_trace", "Position covariance trace (m²)", "#dc2626"),
    )
    for axis, (metric, ylabel, color) in zip(axes, panels):
        values = np.asarray([_number(row.get(metric)) for row in localization], dtype=float)
        axis.plot(timestamps, values, marker="o", linewidth=1.8, markersize=4, color=color)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle(f"Single-UAV localization diagnostics | {episode_id} | {split}")
    _save_figure(figure, output_path)
    plt.close(figure)


def _plot_goal_distance(
    estimates: Sequence[Mapping[str, Any]],
    goal: np.ndarray,
    *,
    trajectory_rows: Sequence[Mapping[str, Any]] | None = None,
    episode_id: str,
    split: str,
    output_path: Path,
    goal_threshold_m: float = 1.0,
    near_goal_radius_m: float = 5.0,
    speed_threshold_mps: float = 0.5,
) -> None:
    """Plot goal distance and the low-speed condition required for stable arrival."""

    import matplotlib.pyplot as plt

    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("single-UAV goal-distance plot requires a finite XYZ goal")
    grouped = _group_by_timestamp(estimates)
    velocity_keys = ("vx", "vy", "vz")
    has_actual_velocity = bool(
        trajectory_rows
        and any(all(key in row for key in velocity_keys) for row in trajectory_rows)
    )
    speed_rows = trajectory_rows if has_actual_velocity else estimates
    speed_grouped = _group_by_timestamp(speed_rows)
    speed_label = "Actual UAV speed (reward)" if has_actual_velocity else "EKF estimated speed"
    timestamps: list[float] = []
    distances: list[float] = []
    speeds: list[float] = []
    for timestamp, rows in grouped.items():
        values: list[float] = []
        speed_values: list[float] = []
        for row in rows:
            estimate = np.asarray(
                [_number(row.get(axis)) for axis in ("x", "y", "z")],
                dtype=float,
            )
            distance = float(np.linalg.norm(estimate - goal))
            if np.isfinite(distance):
                values.append(distance)
        speed_rows_at_timestamp = speed_grouped.get(timestamp, rows)
        velocity_values = [
            np.asarray(
                [_number(speed_row.get(axis)) for axis in velocity_keys],
                dtype=float,
            )
            for speed_row in speed_rows_at_timestamp
        ]
        speed_values = [
            float(np.linalg.norm(value))
            for value in velocity_values
            if np.isfinite(value).all()
        ]
        if values:
            timestamps.append(float(timestamp))
            distances.append(float(np.mean(values)))
            speeds.append(float(np.mean(speed_values)) if speed_values else float("nan"))
    if not distances:
        raise ValueError("estimates.csv contains no finite goal-distance samples")

    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.plot(
        timestamps,
        distances,
        color="#0891b2",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="EKF estimated distance",
    )
    axis.axhline(
        float(goal_threshold_m),
        color="#dc2626",
        linewidth=1.4,
        linestyle="--",
        label=f"Stable goal radius ({goal_threshold_m:g} m)",
    )
    axis.axhline(
        float(near_goal_radius_m),
        color="#f59e0b",
        linewidth=1.3,
        linestyle=":",
        label=f"Near-goal radius ({near_goal_radius_m:g} m)",
    )
    axis.set_title("Estimated distance to goal")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Distance to goal (m)")
    axis.grid(alpha=0.25)
    speed_axis = axis.twinx()
    speed_axis.plot(
        timestamps,
        speeds,
        color="#16a34a",
        linewidth=1.5,
        marker=".",
        markersize=5,
        alpha=0.9,
        label=speed_label,
    )
    speed_axis.axhline(
        float(speed_threshold_mps),
        color="#166534",
        linewidth=1.3,
        linestyle="--",
        label=f"Stable speed limit ({speed_threshold_mps:g} m/s)",
    )
    speed_axis.set_ylabel("Estimated speed (m/s)")
    handles, labels = axis.get_legend_handles_labels()
    speed_handles, speed_labels = speed_axis.get_legend_handles_labels()
    axis.legend(handles + speed_handles, labels + speed_labels, loc="best")
    figure.suptitle(f"Single-UAV goal-progress diagnostics | {episode_id} | {split}")
    _save_figure(figure, output_path)
    plt.close(figure)


def _plot_trajectory(
    trajectory_rows: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    obstacles: Sequence[Mapping[str, Any]],
    goal: np.ndarray,
    *,
    uav_id: str,
    episode_id: str,
    split: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _, actual = _point_series(trajectory_rows, axes=("x", "y", "z"))
    truth_rows = [row for row in ground_truth if str(row.get("uav_id", "")) == uav_id]
    _, truth = _point_series(truth_rows, axes=("x", "y", "z"))
    if actual.shape[0] == 0:
        raise ValueError("trajectory.csv contains no finite XYZ samples")

    figure = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    xy_axis = figure.add_subplot(1, 2, 1)
    xy_axis.plot(actual[:, 0], actual[:, 1], color="#7c3aed", linewidth=2.0, label="UAV state")
    if truth.shape[0]:
        xy_axis.plot(
            truth[:, 0],
            truth[:, 1],
            color="#111827",
            linewidth=1.5,
            linestyle="--",
            label="Ground truth",
        )
    _draw_xy_obstacles(xy_axis, obstacles)
    xy_axis.scatter(actual[0, 0], actual[0, 1], color="#16a34a", marker="o", s=45, label="Start")
    if goal.shape == (3,) and np.isfinite(goal).all():
        xy_axis.scatter(goal[0], goal[1], color="#dc2626", marker="*", s=90, label="Goal")
    xy_axis.set_title("XY flight trajectory")
    xy_axis.set_xlabel("x (m)")
    xy_axis.set_ylabel("y (m)")
    xy_axis.set_aspect("equal", adjustable="datalim")
    xy_axis.grid(alpha=0.25)
    handles, labels = xy_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    xy_axis.legend(unique.values(), unique.keys())

    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")
    axis_3d.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#7c3aed", linewidth=2.0, label="UAV state")
    if truth.shape[0]:
        axis_3d.plot(
            truth[:, 0],
            truth[:, 1],
            truth[:, 2],
            color="#111827",
            linewidth=1.5,
            linestyle="--",
            label="Ground truth",
        )
    axis_3d.scatter(actual[0, 0], actual[0, 1], actual[0, 2], color="#16a34a", marker="o", s=45, label="Start")
    if goal.shape == (3,) and np.isfinite(goal).all():
        axis_3d.scatter(goal[0], goal[1], goal[2], color="#dc2626", marker="*", s=90, label="Goal")
    axis_3d.set_title("3D flight trajectory")
    axis_3d.set_xlabel("x (m)")
    axis_3d.set_ylabel("y (m)")
    axis_3d.set_zlabel("z (m)")
    axis_3d.legend()
    figure.suptitle(f"Single-UAV flight trajectory | {episode_id} | {split}")
    _save_figure(figure, output_path)
    plt.close(figure)


def _action_response_rows(
    action_rows: Sequence[Mapping[str, Any]],
    reward_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    """Return finite action-to-motion response samples for one UAV.

    The action log is authoritative for the command/execution response.  A
    reward timestamp is joined only to overlay the distance-dependent target
    speed; no trajectory smoothing or interpolation is performed.
    """

    target_by_timestamp = {
        _timestamp(row): _number(row.get("single_uav_target_speed_mps"))
        for row in reward_rows
        if np.isfinite(_timestamp(row))
    }
    rows: list[dict[str, float]] = []
    for row in sorted(action_rows, key=_timestamp):
        timestamp = _timestamp(row)
        values = {
            "timestamp_s": timestamp,
            "pre_speed_mps": _number(row.get("pre_speed_mps")),
            "post_speed_mps": _number(row.get("post_speed_mps")),
            "speed_reduction_mps": _number(row.get("speed_reduction_mps")),
            "applied_longitudinal_delta_v_mps": _number(
                row.get("applied_longitudinal_delta_v_mps")
            ),
            "braking_delta_v_mps": _number(row.get("braking_delta_v_mps")),
            "braking_action": _number(row.get("braking_action")),
            "brake_effective": _number(row.get("brake_effective")),
            "brake_blocked": _number(row.get("brake_blocked")),
            "speed_limit_limited": _number(row.get("speed_limit_limited"), 0.0),
            "target_speed_mps": target_by_timestamp.get(timestamp, float("nan")),
        }
        required = (
            "timestamp_s",
            "pre_speed_mps",
            "post_speed_mps",
            "applied_longitudinal_delta_v_mps",
            "braking_delta_v_mps",
            "braking_action",
            "brake_effective",
        )
        if all(np.isfinite(values[key]) for key in required):
            rows.append(values)
    return rows


def _plot_action_response(
    action_rows: Sequence[Mapping[str, Any]],
    reward_rows: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
    split: str,
    output_path: Path,
    target_speed_label: str = "Target speed",
) -> bool:
    """Plot commanded braking, executed braking, and the speed response."""

    response = _action_response_rows(action_rows, reward_rows)
    if not response:
        return False

    import matplotlib.pyplot as plt

    timestamps = np.asarray([row["timestamp_s"] for row in response], dtype=float)
    pre_speed = np.asarray([row["pre_speed_mps"] for row in response], dtype=float)
    post_speed = np.asarray([row["post_speed_mps"] for row in response], dtype=float)
    target_speed = np.asarray([row["target_speed_mps"] for row in response], dtype=float)
    applied_longitudinal = np.asarray(
        [row["applied_longitudinal_delta_v_mps"] for row in response], dtype=float
    )
    braking_delta = np.asarray(
        [row["braking_delta_v_mps"] for row in response], dtype=float
    )
    braking_action = np.asarray([row["braking_action"] for row in response], dtype=bool)
    brake_effective = np.asarray([row["brake_effective"] for row in response], dtype=bool)
    brake_blocked = np.asarray([row["brake_blocked"] for row in response], dtype=bool)
    speed_limit_limited = np.asarray(
        [row["speed_limit_limited"] for row in response], dtype=bool
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        timestamps,
        pre_speed,
        color="#64748b",
        linewidth=1.4,
        label="Pre-action speed",
    )
    axes[0].plot(
        timestamps,
        post_speed,
        color="#7c3aed",
        linewidth=1.8,
        label="Post-action speed",
    )
    valid_target = np.isfinite(target_speed)
    if valid_target.any():
        axes[0].plot(
            timestamps[valid_target],
            target_speed[valid_target],
            color="#111827",
            linewidth=1.5,
            linestyle="--",
            label=target_speed_label,
        )
    if braking_action.any():
        axes[0].scatter(
            timestamps[braking_action],
            post_speed[braking_action],
            color="#dc2626",
            s=24,
            zorder=4,
            label="Braking action",
        )
    if brake_effective.any():
        axes[0].scatter(
            timestamps[brake_effective],
            post_speed[brake_effective],
            facecolors="none",
            edgecolors="#16a34a",
            linewidths=1.2,
            s=54,
            zorder=5,
            label="Effective speed reduction",
        )
    if speed_limit_limited.any():
        axes[0].scatter(
            timestamps[speed_limit_limited],
            post_speed[speed_limit_limited],
            color="#f59e0b",
            marker="x",
            s=32,
            zorder=5,
            label="Speed limit active",
        )
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Speed response to the executed motion action")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(
        timestamps,
        applied_longitudinal,
        color="#2563eb",
        linewidth=1.4,
        label="Applied longitudinal Δv",
    )
    axes[1].plot(
        timestamps,
        -braking_delta,
        color="#dc2626",
        linewidth=1.6,
        linestyle="--",
        label="Braking component (negative)",
    )
    axes[1].axhline(0.0, color="#111827", linewidth=0.9)
    if brake_blocked.any():
        axes[1].scatter(
            timestamps[brake_blocked],
            applied_longitudinal[brake_blocked],
            color="#f59e0b",
            marker="x",
            s=32,
            label="Braking command partly blocked",
        )
    if speed_limit_limited.any():
        axes[1].scatter(
            timestamps[speed_limit_limited],
            applied_longitudinal[speed_limit_limited],
            color="#f59e0b",
            marker="x",
            s=32,
            label="Speed limit active",
        )
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Δv along pre-action velocity (m/s)")
    axes[1].set_title("Braking direction and action execution")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    action_rate = float(np.mean(braking_action))
    effective_given_action = (
        f"{float(np.mean(brake_effective[braking_action])):.1%}"
        if braking_action.any()
        else "n/a (no braking actions)"
    )
    figure.suptitle(
        f"Single-UAV action-to-motion braking diagnostics | {episode_id} | {split}\n"
        f"braking action rate={action_rate:.1%}; effective among braking={effective_given_action}"
    )
    _save_figure(figure, output_path)
    plt.close(figure)
    return True


def _plot_reward(
    rewards: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
    split: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    timestamps = np.asarray([float(row["timestamp"]) for row in rewards], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(
        timestamps,
        [_number(row.get("mean_reward")) for row in rewards],
        color="#7c3aed",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Mean reward",
    )
    if any(np.isfinite(_number(row.get("learning_reward"))) for row in rewards):
        axes[0].plot(
            timestamps,
            [_number(row.get("learning_reward"), 0.0) for row in rewards],
            color="#0891b2",
            linewidth=1.6,
            linestyle="--",
            marker=".",
            markersize=5,
            label="Learning reward",
        )
    axes[0].plot(
        timestamps,
        [_number(row.get("joint_objective")) for row in rewards],
        color="#111827",
        linewidth=1.5,
        linestyle="--",
        label="Joint objective",
    )
    axes[0].set_ylabel("Reward / objective")
    axes[0].set_title("Total reward and objective")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    components = (
        ("j_loc", "J_loc", "#2563eb"),
        ("j_com", "J_com", "#16a34a"),
        ("collision_penalty", "Collision penalty", "#dc2626"),
        ("energy_penalty", "Energy penalty", "#f59e0b"),
        ("localization_progress_score", "Localization progress score", "#7c3aed"),
        ("goal_progress_score", "Goal progress score", "#0891b2"),
        ("goal_progress_reward", "Goal progress reward", "#0f766e"),
        ("goal_bonus", "Goal arrival bonus", "#16a34a"),
        ("near_goal_speed_cost", "Near-goal speed cost", "#be123c"),
        ("localization_cost", "Localization cost", "#dc2626"),
    )
    for metric, label, color in components:
        axes[1].plot(
            timestamps,
            [_number(row.get(metric)) for row in rewards],
            linewidth=1.5,
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Component value")
    axes[1].set_title("Reward components")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)
    figure.suptitle(f"Single-UAV reward diagnostics | {episode_id} | {split}")
    _save_figure(figure, output_path)
    plt.close(figure)


def plot_single_uav_diagnostics(config: SingleUAVDiagnosticsConfig) -> dict[str, Any]:
    """Create diagnostic plots for one recorded single-UAV episode."""

    episode_path = Path(config.episode_path)
    if not episode_path.is_dir():
        raise FileNotFoundError(f"episode directory does not exist: {episode_path}")
    scenario_path = episode_path / "scenario.json"
    scenario = _read_json(scenario_path)
    try:
        uav_count = int(scenario.get("uav_count", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid uav_count in {scenario_path}") from exc
    if uav_count != 1:
        raise ValueError(
            f"single-UAV diagnostics require uav_count=1, got {uav_count}: {episode_path}"
        )

    required_files = (
        "trajectory.csv",
        "ground_truth.csv",
        "estimates.csv",
        "rewards.csv",
        "obstacles.csv",
    )
    missing = [name for name in required_files if not (episode_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"episode is missing required diagnostic files: {', '.join(missing)}"
        )

    trajectory_rows = _read_csv(episode_path / "trajectory.csv")
    ground_truth = _read_csv(episode_path / "ground_truth.csv")
    estimates = _read_csv(episode_path / "estimates.csv")
    raw_rewards = _read_csv(episode_path / "rewards.csv")
    obstacles = _read_csv(episode_path / "obstacles.csv")
    action_rows = (
        _read_csv(episode_path / "actions.csv")
        if (episode_path / "actions.csv").is_file()
        else []
    )
    if not trajectory_rows or not raw_rewards:
        raise ValueError(f"episode has no trajectory or reward samples: {episode_path}")

    uav_ids = sorted({str(row.get("uav_id", "")) for row in trajectory_rows if row.get("uav_id")})
    if len(uav_ids) != 1:
        raise ValueError(
            f"single-UAV episode must contain exactly one trajectory uav_id, got {uav_ids}"
        )
    uav_id = uav_ids[0]
    trajectory_rows = sorted(trajectory_rows, key=_timestamp)
    localization, _, _ = _localization_series(ground_truth, estimates)
    if not localization:
        raise ValueError(f"cannot align ground-truth and estimate samples: {episode_path}")
    reward_series = _reward_series(raw_rewards, _reward_parameters(scenario))
    if not reward_series:
        raise ValueError(f"cannot derive reward time series: {episode_path}")

    goals = scenario.get("goals", {})
    raw_goal = goals.get(uav_id, []) if isinstance(goals, Mapping) else []
    goal = np.asarray(raw_goal, dtype=float).reshape(-1) if raw_goal else np.asarray([], dtype=float)
    if goal.size != 3:
        goal = np.asarray([], dtype=float)

    reward_config = scenario.get("reward_config", {})
    reward_config = reward_config if isinstance(reward_config, Mapping) else {}

    output_dir = Path(config.output_dir)
    _prepare_output(output_dir, config.force)
    episode_id = str(scenario.get("episode_id", episode_path.name))
    split = str(scenario.get("split", trajectory_rows[0].get("split", "unknown")))
    _plot_localization(
        localization,
        episode_id=episode_id,
        split=split,
        output_path=output_dir / "localization_error_timeseries.png",
    )
    _plot_goal_distance(
        estimates,
        goal,
        episode_id=episode_id,
        split=split,
        output_path=output_dir / "goal_distance_timeseries.png",
        trajectory_rows=trajectory_rows,
        goal_threshold_m=_number(
            reward_config.get("single_uav_goal_threshold_m"), 1.0
        ),
        near_goal_radius_m=_number(
            reward_config.get("single_uav_near_goal_radius_m"), 5.0
        ),
        speed_threshold_mps=_number(
            reward_config.get("single_uav_goal_speed_threshold_mps"), 0.5
        ),
    )
    _plot_trajectory(
        trajectory_rows,
        ground_truth,
        obstacles,
        goal,
        uav_id=uav_id,
        episode_id=episode_id,
        split=split,
        output_path=output_dir / "flight_trajectory_xy_xyz.png",
    )
    _plot_reward(
        reward_series,
        episode_id=episode_id,
        split=split,
        output_path=output_dir / "reward_timeseries.png",
    )
    action_plot_path = output_dir / "braking_action_response.png"
    action_plot_available = _plot_action_response(
        action_rows,
        raw_rewards,
        episode_id=episode_id,
        split=split,
        output_path=action_plot_path,
        target_speed_label=(
            "Stopping-distance target speed"
            if str(reward_config.get("single_uav_speed_control_mode", ""))
            == "braking_distance"
            else "Distance target speed"
        ),
    )

    scenario_generation = scenario.get("scenario_generation", {})
    geometry_seed = (
        scenario_generation.get("seed")
        if isinstance(scenario_generation, Mapping)
        else None
    )
    timestamps = [_timestamp(row) for row in trajectory_rows]
    manifest = {
        "figure_type": "single_uav_episode_diagnostics",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "episode": {
            "path": str(episode_path.resolve()),
            "episode_id": episode_id,
            "split": split,
            "uav_count": uav_count,
            "uav_id": uav_id,
            "episode_seed": scenario.get("seed"),
            "scenario_geometry_seed": geometry_seed,
            "sample_count": len(trajectory_rows),
            "localization_points": len(localization),
            "reward_points": len(reward_series),
            "time_start_s": float(min(timestamps)) if timestamps else None,
            "time_end_s": float(max(timestamps)) if timestamps else None,
        },
        "motion_action_contract": scenario.get("motion_action_contract", {}),
        "source_files": {
            name: str((episode_path / name).resolve())
            for name in ("scenario.json", *required_files, "actions.csv")
            if (episode_path / name).is_file()
        },
        "x_axes": {
            "localization_error_timeseries": "simulation_time_s",
            "goal_distance_timeseries": "simulation_time_s",
            "flight_trajectory_xy_xyz": "trajectory_samples_in_simulation_time_s",
            "reward_timeseries": "simulation_time_s",
            "braking_action_response": "simulation_time_s",
        },
        "stable_arrival_contract": {
            "goal_threshold_m": _number(
                reward_config.get("single_uav_goal_threshold_m"), 1.0
            ),
            "speed_threshold_mps": _number(
                reward_config.get("single_uav_goal_speed_threshold_mps"), 0.5
            ),
            "near_goal_radius_m": _number(
                reward_config.get("single_uav_near_goal_radius_m"), 5.0
            ),
            "speed_source": (
                "trajectory.csv actual UAV velocity used by the reward"
                if any(
                    all(key in row for key in ("vx", "vy", "vz"))
                    for row in trajectory_rows
                )
                else "estimates.csv EKF velocity fallback"
            ),
            "stable_goal_definition": (
                "estimated goal distance <= goal threshold and actual speed <= speed threshold"
            ),
        },
        "transformations": {
            "localization": "position and velocity RMSE are computed from aligned ground_truth.csv and estimates.csv; covariance trace is read from the estimate covariance matrix",
            "goal_distance": "distance is computed from EKF-estimated XYZ position to the configured goal; estimated speed and the configured near-goal/stable-arrival thresholds are overlaid; simulator truth is not used",
            "trajectory": "trajectory.csv state samples are plotted without smoothing or resampling; ground truth.csv is overlaid when aligned",
            "reward": "reward components are averaged by timestamp across UAV rows; this episode contains one UAV",
            "braking_action_response": "actions.csv pre/post speed and applied delta_v diagnostics; braking is the negative projection of applied delta_v onto pre-action velocity; reward target speed is joined by exact timestamp",
        },
        "plots": {
            "localization_error_timeseries": str((output_dir / "localization_error_timeseries.png").resolve()),
            "goal_distance_timeseries": str((output_dir / "goal_distance_timeseries.png").resolve()),
            "flight_trajectory_xy_xyz": str((output_dir / "flight_trajectory_xy_xyz.png").resolve()),
            "reward_timeseries": str((output_dir / "reward_timeseries.png").resolve()),
            **(
                {"braking_action_response": str(action_plot_path.resolve())}
                if action_plot_available
                else {}
            ),
        },
        "claim_boundary": "single-episode diagnostic only; this does not establish convergence, generalization, or communication cooperation performance",
        "communication_scope": "single-UAV episodes have no inter-UAV communication links; packet loss, delay, and AoI should not be interpreted as learned UAV-to-UAV communication behavior",
    }
    manifest_path = output_dir / "diagnostics_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir.resolve()),
        "episode_id": episode_id,
        "plots": manifest["plots"],
        "manifest": str(manifest_path.resolve()),
    }


__all__ = ["SingleUAVDiagnosticsConfig", "plot_single_uav_diagnostics"]
