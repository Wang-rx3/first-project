"""Reproducible screening for non-saturated communication-localization scenes.

The screening stage deliberately does not train or evaluate MAPPO.  It uses a
truth-free fixed controller, a space-filling Latin-hypercube design, and
episode-level replication to identify channel regimes in which communication
is neither always available nor practically impossible.  Scenario selection
therefore cannot be biased by looking at the learned policy's performance.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from ..simulation.ekf import EKFConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.scenario import add_moving_los_screen, generate_scenario
from ..simulation.scenario_profile import DEFAULT_SCREENING_GEOMETRY
from ..simulation.sensor_model import SensorConfig
from ..simulation.topology import TopologyConfig
from ..synthetic.channel_model import ChannelConfig
from .baseline import independent_fixed_actions
from .formal import default_ekf_parameters


FACTOR_RANGES: dict[str, tuple[float, float]] = {
    "max_comm_range_m": (25.0, 65.0),
    "total_bandwidth_hz": (20_000.0, 2_000_000.0),
    "nlos_extra_loss_db": (16.0, 40.0),
    "message_arrivals_per_window": (1.0, 30.0),
}

FACTOR_SCALES: dict[str, str] = {
    "max_comm_range_m": "linear",
    "total_bandwidth_hz": "log",
    "nlos_extra_loss_db": "linear",
    "message_arrivals_per_window": "linear",
}

PROFILE_NAMES = ("easy", "moderate", "hard")


@dataclass(frozen=True)
class ScenarioScreeningConfig:
    """Configuration for the environment-only screening experiment."""

    output_dir: Path
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    design_seed: int = 2026
    scenario_seeds: tuple[int, ...] = tuple(2026 + 1009 * index for index in range(10))
    design_points: int = 36
    uav_count: int = 3
    max_steps: int = 50
    cruise_speed_mps: float = 3.0
    gnss_mode: str = "denied"
    force: bool = False
    progress: bool = True
    vo_position_std_m: float = 0.03
    process_position_std_m: float = 0.08
    process_velocity_std_mps: float = 0.15
    nlos_variance_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("the first screening stage is restricted to three UAVs")
        if self.design_points < 2:
            raise ValueError("design_points must be at least 2")
        if not self.scenario_seeds:
            raise ValueError("at least one independent scenario seed is required")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.cruise_speed_mps <= 0.0 or not math.isfinite(self.cruise_speed_mps):
            raise ValueError("cruise_speed_mps must be finite and positive")
        for name in (
            "vo_position_std_m",
            "process_position_std_m",
            "process_velocity_std_mps",
            "nlos_variance_multiplier",
        ):
            value = float(getattr(self, name))
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")


def latin_hypercube_design(
    design_points: int,
    *,
    seed: int,
    factor_ranges: Mapping[str, tuple[float, float]] = FACTOR_RANGES,
) -> list[dict[str, Any]]:
    """Return a seeded maximin-like Latin-hypercube layout in real units.

    One independently shuffled stratum is used per factor.  Jitter within a
    stratum prevents a hidden grid while preserving exactly one point in each
    marginal stratum.  A separate run-order permutation is archived.
    """

    if design_points < 2:
        raise ValueError("design_points must be at least 2")
    rng = np.random.default_rng(int(seed))
    names = tuple(factor_ranges)
    unit = np.empty((design_points, len(names)), dtype=float)
    for column in range(len(names)):
        strata = (np.arange(design_points, dtype=float) + rng.random(design_points)) / design_points
        unit[:, column] = strata[rng.permutation(design_points)]
    run_order = rng.permutation(design_points)
    rows: list[dict[str, Any]] = []
    for order, point_index in enumerate(run_order, start=1):
        row: dict[str, Any] = {
            "run_order": order,
            "design_point_id": f"point_{point_index + 1:03d}",
            "design_point_index": int(point_index + 1),
        }
        for column, name in enumerate(names):
            lower, upper = factor_ranges[name]
            if FACTOR_SCALES.get(name, "linear") == "log":
                row[name] = float(
                    np.exp(np.log(lower) + unit[point_index, column] * (np.log(upper) - np.log(lower)))
                )
            else:
                row[name] = float(lower + unit[point_index, column] * (upper - lower))
        rows.append(row)
    return rows


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float(default)


def _quantile(values: Iterable[float], probability: float, default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if array.size else float(default)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_output(config: ScenarioScreeningConfig) -> tuple[Path, Path, Path]:
    output = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/screening") / output.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/screening") / output.name
    managed = (output, logs, plots)
    nonempty = [path for path in managed if path.exists() and any(path.iterdir())]
    if nonempty and not config.force:
        raise FileExistsError(
            "screening output is not empty: " + ", ".join(str(path) for path in nonempty)
            + "; use --force to replace these screening outputs"
        )
    if config.force:
        for path in managed:
            if path.exists():
                shutil.rmtree(path)
    for path in managed:
        path.mkdir(parents=True, exist_ok=True)
    return output, logs, plots


def _make_environment(
    config: ScenarioScreeningConfig,
    parameters: Mapping[str, float],
    seed: int,
) -> MultiUAVJointEnv:
    motion = MotionConfig(
        dt_s=0.2,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )
    sensor = SensorConfig(
        uwb_max_range_m=100.0,
        vo_position_std_m=config.vo_position_std_m,
        uwb_nlos_variance_multiplier=config.nlos_variance_multiplier,
    )
    ekf = EKFConfig(
        process_position_std_m=config.process_position_std_m,
        process_velocity_std_mps=config.process_velocity_std_mps,
        vo_position_std_m=config.vo_position_std_m,
    )
    channel = replace(
        ChannelConfig(),
        total_bandwidth_hz=float(parameters["total_bandwidth_hz"]),
        nlos_extra_loss_db=float(parameters["nlos_extra_loss_db"]),
    )
    topology = TopologyConfig(
        max_comm_range_m=float(parameters["max_comm_range_m"]),
        uwb_range_m=100.0,
        message_arrivals_per_window=float(parameters["message_arrivals_per_window"]),
    )
    return MultiUAVJointEnv(
        SimulationConfig(
            uav_count=config.uav_count,
            max_steps=config.max_steps,
            seed=int(seed),
            motion=motion,
            sensor=sensor,
            channel=channel,
            topology=topology,
            ekf=ekf,
            reward=RewardConfig(),
            gnss_mode=config.gnss_mode,
        )
    )


def _add_screening_blocker(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible wrapper for the shared screening geometry."""

    return add_moving_los_screen(dict(scenario))


def screening_probe_actions(
    environment: MultiUAVJointEnv,
    observations: Mapping[str, Mapping[str, Any]],
    *,
    cruise_speed_mps: float,
) -> dict[str, dict[str, Any]]:
    """Fixed truth-free motion/bandwidth control with neutral UWB selection."""

    actions = independent_fixed_actions(
        observations,
        environment.config.motion,
        cruise_speed_mps=cruise_speed_mps,
    )
    for uav_id, observation in observations.items():
        candidates = sorted(str(source) for source in observation.get("candidate_measurement_sources", ()))
        actions[uav_id]["measurement_uav_id"] = candidates[0] if candidates else "none"
    return actions


def _episode(
    config: ScenarioScreeningConfig,
    design: Mapping[str, Any],
    scenario_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = _make_environment(config, design, scenario_seed)
    scenario = _add_screening_blocker(generate_scenario(
        int(scenario_seed),
        config.uav_count,
        bounds_min_m=environment.config.motion.bounds_min_m,
        bounds_max_m=environment.config.motion.bounds_max_m,
    ))
    observations, _info = environment.reset(seed=int(scenario_seed), scenario_config=scenario)
    previous_effective: set[tuple[str, str]] = set()
    step_rows: list[dict[str, Any]] = []
    edge_change_count = 0
    finite = True
    collision = 0
    for step in range(config.max_steps):
        actions = screening_probe_actions(
            environment,
            observations,
            cruise_speed_mps=config.cruise_speed_mps,
        )
        observations, _rewards, terminated, truncated, info = environment.step(actions)
        snapshot = info["topology"]
        potential = set(snapshot.potential_edges)
        candidate = set(snapshot.candidate_edges)
        effective = set(snapshot.effective_edges)
        successful_transmissions = sum(
            int(snapshot.metrics[key].packet_success)
            for key in candidate
            if key in snapshot.metrics
        )
        if step > 0:
            edge_change_count += len(previous_effective.symmetric_difference(effective))
        previous_effective = effective
        candidate_metrics = [snapshot.metrics[key] for key in candidate if key in snapshot.metrics]
        arrivals = sum(
            str(event.get("status")) == "arrived"
            for event in info.get("message_buffer", {}).get("events", ())
        )
        diagnostics = list(info.get("ekf_diagnostics", {}).values())
        uwb_attempted = sum(int(row.get("uwb_update_attempted", 0)) for row in diagnostics)
        uwb_accepted = sum(int(row.get("uwb_update_accepted", 0)) for row in diagnostics)
        position_errors = [
            float(np.linalg.norm(environment.states[uav_id].position_m - environment.ekfs[uav_id].state[:3]))
            for uav_id in environment.uav_ids
        ]
        covariance_traces = [
            float(np.trace(environment.ekfs[uav_id].covariance[:3, :3]))
            for uav_id in environment.uav_ids
        ]
        all_arrays = [
            environment.ekfs[uav_id].state for uav_id in environment.uav_ids
        ] + [environment.ekfs[uav_id].covariance for uav_id in environment.uav_ids]
        finite = finite and all(np.isfinite(array).all() for array in all_arrays)
        connectivity = float(info.get("global_metrics", {}).get("connectivity", 1.0 / config.uav_count))
        step_rows.append(
            {
                "design_point_id": design["design_point_id"],
                "scenario_seed": int(scenario_seed),
                "step": step + 1,
                "time_s": float((step + 1) * environment.config.motion.dt_s),
                "potential_edges": len(potential),
                "candidate_edges": len(candidate),
                "effective_edges": len(effective),
                "successful_transmissions": successful_transmissions,
                "candidate_effective_ratio": successful_transmissions / len(candidate) if candidate else 0.0,
                "connectivity": connectivity,
                "packet_loss": _mean((metric.packet_loss for metric in candidate_metrics), default=1.0),
                "delay_s": _mean((metric.delay_s for metric in candidate_metrics), default=environment.config.channel.max_delay_s),
                "aoi_s": _mean((metric.aoi_s for metric in candidate_metrics), default=environment.config.channel.aoi_cap_s),
                "queue_length": _mean((metric.queue_length_packets for metric in candidate_metrics)),
                "nlos_fraction": _mean((1.0 - float(metric.los) for metric in candidate_metrics)),
                "message_arrivals": arrivals,
                "uwb_update_attempted": uwb_attempted,
                "uwb_update_accepted": uwb_accepted,
                "uwb_available_without_effective_message": len(set(snapshot.uwb_edges) - effective),
                "position_error_m": _mean(position_errors),
                "position_covariance_trace_m2": _mean(covariance_traces),
                "collision": int(info.get("collision", 0)),
            }
        )
        collision = max(collision, int(info.get("collision", 0)))
        if terminated or truncated:
            break

    steps = max(1, len(step_rows))
    candidate_opportunities = sum(int(row["candidate_edges"]) for row in step_rows)
    arrivals = sum(int(row["message_arrivals"]) for row in step_rows)
    attempts = sum(int(row["uwb_update_attempted"]) for row in step_rows)
    accepted = sum(int(row["uwb_update_accepted"]) for row in step_rows)
    summary = {
        "design_point_id": design["design_point_id"],
        "run_order": int(design["run_order"]),
        "scenario_seed": int(scenario_seed),
        "episode_steps": len(step_rows),
        "full_connectivity_fraction": _mean(float(row["connectivity"] >= 1.0 - 1e-9) for row in step_rows),
        "partial_connectivity_fraction": _mean(
            float(1.0 / config.uav_count + 1e-9 < row["connectivity"] < 1.0 - 1e-9)
            for row in step_rows
        ),
        "disconnected_fraction": _mean(
            float(row["connectivity"] <= 1.0 / config.uav_count + 1e-9) for row in step_rows
        ),
        "mean_connectivity": _mean(row["connectivity"] for row in step_rows),
        "candidate_effective_ratio": (
            sum(int(row["successful_transmissions"]) for row in step_rows) / candidate_opportunities
            if candidate_opportunities else 0.0
        ),
        "mean_packet_loss": _mean(row["packet_loss"] for row in step_rows),
        "mean_delay_s": _mean(row["delay_s"] for row in step_rows),
        "mean_aoi_s": _mean(row["aoi_s"] for row in step_rows),
        "aoi_p95_s": _quantile((row["aoi_s"] for row in step_rows), 0.95),
        "mean_queue_length": _mean(row["queue_length"] for row in step_rows),
        "nlos_fraction": _mean(row["nlos_fraction"] for row in step_rows),
        "message_arrival_rate": arrivals / candidate_opportunities if candidate_opportunities else 0.0,
        "uwb_update_acceptance_rate": accepted / attempts if attempts else 0.0,
        "neighbor_update_window_rate": accepted / (steps * config.uav_count),
        "uwb_update_attempted": attempts,
        "uwb_update_accepted": accepted,
        "uwb_available_without_effective_message": sum(
            int(row["uwb_available_without_effective_message"]) for row in step_rows
        ),
        "effective_edge_change_count": int(edge_change_count),
        "position_rmse_m": float(np.sqrt(_mean(float(row["position_error_m"]) ** 2 for row in step_rows))),
        "mean_position_covariance_trace_m2": _mean(
            row["position_covariance_trace_m2"] for row in step_rows
        ),
        "collision_episode": int(collision),
        "finite_state_fraction": 1.0 if finite else 0.0,
    }
    summary.update({name: float(design[name]) for name in FACTOR_RANGES})
    return summary, step_rows


def classify_parameter_summary(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Classify one replicated parameter point using predeclared gates."""

    full = float(row["full_connectivity_fraction"])
    partial = float(row["partial_connectivity_fraction"])
    disconnected = float(row["disconnected_fraction"])
    conversion = float(row["candidate_effective_ratio"])
    loss = float(row["mean_packet_loss"])
    arrival = float(row["message_arrival_rate"])
    update = float(row["neighbor_update_window_rate"])
    collision = float(row["collision_episode_fraction"])
    finite = float(row["finite_state_fraction"])
    reasons: list[str] = []
    gates = {
        "full_connectivity_fraction": 0.40 <= full <= 0.90,
        "partial_connectivity_fraction": 0.10 <= partial <= 0.50,
        "disconnected_fraction": disconnected < 0.10,
        "candidate_effective_ratio": 0.60 <= conversion <= 0.95,
        "mean_packet_loss": 0.03 <= loss <= 0.20,
        "message_arrival_rate": 0.40 <= arrival <= 0.90,
        "neighbor_update_window_rate": 0.20 <= update <= 0.80,
        "collision_episode_fraction": collision < 0.05,
        "finite_state_fraction": finite >= 1.0 - 1e-12,
    }
    for name, passed in gates.items():
        if not passed:
            reasons.append(name)
    if all(gates.values()):
        return "eligible", reasons
    if full > 0.90 and conversion > 0.95 and loss < 0.03:
        return "simple_saturated", reasons
    if disconnected >= 0.10 or conversion < 0.60 or loss > 0.20:
        return "difficult_saturated", reasons
    return "transition_other", reasons


def _parameter_summaries(
    designs: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for design in designs:
        selected = [row for row in episodes if row["design_point_id"] == design["design_point_id"]]
        aggregate: dict[str, Any] = dict(design)
        aggregate["episode_replicates"] = len(selected)
        metrics = (
            "full_connectivity_fraction",
            "partial_connectivity_fraction",
            "disconnected_fraction",
            "mean_connectivity",
            "candidate_effective_ratio",
            "mean_packet_loss",
            "mean_delay_s",
            "mean_aoi_s",
            "aoi_p95_s",
            "mean_queue_length",
            "nlos_fraction",
            "message_arrival_rate",
            "uwb_update_acceptance_rate",
            "neighbor_update_window_rate",
            "uwb_available_without_effective_message",
            "effective_edge_change_count",
            "position_rmse_m",
            "mean_position_covariance_trace_m2",
            "collision_episode",
            "finite_state_fraction",
        )
        for metric in metrics:
            output_name = "collision_episode_fraction" if metric == "collision_episode" else metric
            aggregate[output_name] = _mean(float(row[metric]) for row in selected)
        # A policy-independent severity score is used only to order already
        # eligible environments.  It never uses MAPPO reward or performance.
        aggregate["severity_score"] = float(
            0.30 * (1.0 - float(aggregate["mean_connectivity"]))
            + 0.25 * float(aggregate["mean_packet_loss"])
            + 0.20 * (1.0 - float(aggregate["candidate_effective_ratio"]))
            + 0.15 * min(1.0, float(aggregate["mean_aoi_s"]) / 0.2)
            + 0.10 * min(1.0, float(aggregate["mean_queue_length"]) / 8.0)
        )
        classification, failures = classify_parameter_summary(aggregate)
        aggregate["classification"] = classification
        aggregate["failed_gates"] = ";".join(failures)
        summaries.append(aggregate)
    return summaries


def _profile_payload(
    name: str,
    row: Mapping[str, Any] | None,
    *,
    config: ScenarioScreeningConfig,
) -> dict[str, Any]:
    if row is None:
        return {
            "profile": name,
            "uav_count": int(config.uav_count),
            "selection_status": "not_selected_no_eligible_point",
            "allowed_for_training": False,
            "reason": "No parameter point passed every predeclared environment gate.",
        }
    parameters = {factor: float(row[factor]) for factor in FACTOR_RANGES}
    return {
        "profile": name,
        "uav_count": int(config.uav_count),
        "selection_status": "eligible_frozen_candidate",
        "allowed_for_training": True,
        "design_point_id": row["design_point_id"],
        "severity_score": float(row["severity_score"]),
        "parameters": parameters,
        "screening_design_seed": int(config.design_seed),
        "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
        "selection_rule": "low/median/high severity among points passing all predeclared gates",
        "mappo_performance_used_for_selection": False,
        "screening_geometry": dict(DEFAULT_SCREENING_GEOMETRY),
    }


def _select_profiles(
    summaries: Sequence[Mapping[str, Any]],
    config: ScenarioScreeningConfig,
) -> dict[str, dict[str, Any]]:
    eligible = sorted(
        (row for row in summaries if row.get("classification") == "eligible"),
        key=lambda row: float(row["severity_score"]),
    )
    if not eligible:
        return {name: _profile_payload(name, None, config=config) for name in PROFILE_NAMES}
    if len(eligible) == 1:
        return {
            "easy": _profile_payload("easy", None, config=config),
            "moderate": _profile_payload("moderate", eligible[0], config=config),
            "hard": _profile_payload("hard", None, config=config),
        }
    if len(eligible) == 2:
        return {
            "easy": _profile_payload("easy", eligible[0], config=config),
            "moderate": _profile_payload("moderate", eligible[1], config=config),
            "hard": _profile_payload("hard", None, config=config),
        }
    indices = (0, len(eligible) // 2, len(eligible) - 1)
    return {
        name: _profile_payload(name, eligible[index], config=config)
        for name, index in zip(PROFILE_NAMES, indices)
    }


def _plot_screening(
    summaries: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    plots_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "eligible": "#2ca02c",
        "simple_saturated": "#1f77b4",
        "difficult_saturated": "#d62728",
        "transition_other": "#7f7f7f",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = (
        ("full_connectivity_fraction", "Fully connected window fraction"),
        ("candidate_effective_ratio", "Candidate-to-effective ratio"),
        ("mean_packet_loss", "Mean packet loss"),
        ("message_arrival_rate", "Message arrivals / candidate opportunities"),
    )
    x = np.arange(len(summaries))
    point_colors = [colors[str(row["classification"])] for row in summaries]
    for axis, (metric, title) in zip(axes.flat, pairs):
        axis.scatter(x, [float(row[metric]) for row in summaries], c=point_colors, s=24)
        axis.set_title(title)
        axis.set_xlabel("Randomized design-point order")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    saturation = plots_dir / "saturation_diagnostics.png"
    figure.savefig(saturation, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(
        [float(row["message_arrival_rate"]) for row in summaries],
        [float(row["neighbor_update_window_rate"]) for row in summaries],
        c=point_colors,
    )
    axes[0].set_xlabel("Message arrival rate")
    axes[0].set_ylabel("Neighbor-update windows / UAV windows")
    axes[0].set_title("Communication-to-EKF coupling")
    axes[1].scatter(
        [float(row["mean_packet_loss"]) for row in summaries],
        [float(row["position_rmse_m"]) for row in summaries],
        c=point_colors,
    )
    axes[1].set_xlabel("Mean packet loss")
    axes[1].set_ylabel("Position RMSE (m)")
    axes[1].set_title("Communication-localization response")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    coupling = plots_dir / "communication_coupling.png"
    figure.savefig(coupling, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for classification, color in colors.items():
        selected = [row for row in summaries if row["classification"] == classification]
        if selected:
            axis.scatter(
                [float(row["mean_connectivity"]) for row in selected],
                [float(row["mean_packet_loss"]) for row in selected],
                c=color,
                label=classification.replace("_", " "),
            )
    axis.set_xlabel("Mean largest-component connectivity")
    axis.set_ylabel("Mean packet loss")
    axis.set_title("Predeclared feasible-region classification")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    feasible = plots_dir / "feasible_region.png"
    figure.savefig(feasible, dpi=180)
    plt.close(figure)
    return {
        "saturation_diagnostics": str(saturation),
        "communication_coupling": str(coupling),
        "feasible_region": str(feasible),
    }


def run_scenario_screening(config: ScenarioScreeningConfig) -> dict[str, Any]:
    """Run the environment-only screening and archive its auditable design."""

    output, logs, plots = _prepare_output(config)
    designs = latin_hypercube_design(config.design_points, seed=config.design_seed)
    _write_csv(logs / "design_matrix.csv", designs)
    schedule = [
        {**design, "scenario_seed": int(seed)}
        for design in designs
        for seed in config.scenario_seeds
    ]
    rng = np.random.default_rng(config.design_seed + 7919)
    schedule = [schedule[index] for index in rng.permutation(len(schedule))]
    for run_index, row in enumerate(schedule, start=1):
        row["screening_run_order"] = run_index
    _write_csv(logs / "run_schedule.csv", schedule)

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for index, scheduled in enumerate(schedule, start=1):
        if config.progress:
            print(
                f"[screen-scenarios] run {index}/{len(schedule)} "
                f"point={scheduled['design_point_id']} seed={scheduled['scenario_seed']}",
                flush=True,
            )
        summary, steps = _episode(config, scheduled, int(scheduled["scenario_seed"]))
        summary["screening_run_order"] = int(scheduled["screening_run_order"])
        episode_rows.append(summary)
        step_rows.extend(steps)
    summaries = _parameter_summaries(designs, episode_rows)
    profiles = _select_profiles(summaries, config)

    _write_csv(logs / "episode_metrics.csv", episode_rows)
    _write_csv(logs / "step_metrics.csv", step_rows)
    _write_csv(logs / "parameter_summary.csv", summaries)
    _write_csv(
        logs / "eligibility_report.csv",
        [
            {
                "design_point_id": row["design_point_id"],
                "classification": row["classification"],
                "failed_gates": row["failed_gates"],
                "severity_score": row["severity_score"],
            }
            for row in summaries
        ],
    )
    profile_dir = output / "selected_profiles"
    for name, payload in profiles.items():
        _write_json(profile_dir / f"{name}.json", payload)
    plot_paths = _plot_screening(summaries, step_rows, plots)
    counts = {
        label: sum(row["classification"] == label for row in summaries)
        for label in ("eligible", "simple_saturated", "difficult_saturated", "transition_other")
    }
    manifest = {
        "experiment": "communication_localization_environment_screening",
        "result_status": "environment_screening_only_not_mappo_evidence",
        "design": "seeded_randomized_latin_hypercube",
        "independent_replication_unit": "episode/scenario_seed",
        "time_steps_are_independent_replicates": False,
        "mappo_performance_used_for_selection": False,
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "logs_dir": str(logs),
            "plots_dir": str(plots),
        },
        "factor_ranges": FACTOR_RANGES,
        "factor_scales": FACTOR_SCALES,
        "screening_geometry": "moving_3d_los_screen; packet loss is never changed directly",
        "classification_counts": counts,
        "profiles": profiles,
        "files": {
            "design_matrix": str(logs / "design_matrix.csv"),
            "run_schedule": str(logs / "run_schedule.csv"),
            "episode_metrics": str(logs / "episode_metrics.csv"),
            "step_metrics": str(logs / "step_metrics.csv"),
            "parameter_summary": str(logs / "parameter_summary.csv"),
            "eligibility_report": str(logs / "eligibility_report.csv"),
            "plots": plot_paths,
        },
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


__all__ = [
    "FACTOR_RANGES",
    "FACTOR_SCALES",
    "ScenarioScreeningConfig",
    "classify_parameter_summary",
    "latin_hypercube_design",
    "run_scenario_screening",
    "screening_probe_actions",
]
