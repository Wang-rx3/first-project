"""Generate a deterministic, schema-compatible multi-UAV benchmark.

This module is intentionally dependency-free and remains as a flat CSV
compatibility generator. The action-driven multi-UAV environment used for the
main experiments lives in ``src.simulation`` and is exported by
``episode_generator.py``. This module exercises the canonical flat schema and
provides reproducible fixtures; it is not the publication-level dynamics
source.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..schema import LINK_COLUMNS, OBSTACLE_COLUMNS, TRAJECTORY_COLUMNS
from .channel_model import ChannelConfig, LinkMetrics, compute_link_metrics


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for the local benchmark scaffold."""

    output_dir: Path
    seed: int = 2026
    scenarios: int = 9
    steps: int = 120
    dt: float = 0.2
    area_size: float = 200.0
    max_comm_range: float = 75.0
    uav_counts: tuple[int, ...] = (3, 5, 7)
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def count_for(self, scenario_index: int) -> int:
        return self.uav_counts[scenario_index % len(self.uav_counts)]


def _write_csv(path: Path, columns: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_for(index: int, scenarios: int) -> str:
    train_end = max(1, math.floor(scenarios * 0.6))
    validation_end = max(train_end + 1, math.floor(scenarios * 0.8))
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def _inside_building(x: float, y: float, building: dict[str, float]) -> bool:
    return (
        building["x_min"] <= x <= building["x_max"]
        and building["y_min"] <= y <= building["y_max"]
    )


def _line_blocked(
    source: tuple[float, float],
    target: tuple[float, float],
    buildings: list[dict[str, float]],
) -> bool:
    for step in range(1, 20):
        ratio = step / 20.0
        x = source[0] + ratio * (target[0] - source[0])
        y = source[1] + ratio * (target[1] - source[1])
        if any(_inside_building(x, y, building) for building in buildings):
            return True
    return False


def _obstacles(scenario_index: int, area_size: float) -> list[dict[str, Any]]:
    center = area_size / 2.0
    shift = (scenario_index % 3 - 1) * 18.0
    buildings = [
        {
            "obstacle_id": "building_0",
            "obstacle_type": "building",
            "x_min": center - 34 + shift,
            "x_max": center - 8 + shift,
            "y_min": center - 20,
            "y_max": center + 24,
            "z": 25.0,
            "radius": 22.0,
        },
        {
            "obstacle_id": "building_1",
            "obstacle_type": "building",
            "x_min": center + 10 - shift,
            "x_max": center + 42 - shift,
            "y_min": center - 46,
            "y_max": center - 18,
            "z": 40.0,
            "radius": 20.0,
        },
    ]
    return buildings


def _ground_truth(
    scenario_index: int,
    uav_index: int,
    timestamp: float,
    area_size: float,
) -> dict[str, float]:
    center = area_size / 2.0
    radius = 18.0 + uav_index * 2.5 + (scenario_index % 3) * 2.0
    omega = 0.18 + (uav_index % 3) * 0.025
    phase = scenario_index * 0.31 + uav_index * 0.73
    angle = omega * timestamp + phase
    x = center + radius * math.cos(angle)
    y = center + radius * math.sin(angle)
    z = 12.0 + 1.5 * math.sin(angle * 0.7 + uav_index)
    vx = -radius * omega * math.sin(angle)
    vy = radius * omega * math.cos(angle)
    vz = 1.5 * 0.7 * omega * math.cos(angle * 0.7 + uav_index)
    ax = -radius * omega * omega * math.cos(angle)
    ay = -radius * omega * omega * math.sin(angle)
    az = -1.5 * (0.7 * omega) ** 2 * math.sin(angle * 0.7 + uav_index)
    return {
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "yaw": math.atan2(vy, vx),
    }


def _gnss_mode(scenario_index: int, timestamp: float) -> str:
    if scenario_index % 3 == 0 and 8.0 <= timestamp <= 16.0:
        return "outage"
    if scenario_index % 3 == 1:
        return "degraded"
    return "nominal"


def _obstacle_state(
    position: tuple[float, float],
    buildings: list[dict[str, float]],
    timestamp: float,
) -> str:
    states: list[str] = []
    if any(
        math.hypot(position[0] - (b["x_min"] + b["x_max"]) / 2, position[1] - (b["y_min"] + b["y_max"]) / 2)
        < b["radius"] + 12
        for b in buildings
    ):
        states.append("building_near")
    vehicle_x = 100.0 + 30.0 * math.cos(timestamp * 0.12)
    vehicle_y = 100.0 + 30.0 * math.sin(timestamp * 0.12)
    if math.hypot(position[0] - vehicle_x, position[1] - vehicle_y) < 18:
        states.append("dynamic_vehicle_near")
    return ";".join(states) if states else "clear"


def generate_benchmark(config: SyntheticConfig, allow_overwrite: bool = False) -> dict[str, Any]:
    """Generate trajectory, edge, obstacle and metadata files."""

    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}; use allow_overwrite=True or --force"
        )
    output.mkdir(parents=True, exist_ok=True)

    trajectory_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    obstacle_rows: list[dict[str, Any]] = []
    scenario_records: list[dict[str, Any]] = []
    total_uavs = 0

    for scenario_index in range(config.scenarios):
        scenario_id = f"scenario_{scenario_index:03d}"
        split = _split_for(scenario_index, config.scenarios)
        uav_count = config.count_for(scenario_index)
        total_uavs += uav_count
        buildings = _obstacles(scenario_index, config.area_size)
        scenario_rng = random.Random(config.seed + scenario_index * 1009)
        aoi_state: dict[tuple[int, int], float] = {}
        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "split": split,
                "uav_count": uav_count,
                "steps": config.steps,
                "dt": config.dt,
                "obstacle_types": ["building", "dynamic_vehicle"],
            }
        )

        for step in range(config.steps):
            timestamp = round(step * config.dt, 6)
            states: dict[int, dict[str, float]] = {
                uav_index: _ground_truth(
                    scenario_index, uav_index, timestamp, config.area_size
                )
                for uav_index in range(uav_count)
            }

            for building in buildings:
                obstacle_rows.append(
                    {
                        "timestamp": timestamp,
                        "scenario_id": scenario_id,
                        "split": split,
                        "obstacle_id": building["obstacle_id"],
                        "obstacle_type": building["obstacle_type"],
                        "x": round((building["x_min"] + building["x_max"]) / 2, 6),
                        "y": round((building["y_min"] + building["y_max"]) / 2, 6),
                        "z": building["z"],
                        "vx": 0.0,
                        "vy": 0.0,
                        "radius": building["radius"],
                        "active": 1,
                    }
                )
            obstacle_rows.append(
                {
                    "timestamp": timestamp,
                    "scenario_id": scenario_id,
                    "split": split,
                    "obstacle_id": "vehicle_0",
                    "obstacle_type": "dynamic_vehicle",
                    "x": 100.0 + 30.0 * math.cos(timestamp * 0.12),
                    "y": 100.0 + 30.0 * math.sin(timestamp * 0.12),
                    "z": 1.5,
                    "vx": -3.6 * math.sin(timestamp * 0.12),
                    "vy": 3.6 * math.cos(timestamp * 0.12),
                    "radius": 3.0,
                    "active": 1,
                }
            )

            directed_metrics: dict[tuple[int, int], LinkMetrics] = {}
            neighbors: dict[int, list[int]] = {uav_index: [] for uav_index in range(uav_count)}
            for src in range(uav_count):
                for dst in range(uav_count):
                    if src == dst:
                        continue
                    source = states[src]
                    target = states[dst]
                    distance = math.sqrt(
                        (source["x"] - target["x"]) ** 2
                        + (source["y"] - target["y"]) ** 2
                        + (source["z"] - target["z"]) ** 2
                    )
                    los = not _line_blocked(
                        (source["x"], source["y"]),
                        (target["x"], target["y"]),
                        buildings,
                    )
                    key = (src, dst)
                    metrics = compute_link_metrics(
                        distance_m=distance,
                        los=los,
                        bandwidth_ratio=1.0 / max(1, uav_count - 1),
                        queue_length_packets=0.0,
                        previous_aoi_s=aoi_state.get(key, 0.0),
                        dt_s=config.dt,
                        rng=scenario_rng,
                        config=config.channel,
                        max_comm_range_m=config.max_comm_range,
                    )
                    aoi_state[key] = metrics.aoi_s
                    directed_metrics[key] = metrics
                    if metrics.effective_edge:
                        neighbors[src].append(dst)
                    link_rows.append(
                        {
                            "timestamp": timestamp,
                            "scenario_id": scenario_id,
                            "split": split,
                            "src_uav_id": f"uav_{src:02d}",
                            "dst_uav_id": f"uav_{dst:02d}",
                            "distance": round(distance, 6),
                            "los": metrics.los,
                            "tx_power_dbm": round(metrics.tx_power_dbm, 6),
                            "rx_power_dbm": round(metrics.rx_power_dbm, 6),
                            "noise_power_dbm": round(metrics.noise_power_dbm, 6),
                            "rssi": round(metrics.rssi_dbm, 6),
                            "snr": round(metrics.snr_db, 6),
                            "channel_bandwidth_hz": round(metrics.channel_bandwidth_hz, 6),
                            "rate_bps": round(metrics.rate_bps, 6),
                            "packet_loss": round(metrics.packet_loss, 6),
                            "delay": round(metrics.delay_s, 9),
                            "aoi": round(metrics.aoi_s, 6),
                            "queue_length": round(metrics.queue_length_packets, 6),
                            "potential_edge": int(distance <= config.max_comm_range),
                            "candidate_edge": metrics.candidate_edge,
                            "effective_edge": metrics.effective_edge,
                            "uwb_available": int(distance <= 100.0),
                            "link_state": metrics.link_state,
                            "bandwidth_kbps": round(metrics.bandwidth_kbps, 6),
                        }
                    )

            for uav_index, state in states.items():
                uav_id = f"uav_{uav_index:02d}"
                metrics_for_uav = [
                    directed_metrics[(uav_index, neighbor)]
                    for neighbor in neighbors[uav_index]
                ]
                averages = (
                    [
                        sum(metric.rssi_dbm for metric in metrics_for_uav) / len(metrics_for_uav),
                        sum(metric.snr_db for metric in metrics_for_uav) / len(metrics_for_uav),
                        sum(metric.packet_loss for metric in metrics_for_uav) / len(metrics_for_uav),
                        sum(metric.delay_s for metric in metrics_for_uav) / len(metrics_for_uav),
                        sum(metric.aoi_s for metric in metrics_for_uav) / len(metrics_for_uav),
                    ]
                    if metrics_for_uav
                    else ["", "", "", "", ""]
                )
                ranges = [
                    math.sqrt(
                        (state["x"] - states[neighbor]["x"]) ** 2
                        + (state["y"] - states[neighbor]["y"]) ** 2
                        + (state["z"] - states[neighbor]["z"]) ** 2
                    )
                    for neighbor in neighbors[uav_index]
                ]
                noisy_x = state["x"] + scenario_rng.gauss(0.0, 0.08)
                noisy_y = state["y"] + scenario_rng.gauss(0.0, 0.08)
                noisy_z = state["z"] + scenario_rng.gauss(0.0, 0.05)
                trajectory_rows.append(
                    {
                        "timestamp": timestamp,
                        "scenario_id": scenario_id,
                        "split": split,
                        "uav_id": uav_id,
                        "x": round(noisy_x, 6),
                        "y": round(noisy_y, 6),
                        "z": round(noisy_z, 6),
                        "vx": round(state["vx"], 6),
                        "vy": round(state["vy"], 6),
                        "vz": round(state["vz"], 6),
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": round(state["yaw"], 6),
                        "accel_x": round(state["ax"] + scenario_rng.gauss(0.0, 0.03), 6),
                        "accel_y": round(state["ay"] + scenario_rng.gauss(0.0, 0.03), 6),
                        "accel_z": round(state["az"] + scenario_rng.gauss(0.0, 0.03), 6),
                        "gyro_x": round(scenario_rng.gauss(0.0, 0.01), 6),
                        "gyro_y": round(scenario_rng.gauss(0.0, 0.01), 6),
                        "gyro_z": round((state["yaw"] + math.pi) % (2 * math.pi) - math.pi, 6),
                        "uwb_range": round(min(ranges), 6) if ranges else "",
                        "neighbor_ids": ";".join(f"uav_{neighbor:02d}" for neighbor in sorted(neighbors[uav_index])),
                        "rssi": round(averages[0], 6) if averages[0] != "" else "",
                        "snr": round(averages[1], 6) if averages[1] != "" else "",
                        "packet_loss": round(averages[2], 6) if averages[2] != "" else "",
                        "delay": round(averages[3], 9) if averages[3] != "" else "",
                        "aoi": round(averages[4], 6) if averages[4] != "" else "",
                        "gnss_mode": _gnss_mode(scenario_index, timestamp),
                        "obstacle_state": _obstacle_state((state["x"], state["y"]), buildings, timestamp),
                        "action_vx": round(state["vx"], 6),
                        "action_vy": round(state["vy"], 6),
                        "action_vz": round(state["vz"], 6),
                        "ground_truth_x": round(state["x"], 6),
                        "ground_truth_y": round(state["y"], 6),
                        "ground_truth_z": round(state["z"], 6),
                    }
                )

    _write_csv(output / "trajectory.csv", TRAJECTORY_COLUMNS, trajectory_rows)
    _write_csv(output / "links.csv", LINK_COLUMNS, link_rows)
    _write_csv(output / "obstacles.csv", OBSTACLE_COLUMNS, obstacle_rows)

    metadata = {
        "dataset_id": "python_sim_compatibility_benchmark",
        "generator": "research-project.src.synthetic.generator",
        "target_engine": "python_sim",
        "engine_status": "dependency-free compatibility generator; use generate-episodes for the action-driven environment",
        "seed": config.seed,
        "config": {
            **asdict(config),
            "output_dir": str(output),
            "uav_counts": list(config.uav_counts),
        },
        "scenario_records": scenario_records,
        "files": {
            "trajectory": "trajectory.csv",
            "links": "links.csv",
            "obstacles": "obstacles.csv",
        },
        "notes": [
            "Use this output to test schema, preprocessing, graph construction, and experiment plumbing.",
            "The action-driven pure-Python environment in src.simulation is the main research source; this flat generator remains for compatibility tests.",
            "All noise and channel parameters come from the versioned Python configuration; formal studies use sensitivity and repeated-seed checks rather than external datasets.",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "scenarios": config.scenarios,
        "total_uavs_across_scenarios": total_uavs,
        "trajectory_rows": len(trajectory_rows),
        "link_rows": len(link_rows),
        "obstacle_rows": len(obstacle_rows),
        "seed": config.seed,
    }
