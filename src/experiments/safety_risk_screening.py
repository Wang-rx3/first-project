"""Safety-risk scenario screening without MAPPO training.

This module is deliberately separate from formal algorithm evaluation.  It
answers one prerequisite question: does a truth-free, independent controller
encounter a measurable but non-degenerate safety problem in the proposed
three-UAV geometry?  The protected replay keeps the normal safety projector;
an unprotected replay is a diagnostic only and never represents the proposed
algorithm's final operating mode.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..simulation.environment import MultiUAVJointEnv, RewardConfig, SimulationConfig
from ..simulation.ekf import EKFConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.obstacle_model import collision_event
from ..simulation.scenario import generate_safety_risk_scenario
from ..simulation.scenario_profile import load_scenario_profile
from ..simulation.sensor_model import SensorConfig
from ..simulation.topology import TopologyConfig
from ..synthetic.channel_model import ChannelConfig
from .baseline import independent_fixed_actions
from .formal import load_selected_ekf_config


DEFAULT_SCENARIO_SEEDS = (
    2026,
    3035,
    4044,
    5053,
    6062,
    7071,
    8080,
    9089,
    10098,
    11107,
)


@dataclass(frozen=True)
class SafetyRiskScreeningConfig:
    """Configuration for the independent safety-risk screening experiment."""

    output_dir: Path
    logs_dir: Path | None = None
    plots_dir: Path | None = None
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    uav_count: int = 3
    max_steps: int = 50
    cruise_speed_mps: float = 3.0
    gnss_mode: str = "denied"
    scenario_profile_path: Path = Path(
        "data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json"
    )
    ekf_config_path: Path = Path(
        "data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json"
    )
    safety_projection_enabled: bool = True
    dynamic_obstacle_swept_projection_enabled: bool = False
    include_unprotected_diagnostic: bool = True
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("the safety-risk screening is restricted to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.gnss_mode != "denied":
            raise ValueError("the safety-risk screening requires gnss_mode='denied'")
        if not self.scenario_seeds:
            raise ValueError("at least one scenario seed is required")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if not math.isfinite(float(self.cruise_speed_mps)) or self.cruise_speed_mps <= 0.0:
            raise ValueError("cruise_speed_mps must be finite and positive")


def _progress(config: SafetyRiskScreeningConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-screening] {message}", flush=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_outputs(config: SafetyRiskScreeningConfig) -> tuple[Path, Path, Path]:
    output = Path(config.output_dir)
    logs = Path(config.logs_dir) if config.logs_dir is not None else Path("logs/preliminary") / output.name
    plots = Path(config.plots_dir) if config.plots_dir is not None else Path("plots/preliminary") / output.name
    managed = (output, logs, plots)
    nonempty = [path for path in managed if path.exists() and any(path.iterdir())]
    if nonempty and not config.force:
        raise FileExistsError(
            "safety screening output is not empty: "
            + ", ".join(str(path) for path in nonempty)
            + "; use --force to replace these outputs"
        )
    if config.force:
        for path in managed:
            if path.exists():
                shutil.rmtree(path)
    for path in managed:
        path.mkdir(parents=True, exist_ok=True)
    return output, logs, plots


def _make_environment(
    config: SafetyRiskScreeningConfig,
    parameters: Mapping[str, float],
    ekf_parameters: Mapping[str, float],
    seed: int,
    *,
    safety_projection_enabled: bool,
) -> MultiUAVJointEnv:
    motion = MotionConfig(
        dt_s=0.2,
        bounds_min_m=(0.0, 0.0, 2.0),
        bounds_max_m=(200.0, 200.0, 60.0),
    )
    sensor = SensorConfig(
        uwb_max_range_m=100.0,
        vo_position_std_m=float(ekf_parameters["vo_position_std_m"]),
        uwb_nlos_variance_multiplier=float(ekf_parameters["nlos_variance_multiplier"]),
    )
    ekf = EKFConfig(
        process_position_std_m=float(ekf_parameters["process_position_std_m"]),
        process_velocity_std_mps=float(ekf_parameters["process_velocity_std_mps"]),
        vo_position_std_m=float(ekf_parameters["vo_position_std_m"]),
    )
    channel = ChannelConfig(
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
            safety_projection_enabled=bool(safety_projection_enabled),
            dynamic_obstacle_swept_projection_enabled=bool(
                config.dynamic_obstacle_swept_projection_enabled
            ),
        )
    )


def _finite_environment(environment: MultiUAVJointEnv) -> bool:
    arrays: list[np.ndarray] = []
    arrays.extend(state.position_m for state in environment.states.values())
    arrays.extend(state.velocity_mps for state in environment.states.values())
    arrays.extend(ekf.state for ekf in environment.ekfs.values())
    arrays.extend(ekf.covariance for ekf in environment.ekfs.values())
    return bool(all(np.isfinite(array).all() for array in arrays))


def _episode(
    config: SafetyRiskScreeningConfig,
    parameters: Mapping[str, float],
    ekf_parameters: Mapping[str, float],
    scenario_seed: int,
    scenario: Mapping[str, Any],
    *,
    mode: str,
    safety_projection_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = _make_environment(
        config,
        parameters,
        ekf_parameters,
        scenario_seed,
        safety_projection_enabled=safety_projection_enabled,
    )
    observations, _reset_info = environment.reset(
        seed=int(scenario_seed),
        scenario_config=scenario,
    )
    initial_positions = {
        uav_id: state.position_m.copy() for uav_id, state in environment.states.items()
    }
    initial_collision, initial_clearance = collision_event(
        initial_positions,
        environment.obstacles,
        environment.config.motion.safety_distance_m,
    )
    previous_positions = {key: value.copy() for key, value in initial_positions.items()}
    step_rows: list[dict[str, Any]] = []
    trajectory_length = 0.0
    energy_sum = 0.0
    finite = int(initial_collision == 0 and _finite_environment(environment))
    collision_occurred = 0
    pre_projection_collision_occurred = 0
    min_clearance = float(initial_clearance)
    pre_projection_min_clearance = float(initial_clearance)
    projection_count = 0
    boundary_projection_count = 0
    selected_measurement_action_count = 0
    cooperative_ekf_update_count = 0

    for step in range(config.max_steps):
        actions = independent_fixed_actions(
            observations,
            environment.config.motion,
            cruise_speed_mps=config.cruise_speed_mps,
        )
        # The fixed controller is explicitly independent: it never selects a
        # neighbor measurement.  Bandwidth remains a deterministic neutral
        # allocation so the Moderate communication configuration is present.
        for action in actions.values():
            action["measurement_uav_id"] = "none"
        selected_measurement_action_count += sum(
            str(action.get("measurement_uav_id", "none")) != "none"
            for action in actions.values()
        )
        observations, rewards, terminated, truncated, info = environment.step(actions)
        cooperative_ekf_update_count = len(info.get("relative_range_update_events", []))
        collision = int(info.get("collision", 0))
        pre_collision = int(info.get("pre_projection_collision", 0))
        minimum = float(info.get("minimum_clearance", float("inf")))
        pre_minimum = float(info.get("pre_projection_minimum_clearance", float("inf")))
        executed_deltas = [
            np.asarray(info.get("action_execution", {}).get(uav_id, {}).get("delta_v_applied", np.zeros(3)), dtype=float)
            for uav_id in environment.uav_ids
        ]
        delta_norms = [float(np.linalg.norm(delta)) for delta in executed_deltas]
        action_norm_mean = float(np.mean(delta_norms)) if delta_norms else 0.0
        energy_step = float(
            np.mean(
                [
                    norm / (environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s)
                    for norm in delta_norms
                ]
            )
            if delta_norms
            else 0.0
        )
        displacement = [
            float(np.linalg.norm(environment.states[uav_id].position_m - previous_positions[uav_id]))
            for uav_id in environment.uav_ids
        ]
        path_step = float(sum(displacement))
        trajectory_length += path_step
        energy_sum += energy_step
        projection_step = int(info.get("safety_projection_count", 0))
        boundary_step = int(info.get("boundary_projection_count", 0))
        projection_count += projection_step
        boundary_projection_count += boundary_step
        collision_occurred = max(collision_occurred, collision)
        pre_projection_collision_occurred = max(pre_projection_collision_occurred, pre_collision)
        min_clearance = min(min_clearance, minimum)
        pre_projection_min_clearance = min(pre_projection_min_clearance, pre_minimum)
        finite = int(
            finite
            and _finite_environment(environment)
            and all(np.isfinite(value) for value in (minimum, pre_minimum, action_norm_mean, energy_step, path_step))
        )
        step_rows.append(
            {
                "scenario_seed": int(scenario_seed),
                "mode": mode,
                "step": int(step + 1),
                "time_s": float((step + 1) * environment.config.motion.dt_s),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "termination_reason": str(info.get("termination_reason", "running")),
                "collision": collision,
                "pre_projection_collision": pre_collision,
                "minimum_clearance_m": minimum,
                "pre_projection_minimum_clearance_m": pre_minimum,
                "safety_projection_count": projection_step,
                "boundary_projection_count": boundary_step,
                "action_magnitude_mean": action_norm_mean,
                "action_magnitude_max": float(max(delta_norms)) if delta_norms else 0.0,
                "energy_proxy_step": energy_step,
                "path_length_step_m": path_step,
                "trajectory_length_m": trajectory_length,
                "state_finite": int(_finite_environment(environment)),
                "covariance_finite": int(
                    all(np.isfinite(ekf.covariance).all() for ekf in environment.ekfs.values())
                ),
                "selected_measurement_actions": int(selected_measurement_action_count),
                "cooperative_ekf_updates": int(cooperative_ekf_update_count),
                "team_reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
                "dynamic_obstacle_altitude_overlap": 1,
            }
        )
        previous_positions = {
            uav_id: state.position_m.copy() for uav_id, state in environment.states.items()
        }
        if terminated or truncated:
            break

    steps = max(1, len(step_rows))
    risk_clearance_threshold = float(environment.config.motion.safety_distance_m)
    summary = {
        "scenario_seed": int(scenario_seed),
        "mode": mode,
        "uav_count": int(config.uav_count),
        "episode_steps": int(len(step_rows)),
        "initial_collision": int(initial_collision),
        "initial_minimum_clearance_m": float(initial_clearance),
        "collision_episode": int(collision_occurred),
        "pre_projection_collision_episode": int(pre_projection_collision_occurred),
        "minimum_clearance_m": float(min_clearance),
        "pre_projection_minimum_clearance_m": float(pre_projection_min_clearance),
        "near_safety_threshold": int(min_clearance <= risk_clearance_threshold),
        "pre_projection_near_safety_threshold": int(pre_projection_min_clearance <= risk_clearance_threshold),
        "safety_projection_count": int(projection_count),
        "boundary_projection_count": int(boundary_projection_count),
        "action_magnitude_mean": float(np.mean([row["action_magnitude_mean"] for row in step_rows])) if step_rows else 0.0,
        "action_magnitude_max": float(max((row["action_magnitude_max"] for row in step_rows), default=0.0)),
        "energy_proxy": float(energy_sum / steps),
        "trajectory_length_m": float(trajectory_length),
        "selected_measurement_action_count": int(selected_measurement_action_count),
        "cooperative_ekf_update_count": int(cooperative_ekf_update_count),
        "finite_state": int(finite),
        "terminated": int(bool(step_rows and step_rows[-1]["terminated"])),
        "truncated": int(bool(step_rows and step_rows[-1]["truncated"])),
        "termination_reason": str(step_rows[-1]["termination_reason"] if step_rows else "running"),
        "flight_altitude_m": float(scenario["safety_risk_design"]["flight_altitude_m"]),
        "dynamic_obstacle_altitude_overlap": 1,
        "controller": "independent_fixed_goal_tracking_without_neighbor_measurements",
        "safety_projection_enabled": int(bool(safety_projection_enabled)),
        "dynamic_obstacle_swept_projection_enabled": int(
            bool(config.dynamic_obstacle_swept_projection_enabled)
        ),
    }
    return step_rows, summary


def _plot_screening(path: Path, scenario_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ("projection_on", "projection_off")
    colors = {"projection_on": "#2f6de1", "projection_off": "#d62728"}
    grouped = {
        mode: [row for row in scenario_rows if row["mode"] == mode]
        for mode in modes
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    collision_rates = [
        float(np.mean([float(row["collision_episode"]) for row in grouped[mode]]))
        if grouped[mode]
        else 0.0
        for mode in modes
    ]
    pre_rates = [
        float(np.mean([float(row["pre_projection_collision_episode"]) for row in grouped[mode]]))
        if grouped[mode]
        else 0.0
        for mode in modes
    ]
    x = np.arange(len(modes), dtype=float)
    width = 0.36
    axes[0, 0].bar(x - width / 2, collision_rates, width, label="actual collision", color="#444444")
    axes[0, 0].bar(x + width / 2, pre_rates, width, label="pre-projection risk", color="#f0a202")
    axes[0, 0].set_xticks(x, ["projection on", "projection off"])
    axes[0, 0].set_ylim(bottom=0.0)
    axes[0, 0].set_ylabel("episode fraction")
    axes[0, 0].set_title("Collision and pre-projection risk")
    axes[0, 0].legend(fontsize=8)

    data = [
        [float(row["minimum_clearance_m"]) for row in grouped[mode]] or [np.nan]
        for mode in modes
    ]
    axes[0, 1].boxplot(data, labels=["projection on", "projection off"], showmeans=True)
    axes[0, 1].axhline(2.0, color="#d62728", linestyle="--", linewidth=1.0, label="safety threshold")
    axes[0, 1].set_ylabel("minimum clearance (m)")
    axes[0, 1].set_title("Scenario-level minimum clearance")
    axes[0, 1].legend(fontsize=8)

    for mode in modes:
        rows = grouped[mode]
        axes[1, 0].plot(
            [int(row["scenario_seed"]) for row in rows],
            [float(row["safety_projection_count"]) for row in rows],
            marker="o",
            label=mode.replace("_", " "),
            color=colors[mode],
        )
    axes[1, 0].set_xlabel("scenario seed")
    axes[1, 0].set_ylabel("safety projection count")
    axes[1, 0].set_title("Safety projection diagnostics")
    axes[1, 0].legend(fontsize=8)

    for mode in modes:
        rows = grouped[mode]
        axes[1, 1].scatter(
            [float(row["energy_proxy"]) for row in rows],
            [float(row["trajectory_length_m"]) for row in rows],
            label=mode.replace("_", " "),
            color=colors[mode],
            alpha=0.85,
        )
    axes[1, 1].set_xlabel("energy proxy")
    axes[1, 1].set_ylabel("trajectory length (m)")
    axes[1, 1].set_title("Motion cost diagnostics")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_safety_risk_screening(config: SafetyRiskScreeningConfig) -> dict[str, Any]:
    """Run the fixed-controller safety-risk screening and write its audit."""

    output, logs, plots = _prepare_outputs(config)
    profile = load_scenario_profile(config.scenario_profile_path, expected_uav_count=config.uav_count)
    ekf_parameters = load_selected_ekf_config(config.ekf_config_path)
    parameters = dict(profile["parameters"])
    step_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    modes = ["projection_on"]
    if config.include_unprotected_diagnostic:
        modes.append("projection_off")

    scenarios_dir = output / "scenarios"
    for scenario_seed in config.scenario_seeds:
        scenario = generate_safety_risk_scenario(
            int(scenario_seed),
            config.uav_count,
        )
        _write_json(scenarios_dir / f"scenario_{int(scenario_seed)}.json", scenario)
        for mode in modes:
            _progress(config, f"seed={scenario_seed} mode={mode} start")
            rows, summary = _episode(
                config,
                parameters,
                ekf_parameters,
                int(scenario_seed),
                scenario,
                mode=mode,
                safety_projection_enabled=(mode == "projection_on"),
            )
            step_rows.extend(rows)
            scenario_rows.append(summary)
            _progress(
                config,
                f"seed={scenario_seed} mode={mode} steps={summary['episode_steps']} "
                f"collision={summary['collision_episode']} "
                f"pre_risk={summary['pre_projection_collision_episode']}",
            )

    protected = [row for row in scenario_rows if row["mode"] == "projection_on"]
    diagnostic = [row for row in scenario_rows if row["mode"] == "projection_off"]
    all_rows = scenario_rows
    initial_collision_count = int(sum(int(row["initial_collision"]) for row in all_rows))
    finite_all = bool(all(int(row["finite_state"]) == 1 for row in all_rows))
    protected_collision_rate = float(np.mean([row["collision_episode"] for row in protected])) if protected else 0.0
    protected_pre_risk_rate = float(np.mean([row["pre_projection_collision_episode"] for row in protected])) if protected else 0.0
    diagnostic_collision_rate = float(np.mean([row["collision_episode"] for row in diagnostic])) if diagnostic else 0.0
    near_threshold_rate = float(
        np.mean(
            [
                int(row["near_safety_threshold"] or row["pre_projection_near_safety_threshold"])
                for row in protected
            ]
        )
        if protected
        else 0.0
    )
    if protected_collision_rate > 0.0:
        gate_rate = protected_collision_rate
        gate_basis = "projection_on_actual_collision"
    elif protected_pre_risk_rate > 0.0:
        gate_rate = protected_pre_risk_rate
        gate_basis = "projection_on_pre_projection_risk"
    else:
        gate_rate = diagnostic_collision_rate
        gate_basis = "projection_off_diagnostic_collision"
    collision_gate_passed = bool(0.0 < gate_rate < 0.80)
    preferred_rate_range = bool(0.05 <= gate_rate <= 0.30)
    screening_passed = bool(
        initial_collision_count == 0
        and finite_all
        and near_threshold_rate > 0.0
        and collision_gate_passed
        and preferred_rate_range
    )
    audit_ok = bool(
        len(scenario_rows) == len(config.scenario_seeds) * len(modes)
        and initial_collision_count == 0
        and finite_all
        and all(row["dynamic_obstacle_altitude_overlap"] == 1 for row in all_rows)
    )
    report: dict[str, Any] = {
        "schema_version": "safety-risk-screening-v1.0",
        "audit_ok": audit_ok,
        "screening_passed": screening_passed,
        "design": {
            "uav_count": config.uav_count,
            "max_steps": config.max_steps,
            "cruise_speed_mps": config.cruise_speed_mps,
            "gnss_mode": config.gnss_mode,
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "independent_unit": "scenario_seed",
            "controller": "independent_fixed_goal_tracking_without_neighbor_measurements",
            "uses_neighbor_information": False,
            "trainer_updates": 0,
            "test_data_used_for_fitting_or_selection": False,
            "time_windows_are_diagnostic_only": True,
            "safety_projection_default": True,
            "dynamic_obstacle_swept_projection_enabled": bool(
                config.dynamic_obstacle_swept_projection_enabled
            ),
            "unprotected_replay_is_diagnostic_only": True,
        },
        "fixed_configuration": {
            "scenario_profile": str(Path(config.scenario_profile_path).resolve()),
            "scenario_profile_name": profile.get("profile"),
            "scenario_parameters": parameters,
            "ekf_config": str(Path(config.ekf_config_path).resolve()),
            "ekf_parameters": ekf_parameters,
            "motion": asdict(MotionConfig()),
            "dynamic_obstacle_swept_projection_enabled": bool(
                config.dynamic_obstacle_swept_projection_enabled
            ),
        },
        "counts": {
            "scenario_units": len(config.scenario_seeds),
            "modes": modes,
            "scenario_rows": len(scenario_rows),
            "step_rows": len(step_rows),
            "initial_collision_count": initial_collision_count,
            "finite_scenario_count": int(sum(int(row["finite_state"]) for row in all_rows)),
            "selected_measurement_action_count": int(
                sum(int(row["selected_measurement_action_count"]) for row in all_rows)
            ),
            "cooperative_ekf_update_count": int(
                sum(int(row["cooperative_ekf_update_count"]) for row in all_rows)
            ),
        },
        "protected_projection_on": {
            "collision_episode_rate": protected_collision_rate,
            "pre_projection_risk_episode_rate": protected_pre_risk_rate,
            "minimum_clearance_near_threshold_rate": near_threshold_rate,
            "mean_safety_projection_count": float(np.mean([row["safety_projection_count"] for row in protected])) if protected else 0.0,
            "mean_energy_proxy": float(np.mean([row["energy_proxy"] for row in protected])) if protected else 0.0,
        },
        "unprotected_diagnostic": {
            "available": bool(diagnostic),
            "collision_episode_rate": diagnostic_collision_rate,
        },
        "predeclared_gate": {
            "collision_rate_used": gate_rate,
            "collision_rate_basis": gate_basis,
            "collision_gate_passed_0_to_80_percent": collision_gate_passed,
            "preferred_rate_range_5_to_30_percent": preferred_rate_range,
            "near_threshold_present": bool(near_threshold_rate > 0.0),
            "initial_state_safe": bool(initial_collision_count == 0),
            "all_state_covariance_energy_finite": finite_all,
        },
        "interpretation": {
            "mappo_trained": False,
            "algorithm_comparison_allowed": bool(screening_passed),
            "current_result_scope": "scenario screening only",
            "next_step_if_passed": "paired baseline versus full MAPPO safety-energy evaluation on the same scenario seeds",
        },
    }
    _write_csv(logs / "step_metrics.csv", step_rows)
    _write_csv(logs / "scenario_metrics.csv", scenario_rows)
    _write_csv(
        logs / "screening_summary.csv",
        [
            {"metric": "protected_collision_episode_rate", "value": protected_collision_rate},
            {"metric": "protected_pre_projection_risk_episode_rate", "value": protected_pre_risk_rate},
            {"metric": "diagnostic_unprotected_collision_episode_rate", "value": diagnostic_collision_rate},
            {"metric": "gate_rate", "value": gate_rate},
            {"metric": "near_threshold_rate", "value": near_threshold_rate},
            {"metric": "screening_passed", "value": int(screening_passed)},
        ],
    )
    _write_json(logs / "audit_report.json", report)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "safety-risk-screening-v1.0",
            "output_dir": str(output.resolve()),
            "logs_dir": str(logs.resolve()),
            "plots_dir": str(plots.resolve()),
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "fixed_controller": "independent_fixed_goal_tracking_without_neighbor_measurements",
            "uses_neighbor_information": False,
            "trainer_updates": 0,
            "test_data_used_for_fitting_or_selection": False,
            "dynamic_obstacle_swept_projection_enabled": bool(
                config.dynamic_obstacle_swept_projection_enabled
            ),
            "formal_episode_schema_used": False,
            "audit_report": str((logs / "audit_report.json").resolve()),
        },
    )
    _plot_screening(plots / "safety_risk_screening.png", scenario_rows)
    return {
        "audit_ok": audit_ok,
        "screening_passed": screening_passed,
        "collision_rate": gate_rate,
        "collision_rate_basis": gate_basis,
        "scenario_units": len(config.scenario_seeds),
        "independent_unit": "scenario_seed",
        "data": str(output),
        "logs": str(logs),
        "plot": str(plots / "safety_risk_screening.png"),
        "audit_report": str(logs / "audit_report.json"),
    }


__all__ = [
    "DEFAULT_SCENARIO_SEEDS",
    "SafetyRiskScreeningConfig",
    "run_safety_risk_screening",
]
