"""Training-only calibration sweeps for the synthetic EKF configuration.

The sweep is deliberately independent of MAPPO training.  It uses the same
seeded Python scenes and a fixed equal-bandwidth probe, so a selected sensor
configuration cannot be chosen by looking at validation or test episodes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .formal import FormalExperimentConfig, _make_environment, _scenario_for
from .pilot import _probe_actions, _seed_everything


@dataclass(frozen=True)
class EKFCalibrationConfig:
    output_dir: Path
    seed: int = 2026
    uav_count: int = 3
    episodes: int = 3
    max_steps: int = 50
    vo_stds_m: tuple[float, ...] = (0.015, 0.03, 0.05)
    process_position_stds_m: tuple[float, ...] = (0.03, 0.08, 0.15)
    process_velocity_stds_mps: tuple[float, ...] = (0.08, 0.15, 0.25)
    nlos_variance_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0)
    force: bool = False

    def __post_init__(self) -> None:
        if self.uav_count not in {3, 5, 7}:
            raise ValueError("uav_count must be one of 3, 5, or 7")
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")
        for values in (
            self.vo_stds_m,
            self.process_position_stds_m,
            self.process_velocity_stds_mps,
            self.nlos_variance_multipliers,
        ):
            if not values or any(float(value) <= 0.0 for value in values):
                raise ValueError("all calibration candidates must be positive")


_NEES_95_LOWER = 0.2158
_NEES_95_UPPER = 9.3484


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else default


def _nees(estimate: np.ndarray, truth: np.ndarray, covariance: np.ndarray) -> float:
    estimate_position = np.asarray(estimate, dtype=float).reshape(-1)[:3]
    error = np.asarray(truth, dtype=float).reshape(3) - estimate_position
    position_covariance = np.asarray(covariance, dtype=float)[:3, :3]
    value = float(error @ np.linalg.pinv(position_covariance) @ error)
    return value if np.isfinite(value) else float("nan")


def _candidate_rows(config: EKFCalibrationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_index = 0
    total_candidates = (
        len(config.vo_stds_m)
        * len(config.process_position_stds_m)
        * len(config.process_velocity_stds_mps)
        * len(config.nlos_variance_multipliers)
    )
    for vo_std in config.vo_stds_m:
        for process_position_std in config.process_position_stds_m:
            for process_velocity_std in config.process_velocity_stds_mps:
                for nlos_multiplier in config.nlos_variance_multipliers:
                    candidate_index += 1
                    formal = FormalExperimentConfig(
                        output_dir=config.output_dir / "_scan_runtime",
                        uav_count=config.uav_count,
                        seed=config.seed,
                        max_steps=config.max_steps,
                        warmup_episodes=1,
                        train_episodes=1,
                        validation_episodes=1,
                        test_episodes=1,
                        device="cpu",
                        vo_position_std_m=float(vo_std),
                        process_position_std_m=float(process_position_std),
                        process_velocity_std_mps=float(process_velocity_std),
                        nlos_variance_multiplier=float(nlos_multiplier),
                        progress=False,
                    )
                    nees_values: list[float] = []
                    final_nees: list[float] = []
                    position_errors: list[float] = []
                    covariance_traces: list[float] = []
                    accepted_nis: list[float] = []
                    rejected_nis: list[float] = []
                    accepted_updates = 0
                    rejected_updates = 0
                    for episode_index in range(config.episodes):
                        seed = int(config.seed + 1009 * episode_index)
                        _seed_everything(seed)
                        environment = _make_environment(formal, seed)
                        _observations, _info = environment.reset(
                            seed=seed,
                            scenario_config=_scenario_for(formal, seed),
                        )
                        episode_nees: list[float] = []
                        for _step in range(config.max_steps):
                            actions = _probe_actions(environment)
                            _next_observations, _rewards, terminated, truncated, info = environment.step(actions)
                            for uav_id in environment.uav_ids:
                                estimate = info["central_state"]["estimates"][uav_id]
                                covariance = info["central_state"]["covariances"][uav_id]
                                truth = environment.states[uav_id].position_m
                                value = _nees(estimate, truth, covariance)
                                if np.isfinite(value):
                                    nees_values.append(value)
                                    episode_nees.append(value)
                                position_errors.append(float(np.linalg.norm(np.asarray(truth) - np.asarray(estimate[:3]))))
                                covariance_traces.append(float(np.trace(np.asarray(covariance)[:3, :3])))
                                diagnostic = info.get("ekf_diagnostics", {}).get(uav_id, {})
                                for key in ("nis_vo_accepted", "nis_uwb_accepted_mean", "nis_gnss_accepted"):
                                    value = float(diagnostic.get(key, -1.0))
                                    if np.isfinite(value) and value >= 0.0:
                                        accepted_nis.append(value)
                                for key in ("nis_vo_rejected", "nis_uwb_rejected_mean", "nis_gnss_rejected"):
                                    value = float(diagnostic.get(key, -1.0))
                                    if np.isfinite(value) and value >= 0.0:
                                        rejected_nis.append(value)
                                accepted_updates += int(diagnostic.get("vo_update_accepted", 0))
                                accepted_updates += int(diagnostic.get("uwb_update_accepted", 0))
                                accepted_updates += int(diagnostic.get("gnss_update_accepted", 0))
                                rejected_updates += int(diagnostic.get("vo_update_rejected", 0))
                                rejected_updates += int(diagnostic.get("uwb_update_rejected", 0))
                                rejected_updates += int(diagnostic.get("gnss_update_rejected", 0))
                            if terminated or truncated:
                                break
                        if episode_nees:
                            final_nees.append(float(episode_nees[-1]))
                    coverage = _mean(
                        float(_NEES_95_LOWER <= value <= _NEES_95_UPPER)
                        for value in nees_values
                    )
                    accepted_high_fraction = _mean(float(value > 9.0) for value in accepted_nis)
                    rejected_high_fraction = _mean(float(value > 9.0) for value in rejected_nis)
                    final_nees_mean = _mean(final_nees)
                    rmse = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else 0.0
                    score = (
                        abs(coverage - 0.95)
                        + 2.0 * max(0.0, final_nees_mean / _NEES_95_UPPER - 1.0)
                        + accepted_high_fraction
                        + 0.01 * rmse
                    )
                    rows.append(
                        {
                            "candidate_index": candidate_index,
                            "vo_std_m": float(vo_std),
                            "process_position_std_m": float(process_position_std),
                            "process_velocity_std_mps": float(process_velocity_std),
                            "nlos_variance_multiplier": float(nlos_multiplier),
                            "nees_coverage_95": coverage,
                            "nees_final_mean": final_nees_mean,
                            "position_rmse_m": rmse,
                            "position_covariance_trace_m2": _mean(covariance_traces),
                            "accepted_nis_mean": _mean(accepted_nis, -1.0),
                            "rejected_nis_mean": _mean(rejected_nis, -1.0),
                            "accepted_nis_high_fraction": accepted_high_fraction,
                            "rejected_nis_high_fraction": rejected_high_fraction,
                            "accepted_updates": accepted_updates,
                            "rejected_updates": rejected_updates,
                            "selection_score": float(score),
                        }
                    )
                    print(
                        f"[calibrate-ekf] candidate {candidate_index}/{total_candidates} "
                        f"vo={vo_std:g} pos={process_position_std:g} "
                        f"vel={process_velocity_std:g} nlos={nlos_multiplier:g} "
                        f"score={score:.6f}",
                        flush=True,
                    )
    return rows


def run_ekf_parameter_scan(config: EKFCalibrationConfig) -> dict[str, Any]:
    """Run the training-only grid and write the selected configuration."""

    root = Path(config.output_dir)
    if root.exists() and any(root.iterdir()) and not config.force:
        raise FileExistsError(f"calibration output is not empty: {root}; use --force to replace it")
    root.mkdir(parents=True, exist_ok=True)
    rows = _candidate_rows(config)
    rows.sort(key=lambda row: float(row["selection_score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = int(rank == 1)
    results_path = root / "ekf_parameter_scan.csv"
    columns = list(rows[0]) if rows else []
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    selected = dict(rows[0]) if rows else {}
    selected_coverage = float(selected.get("nees_coverage_95", float("nan")))
    coverage_target_met = bool(
        np.isfinite(selected_coverage) and selected_coverage >= 0.90
    )
    payload = {
        "schema_version": "ekf-calibration-v1",
        "scope": "training_only",
        "data_source": "pure_python_seeded_scenarios",
        "selection_rule": {
            "nees_interval_95": [_NEES_95_LOWER, _NEES_95_UPPER],
            "description": "Prefer NEES coverage near 0.95, finite final NEES, and high NIS concentrated in rejected observations.",
        },
        "config": asdict(config),
        "selected": selected,
        "selected_parameters": {
            "vo_position_std_m": selected.get("vo_std_m"),
            "process_position_std_m": selected.get("process_position_std_m"),
            "process_velocity_std_mps": selected.get("process_velocity_std_mps"),
            "nlos_variance_multiplier": selected.get("nlos_variance_multiplier"),
        },
        "selection_diagnostics": {
            "target_nees_coverage_95": 0.95,
            "minimum_acceptable_nees_coverage_95": 0.90,
            "selected_nees_coverage_95": selected_coverage,
            "target_met": coverage_target_met,
            "interpretation": (
                "The selected candidate is suitable for a preliminary run."
                if coverage_target_met
                else "No scanned candidate met the NEES coverage target; do not call this final calibration."
            ),
        },
        "results": str(results_path),
        "warning": (
            "This does not use validation/test data and does not prove real-world sensor calibration. "
            + ("The selected candidate is outside the target NEES coverage range." if not coverage_target_met else "")
        ),
    }
    (root / "selected_ekf_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return payload


__all__ = ["EKFCalibrationConfig", "run_ekf_parameter_scan"]
