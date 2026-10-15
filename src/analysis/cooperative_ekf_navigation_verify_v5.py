"""Independent navigation/geometry recomputation from saved v5 truth logs."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .cooperative_ekf_reliability_report_v5 import ROOT, STEM, UIDS, read_rows, value
from ..control.fixed_navigation import fixed_navigation_command
from ..experiments import three_uav_cooperative_ekf_reliability_v5 as v5


def verify(run_id):
    base = ROOT / "logs/preliminary" / STEM / run_id
    rows = read_rows(base / "localization.csv")
    cfg = v5.read_json(v5.CONFIG)
    cases = {v5.episode_id(r["arm"], r): r for r in cfg["run_matrix"]}
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["uav_id"] in UIDS:
            groups[row["episode_id"]][int(row["step"])][row["uav_id"]] = row
    minimum = float("inf")
    max_acceleration = max_speed = max_nav_difference = 0.0
    bounds_ok = True
    for episode, steps in groups.items():
        case = cases[episode]
        initial = case["scenario"]
        previous_positions = {u: np.asarray(initial["initial_positions"][u]) for u in UIDS}
        previous_velocities = {u: np.asarray(initial["initial_velocities"][u]) for u in UIDS}
        previous_estimates = {u: np.r_[previous_positions[u], previous_velocities[u]] + np.asarray(cfg["initial_errors_by_seed"][str(case["seed"])][u]) for u in UIDS}
        for step in sorted(steps):
            current = steps[step]
            positions = {u: np.asarray(value(current[u], "true_position_m")) for u in UIDS}
            velocities = {u: np.asarray(value(current[u], "true_velocity_mps")) for u in UIDS}
            for u in UIDS:
                max_speed = max(max_speed, float(np.linalg.norm(velocities[u])))
                max_acceleration = max(max_acceleration, float(np.linalg.norm((velocities[u] - previous_velocities[u]) / .2)))
                bounds_ok &= bool(np.all(positions[u] >= [0, 0, 2]) and np.all(positions[u] <= [200, 200, 60]))
                expected = fixed_navigation_command(previous_estimates[u][:6], initial["goals"][u])["command_delta_v_mps"]
                max_nav_difference = max(max_nav_difference, float(np.max(np.abs(expected - np.asarray(value(current[u], "actual_delta_v"))))))
            for i, left in enumerate(UIDS):
                for right in UIDS[i + 1:]:
                    r0 = previous_positions[left] - previous_positions[right]
                    increment = positions[left] - positions[right] - r0
                    denominator = float(increment @ increment)
                    tau = float(np.clip(-(r0 @ increment) / denominator, 0, 1)) if denominator > 1e-20 else 0.0
                    minimum = min(minimum, float(np.linalg.norm(r0 + tau * increment)))
            previous_positions, previous_velocities = positions, velocities
            previous_estimates = {u: np.asarray(value(current[u], "estimate_state")) for u in UIDS}
    result = {"minimum_continuous_segment_separation_m": minimum, "maximum_speed_mps": max_speed,
              "maximum_actual_acceleration_mps2": max_acceleration, "bounds_ok": bounds_ok,
              "raw_own_ekf_navigation_delta_v_max_abs_difference": max_nav_difference,
              "ok": bounds_ok and minimum >= 2 and max_speed <= 12 and max_acceleration <= 3 + 1e-8 and max_nav_difference <= 1e-10,
              "environment_steps": 0}
    v5.write_json(base / "navigation_recomputation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe run id")
    result = verify(args.run_id)
    print(v5.encode(result))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
