"""Fixed 256-draw linear comparison, independent of all simulator records."""
from __future__ import annotations

import argparse
import numpy as np

from ..experiments import three_uav_fusion_validation_v6 as v6
from ..simulation.random_streams import indexed_rng
from .cooperative_ekf_reliability_reference_v5 import fixed_gain_covariance_bound, young_bound_rows


def run(run_id):
    output = v6.Output(run_id)
    draws = indexed_rng(2026, 0, "diag_v6_fixed_linear_gaussian").normal(size=(256, 2))
    e_own = draws[:, 0]
    e_source = 2 * draws[:, 0]
    noise = .08 * draws[:, 1]
    gain = 1 / 9.0064
    error = (1 - gain) * e_own + gain * e_source + gain * noise
    bound = fixed_gain_covariance_bound(1., 1., gain, 4., .0064)["bound"]
    known_cross_gain = -1 / 1.0064
    joint_error = (1 - known_cross_gain) * e_own + known_cross_gain * e_source + known_cross_gain * noise
    joint_var = 1 - 1 / 1.0064
    actual_var = (1 + gain) ** 2 + gain ** 2 * .0064
    entries = []
    for method, errors, variance in (("restore_prior", error, 1.), ("young_same_gain", error, bound), ("known_cross_joint_linear", joint_error, joint_var)):
        entries.append({"method": method, "draws": 256, "mean_error": float(errors.mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                        "reported_variance": variance, "empirical_centered_variance_ddof1": float(np.var(errors, ddof=1)),
                        "mean_nees": float(np.mean(errors ** 2 / variance)),
                        "expected_error_variance": actual_var if method != "known_cross_joint_linear" else joint_var,
                        "mean_nis": float(np.mean((noise - (e_own - e_source)) ** 2 / (9.0064 if method != "known_cross_joint_linear" else 1.0064))),
                        "fixed_gain": gain if method != "known_cross_joint_linear" else known_cross_gain})
        output.row("linear_comparison", entries[-1])
    grid = young_bound_rows(v6.old.CONFIG)
    if not all(r.get("bound_ge_actual", r.get("bound_ge_actual_psd")) for r in grid):
        raise AssertionError("fixed-gain bound grid failure")
    result = {"draw_hash": v6.old.content_hash(draws), "actual_scalar_variance": actual_var,
              "draws": 256, "entries": entries, "environment_steps": 0,
              "mean_equality_restore_young_max_abs": 0., "young_bound_grid_checks": len(grid),
              "claim": "Fixed linear Gaussian benchmark, known cross covariance for joint; not nonlinear recursive proof."}
    v6.old.write_json(output.data / "draws.json", draws)
    v6.old.write_json(output.data / "linear_report.json", result)
    output.flush()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(v6.old.encode(run(args.run_id)))


if __name__ == "__main__":
    main()
