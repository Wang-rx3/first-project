"""Offline reference calculations for the cooperative-EKF reliability diagnostic.

The functions in this module deliberately operate on the locked JSON and the
historical v4 files only.  They do not construct an environment, train a
policy, fit a normalizer, or change the deployed EKF.  The matrices here are
diagnostic reference calculations; they are not a runtime covariance rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


UIDS = ("uav_00", "uav_01", "uav_02")
RANGE_VARIANCE_M2 = 0.08**2
V4_STEM = "three_uav_measurement_learnability_v4_seed2026_20261012"


def _config(config: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    return json.loads(Path(config).read_text(encoding="utf-8"))


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("non-finite numeric value")
    return result


def _sym(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=float) + np.asarray(value, dtype=float).T) / 2.0


def _eig(value: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(_sym(value))


def _tolist(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _restricted_logdet(value: np.ndarray, tolerance: float = 1e-10) -> float:
    """Log determinant over numerically positive eigenvalues.

    Correlation one produces a singular joint covariance.  Reporting this
    restricted value keeps the diagnostic finite while retaining the rank
    information in the row itself.
    """

    eigenvalues = _eig(value)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues[eigenvalues > tolerance * scale]
    return float(np.sum(np.log(positive))) if positive.size else 0.0


def _restricted_logdet_gain(before: np.ndarray, after: np.ndarray, tolerance: float = 1e-10) -> float:
    """Log-volume reduction on the positive-eigenvalue support of ``before``."""

    eigenvalues, vectors = np.linalg.eigh(_sym(before))
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    support = eigenvalues > tolerance * scale
    if not np.any(support):
        return 0.0
    before_support = eigenvalues[support]
    after_support = _sym(vectors[:, support].T @ _sym(after) @ vectors[:, support])
    after_eigenvalues = np.linalg.eigvalsh(after_support)
    positive_after = after_eigenvalues[after_eigenvalues > tolerance * scale]
    if positive_after.size != before_support.size:
        return float("inf")
    return float(np.sum(np.log(before_support)) - np.sum(np.log(positive_after)))


def _rank(value: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(value, dtype=float), tol=1e-10))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _state_index(uid_index: int) -> slice:
    return slice(6 * uid_index, 6 * uid_index + 6)


def _position_prior(config: Mapping[str, Any], correlation: float) -> np.ndarray:
    marginal = _array(config["initial_position_marginal_covariance"], (3, 3))
    if not np.allclose(marginal, 4.0 * np.eye(3), atol=1e-12):
        raise ValueError("v5 reference requires a 4 I position marginal")
    matrix = np.zeros((9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = marginal if i == j else correlation * marginal
    return _sym(matrix)


def _measurement_row(source_index: int, mean_position: np.ndarray | None = None) -> np.ndarray:
    row = np.zeros(9, dtype=float)
    if mean_position is None:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        mean = np.asarray(mean_position, dtype=float).reshape(9)
        direction = mean[:3] - mean[3 * source_index : 3 * source_index + 3]
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-15 else np.asarray([1.0, 0.0, 0.0])
    row[:3] = direction
    row[3 * source_index : 3 * source_index + 3] = -direction
    return row


def _relative_selector(source_index: int) -> np.ndarray:
    selector = np.zeros((3, 9), dtype=float)
    selector[:, :3] = np.eye(3)
    selector[:, 3 * source_index : 3 * source_index + 3] = -np.eye(3)
    return selector


def _measurement_matrix(case: str, mean_position: np.ndarray | None = None) -> np.ndarray:
    if case == "B":
        return _measurement_row(1, mean_position)[None, :]
    if case == "C":
        return _measurement_row(2, mean_position)[None, :]
    if case == "BCstack":
        return np.vstack((_measurement_row(1, mean_position), _measurement_row(2, mean_position)))
    if case == "BBtwo":
        row = _measurement_row(1, mean_position)
        return np.vstack((row, row))
    raise ValueError(f"unknown reference measurement case: {case}")


def _initial_mean_position(case: Mapping[str, Any], config: Mapping[str, Any], seed_override: str | int | None = None) -> np.ndarray:
    seed = str(case["seed"] if seed_override is None else seed_override)
    errors = config["initial_errors_by_seed"].get(seed)
    if errors is None:
        errors = config["initial_errors_by_seed"].get(int(seed))
    if errors is None:
        raise ValueError(f"missing locked initial errors for seed {seed}")
    positions = case["scenario"]["initial_positions"]
    values: list[float] = []
    for uid in UIDS:
        values.extend((_array(positions[uid], (3,)) + _array(errors[uid], (6,))[:3]).tolist())
    return np.asarray(values, dtype=float)


def _case_rows(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = list(config.get("cases", ()))
    if not cases:
        raise ValueError("v5 config has no locked cases")
    return cases


def initial_prior_audit(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate the locked 18-dimensional position/velocity prior law.

    The fixed vectors in ``initial_errors_by_seed`` are treated as the entire
    two-family realization.  Their empirical moments are reported as a
    description, never used to redraw or retune the prescribed law.
    """

    cfg = _config(config)
    order = list(cfg.get("initial_error_order", ()))
    expected_order = ["position_x_m", "position_y_m", "position_z_m", "velocity_x_mps", "velocity_y_mps", "velocity_z_mps"]
    if order != expected_order:
        raise ValueError(f"unexpected initial error order: {order}")
    marginal_p = _array(cfg["initial_position_marginal_covariance"], (3, 3))
    marginal_v = _array(cfg["initial_velocity_marginal_covariance"], (3, 3))
    cross_p = _array(cfg["initial_cross_position_covariance"], (3, 3))
    if not np.allclose(marginal_p, 4.0 * np.eye(3), atol=1e-12):
        raise ValueError("position marginal must be exactly 4 I")
    if not np.allclose(marginal_v, np.eye(3), atol=1e-12):
        raise ValueError("velocity marginal must be exactly I")
    if not np.allclose(cross_p, 2.0 * np.eye(3), atol=1e-12):
        raise ValueError("cross position covariance must be exactly 2 I")

    errors_by_seed = cfg.get("initial_errors_by_seed", {})
    seeds = [str(seed) for seed in cfg.get("seeds", errors_by_seed.keys())]
    if len(seeds) != 2:
        raise ValueError("the locked v5 audit expects exactly two seed families")
    fixed_errors: dict[str, dict[str, list[float]]] = {}
    rows: list[np.ndarray] = []
    per_uid: dict[str, list[np.ndarray]] = {uid: [] for uid in UIDS}
    for seed in seeds:
        source = errors_by_seed.get(seed, errors_by_seed.get(int(seed)))
        if source is None:
            raise ValueError(f"missing fixed errors for seed {seed}")
        fixed_errors[seed] = {}
        for uid in UIDS:
            vector = _array(source[uid], (6,))
            fixed_errors[seed][uid] = vector.tolist()
            rows.append(vector)
            per_uid[uid].append(vector)
    realized = np.asarray(rows, dtype=float)
    position_samples = realized[:, :3]
    velocity_samples = realized[:, 3:]
    # Six UID/seed rows are the descriptive sample; the two-family count is
    # kept explicit so it cannot be mistaken for a large random sample.
    covariance_ddof1 = np.cov(realized, rowvar=False, ddof=1)
    per_uid_report: dict[str, Any] = {}
    for uid, values in per_uid.items():
        sample = np.asarray(values, dtype=float)
        per_uid_report[uid] = {
            "seed_family_count": int(sample.shape[0]),
            "position_mean": np.mean(sample[:, :3], axis=0).tolist(),
            "velocity_mean": np.mean(sample[:, 3:], axis=0).tolist(),
            "position_covariance_ddof1": np.cov(sample[:, :3], rowvar=False, ddof=1).tolist(),
            "velocity_covariance_ddof1": np.cov(sample[:, 3:], rowvar=False, ddof=1).tolist(),
        }
    per_seed_report: dict[str, Any] = {}
    for seed in seeds:
        sample = np.asarray([fixed_errors[seed][uid] for uid in UIDS], dtype=float)
        per_seed_report[seed] = {
            "uid_count": int(sample.shape[0]),
            "position_mean": np.mean(sample[:, :3], axis=0).tolist(),
            "velocity_mean": np.mean(sample[:, 3:], axis=0).tolist(),
            "position_covariance_ddof1": np.cov(sample[:, :3], rowvar=False, ddof=1).tolist(),
            "velocity_covariance_ddof1": np.cov(sample[:, 3:], rowvar=False, ddof=1).tolist(),
        }

    prior18 = np.zeros((18, 18), dtype=float)
    for i in range(3):
        for j in range(3):
            si, sj = _state_index(i), _state_index(j)
            block = np.zeros((6, 6), dtype=float)
            block[:3, :3] = marginal_p if i == j else cross_p
            if i == j:
                block[3:, 3:] = marginal_v
            prior18[si, sj] = block
    position_prior = prior18[np.ix_([0, 1, 2, 6, 7, 8, 12, 13, 14], [0, 1, 2, 6, 7, 8, 12, 13, 14])]
    velocity_prior = prior18[np.ix_([3, 4, 5, 9, 10, 11, 15, 16, 17], [3, 4, 5, 9, 10, 11, 15, 16, 17])]
    # Explicit 18 x 21 sampling factor: one shared 3-D position component,
    # one private 3-D position component per UID, and one private 3-D
    # velocity component per UID.  This is a validation witness for the law,
    # not a redraw mechanism.
    factor = np.zeros((18, 21), dtype=float)
    shared_scale = math.sqrt(2.0)
    for i in range(3):
        state = _state_index(i)
        factor[state.start : state.start + 3, :3] = shared_scale * np.eye(3)
        factor[state.start : state.start + 3, 3 + 3 * i : 3 + 3 * i + 3] = shared_scale * np.eye(3)
        factor[state.start + 3 : state.stop, 12 + 3 * i : 12 + 3 * i + 3] = np.eye(3)
    factor_covariance = _sym(factor @ factor.T)
    return {
        "schema": "cooperative_ekf_reliability_reference_v5_initial_prior",
        "validated": True,
        "sampling_law": {
            "state_dimension": 18,
            "per_uid_dimension": 6,
            "uid_order": list(UIDS),
            "component_order": order,
            "position_marginal": marginal_p.tolist(),
            "velocity_marginal": marginal_v.tolist(),
            "cross_position_covariance": cross_p.tolist(),
            "cross_velocity_covariance": np.zeros((3, 3)).tolist(),
            "position_velocity_cross_covariance": np.zeros((3, 3)).tolist(),
            "factor_dimension": 21,
            "factor_matrix_18x21": factor.tolist(),
            "factor_reconstructs_prior_max_abs": float(np.max(np.abs(factor_covariance - prior18))),
            "factor_reconstructs_prior": bool(np.allclose(factor_covariance, prior18, atol=1e-12)),
        },
        "prior_covariance_18x18": prior18.tolist(),
        "position_prior_covariance_9x9": position_prior.tolist(),
        "velocity_prior_covariance_9x9": velocity_prior.tolist(),
        "fixed_errors_by_seed": fixed_errors,
        "errors_match_locked_config": True,
        "realized_moments": {
            "seed_family_count": 2,
            "uid_seed_row_count": int(realized.shape[0]),
            "position_mean": np.mean(position_samples, axis=0).tolist(),
            "velocity_mean": np.mean(velocity_samples, axis=0).tolist(),
            "covariance_ddof1": covariance_ddof1.tolist(),
            "position_covariance_ddof1": np.cov(position_samples, rowvar=False, ddof=1).tolist(),
            "velocity_covariance_ddof1": np.cov(velocity_samples, rowvar=False, ddof=1).tolist(),
            "descriptive_only": True,
            "no_redraw": True,
        },
        "per_uid_realized_moments": per_uid_report,
        "per_seed_realized_moments": per_seed_report,
        "attitude_error_injected": False,
        "bias_error_injected": False,
        "full15_nees_claim": False,
        "coverage_claim": None,
    }


def joint_reference_rows(config: Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    """Return static 9-D joint Gaussian reference rows for B/C combinations."""

    cfg = _config(config)
    correlations = [float(x) for x in cfg.get("offline_joint_reference_correlations", (0.0, 0.5, 1.0))]
    rows: list[dict[str, Any]] = []
    common_map = np.kron(np.ones((3, 1)) / math.sqrt(3.0), np.eye(3))
    seeds = [str(seed) for seed in cfg.get("seeds", cfg.get("initial_errors_by_seed", {}).keys())]
    for case in _case_rows(cfg):
        for seed in seeds:
            mean = _initial_mean_position(case, cfg, seed)
            for correlation in correlations:
                prior = _position_prior(cfg, correlation)
                common_before = common_map.T @ prior @ common_map
                for measurement_case in ("B", "C", "BCstack", "BBtwo"):
                    jacobian = _measurement_matrix(measurement_case, mean)
                    innovation = jacobian @ prior @ jacobian.T + RANGE_VARIANCE_M2 * np.eye(jacobian.shape[0])
                    posterior = _sym(prior - prior @ jacobian.T @ np.linalg.pinv(innovation) @ jacobian @ prior)
                    delta = _sym(posterior - prior)
                    relative_01 = _measurement_row(1, mean)
                    relative_02 = _measurement_row(2, mean)
                    common_after = common_map.T @ posterior @ common_map
                    rows.append(
                        {
                        "schema": "cooperative_ekf_reliability_reference_v5_joint",
                        "scene_id": case["scene_id"],
                        "seed": int(seed),
                        "initial_mean_position": mean.tolist(),
                        "correlation": correlation,
                        "measurement_case": measurement_case,
                        "measurement_noise_variance_m2": RANGE_VARIANCE_M2,
                        "jacobian": jacobian.tolist(),
                        "prior_covariance_9x9": prior.tolist(),
                        "posterior_covariance_9x9": posterior.tolist(),
                        "delta_covariance_9x9": delta.tolist(),
                        "prior_eigenvalues": _eig(prior).tolist(),
                        "posterior_eigenvalues": _eig(posterior).tolist(),
                        "delta_min_eigenvalue": float(np.min(_eig(delta))),
                        "prior_psd_min_eigenvalue": float(np.min(_eig(prior))),
                        "posterior_psd_min_eigenvalue": float(np.min(_eig(posterior))),
                        "own_uav_00_position_trace_before_m2": float(np.trace(prior[:3, :3])),
                        "own_uav_00_position_trace_after_m2": float(np.trace(posterior[:3, :3])),
                        "own_uav_00_position_eigenvalues_before": _eig(prior[:3, :3]).tolist(),
                        "own_uav_00_position_eigenvalues_after": _eig(posterior[:3, :3]).tolist(),
                        "relative_covariance_trace_uav00_uav01_m2": float(np.trace(_relative_selector(1) @ prior @ _relative_selector(1).T)),
                        "relative_covariance_trace_uav00_uav02_m2": float(np.trace(_relative_selector(2) @ prior @ _relative_selector(2).T)),
                        "relative_covariance_trace_uav00_uav01_after_m2": float(np.trace(_relative_selector(1) @ posterior @ _relative_selector(1).T)),
                        "relative_covariance_trace_uav00_uav02_after_m2": float(np.trace(_relative_selector(2) @ posterior @ _relative_selector(2).T)),
                        "relative_radial_variance_uav00_uav01_before_m2": float(relative_01 @ prior @ relative_01.T),
                        "relative_radial_variance_uav00_uav02_before_m2": float(relative_02 @ prior @ relative_02.T),
                        "relative_radial_variance_uav00_uav01_after_m2": float(relative_01 @ posterior @ relative_01.T),
                        "relative_radial_variance_uav00_uav02_after_m2": float(relative_02 @ posterior @ relative_02.T),
                        "joint_trace_before_m2": float(np.trace(prior)),
                        "joint_trace_after_m2": float(np.trace(posterior)),
                        "common_translation_covariance_before_m2": common_before.tolist(),
                        "common_translation_covariance_after_m2": common_after.tolist(),
                        "common_translation_gauge_change_max_abs_m2": float(np.max(np.abs(common_after - common_before))),
                        "jacobian_rank": _rank(jacobian),
                        "gram_rank": _rank(jacobian.T @ jacobian),
                        "trace_delta_m2": float(np.trace(delta)),
                        "logdet_prior_restricted_positive": _restricted_logdet(prior),
                        "logdet_posterior_restricted_positive": _restricted_logdet(posterior),
                        "logdet_gain_restricted_positive": _restricted_logdet_gain(prior, posterior),
                        "hypothetical_static_joint_reference": True,
                        "period_bypass": False,
                        "hypothetical_not_runtime_samples": True,
                        "runtime_covariance_rule": False,
                        }
                    )
    grouped: dict[tuple[str, int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scene_id"], row["seed"], row["correlation"]), {})[row["measurement_case"]] = row
    for group in grouped.values():
        first_b = group["B"]["own_uav_00_position_trace_after_m2"]
        for second_case in ("BCstack", "BBtwo"):
            second = group[second_case]["own_uav_00_position_trace_after_m2"]
            gain = _safe_divide(first_b - second, first_b)
            group[second_case]["conditional_incremental_after_B_own_trace_gain_fraction"] = gain
        group["B"]["conditional_after_B_second_C_own_trace_gain_fraction"] = group["BCstack"]["conditional_incremental_after_B_own_trace_gain_fraction"]
        group["B"]["conditional_after_B_second_B_own_trace_gain_fraction"] = group["BBtwo"]["conditional_incremental_after_B_own_trace_gain_fraction"]
    return rows


def _safe_divide(numerator: float, denominator: float, zero_value: float = 0.0) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-15 else float(zero_value)


def fixed_gain_covariance_bound(
    prior: float | np.ndarray,
    jacobian: float | np.ndarray,
    gain: float | np.ndarray,
    neighbor_projected_variance: float,
    range_variance: float,
) -> dict[str, Any]:
    """Apply a Young inequality covariance upper bound for a fixed gain.

    ``prior`` may be a scalar or a square covariance matrix.  ``jacobian`` is
    one measurement row and ``gain`` is its corresponding column.  The
    returned ``bound`` is a scalar for the scalar case and a list matrix for a
    vector case; ``bound_matrix`` is always included for callers that want a
    uniform representation.
    """

    p = np.asarray(prior, dtype=float)
    scalar = p.ndim == 0
    if scalar:
        p = p.reshape(1, 1)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("prior must be scalar or square")
    n = p.shape[0]
    h = np.asarray(jacobian, dtype=float)
    h = h.reshape(1, 1) if h.ndim == 0 else h.reshape(1, -1)
    k = np.asarray(gain, dtype=float)
    k = k.reshape(1, 1) if k.ndim == 0 else k.reshape(-1, 1)
    if h.shape != (1, n) or k.shape != (n, 1):
        raise ValueError(f"one-measurement dimensions required: H={h.shape}, K={k.shape}, n={n}")
    if neighbor_projected_variance < 0.0 or range_variance < 0.0:
        raise ValueError("variances must be non-negative")
    m = np.eye(n) - k @ h
    a_matrix = _sym(m @ p @ m.T)
    # Explicit parentheses keep the projected source variance scalar.
    b_matrix = _sym(float(neighbor_projected_variance) * (k @ k.T))
    noise_matrix = _sym(float(range_variance) * (k @ k.T))
    trace_a = max(0.0, float(np.trace(a_matrix)))
    trace_b = max(0.0, float(np.trace(b_matrix)))
    root_a, root_b = math.sqrt(trace_a), math.sqrt(trace_b)
    if root_a == 0.0 and root_b == 0.0:
        weight = 0.5
        bound_matrix = noise_matrix.copy()
    elif root_a == 0.0:
        weight = 1.0
        bound_matrix = b_matrix + noise_matrix
    elif root_b == 0.0:
        weight = 0.0
        bound_matrix = a_matrix + noise_matrix
    else:
        weight = root_b / (root_a + root_b)
        bound_matrix = a_matrix / (1.0 - weight) + b_matrix / weight + noise_matrix
    bound_matrix = _sym(bound_matrix)
    output: dict[str, Any] = {
        "bound": float(bound_matrix[0, 0]) if scalar else bound_matrix.tolist(),
        "bound_matrix": bound_matrix.tolist(),
        "weight": float(weight),
        "A": a_matrix.tolist(),
        "B": b_matrix.tolist(),
        "noise_term": noise_matrix.tolist(),
        "trace_A": trace_a,
        "trace_B": trace_b,
        "trace_bound": float(np.trace(bound_matrix)),
        "degenerate_zero_trace_handled": bool(root_a == 0.0 or root_b == 0.0),
    }
    return output


def young_bound_rows(config: Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    """Evaluate the scalar and 3-D fixed-gain Young-bound grids."""

    cfg = _config(config)
    own_grid = [float(x) for x in cfg.get("young_bound_variance_grid", (1.0, 4.0, 9.0))]
    corr_grid = [float(x) for x in cfg.get("young_bound_correlation_grid", (-1.0, -0.5, 0.0, 0.5, 1.0))]
    rows: list[dict[str, Any]] = []
    unit = np.asarray([[1.0, 0.0, 0.0]])
    for own in own_grid:
        for source in own_grid:
            for correlation in corr_grid:
                root = math.sqrt(own * source)
                bound_variance = own + source + 2.0 * root + RANGE_VARIANCE_M2
                gain = own / bound_variance
                actual = (1.0 - gain) ** 2 * own + gain**2 * source + 2.0 * gain * (1.0 - gain) * correlation * root + gain**2 * RANGE_VARIANCE_M2
                restored = own
                fake = own - own**2 / bound_variance
                helper = fixed_gain_covariance_bound(own, 1.0, gain, source, RANGE_VARIANCE_M2)
                upper = float(helper["bound"])
                rows.append(
                    {
                        "schema": "cooperative_ekf_reliability_reference_v5_young",
                        "grid_kind": "scalar",
                        "own_variance": own,
                        "source_variance": source,
                        "correlation": correlation,
                        "range_variance": RANGE_VARIANCE_M2,
                        "bound_innovation_variance": bound_variance,
                        "current_gain": gain,
                        "actual_fixed_gain_posterior_variance": actual,
                        "restored_prior_variance": restored,
                        "linear_fake_posterior_variance": fake,
                        "young_upper_bound_variance": upper,
                        "young_weight": helper["weight"],
                        "actual_minus_restored_gap": actual - restored,
                        "upper_minus_actual_gap": upper - actual,
                        "bound_ge_actual": bool(upper >= actual - 1e-10),
                        "restored_prior_is_conservative": bool(restored >= actual - 1e-10),
                        "joint_scalar_covariance_eigenvalues": np.linalg.eigvalsh(np.asarray([[own, correlation * root], [correlation * root, source]])).tolist(),
                    }
                )

                p = own * np.eye(3)
                q = source * np.eye(3)
                cross = correlation * root * np.eye(3)
                h = unit
                k = gain * unit.T
                m = np.eye(3) - k @ h
                # The source error enters the residual with a positive error
                # update term after the own-minus-source measurement Jacobian.
                source_projected = float((h @ q @ h.T).item())
                actual_cov = _sym(m @ p @ m.T + source_projected * (k @ k.T) + m @ cross @ h.T @ k.T + k @ h @ cross @ m.T + RANGE_VARIANCE_M2 * (k @ k.T))
                helper_matrix = fixed_gain_covariance_bound(p, h, k, source, RANGE_VARIANCE_M2)
                upper_matrix = np.asarray(helper_matrix["bound_matrix"], dtype=float)
                fake_matrix = _sym(p - (p @ h.T @ h @ p) / bound_variance)
                residual = _sym(upper_matrix - actual_cov)
                prior_residual = _sym(p - actual_cov)
                rows.append(
                    {
                        "schema": "cooperative_ekf_reliability_reference_v5_young",
                        "grid_kind": "matrix3d",
                        "own_variance": own,
                        "source_variance": source,
                        "correlation": correlation,
                        "range_variance": RANGE_VARIANCE_M2,
                        "bound_innovation_variance": bound_variance,
                        "current_gain": k.tolist(),
                        "M": m.tolist(),
                        "actual_fixed_gain_posterior_covariance": actual_cov.tolist(),
                        "restored_prior_covariance": p.tolist(),
                        "linear_fake_posterior_covariance": fake_matrix.tolist(),
                        "young_upper_bound_covariance": upper_matrix.tolist(),
                        "young_weight": helper_matrix["weight"],
                        "actual_minus_restored_min_eigenvalue": float(np.min(_eig(actual_cov - p))),
                        "upper_minus_actual_min_eigenvalue": float(np.min(_eig(residual))),
                        "bound_ge_actual_psd": bool(np.min(_eig(residual)) >= -1e-10),
                        "restored_prior_is_conservative_psd": bool(np.min(_eig(prior_residual)) >= -1e-10),
                        "joint_6d_covariance_eigenvalues": np.linalg.eigvalsh(np.block([[p, cross], [cross, q]])).tolist(),
                        "cross_term_sign": "positive_source_error_update",
                    }
                )
    return rows


def _pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa, ya = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if xa.size < 2 or ya.size != xa.size or np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _native_mask(observation: Mapping[str, Any]) -> list[bool]:
    own = str(observation.get("uav_id", ""))
    budget = _float(observation.get("measurement_budget_remaining")) > 0.0
    eligibility = [bool(x) for x in observation.get("measurement_update_eligibility", ())]
    if len(eligibility) != 3:
        return []
    messages = observation.get("neighbor_messages", {})
    candidates = set(observation.get("candidate_measurement_sources", ()))
    effective = set(observation.get("effective_sources", ()))
    uwb = observation.get("uwb_available", {})
    base_mask = observation.get("native_base_action_mask", observation.get("native_admissible_mask"))
    if isinstance(base_mask, (list, tuple)) and len(base_mask) == 4:
        base_mask = [bool(x) for x in base_mask]
    else:
        base_mask = [True] * 4
    mask = [False] * 4
    for i, source in enumerate(UIDS):
        if source == own or not budget or not base_mask[i] or not eligibility[i] or source not in candidates or source not in effective or not bool(uwb.get(source, False)):
            continue
        message = messages.get(source) if isinstance(messages, Mapping) else None
        try:
            estimate = _array(message["estimate"], (15,))
            covariance = _array(message["covariance"], (15, 15))
            if np.allclose(covariance, covariance.T, atol=1e-10) and float(np.min(_eig(covariance))) >= -1e-9:
                mask[i] = True
        except (TypeError, KeyError, ValueError, np.linalg.LinAlgError):
            continue
    mask[3] = True
    return mask


def _expected_credit(observation: Mapping[str, Any], source: str) -> float:
    try:
        own = _array(observation["estimate"], (15,))
        message = observation["neighbor_messages"][source]
        neighbor = _array(message["estimate"], (15,))
        own_cov = _array(observation["covariance"], (15, 15))[:3, :3]
        neighbor_cov = _array(message["covariance"], (15, 15))[:3, :3]
        direction = neighbor[:3] - own[:3]
        norm = np.linalg.norm(direction)
        if norm <= 1e-15:
            return 0.0
        direction = direction / norm
        local = float(direction @ own_cov @ direction)
        source_var = float(direction @ neighbor_cov @ direction)
        denominator = local + RANGE_VARIANCE_M2 + source_var + 2.0 * math.sqrt(max(local, 0.0) * max(source_var, 0.0))
        numerator = float(direction @ own_cov @ own_cov @ direction)
        return float(np.clip((numerator / denominator) / max(float(np.trace(own_cov)), 1e-15), 0.0, 1.0))
    except (TypeError, KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return 0.0


def audit_v4(project_root: str | Path, config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Audit the frozen v4 observation bank and CSVs without writing files."""

    cfg = _config(config)
    root = Path(project_root)
    stem = Path(str(cfg.get("base_config", V4_STEM))).stem
    data_dir = root / "data" / "synthetic" / stem
    log_dir = root / "logs" / "preliminary" / stem / str(cfg.get("source_calibration", "calibration_v4_s2026"))
    bank_path = data_dir / "calibration_v4_s2026" / "observation_bank.json"
    if not bank_path.exists():
        bank_path = data_dir / str(cfg.get("source_calibration", "calibration_v4_s2026")) / "observation_bank.json"
    if not bank_path.exists():
        raise FileNotFoundError(bank_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    actions = _read_csv(log_dir / "actions.csv")
    candidates = _read_csv(log_dir / "candidate_values.csv")
    events = _read_csv(log_dir / "fusion_events.csv")
    localizations = _read_csv(log_dir / "localization_diagnostics.csv")
    by_frame = {(str(frame.get("scene_id")), int(frame.get("step", -1))): frame for frame in bank}

    dual = [row for row in candidates if int(_float(row.get("legal_peer_count"), 0)) >= 2]
    dual_keys = {(row.get("scene_id", ""), int(_float(row.get("step"), -1)), row.get("uav_id", "")) for row in dual}
    spread_rows = [row for row in dual if _float(row.get("expected_credit_spread")) > 1e-6]
    pre_checks = 0
    mask_mismatches: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for action in actions:
        step = int(_float(action.get("step"), -1))
        # The bank stores the state before decision step t at frame t-1.
        if step < 1 or step > 120:
            continue
        frame = by_frame.get((action.get("scene_id", ""), step - 1))
        if frame is None:
            continue
        pre_checks += 1
        uid = action.get("uav_id", "")
        observation = frame.get("observations", {}).get(uid)
        if not isinstance(observation, Mapping):
            continue
        expected = _native_mask({**observation, "uav_id": uid})
        actual = [bool(x) for x in _json_field(action.get("native_action_mask", "[]"))]
        if expected != actual:
            mask_mismatches.append({"scene_id": frame.get("scene_id"), "step": step, "uav_id": uid, "expected": expected, "actual": actual})
        if (action.get("scene_id", ""), step, uid) not in dual_keys:
            continue
        legal = [source for source in UIDS if source != uid and _native_mask({**observation, "uav_id": uid})[UIDS.index(source)]]
        vectors: list[np.ndarray] = []
        for source in legal:
            msg = observation.get("neighbor_messages", {}).get(source, {})
            try:
                own = _array(observation["estimate"], (15,))[:3]
                peer = _array(msg["estimate"], (15,))[:3]
                d = peer - own
                d = d / max(np.linalg.norm(d), 1e-15)
                vectors.append(d)
            except (TypeError, KeyError, ValueError):
                pass
        if len(vectors) >= 2:
            gram = np.asarray(vectors) @ np.asarray(vectors).T
            direction_rows.append({"scene_id": frame.get("scene_id"), "step": step, "uav_id": uid, "direction_gram_eigenvalues": np.linalg.eigvalsh(_sym(gram)).tolist(), "direction_rank": _rank(np.asarray(vectors)), "direction_score": float(np.linalg.det(np.asarray(gram)))})

    candidate_compare: list[dict[str, Any]] = []
    for row in dual:
        step = int(_float(row.get("step"), -1))
        frame = by_frame.get((row.get("scene_id", ""), step - 1))
        if frame is None:
            continue
        obs = frame.get("observations", {}).get(row.get("uav_id", ""))
        if not isinstance(obs, Mapping):
            continue
        legal = [source for source in UIDS if source != row.get("uav_id") and _native_mask({**obs, "uav_id": row.get("uav_id")})[UIDS.index(source)]]
        values = [_expected_credit(obs, source) for source in legal]
        computed_spread = max(values, default=0.0) - min(values, default=0.0)
        candidate_compare.append({"scene_id": row.get("scene_id"), "step": step, "uav_id": row.get("uav_id"), "logged_spread": _float(row.get("expected_credit_spread")), "computed_spread": computed_spread, "exact": abs(computed_spread - _float(row.get("expected_credit_spread"))) <= 1e-10})

    isotropy_values: list[float] = []
    covariance_count = 0
    for frame in bank:
        for obs in frame.get("observations", {}).values():
            try:
                cov = _array(obs["covariance"], (15, 15))[:3, :3]
                eig = _eig(cov)
                isotropy_values.append(float(np.max(eig) / max(np.min(eig), 1e-15)))
                covariance_count += 1
            except (TypeError, KeyError, ValueError, np.linalg.LinAlgError):
                pass

    accepted = [row for row in events if str(row.get("accepted", "")).lower() == "true"]
    raw_trace_proxy: list[float] = []
    final_gains: list[float] = []
    error_improvements: list[float] = []
    for row in accepted:
        before = _float(row.get("position_covariance_trace_before_m2"))
        middle = _float(row.get("position_covariance_trace_after_linear_m2"), before)
        final = _float(row.get("position_covariance_trace_after_final_m2"), before)
        raw_trace_proxy.append(_safe_divide(before - middle, before))
        final_gains.append(before - final)
        error_improvements.append(_float(row.get("true_position_error_before_m")) - _float(row.get("true_position_error_after_m")))
    action_raw = [_float(row.get("raw_relative_credit_proxy")) for row in actions if math.isfinite(_float(row.get("raw_relative_credit_proxy"), math.nan))]
    configured_scale = _float(cfg.get("credit_scale"), 1.0)
    scaled_comparison = [
        _float(row.get("scaled_credit_proxy"))
        - float(np.clip(_float(row.get("raw_relative_credit_proxy")) / max(_float(row.get("credit_scale_used"), configured_scale), 1e-15), 0.0, 1.0))
        for row in actions
    ]
    selected_age = [_float(row.get("message_age_s"), math.nan) for row in accepted if math.isfinite(_float(row.get("message_age_s"), math.nan))]

    return {
        "schema": "cooperative_ekf_reliability_reference_v5_v4_audit",
        "offline_only": True,
        "optimizer_updates": 0,
        "environment_steps": 0,
        "input_observation_bank": str(bank_path.relative_to(root)),
        "raw_decision_count": len(actions),
        "pre_decision_frames_checked_steps_1_to_119": pre_checks,
        "mask_audit": {"checked_decision_rows": pre_checks, "checked_observations": pre_checks, "mismatch_count": len(mask_mismatches), "mismatches": mask_mismatches, "compared_field": "actions.native_action_mask"},
        "decision_counts": {"candidate_rows": len(candidates), "dual_peer_rows": len(dual), "spread_gt_1e-6_rows": len(spread_rows), "fusion_event_rows": len(events), "accepted_event_rows": len(accepted), "localization_rows": len(localizations)},
        "dual_peer_csv_comparison": {"rows_compared": len(candidate_compare), "exact_match_count": sum(int(row["exact"]) for row in candidate_compare), "all_exact": all(row["exact"] for row in candidate_compare), "rows": candidate_compare},
        "geometry": {"rows_with_two_directions": len(direction_rows), "dual_legal_scope": len(dual), "direction_rows": direction_rows, "pre_decision_only": True},
        "covariance_isotropy": {"reported_position_covariance_count": covariance_count, "mean_max_over_min_eigenvalue": float(np.mean(isotropy_values)) if isotropy_values else None, "p95_max_over_min_eigenvalue": float(np.percentile(isotropy_values, 95)) if isotropy_values else None},
        "error_proxy_correlation": {"accepted_rows": len(accepted), "trace_proxy_vs_true_error_improvement_pearson": _pearson(raw_trace_proxy, error_improvements), "diagnostic_truth_used_as_reward": False, "actor_truth_available": False},
        "reward_trace": {"accepted_relative_credit_from_trace_before_minus_middle": raw_trace_proxy, "logged_linear_relative_credit_proxy": [_float(row.get("linear_relative_credit_proxy")) for row in accepted], "max_trace_proxy_log_abs_error": float(max((abs(a - _float(row.get("linear_relative_credit_proxy"))) for a, row in zip(raw_trace_proxy, accepted)), default=0.0)), "final_covariance_gain_values_m2": final_gains, "final_covariance_gain_max_abs_m2": float(max((abs(value) for value in final_gains), default=0.0)), "scaled_minus_clipped_raw_div_scale_max_abs": float(max((abs(value) for value in scaled_comparison), default=0.0)), "raw_credit_p95_before_scaling": float(np.percentile(action_raw, 95)) if action_raw else None, "credit_scale_configured": configured_scale, "communication_cost_codepath": {"measurement_cost_coef": 0.05, "actual_overhead_equals_512_byte_packet": all(abs(_float(row.get("communication_cost")) - 0.05 * _float(row.get("actual_overhead_packets"))) <= 1e-12 for row in actions), "wasted_budget_units_invented_cost": False}},
        "nis_diagnostic": {"accepted_rows": len(accepted), "native_nis_values": [_float(row.get("nis")) for row in accepted], "unknown_cross_bound_values_m2": [_float(row.get("unknown_cross_bound_m2")) for row in accepted], "unknown_cross_is_not_labeled_proof": True},
        "neighbor_belief_age_s": None,
        "delivered_source_covariance_propagation": {"selected_fusion_message_age_values_s": selected_age, "unselected_or_bank_age": None},
        "truth_and_reward_guard": {"actor_truth_forbidden": True, "true_error_fields_audit_only": True, "true_error_used_for_training_reward": False},
        "claim_limitation": "Historical v4 logs and frozen observation-bank geometry are audited descriptively; this does not establish a runtime covariance guarantee, policy learnability, or true actual-trajectory RMSE claim.",
    }


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing reference output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/configs/three_uav_cooperative_ekf_reliability_diag_v5_seed2026_20261012.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", required=True, help="safe output run identifier")
    args = parser.parse_args(argv)
    if not args.run_id.isascii() or not args.run_id.replace("_", "").replace("-", "").isalnum() or not args.run_id[0].isalnum() or len(args.run_id) > 64:
        raise ValueError("unsafe run-id")
    report = {"initial_prior": initial_prior_audit(args.config), "joint_reference_rows": joint_reference_rows(args.config), "young_bound_rows": young_bound_rows(args.config), "v4_audit": audit_v4(args.project_root, args.config)}
    cfg = _config(args.config)
    stem = str(cfg.get("experiment_name", "cooperative_ekf_reliability_diag_v5"))
    log_path = args.project_root / "logs" / "preliminary" / stem / args.run_id / "reference_report.json"
    data_path = args.project_root / "data" / "synthetic" / stem / args.run_id / "reference_report.json"
    if log_path.exists() or data_path.exists():
        raise FileExistsError(f"refusing to reuse output run: {args.run_id}")
    _write_json(log_path, report)
    _write_json(data_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
