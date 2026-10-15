"""Pure checks for the v5 cooperative-EKF reference calculations.

These tests read the locked configuration and frozen v4 files only.  They do
not construct an environment or run a policy.
"""

from pathlib import Path

import numpy as np

from src.analysis.cooperative_ekf_reliability_reference_v5 import (
    audit_v4,
    fixed_gain_covariance_bound,
    initial_prior_audit,
    joint_reference_rows,
    young_bound_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "configs" / "three_uav_cooperative_ekf_reliability_diag_v5_seed2026_20261012.json"


def test_initial_prior_audit_validates_locked_18d_law_without_redraw() -> None:
    report = initial_prior_audit(CONFIG)
    assert report["validated"] is True
    assert report["sampling_law"]["state_dimension"] == 18
    assert report["sampling_law"]["per_uid_dimension"] == 6
    assert report["sampling_law"]["component_order"][:3] == ["position_x_m", "position_y_m", "position_z_m"]
    assert np.allclose(np.asarray(report["position_prior_covariance_9x9"])[:3, :3], 4.0 * np.eye(3))
    assert np.allclose(np.asarray(report["position_prior_covariance_9x9"])[:3, 3:6], 2.0 * np.eye(3))
    assert report["sampling_law"]["factor_dimension"] == 21
    assert report["sampling_law"]["factor_reconstructs_prior"] is True
    assert report["realized_moments"]["seed_family_count"] == 2
    assert report["realized_moments"]["descriptive_only"] is True
    assert report["realized_moments"]["no_redraw"] is True
    assert report["attitude_error_injected"] is False
    assert report["bias_error_injected"] is False
    assert report["full15_nees_claim"] is False


def test_joint_reference_has_scene_seed_correlation_and_geometry_rows() -> None:
    rows = joint_reference_rows(CONFIG)
    assert len(rows) == 2 * 2 * 3 * 4
    assert {row["measurement_case"] for row in rows} == {"B", "C", "BCstack", "BBtwo"}
    assert {row["correlation"] for row in rows} == {0.0, 0.5, 1.0}
    assert len({(row["scene_id"], row["seed"]) for row in rows}) == 4

    b = next(row for row in rows if row["measurement_case"] == "B" and row["correlation"] == 0.5)
    c = next(row for row in rows if row["measurement_case"] == "C" and row["correlation"] == 0.5)
    bc = next(row for row in rows if row["measurement_case"] == "BCstack" and row["correlation"] == 0.5)
    bb = next(row for row in rows if row["measurement_case"] == "BBtwo" and row["correlation"] == 0.5)
    assert np.asarray(b["jacobian"]).shape == (1, 9)
    assert np.asarray(bc["jacobian"]).shape == (2, 9)
    assert np.asarray(bb["jacobian"]).shape == (2, 9)
    assert b["jacobian"] == bb["jacobian"][0:1]
    assert bb["jacobian"][0] == bb["jacobian"][1]
    assert b["jacobian_rank"] == 1
    assert bc["jacobian_rank"] == 2
    assert bb["gram_rank"] == 1
    assert b["period_bypass"] is False
    assert b["hypothetical_not_runtime_samples"] is True
    assert np.isclose(np.linalg.norm(np.asarray(b["jacobian"])), np.sqrt(2.0))
    assert np.isclose(b["own_uav_00_position_trace_after_m2"], 11.001597444089457)
    assert b["conditional_after_B_second_C_own_trace_gain_fraction"] >= b["conditional_after_B_second_B_own_trace_gain_fraction"]
    assert abs(b["common_translation_gauge_change_max_abs_m2"]) <= 1e-10
    assert np.min(np.asarray(b["posterior_eigenvalues"])) >= -1e-9


def test_fixed_gain_bound_and_young_grid_are_psd_safe() -> None:
    helper = fixed_gain_covariance_bound(1.0, 1.0, 1.0 / 9.0064, 4.0, 0.0064)
    assert helper["weight"] == np.sqrt(helper["trace_B"]) / (np.sqrt(helper["trace_A"]) + np.sqrt(helper["trace_B"]))
    assert helper["bound"] > 1.0
    assert helper["bound"] >= 1.2344713493

    rows = young_bound_rows(CONFIG)
    assert len(rows) == 2 * 3 * 3 * 5
    scalar = [row for row in rows if row["grid_kind"] == "scalar"]
    matrix = [row for row in rows if row["grid_kind"] == "matrix3d"]
    assert len(scalar) == len(matrix) == 45
    assert all(row["bound_ge_actual"] for row in scalar)
    assert all(row["bound_ge_actual_psd"] for row in matrix)
    endpoint = next(row for row in scalar if row["own_variance"] == 1.0 and row["source_variance"] == 4.0 and row["correlation"] == 1.0)
    assert endpoint["actual_fixed_gain_posterior_variance"] > endpoint["restored_prior_variance"]
    assert np.isclose(endpoint["actual_fixed_gain_posterior_variance"], 1.2344713493408577)
    matrix_endpoint = next(row for row in matrix if row["own_variance"] == 1.0 and row["source_variance"] == 4.0 and row["correlation"] == 1.0)
    assert matrix_endpoint["restored_prior_is_conservative_psd"] is False
    assert np.min(np.asarray(matrix_endpoint["joint_6d_covariance_eigenvalues"])) >= -1e-10


def test_v4_audit_preserves_frozen_counts_and_truth_boundaries() -> None:
    report = audit_v4(ROOT, CONFIG)
    assert report["offline_only"] is True
    assert report["optimizer_updates"] == 0
    assert report["environment_steps"] == 0
    assert report["raw_decision_count"] == 1440
    assert report["decision_counts"]["dual_peer_rows"] == 331
    assert report["decision_counts"]["spread_gt_1e-6_rows"] == 245
    assert report["decision_counts"]["fusion_event_rows"] == 342
    assert report["dual_peer_csv_comparison"]["all_exact"] is True
    assert report["geometry"]["rows_with_two_directions"] == 331
    assert report["geometry"]["dual_legal_scope"] == 331
    assert report["mask_audit"]["mismatch_count"] == 0
    assert report["error_proxy_correlation"]["diagnostic_truth_used_as_reward"] is False
    assert report["truth_and_reward_guard"]["actor_truth_forbidden"] is True
    assert report["neighbor_belief_age_s"] is None
    assert report["delivered_source_covariance_propagation"]["unselected_or_bank_age"] is None
    assert report["reward_trace"]["final_covariance_gain_max_abs_m2"] <= 1e-10
    assert report["reward_trace"]["max_trace_proxy_log_abs_error"] <= 1e-12
    assert report["reward_trace"]["scaled_minus_clipped_raw_div_scale_max_abs"] <= 1e-12
    assert report["reward_trace"]["communication_cost_codepath"]["actual_overhead_equals_512_byte_packet"] is True
