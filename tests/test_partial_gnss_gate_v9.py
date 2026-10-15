import copy
import pytest

from src.analysis.partial_gnss_gate_v9 import admission


def fixture():
    spec = {"cases": [{"scene_id": s, "seed": n} for s in ("col", "noncol") for n in range(4)],
            "gate": {"positive_seeds_per_geometry": 3, "pooled_rmse_improvement_fraction": .05,
                     "maximum_case_degradation_fraction": .05, "reference_rmse_limit_m": 1.,
                     "reference_gnss_acceptance_min": .9, "covariance_full_trace_limit": 1e4,
                     "target_nees_mean_limit": 7.815, "effective_fusion_rate_min": .2}}
    rows = []
    for c in spec["cases"]:
        for policy in ("none", "random_peer"):
            rows.append({**c, "method": "measurement_ci", "policy": policy, "steps": 120, "status": "complete",
                "uav2_absolute_rmse_m": 2. if policy == "none" else 1.8,
                "cluster_relative_rmse_m": 1. if policy == "none" else .9,
                "uav2_position_nees_mean": 2., "covariance_finite": True,
                "covariance_min_eigenvalue": .01, "full_covariance_trace_max": 20.,
                "collision_count": 0, "boundary_projections": 0, "safety_projections": 0,
                "uav0_rmse_m": .2, "uav1_rmse_m": .2,
                "uav0_gnss_accepted": 119, "uav1_gnss_accepted": 119,
                "uav0_gnss_valid": 120, "uav1_gnss_valid": 120,
                "uwb_requested": 0 if policy == "none" else 29,
                "budget_used": 0 if policy == "none" else 29,
                "uwb_attempted": 0 if policy == "none" else 25,
                "uwb_accepted": 0 if policy == "none" else 10})
    return spec, rows


def test_approved_fixture_is_not_training_release():
    spec, rows = fixture()
    result = admission(rows, spec)
    assert result["passed"]
    assert result["candidate_closed_loop_allowed"]
    assert result["scientific_training_ready"] is False
    assert result["independent_seeds"] == 4


@pytest.mark.parametrize("field,value", [("steps", 119), ("uav0_rmse_m", 1.01),
    ("uav0_gnss_accepted", 107), ("covariance_finite", False),
    ("covariance_min_eigenvalue", -1e-6), ("full_covariance_trace_max", 10000),
    ("uav2_position_nees_mean", 8.), ("uwb_requested", 30), ("budget_used", 31),
    ("boundary_projections", 1), ("safety_projections", 1), ("collision_count", 1),
    ("uav2_absolute_rmse_m", 2.11)])
def test_individual_failed_rule_blocks_all_closed_loop(field, value):
    spec, rows = fixture()
    next(r for r in rows if r["policy"] == "random_peer")[field] = value
    result = admission(rows, spec)
    assert not result["passed"]
    assert result["candidate_closed_loop_allowed"] is False


def test_low_absorption_blocks_even_if_true_error_improves():
    spec, rows = fixture()
    for r in rows:
        r["uwb_accepted"] = 1
    assert not admission(rows, spec)["passed"]


def test_cross_seed_requirement_not_replaced_by_pooled_gain():
    spec, rows = fixture()
    for r in rows:
        if r["policy"] == "random_peer" and r["seed"] < 2:
            r["uav2_absolute_rmse_m"] = 2.
    assert not admission(rows, spec)["passed"]


def test_near_zero_baseline_fails_closed():
    spec, rows = fixture()
    next(r for r in rows if r["policy"] == "none")["uav2_absolute_rmse_m"] = 0.
    assert not admission(rows, spec)["passed"]


def test_unrelated_policy_does_not_release_primary():
    spec, rows = fixture()
    rows = [dict(r, policy="fixed0") if r["policy"] == "random_peer" else r for r in rows]
    assert not admission(rows, spec)["passed"]
