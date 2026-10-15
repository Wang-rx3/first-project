import copy
from src.analysis.partial_gnss_validation_gate_v10 import safety_admission, select_candidate, METHODS


def fixture(fine_gain=.03):
    spec={"development_seeds":[1,2,3,4],"safety":{"kernel_p95_limit_ms":200},
          "selection":{"minimum_fractional_improvement":.01,"absolute_rmse_min_gain_m":.01,
                       "positive_independent_seeds":3,"maximum_case_degradation_fraction":.05}}
    rows=[]
    for method in METHODS:
        for scene in ("col","noncol"):
            for seed in (1,2,3,4):
                for policy in ("none","fixed0","fixed1","random_peer"):
                    rows.append({"method":method,"scene_id":scene,"seed":seed,"policy":policy,
                        "full_horizon_eligible":True,"covariance_finite":True,"covariance_min_eigenvalue":1e-5,
                        "full_covariance_trace_max":30.,"state_abs_max":100.,"conditional_bound_mismatch_count":0,
                        "replay_interface_valid":True,"packet_reconstruction_max":0.,"source_reconstruction_max":0.,
                        "kernel_timing_samples_ms":[2.],"uav2_absolute_rmse_m":2.-fine_gain if method=="refined_ci" else 2.,
                        "cluster_relative_rmse_m":1.-fine_gain if method=="refined_ci" else 1.,
                        "uav2_position_nees_mean":100.,"uwb_accepted":0,"effective_fusion_rate":0.})
    return spec,rows


def test_zero_fusion_and_high_error_nees_do_not_block_safe_execution():
    spec,rows=fixture()
    safety=safety_admission(rows,spec,True)
    assert all(v["passed"] for v in safety.values())
    decision=select_candidate(rows,spec,safety)
    assert decision["selected_method"]=="refined_ci"
    assert decision["closed_loop_execution_allowed"]


def test_weak_fine_accuracy_retains_original_and_still_executes():
    spec,rows=fixture(fine_gain=.0001)
    decision=select_candidate(rows,spec,safety_admission(rows,spec,True))
    assert decision["selected_method"]=="original_ci"
    assert decision["closed_loop_execution_allowed"]
    assert not decision["refined_development_benefit"]


def test_both_numerically_unsafe_block_physical_execution():
    spec,rows=fixture()
    for r in rows:r["covariance_finite"]=False
    decision=select_candidate(rows,spec,safety_admission(rows,spec,True))
    assert decision["selected_method"] is None
    assert not decision["closed_loop_execution_allowed"]


def test_only_fine_safe_executes_without_accuracy_claim():
    spec,rows=fixture(fine_gain=-.1)
    for r in rows:
        if r["method"]=="original_ci":r["covariance_finite"]=False
    decision=select_candidate(rows,spec,safety_admission(rows,spec,True))
    assert decision["selected_method"]=="refined_ci"
    assert decision["closed_loop_execution_allowed"]
    assert not decision["refined_development_benefit"]


def test_case_harm_cannot_be_hidden_by_average_improvement():
    spec,rows=fixture(fine_gain=.2)
    next(r for r in rows if r["method"]=="refined_ci" and r["policy"]=="random_peer")["uav2_absolute_rmse_m"]=2.2
    assert select_candidate(rows,spec,safety_admission(rows,spec,True))["selected_method"]=="original_ci"


def test_unit_release_and_kernel_compute_budget_are_safety_requirements():
    spec,rows=fixture()
    assert not any(v["passed"] for v in safety_admission(rows,spec,False).values())
    for r in rows:r["kernel_timing_samples_ms"]=[201.]
    assert not any(v["passed"] for v in safety_admission(rows,spec,True).values())
