from src.analysis.neighbor_selection_gate_v11 import assess, correlation


def fixture():
    spec={"geometry_ids":["col","noncol"],"confirmation_seeds":[11,12,13,14]}
    physical=[]
    cf=[]
    for scene in spec["geometry_ids"]:
        for seed in spec["confirmation_seeds"]:
            for policy,value in (("random_peer",1.),("greedy",.9)):
                physical.append({"study_phase":"confirm","scene_id":scene,"seed":seed,"policy":policy,
                    "uav2_absolute_rmse_m":value,"cluster_relative_rmse_m":value,
                    "request_windows":[6,10],"uwb_requested":2})
            for phase in ("develop","confirm"):
                for value in (-.04,-.02,.03,.05,.08,.1):
                    cf.append({"study_phase":phase,"scene_id":scene,"seed":seed,"dual_eligible":True,
                        "meaningful":True,"score_difference":value,"utility_difference":value,
                        "ranking_correct":1.,"preferred_peer":"uav_00" if value>0 else "uav_01"})
    return physical,cf,spec


def test_peer_identity_admission_requires_matching_request_windows():
    physical,cf,spec=fixture()
    assert assess(physical,cf,spec,True)["status"]=="pass_limited_scope"
    physical[1]["request_windows"]=[6,14]
    result=assess(physical,cf,spec,True)
    assert result["trajectory_passed"] and not result["cost_matched"]
    assert result["status"]=="conditional_pass"


def test_zero_denominator_cannot_claim_gain():
    physical,cf,spec=fixture()
    physical[0]["uav2_absolute_rmse_m"]=0.
    result=assess(physical,cf,spec,True)
    assert not result["trajectory_passed"]


def test_counterfactual_windows_cannot_replace_missing_seed_coverage():
    physical,cf,spec=fixture()
    cf=[r for r in cf if not(r["study_phase"]=="confirm" and r["seed"]==14)]
    cf += [dict(r) for r in cf]
    result=assess(physical,cf,spec,True)
    assert not result["coverage_passed"] and result["status"]=="not_passed"


def test_integrity_failure_blocks_scientific_pass():
    physical,cf,spec=fixture()
    assert assess(physical,cf,spec,False)["status"]=="not_passed"


def test_constant_score_is_undefined_and_cannot_pass():
    assert correlation([1,1,1],[1,2,3],True) is None
    assert correlation([1,2,3],[3,2,1],True)==-1
