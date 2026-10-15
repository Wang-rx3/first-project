from src.analysis.neighbor_selection_joint_audit_v11 import joint_assessment
from src.analysis.neighbor_selection_gate_v11 import assess


def test_three_marginal_positive_seeds_are_not_three_joint_positive_seeds():
    spec={"geometry_ids":["col","noncol"],"confirmation_seeds":[11,12,13,14]}
    rows=[];cf=[]
    for scene in spec["geometry_ids"]:
        for seed in spec["confirmation_seeds"]:
            rows.extend([
                {"study_phase":"confirm","scene_id":scene,"seed":seed,"policy":"random_peer","uav2_absolute_rmse_m":1.,"cluster_relative_rmse_m":1.,"request_windows":[6],"uwb_requested":1},
                {"study_phase":"confirm","scene_id":scene,"seed":seed,"policy":"greedy","uav2_absolute_rmse_m":1.01 if seed==14 else .8,"cluster_relative_rmse_m":1.01 if seed==11 else .8,"request_windows":[6],"uwb_requested":1}])
            for phase in ("develop","confirm"):
                for value in (-.04,-.02,.03,.05,.08,.1):
                    cf.append({"study_phase":phase,"scene_id":scene,"seed":seed,"dual_eligible":True,"meaningful":True,"score_difference":value,"utility_difference":value,"ranking_correct":1.,"preferred_peer":"uav_00" if value>0 else "uav_01"})
    assert assess(rows,cf,spec,True)["trajectory_passed"]
    result=joint_assessment(rows,cf,spec,True)
    assert result["joint_positive_seeds_per_geometry"]=={"col":2,"noncol":2}
    assert not result["trajectory_passed"]
