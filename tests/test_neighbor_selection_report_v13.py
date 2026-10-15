"""Pure V13 reporting tests; no simulator, filter, or training is imported."""
from src.analysis.neighbor_selection_report_v13 import _feature_correlations, _preferred, assess, correlation, groups


def _physical(phase="confirm", scene="g0", seed=1, policy="random_peer", value=10.0, complete=True):
    return {
        "study_phase": phase, "scene_id": scene, "seed": seed, "policy": policy,
        "full_horizon_eligible": complete,
        "uav2_absolute_rmse_m": value if complete else None,
        "cluster_relative_rmse_m": value if complete else None,
        "uav2_position_nees_mean": 2.0, "uwb_nis_mean": None,
        "uwb_requested": 3, "uwb_attempted": 3, "uwb_accepted": 2,
        "measurement_bytes_equivalent": 1536, "uav2_strict_parking": True,
        "uav2_final_localization_error_m": value if complete else None,
    }


def _point(phase, scene, seed, index, sign=1.0):
    norm = sign * (0.02 + index * 0.01)
    return {"study_phase": phase, "scene_id": scene, "seed": seed, "step": index,
            "dual_eligible": True, "meaningful_norm": True,
            "norm_utility_difference": norm, "v13_score_difference": norm,
            "v13_ranking_correct": 1.0, "legacy_norm_ranking_correct": 1.0,
            "identity_matches": True, "v13_features": {"range_m": float(index), "rssi": -50.0 + index}}


def test_groups_null_full_metrics_and_report_completed_subset():
    rows = [_physical(value=2.0), _physical(seed=2, value=2.0, complete=False)]
    row = groups(rows)[0]
    assert row["blocks"] == 2
    assert row["complete_blocks"] == 1
    assert row["censored_blocks"] == 1
    assert row["uav2_absolute_rmse_m"] is None
    assert row["complete_case_uav2_absolute_rmse_m"] == 2.0
    assert row["complete_case_uav2_position_nees_mean"] == 2.0


def test_groups_pool_is_root_mean_square_for_complete_cohort():
    rows = [_physical(seed=1, value=3.0), _physical(seed=2, value=4.0)]
    row = groups(rows)[0]
    assert row["full_horizon_eligible"]
    assert row["uav2_absolute_rmse_m"] == 3.5355339059327378
    assert row["complete_case_uav2_absolute_rmse_m"] == row["uav2_absolute_rmse_m"]


def test_correlation_null_handling_and_tied_rank_contract():
    assert correlation([1.0, 2.0], [1.0, 2.0]) is None
    assert correlation([1.0, None, 3.0], [1.0, 2.0, 3.0]) is None
    assert correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], True) is None


def test_norm_preferred_label_does_not_reuse_old_sse_preference():
    point = {"norm_utility_difference": 0.02, "preferred_peer": "uav_01"}
    assert _preferred(point) == "uav_00"


def test_feature_relationships_retain_all_dual_and_mark_informative_stratum():
    points = [_point("confirm", "g0", 1, 1, 1.0),
              {**_point("confirm", "g0", 1, 2, 1.0), "meaningful_norm": False,
               "norm_utility_difference": 0.001}]
    rows = _feature_correlations(points)
    assert {row["stratum"] for row in rows} == {"all_dual", "informative"}
    assert max(row["n"] for row in rows if row["stratum"] == "all_dual") == 2


def test_assess_freezes_v13_ranking_and_physical_gates():
    spec = {"geometry_ids": ["g0", "g1"], "confirmation_seeds": [1, 2, 3, 4],
            "development_seeds": [1, 2, 3, 4]}
    rows = []
    for phase in ("develop", "confirm"):
        for scene in spec["geometry_ids"]:
            for seed in spec["confirmation_seeds"]:
                rows.extend((_physical(phase, scene, seed, "random_peer", 10.0),
                             _physical(phase, scene, seed, "v13_greedy", 9.0)))
    points = []
    for phase in ("develop", "confirm"):
        for scene in spec["geometry_ids"]:
            for seed in spec["confirmation_seeds"]:
                for index in range(3):
                    points.append(_point(phase, scene, seed, seed * 10 + index,
                                         1.0 if index % 2 == 0 else -1.0))
    result = assess(rows, points, spec, integrity=True)
    assert result["research_status"] == "通过"
    assert result["training_ready"]
    assert result["gates"]["v13_ranking"]
    assert result["physical"]["passed"]
    assert result["legacy_reference"]["historical_source30"]["passed"] is None
    assert result["legacy_reference"]["historical_source30"]["evaluated_this_round"] is False
    assert result["legacy_reference"]["separate_reference"]
    assert result["legacy_reference"]["does_not_lower_v13_gates"]


def test_assess_failure_aware_missing_case_is_not_conditional():
    spec = {"geometry_ids": ["g0"], "confirmation_seeds": [1], "development_seeds": [1]}
    rows = [_physical("confirm", "g0", 1, "random_peer", 10.0)]
    result = assess(rows, [], spec, integrity=False)
    assert result["research_status"] == "未通过"
    assert not result["training_ready"]
    assert result["physical"]["missing_cases"]

