"""Post-data pure reporting tests; no thresholds or algorithms are changed."""
from src.analysis.neighbor_selection_failure_report_v12 import groups


def _row(complete, value):
    return {"study_phase": "confirm", "scene_id": "train_collinear", "policy": "random_peer",
            "full_horizon_eligible": complete, "uav2_absolute_rmse_m": value, "cluster_relative_rmse_m": value,
            "uav2_position_nees_mean": 2., "uwb_nis_mean": None, "uwb_requested": 3 if complete else 0,
            "uwb_attempted": 3 if complete else 0, "uwb_accepted": 1 if complete else 0,
            "measurement_bytes_equivalent": 1536 if complete else 0, "uav2_strict_parking": False,
            "uav2_final_localization_error_m": value}


def test_incomplete_cohort_keeps_null_rmse_and_explicit_subset():
    row = groups([_row(True, 2.), _row(False, None)])[0]
    assert row["blocks"] == 2
    assert row["complete_blocks"] == 1
    assert row["censored_blocks"] == 1
    assert row["uav2_absolute_rmse_m"] is None
    assert row["complete_case_uav2_absolute_rmse_m"] == 2.
    assert row["position_nees_mean"] is None
    assert row["complete_case_position_nees_mean"] == 2.


def test_all_censored_group_has_no_subset_rmse():
    row = groups([_row(False, None)])[0]
    assert row["complete_case_uav2_absolute_rmse_m"] is None
    assert row["complete_blocks"] == 0


def test_complete_group_reports_same_full_and_subset_metric():
    row = groups([_row(True, 2.), _row(True, 2.)])[0]
    assert row["full_horizon_eligible"]
    assert row["complete_case_uav2_absolute_rmse_m"] == row["uav2_absolute_rmse_m"]
