from src.analysis.neighbor_selection_evidence_audit_v12 import classify_points, legal_regret


def test_duplicate_and_missing_cannot_masquerade_as_valid_count():
    first = {"study_phase": "confirm", "scene_id": "c", "seed": 1, "step": 6, "label_complete": True, "identity_matches": True}
    expected = {("confirm", "c", 1, 6), ("confirm", "c", 1, 10)}
    result, valid = classify_points([first, first], expected)
    assert result["entries"] == 2
    assert not result["exhaustive"]
    assert result["duplicate"] and result["missing"]


def test_failure_record_is_planned_entry_and_not_valid_point():
    point = {"study_phase": "confirm", "scene_id": "c", "seed": 1, "step": 6, "label_complete": False}
    result, valid = classify_points([point], {("confirm", "c", 1, 6)})
    assert result["exhaustive"] and result["unavailable"] == 1
    assert result["valid"] == 0 and valid == []


def test_illegal_profitable_action_is_excluded_from_oracle():
    row = {"study_phase": "confirm", "scene_id": "c", "seed": 1, "step": 30, "label_complete": True,
           "pre_context": {"legal_sources": ["uav_01"]}, "old_choice": "none", "minimal_choice": "none"}
    for timing in ("current", "fourstate"):
        row["offline_reward_"+timing+"_none"] = 0.
        row["offline_reward_"+timing+"_uav_00"] = 100.
        row["offline_reward_"+timing+"_uav_01"] = -.01
    result = legal_regret(row)
    assert result["allowed_actions"] == ["none", "uav_01"]
    assert result["oracle_four_net_ceiling_m2"] == 0.
    assert result["minimal_four_net_regret_m2"] == 0.
    assert not result["oracle_deployable"]
