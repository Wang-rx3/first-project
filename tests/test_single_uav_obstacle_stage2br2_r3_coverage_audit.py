"""Contract tests for the isolated Stage 2B-R2-R3 coverage audit."""

from src.experiments.single_uav_obstacle_stage2br2_r3_coverage_audit import (
    PHASE_DEFINITION,
    TARGET_PHASES,
    _balanced_records,
    _coverage_phase_for_step,
    coverage_scenarios,
)


def test_r3_phase_contract_is_explicit_and_complete():
    assert tuple(PHASE_DEFINITION) == (
        "approach",
        "decision",
        "bypass",
        "post_bypass",
        "recovery",
        "boundary_onset",
    )
    assert TARGET_PHASES == tuple(PHASE_DEFINITION)


def test_r3_scenarios_are_eight_disjoint_mirror_pairs():
    scenarios = coverage_scenarios(object())
    assert len(scenarios) == 16
    pairs = {}
    for scenario in scenarios:
        metadata = scenario["stage2br2_r3"]
        pairs.setdefault(metadata["pair_id"], []).append(scenario)
    assert len(pairs) == 8
    assert all(len(members) == 2 for members in pairs.values())
    train_pairs = {pair for pair, members in pairs.items() if members[0]["stage2br2_r3"]["split"] == "train"}
    validation_pairs = {pair for pair, members in pairs.items() if members[0]["stage2br2_r3"]["split"] == "validation"}
    assert len(train_pairs) == 6
    assert len(validation_pairs) == 2
    assert not train_pairs & validation_pairs


def test_state_only_classifier_retains_far_field_approach():
    phase, base = _coverage_phase_for_step(
        phase_state={"obstacle_center": [54.0, 104.5, 20.0], "decision_seen": False, "decision_lateral_abs": 0.0},
        step=0,
        position_before=[34.0, 100.0, 20.0],
        pre_diag={"pre_candidate_static_swept_conflict": 0, "pre_candidate_static_unsafe": 0},
        raw_lateral=0.35,
        radial_progress=0.0,
        passed_before=False,
        first_pass_step=None,
        first_boundary_step=None,
    )
    assert phase == "approach"
    assert base == "approach"


def test_state_only_classifier_starts_decision_at_eight_meters():
    state = {"obstacle_center": [54.0, 104.5, 20.0], "decision_seen": False, "decision_lateral_abs": 0.0}
    phase, base = _coverage_phase_for_step(
        phase_state=state,
        step=10,
        position_before=[46.0, 100.0, 20.0],
        pre_diag={"pre_candidate_static_swept_conflict": 0, "pre_candidate_static_unsafe": 0},
        raw_lateral=0.0,
        radial_progress=0.0,
        passed_before=False,
        first_pass_step=None,
        first_boundary_step=None,
    )
    assert phase == "decision"
    assert base == "decision"
    assert state["decision_seen"] is True


def test_balanced_records_cap_each_episode_phase_deterministically():
    records = []
    for step in range(20):
        records.append({"split": "train", "episode_id": "ep_a", "phase": "post_bypass", "step": step})
    for step in range(3):
        records.append({"split": "train", "episode_id": "ep_a", "phase": "recovery", "step": step})
    for step in range(20):
        records.append({"split": "validation", "episode_id": "ep_b", "phase": "post_bypass", "step": step})
    selected = _balanced_records(records, max_per_episode_phase=8)
    assert len([row for row in selected if row["episode_id"] == "ep_a" and row["phase"] == "post_bypass"]) == 8
    assert len([row for row in selected if row["episode_id"] == "ep_a" and row["phase"] == "recovery"]) == 3
    assert len([row for row in selected if row["episode_id"] == "ep_b" and row["phase"] == "post_bypass"]) == 8
    assert selected == _balanced_records(records, max_per_episode_phase=8)
