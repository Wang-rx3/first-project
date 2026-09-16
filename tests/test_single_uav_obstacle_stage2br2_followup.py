"""Contract tests for the isolated Stage 2B-R2 sequential follow-up."""

from pathlib import Path

import numpy as np

from src.experiments.single_uav_obstacle_stage2br2_followup import (
    ALL_CATEGORIES,
    EXACT_R2_CATEGORIES,
    PHASE_DEFINITION,
    _scenario_category,
    _scenario_side,
    _dataset_record,
    _phase_action_audit,
    _select_training_scope,
    exact_r2_smoke_scenarios,
    four_category_scenarios,
    sequential_pair_scenarios,
)


def _config():
    from src.experiments.single_uav_obstacle_stage2br2_followup import DEFAULT_STAGE2A_CHECKPOINT, _load_policy

    _payload, _trainer, config, _identity = _load_policy(
        Path(DEFAULT_STAGE2A_CHECKPOINT), output_dir=Path(".tmp") / "stage2br2-followup-test-source"
    )
    return config


def test_exact_r2_scene_contract_has_four_fixed_entries():
    scenarios = exact_r2_smoke_scenarios(_config())
    assert len(scenarios) == 4
    assert tuple(_scenario_category(scenario) for scenario in scenarios) == EXACT_R2_CATEGORIES
    assert len({_scenario_category(scenario) for scenario in scenarios}) == 2


def test_exact_r2_scenes_are_reusable_across_roles():
    scenarios = exact_r2_smoke_scenarios(_config())
    hashes = []
    from src.experiments.single_uav_oracle_behavior_cloning import _canonical_hash

    for scenario in scenarios:
        hashes.append(_canonical_hash(scenario))
    assert len(hashes) == 4
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes)


def test_four_category_probe_covers_all_mirror_categories():
    scenarios = four_category_scenarios(_config())
    assert tuple(_scenario_category(scenario) for scenario in scenarios) == ALL_CATEGORIES
    assert [_scenario_side(scenario) for scenario in scenarios] == ["left", "right", "left", "right"]


def test_sequential_dataset_has_four_disjoint_mirror_pairs():
    scenarios = sequential_pair_scenarios(_config())
    assert len(scenarios) == 8
    pairs = {}
    for scenario in scenarios:
        pair_id = scenario["stage2br2_followup"]["pair_id"]
        pairs.setdefault(pair_id, []).append(scenario)
    assert len(pairs) == 4
    assert all(len(members) == 2 for members in pairs.values())
    assert all(
        members[0]["stage2br2_followup"]["split"] == members[1]["stage2br2_followup"]["split"]
        for members in pairs.values()
    )


def test_sequential_pair_split_does_not_share_pair_between_splits():
    scenarios = sequential_pair_scenarios(_config())
    train_pairs = {s["stage2br2_followup"]["pair_id"] for s in scenarios if s["stage2br2_followup"]["split"] == "train"}
    validation_pairs = {s["stage2br2_followup"]["pair_id"] for s in scenarios if s["stage2br2_followup"]["split"] == "validation"}
    assert train_pairs == {"pair_8101", "pair_8102", "pair_8103"}
    assert validation_pairs == {"pair_8104"}
    assert not train_pairs & validation_pairs


def test_phase_contract_is_explicit_and_complete():
    assert set(PHASE_DEFINITION) == {"approach", "decision", "bypass", "post_bypass", "recovery", "boundary_onset"}


def test_dataset_record_preserves_overlay_phase_and_base_phase():
    scenario = sequential_pair_scenarios(_config())[0]
    row = {
        "phase": "boundary_onset",
        "base_phase": "post_bypass",
        "true_goal_distance_before_m": 4.0,
        "total_speed_before_mps": 1.0,
        "true_radial_velocity_before_mps": 0.5,
        "tangential_speed_before_mps": 0.25,
        "goal_direction_x": 1.0,
        "goal_direction_y": 0.0,
        "goal_direction_z": 0.0,
        "velocity_before_x_mps": 1.0,
        "velocity_before_y_mps": 0.0,
        "velocity_before_z_mps": 0.0,
        "lateral_forward_action_ratio": 0.5,
        "boundary_projected": 1,
        "safety_projected": 0,
        "action_clipped": 0,
    }
    record = _dataset_record(
        scenario=scenario,
        split="train",
        pair_id="pair_8101",
        episode_id="episode_1",
        episode_index=1,
        step=77,
        actor_observation={"node_features": [[0.0] * 42]},
        teacher_action=np.array([0.1, -0.2, 0.0]),
        student_action=np.array([0.2, -0.1, 0.0]),
        row=row,
        control={"route_phase": "goal_after_bypass", "truth_state_read_for_control": False},
        scenario_seed=2600000,
        rollout_seed=2708101,
    )
    assert record["phase"] == "boundary_onset"
    assert record["base_phase"] == "post_bypass"
    assert record["teacher_delta_v_y_mps"] == -0.2


def test_phase_action_audit_uses_overlay_phase_and_teacher_action():
    rows = [
        {
            "evaluation_stage": "sequential_dagger_data_collection",
            "policy": "sequential_dagger_student",
            "category": "left",
            "phase": "boundary_onset",
            "base_phase": "post_bypass",
            "raw_actor_delta_v_y_mps": -0.2,
            "teacher_delta_v_y_mps": -0.4,
            "raw_actor_delta_v_x_mps": 0.3,
            "oracle_delta_v_x_mps": 0.5,
            "lateral_forward_action_ratio": 0.67,
            "raw_action_norm_mps": 0.36,
            "after_boundary_delta_v_x_mps": 0.3,
            "after_boundary_delta_v_y_mps": -0.2,
            "after_boundary_delta_v_z_mps": 0.0,
            "true_goal_distance_before_m": 3.0,
            "true_radial_progress_mps": 0.2,
            "total_speed_mps": 1.0,
            "oracle_desired_speed_mps": 0.5,
            "boundary_projected": 1,
            "safety_projected": 0,
        }
    ]
    audit = _phase_action_audit(rows)
    assert len(audit) == 1
    assert audit[0]["phase"] == "boundary_onset"
    assert audit[0]["base_phases_present"] == "post_bypass"
    assert audit[0]["mean_oracle_delta_v_y_mps"] == -0.4


def test_scope_selection_never_chooses_full_actor_by_default():
    pre = {
        "r2_39_42": {"direction_adaptive": 4, "passed_obstacle_rate": 0.5, "post_bypass_goal_progress_rate": 0.5, "collision": 0, "passed_obstacle": 2, "post_bypass_goal_progress": 2, "boundary_projection_steps": 110},
        "r2_actor_head": {"direction_adaptive": 4, "passed_obstacle_rate": 0.5, "post_bypass_goal_progress_rate": 0.5, "collision": 0, "passed_obstacle": 2, "post_bypass_goal_progress": 2, "boundary_projection_steps": 110},
        "r2_full_actor": {"direction_adaptive": 4, "passed_obstacle_rate": 0.5, "post_bypass_goal_progress_rate": 0.5, "collision": 0, "passed_obstacle": 2, "post_bypass_goal_progress": 2, "boundary_projection_steps": 110},
    }
    scope, reason = _select_training_scope(pre)
    assert scope == "39_42"
    assert reason == "all_tiny_scopes_fail_or_are_not_clearly_better"


def test_scope_selection_can_identify_actor_head_without_selecting_full_actor():
    pre = {
        "r2_39_42": {"direction_adaptive": 4, "passed_obstacle_rate": 0.5, "post_bypass_goal_progress_rate": 0.5, "collision": 0, "passed_obstacle": 2, "post_bypass_goal_progress": 2, "boundary_projection_steps": 110},
        "r2_actor_head": {"direction_adaptive": 4, "passed_obstacle_rate": 0.75, "post_bypass_goal_progress_rate": 0.75, "collision": 0, "passed_obstacle": 3, "post_bypass_goal_progress": 3, "boundary_projection_steps": 80},
        "r2_full_actor": {"direction_adaptive": 4, "passed_obstacle_rate": 1.0, "post_bypass_goal_progress_rate": 1.0, "collision": 0, "passed_obstacle": 4, "post_bypass_goal_progress": 4, "boundary_projection_steps": 0},
    }
    scope, _reason = _select_training_scope(pre)
    assert scope == "actor_head"
