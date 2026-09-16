"""Contract tests for the read-only Stage 2B-R3 identifiability audit."""

import numpy as np

from src.experiments.single_uav_stage2br3_observation_identifiability import (
    FEATURE_GROUPS,
    NODE_DIM,
    _action_decomposition,
    _canonical_phase,
    _classify_causes,
    _observation_vector,
    _phase_pair,
    _smoke_subset,
    _transition_windows,
    build_feature_map,
)


def _record(*, phase="normal", side="none", split="train", episode_id="ep", pair_id="", step=0, action=(0.1, 0.0, 0.0)):
    node = np.zeros(NODE_DIM, dtype=float).tolist()
    node[3] = 1.0
    node[4] = 0.0
    node[5] = 0.0
    return {
        "split": split,
        "episode_id": episode_id,
        "pair_id": pair_id,
        "scenario_hash": f"scenario-{episode_id}",
        "scenario_seed": 2026,
        "step_index": step,
        "phase": phase,
        "base_phase": phase,
        "side": side,
        "kind": "static_obstacle" if side != "none" else "normal_far_distance",
        "observation": {"node_features": [node]},
        "teacher_action_delta_v_mps": list(action),
        "goal_direction": [1.0, 0.0, 0.0],
        "goal_distance_m": 5.0,
        "speed_mps": 1.0,
        "radial_velocity_mps": 1.0,
        "tangential_velocity_mps": 0.0,
        "obstacle_direction_body_m": [1.0, 1.0 if side == "left" else -1.0 if side == "right" else 0.0, 0.0],
    }


def test_phase_mapping_keeps_bypass_sides_and_boundary_overlay_separate():
    assert _canonical_phase(_record(phase="bypass", side="left")) == "BYPASS_LEFT"
    assert _canonical_phase(_record(phase="bypass", side="right")) == "BYPASS_RIGHT"
    assert _canonical_phase(_record(phase="post_bypass", side="left")) == "POST_BYPASS"
    assert _canonical_phase(_record(phase="recovery", side="left")) == "RECOVERY"
    assert _canonical_phase(_record(phase="boundary_onset", side="left")) == "BOUNDARY_ONSET"


def test_observation_contract_is_exactly_42_node_features():
    record = _record()
    assert _observation_vector(record).shape == (42,)
    assert sum(len(indices) for name, indices in FEATURE_GROUPS.items() if name == "obstacle_geometry") == 6


def test_teacher_action_decomposition_reports_radial_and_braking_components():
    record = _record(action=(-0.25, 0.2, 0.0))
    parts = _action_decomposition(record)
    assert parts["teacher_radial_action_mps"] < 0.0
    assert parts["teacher_braking_magnitude_mps"] > 0.0
    assert parts["teacher_tangential_action_mps"] > 0.0


def test_phase_pair_is_order_invariant():
    assert _phase_pair("BYPASS_LEFT", "RECOVERY") == _phase_pair("RECOVERY", "BYPASS_LEFT")


def test_smoke_subset_is_deterministic_and_episode_bounded():
    records = [_record(phase="bypass", side="left", episode_id=f"ep-{index}", step=step) for index in range(6) for step in range(5)]
    first = _smoke_subset(records, max_episodes_per_phase=2, max_rows_per_episode_phase=2)
    second = _smoke_subset(records, max_episodes_per_phase=2, max_rows_per_episode_phase=2)
    assert first == second
    assert len({row["episode_id"] for row in first}) == 2
    assert len(first) == 4


def test_transition_window_uses_bypass_to_recovery_and_preserves_offsets():
    rows = []
    for step, phase in enumerate(["bypass", "bypass", "post_bypass", "recovery", "recovery"]):
        rows.append(_record(phase=phase, side="left", episode_id="ep-transition", pair_id="pair-1", step=step))
    windows, transitions = _transition_windows(rows, radius=1)
    assert len(transitions) == 1
    assert transitions[0]["from_phase"] == "BYPASS_LEFT"
    assert transitions[0]["to_phase"] == "POST_BYPASS"
    assert {row["offset"] for row in windows} == {-1, 0, 1}


def test_feature_map_marks_boundary_as_unavailable_direct_feature():
    layout = [{"index": index, "feature": f"f{index}", "source": "test", "notes": ""} for index in range(42)]
    result = build_feature_map(layout)
    assert len(result) == 42
    assert result[39]["obstacle_related"] == 1
    assert result[39]["available_to_actor"] == 1
    assert not FEATURE_GROUPS["boundary"]


def test_classification_returns_inconclusive_without_flags():
    classification = _classify_causes([], [], [], [])
    assert classification["case"] == "INCONCLUSIVE"
    assert classification["next_step_choice"] == "E"
