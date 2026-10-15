"""Pure mutation tests for the independent three-UAV auditor."""

from __future__ import annotations

import copy

import numpy as np

from src.analysis.three_uav_fixed_navigation_audit import (
    audit_flow_rows,
    audit_navigation_row,
    audit_pair_rows,
    derive_strict_status,
    packet_id,
    synchronized_pair_distance,
)
from src.control.fixed_navigation import FixedNavigationConfig, fixed_navigation_command


def _navigation_row(*, post_position=None, post_velocity=None):
    cfg = FixedNavigationConfig()
    estimate = np.array([0.0, 0.0, 10.0, 4.0, 0.0, 0.0])
    goal = np.array([10.0, 0.0, 10.0])
    command = fixed_navigation_command(estimate, goal, cfg)
    requested = np.asarray(command["command_delta_v_mps"], dtype=float)
    pre_position = estimate[:3].copy()
    pre_velocity = estimate[3:].copy()
    post_velocity = pre_velocity + requested if post_velocity is None else np.asarray(post_velocity, dtype=float)
    post_position = pre_position + (2.0 * pre_velocity + requested) * 0.1 if post_position is None else np.asarray(post_position, dtype=float)
    return {
        "episode_id": "synthetic", "variant": "normal", "step": 1, "uav_id": "uav_00", "goal_m": goal,
        "ekf_state_pre": estimate, "ekf_state_post": np.r_[post_position, post_velocity],
        "ekf_covariance_post": np.eye(15), "true_position_pre_m": pre_position,
        "true_velocity_pre_mps": pre_velocity, "true_position_post_m": post_position,
        "true_velocity_post_mps": post_velocity, "requested_delta_v_mps": requested,
        "env_input_delta_v_mps": requested.copy(), "reported_applied_delta_v_mps": requested.copy(),
        "actual_delta_v_mps": requested.copy(), "actual_acceleration_mps2": requested / 0.2,
        "desired_velocity_mps": command["desired_velocity_mps"],
        "requested_acceleration_mps2": command["requested_acceleration_mps2"],
        "raw_delta_v_mps": command["raw_delta_v_mps"], "axis_limited_delta_v_mps": command["axis_limited_delta_v_mps"],
        "qualifying": False, "safety2_enabled": True, "assist_mask": "00000",
        "boundary_projected": False, "safety_projected": False, "environment_action_clipped": False,
        "collision": False,
    }


def _nav(uid, pre, post, step=1):
    return {
        "episode_id": "synthetic", "step": step, "uav_id": uid,
        "true_position_pre_m": np.asarray(pre, dtype=float), "true_position_post_m": np.asarray(post, dtype=float),
        "true_velocity_pre_mps": np.zeros(3), "true_velocity_post_mps": np.zeros(3),
    }


def _flow(source, target, *, variant="normal", requested=.5, applied=.5, send=.2, previous=None):
    state = np.arange(15, dtype=float)
    positions = {"uav_00": np.array([0.0, 0.0, 2.0]), "uav_01": np.array([4.0, 0.0, 2.0]), "uav_02": np.array([8.0, 0.0, 2.0])}
    return {
        "episode_id": "synthetic", "variant": variant, "step": 1, "source": source, "target": target,
        "physical_distance_m": float(np.linalg.norm(positions[source] - positions[target])), "potential": True, "requested_bandwidth": requested,
        "applied_bandwidth": applied, "controlled_link_selected": applied > 0, "transmitted": applied > 0,
        "message_send_time_s": send, "message_age_s": 0.0, "fresh": True, "effective_visible": True,
        "previous_consumed_send_time_s": previous, "new_since_last_consumed": True, "reused": False,
        "newly_arrived_this_step": True, "packet_id": packet_id(source, target, send),
        "received_message_state": state, "received_message_covariance_diagonal": np.ones(15),
        "actor_neighbor_state": state.copy(), "actor_neighbor_covariance_diagonal": np.ones(15),
        "actor_node_state": state.copy(), "actor_node_covariance_diagonal": np.ones(15),
    }


def test_navigation_auditor_recomputes_actual_delta_v_and_rejects_mutation():
    row = _navigation_row()
    assert audit_navigation_row(row) == []
    mutated = copy.deepcopy(row)
    mutated["reported_applied_delta_v_mps"] = np.asarray(mutated["reported_applied_delta_v_mps"]) + [0.01, 0.0, 0.0]
    errors = audit_navigation_row(mutated)
    assert "reported_applied_delta_v" in errors


def test_strict_threshold_and_consecutive_streak_are_recomputed():
    rows = []
    for step in range(1, 6):
        rows.append({"goal_m": [0.0, 0.0, 2.0], "true_position_post_m": [1.0, 0.0, 2.0],
                     "true_velocity_post_mps": [0.5, 0.0, 0.0], "collision": False})
    status = derive_strict_status(rows)
    assert status[-1]["qualifying"] is True
    assert status[-1]["hold_steps"] == 5
    assert status[-1]["ever_success"] is True
    rows[2]["true_velocity_post_mps"] = [0.500001, 0.0, 0.0]
    status = derive_strict_status(rows)
    assert status[2]["qualifying"] is False
    assert status[-1]["ever_success"] is False


def test_synchronized_pair_audit_detects_midpoint_crossing_and_unsafe_distance():
    pre = {"uav_00": [0, 0, 2], "uav_01": [2, 0, 2], "uav_02": [10, 0, 2]}
    post = {"uav_00": [2, 0, 2], "uav_01": [0, 0, 2], "uav_02": [10, 0, 2]}
    distance, fraction = synchronized_pair_distance(pre["uav_00"], post["uav_00"], pre["uav_01"], post["uav_01"])
    assert fraction == .5
    assert distance == 0.0
    navigation = [_nav(uid, pre[uid], post[uid]) for uid in ("uav_00", "uav_01", "uav_02")]
    pair = [{"episode_id": "synthetic", "step": 1, "source": "uav_00", "target": "uav_01",
             "distance_m": 0.0, "time_fraction": .5}]
    result = audit_pair_rows(pair, navigation, minimum_distance=2.0)
    assert any("pair_minimum" in error for error in result["errors"])


def test_flow_audit_rejects_wrong_direction_bandwidth_cut_and_reused_packet():
    navigation = [_nav("uav_00", [0, 0, 2], [0, 0, 2]), _nav("uav_01", [4, 0, 2], [4, 0, 2]), _nav("uav_02", [8, 0, 2], [8, 0, 2])]
    valid = _flow("uav_00", "uav_01")
    valid_rows = [_flow(source, target) for source, target in (("uav_00", "uav_01"), ("uav_00", "uav_02"), ("uav_01", "uav_00"), ("uav_01", "uav_02"), ("uav_02", "uav_00"), ("uav_02", "uav_01"))]
    arrivals = [{"episode_id": "synthetic", "step": 1, "source": row["source"], "target": row["target"], "status": "arrived"} for row in valid_rows]
    valid_result = audit_flow_rows(valid_rows, navigation, arrivals)
    assert not [error for error in valid_result["errors"] if error != "flow_row_count"]
    mutated = copy.deepcopy(valid)
    mutated["target"], mutated["requested_bandwidth"] = "uav_02", 0.0
    result = audit_flow_rows([mutated], navigation, [])
    assert any("flow_direction" in error or "flow_requested_bandwidth" in error for error in result["errors"])
    cut = _flow("uav_00", "uav_02", variant="cut", requested=.5, applied=.5)
    cut["step"] = 20
    result = audit_flow_rows([cut], [dict(row, step=20) for row in navigation], [])
    assert any("flow_requested_bandwidth" in error or "flow_applied_bandwidth" in error for error in result["errors"])
    reused = _flow("uav_00", "uav_01", previous=.2)
    reused["new_since_last_consumed"] = True
    reused["reused"] = True
    result = audit_flow_rows([reused], navigation, [{"episode_id": "synthetic", "step": 1, "source": "uav_00", "target": "uav_01", "status": "arrived"}])
    assert any("flow_new_packet" in error or "flow_reused" in error for error in result["errors"])


def test_flow_auditor_allows_radio_loss_and_cached_unconsumed_packet():
    navigation = [_nav(uid, position, position, step=step) for step in (1, 2)
                  for uid, position in (("uav_00", [0, 0, 2]), ("uav_01", [4, 0, 2]), ("uav_02", [8, 0, 2]))]
    first = _flow("uav_00", "uav_01")
    second = copy.deepcopy(first)
    second.update(step=2, message_age_s=.2, newly_arrived_this_step=False, transmitted=False)
    arrivals = [{"episode_id": "synthetic", "step": 1, "source": "uav_00", "target": "uav_01", "status": "arrived"}]
    result = audit_flow_rows([first, second], navigation, arrivals)
    assert result["errors"] == ["flow_row_count"]
    second["actor_node_covariance_diagonal"][14] = 10.
    result = audit_flow_rows([first, second], navigation, arrivals)
    assert any("flow_actor_node_covariance" in error for error in result["errors"])


def test_navigation_auditor_uses_full_fifteen_state_covariance():
    row = _navigation_row()
    row["ekf_state_pre"] = np.r_[row["ekf_state_pre"], np.zeros(9)]
    assert audit_navigation_row(row) == []
    row["ekf_covariance_post"][14, 14] = -1.
    assert "covariance_psd" in audit_navigation_row(row)
