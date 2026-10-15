"""Pure mutation tests for the fixed-navigation independent auditor."""
from __future__ import annotations

import numpy as np

from src.analysis.single_uav_fixed_navigation_audit import audit_group, audit_row
from src.control.fixed_navigation import FixedNavigationConfig, fixed_navigation_command


def _case(*, at_goal: bool = False) -> dict:
    goal = [10.0, 0.0, 10.0]
    start = goal if at_goal else [0.0, 0.0, 10.0]
    return {
        "scenario_id": "synthetic_fixednav_case",
        "kind": "normal",
        "regime": "toward",
        "noise_seed": 7,
        "scenario": {"initial_positions": {"uav_00": start}, "goals": {"uav_00": goal}, "obstacle_specs": [], "obstacles": []},
        "references": [goal],
        "geometry_sha256": "synthetic",
        "safety2_enabled": True,
        "assist_mask": "00000",
        "pass_commit": False,
        "lateral_assist": False,
        "radial_assist": False,
        "goal_recovery": False,
        "terminal_capture": False,
    }


def _row(case: dict, *, step: int = 1, at_goal: bool = False) -> dict:
    config = FixedNavigationConfig()
    goal = np.asarray(case["references"][0], dtype=float)
    if at_goal:
        estimate = np.r_[goal, [0.0, 0.0, 0.0]]
        pre_position = goal.copy()
        pre_velocity = np.zeros(3)
    else:
        estimate = np.array([0.0, 0.0, 10.0, 4.0, 0.0, 0.0])
        pre_position = np.array([0.0, 0.0, 10.0])
        pre_velocity = np.array([4.0, 0.0, 0.0])
    command = fixed_navigation_command(estimate, goal, config)
    requested = np.asarray(command["command_delta_v_mps"], dtype=float)
    post_velocity = pre_velocity + requested
    post_position = pre_position + (2.0 * pre_velocity + requested) * 0.1
    distance = float(np.linalg.norm(post_position - goal))
    speed = float(np.linalg.norm(post_velocity))
    direction = (goal - estimate[:3]) / max(float(np.linalg.norm(goal - estimate[:3])), 1e-12)
    witness = bool(np.dot(pre_velocity, direction) > 0.05 and np.dot(requested, direction) < -1e-4
                   and -np.dot(requested, direction) > 0.001 and speed < np.linalg.norm(pre_velocity))
    return {
        "phase": "development", "scenario_id": case["scenario_id"], "noise_seed": 7,
        "geometry_sha256": case["geometry_sha256"], "config_sha256": "config",
        "regime": case["regime"], "step": step, "controller_mode": "fixed_navigation",
        "safety2_enabled": True, "assist_mask": "00000", "pass_commit": False,
        "lateral_assist": False, "radial_assist": False, "goal_recovery": False, "terminal_capture": False,
        "ekf_position_pre_m": estimate[:3], "ekf_velocity_pre_mps": estimate[3:],
        "reference_position_m": goal, "reference_index": 0,
        "desired_velocity_mps": command["desired_velocity_mps"], "requested_acceleration_mps2": command["requested_acceleration_mps2"],
        "raw_delta_v_mps": command["raw_delta_v_mps"], "axis_limited_delta_v_mps": command["axis_limited_delta_v_mps"],
        "requested_delta_v_mps": requested, "env_input_command_mps": requested,
        "reported_applied_delta_v_mps": requested, "actual_delta_v_mps": requested,
        "execution_correction_delta_v_mps": np.zeros(3), "actual_acceleration_mps2": requested / 0.2,
        "nominal_position_after_command_m": post_position,
        "true_position_pre_m": pre_position, "true_velocity_pre_mps": pre_velocity,
        "true_position_post_m": post_position, "true_velocity_post_mps": post_velocity,
        "goal_distance_m_actual": distance, "speed_mps_actual": speed,
        "strict_hold_steps": step if at_goal else 0, "strict_success": bool(at_goal and step >= 5),
        "strict_physical_pass_complete": False, "qualifying": bool(at_goal), "braking_witness": witness,
        "braking_before_first_hold": bool(witness and not at_goal),
        "requested_projection_mps": float(np.dot(requested, direction)),
        "pre_true_projection_mps": float(np.dot(pre_velocity, direction)),
        "actual_projected_speed_reduction_mps": float(-np.dot(requested, direction)),
        "boundary_projected": False, "safety_projected": False, "environment_action_clipped": False,
        "controller_limited": bool(command["controller_limited"]), "collision": False, "physical_contact": False,
        "segment_intersects": False, "segment_clearance_lower_bound_m": None, "legal_action_and_acceleration": True,
        "terminated": bool(at_goal and step == 5), "truncated": False, "termination_reason": "",
    }


def test_audit_row_detects_actuator_semantic_mutation() -> None:
    case = _case()
    row = _row(case)
    row["reported_applied_delta_v_mps"] = np.asarray(row["reported_applied_delta_v_mps"]) + [0.01, 0.0, 0.0]
    result = audit_row(row, case, FixedNavigationConfig())
    assert not result["ok"]
    assert "actuator_semantic_error" in result["errors"]


def test_audit_row_detects_mode_and_fixed_command_mutation() -> None:
    case = _case()
    row = _row(case)
    row["controller_mode"] = "coast_zero_delta_v"
    result = audit_row(row, case, FixedNavigationConfig())
    assert not result["ok"]
    assert "controller_mode" in result["errors"]
    assert "coast_zero_requested" in result["errors"]


def test_audit_row_detects_braking_witness_mutation() -> None:
    case = _case()
    row = _row(case)
    assert row["braking_witness"] is True
    row["braking_witness"] = False
    result = audit_row(row, case, FixedNavigationConfig())
    assert not result["ok"]
    assert "braking_witness" in result["errors"]


def test_audit_group_detects_strict_five_hold_mutation() -> None:
    case = _case(at_goal=True)
    rows = [_row(case, step=step, at_goal=True) for step in range(1, 6)]
    result = audit_group(rows, case, FixedNavigationConfig(), config_hash="config")
    assert result["ok"]
    rows[4]["strict_hold_steps"] = 4
    result = audit_group(rows, case, FixedNavigationConfig(), config_hash="config")
    assert not result["ok"]
    assert any("strict_hold_steps" in error for error in result["errors"])
