from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.analysis.single_uav_current_run_visualization import (
    aggregate_episode_returns,
    funnel_counts,
    parse_mappo_loss_rows,
    phase_distance_counts,
    radial_tangential,
    same_scenario_pairing,
    vector_from_row,
)


def test_reward_aggregation_does_not_double_accumulate() -> None:
    returns = [2.0, -1.0, 3.0]
    assert returns == [2.0, -1.0, 3.0]
    assert aggregate_episode_returns(returns) == [2.0, 0.5, 1.3333333333333333]


def test_mappo_parser_does_not_read_bc_loss() -> None:
    rows = [
        {"update": "1", "actor_loss": "-0.2", "critic_loss": "1.0", "bc_loss": "0.001"},
        {"update": "2", "bc_loss": "0.0001"},
    ]
    parsed = parse_mappo_loss_rows(rows)
    assert len(parsed) == 2
    assert "bc_loss" not in parsed[0]
    assert parsed[1]["actor_loss"] == ""


def test_same_scenario_pairing_is_exact() -> None:
    rows = [
        {"policy": "baseline_rl", "scenario_hash": "a"},
        {"policy": "bc_actor", "scenario_hash": "a"},
        {"policy": "oracle_teacher", "scenario_hash": "a"},
    ]
    assert same_scenario_pairing(rows, ("baseline_rl", "bc_actor", "oracle_teacher"))["paired"]
    rows.append({"policy": "bc_actor", "scenario_hash": "b"})
    assert not same_scenario_pairing(rows, ("baseline_rl", "bc_actor", "oracle_teacher"))["paired"]


def test_distance_bin_boundaries() -> None:
    from src.analysis.single_uav_current_run_visualization import _distance_bin

    assert _distance_bin(8.0001) == ">8m"
    assert _distance_bin(8.0) == "8–5m"
    assert _distance_bin(5.0) == "5–3m"
    assert _distance_bin(3.0) == "3–1m"
    assert _distance_bin(1.0) == "≤1m"


def test_phase_distance_counts() -> None:
    rows = [
        {"split": "train", "collection_mode": "student_closed_loop", "phase": "recovery", "goal_distance_m": "2"},
        {"split": "train", "collection_mode": "student_closed_loop", "phase": "recovery", "goal_distance_m": "2"},
        {"split": "validation", "collection_mode": "teacher_bootstrap", "phase": "bypass", "goal_distance_m": "6"},
    ]
    counts = phase_distance_counts(rows)
    assert {row["count"] for row in counts} == {1, 2}
    recovery = next(row for row in counts if row["phase"] == "recovery")
    assert recovery["distance_bin"] == "3–1m"
    assert recovery["fraction"] == 1.0


def test_vector_decomposition() -> None:
    radial, tangential = radial_tangential([1.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assert np.isclose(radial, 1.0)
    assert np.isclose(tangential, 1.0)


def test_radial_tangential_is_nonnegative_tangent() -> None:
    radial, tangential = radial_tangential([-2.0, 3.0, 0.0], [1.0, 0.0, 0.0])
    assert radial == -2.0
    assert tangential == 3.0


def test_vector_from_row_accepts_physical_action_unit_suffix() -> None:
    vector = vector_from_row(
        {
            "raw_actor_delta_v_x_mps": "1.0",
            "raw_actor_delta_v_y_mps": "-2.0",
            "raw_actor_delta_v_z_mps": "0.5",
        },
        "raw_actor_delta_v",
    )
    assert np.allclose(vector, [1.0, -2.0, 0.5])


def test_funnel_counts_are_monotone() -> None:
    rows = [
        {"policy": "bc_actor", "total": "1", "passed_obstacle": "1", "post_bypass_recovery": "1", "entered_8m": "1", "entered_5m": "1", "entered_3m": "0", "entered_1m": "0", "stable_goal": "0"},
        {"policy": "bc_actor", "passed_obstacle": "0", "post_bypass_recovery": "0", "entered_8m": "1", "entered_5m": "0", "entered_3m": "0", "entered_1m": "0", "stable_goal": "0"},
    ]
    counts = funnel_counts(rows, "bc_actor")
    values = [row["episode_count"] for row in counts]
    assert values == sorted(values, reverse=True)


def test_missing_optional_loss_field_is_not_filled_with_zero() -> None:
    parsed = parse_mappo_loss_rows([{"update": "1", "actor_loss": "-0.2"}])
    assert parsed[0]["critic_loss"] == ""
    assert parsed[0]["entropy"] == ""


def test_visualization_helpers_do_not_modify_training_source() -> None:
    path = Path("src/experiments/single_uav_external_recovery_method_study.py")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _ = aggregate_episode_returns([1.0, 2.0])
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
