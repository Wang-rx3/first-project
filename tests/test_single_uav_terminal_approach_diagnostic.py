import pytest

from src.experiments.single_uav_terminal_approach_diagnostic import (
    _episode_segment_rows,
    approach_segment_for_distance,
    distance_bin_for_distance,
)


def test_terminal_approach_segments_have_explicit_half_open_boundaries():
    assert approach_segment_for_distance(8.0) == "outside_near_goal"
    assert approach_segment_for_distance(7.999) == "8_5m"
    assert approach_segment_for_distance(5.0) == "8_5m"
    assert approach_segment_for_distance(4.999) == "5_3m"
    assert approach_segment_for_distance(3.0) == "5_3m"
    assert approach_segment_for_distance(2.999) == "3_1m"
    assert approach_segment_for_distance(1.0) == "3_1m"
    assert approach_segment_for_distance(0.999) == "goal_radius"


def test_three_meter_transition_bins_are_unambiguous():
    assert distance_bin_for_distance(3.0) == "3_3p5m"
    assert distance_bin_for_distance(3.499999) == "3_3p5m"
    assert distance_bin_for_distance(3.5) == "3p5_5m"


def test_episode_segment_rows_record_progression_and_dwell_steps():
    distances = (7.0, 4.0, 2.0, 0.8)
    step_rows = [
        {
            "approach_segment": approach_segment_for_distance(distance),
            "goal_distance_m": distance,
            "terminal_radial_progress_active": float(1.0 <= distance < 3.0),
            "terminal_radial_progress_m": 0.1 if 1.0 <= distance < 3.0 else 0.0,
            "terminal_radial_progress_speed_excess_mps": 0.2 if 1.0 <= distance < 3.0 else 0.0,
            "terminal_radial_progress_speed_gate": 0.96 if 1.0 <= distance < 3.0 else 0.0,
            "terminal_radial_progress_signal": 0.19 if 1.0 <= distance < 3.0 else 0.0,
            "terminal_radial_progress_reward": 0.095 if 1.0 <= distance < 3.0 else 0.0,
        }
        for distance in distances
    ]

    rows = _episode_segment_rows(
        step_rows,
        policy="baseline",
        episode_index=1,
        seed=2026,
    )

    by_segment = {row["segment"]: row for row in rows}
    assert by_segment["8_5m"]["entered_segment"] == 1
    assert by_segment["8_5m"]["continued_to_next_segment"] == 1
    assert by_segment["5_3m"]["step_count"] == 1
    assert by_segment["5_3m"]["continued_to_next_segment"] == 1
    assert by_segment["3_1m"]["continued_to_next_segment"] == 1
    assert by_segment["3_1m"]["terminal_radial_progress_active_fraction"] == 1.0
    assert by_segment["3_1m"]["mean_terminal_radial_progress_reward"] == 0.095


@pytest.mark.parametrize("distance", (float("nan"), float("inf"), -1.0))
def test_invalid_or_out_of_range_distances_are_not_assigned_to_approach_segments(distance):
    assert approach_segment_for_distance(distance) in {
        "invalid",
        "below_zero",
    }
