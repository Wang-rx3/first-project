import pytest

from src.experiments.single_uav_navigation_preserving_finetune import (
    FINE_TUNING_TERMINAL_RANGE_M,
    fine_tuning_episode_kind,
    fine_tuning_schedule_counts,
    fine_tuning_terminal_distance_values,
)


def test_navigation_preserving_schedule_is_ninety_ten_for_formal_round():
    counts = fine_tuning_schedule_counts(100)

    assert counts["normal_far_distance"] == 90
    assert counts["terminal"] == 10
    assert counts["standard_braking"] == 0
    assert counts["normal_far_distance_rate"] == pytest.approx(0.90)
    assert counts["terminal_rate"] == pytest.approx(0.10)


def test_navigation_preserving_schedule_rounds_small_smoke_deterministically():
    counts = fine_tuning_schedule_counts(30)

    assert counts["normal_far_distance"] == 27
    assert counts["terminal"] == 3
    assert [fine_tuning_episode_kind(index) for index in range(10)].count("terminal") == 1


def test_terminal_fine_tuning_distance_grid_stays_in_five_to_eight_meters():
    values = fine_tuning_terminal_distance_values(10)

    assert values[0] == pytest.approx(FINE_TUNING_TERMINAL_RANGE_M[0])
    assert values[-1] == pytest.approx(FINE_TUNING_TERMINAL_RANGE_M[1])
    assert all(FINE_TUNING_TERMINAL_RANGE_M[0] <= value <= FINE_TUNING_TERMINAL_RANGE_M[1] for value in values)


def test_schedule_rejects_invalid_indices_and_counts():
    with pytest.raises(ValueError):
        fine_tuning_episode_kind(-1)
    with pytest.raises(ValueError):
        fine_tuning_schedule_counts(0)
