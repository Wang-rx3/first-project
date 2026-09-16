from __future__ import annotations

import numpy as np

from src.experiments.formal import (
    FormalExperimentConfig,
    _apply_single_uav_braking_curriculum,
    _single_uav_braking_curriculum_kind,
    _single_uav_terminal_curriculum_distance,
)


def _config(
    tmp_path,
    *,
    fraction: float = 0.5,
    terminal_fraction: float = 0.0,
    train_episodes: int = 10,
) -> FormalExperimentConfig:
    return FormalExperimentConfig(
        output_dir=tmp_path,
        uav_count=1,
        train_episodes=train_episodes,
        single_uav_learning_mode="loc_goal_v2",
        single_uav_speed_control_mode="braking_distance",
        single_uav_braking_curriculum_fraction=fraction,
        single_uav_terminal_braking_curriculum_fraction=terminal_fraction,
    )


def _scenario() -> dict[str, object]:
    return {
        "bounds_min_m": [0.0, 0.0, 2.0],
        "bounds_max_m": [200.0, 200.0, 60.0],
        "initial_positions": {"uav_00": [0.0, 0.0, 10.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [20.0, 0.0, 10.0]},
        "obstacle_specs": [],
    }


def test_curriculum_creates_controlled_local_braking_starts(tmp_path) -> None:
    source = _scenario()
    result = _apply_single_uav_braking_curriculum(
        _config(tmp_path), source, split="train", episode_index=1
    )

    metadata = result["single_uav_braking_curriculum"]
    assert metadata["enabled"] is True
    assert metadata["requested_distance_m"] == 5.0
    assert np.isclose(
        np.linalg.norm(
            np.asarray(result["goals"]["uav_00"])
            - np.asarray(result["initial_positions"]["uav_00"])
        ),
        5.0,
    )
    velocity = np.asarray(result["initial_velocities"]["uav_00"], dtype=float)
    assert velocity[0] > 0.0
    assert np.linalg.norm(velocity) > 0.5
    assert result["obstacle_specs"] == source["obstacle_specs"]
    assert metadata["source_initial_velocity_mps"] == [0.0, 0.0, 0.0]


def test_curriculum_does_not_change_validation_or_late_training_starts(tmp_path) -> None:
    source = _scenario()
    config = _config(tmp_path)

    validation = _apply_single_uav_braking_curriculum(
        config, source, split="validation", episode_index=0
    )
    late_train = _apply_single_uav_braking_curriculum(
        config, source, split="train", episode_index=9
    )

    assert validation == source
    assert late_train == source


def test_terminal_curriculum_uses_ten_slot_schedule_and_progressive_distances(
    tmp_path,
) -> None:
    source = _scenario()
    config = _config(
        tmp_path,
        fraction=0.4,
        terminal_fraction=0.3,
        train_episodes=100,
    )
    kinds = [
        _single_uav_braking_curriculum_kind(index, 0.4, 0.3)
        for index in range(100)
    ]
    assert kinds.count("standard") == 10
    assert kinds.count("terminal") == 30
    assert kinds.count(None) == 60
    assert kinds[:10] == [
        "standard",
        "terminal",
        "terminal",
        "terminal",
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    assert _single_uav_terminal_curriculum_distance(4, 100)[0] == 1.5
    assert _single_uav_terminal_curriculum_distance(34, 100)[0] == 3.0
    assert _single_uav_terminal_curriculum_distance(94, 100)[0] == 8.0

    result = _apply_single_uav_braking_curriculum(
        config,
        source,
        split="train",
        episode_index=1,
        phase="train",
    )

    metadata = result["single_uav_braking_curriculum"]
    assert metadata["enabled"] is True
    assert metadata["curriculum_kind"] == "terminal"
    assert metadata["requested_distance_m"] == 1.5
    assert metadata["terminal_curriculum_stage_index"] == 0
    assert metadata["terminal_curriculum_level_index"] == 0
    assert metadata["terminal_curriculum_distance_range_m"] == [1.5, 3.0]
    assert metadata["terminal_curriculum_schedule"]["cycle_length_episodes"] == 10
    assert metadata["terminal_curriculum_schedule"]["braking_budget_slots_per_cycle"] == 4
    assert metadata["terminal_curriculum_schedule"]["standard_slots_per_cycle"] == 1
    assert metadata["terminal_curriculum_schedule"]["terminal_slots_per_cycle"] == 3
    assert metadata["terminal_curriculum_schedule"]["untouched_far_slots_per_cycle"] == 6
    assert metadata["terminal_curriculum_schedule"]["far_fraction_realized"] == 0.6
    assert metadata["terminal_curriculum_schedule"]["non_terminal_fraction_realized"] == 0.7
    assert np.isclose(
        np.linalg.norm(
            np.asarray(result["goals"]["uav_00"])
            - np.asarray(result["initial_positions"]["uav_00"])
        ),
        1.5,
    )
    assert np.linalg.norm(np.asarray(result["initial_velocities"]["uav_00"])) > 0.5

    # A terminal slot in the next cycle remains a terminal example, while the
    # warm-up phase and validation keep their original starts.
    terminal_next = _apply_single_uav_braking_curriculum(
        config,
        source,
        split="train",
        episode_index=11,
        phase="train",
    )
    assert terminal_next["single_uav_braking_curriculum"]["curriculum_kind"] == "terminal"
    warmup = _apply_single_uav_braking_curriculum(
        config,
        source,
        split="train",
        episode_index=4,
        phase="warmup",
    )
    assert warmup == source
    validation = _apply_single_uav_braking_curriculum(
        config,
        source,
        split="validation",
        episode_index=2,
        phase="validation",
    )
    assert validation == source
