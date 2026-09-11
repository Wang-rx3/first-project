from __future__ import annotations

import numpy as np

from src.experiments.formal import (
    FormalExperimentConfig,
    _apply_single_uav_braking_curriculum,
)


def _config(tmp_path, *, fraction: float = 0.5) -> FormalExperimentConfig:
    return FormalExperimentConfig(
        output_dir=tmp_path,
        uav_count=1,
        train_episodes=10,
        single_uav_learning_mode="loc_goal_v2",
        single_uav_speed_control_mode="braking_distance",
        single_uav_braking_curriculum_fraction=fraction,
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
