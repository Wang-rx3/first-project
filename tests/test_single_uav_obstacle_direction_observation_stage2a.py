from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.formal import FormalExperimentConfig
from src.experiments.single_uav_obstacle_direction_observation_stage2a import (
    NEW_NODE_DIM,
    OLD_NODE_DIM,
    migrate_central_normalizer_state,
    migrate_observation_normalizer_state,
    migrate_policy_state,
)
from src.experiments.single_uav_obstacle_observation_audit import probe_scenarios
from src.experiments.single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from src.learning.mappo import central_state_vector_dim
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig
from src.simulation.motion_model import MotionConfig
from src.simulation.obstacle_model import SphereObstacle, point_obstacle_distance
from src.simulation.environment import RewardConfig


def _direction_environment() -> MultiUAVJointEnv:
    return MultiUAVJointEnv(
        SimulationConfig(
            uav_count=1,
            seed=2026,
            reward=RewardConfig(single_uav_speed_control_mode="braking_distance"),
            single_uav_obstacle_direction_features_enabled=True,
        )
    )


def _direction_scenario(case: str) -> dict:
    return probe_scenarios()[case]


def test_formal_node_dimension_is_opt_in() -> None:
    legacy = FormalExperimentConfig(
        output_dir=Path("."),
        uav_count=1,
        single_uav_speed_control_mode="braking_distance",
    )
    extended = FormalExperimentConfig(
        output_dir=Path("."),
        uav_count=1,
        single_uav_speed_control_mode="braking_distance",
        single_uav_obstacle_direction_features_enabled=True,
    )
    assert legacy.node_feature_dim == OLD_NODE_DIM
    assert extended.node_feature_dim == NEW_NODE_DIM


def test_direction_features_have_expected_signs_and_clearance_slot_is_unchanged() -> None:
    environment = _direction_environment()
    front_obs, _ = environment.reset(seed=2026, scenario_config=_direction_scenario("front"))
    left_obs, _ = environment.reset(seed=2026, scenario_config=_direction_scenario("left"))
    right_obs, _ = environment.reset(seed=2026, scenario_config=_direction_scenario("right"))
    behind_obs, _ = environment.reset(seed=2026, scenario_config=_direction_scenario("behind"))
    front = np.asarray(front_obs["uav_00"]["node_features"], dtype=float).reshape(-1)
    left = np.asarray(left_obs["uav_00"]["node_features"], dtype=float).reshape(-1)
    right = np.asarray(right_obs["uav_00"]["node_features"], dtype=float).reshape(-1)
    behind = np.asarray(behind_obs["uav_00"]["node_features"], dtype=float).reshape(-1)
    assert front.shape == (NEW_NODE_DIM,)
    assert np.isfinite(front).all()
    assert left[40] > 0.0 and right[40] < 0.0
    assert front[39] > 0.0 and behind[39] < 0.0
    assert np.isclose(left[33], right[33])
    assert np.isclose(front[33], behind[33])


def test_world_to_yaw_body_frame_uses_forward_and_left_convention() -> None:
    assert np.allclose(
        MultiUAVJointEnv._world_to_uav_yaw_frame(np.asarray([1.0, 0.0, 0.0]), 0.0),
        [1.0, 0.0, 0.0],
    )
    assert np.allclose(
        MultiUAVJointEnv._world_to_uav_yaw_frame(np.asarray([0.0, 1.0, 0.0]), np.pi / 2.0),
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )
    assert np.allclose(
        MultiUAVJointEnv._world_to_uav_yaw_frame(np.asarray([-1.0, 0.0, 0.0]), np.pi / 2.0),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_tie_break_selects_smallest_obstacle_id_and_matches_clearance() -> None:
    environment = _direction_environment()
    scenario = _direction_scenario("front")
    scenario["obstacle_specs"] = [
        {
            "obstacle_id": "b",
            "obstacle_type": "static",
            "geometry_type": "sphere",
            "center_m": [56.0, 100.0, 20.0],
            "radius_m": 1.0,
            "velocity_mps": [0.0, 0.0, 0.0],
            "active": 1,
            "physical_collision_enabled": True,
        },
        {
            "obstacle_id": "a",
            "obstacle_type": "static",
            "geometry_type": "sphere",
            "center_m": [50.0, 106.0, 20.0],
            "radius_m": 1.0,
            "velocity_mps": [0.0, 0.0, 0.0],
            "active": 1,
            "physical_collision_enabled": True,
        },
    ]
    observations, _ = environment.reset(seed=2026, scenario_config=scenario)
    node = np.asarray(observations["uav_00"]["node_features"], dtype=float).reshape(-1)
    assert np.isclose(node[33], 5.0)
    assert np.allclose(node[39:42], [0.0, 6.0, 0.0])
    position = np.asarray(environment.ekfs["uav_00"].state[:3], dtype=float)
    selected = environment._nearest_physical_obstacle(position, environment.obstacles)
    assert selected is not None and selected.obstacle_id == "a"
    assert np.isclose(node[33], point_obstacle_distance(position, selected))


def test_no_obstacle_direction_vector_is_finite_zero() -> None:
    environment = _direction_environment()
    scenario = _direction_scenario("front")
    scenario["obstacles"] = []
    scenario.pop("obstacle_specs", None)
    observations, _ = environment.reset(seed=2026, scenario_config=scenario)
    node = np.asarray(observations["uav_00"]["node_features"], dtype=float).reshape(-1)
    assert node.shape == (NEW_NODE_DIM,)
    assert np.isfinite(node).all()
    assert np.allclose(node[39:42], 0.0)
    assert np.allclose(node[34:36], 0.0)


def test_policy_migration_copies_old_slices_and_zeros_new_columns() -> None:
    source = {
        "gat.layers.0.node_projection.weight": torch.arange(12.0).reshape(3, 4),
        "gat.layers.0.residual.weight": torch.arange(12.0, 24.0).reshape(3, 4),
        "critic.0.weight": torch.arange(2.0 * 7.0).reshape(2, 7),
        "other": torch.ones(2),
    }
    template = {
        "gat.layers.0.node_projection.weight": torch.zeros(3, 7),
        "gat.layers.0.residual.weight": torch.zeros(3, 7),
        "critic.0.weight": torch.zeros(2, 10),
        "other": torch.zeros(2),
    }
    migrated, report = migrate_policy_state(
        source,
        template,
        old_node_dim=4,
        new_node_dim=7,
        uav_count=1,
        old_central_dim=7,
        new_central_dim=10,
    )
    assert torch.equal(migrated["gat.layers.0.node_projection.weight"][:, :4], source["gat.layers.0.node_projection.weight"])
    assert torch.equal(migrated["gat.layers.0.node_projection.weight"][:, 4:], torch.zeros(3, 3))
    assert torch.equal(migrated["critic.0.weight"][:, :4], source["critic.0.weight"][:, :4])
    assert torch.equal(migrated["critic.0.weight"][:, 4:7], torch.zeros(2, 3))
    assert torch.equal(migrated["critic.0.weight"][:, 7:], source["critic.0.weight"][:, 4:])
    assert len(report["shape_changed_parameters"]) == 3


def test_normalizer_migration_preserves_old_entries_and_uses_neutral_new_entries() -> None:
    source = {
        "eps": 1e-6,
        "clip": 10.0,
        "normalizers": {
            "node_features": {
                "eps": 1e-6,
                "clip": 10.0,
                "mean": list(np.arange(39.0)),
                "scale": list(np.arange(1.0, 40.0)),
                "count": list(np.arange(39, dtype=np.int64)),
            }
        },
    }
    migrated, report = migrate_observation_normalizer_state(source)
    node = migrated["normalizers"]["node_features"]
    assert node["mean"][:39] == source["normalizers"]["node_features"]["mean"]
    assert node["scale"][:39] == source["normalizers"]["node_features"]["scale"]
    assert node["count"][:39] == source["normalizers"]["node_features"]["count"]
    assert node["mean"][39:] == [0.0, 0.0, 0.0]
    assert node["scale"][39:] == [1.0, 1.0, 1.0]
    assert node["count"][39:] == [0, 0, 0]
    assert report["new_entries_neutral"] is True


def test_central_normalizer_migration_inserts_neutral_node_slots() -> None:
    old_dim = central_state_vector_dim(1, 39, 11, 8, False)
    new_dim = central_state_vector_dim(1, 42, 11, 8, False)
    source = {
        "eps": 1e-6,
        "clip": 10.0,
        "mean": list(np.arange(old_dim, dtype=float)),
        "scale": list(np.arange(1.0, old_dim + 1.0)),
        "count": list(np.arange(old_dim, dtype=np.int64)),
    }
    migrated, report = migrate_central_normalizer_state(
        source,
        uav_count=1,
        old_node_dim=39,
        new_node_dim=42,
        edge_feature_dim=11,
        max_obstacles=8,
        include_remaining_time=False,
    )
    assert len(migrated["mean"]) == new_dim
    assert migrated["mean"][39:42] == [0.0, 0.0, 0.0]
    assert migrated["scale"][39:42] == [1.0, 1.0, 1.0]
    assert migrated["count"][39:42] == [0, 0, 0]
    assert report["old_mean_preserved_exact"] is True
    assert report["new_count_slots_neutral"] is True


def test_old_stage1_checkpoint_loads_with_flag_false_without_update(tmp_path: Path) -> None:
    checkpoint = Path(
        "data/synthetic/single_uav_static_obstacle_avoidance_stage1_formal_seed2026/"
        "scale_1/checkpoints/final.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("historical Stage 1 checkpoint is not available")
    import torch as torch_module

    payload = torch_module.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=tmp_path, progress=False)
    trainer, _identity, _checkpoint_identity = _load_frozen_trainer(
        payload, config, metric_scales=config.metric_scales
    )
    assert config.single_uav_obstacle_direction_features_enabled is False
    assert trainer.config.node_feature_dim == OLD_NODE_DIM
    assert _state_dict_hash(trainer.policy.state_dict()) == _state_dict_hash(payload["policy_state_dict"])
