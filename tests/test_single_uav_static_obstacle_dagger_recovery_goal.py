"""Contract tests for the static-obstacle recovery/goal DAgger entry point."""

import json
from pathlib import Path

import numpy as np

from src.experiments.single_uav_static_obstacle_dagger_recovery_goal import (
    ActorConstrainedBoundaryAwareGeometricObstacleOracleController,
    NEW_NODE_DIM,
    _balance_demo_kinds,
    _build_pool,
    _evenly_cap,
    _reference_normal_pool,
    _static_variant,
)


def _config():
    from src.experiments.single_uav_obstacle_stage2br2_followup import (
        DEFAULT_STAGE2A_CHECKPOINT,
        _load_policy,
    )

    _payload, _trainer, config, _identity = _load_policy(
        Path(DEFAULT_STAGE2A_CHECKPOINT),
        output_dir=Path(".tmp") / "recovery-goal-contract-test",
    )
    return config


def test_recovery_goal_pool_is_balanced_and_split_disjoint():
    config = _config()
    train = _build_pool(
        config,
        split="train",
        normal_count=4,
        static_count=4,
        seed_base=7100000,
        round_index=1,
    )
    validation = _build_pool(
        config,
        split="validation",
        normal_count=2,
        static_count=4,
        seed_base=7200000,
        round_index=1,
    )
    assert len(train) == 8
    assert len(validation) == 6
    assert sum(item["kind"] == "normal_far_distance" for item in train) == 4
    assert sum(item["kind"] == "static_obstacle" for item in train) == 4
    train_hashes = {item["scenario"]["recovery_goal"]["scenario_hash"] for item in train}
    validation_hashes = {item["scenario"]["recovery_goal"]["scenario_hash"] for item in validation}
    assert train_hashes.isdisjoint(validation_hashes)
    train_pairs = {item["pair_id"] for item in train if item["kind"] == "static_obstacle"}
    assert len(train_pairs) == 2
    assert all(
        sum(item["pair_id"] == pair_id for item in train if item["kind"] == "static_obstacle") == 2
        for pair_id in train_pairs
    )


def test_pool_keeps_actor_observation_contract_metadata():
    config = _config()
    pool = _build_pool(
        config,
        split="train",
        normal_count=2,
        static_count=2,
        seed_base=7300000,
        round_index=2,
    )
    assert config.node_feature_dim == NEW_NODE_DIM
    static = next(item for item in pool if item["kind"] == "static_obstacle")
    assert static["scenario"]["stage2br"]["direct_path_crosses_static_obstacle"] is True
    assert static["scenario"]["stage2br"]["side"] in {"left", "right"}
    assert static["scenario"]["recovery_goal"]["noise_seed"] != static["scenario_seed"]


def test_phase_balancing_caps_each_episode_phase_without_cross_episode_mixing():
    records = []
    for episode_id in ("e1", "e2"):
        for phase in ("post_bypass", "recovery"):
            for step in range(30):
                records.append({"episode_id": episode_id, "phase": phase, "step_index": step})
    selected = _evenly_cap(records, cap=6)
    assert len(selected) == 24
    for episode_id in ("e1", "e2"):
        for phase in ("post_bypass", "recovery"):
            rows = [row for row in selected if row["episode_id"] == episode_id and row["phase"] == phase]
            assert len(rows) == 6


def test_demo_kind_balance_preserves_phase_coverage_and_equalizes_steps():
    records = []
    for kind in ("normal_far_distance", "static_obstacle"):
        phases = ("normal",) if kind == "normal_far_distance" else ("bypass", "recovery", "post_bypass")
        for episode_index in range(2):
            for phase in phases:
                for step in range(20):
                    records.append(
                        {
                            "kind": kind,
                            "episode_id": f"{kind}-{episode_index}",
                            "phase": phase,
                            "step_index": step + (100 if phase != "normal" else 0),
                        }
                    )
    selected = _balance_demo_kinds(records, cap=12)
    counts = {kind: sum(row["kind"] == kind for row in selected) for kind in ("normal_far_distance", "static_obstacle")}
    assert counts["normal_far_distance"] == counts["static_obstacle"] == 24
    for episode_index in range(2):
        static_phases = {row["phase"] for row in selected if row["kind"] == "static_obstacle" and row["episode_id"] == f"static_obstacle-{episode_index}"}
        assert static_phases == {"bypass", "recovery", "post_bypass"}


def test_normal_evaluation_reuses_reference_physical_scenes(tmp_path):
    config = _config()
    episode_root = tmp_path / "test" / "episode_500001"
    episode_root.mkdir(parents=True)
    generation = {
        "seed": 2026,
        "uav_count": 1,
        "initial_positions": {"uav_00": [10.0, 20.0, 20.0]},
        "initial_velocities": {"uav_00": [0.0, 0.0, 0.0]},
        "goals": {"uav_00": [30.0, 20.0, 20.0]},
        "obstacle_specs": [],
    }
    (episode_root / "scenario.json").write_text(
        json.dumps({"seed": 1102026, "scenario_generation": generation}),
        encoding="utf-8",
    )
    pool = _reference_normal_pool(
        config,
        count=1,
        reference_root=tmp_path / "test",
        round_index=99,
    )
    assert pool[0]["scenario"]["initial_positions"]["uav_00"] == [10.0, 20.0, 20.0]
    assert pool[0]["scenario"]["goals"]["uav_00"] == [30.0, 20.0, 20.0]
    assert pool[0]["scenario_seed"] == 2026
    assert pool[0]["noise_seed"] == 1102026
    assert pool[0]["scenario"]["recovery_goal"]["reference_scenario_file"].endswith(
        "episode_500001\\scenario.json"
    )


def test_obstacle_oracle_teacher_action_matches_actor_component_box():
    from src.experiments.formal import _make_environment

    config = _config()
    seed = 7400001
    scenario = _static_variant(
        config,
        split="train",
        ordinal=0,
        scenario_seed=seed,
        pair_id="actor_constraint_contract",
        side="left",
        round_index=1,
    )
    environment = _make_environment(
        config,
        seed,
        metric_scales=config.metric_scales,
        safety_projection_enabled=True,
    )
    observations, _info = environment.reset(seed=seed, scenario_config=scenario)
    oracle = ActorConstrainedBoundaryAwareGeometricObstacleOracleController(environment)
    payload, control = oracle.action(observations)
    action = np.asarray(payload["uav_00"]["delta_v"], dtype=float)
    component_limit = float(control["actor_component_limit_mps"])
    assert np.all(np.abs(action) <= component_limit + 1.0e-9)
    assert np.linalg.norm(action) <= environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s + 1.0e-9
    assert control["teacher_action_componentwise_legal"] == 1
    assert control["truth_state_read_for_control"] is False
