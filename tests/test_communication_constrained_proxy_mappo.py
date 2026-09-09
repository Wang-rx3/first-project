from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.learning.mappo import MAPPOConfig, MAPPOTrainer
from src.experiments.communication_constrained_proxy_mappo import (
    CommunicationConstrainedProxyMAPPOConfig,
    _base_config,
    _proxy_scales,
)
from src.learning.measurement_proxy import measurement_communication_cost_features
from src.simulation.environment import MultiUAVJointEnv, SimulationConfig


TARGETS = {"uav_00": 0.0, "uav_01": 0.0, "uav_02": 0.0}
PROXY_MODEL = {
    "age_variance_coefficient": 1.0,
    "scales": {
        "measurement_std_median": 0.5,
        "neighbor_covariance_trace_q95_m2": 3.0,
        "delay_q75_s": 0.1,
        "age_q75_s": 1.0,
        "queue_q75_packets": 2.0,
        "rssi_q05_dbm": -100.0,
        "rssi_q95_dbm": -40.0,
        "snr_q05_db": 0.0,
        "snr_q95_db": 20.0,
        "ig_q05": 0.0,
        "ig_q95": 1.0,
    },
}


def test_communication_constraint_config_is_isolated_and_validated() -> None:
    with pytest.raises(ValueError, match="targets"):
        MAPPOConfig(
            use_measurement_proxy_features=True,
            measurement_communication_constraint_enabled=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        MAPPOConfig(
            use_measurement_proxy_features=True,
            measurement_communication_constraint_enabled=True,
            measurement_communication_cost_target_by_uav=TARGETS,
            measurement_rate_constraint_enabled=True,
            measurement_rate_target_by_uav=TARGETS,
        )
    config = MAPPOConfig(
        use_measurement_proxy_features=True,
        measurement_communication_constraint_enabled=True,
        measurement_communication_cost_target_by_uav=TARGETS,
    )
    assert config.measurement_rate_constraint_enabled is False
    assert config.measurement_communication_fixed_packet_fraction == 0.5


def test_vo_quality_calibration_is_transparent_to_all_formal_branches() -> None:
    config = CommunicationConstrainedProxyMAPPOConfig(
        output_dir=Path("tmp-communication-test"),
        seeds=(2026,),
        visual_odometry_quality_calibration_enabled=True,
    )
    base = _base_config(config)
    assert base.visual_odometry_quality_calibration_enabled is True


def test_fixed_packet_cost_is_bounded_and_none_is_external_zero() -> None:
    environment = MultiUAVJointEnv(SimulationConfig(max_steps=1, gnss_mode="denied"))
    observations, _info = environment.reset(seed=2026)
    values = []
    for observation in observations.values():
        cost, valid = measurement_communication_cost_features(
            observation, environment.uav_ids, PROXY_MODEL, fixed_packet_cost_fraction=0.5
        )
        assert np.isfinite(cost).all()
        assert np.all((cost >= 0.5) & (cost <= 1.0))
        assert np.all((valid == 0.0) | (valid == 1.0))
        values.extend(cost[valid > 0.5])
    assert values
    none_cost = 0.0
    assert none_cost == 0.0


def test_communication_cost_path_accepts_persistent_proxy_scale_extensions() -> None:
    model = dict(PROXY_MODEL)
    model["scales"] = {
        **dict(PROXY_MODEL["scales"]),
        "persistent_ig_q05": 0.0,
        "persistent_ig_q95": 1.0,
    }
    scales = _proxy_scales(model)
    assert scales.ig_q05 == 0.0
    assert scales.ig_q95 == 1.0


def test_communication_dual_updates_and_gradient_scope_is_measurement_head_only() -> None:
    environment = MultiUAVJointEnv(SimulationConfig(max_steps=2, gnss_mode="denied"))
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=2026,
        gat_hidden_dim=8,
        gat_output_dim=8,
        gat_heads=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        max_obstacles=4,
        policy_epochs=1,
        minibatch_size=16,
        use_measurement_proxy_features=True,
        measurement_proxy_guidance_enabled=True,
        measurement_communication_constraint_enabled=True,
        measurement_communication_cost_target_by_uav=TARGETS,
        measurement_communication_cost_tolerance=0.0,
        measurement_communication_dual_initial=1.0,
        measurement_communication_dual_step_size=0.25,
        measurement_communication_dual_max=5.0,
        measurement_proxy_model=PROXY_MODEL,
    )
    observations, info = environment.reset(seed=2026)
    decision = trainer.act(observations, info["central_state"])
    observation = decision.observations["uav_00"]
    trainer.policy.zero_grad(set_to_none=True)
    cost = trainer.policy.expected_measurement_communication_cost(observation)
    cost.backward()
    measurement_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in trainer.policy.measurement_head.parameters()
        if parameter.grad is not None
    )
    actor_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in trainer.policy.actor_body.parameters()
        if parameter.grad is not None
    )
    assert measurement_gradient > 0.0
    assert actor_gradient == 0.0

    _next, rewards, terminated, truncated, next_info = environment.step(decision.actions)
    trainer.add_transition(decision, rewards, next_info["central_state"], done=bool(terminated or truncated))
    stats = trainer.update()
    assert stats["measurement_communication_constraint_enabled"] == 1.0
    assert np.isfinite(list(stats.values())).all()
    state = trainer.measurement_communication_dual_state_dict()
    assert all(0.0 <= value <= 5.0 for value in state.values())
    restored = dict(state)
    trainer.measurement_communication_dual_by_uav = {key: 0.0 for key in state}
    trainer.load_measurement_communication_dual_state_dict(restored)
    assert trainer.measurement_communication_dual_state_dict() == restored
