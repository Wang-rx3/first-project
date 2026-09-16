"""Contract tests for the isolated Stage 2B-R2 learning-bottleneck audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.single_uav_obstacle_dagger_stage2br import mirror_static_scenario
from src.experiments.single_uav_obstacle_stage2br2_learning_bottleneck import (
    ALL_ABLATION_SCOPES,
    FIRST_LAYER_DIRECTION_NAMES,
    audit_mirror_targets,
    configure_ablation_scope,
    _scope_parameter_selected,
)


def _synthetic_mirror_samples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, left_category, right_category, x, lateral in (
        ("side", "left", "right", 42.0, 4.0),
        ("front", "left_front", "right_front", 46.0, 4.5),
    ):
        for category, side in ((left_category, "left"), (right_category, "right")):
            sign = -1.0 if side == "left" else 1.0
            body_dy = lateral if side == "left" else -lateral
            rows.append(
                {
                    "pair_id": f"pair_{group}",
                    "mirror_group": group,
                    "category": category,
                    "side": side,
                    "obstacle_position_world_m": [x, 100.0 + body_dy, 20.0],
                    "obstacle_relative_dx_body_m": x - 34.0,
                    "obstacle_relative_dy_body_m": body_dy,
                    "obstacle_relative_dz_body_m": 0.0,
                    "teacher_action_delta_v_mps": [0.2, sign * 0.4, 0.0],
                    "expected_lateral_sign": sign,
                }
            )
    return rows


def test_mirror_geometry_changes_only_lateral_sign():
    left = mirror_static_scenario(
        None, category="left", pair_index=1, side="left", seed=10, round_index=0
    )
    right = mirror_static_scenario(
        None, category="right", pair_index=1, side="right", seed=10, round_index=0
    )
    left_spec = left["obstacle_specs"][0]
    right_spec = right["obstacle_specs"][0]
    assert left_spec["center_m"][0] == right_spec["center_m"][0]
    assert left_spec["center_m"][1] + right_spec["center_m"][1] == pytest.approx(200.0)
    assert left_spec["radius_m"] == right_spec["radius_m"]
    assert left["initial_positions"] == right["initial_positions"]
    assert left["goals"] == right["goals"]
    assert left["initial_velocities"] == right["initial_velocities"]


def test_mirror_target_audit_accepts_correct_oracle_and_bc_signs():
    report = audit_mirror_targets(_synthetic_mirror_samples())
    assert report["passed"] is True
    assert all(row["left_target_sign_correct"] == 1 for row in report["pair_rows"])
    assert all(row["right_target_sign_correct"] == 1 for row in report["pair_rows"])


def test_mirror_target_audit_rejects_wrong_right_target():
    samples = _synthetic_mirror_samples()
    samples[1]["teacher_action_delta_v_mps"] = [0.2, -0.4, 0.0]
    report = audit_mirror_targets(samples)
    assert report["passed"] is False
    assert report["invalid_reasons"]


def test_ablation_scope_contract_excludes_critic_and_logstd():
    names = [
        "gat.layers.0.node_projection.weight",
        "gat.layers.0.residual.weight",
        "gat.layers.0.bias",
        "gat.layers.1.bias",
        "actor_body.0.weight",
        "delta_mean_head.weight",
        "measurement_head.weight",
        "critic.0.weight",
        "delta_logstd",
    ]
    assert {
        name for name in names if _scope_parameter_selected(name, "39_42_direction_columns")
    } == set(FIRST_LAYER_DIRECTION_NAMES)
    assert all(
        not _scope_parameter_selected(name, "first_layer")
        for name in ("actor_body.0.weight", "critic.0.weight", "delta_logstd")
    )
    assert _scope_parameter_selected("actor_body.0.weight", "actor_head")
    assert _scope_parameter_selected("delta_mean_head.weight", "actor_head")
    assert not _scope_parameter_selected("critic.0.weight", "actor_head")
    assert not _scope_parameter_selected("delta_logstd", "full_actor")
    assert not _scope_parameter_selected("critic.0.weight", "full_actor")
    assert _scope_parameter_selected("measurement_head.weight", "full_actor")


def test_all_declared_scopes_are_supported():
    for scope in ALL_ABLATION_SCOPES:
        assert isinstance(scope, str) and scope


def test_scope_configuration_freezes_critic_and_logstd():
    pytest.importorskip("torch")
    from src.learning.mappo import MAPPOConfig, MAPPOPolicy

    policy = MAPPOPolicy(
        MAPPOConfig(uav_count=1, node_feature_dim=42, device="cpu"),
        uav_ids=("uav_00",),
    )
    for scope in ALL_ABLATION_SCOPES:
        trainable = configure_ablation_scope(policy, scope)
        assert trainable
        assert policy.delta_logstd.requires_grad is False
        assert all(
            not parameter.requires_grad
            for name, parameter in policy.named_parameters()
            if name.startswith("critic.")
        )


def test_r2_output_paths_are_independent_from_previous_stage():
    from src.experiments.single_uav_obstacle_stage2br2_learning_bottleneck import (
        DEFAULT_DATA_ROOT,
        DEFAULT_LOG_ROOT,
        DEFAULT_PLOT_ROOT,
    )

    paths = {Path(DEFAULT_DATA_ROOT), Path(DEFAULT_LOG_ROOT), Path(DEFAULT_PLOT_ROOT)}
    assert all("stage2br2" in str(path).lower() for path in paths)
    assert not any("stage2br_smoke" in str(path).lower() for path in paths)
