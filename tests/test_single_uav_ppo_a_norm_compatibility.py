"""Pure/reset-only checks for the normalization-transfer compatibility APIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.experiments.single_uav_ppo_a_foundation import load_run
from src.experiments.single_uav_ppo_a_norm_compatibility import (
    _runtime_graph_probe,
    _node_arrays,
    initialize_condition,
    license_output_head,
    tensor_pair_audit,
)
from src.experiments.single_uav_ppo_a_scenarios import build_a_foundation_scenarios
from src.experiments.single_uav_ppo_a_environment import create_foundation_environment
from src.experiments.single_uav_ppo_baseline import create_single_uav_ppo_trainer
from src.learning.normalization import FeatureNormalizer, ObservationNormalizer


def _source_fixture():
    spec, base, stats = load_run()
    return spec, base, stats


def test_singleton_first_layer_runtime_preactivation_matches_elu(tmp_path: Path) -> None:
    spec, base, stats = _source_fixture()
    env = create_foundation_environment(base)
    action_log = tmp_path / "construction.csv"
    trainer = create_single_uav_ppo_trainer(
        env,
        base,
        normalizer=stats[0],
        central_normalizer=stats[1],
        normalizer_provenance=stats[3],
        action_log_path=action_log,
    )
    initialize_condition(trainer, spec["A_source"]["checkpoint"], "N")
    normalized = np.linspace(-0.5, 0.5, 42, dtype=np.float32)
    _pre, _expected, error = _runtime_graph_probe(trainer, normalized)
    assert error <= 1e-5


def test_clip_noninvertibility_is_explicit() -> None:
    old = ObservationNormalizer()
    old_node = FeatureNormalizer(clip=1.0).fit(np.array([[0.0] * 42, [1.0] * 42]))
    old.normalizers = {"node_features": old_node}
    mean, scale, clip = _node_arrays(old)
    assert clip == 1.0
    x = mean + scale * 3.0
    clipped_a = np.clip((x - mean) / scale, -clip, clip)
    clipped_b = np.clip((x + scale - mean) / scale, -clip, clip)
    assert np.array_equal(clipped_a, clipped_b)


def test_r_condition_preserves_source_except_two_head_tensors(tmp_path: Path) -> None:
    spec, base, stats = _source_fixture()
    env = create_foundation_environment(base)
    action_log = tmp_path / "construction.csv"
    trainer = create_single_uav_ppo_trainer(
        env,
        base,
        normalizer=stats[0],
        central_normalizer=stats[1],
        normalizer_provenance=stats[3],
        action_log_path=action_log,
    )
    report = initialize_condition(trainer, spec["A_source"]["checkpoint"], "R")
    assert report["actual_source_preserved_tensors"] == 21
    assert set(report["resetparameter_names"]) == {"delta_mean_head.weight", "delta_mean_head.bias"}
    assert report["rollout_size"] == 0
    assert report["target_optimizer_state_entries"] == 0


def test_license_uses_unreset_actor_and_zero_environment_steps(tmp_path: Path) -> None:
    spec, base, stats = _source_fixture()
    result = license_output_head(
        base,
        {"train": build_a_foundation_scenarios()["normal_train"]},
        spec["A_source"]["checkpoint"],
        stats,
        tmp_path / "data",
        tmp_path / "logs",
    )
    assert result["env_steps"] == 0
    assert result["ppo_updates"] == 0
    assert result["primary"] in {"N", "R"}
    assert all(float(row["bounded_mean_norm_mps"]) > 0.05 for row in result["probe_rows"]) == result["licensed"]


def test_tensor_pair_audit_scopes_condition_differences() -> None:
    common = {
        "critic.0.weight": torch.ones(2),
        "critic.0.bias": torch.ones(2),
        "actor_body.0.weight": torch.ones(2),
        "delta_mean_head.weight": torch.ones(2),
        "delta_mean_head.bias": torch.ones(2),
        "gat.layers.0.attention": torch.ones(2),
    }
    states = {condition: {name: value.clone() for name, value in common.items()} for condition in ("C", "N", "G", "R")}
    states["G"]["gat.layers.0.attention"].zero_()
    states["R"]["delta_mean_head.weight"].zero_()
    states["R"]["delta_mean_head.bias"].zero_()
    report = tensor_pair_audit(states)
    assert report["passed"]
    assert report["actual_changed_names"]["C_N"] == []
    assert report["actual_changed_names"]["N_R"] == ["delta_mean_head.bias", "delta_mean_head.weight"]
