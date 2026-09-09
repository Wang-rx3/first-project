"""Evidence/aggregation checks; synthetic fixtures are test-only."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments import final_mainline_audit as base
from src.experiments.mainline_figures_v2 import (
    PARTS, SourceGuard, bootstrap_indices, curve_stats, quality_episode,
    reward_windows, training_evidence, write_table, zoom_limits,
)


def scenario(episode="episode_000001"):
    return {"episode_id": episode, "split": "train", "uav_count": 3,
            "reward_config": {"weights": {"uncertainty": .35, "loss": .2, "aoi": .15,
                                             "delay": .15, "connectivity": .15},
                              "lambda_coll": 10., "lambda_energy": .01}}


def rewards(episode="episode_000001", collision=0, connectivity=1.):
    joint = .35 * .5 - .15 * connectivity
    return [{"episode_id": episode, "split": "train", "uav_id": f"uav_0{i}", "timestamp": .2,
             "reward_total": -joint - 10 * collision - .002, "joint_objective": joint,
             "normalized_uncertainty": .5, "normalized_packet_loss": 0., "normalized_aoi": 0.,
             "normalized_delay": 0., "normalized_connectivity": connectivity,
             "reward_collision": -10 * collision, "reward_energy": -.002, "collision": collision}
            for i in range(3)]


def test_signed_communication_and_sparse_collision_close():
    normal = reward_windows(rewards(), scenario())[0]
    assert normal["reward_com"] == pytest.approx(.15)
    assert normal["reward_energy"] == -.002
    assert sum(normal[k] for k in PARTS) == pytest.approx(normal["mean_reward"])
    collision = reward_windows(rewards(collision=1, connectivity=0), scenario())[0]
    assert collision["reward_collision"] == -10
    assert abs(collision["closure_error"]) < 1e-6


def test_reward_missing_or_duplicate_uav_and_bad_formula_fail():
    with pytest.raises(ValueError, match="UAV"):
        reward_windows(rewards()[:2], scenario())
    rows = rewards()
    rows[0]["reward_total"] += 1
    with pytest.raises(ValueError, match="closure"):
        reward_windows(rows, scenario())
    rows = rewards()
    del rows[0]["reward_energy"]
    with pytest.raises(KeyError):
        reward_windows(rows, scenario())


def test_training_binds_episode_id_not_directory_order(tmp_path):
    config = base.FinalMainlineAuditConfig(reference_root=tmp_path, seeds=(2026,),
        branches=(base.BASELINE_BRANCH, base.PROPOSED_BRANCH), expected_training_updates=2, progress=False)
    for branch in config.branches:
        scale = tmp_path / "seed_2026" / branch / "scale_3"
        diagnostics = scale / "diagnostics"
        diagnostics.mkdir(parents=True)
        logs = []
        for update, eid in ((2, "episode_000001"), (1, "episode_000002")):
            ep = scale / "train" / eid
            ep.mkdir(parents=True)
            data = rewards(eid, collision=int(update == 2))
            write_table(ep / "rewards.csv", data)
            (ep / "scenario.json").write_text(json.dumps(scenario(eid)))
            logs.append({"phase": "train", "episode_id": eid, "update": update,
                         **{k: .1 for k in base.TRAINING_METRICS}, "mean_reward": data[0]["reward_total"]})
        (diagnostics / "training_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in logs))
    rows, _, _ = training_evidence(config)
    assert rows[0]["episode_id"] == "episode_000002"
    assert rows[0]["reward_collision"] == 0
    assert rows[1]["reward_collision"] == -10
    path = diagnostics / "training_metrics.jsonl"
    path.write_text(json.dumps(logs[0]))
    with pytest.raises(ValueError, match="missing/duplicate"):
        training_evidence(config)


def test_complete_seed_curves_and_equal_weights():
    config = base.FinalMainlineAuditConfig(reference_root=Path("."), seeds=(1, 2, 3, 4, 5),
        expected_training_updates=2, bootstrap_samples=100)
    rows = [{"base_seed": s, "branch": base.BASELINE_BRANCH, "update": u, "mean_reward": s * u}
            for s in config.seeds for u in (1, 2)]
    indices = bootstrap_indices(config)
    assert np.array_equal(indices, bootstrap_indices(config))
    mean, lo, hi, _ = curve_stats(rows, base.BASELINE_BRANCH, "mean_reward", config, indices)
    np.testing.assert_allclose(mean, [3, 6])
    assert lo[1] == pytest.approx(lo[0] * 2)
    assert hi[1] == pytest.approx(hi[0] * 2)
    with pytest.raises(KeyError):
        curve_stats(rows[:-1], base.BASELINE_BRANCH, "mean_reward", config, indices)


def test_no_update_is_na_for_fraction_and_zero_for_per_window():
    event = {"accepted_update_count": 0, "beneficial_update_count": 0,
             "neutral_update_count": 0, "harmful_update_count": 0}
    result = quality_episode({"accepted_update_count": 0, "measurement_overhead_packets": 2}, event, 50)
    assert np.isnan(result["harmful_fraction"])
    assert result["harmful_per_window"] == 0
    assert result["neutral_per_window"] == 0
    assert result["measurement_load_bytes"] == 1024
    assert np.isnan(result["h5_benefit_retained_fraction"])


def test_source_guard_blocks_test_reads_and_source_writes(tmp_path):
    src, out = tmp_path / "source", tmp_path / "output"
    src.mkdir()
    out.mkdir()
    normal = src / "checkpoint.pt"
    normal.write_bytes(b"frozen checkpoint")
    sealed = src / "test_cooperative_updates.jsonl"
    sealed.write_text("secret")
    mixed = src / "diagnostics" / "cooperative_updates.jsonl"
    mixed.parent.mkdir()
    mixed.write_text("mixed test content")
    with SourceGuard((src,), (out,)) as guard:
        normal.read_bytes()
        with pytest.raises(PermissionError, match="sealed test"):
            sealed.read_text()
        with pytest.raises(PermissionError, match="source write"):
            normal.write_bytes(b"changed")
        with pytest.raises(PermissionError, match="mixed-split"):
            mixed.read_text()
        (out / "derived.csv").write_text("allowed")
        result = guard.verify()
        assert len(result) == 1 and result[0]["unchanged"]
    assert normal.read_bytes() == b"frozen checkpoint"


def test_nested_audits_prefer_validation_files_without_opening_mixed(tmp_path):
    from src.experiments import metric_consistency_audit as metric
    from src.experiments import ekf_update_persistence_audit as persistence
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    write_table(diag / "validation_episode_metrics.csv", [{"episode_id": "episode_400001", "split": "validation", "mean_reward": -.5}])
    (diag / "validation_cooperative_updates.jsonl").write_text(json.dumps({"episode_id": "episode_400001", "split": "validation", "accepted": True}))
    (diag / "cooperative_updates.jsonl").write_text("invalid mixed log must never be read")
    (diag / "episode_metrics.csv").write_text("invalid mixed metrics must never be read")
    with SourceGuard((tmp_path,), ()):
        assert metric._accepted_updates(tmp_path, "validation") == {"episode_400001": 1}
        assert "episode_400001" in metric._metric_rows(tmp_path, "validation")
        assert "episode_400001" in persistence._episode_metric_map(tmp_path, 2026, base.BASELINE_BRANCH, "validation")


def test_zoom_is_pooled_across_branches_and_preserves_zero():
    assert zoom_limits([np.zeros(10)]) is None
    groups = [np.linspace(-.004, -.001, 100), np.linspace(-.008, -.002, 100)]
    assert zoom_limits(groups) == zoom_limits(groups[::-1])
    lo, hi = zoom_limits(groups)
    assert lo < -.004 and hi > 0


def test_cli_profile_is_opt_in():
    from src.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["run-final-mainline-audit", "--reference", "example"])
    assert args.figure_profile == "legacy"
    args = parser.parse_args(["run-final-mainline-audit", "--reference", "example", "--figure-profile", "mainline-v2"])
    assert args.figure_profile == "mainline-v2"
