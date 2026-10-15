"""Pure tests for canonical actual-stable logging; no environment is created."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "experiments" / "single_uav_goal_recovery_stable_contract.py"
SPEC = importlib.util.spec_from_file_location("stable_contract", HELPER)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def _info(*, distance=1.0, speed=0.5, distance_ok=True, speed_ok=True,
          candidate=True, counter=1, hold=5, stable=False):
    return {
        "single_uav_goal_in_radius": distance_ok,
        "single_uav_goal_speed_ok": speed_ok,
        "single_uav_goal_condition": candidate,
        "single_uav_stable_steps": counter,
        "single_uav_goal_stable": stable,
        "learning_reward_components": {
            "current_goal_distance": distance,
            "single_uav_speed_mps": speed,
            "single_uav_goal_threshold_m": 1.0,
            "single_uav_goal_speed_threshold_mps": 0.5,
            "single_uav_stable_hold_steps": hold,
        },
    }


def test_extracts_exact_actual_speed_and_threshold_boundary_without_mutation():
    info = _info(distance=1.0, speed=0.5)
    before = copy.deepcopy(info)
    row = contract.extract_canonical_stable_fields(info)
    assert info == before
    assert row["post_step_actual_speed_mps"] == 0.5
    assert row["goal_speed_for_stable_mps"] == 0.5
    assert row["goal_distance_for_stable_m"] == 1.0
    assert row["goal_distance_ok"] == row["goal_speed_ok"] == 1
    assert row["goal_speed_source"] == "simulator_actual_state"


@pytest.mark.parametrize("counter,stable", [(0, False), (1, False), (4, False), (5, True)])
def test_counter_and_terminal_stable_step_are_copied(counter, stable):
    candidate = counter > 0
    row = contract.extract_canonical_stable_fields(
        _info(candidate=candidate, distance_ok=candidate, speed_ok=candidate,
              counter=counter, stable=stable)
    )
    assert row["stable_hold_counter"] == counter
    assert bool(row["stable_goal_now"]) is stable


def test_counter_reset_and_inconsistent_flags_fail_closed():
    reset = contract.extract_canonical_stable_fields(
        _info(distance_ok=False, speed_ok=True, candidate=False, counter=0)
    )
    assert reset["stable_hold_counter"] == 0
    with pytest.raises(ValueError):
        contract.extract_canonical_stable_fields(
            _info(distance_ok=False, speed_ok=True, candidate=False, counter=2)
        )
    with pytest.raises(ValueError):
        contract.extract_canonical_stable_fields(_info(candidate=False, counter=0))


def _row(step, *, distance_ok, speed_ok, counter, stable=False, speed=1.0):
    return {
        "step": step, "goal_distance_ok": int(distance_ok),
        "goal_speed_ok": int(speed_ok),
        "stable_candidate_step": int(distance_ok and speed_ok),
        "stable_hold_counter": counter, "stable_goal_now": int(stable),
        "post_step_actual_speed_mps": speed,
        "post_step_actual_radial_velocity_mps": 0.4,
        "post_step_actual_tangential_velocity_mps": 0.3,
        "goal_recovery_correction_active": 1,
        "goal_recovery_correction_norm_mps": 0.6,
    }


def test_episode_aggregation_uses_logged_canonical_rows():
    rows = [
        _row(7, distance_ok=False, speed_ok=False, counter=0, speed=1.2),
        _row(8, distance_ok=True, speed_ok=False, counter=0, speed=0.8),
        _row(9, distance_ok=True, speed_ok=True, counter=1, speed=0.5),
        _row(10, distance_ok=True, speed_ok=True, counter=5, stable=True, speed=0.4),
    ]
    result = contract.aggregate_episode_stable_fields(rows)
    assert result["first_entered_1m_step"] == 8
    assert result["first_goal_speed_ok_step"] == 9
    assert result["first_stable_candidate_step"] == 9
    assert result["max_stable_hold_counter"] == 5
    assert result["stable_goal_step"] == 10
    assert result["actual_speed_at_first_entered_1m_mps"] == 0.8
    assert result["fraction_inside_1m_speed_ok"] == pytest.approx(2 / 3)


def _reference():
    episode = {
        "episode_index": 1, "episode_id": "e1", "scenario_seed": 11,
        "noise_seed": 12, "scenario_hash": "abc",
    }
    return {
        "checkpoint": {
            "resolved_path": str((ROOT / "actor.pt").resolve()),
            "sha256": "sha", "policy_hash": "policy",
        },
        "scenario_manifest_hash": "manifest",
        "results": {"on": {"episodes": [episode]}},
    }, episode


def test_identity_accepts_exact_and_rejects_checkpoint_mismatch():
    reference, episode = _reference()
    checks = contract.assert_previous_formal_identity(
        reference, checkpoint_path=ROOT / "actor.pt", checkpoint_sha256="sha",
        policy_hash="policy", scenario_manifest_hash="manifest",
        current_episodes=[episode],
    )
    assert all(checks.values())
    with pytest.raises(ValueError, match="checkpoint"):
        contract.assert_previous_formal_identity(
            reference, checkpoint_path=ROOT / "other.pt", checkpoint_sha256="sha",
            policy_hash="policy", scenario_manifest_hash="manifest",
            current_episodes=[episode],
        )


@pytest.mark.parametrize("field,value", [
    ("scenario_manifest_hash", "different"),
    ("policy_hash", "different"),
])
def test_identity_rejects_manifest_and_policy_mismatch(field, value):
    reference, episode = _reference()
    kwargs = dict(
        checkpoint_path=ROOT / "actor.pt", checkpoint_sha256="sha",
        policy_hash="policy", scenario_manifest_hash="manifest",
        current_episodes=[episode],
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        contract.assert_previous_formal_identity(reference, **kwargs)


def test_identity_rejects_reordered_or_changed_episode():
    reference, episode = _reference()
    changed = dict(episode, noise_seed=99)
    with pytest.raises(ValueError, match="episode"):
        contract.assert_previous_formal_identity(
            reference, checkpoint_path=ROOT / "actor.pt", checkpoint_sha256="sha",
            policy_hash="policy", scenario_manifest_hash="manifest",
            current_episodes=[changed],
        )


def test_run_summary_counter_distribution():
    rows = []
    for counter in (0, 1, 2, 3, 4, 5, 7):
        rows.append({
            "entered_1m": 1, "stable_goal": int(counter >= 5),
            "actual_speed_at_first_entered_1m_mps": 0.6,
            "mean_actual_speed_inside_1m_mps": 0.5,
            "fraction_inside_1m_speed_ok": 0.5,
            "mean_tangential_velocity_inside_1m_mps": 0.2,
            "correction_active_fraction_inside_1m": 1.0,
            "mean_correction_norm_inside_1m_mps": 0.4,
            "max_stable_hold_counter": counter,
            "stable_speed_threshold_mps": 0.5,
        })
    summary = contract.aggregate_actual_stable_validation(rows)
    assert summary["max_stable_hold_counter_distribution"] == {
        "counter_0": 1, "counter_1": 1, "counter_2": 1,
        "counter_3": 1, "counter_4": 1, "counter_5_plus": 2,
    }


def test_evaluation_source_forces_on_formal_and_only_logs_after_step():
    source = (ROOT / "src" / "experiments" /
              "single_uav_goal_recovery_correction_evaluation.py").read_text(encoding="utf-8")
    assert 'condition, formal = "on", True' in source
    assert "--actual-stable-validation" in source
    assert source.index("stable_fields = extract_canonical_stable_fields(next_info)") > source.index("environment.step(")
    assert "teacher.action" not in source
    assert "trainer.update(" not in source
    assert "optimizer.step(" not in source

