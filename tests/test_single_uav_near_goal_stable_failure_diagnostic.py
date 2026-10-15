"""Fast tests for the file-only near-goal stable-failure diagnostic."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "experiments" / "single_uav_near_goal_stable_failure_diagnostic.py"
)
_SPEC = importlib.util.spec_from_file_location("near_goal_stable_failure_diagnostic", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)


def _episode(**updates):
    row = {
        "condition": "on", "episode_id": "episode_1", "entered_1m": "1",
        "stable_goal": "0", "first_entered_1m_step": "1",
        "goal_threshold_m": "1.0", "stable_hold_steps": "5",
    }
    row.update({key: str(value) for key, value in updates.items()})
    return row


def _step(step, distance_after, radial=0.2, tangential=0.1, **updates):
    row = {
        "condition": "on", "episode_id": "episode_1", "step": str(step),
        "goal_distance_m": str(distance_after + 0.1),
        "goal_distance_after_m": str(distance_after),
        "current_radial_velocity_mps": str(radial),
        "current_tangential_velocity_mps": str(tangential),
        "desired_radial_velocity_mps": "0.0",
        "actor_raw_delta_v_mps": "[0.1,0.0,0.0]",
        "actor_raw_delta_v_norm_mps": "0.1",
        "goal_recovery_correction_active": "0",
        "goal_recovery_correction_alpha": "0.0",
        "goal_recovery_correction_delta_v_mps": "[0.0,0.0,0.0]",
        "goal_recovery_correction_norm_mps": "0.0",
        "goal_recovery_radial_delta_v_mps": "[0.0,0.0,0.0]",
        "goal_recovery_tangential_delta_v_mps": "[0.0,0.0,0.0]",
        "corrected_delta_v_pre_transform_mps": "[0.1,0.0,0.0]",
        "corrected_delta_v_pre_safety_mps": "[0.1,0.0,0.0]",
        "applied_delta_v_mps": "[0.1,0.0,0.0]",
        "safety_projection_applied": "0", "boundary_projection_applied": "0",
        "passed_obstacle": "1", "post_bypass_recovery": "1", "stable_goal": "0",
    }
    row.update({key: str(value) for key, value in updates.items()})
    return row


def test_selection_excludes_not_entered_and_stable():
    rows = [_episode(), _episode(episode_id="not_entered", entered_1m=0),
            _episode(episode_id="stable", stable_goal=1)]
    assert [row["episode_id"] for row in diagnostic.select_failure_episodes(rows)] == ["episode_1"]


def test_alignment_uses_next_contiguous_row_and_high_entry_speed():
    rows = [_step(0, 1.2), _step(1, 0.8), _step(2, 0.7, radial=0.6, tangential=0.8), _step(3, 0.6)]
    result, window = diagnostic.diagnose_episode(_episode(), rows, 1, 2)
    assert result["entry_total_speed_mps"] == pytest.approx(1.0)
    assert result["high_entry_speed"] is True
    assert next(row for row in window if row["step"] == "1")["post_state_proxy_source_step"] == 2
    assert window[-1]["post_total_speed_proxy_mps"] is None


def test_overshoot_conflict_and_insufficient_braking_are_traceable():
    active = {
        "goal_recovery_correction_active": 1,
        "goal_recovery_correction_delta_v_mps": "[0.2,0,0]",
        "goal_recovery_correction_norm_mps": 0.2,
        "goal_recovery_radial_delta_v_mps": "[0.2,0,0]",
        "actor_raw_delta_v_mps": "[-0.2,0,0]",
        "desired_radial_velocity_mps": 1.0,
    }
    rows = [_step(0, 1.2), _step(1, 0.8, radial=0.4, **active),
            _step(2, 0.2, radial=0.4, **active), _step(3, 1.2, radial=0.4, **active),
            _step(4, 1.4, radial=0.4)]
    result, _ = diagnostic.diagnose_episode(_episode(), rows, 1, 3)
    assert result["correction_overshoot"] is True
    assert result["actor_correction_conflict"] is True
    assert result["primary_failure_mode"] == "CORRECTION_OVERSHOOT"
    assert result["mode_diagnostics"]["CORRECTION_OVERSHOOT"]["confidence"] == "association"


def test_tangential_residual_and_hold_interrupted():
    rows = [_step(0, 1.2), _step(1, 0.9), _step(2, 0.8, radial=0.1, tangential=0.8),
            _step(3, 0.7, radial=0.1, tangential=0.8),
            _step(4, 0.6, radial=0.1, tangential=0.1),
            _step(5, 0.6, radial=0.8, tangential=0.1)]
    result, _ = diagnostic.diagnose_episode(_episode(), rows, 1, 4)
    assert result["tangential_residual"] is True
    assert result["stable_hold_interrupted"] is True
    assert result["max_consecutive_stable_candidate_steps"] < 5


def test_oscillation_counts_reentries_and_exits():
    rows = [_step(0, 1.2), _step(1, 0.8, goal_distance_m=1.2),
            _step(2, 1.1), _step(3, 0.8), _step(4, 0.7)]
    result, _ = diagnostic.diagnose_episode(_episode(), rows, 1, 3)
    assert result["number_of_1m_entries"] == 2
    assert result["number_of_1m_exits"] == 1
    assert result["goal_region_oscillation"] is True


def test_braking_rule_detects_weak_deceleration():
    rows = [_step(0, 1.2, radial=1.0), _step(1, 0.8, radial=1.0),
            _step(2, 0.7, radial=0.99), _step(3, 0.6, radial=0.98)]
    result, _ = diagnostic.diagnose_episode(_episode(), rows, 1, 2)
    assert result["insufficient_braking"] is True
    assert result["braking_required_steps"] >= 1


def _write_csv(path: Path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerow(row)


def test_existing_output_root_is_rejected_before_reading(tmp_path):
    existing = tmp_path / "existing"; existing.mkdir()
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(evaluation_root=tmp_path / "missing", logs_root=existing,
                                  plots_root=tmp_path / "plots", plots=False)


def test_classifier_metadata_and_flags_are_diagnostic_only():
    result, _ = diagnostic.diagnose_episode(
        _episode(), [_step(0, 1.2), _step(1, 0.8), _step(2, 0.7)], 1, 1
    )
    assert result["stable_criterion"]["exact_recomputation_possible"] is False
    assert result["training_performed"] is False
    assert result["rollout_executed"] is False
    assert result["control_algorithm_modified"] is False
    assert set(result["mode_diagnostics"]) == set(diagnostic.MODE_ORDER)


def test_source_has_no_rollout_training_or_control_imports():
    path = Path(diagnostic.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith(("src.simulation", "src.control", "src.learning")) for name in imports)
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not called & {"run_goal_recovery_correction_evaluation", "_rollout_frozen_actor_episode", "Teacher"}


def test_rules_and_cli_defaults_are_stable():
    args = diagnostic.build_parser().parse_args([])
    assert args.pre_steps == 15 and args.post_steps == 20
    assert diagnostic.RULES["version"] == "near_goal_failure_rules_v1"
    assert diagnostic.PRIMARY_PRIORITY[0] == "CORRECTION_OVERSHOOT"
