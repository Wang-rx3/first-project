"""Pure contract tests for the v12 read-only neighbour diagnostic."""
from __future__ import annotations

import copy

import numpy as np

from src.analysis.neighbor_selection_diagnostic_v12 import (
    correlation,
    derive_directional_features,
    label_point,
    recover_prior_frame_delay,
    summarize_points,
)


def _branch(values: list[float], *, accepted: bool = True) -> dict[str, object]:
    return {
        "sse_m2": values,
        "accepted": accepted,
        "attempted": accepted,
        "request_cost": 1 if accepted else 0,
        "event": {"P_initial": np.eye(3).tolist(), "P_post": (0.5 * np.eye(3)).tolist()},
    }


def _point(a: float = 1.0, b: float = 0.5) -> dict[str, object]:
    return {
        "study_phase": "develop",
        "scene_id": "g",
        "seed": 1,
        "step": 6,
        "dual_eligible": True,
        "score_difference": 0.2,
        "branches": {
            "none": _branch([4.0, 4.0, 4.0, 4.0], accepted=False),
            "uav_00": _branch([a, a, a, a]),
            "uav_01": _branch([b, b, b, b]),
        },
    }


def test_original_correlation_handles_constants_and_tied_ranks() -> None:
    assert correlation([1, 1, 1], [1, 2, 3]) is None
    value = correlation([1, 1, 2, 3], [1, 2, 2, 3], True)
    assert value is not None and 0.0 < value <= 1.0


def test_labels_use_fixed_cost_and_fractional_ties() -> None:
    no_effect = label_point(_point(a=3.0, b=3.0))
    assert no_effect["sensitivity_no_effect"] is True
    point = _point(a=3.0, b=3.5)
    point["score_difference"] = 0.0
    row = label_point(point)
    assert row["offline_R_current_uav_00"] == pytest.approx(0.99)
    assert row["offline_R_fourstate_uav_01"] == pytest.approx(0.49)
    assert row["true_peer_difference_fourstate"] != 0.0
    assert row["ranking_correct"] == 0.5


def test_non_deployable_labels_do_not_create_regret_claim() -> None:
    point = _point()
    point["branches"] = {"none": _branch([1.0]), "uav_00": {}, "uav_01": {}}
    row = label_point(point)
    assert row["label_deployable"] is False
    assert row["regret_all3actions"]["offline_reward_fourstate"]["label_ceiling"] is None


def test_observable_choices_require_positive_candidate_scores_and_keep_ties() -> None:
    point = _point()
    point["pre_context"] = {"measurement_budget_remaining": 2}
    point["old_scoring"] = {
        "candidates": {
            "uav_00": {"score": -0.2, "valid": True},
            "uav_01": {"score": -0.1, "valid": True},
        }
    }
    point["minimal_scoring"] = {
        "candidates": {
            "uav_00": {"trace_gain_m2": 0.02, "valid": True},
            "uav_01": {"trace_gain_m2": 0.02, "valid": True},
        }
    }
    row = label_point(point)
    assert row["old_choice"] == "none"
    assert row["minimal_pred_choice"] in {"uav_00", "uav_01"}
    assert row["minimal_pred_choice"] == label_point(point)["minimal_pred_choice"]


def test_directional_features_preserve_anisotropic_direction_and_isotropic_alias() -> None:
    context = {
        "own_state": [0.0, 0.0, 0.0],
        "own_covariance": np.diag([1.0, 2.0, 3.0]).tolist(),
        "peers": {
            "uav_00": {"state": [1.0, 0.0, 0.0], "covariance": np.eye(3).tolist()},
            "uav_01": {"state": [0.0, 0.0, 1.0], "covariance": (2.0 * np.eye(3)).tolist()},
        },
    }
    result = derive_directional_features(context)
    assert result["directional_self_variance_uav_00_m2"] == 1.0
    assert result["directional_self_variance_uav_01_m2"] == 3.0
    assert result["world_unit_direction_diff"] == [1.0, 0.0, -1.0]
    isotropic = copy.deepcopy(context)
    isotropic["own_covariance"] = (2.0 * np.eye(3)).tolist()
    iso = derive_directional_features(isotropic)
    assert iso["directional_self_variance_uav_00_m2"] == iso["directional_self_variance_uav_01_m2"]


def test_delay_uses_only_previous_frame_and_rejects_mismatch() -> None:
    context = {
        "step": 2,
        "time_s": 1.0,
        "peers": {"uav_00": {"send_time_s": 0.8, "state_time_s": 1.0}},
    }
    record = {
        "frames": [
            {"audit": {"live_queue": {"arrived_messages_after": {"('uav_00', 'uav_02')": {
                "source": "uav_00", "target": "uav_02", "send_time_s": 0.8,
                "last_propagated_time_s": 1.0, "delay_s": 0.03,
            }}}}},
            {"audit": {"live_queue": {"arrived_messages_after": {"('uav_00', 'uav_02')": {
                "source": "uav_00", "target": "uav_02", "send_time_s": 0.8,
                "last_propagated_time_s": 2.0, "delay_s": 9.0,
            }}}}},
        ]
    }
    assert recover_prior_frame_delay(record, context, "uav_00") == 0.03
    bad = copy.deepcopy(context)
    bad["peers"]["uav_00"]["send_time_s"] = 0.7
    assert recover_prior_frame_delay(record, bad, "uav_00") is None


def test_prelink_delay_is_not_reported_as_packet_delay() -> None:
    context = {
        "step": 2,
        "time_s": 1.0,
        "peers": {"uav_00": {"prelink_delay_s": 0.25}},
    }
    assert recover_prior_frame_delay({"frames": []}, context, "uav_00") is None


def test_summary_groups_seed_replicates_and_keeps_no_effect_non_deployable() -> None:
    first = _point()
    second = _point(a=2.0, b=1.5)
    second["seed"] = 2
    summary = summarize_points([first, second])
    assert summary["n"] == 2
    assert summary["pooled"]["n"] == 2
    assert {item["seed"] for item in summary["groups"] if item["seed"] != "all"} == {1, 2}
    assert isinstance(summary["correlations"], list)


# pytest is imported at the bottom so the module remains importable in tiny
# environments that only exercise the diagnostic functions.
import pytest
