from __future__ import annotations

import math

from src.experiments.vo_interaction_audit import (
    _event_class,
    _jsonable,
    _next_sensor,
    _validation_step,
)


def _event(before: float, after: float, accepted: bool = True) -> dict[str, object]:
    return {
        "accepted": accepted,
        "true_position_error_before_m": before,
        "true_position_error_after_m": after,
    }


def test_vo_audit_classifies_accepted_update_quality() -> None:
    assert _event_class(_event(1.0, 0.9)) == "beneficial"
    assert _event_class(_event(1.0, 1.1)) == "harmful"
    assert _event_class(_event(1.0, 1.0005)) == "neutral"
    assert _event_class(_event(1.0, 0.9, accepted=False)) == "rejected"


def test_vo_audit_maps_post_step_timestamp_without_cross_episode_state() -> None:
    assert _validation_step(0.2) == 0
    assert _validation_step(1.0) == 4
    mapping = {
        ("uav_00", 1): {"vo_update_accepted": 1, "uwb_update_accepted": 0},
        ("uav_00", 2): {"vo_update_accepted": 0, "uwb_update_accepted": 1},
    }
    assert _next_sensor(mapping, "uav_00", 0) == "visual_odometry"
    # The audit receives a per-episode mapping; absent future rows do not
    # borrow a sensor event from another episode.
    assert _next_sensor({}, "uav_00", 0) == "none_observed"


def test_vo_audit_converts_nonfinite_values_to_na_for_json() -> None:
    assert _jsonable({"missing": float("nan"), "value": [float("inf")]}) == {
        "missing": "NA",
        "value": ["NA"],
    }
