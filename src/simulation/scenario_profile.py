"""Validated frozen environment profiles for reproducible experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


PROFILE_PARAMETER_NAMES = (
    "max_comm_range_m",
    "total_bandwidth_hz",
    "nlos_extra_loss_db",
    "message_arrivals_per_window",
)

DEFAULT_SCREENING_GEOMETRY: dict[str, Any] = {
    "type": "moving_3d_los_screen",
    "construction": "pair_00_01_midpoint_then_perpendicular_translation",
    "packet_loss_modified_directly": False,
}


def load_scenario_profile(
    path: str | Path,
    *,
    expected_uav_count: int = 3,
) -> dict[str, Any]:
    """Load and validate one frozen environment profile.

    The first screening artifact predates the explicit ``uav_count`` and
    ``screening_geometry`` fields.  Missing values are filled with the
    documented three-UAV screening defaults for backward compatibility; no
    communication parameter is filled implicitly.
    """

    profile_path = Path(path)
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read scenario profile {profile_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"scenario profile must be a JSON object: {profile_path}")
    if payload.get("allowed_for_training") is not True:
        raise ValueError(
            f"scenario profile is not allowed for training: {profile_path}"
        )
    profile_uav_count = payload.get("uav_count", 3)
    try:
        profile_uav_count = int(profile_uav_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid uav_count in scenario profile: {profile_path}") from exc
    if profile_uav_count != int(expected_uav_count):
        raise ValueError(
            f"scenario profile uav_count={profile_uav_count} does not match "
            f"the requested experiment uav_count={expected_uav_count}"
        )
    if payload.get("mappo_performance_used_for_selection") is not False:
        raise ValueError(
            "scenario profile selection must not depend on MAPPO performance"
        )
    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise ValueError(f"scenario profile has no parameters object: {profile_path}")
    parameters: dict[str, float] = {}
    for name in PROFILE_PARAMETER_NAMES:
        if name not in raw_parameters:
            raise ValueError(f"missing {name} in scenario profile: {profile_path}")
        try:
            value = float(raw_parameters[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name} in scenario profile: {profile_path}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive: {profile_path}")
        parameters[name] = value

    scenario_seeds = payload.get("scenario_seeds", [])
    if not isinstance(scenario_seeds, list):
        raise ValueError(f"scenario_seeds must be a list: {profile_path}")
    geometry = payload.get("screening_geometry", DEFAULT_SCREENING_GEOMETRY)
    if not isinstance(geometry, Mapping):
        raise ValueError(f"screening_geometry must be an object: {profile_path}")
    geometry_payload = dict(DEFAULT_SCREENING_GEOMETRY)
    geometry_payload.update({str(key): value for key, value in geometry.items()})
    return {
        "profile": str(payload.get("profile", profile_path.stem)),
        "uav_count": profile_uav_count,
        "profile_path": str(profile_path.resolve()),
        "selection_status": str(payload.get("selection_status", "unknown")),
        "allowed_for_training": True,
        "design_point_id": str(payload.get("design_point_id", "unknown")),
        "severity_score": payload.get("severity_score"),
        "parameters": parameters,
        "screening_design_seed": payload.get("screening_design_seed"),
        "scenario_seeds": [int(seed) for seed in scenario_seeds],
        "selection_rule": str(payload.get("selection_rule", "unknown")),
        "mappo_performance_used_for_selection": False,
        "screening_geometry": geometry_payload,
    }


__all__ = [
    "DEFAULT_SCREENING_GEOMETRY",
    "PROFILE_PARAMETER_NAMES",
    "load_scenario_profile",
]
