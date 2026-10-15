"""Deterministic scene manifest for the fixed single-UAV navigation audit.

This module only constructs JSON-compatible reset dictionaries.  It does not
create an environment, consume a random generator, load a model, or advance
any simulation step.  The four returned phase lists are intentionally frozen
so that the fixed controller audit can be reproduced from the manifest alone.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


_UAV_ID = "uav_00"
_REGIMES = ("rest", "toward", "away", "tangent")
_DISTANCES = (4.0, 12.0, 20.0, 30.0)
_BOUNDS_MIN = (0.0, 0.0, 2.0)
_BOUNDS_MAX = (200.0, 200.0, 60.0)


def _json(value: Any) -> Any:
    """Return a JSON-compatible copy used by the canonical hash."""

    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    # The manifest is built from Python scalars, but accepting scalar-like
    # values here keeps the canonical helper compatible with the historical
    # geometry contract without importing NumPy.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _geometry_hash(scenario: Mapping[str, Any]) -> str:
    """Hash positions, goals, bounds, and box geometry, omitting IDs/metadata.

    This is the same payload contract as
    ``single_uav_ppo_a_scenarios._geometry_hash``.  In particular, the
    presence of the explicit ``obstacles=[]`` reset override is retained in
    the payload while obstacle IDs, type labels, and velocities are omitted.
    """

    obstacle_specs: list[dict[str, Any]] = []
    for spec in scenario.get("obstacle_specs", []):
        if not isinstance(spec, Mapping):
            continue
        obstacle_specs.append(
            {
                "geometry_type": str(spec["geometry_type"]),
                "center_m": [float(value) for value in spec["center_m"]],
                "size_m": [float(value) for value in spec["size_m"]],
                "physical_collision_enabled": bool(spec["physical_collision_enabled"]),
            }
        )
    payload: dict[str, Any] = {
        "initial_positions": {
            str(key): [float(value) for value in values]
            for key, values in scenario.get("initial_positions", {}).items()
        },
        "goals": {
            str(key): [float(value) for value in values]
            for key, values in scenario.get("goals", {}).items()
        },
        "obstacle_specs": obstacle_specs,
        "bounds_min_m": list(_BOUNDS_MIN),
        "bounds_max_m": list(_BOUNDS_MAX),
    }
    if "obstacles" in scenario:
        payload["obstacles"] = []
    return _canonical(payload)


def _vec(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("scene vectors must contain three finite values")
    return result


def _add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return _vec([float(a) + float(b) for a, b in zip(left, right)])


def _sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return _vec([float(a) - float(b) for a, b in zip(left, right)])


def _scale(vector: Sequence[float], factor: float) -> list[float]:
    return _vec([float(value) * float(factor) for value in vector])


def _normalise(vector: Sequence[float]) -> list[float]:
    values = _vec(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    return _scale(values, 1.0 / norm)


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    a = _vec(left)
    b = _vec(right)
    return _vec((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def _first_nonparallel_axis(direction: Sequence[float]) -> list[float]:
    """Return the first Cartesian basis vector not parallel to direction."""

    values = _vec(direction)
    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        # This also handles non-axis directions without an arbitrary epsilon
        # choice: a basis is parallel only when the cross product vanishes.
        if math.sqrt(sum(component * component for component in _cross(values, axis))) > 1.0e-12:
            return list(axis)
    raise ValueError("direction is parallel to every Cartesian axis")


def _tangent(direction: Sequence[float]) -> list[float]:
    basis = _first_nonparallel_axis(direction)
    return _normalise(_cross(direction, basis))


def _velocity_for_regime(direction: Sequence[float], regime: str) -> list[float]:
    if regime == "rest":
        return [0.0, 0.0, 0.0]
    if regime == "toward":
        return _scale(direction, 1.5)
    if regime == "away":
        return _scale(direction, -1.0)
    if regime == "tangent":
        return _scale(_tangent(direction), 1.0)
    raise ValueError(f"unknown navigation regime: {regime!r}")


def _scene(
    *,
    kind: str,
    start: Sequence[float],
    velocity: Sequence[float],
    goal: Sequence[float],
    obstacle_specs: Sequence[Mapping[str, Any]] = (),
    explicit_empty_obstacles: bool,
) -> dict[str, Any]:
    specs = [copy.deepcopy(dict(spec)) for spec in obstacle_specs]
    scene: dict[str, Any] = {
        "initial_positions": {_UAV_ID: _vec(start)},
        "initial_velocities": {_UAV_ID: _vec(velocity)},
        "initial_attitudes": {_UAV_ID: [0.0, 0.0, 0.0]},
        "goals": {_UAV_ID: _vec(goal)},
        "obstacle_specs": specs,
        "foundation_task_kind": str(kind),
    }
    if explicit_empty_obstacles:
        scene["obstacles"] = []
    return scene


def _case(
    *,
    scenario_id: str,
    kind: str,
    regime: str,
    noise_seed: int,
    scenario: Mapping[str, Any],
    references: Sequence[Sequence[float]],
    side: str = "none",
    belief_position_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
    belief_velocity_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    if kind not in {"normal", "single"}:
        raise ValueError(f"unsupported fixed-navigation kind: {kind!r}")
    if regime not in _REGIMES:
        raise ValueError(f"unsupported fixed-navigation regime: {regime!r}")
    scene = copy.deepcopy(dict(scenario))
    goal = scene["goals"][_UAV_ID]
    refs = [_vec(point) for point in references]
    if not refs:
        raise ValueError("each fixed-navigation case needs a reference point")
    if refs[-1] != _vec(goal):
        raise ValueError("the final static reference must equal the mission goal")
    return {
        "scenario_id": str(scenario_id),
        "kind": str(kind),
        "regime": str(regime),
        "noise_seed": int(noise_seed),
        "scenario": scene,
        "references": refs,
        "belief_position_offset_m": _vec(belief_position_offset_m),
        "belief_velocity_offset_m": _vec(belief_velocity_offset_m),
        "side": str(side),
        "geometry_sha256": _geometry_hash(scene),
        # Every phase uses the same frozen execution mode.  Keep this
        # explicit on every case so logs and manifests are self-contained.
        "safety2_enabled": True,
        "assist_mask": "00000",
        "pass_commit": False,
        "lateral_assist": False,
        "radial_assist": False,
        "goal_recovery": False,
        "terminal_capture": False,
    }


def _midpoint(index: int, base_x: float, base_y: float) -> list[float]:
    return [base_x + 0.7 * float(index), base_y + 0.55 * float(index), 30.0]


def _direct_case(
    *,
    scenario_id: str,
    kind: str,
    regime: str,
    noise_seed: int,
    direction: Sequence[float],
    distance: float,
    midpoint: Sequence[float],
    side: str = "none",
    belief_position_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
    belief_velocity_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    unit_direction = _normalise(direction)
    half_displacement = _scale(unit_direction, float(distance) / 2.0)
    start = _sub(midpoint, half_displacement)
    goal = _add(midpoint, half_displacement)
    scene = _scene(
        kind=kind,
        start=start,
        velocity=_velocity_for_regime(unit_direction, regime),
        goal=goal,
        explicit_empty_obstacles=True,
    )
    return _case(
        scenario_id=scenario_id,
        kind=kind,
        regime=regime,
        noise_seed=noise_seed,
        scenario=scene,
        references=[goal],
        side=side,
        belief_position_offset_m=belief_position_offset_m,
        belief_velocity_offset_m=belief_velocity_offset_m,
    )


def _development_cases() -> list[dict[str, Any]]:
    directions = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    distances = (8.0, 12.0, 8.0, 12.0, 8.0, 12.0)
    regimes = ("rest", "toward", "away", "tangent", "rest", "toward")
    cases: list[dict[str, Any]] = []
    for index, (direction, distance, regime) in enumerate(zip(directions, distances, regimes)):
        cases.append(
            _direct_case(
                scenario_id=f"fixednav_dev{index:02d}",
                kind="normal",
                regime=regime,
                noise_seed=2026114100 + index,
                direction=direction,
                distance=distance,
                midpoint=[40.0 + 1.2 * index, 40.0 + 1.7 * index, 30.0],
            )
        )
    return cases


def _normal_evaluation_cases() -> list[dict[str, Any]]:
    axes = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    cases: list[dict[str, Any]] = []
    index = 0
    for axis_index, direction in enumerate(axes):
        for regime_index, regime in enumerate(_REGIMES):
            cases.append(
                _direct_case(
                    scenario_id=f"fixednav_normal{index:02d}",
                    kind="normal",
                    regime=regime,
                    noise_seed=2026114200 + index,
                    direction=direction,
                    distance=_DISTANCES[(axis_index + regime_index) % len(_DISTANCES)],
                    midpoint=_midpoint(index, 85.0, 60.0),
                )
            )
            index += 1

    corner_signs = (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    )
    for corner_index, direction in enumerate(corner_signs):
        regime = _REGIMES[corner_index % len(_REGIMES)]
        cases.append(
            _direct_case(
                scenario_id=f"fixednav_normal{index:02d}",
                kind="normal",
                regime=regime,
                noise_seed=2026114200 + index,
                direction=direction,
                distance=_DISTANCES[corner_index % len(_DISTANCES)],
                midpoint=_midpoint(corner_index, 115.0, 85.0),
            )
        )
        index += 1

    unique_directions = (
        (1.0, 2.0, 1.0),
        (-2.0, 1.0, 1.0),
        (1.0, -1.0, 2.0),
        (-1.0, -2.0, 1.0),
    )
    position_signs = (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    for unique_index, direction in enumerate(unique_directions):
        position_offset = _scale(_normalise(position_signs[unique_index]), 0.4)
        # The velocity tetrahedron is rotated by one element relative to the
        # position tetrahedron, so each challenge has a distinct paired bias.
        velocity_offset = _scale(_normalise(position_signs[(unique_index + 1) % 4]), 0.2)
        cases.append(
            _direct_case(
                scenario_id=f"fixednav_normal{index:02d}",
                kind="normal",
                regime=_REGIMES[unique_index],
                noise_seed=2026114200 + index,
                direction=direction,
                distance=_DISTANCES[unique_index],
                midpoint=_midpoint(unique_index, 135.0, 110.0),
                belief_position_offset_m=position_offset,
                belief_velocity_offset_m=velocity_offset,
            )
        )
        index += 1
    return cases


def _paired_cases() -> list[dict[str, Any]]:
    direction = (1.0, 0.0, 0.0)
    cases: list[dict[str, Any]] = []
    for index, regime in enumerate(_REGIMES):
        cases.append(
            _direct_case(
                scenario_id=f"fixednav_paired{index:02d}",
                kind="normal",
                regime=regime,
                noise_seed=2026114300 + index,
                direction=direction,
                distance=12.0,
                midpoint=[75.0 + 2.0 * index, 145.0 + float(index), 30.0],
            )
        )
    return cases


def _obstacle_spec(obstacle_id: str, center: Sequence[float]) -> dict[str, Any]:
    return {
        "obstacle_id": obstacle_id,
        "geometry_type": "box",
        "center_m": _vec(center),
        "size_m": [3.0, 3.0, 8.0],
        "velocity_mps": [0.0, 0.0, 0.0],
        "obstacle_type": "building",
        "active": True,
        "physical_collision_enabled": True,
    }


def _obstacle_case(
    *,
    scenario_id: str,
    noise_seed: int,
    side: str,
    start: Sequence[float],
    goal: Sequence[float],
    center: Sequence[float],
    references: Sequence[Sequence[float]],
) -> dict[str, Any]:
    scene = _scene(
        kind="single",
        start=start,
        velocity=(0.0, 0.0, 0.0),
        goal=goal,
        obstacle_specs=[_obstacle_spec(scenario_id, center)],
        explicit_empty_obstacles=False,
    )
    return _case(
        scenario_id=scenario_id,
        kind="single",
        regime="rest",
        noise_seed=noise_seed,
        scenario=scene,
        references=references,
        side=side,
    )


def _obstacle_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    x_start = [55.0, 110.0, 22.0]
    x_goal = [79.0, 110.0, 22.0]
    x_center = [67.0, 110.0, 22.0]
    for index, side_sign in enumerate((1.0, -1.0)):
        x_refs = [
            _add(x_start, (7.0, 5.5 * side_sign, 0.0)),
            _add(x_start, (17.0, 5.5 * side_sign, 0.0)),
            _add(x_start, (24.0, 0.0, 0.0)),
        ]
        side = "left" if side_sign > 0 else "right"
        cases.append(
            _obstacle_case(
                scenario_id=f"box_x_{side}",
                noise_seed=2026114400 + index,
                side=side,
                start=x_start,
                goal=x_goal,
                center=x_center,
                references=x_refs,
            )
        )

    y_start = [110.0, 70.0, 28.0]
    y_goal = [110.0, 94.0, 28.0]
    y_center = [110.0, 82.0, 28.0]
    for index, side_sign in enumerate((1.0, -1.0), start=2):
        # Rotate the X-route displacement by (-y, x), as specified by the
        # second independent geometry.  X-left (+y) therefore maps to -x.
        x_route_offsets = ((7.0, 5.5 * side_sign, 0.0), (17.0, 5.5 * side_sign, 0.0), (24.0, 0.0, 0.0))
        y_refs = [_add(y_start, (-offset[1], offset[0], offset[2])) for offset in x_route_offsets]
        side = "left" if side_sign > 0 else "right"
        cases.append(
            _obstacle_case(
                scenario_id=f"box_y_{side}",
                noise_seed=2026114400 + index,
                side=side,
                start=y_start,
                goal=y_goal,
                center=y_center,
                references=y_refs,
            )
        )
    return cases


def build_fixed_navigation_scenarios() -> dict[str, list[dict[str, Any]]]:
    """Return the frozen, side-effect-free fixed-navigation scenario manifest."""

    # Construct fresh records each time.  Callers may pass the nested reset
    # dictionaries to an environment or annotate records without changing a
    # later manifest produced by this function.
    return {
        "development": _development_cases(),
        "normal_evaluation": _normal_evaluation_cases(),
        "paired": _paired_cases(),
        "obstacle": _obstacle_cases(),
    }


__all__ = ["build_fixed_navigation_scenarios"]
