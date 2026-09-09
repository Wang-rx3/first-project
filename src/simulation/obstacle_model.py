"""Three-dimensional obstacle, collision, and LOS geometry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class BoxObstacle:
    obstacle_id: str
    center_m: np.ndarray
    size_m: np.ndarray
    velocity_mps: np.ndarray | None = None
    obstacle_type: str = "building"
    active: bool = True
    # Communication-only geometry remains visible to line_of_sight(), but is
    # not part of the physical collision/safety geometry.  Keep this field at
    # the end so historical positional constructors remain compatible.
    physical_collision_enabled: bool = True

    @property
    def minimum_m(self) -> np.ndarray:
        return self.center_m - self.size_m / 2.0

    @property
    def maximum_m(self) -> np.ndarray:
        return self.center_m + self.size_m / 2.0

    def update(self, dt_s: float, lower: np.ndarray, upper: np.ndarray) -> None:
        if self.velocity_mps is None or not self.active:
            return
        next_center = self.center_m + self.velocity_mps * dt_s
        for axis in range(3):
            half = self.size_m[axis] / 2.0
            if next_center[axis] - half < lower[axis] or next_center[axis] + half > upper[axis]:
                self.velocity_mps[axis] *= -1.0
                next_center[axis] = np.clip(next_center[axis], lower[axis] + half, upper[axis] - half)
        self.center_m = next_center


@dataclass
class SphereObstacle:
    obstacle_id: str
    center_m: np.ndarray
    radius_m: float
    velocity_mps: np.ndarray | None = None
    obstacle_type: str = "dynamic_vehicle"
    active: bool = True
    # See BoxObstacle.physical_collision_enabled.
    physical_collision_enabled: bool = True

    def update(self, dt_s: float, lower: np.ndarray, upper: np.ndarray) -> None:
        if self.velocity_mps is None or not self.active:
            return
        next_center = self.center_m + self.velocity_mps * dt_s
        for axis in range(3):
            if next_center[axis] - self.radius_m < lower[axis] or next_center[axis] + self.radius_m > upper[axis]:
                self.velocity_mps[axis] *= -1.0
                next_center[axis] = np.clip(next_center[axis], lower[axis] + self.radius_m, upper[axis] - self.radius_m)
        self.center_m = next_center


Obstacle = BoxObstacle | SphereObstacle


def is_physical_obstacle(obstacle: Obstacle) -> bool:
    """Return whether ``obstacle`` participates in physical safety geometry.

    Missing attributes are treated as physical so old in-memory objects and
    old scenario files retain their historical semantics.
    """

    return bool(getattr(obstacle, "physical_collision_enabled", True))


def point_box_distance(point: np.ndarray, obstacle: BoxObstacle) -> float:
    point = np.asarray(point, dtype=float)
    closest = np.minimum(np.maximum(point, obstacle.minimum_m), obstacle.maximum_m)
    return float(np.linalg.norm(point - closest))


def point_obstacle_distance(point: np.ndarray, obstacle: Obstacle) -> float:
    if isinstance(obstacle, BoxObstacle):
        return point_box_distance(point, obstacle)
    return max(0.0, float(np.linalg.norm(np.asarray(point) - obstacle.center_m) - obstacle.radius_m))


def segment_box_intersects(start: np.ndarray, end: np.ndarray, obstacle: BoxObstacle) -> bool:
    """Slab intersection test for a closed 3-D line segment and AABB."""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < obstacle.minimum_m[axis] or start[axis] > obstacle.maximum_m[axis]:
                return False
            continue
        inv = 1.0 / direction[axis]
        near = (obstacle.minimum_m[axis] - start[axis]) * inv
        far = (obstacle.maximum_m[axis] - start[axis]) * inv
        if near > far:
            near, far = far, near
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return False
    return True


def segment_sphere_intersects(start: np.ndarray, end: np.ndarray, obstacle: SphereObstacle) -> bool:
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return float(np.linalg.norm(start - obstacle.center_m)) <= obstacle.radius_m
    projection = float(np.dot(obstacle.center_m - start, direction) / denominator)
    projection = min(1.0, max(0.0, projection))
    closest = start + projection * direction
    return float(np.linalg.norm(closest - obstacle.center_m)) <= obstacle.radius_m


def line_of_sight(start: np.ndarray, end: np.ndarray, obstacles: Iterable[Obstacle]) -> bool:
    """Return true if no active 3-D obstacle intersects the UAV segment."""

    for obstacle in obstacles:
        if not obstacle.active:
            continue
        if isinstance(obstacle, BoxObstacle) and segment_box_intersects(start, end, obstacle):
            return False
        if isinstance(obstacle, SphereObstacle) and segment_sphere_intersects(start, end, obstacle):
            return False
    return True


def segment_intersects_obstacle(start: np.ndarray, end: np.ndarray, obstacle: Obstacle) -> bool:
    """Return whether a 3-D motion segment intersects one active obstacle."""

    if not obstacle.active or not is_physical_obstacle(obstacle):
        return False
    if isinstance(obstacle, BoxObstacle):
        return segment_box_intersects(start, end, obstacle)
    return segment_sphere_intersects(start, end, obstacle)


def moving_obstacle_swept_segment_intersects(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: Obstacle,
    dt_s: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    safety_distance_m: float,
) -> bool:
    """Return whether a UAV segment intersects a one-step moving safety envelope.

    The existing safety projector checks a moving UAV against the obstacle at
    the beginning of the step.  This helper adds a conservative, deterministic
    one-step envelope for obstacles with a velocity.  The envelope contains
    the obstacle geometry, the configured safety distance, and the complete
    center sweep between the current and next obstacle states.  Boundary
    reflections use the same ``Obstacle.update`` rule as the environment; a
    possible boundary contact is included in the envelope as well.

    The calculation is pure: it deep-copies the obstacle before prediction and
    does not consume randomness or mutate the live simulation state.
    """

    if (
        not obstacle.active
        or not is_physical_obstacle(obstacle)
        or obstacle.velocity_mps is None
    ):
        return False
    try:
        start = np.asarray(start, dtype=float).reshape(3)
        end = np.asarray(end, dtype=float).reshape(3)
        lower = np.asarray(bounds_min, dtype=float).reshape(3)
        upper = np.asarray(bounds_max, dtype=float).reshape(3)
        velocity = np.asarray(obstacle.velocity_mps, dtype=float).reshape(3)
        center = np.asarray(obstacle.center_m, dtype=float).reshape(3)
        dt = float(dt_s)
        safety = float(safety_distance_m)
    except (TypeError, ValueError):
        return False
    if (
        not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(center).all()
        or not np.isfinite(dt)
        or not np.isfinite(safety)
        or dt < 0.0
        or safety < 0.0
        or np.any(lower >= upper)
    ):
        return False

    # Predict the endpoint with the exact environment update, including a
    # possible boundary reflection.  Deep copy is required because
    # ``Obstacle.update`` flips velocity in place at a boundary.
    predicted = copy.deepcopy(obstacle)
    predicted.update(dt, lower, upper)
    predicted_center = np.asarray(predicted.center_m, dtype=float).reshape(3)

    if isinstance(obstacle, BoxObstacle):
        extent = np.asarray(obstacle.size_m, dtype=float).reshape(3) / 2.0
    elif isinstance(obstacle, SphereObstacle):
        radius = float(obstacle.radius_m)
        if not np.isfinite(radius) or radius < 0.0:
            return False
        extent = np.full(3, radius, dtype=float)
    else:
        return False
    if not np.isfinite(extent).all() or np.any(extent < 0.0):
        return False

    center_min = np.minimum(center, predicted_center)
    center_max = np.maximum(center, predicted_center)

    # If the one-step update reflects at a bound, the contact point can lie
    # beyond the interval [center, predicted_center]. Include it so the
    # envelope remains conservative under the environment's bounce rule.
    raw_center = center + velocity * dt
    feasible_min = lower + extent
    feasible_max = upper - extent
    for axis in range(3):
        if raw_center[axis] < feasible_min[axis]:
            center_min[axis] = min(center_min[axis], feasible_min[axis])
        if raw_center[axis] > feasible_max[axis]:
            center_max[axis] = max(center_max[axis], feasible_max[axis])

    envelope_min = center_min - extent - safety
    envelope_max = center_max + extent + safety
    envelope = BoxObstacle(
        obstacle_id=f"{obstacle.obstacle_id}__one_step_swept_safety",
        center_m=(envelope_min + envelope_max) / 2.0,
        size_m=envelope_max - envelope_min,
        velocity_mps=np.zeros(3, dtype=float),
        obstacle_type="swept_safety_envelope",
        active=True,
    )
    return bool(segment_box_intersects(start, end, envelope))


def segment_minimum_distance(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    """Return the minimum Euclidean distance between two 3-D segments."""

    p = np.asarray(start_a, dtype=float).reshape(3)
    q = np.asarray(end_a, dtype=float).reshape(3)
    r = np.asarray(start_b, dtype=float).reshape(3)
    s = np.asarray(end_b, dtype=float).reshape(3)
    u = q - p
    v = s - r
    w = p - r
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    eps = 1e-12
    denominator = a * c - b * b

    if a <= eps and c <= eps:
        return float(np.linalg.norm(p - r))
    if a <= eps:
        t = np.clip(e / c, 0.0, 1.0)
        return float(np.linalg.norm(p - (r + t * v)))
    if c <= eps:
        t = np.clip(-d / a, 0.0, 1.0)
        return float(np.linalg.norm((p + t * u) - r))

    # Closest points on two non-parallel segments.  The boundary branches
    # below are the standard clamped solution for the parallel/endpoint cases.
    if denominator <= eps:
        sc = 0.0
        tc = np.clip(e / c, 0.0, 1.0)
    else:
        sc = (b * e - c * d) / denominator
        tc = (a * e - b * d) / denominator
        if sc < 0.0:
            sc = 0.0
            tc = np.clip(e / c, 0.0, 1.0)
        elif sc > 1.0:
            sc = 1.0
            tc = np.clip((e + b) / c, 0.0, 1.0)
        elif tc < 0.0:
            tc = 0.0
            sc = np.clip(-d / a, 0.0, 1.0)
        elif tc > 1.0:
            tc = 1.0
            sc = np.clip((b - d) / a, 0.0, 1.0)
    return float(np.linalg.norm((p + sc * u) - (r + tc * v)))


def joint_motion_collision_diagnostics(
    previous_positions: dict[str, np.ndarray],
    next_positions: dict[str, np.ndarray],
    obstacles_before: Iterable[Obstacle],
    obstacles_after: Iterable[Obstacle],
    *,
    dt_s: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    safety_distance_m: float,
) -> dict[str, object]:
    """Check simultaneous UAV motion against physical obstacles.

    Communication-only geometry is intentionally excluded from this function;
    :func:`line_of_sight` remains the separate link-visibility path and still
    considers every active geometry.  The result is pure and does not mutate
    either obstacle list.
    """

    before = list(obstacles_before)
    after = list(obstacles_after)
    pair_conflicts: list[dict[str, object]] = []
    obstacle_conflicts: list[dict[str, object]] = []
    minimum_clearance = float("inf")
    ids = sorted(next_positions)

    for index, source in enumerate(ids):
        for target in ids[index + 1 :]:
            endpoint_distance = float(
                np.linalg.norm(next_positions[source] - next_positions[target])
            )
            swept_distance = segment_minimum_distance(
                previous_positions[source],
                next_positions[source],
                previous_positions[target],
                next_positions[target],
            )
            minimum_clearance = min(minimum_clearance, endpoint_distance, swept_distance)
            if endpoint_distance < safety_distance_m or swept_distance < safety_distance_m:
                physical_contact = bool(
                    endpoint_distance <= 1e-9 or swept_distance <= 1e-9
                )
                pair_conflicts.append(
                    {
                        "source_uav_id": source,
                        "target_uav_id": target,
                        "endpoint_distance_m": endpoint_distance,
                        "swept_distance_m": swept_distance,
                        "physical_contact": physical_contact,
                        "clearance_violation": not physical_contact,
                        "conflict_type": (
                            "endpoint_clearance"
                            if endpoint_distance < safety_distance_m
                            else "swept_path_clearance"
                        ),
                    }
                )

    physical_before = [obstacle for obstacle in before if is_physical_obstacle(obstacle)]
    after_by_id = {
        str(obstacle.obstacle_id): obstacle
        for obstacle in after
        if is_physical_obstacle(obstacle)
    }
    for uav_id in ids:
        start = previous_positions[uav_id]
        end = next_positions[uav_id]
        for obstacle in physical_before:
            predicted = after_by_id.get(str(obstacle.obstacle_id), obstacle)
            endpoint_clearance = point_obstacle_distance(end, predicted)
            minimum_clearance = min(minimum_clearance, endpoint_clearance)
            static_segment = segment_intersects_obstacle(start, end, obstacle)
            moving_segment = moving_obstacle_swept_segment_intersects(
                start,
                end,
                obstacle,
                dt_s,
                bounds_min,
                bounds_max,
                safety_distance_m,
            )
            if endpoint_clearance < safety_distance_m or static_segment or moving_segment:
                physical_contact = bool(
                    endpoint_clearance <= 1e-9
                    or static_segment
                    or moving_obstacle_swept_segment_intersects(
                        start,
                        end,
                        obstacle,
                        dt_s,
                        bounds_min,
                        bounds_max,
                        0.0,
                    )
                )
                obstacle_conflicts.append(
                    {
                        "uav_id": uav_id,
                        "obstacle_id": str(obstacle.obstacle_id),
                        "endpoint_clearance_m": endpoint_clearance,
                        "physical_contact": physical_contact,
                        "clearance_violation": not physical_contact,
                        "conflict_type": (
                            "endpoint_clearance"
                            if endpoint_clearance < safety_distance_m
                            else "moving_swept_path"
                            if moving_segment
                            else "static_swept_path"
                        ),
                    }
                )

    if minimum_clearance == float("inf"):
        minimum_clearance = float(safety_distance_m)
    all_conflicts = pair_conflicts + obstacle_conflicts
    return {
        "collision": bool(all_conflicts),
        "physical_contact": any(
            bool(conflict.get("physical_contact", False)) for conflict in all_conflicts
        ),
        "clearance_violation": any(
            bool(conflict.get("clearance_violation", False)) for conflict in all_conflicts
        ),
        "minimum_clearance_m": float(minimum_clearance),
        "uav_pair_conflicts": pair_conflicts,
        "obstacle_conflicts": obstacle_conflicts,
        "ignored_nonphysical_obstacle_ids": sorted(
            str(obstacle.obstacle_id)
            for obstacle in before
            if not is_physical_obstacle(obstacle)
        ),
    }


def collision_event(
    positions: dict[str, np.ndarray],
    obstacles: Iterable[Obstacle],
    safety_distance_m: float,
) -> tuple[int, float]:
    """Return collision flag and minimum scene clearance."""

    minimum = float("inf")
    ids = sorted(positions)
    for index, source in enumerate(ids):
        for target in ids[index + 1 :]:
            minimum = min(minimum, float(np.linalg.norm(positions[source] - positions[target])))
    for position in positions.values():
        for obstacle in obstacles:
            if obstacle.active and is_physical_obstacle(obstacle):
                minimum = min(minimum, point_obstacle_distance(position, obstacle))
    if minimum == float("inf"):
        minimum = safety_distance_m
    return int(minimum < safety_distance_m), minimum


__all__ = [
    "BoxObstacle",
    "SphereObstacle",
    "Obstacle",
    "is_physical_obstacle",
    "collision_event",
    "line_of_sight",
    "point_obstacle_distance",
    "segment_intersects_obstacle",
    "moving_obstacle_swept_segment_intersects",
    "segment_minimum_distance",
    "joint_motion_collision_diagnostics",
    "segment_box_intersects",
    "segment_sphere_intersects",
]
