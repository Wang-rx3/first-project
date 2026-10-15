"""Pure helpers for the Single-UAV signed bypass observability trace.

The helpers in this module are deliberately side-effect free.  They only
project already available vectors into the frozen horizontal goal frame and
maintain diagnostic bookkeeping; they never call an environment, policy,
controller, random-number generator, or safety projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION = "signed_bypass_observability_v1"
LATERAL_AXIS_SOURCE = "goal_projected_xy_left"
LATERAL_SIGN_CONVENTION = "+1=goal-frame-left,-1=goal-frame-right"
VERTICAL_AXIS_SOURCE = "enu_world_up"
DEFAULT_SIGN_DEADBAND_MPS = 1.0e-8
DEFAULT_SIGN_DEADBAND_M = 1.0e-8
NOT_INDEPENDENTLY_VERIFIED = "NOT_INDEPENDENTLY_VERIFIED"


_IDENTITY_COMPONENT_CHECK_KEYS = {
    "reward_identity": "same_reward",
    "dynamics_identity": "same_dynamics",
    "action_transform_identity": "same_action_transform",
    "safety_projection_identity": "same_safety_projection",
    "boundary_projection_identity": "same_boundary_projection",
}

_INDEPENDENT_IDENTITY_FIELD_ALIASES = {
    "reward_identity": ("reward_identity", "reward_identity_hash", "reward_config_hash"),
    "dynamics_identity": (
        "dynamics_identity",
        "dynamics_identity_hash",
        "dynamics_config_hash",
    ),
    "action_transform_identity": (
        "action_transform_identity",
        "action_transform_identity_hash",
        "action_transform_hash",
    ),
    "safety_projection_identity": (
        "safety_projection_identity",
        "safety_projection_identity_hash",
        "safety_projection_hash",
    ),
    "boundary_projection_identity": (
        "boundary_projection_identity",
        "boundary_projection_identity_hash",
        "boundary_projection_hash",
    ),
}


class SignedLateralAxisNotWellDefined(ValueError):
    """Raised when no deterministic horizontal signed axis can be built."""

    code = "SIGNED_LATERAL_AXIS_NOT_WELL_DEFINED"


@dataclass(frozen=True)
class SignedFrame:
    """An orthonormal horizontal goal frame with an explicit sign contract."""

    radial_unit: np.ndarray
    lateral_unit: np.ndarray
    vertical_unit: np.ndarray
    axis_source: str = LATERAL_AXIS_SOURCE
    sign_convention: str = LATERAL_SIGN_CONVENTION
    vertical_axis_source: str = VERTICAL_AXIS_SOURCE

    def __post_init__(self) -> None:
        for name in ("radial_unit", "lateral_unit", "vertical_unit"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(3)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value.copy())


def _vector3(value: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def build_horizontal_goal_frame(
    goal_delta_m: Iterable[float],
    *,
    tolerance_m: float = 1.0e-9,
) -> SignedFrame:
    """Build the canonical horizontal frame used by the frozen static pool.

    The frozen static holdout has a horizontal start--goal ray and a separate
    ENU vertical axis.  A non-horizontal goal ray is rejected rather than
    silently inventing a tangent-plane left/right sign.
    """

    delta = _vector3(goal_delta_m, name="goal_delta_m")
    if abs(float(delta[2])) > max(float(tolerance_m), 0.0):
        raise SignedLateralAxisNotWellDefined(
            f"{SignedLateralAxisNotWellDefined.code}: goal ray is not horizontal"
        )
    horizontal = delta.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if not np.isfinite(norm) or norm <= max(float(tolerance_m), 0.0):
        raise SignedLateralAxisNotWellDefined(
            f"{SignedLateralAxisNotWellDefined.code}: goal XY projection is degenerate"
        )
    radial = horizontal / norm
    # ENU +z is up; rotating +x/+y left gives the existing body +y=left sign.
    lateral = np.asarray([-radial[1], radial[0], 0.0], dtype=float)
    vertical = np.asarray([0.0, 0.0, 1.0], dtype=float)
    return SignedFrame(radial, lateral, vertical)


def frame_orthogonality(frame: SignedFrame) -> dict[str, float]:
    """Return norm and pairwise-dot checks for a frame."""

    r, l, z = frame.radial_unit, frame.lateral_unit, frame.vertical_unit
    return {
        "radial_norm_error": abs(float(np.linalg.norm(r)) - 1.0),
        "lateral_norm_error": abs(float(np.linalg.norm(l)) - 1.0),
        "vertical_norm_error": abs(float(np.linalg.norm(z)) - 1.0),
        "radial_lateral_dot": float(np.dot(r, l)),
        "radial_vertical_dot": float(np.dot(r, z)),
        "lateral_vertical_dot": float(np.dot(l, z)),
    }


def decompose_signed_vector(
    vector: Iterable[float],
    frame: SignedFrame,
) -> dict[str, float]:
    """Project one 3-D vector into radial/lateral/vertical coordinates."""

    value = _vector3(vector, name="vector")
    radial = float(np.dot(value, frame.radial_unit))
    lateral = float(np.dot(value, frame.lateral_unit))
    vertical = float(np.dot(value, frame.vertical_unit))
    residual = float(
        np.linalg.norm(
            value
            - radial * frame.radial_unit
            - lateral * frame.lateral_unit
            - vertical * frame.vertical_unit
        )
    )
    return {
        "radial": radial,
        "lateral": lateral,
        "vertical": vertical,
        "reconstruction_residual": residual,
    }


def signed_side(value: float, *, deadband: float = DEFAULT_SIGN_DEADBAND_M) -> str:
    """Classify a signed scalar without forcing a side near zero."""

    number = float(value)
    band = max(float(deadband), 0.0)
    if not np.isfinite(number):
        return "AMBIGUOUS"
    if number > band:
        return "LEFT"
    if number < -band:
        return "RIGHT"
    return "CENTER"


def sign_value(value: float, *, deadband: float = DEFAULT_SIGN_DEADBAND_M) -> int:
    side = signed_side(value, deadband=deadband)
    return 1 if side == "LEFT" else -1 if side == "RIGHT" else 0


def sign_change_count(
    values: Sequence[float],
    *,
    deadband: float = DEFAULT_SIGN_DEADBAND_MPS,
) -> int:
    """Count non-zero sign changes while ignoring CENTER/NA samples."""

    previous = 0
    changes = 0
    for value in values:
        current = sign_value(float(value), deadband=deadband)
        if current == 0:
            continue
        if previous and current != previous:
            changes += 1
        previous = current
    return int(changes)


def same_side_run_length(
    values: Sequence[float],
    *,
    reference_sign: int | None = None,
    deadband: float = DEFAULT_SIGN_DEADBAND_MPS,
) -> int:
    """Return the longest contiguous run on one chosen signed side."""

    reference = int(reference_sign or 0)
    if reference == 0:
        for value in values:
            reference = sign_value(float(value), deadband=deadband)
            if reference:
                break
    if reference == 0:
        return 0
    current = 0
    maximum = 0
    for value in values:
        if sign_value(float(value), deadband=deadband) == reference:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return int(maximum)


def update_side_run(
    previous_run: int,
    value: float,
    reference_sign: int,
    *,
    deadband: float = DEFAULT_SIGN_DEADBAND_MPS,
) -> int:
    """Update a current same-side run length for one step."""

    return int(previous_run + 1) if sign_value(value, deadband=deadband) == int(reference_sign) else 0


def side_agreement(
    first: Sequence[float],
    second: Sequence[float],
    *,
    deadband: float = DEFAULT_SIGN_DEADBAND_MPS,
) -> float:
    """Fraction of paired non-center signs that agree."""

    pairs = zip(first, second)
    comparable = 0
    agree = 0
    for left, right in pairs:
        a = sign_value(float(left), deadband=deadband)
        b = sign_value(float(right), deadband=deadband)
        if not a or not b:
            continue
        comparable += 1
        agree += int(a == b)
    return float(agree / comparable) if comparable else float("nan")


def passed_geometry_fields(
    along_obstacle_m: float,
    obstacle_extent_m: float,
    clearance_m: float,
    *,
    buffer_m: float,
    safe_distance_m: float,
    clearance_margin_m: float = 0.5,
) -> dict[str, float | bool]:
    """Re-express the existing recovery tracker release gate as diagnostics."""

    along = float(along_obstacle_m)
    extent = float(obstacle_extent_m)
    clearance = float(clearance_m)
    threshold = extent + float(buffer_m)
    margin = -(along + threshold)
    soft = bool(np.isfinite(margin) and margin > 0.0)
    clear = bool(
        np.isfinite(clearance)
        and clearance >= float(safe_distance_m) + float(clearance_margin_m)
    )
    return {
        "passed_threshold_m": float(threshold),
        "passed_margin_m": float(margin),
        "passed_condition_now": bool(soft and clear),
        "soft_recovery_condition_now": bool(soft),
    }


def pipeline_signed_components(
    vectors: Mapping[str, Iterable[float]],
    frame: SignedFrame,
) -> dict[str, float]:
    """Project each action-pipeline vector with stable field names."""

    result: dict[str, float] = {}
    for name, vector in vectors.items():
        components = decompose_signed_vector(vector, frame)
        result[f"{name}_radial"] = components["radial"]
        result[f"{name}_lateral"] = components["lateral"]
        result[f"{name}_vertical"] = components["vertical"]
    return result


def trajectory_fingerprint(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> str:
    """Hash only pre-existing behavior fields for logging-perturbation checks."""

    payload = [[row.get(key) for key in keys] for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identity_check_container(
    evaluation: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the existing identity-check mapping, if one is present."""

    checks = evaluation.get("previous_formal_identity_checks")
    return checks if isinstance(checks, Mapping) else None


def _independent_identity_value(
    evaluation: Mapping[str, Any],
    *,
    component: str,
) -> tuple[Any, str | None]:
    """Find an explicitly persisted component identity value.

    Aggregate checks such as ``same_reward`` are intentionally excluded here:
    they are useful supporting context, but are not independent component
    digests/config snapshots.
    """

    aliases = _INDEPENDENT_IDENTITY_FIELD_ALIASES[component]
    for container_name in (
        "independent_identity_fields",
        "component_identity_fields",
        "identity_fields",
    ):
        container = evaluation.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for alias in aliases:
            if alias in container:
                value = container[alias]
                if isinstance(value, Mapping) and "status" in value:
                    value = value.get("value", value.get("status"))
                return value, f"{container_name}.{alias}"
    for alias in aliases:
        if alias in evaluation:
            value = evaluation[alias]
            if isinstance(value, Mapping) and "status" in value:
                value = value.get("value", value.get("status"))
            return value, alias
    return None, None


def _independent_identity_status(current: Any, baseline: Any) -> bool | str:
    """Compare two explicit identity values, failing closed on missing data."""

    if current is None or baseline is None:
        return NOT_INDEPENDENTLY_VERIFIED
    if (
        isinstance(current, str)
        and current == NOT_INDEPENDENTLY_VERIFIED
    ) or (
        isinstance(baseline, str)
        and baseline == NOT_INDEPENDENTLY_VERIFIED
    ):
        return NOT_INDEPENDENTLY_VERIFIED
    if isinstance(current, bool) or isinstance(baseline, bool):
        if not isinstance(current, bool) or not isinstance(baseline, bool):
            return NOT_INDEPENDENTLY_VERIFIED
        return bool(current and baseline)
    if isinstance(current, (str, int, float)) and isinstance(baseline, (str, int, float)):
        if isinstance(current, str) and not current.strip():
            return NOT_INDEPENDENTLY_VERIFIED
        if isinstance(baseline, str) and not baseline.strip():
            return NOT_INDEPENDENTLY_VERIFIED
        return bool(current == baseline)
    return NOT_INDEPENDENTLY_VERIFIED


def compare_identity_component(
    current_evaluation: Mapping[str, Any],
    baseline_evaluation: Mapping[str, Any],
    *,
    component: str,
) -> dict[str, Any]:
    """Report one frozen-component identity without fabricating a pass.

    The signed diagnostic must not infer reward/dynamics/action/safety/boundary
    identity from the fact that logging was added.  The current horizon120
    evaluation records aggregate comparisons in
    ``previous_formal_identity_checks``.  Those values are retained as audit
    context, but are not independent component identity fields.  Only an
    explicitly persisted component identity value (for example a component
    digest in ``identity_fields``) can produce a boolean result here.

    Missing or malformed independent evidence is explicitly reported as
    ``NOT_INDEPENDENTLY_VERIFIED``.
    """

    if component not in _IDENTITY_COMPONENT_CHECK_KEYS:
        raise KeyError(f"unknown identity component: {component}")
    check_key = _IDENTITY_COMPONENT_CHECK_KEYS[component]
    current_value, current_source = _independent_identity_value(
        current_evaluation,
        component=component,
    )
    baseline_value, baseline_source = _independent_identity_value(
        baseline_evaluation,
        component=component,
    )
    status = _independent_identity_status(current_value, baseline_value)
    current_checks = _identity_check_container(current_evaluation)
    baseline_checks = _identity_check_container(baseline_evaluation)
    legacy_current = current_checks.get(check_key) if current_checks is not None else None
    legacy_baseline = baseline_checks.get(check_key) if baseline_checks is not None else None
    source = (
        "explicit_component_identity_field"
        if current_source is not None and baseline_source is not None
        else "previous_formal_identity_checks_supporting_only"
        if legacy_current is not None or legacy_baseline is not None
        else "unavailable"
    )
    reason = (
        "explicit_component_identity_values_compared"
        if current_source is not None and baseline_source is not None
        else "independent_component_identity_field_unavailable"
    )

    return {
        "status": status,
        "check_key": check_key,
        "source": source,
        "current_value": current_value,
        "baseline_value": baseline_value,
        "current_source": current_source,
        "baseline_source": baseline_source,
        "baseline_value_available": bool(baseline_source),
        "supporting_current_check": legacy_current,
        "supporting_baseline_check": legacy_baseline,
        "reason": reason,
    }


def compare_identity_components(
    current_evaluation: Mapping[str, Any],
    baseline_evaluation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Report all five frozen components with explicit missing-field status."""

    return {
        component: compare_identity_component(
            current_evaluation,
            baseline_evaluation,
            component=component,
        )
        for component in _IDENTITY_COMPONENT_CHECK_KEYS
    }


def classify_signed_failure(
    *,
    assessable: bool,
    initial_obstacle_side: str,
    recommended_side: str,
    passed: bool,
    lateral_sign_changes: int,
    actor_motion_agreement: float,
    radial_progress_mean: float,
    lateral_valid: bool,
) -> str:
    """Apply the preparation-stage mutually exclusive failure labels.

    The function is intentionally conservative: an unavailable geometric
    recommendation or missing signed evidence yields ``NOT_ASSESSABLE``
    instead of fabricating a wrong-side conclusion.
    """

    if not assessable:
        return "NOT_ASSESSABLE"
    if (
        recommended_side in {"LEFT", "RIGHT"}
        and initial_obstacle_side in {"LEFT", "RIGHT"}
        and initial_obstacle_side != recommended_side
    ):
        return "WRONG_INITIAL_BYPASS_SIDE"
    if lateral_sign_changes >= 2:
        return "LATERAL_SIGN_OSCILLATION"
    if np.isfinite(actor_motion_agreement) and actor_motion_agreement < 0.5:
        return "ACTOR_TO_MOTION_DIRECTION_MISMATCH"
    if lateral_valid and np.isfinite(radial_progress_mean) and radial_progress_mean <= 0.0:
        return "RADIAL_COLLAPSE_WITH_VALID_LATERAL"
    if not passed:
        return "CORRECT_INITIAL_SIDE_BUT_NO_EXIT"
    return "NOT_ASSESSABLE"


__all__ = [
    "DEFAULT_SIGN_DEADBAND_M",
    "DEFAULT_SIGN_DEADBAND_MPS",
    "LATERAL_AXIS_SOURCE",
    "LATERAL_SIGN_CONVENTION",
    "NOT_INDEPENDENTLY_VERIFIED",
    "SIGNED_BYPASS_OBSERVABILITY_SCHEMA_VERSION",
    "SignedFrame",
    "SignedLateralAxisNotWellDefined",
    "build_horizontal_goal_frame",
    "classify_signed_failure",
    "compare_identity_component",
    "compare_identity_components",
    "decompose_signed_vector",
    "frame_orthogonality",
    "passed_geometry_fields",
    "pipeline_signed_components",
    "same_side_run_length",
    "side_agreement",
    "sign_change_count",
    "signed_side",
    "sign_value",
    "trajectory_fingerprint",
    "update_side_run",
]
