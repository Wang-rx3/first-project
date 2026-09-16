"""Read-only feasibility test under the frozen Actor action parameterization.

This module is intentionally separate from the previous RL-equivalent oracle
audit.  It reuses that audit's four frozen roles and paired scenarios, then
adds a fifth role whose controller has exactly the same estimated-state inputs
as the RL actor and exactly the same final componentwise action set.  No
policy, reward, curriculum, dynamics, or success criterion is changed.
"""

from __future__ import annotations

import csv
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    _copy_calibration_files,
    _episode_metrics,
    _make_environment,
    _scale_seed,
    _write_episode_metrics,
    _write_scale_manifest,
)
from .pilot import _EpisodeTraceWriter, _seed_everything
from .single_uav_checkpoint_evaluation import _config_from_checkpoint
from .single_uav_rl_equivalent_oracle_audit import (
    BASELINE_RL,
    ESTIMATED_ORACLE,
    FINE_RL,
    POLICY_LABELS as BASE_POLICY_LABELS,
    POLICY_NAMES as BASE_POLICY_NAMES,
    _angle_deg,
    _base_step_row,
    _complete_geometry_flags,
    _distance_bin_summary,
    _extract_rl_visible_state,
    _maximum,
    _mean,
    _number,
    _percentile,
    _runner_geometry,
    _write_rows_csv,
)
from .single_uav_scripted_feasibility import (
    _canonical_hash,
    _paired_scenarios,
    _prepare_root,
)


ACTOR_CONSTRAINED_ORACLE = "actor_constrained_estimated_state_oracle"
POLICY_NAMES: tuple[str, ...] = (
    "truth_oracle",
    ESTIMATED_ORACLE,
    ACTOR_CONSTRAINED_ORACLE,
    BASELINE_RL,
    FINE_RL,
)
POLICY_LABELS = {
    **BASE_POLICY_LABELS,
    ACTOR_CONSTRAINED_ORACLE: "Actor-action-constrained estimated-state oracle",
}
OLD_POLICY_NAMES = tuple(BASE_POLICY_NAMES)

DISTANCE_BINS: tuple[tuple[str, str], ...] = (
    ("gt_8m", ">8 m"),
    ("8_5m", "8–5 m"),
    ("5_3m", "5–3 m"),
    ("3_1m", "3–1 m"),
    ("le_1m", "≤1 m"),
)


def actor_component_limit(max_delta_v_mps: float) -> float:
    """Return the exact per-axis bound used by ``MAPPOPolicy``."""

    return float(max_delta_v_mps / math.sqrt(3.0))


def actor_reachable_radial_norm(
    goal_direction: Iterable[float], max_delta_v_mps: float
) -> float:
    """Return the largest action norm along a unit goal direction.

    The actor action set is a componentwise box and the environment action set
    is a Euclidean norm ball.  For a unit direction ``u`` the intersection is:

    ``min(max_delta, component_limit / max(abs(u)))``.
    """

    direction = np.asarray(tuple(goal_direction), dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return float("nan")
    direction = direction / norm
    component = actor_component_limit(max_delta_v_mps)
    largest_component = float(np.max(np.abs(direction)))
    return float(min(float(max_delta_v_mps), component / largest_component))


def actor_action_transform(
    desired_delta_v_mps: Iterable[float], max_delta_v_mps: float
) -> dict[str, Any]:
    """Apply the frozen Actor's final action parameterization to a desired vector.

    A desired oracle action is first restricted to the Actor's componentwise
    box.  The inverse/forward tanh pair is included explicitly so the audit
    records the same latent, squash, and scale stages as the policy.  The
    returned final vector is the one sent to ``environment.step``.
    """

    desired = np.asarray(tuple(desired_delta_v_mps), dtype=float).reshape(3)
    if not np.isfinite(desired).all():
        raise ValueError("desired delta-v must be finite")
    max_delta = float(max_delta_v_mps)
    component = actor_component_limit(max_delta)
    component_limited = np.clip(desired, -component, component)
    if component <= 0.0:
        normalized = np.zeros(3, dtype=float)
    else:
        normalized = component_limited / component
    # A finite neural pre-tanh value cannot represent exactly +/-1.  This is
    # only a numerical epsilon; it does not change the closed action set.
    tanh_epsilon = 1.0e-12
    tanh_output = np.clip(normalized, -1.0 + tanh_epsilon, 1.0 - tanh_epsilon)
    pre_tanh = np.arctanh(tanh_output)
    reconstructed = np.tanh(pre_tanh) * component
    return {
        "desired_delta_v_mps": desired.tolist(),
        "actor_component_limited_delta_v_mps": component_limited.tolist(),
        "actor_pre_tanh_raw": pre_tanh.tolist(),
        "actor_tanh_output": tanh_output.tolist(),
        "actor_scaled_delta_v_mps": reconstructed.tolist(),
        "constrained_delta_v_mps": reconstructed.tolist(),
        "actor_component_limit_mps": component,
        "desired_action_norm_mps": float(np.linalg.norm(desired)),
        "constrained_action_norm_mps": float(np.linalg.norm(reconstructed)),
        "actor_constraint_delta_v_difference_norm_mps": float(
            np.linalg.norm(desired - reconstructed)
        ),
        "actor_constraint_affected": int(
            np.linalg.norm(desired - reconstructed) > 1.0e-9
        ),
        "actor_any_axis_saturated": int(
            bool(np.any(np.abs(desired) > component + 1.0e-9))
        ),
    }


def _projection_diagnostics(
    *,
    desired: np.ndarray,
    constrained: np.ndarray,
    visible: Mapping[str, Any],
    max_delta_v_mps: float,
) -> dict[str, Any]:
    goal_direction = np.asarray(visible["goal_direction"], dtype=float).reshape(3)
    velocity = np.asarray(visible["velocity_mps"], dtype=float).reshape(3)
    velocity_direction = velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) > 1.0e-12 else np.zeros(3)
    reachable = actor_reachable_radial_norm(goal_direction, max_delta_v_mps)
    desired_radial = float(np.dot(desired, goal_direction))
    constrained_radial = float(np.dot(constrained, goal_direction))
    desired_tangent = desired - desired_radial * goal_direction
    constrained_tangent = constrained - constrained_radial * goal_direction
    return {
        "actor_direction_reachable_radial_norm_mps": reachable,
        "desired_action_norm_gt_actor_direction_reachable": int(
            np.isfinite(reachable)
            and float(np.linalg.norm(desired)) > reachable + 1.0e-9
        ),
        "desired_radial_action_projection_mps": desired_radial,
        "constrained_radial_action_projection_mps": constrained_radial,
        "desired_braking_projection_mps": max(
            0.0, -float(np.dot(desired, velocity_direction))
        ),
        "constrained_braking_projection_mps": max(
            0.0, -float(np.dot(constrained, velocity_direction))
        ),
        "desired_tangential_action_mps": float(np.linalg.norm(desired_tangent)),
        "constrained_tangential_action_mps": float(np.linalg.norm(constrained_tangent)),
    }


class ActorConstrainedEstimatedStateOracleController:
    """Estimated-state oracle with the exact frozen Actor action set."""

    def __init__(self, environment: Any) -> None:
        from .single_uav_rl_equivalent_oracle_audit import EstimatedStateOracleController

        self._base = EstimatedStateOracleController(environment)
        self.environment = environment
        self.uav_id = self._base.uav_id
        self.max_delta_v_mps = self._base.max_delta_v_mps

    def action(
        self, observations: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_action, control = self._base.action(observations)
        visible = _extract_rl_visible_state(observations[self.uav_id])
        desired = np.asarray(control["raw_delta_v_mps"], dtype=float).reshape(3)
        transformed = actor_action_transform(desired, self.max_delta_v_mps)
        constrained = np.asarray(
            transformed["constrained_delta_v_mps"], dtype=float
        ).reshape(3)
        transformed.update(
            _projection_diagnostics(
                desired=desired,
                constrained=constrained,
                visible=visible,
                max_delta_v_mps=self.max_delta_v_mps,
            )
        )
        control = dict(control)
        control.update(transformed)
        control.update(
            {
                "legal_delta_v_mps": constrained.tolist(),
                "legal_delta_v_norm_mps": float(np.linalg.norm(constrained)),
                "state_source": "actor_visible_node_features_only",
                "truth_state_read_for_control": False,
                "action_parameterization": "normal_tanh_component_box",
            }
        )
        action = {
            self.uav_id: {
                "delta_v": constrained.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }
        return action, control


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _constraint_episode_summary(
    rows: Sequence[Mapping[str, Any]], writer_summary: Mapping[str, Any]
) -> dict[str, Any]:
    # The established feasibility summary is reused by temporarily using the
    # estimated-oracle label, then the role identity is replaced explicitly.
    from .single_uav_rl_equivalent_oracle_audit import _episode_summary

    legacy_rows = [dict(row, policy=ESTIMATED_ORACLE) for row in rows]
    result = _episode_summary(
        legacy_rows, policy=ESTIMATED_ORACLE, writer_summary=writer_summary
    )
    result["policy"] = ACTOR_CONSTRAINED_ORACLE
    result["policy_label"] = POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE]
    return result


def _aggregate_summaries(
    summaries: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    selected = [row for row in summaries if str(row.get("policy")) == policy]

    def count_field(name: str) -> int:
        return int(sum(int(row.get(name, 0) or 0) for row in selected))

    def mean_field(name: str) -> float:
        return _mean(row.get(name) for row in selected)

    def mean_steps(name: str) -> float:
        return _mean(row.get(name) for row in selected if row.get(name) not in (None, ""))

    count = len(selected)
    return {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "episodes": count,
        "entered_8m_episodes": count_field("entered_8m"),
        "entered_8m_rate": count_field("entered_8m") / count if count else 0.0,
        "entered_5m_episodes": count_field("entered_5m"),
        "entered_5m_rate": count_field("entered_5m") / count if count else 0.0,
        "entered_3m_episodes": count_field("entered_3m"),
        "entered_3m_rate": count_field("entered_3m") / count if count else 0.0,
        "entered_1m_episodes": count_field("entered_1m"),
        "entered_1m_rate": count_field("entered_1m") / count if count else 0.0,
        "stable_goal_episodes": count_field("stable_goal"),
        "stable_goal_rate": count_field("stable_goal") / count if count else 0.0,
        "minimum_goal_distance_m": min(
            (_number(row.get("minimum_goal_distance_m")) for row in selected),
            default=float("nan"),
        ),
        "mean_minimum_goal_distance_m": mean_field("minimum_goal_distance_m"),
        "mean_true_minimum_goal_distance_m": mean_field("minimum_true_goal_distance_m"),
        "mean_speed_mps": mean_field("mean_speed_mps"),
        "mean_speed_within_1m_mps": mean_field("mean_speed_within_1m_mps"),
        "mean_speed_at_8m_entry_mps": mean_field("mean_speed_at_8m_entry_mps"),
        "mean_speed_at_5m_entry_mps": mean_field("mean_speed_at_5m_entry_mps"),
        "mean_speed_at_3m_entry_mps": mean_field("mean_speed_at_3m_entry_mps"),
        "mean_speed_at_1m_entry_mps": mean_field("mean_speed_at_1m_entry_mps"),
        "mean_maximum_consecutive_goal_hold_steps": mean_field(
            "maximum_consecutive_goal_hold_steps"
        ),
        "maximum_consecutive_goal_hold_steps_max": int(
            round(
                _maximum(
                    (row.get("maximum_consecutive_goal_hold_steps") for row in selected),
                    0.0,
                )
            )
        ),
        "episodes_with_hold_steps_ge_1": count_field("hold_steps_ge_1"),
        "episodes_with_hold_steps_ge_1_rate": count_field("hold_steps_ge_1") / count if count else 0.0,
        "episodes_with_hold_steps_ge_3": count_field("hold_steps_ge_3"),
        "episodes_with_hold_steps_ge_3_rate": count_field("hold_steps_ge_3") / count if count else 0.0,
        "episodes_with_hold_steps_ge_5": count_field("hold_steps_ge_5"),
        "episodes_with_hold_steps_ge_5_rate": count_field("hold_steps_ge_5") / count if count else 0.0,
        "timeout_episodes": count_field("timeout"),
        "timeout_rate": count_field("timeout") / count if count else 0.0,
        "mean_steps_to_8m": mean_steps("steps_to_8m"),
        "mean_steps_to_5m": mean_steps("steps_to_5m"),
        "mean_steps_to_3m": mean_steps("steps_to_3m"),
        "mean_steps_to_1m": mean_steps("steps_to_1m"),
        "mean_action_norm_mps": mean_field("mean_action_norm_mps"),
        "mean_action_clipping_fraction": mean_field("action_clipping_fraction"),
        "mean_effective_braking_fraction": mean_field("effective_braking_fraction"),
        "mean_braking_action_fraction": mean_field("braking_action_fraction"),
        "maximum_braking_projection_mps": _maximum(
            (row.get("maximum_braking_projection_mps") for row in selected), 0.0
        ),
    }


def _constraint_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("policy")) == ACTOR_CONSTRAINED_ORACLE]
    return {
        "policy": ACTOR_CONSTRAINED_ORACLE,
        "policy_label": POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
        "rows": len(selected),
        "mean_desired_action_norm_mps": _mean(row.get("desired_action_norm_mps") for row in selected),
        "mean_constrained_action_norm_mps": _mean(row.get("constrained_action_norm_mps") for row in selected),
        "maximum_desired_action_norm_mps": _maximum(row.get("desired_action_norm_mps") for row in selected),
        "maximum_constrained_action_norm_mps": _maximum(row.get("constrained_action_norm_mps") for row in selected),
        "mean_constraint_delta_v_difference_norm_mps": _mean(
            row.get("actor_constraint_delta_v_difference_norm_mps") for row in selected
        ),
        "maximum_constraint_delta_v_difference_norm_mps": _maximum(
            row.get("actor_constraint_delta_v_difference_norm_mps") for row in selected
        ),
        "constraint_affected_fraction": _mean(row.get("actor_constraint_affected") for row in selected),
        "any_axis_saturation_fraction": _mean(row.get("actor_any_axis_saturated") for row in selected),
        "desired_norm_gt_actor_direction_reachable_fraction": _mean(
            row.get("desired_action_norm_gt_actor_direction_reachable") for row in selected
        ),
        "mean_actor_direction_reachable_radial_norm_mps": _mean(
            row.get("actor_direction_reachable_radial_norm_mps") for row in selected
        ),
        "minimum_actor_direction_reachable_radial_norm_mps": min(
            (_number(row.get("actor_direction_reachable_radial_norm_mps")) for row in selected),
            default=float("nan"),
        ),
        "maximum_actor_direction_reachable_radial_norm_mps": _maximum(
            row.get("actor_direction_reachable_radial_norm_mps") for row in selected
        ),
        "mean_desired_radial_action_projection_mps": _mean(
            row.get("desired_radial_action_projection_mps") for row in selected
        ),
        "mean_constrained_radial_action_projection_mps": _mean(
            row.get("constrained_radial_action_projection_mps") for row in selected
        ),
        "mean_desired_braking_projection_mps": _mean(
            row.get("desired_braking_projection_mps") for row in selected
        ),
        "mean_constrained_braking_projection_mps": _mean(
            row.get("constrained_braking_projection_mps") for row in selected
        ),
        "mean_desired_tangential_action_mps": _mean(
            row.get("desired_tangential_action_mps") for row in selected
        ),
        "mean_constrained_tangential_action_mps": _mean(
            row.get("constrained_tangential_action_mps") for row in selected
        ),
        "mean_actual_radial_velocity_mps": _mean(
            row.get("radial_velocity_toward_mps") for row in selected
        ),
        "mean_actual_speed_reduction_mps": _mean(
            row.get("actual_speed_reduction_mps") for row in selected
        ),
    }


def _constraint_bin_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, label in DISTANCE_BINS:
        selected = [
            row
            for row in rows
            if str(row.get("policy")) == ACTOR_CONSTRAINED_ORACLE
            and str(row.get("distance_bin")) == name
        ]
        output.append(
            {
                "policy": ACTOR_CONSTRAINED_ORACLE,
                "policy_label": POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
                "distance_bin": name,
                "distance_bin_label": label,
                "rows": len(selected),
                "mean_action_norm_mps": _mean(row.get("action_norm_mps") for row in selected),
                "action_norm_p50_mps": _percentile((row.get("action_norm_mps") for row in selected), 50),
                "action_norm_p90_mps": _percentile((row.get("action_norm_mps") for row in selected), 90),
                "action_norm_max_mps": _maximum(row.get("action_norm_mps") for row in selected),
                "mean_desired_action_norm_mps": _mean(row.get("desired_action_norm_mps") for row in selected),
                "desired_action_norm_p50_mps": _percentile((row.get("desired_action_norm_mps") for row in selected), 50),
                "desired_action_norm_p90_mps": _percentile((row.get("desired_action_norm_mps") for row in selected), 90),
                "desired_action_norm_max_mps": _maximum(row.get("desired_action_norm_mps") for row in selected),
                "mean_constrained_action_norm_mps": _mean(row.get("constrained_action_norm_mps") for row in selected),
                "constrained_action_norm_p50_mps": _percentile((row.get("constrained_action_norm_mps") for row in selected), 50),
                "constrained_action_norm_p90_mps": _percentile((row.get("constrained_action_norm_mps") for row in selected), 90),
                "constrained_action_norm_max_mps": _maximum(row.get("constrained_action_norm_mps") for row in selected),
                "mean_constraint_delta_v_difference_norm_mps": _mean(row.get("actor_constraint_delta_v_difference_norm_mps") for row in selected),
                "constraint_affected_fraction": _mean(row.get("actor_constraint_affected") for row in selected),
                "any_axis_saturation_fraction": _mean(row.get("actor_any_axis_saturated") for row in selected),
                "desired_norm_gt_actor_direction_reachable_fraction": _mean(row.get("desired_action_norm_gt_actor_direction_reachable") for row in selected),
                "mean_actor_direction_reachable_radial_norm_mps": _mean(row.get("actor_direction_reachable_radial_norm_mps") for row in selected),
                "mean_desired_radial_action_projection_mps": _mean(row.get("desired_radial_action_projection_mps") for row in selected),
                "mean_constrained_radial_action_projection_mps": _mean(row.get("constrained_radial_action_projection_mps") for row in selected),
                "mean_desired_braking_projection_mps": _mean(row.get("desired_braking_projection_mps") for row in selected),
                "mean_constrained_braking_projection_mps": _mean(row.get("constrained_braking_projection_mps") for row in selected),
                "mean_desired_tangential_action_mps": _mean(row.get("desired_tangential_action_mps") for row in selected),
                "mean_constrained_tangential_action_mps": _mean(row.get("constrained_tangential_action_mps") for row in selected),
                # Common names mirror the legacy four-policy distance table;
                # these refer to the *actually constrained* action.
                "mean_radial_action_projection_mps": _mean(
                    row.get("constrained_radial_action_projection_mps") for row in selected
                ),
                "mean_braking_projection_mps": _mean(
                    row.get("constrained_braking_projection_mps") for row in selected
                ),
                "mean_tangential_action_mps": _mean(
                    row.get("constrained_tangential_action_mps") for row in selected
                ),
                "mean_radial_velocity_mps": _mean(row.get("radial_velocity_toward_mps") for row in selected),
                "mean_actual_speed_reduction_mps": _mean(row.get("actual_speed_reduction_mps") for row in selected),
                "effective_braking_fraction": _mean(
                    row.get("effective_braking") for row in selected
                ),
                "mean_total_speed_mps": _mean(
                    row.get("total_speed_mps") for row in selected
                ),
                "mean_target_speed_mps": _mean(
                    row.get("target_speed_mps") for row in selected
                ),
                "action_clipping_fraction": _mean(row.get("action_clipped") for row in selected),
            }
        )
    return output


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _number(numerator, float("nan"))
    denominator_value = _number(denominator, float("nan"))
    if not np.isfinite(numerator_value) or not np.isfinite(denominator_value):
        return float("nan")
    if abs(denominator_value) <= 1.0e-12:
        return float("nan")
    return float(numerator_value / denominator_value)


def _action_amplitude_distance_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize action magnitude and directional components for all roles.

    The four cached roles and the newly evaluated role share the established
    step-level geometry columns.  Keeping this comparison in a separate table
    makes the ``RL / successful constrained-oracle`` magnitude ratio explicit
    without changing any legacy audit files.
    """

    output: list[dict[str, Any]] = []
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    for policy in POLICY_NAMES:
        for name, label in DISTANCE_BINS:
            selected = [
                row
                for row in rows
                if str(row.get("policy")) == policy
                and str(row.get("distance_bin")) == name
            ]
            record = {
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "distance_bin": name,
                "distance_bin_label": label,
                "rows": len(selected),
                "mean_action_norm_mps": _mean(row.get("action_norm_mps") for row in selected),
                "action_norm_p50_mps": _percentile(
                    (row.get("action_norm_mps") for row in selected), 50
                ),
                "action_norm_p90_mps": _percentile(
                    (row.get("action_norm_mps") for row in selected), 90
                ),
                "action_norm_max_mps": _maximum(
                    (row.get("action_norm_mps") for row in selected), 0.0
                ),
                "mean_radial_action_projection_mps": _mean(
                    row.get("action_projection_toward_goal_mps") for row in selected
                ),
                "mean_braking_projection_mps": _mean(
                    row.get("braking_projection_mps") for row in selected
                ),
                "mean_tangential_action_mps": _mean(
                    row.get("tangential_action_mps") for row in selected
                ),
                "mean_radial_velocity_mps": _mean(
                    row.get("radial_velocity_toward_mps") for row in selected
                ),
                "mean_total_speed_mps": _mean(
                    row.get("total_speed_mps") for row in selected
                ),
                "mean_target_speed_mps": _mean(
                    row.get("target_speed_mps") for row in selected
                ),
            }
            raw[(policy, name)] = record

    for policy in POLICY_NAMES:
        for name, _label in DISTANCE_BINS:
            record = dict(raw[(policy, name)])
            oracle = raw[(ACTOR_CONSTRAINED_ORACLE, name)]
            record.update(
                {
                    "mean_action_norm_ratio_vs_actor_constrained_oracle": _safe_ratio(
                        record["mean_action_norm_mps"], oracle["mean_action_norm_mps"]
                    ),
                    "action_norm_p50_ratio_vs_actor_constrained_oracle": _safe_ratio(
                        record["action_norm_p50_mps"], oracle["action_norm_p50_mps"]
                    ),
                    "action_norm_p90_ratio_vs_actor_constrained_oracle": _safe_ratio(
                        record["action_norm_p90_mps"], oracle["action_norm_p90_mps"]
                    ),
                }
            )
            output.append(record)
    return output


def _run_constrained_episode(
    *,
    config: Any,
    scale_root: Path,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(
        seed=seed, scenario_config=copy.deepcopy(dict(scenario))
    )
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / "test" / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split="test",
        seed=seed,
        phase="actor_constrained_oracle_feasibility",
        environment=environment,
        normalization_fitted=False,
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=ACTOR_CONSTRAINED_ORACLE,
    )
    controller = ActorConstrainedEstimatedStateOracleController(environment)
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        pre_state = environment.states["uav_00"].copy()
        action, control = controller.action(observations)
        control = dict(control)
        control["pre_velocity_mps"] = pre_state.velocity_mps.tolist()
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        commanded = np.asarray(action["uav_00"]["delta_v"], dtype=float).reshape(3)
        applied = np.asarray(execution.get("delta_v_applied", commanded), dtype=float).reshape(3)
        geometry = _runner_geometry(
            environment,
            goal=goal,
            pre_state=pre_state,
            post_state=environment.states["uav_00"],
            commanded=commanded,
            applied=applied,
        )
        estimated_before = _extract_rl_visible_state(observations["uav_00"])
        estimated_after = _extract_rl_visible_state(next_observations["uav_00"])
        pre_goal_delta = goal - np.asarray(pre_state.position_m, dtype=float)
        pre_goal_norm = float(np.linalg.norm(pre_goal_delta))
        pre_goal_direction = (
            pre_goal_delta / pre_goal_norm if pre_goal_norm > 1.0e-12 else np.zeros(3)
        )
        row = _base_step_row(
            # The shared row builder owns the legacy four-policy label map;
            # replace the identity immediately after construction so all
            # established geometry/diagnostic columns remain identical.
            policy=ESTIMATED_ORACLE,
            episode_index=episode_index,
            seed=seed,
            step_index=step,
            timestamp_s=(step + 1) * environment.config.motion.dt_s,
            scenario_hash=scenario_hash,
            geometry=geometry,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            control=control,
            execution=execution,
            next_info=next_info,
            reward=next_info.get("learning_reward_components", {}),
        )
        row.update(
            {
                "policy": ACTOR_CONSTRAINED_ORACLE,
                "policy_label": POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
            }
        )
        row = _complete_geometry_flags(row)
        row.update(
            {
                "controller_state_source": "actor_visible_node_features_only",
                "truth_state_read_for_control": False,
                "estimated_true_distance_error_before_m": float(
                    _number(estimated_before.get("distance_m"))
                    - float(geometry["true_goal_distance_before_m"])
                ),
                "estimated_true_direction_angular_error_deg": _angle_deg(
                    np.asarray(estimated_before["goal_direction"], dtype=float),
                    pre_goal_direction,
                ),
                "estimated_true_radial_velocity_error_mps": float(
                    _number(estimated_before.get("radial_velocity_mps"))
                    - float(geometry["true_radial_velocity_before_mps"])
                ),
                "action_parameterization": "normal_tanh_component_box",
                "actor_action_scale": "component_limit=max_delta_v/sqrt(3)",
                "actor_component_limit_mps": control["actor_component_limit_mps"],
                "desired_action_norm_mps": control["desired_action_norm_mps"],
                "constrained_action_norm_mps": control["constrained_action_norm_mps"],
                "actor_constraint_delta_v_difference_norm_mps": control[
                    "actor_constraint_delta_v_difference_norm_mps"
                ],
                "actor_constraint_affected": control["actor_constraint_affected"],
                "actor_any_axis_saturated": control["actor_any_axis_saturated"],
                "actor_direction_reachable_radial_norm_mps": control[
                    "actor_direction_reachable_radial_norm_mps"
                ],
                "desired_action_norm_gt_actor_direction_reachable": control[
                    "desired_action_norm_gt_actor_direction_reachable"
                ],
                "desired_radial_action_projection_mps": control[
                    "desired_radial_action_projection_mps"
                ],
                "constrained_radial_action_projection_mps": control[
                    "constrained_radial_action_projection_mps"
                ],
                "desired_braking_projection_mps": control[
                    "desired_braking_projection_mps"
                ],
                "constrained_braking_projection_mps": control[
                    "constrained_braking_projection_mps"
                ],
                "desired_tangential_action_mps": control[
                    "desired_tangential_action_mps"
                ],
                "constrained_tangential_action_mps": control[
                    "constrained_tangential_action_mps"
                ],
                "actor_pre_tanh_raw_json": json.dumps(control["actor_pre_tanh_raw"]),
                "actor_tanh_output_json": json.dumps(control["actor_tanh_output"]),
                "actor_scaled_delta_v_json": json.dumps(control["actor_scaled_delta_v_mps"]),
            }
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=action,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source=ACTOR_CONSTRAINED_ORACLE,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    summary = writer.finalize()
    summary.update(
        {
            "seed": int(seed),
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "curriculum_used": False,
            "policy_source": ACTOR_CONSTRAINED_ORACLE,
            "controller_state_source": "actor_visible_node_features_only",
            "truth_state_read_for_control": False,
            "paired_episode_index": int(episode_index),
            "paired_scenario_hash": scenario_hash,
        }
    )
    return summary, rows


def _write_constrained_role_outputs(
    *, config: Any, role_root: Path, summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    scale_root = role_root / "scale_1"
    _copy_calibration_files(scale_root, config)
    _write_scale_manifest(scale_root=scale_root, config=config, episodes=summaries)
    manifest_path = scale_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_id": "single_uav_actor_constrained_oracle_feasibility",
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "policy_source": ACTOR_CONSTRAINED_ORACLE,
            "policy_label": POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
            "feasibility_contract": {
                "normal_far_distance_test_only": True,
                "curriculum_used": False,
                "controller_uses_simulator_truth": False,
                "controller_state_source": "actor_visible_node_features_only",
                "action_interface": "environment.step(delta_v)",
                "action_parameterization": "normal_tanh_component_box",
                "direct_position_assignment": False,
                "direct_velocity_assignment": False,
                "stable_judgement_reused": True,
            },
            "read_only_contract": [
                "no neural policy training",
                "no optimizer or PPO update",
                "no rollout transition added",
                "no reward, curriculum, dynamics, or success criterion change",
            ],
        }
    )
    _write_json(manifest_path, _jsonable(manifest))

    standard_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    for writer_summary in summaries:
        episode_path = Path(str(writer_summary["path"]))
        standard = _episode_metrics(episode_path, writer_summary)
        standard.update(
            {
                "policy": ACTOR_CONSTRAINED_ORACLE,
                "policy_label": POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "curriculum_used": False,
            }
        )
        standard_rows.append(standard)
        episode_index = str(writer_summary.get("paired_episode_index"))
        episode_rows = [row for row in rows if str(row.get("episode_index")) == episode_index]
        feasibility_rows.append(_constraint_episode_summary(episode_rows, writer_summary))
    _write_episode_metrics(scale_root / "diagnostics" / "episode_metrics.csv", standard_rows)
    _write_rows_csv(scale_root / "diagnostics" / "feasibility_episode_metrics.csv", feasibility_rows)
    _write_rows_csv(scale_root / "diagnostics" / "step_metrics.csv", rows)
    _write_rows_csv(scale_root / "diagnostics" / "distance_bin_metrics.csv", _constraint_bin_summary(rows))
    validation = validate_episode_benchmark(scale_root)
    return {
        "role": ACTOR_CONSTRAINED_ORACLE,
        "policy_label": POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
        "episodes": len(summaries),
        "schema_validation": validation,
        "manifest": str(manifest_path.resolve()),
        "episode_metrics_path": str((scale_root / "diagnostics" / "episode_metrics.csv").resolve()),
        "feasibility_episode_metrics_path": str((scale_root / "diagnostics" / "feasibility_episode_metrics.csv").resolve()),
        "step_metrics_path": str((scale_root / "diagnostics" / "step_metrics.csv").resolve()),
        "distance_bin_metrics_path": str((scale_root / "diagnostics" / "distance_bin_metrics.csv").resolve()),
        "no_training_update_call": True,
        "no_transition_added": True,
        "episode_summaries": [dict(row) for row in summaries],
        "feasibility_episode_metrics": feasibility_rows,
    }


def _write_root_manifest(output_root: Path, reports: Mapping[str, Mapping[str, Any]]) -> None:
    _write_json(
        output_root / "manifest.json",
        _jsonable(
            {
                "dataset_id": "single_uav_actor_constrained_oracle_feasibility",
                "schema_version": EPISODE_SCHEMA_VERSION,
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "roles": {
                    role: {
                        "path": role,
                        "scale": "scale_1",
                        "manifest": f"{role}/scale_1/manifest.json",
                        "schema_ok": bool(report.get("schema_validation", {}).get("ok", False)),
                    }
                    for role, report in reports.items()
                },
            }
        ),
    )


def _copy_cached_roles(previous_root: Path, output_root: Path) -> None:
    for role in OLD_POLICY_NAMES:
        source = previous_root / role
        destination = output_root / role
        if not source.is_dir():
            raise FileNotFoundError(f"cached role is missing: {source}")
        shutil.copytree(source, destination)


def _action_set_audit(config: Any, environment: Any) -> dict[str, Any]:
    max_delta = float(environment.config.motion.max_acceleration_mps2 * environment.config.motion.dt_s)
    component = actor_component_limit(max_delta)
    return {
        "source_code_contract": {
            "policy_file": "src/learning/mappo.py",
            "distribution": "Normal(mean, exp(delta_logstd))",
            "squash": "tanh",
            "scale": "tanh_output * (max_delta_v / sqrt(3))",
            "environment_mapping": "environment.step(delta_v); motion speed update clips only the physical speed",
        },
        "actor_axis_range_mps": [-component, component],
        "actor_component_limit_mps": component,
        "actor_action_set": "{delta_v: abs(delta_v_i) <= max_delta_v/sqrt(3), i=1..3}",
        "actor_action_constraint_type": "componentwise_box",
        "actor_theoretical_max_action_norm_mps": float(math.sqrt(3.0) * component),
        "environment_action_set": "{delta_v: ||delta_v||_2 <= max_delta_v}",
        "environment_action_constraint_type": "euclidean_norm_ball",
        "environment_max_action_norm_mps": max_delta,
        "actor_and_environment_sets_equal": False,
        "actor_set_subset_of_environment_norm_ball": True,
        "radial_reachable_formula": "min(max_delta_v, (max_delta_v/sqrt(3))/max(abs(u_i))) for unit goal direction u",
        "max_acceleration_mps2": float(environment.config.motion.max_acceleration_mps2),
        "dt_s": float(environment.config.motion.dt_s),
        "max_steps": int(config.max_steps),
        "safety_projection": {
            "environment_safety_projection_enabled": bool(environment.config.safety_projection_enabled),
            "braking_safety_projection_enabled": bool(
                environment.config.reward.single_uav_braking_safety_projection_enabled
            ),
        },
    }


def _step_mean_by_policy(rows: Sequence[Mapping[str, Any]], policy: str, field: str) -> tuple[list[float], list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if str(row.get("policy")) != policy:
            continue
        value = _number(row.get(field))
        if not np.isfinite(value):
            continue
        grouped.setdefault(int(row.get("step_index", 0)), []).append(value)
    steps = sorted(grouped)
    return [float(step) * 0.2 for step in steps], [float(np.mean(grouped[step])) for step in steps]


def _plot_outputs(rows: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Mapping[str, Any]], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "truth_oracle": "#009E73",
        ESTIMATED_ORACLE: "#E69F00",
        ACTOR_CONSTRAINED_ORACLE: "#CC79A7",
        BASELINE_RL: "#0072B2",
        FINE_RL: "#D55E00",
    }
    paths: dict[str, str] = {}

    def save(fig: Any, name: str) -> None:
        path = output_dir / name
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[name.rsplit(".", 1)[0]] = str(path.resolve())

    def time_plot(policies: Sequence[str], field: str, title: str, ylabel: str, name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        for policy in policies:
            x, y = _step_mean_by_policy(rows, policy, field)
            if x:
                ax.plot(x, y, linewidth=2.0, color=colors[policy], label=POLICY_LABELS[policy])
        ax.set_title(title)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        save(fig, name)

    time_plot(
        POLICY_NAMES,
        "true_goal_distance_m",
        "Five-policy goal distance comparison",
        "True goal distance (m)",
        "goal_distance_vs_time.png",
    )
    time_plot(
        (ESTIMATED_ORACLE, ACTOR_CONSTRAINED_ORACLE),
        "total_speed_mps",
        "Estimated oracle speed vs Actor-constrained oracle",
        "Total speed (m/s)",
        "speed_estimated_vs_actor_constrained.png",
    )
    time_plot(
        POLICY_NAMES,
        "radial_velocity_toward_mps",
        "Five-policy radial velocity toward goal",
        "Radial velocity (m/s)",
        "radial_velocity_vs_time.png",
    )

    constrained = [row for row in rows if str(row.get("policy")) == ACTOR_CONSTRAINED_ORACLE]
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    x, desired = _step_mean_by_policy(constrained, ACTOR_CONSTRAINED_ORACLE, "desired_action_norm_mps")
    _x, actual = _step_mean_by_policy(constrained, ACTOR_CONSTRAINED_ORACLE, "constrained_action_norm_mps")
    if x:
        ax.plot(x, desired, color="#7B2CBF", linewidth=2, label="desired delta-v norm")
        ax.plot(_x, actual, color="#CC79A7", linewidth=2, label="Actor-constrained delta-v norm")
    ax.axhline(0.6, color="#111827", linestyle="--", linewidth=1, label="environment max norm")
    ax.set_title("Desired versus Actor-constrained delta-v norm")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("delta-v norm (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "desired_vs_constrained_delta_v_norm.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    x = [_number(row.get("true_goal_distance_before_m")) for row in constrained]
    before = [_number(row.get("desired_radial_action_projection_mps")) for row in constrained]
    after = [_number(row.get("constrained_radial_action_projection_mps")) for row in constrained]
    ax.scatter(x, before, s=12, alpha=0.25, color="#7B2CBF", label="before Actor constraint")
    ax.scatter(x, after, s=12, alpha=0.25, color="#CC79A7", label="after Actor constraint")
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_title("Radial action projection before versus after Actor constraint")
    ax.set_xlabel("True goal distance before action (m)")
    ax.set_ylabel("Radial action projection (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "radial_projection_before_after_constraint.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    x = [_number(row.get("true_goal_distance_before_m")) for row in constrained]
    reachable = [_number(row.get("actor_direction_reachable_radial_norm_mps")) for row in constrained]
    desired_norm = [_number(row.get("desired_action_norm_mps")) for row in constrained]
    actual_norm = [_number(row.get("constrained_action_norm_mps")) for row in constrained]
    ax.scatter(x, reachable, s=10, alpha=0.2, color="#009E73", label="Actor-direction reachable max")
    ax.scatter(x, desired_norm, s=10, alpha=0.15, color="#7B2CBF", label="oracle desired norm")
    ax.scatter(x, actual_norm, s=10, alpha=0.2, color="#CC79A7", label="constrained oracle norm")
    for policy in (BASELINE_RL, FINE_RL):
        xr = [_number(row.get("true_goal_distance_before_m")) for row in rows if str(row.get("policy")) == policy]
        yr = [_number(row.get("action_norm_mps")) for row in rows if str(row.get("policy")) == policy]
        ax.scatter(xr, yr, s=8, alpha=0.12, color=colors[policy], label=f"{POLICY_LABELS[policy]} actual")
    ax.axhline(0.6, color="#111827", linestyle="--", linewidth=1, label="environment max")
    ax.set_title("Goal-direction reachable delta-v versus goal distance")
    ax.set_xlabel("True goal distance before action (m)")
    ax.set_ylabel("delta-v norm / radial authority (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    save(fig, "goal_direction_reachable_delta_v_vs_distance.png")

    bin_rows = _constraint_bin_summary(constrained)
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    labels = [row["distance_bin_label"] for row in bin_rows]
    values = [100.0 * _number(row.get("any_axis_saturation_fraction"), 0.0) for row in bin_rows]
    ax.bar(labels, values, color="#CC79A7")
    ax.set_title("Actor per-axis saturation by distance segment")
    ax.set_xlabel("Distance segment")
    ax.set_ylabel("Saturated steps (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "per_axis_saturation_by_distance.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    x_labels = [POLICY_LABELS[p] for p in POLICY_NAMES]
    x_positions = np.arange(len(POLICY_NAMES))
    width = 0.24
    for offset, field, label, color in (
        (-width, "entered_1m_rate", "entered 1 m", "#0072B2"),
        (0.0, "entered_3m_rate", "entered 3 m", "#E69F00"),
        (width, "stable_goal_rate", "stable", "#009E73"),
    ):
        ax.bar(
            x_positions + offset,
            [100.0 * _number(aggregates[p].get(field), 0.0) for p in POLICY_NAMES],
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks(x_positions, x_labels, rotation=18, ha="right")
    ax.set_ylabel("Episodes (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Entry and stable-goal comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "stable_entry_comparison.png")
    return paths


def evaluate_actor_constrained_oracle_feasibility(
    baseline_checkpoint: str | Path,
    fine_tuned_checkpoint: str | Path,
    *,
    previous_audit_dir: str | Path,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the frozen five-policy Actor-action-constrained feasibility test."""

    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    previous_root = Path(previous_audit_dir)
    previous_logs = Path("logs/preliminary") / previous_root.name
    previous_summary_path = previous_root / "summary.json"
    if not previous_summary_path.is_file():
        raise FileNotFoundError(f"previous audit summary is missing: {previous_summary_path}")
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
    if not bool(previous_summary.get("overall_schema_ok", False)):
        raise ValueError("previous audit schema is not valid; refusing to reuse it")
    if int(previous_summary.get("paired_design", {}).get("episodes", 0)) != int(episodes):
        raise ValueError("episodes must match the cached paired audit exactly")
    for role in OLD_POLICY_NAMES:
        if role not in previous_summary.get("policies", {}):
            raise ValueError(f"previous audit does not contain role {role}")

    import torch

    baseline_path = Path(baseline_checkpoint)
    fine_path = Path(fine_tuned_checkpoint)
    baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    fine_payload = torch.load(fine_path, map_location="cpu", weights_only=False)
    baseline_config = _config_from_checkpoint(
        dict(baseline_payload), output_dir=Path(output_dir) / BASELINE_RL, progress=progress
    )
    fine_config = _config_from_checkpoint(
        dict(fine_payload), output_dir=Path(output_dir) / FINE_RL, progress=progress
    )
    if baseline_config.uav_count != 1 or fine_config.uav_count != 1:
        raise ValueError("Actor-constrained audit requires 1-UAV checkpoints")

    scenarios = _paired_scenarios(baseline_config, int(episodes))
    previous_manifest = json.loads(
        (previous_root / "paired_scenario_manifest.json").read_text(encoding="utf-8")
    )
    current_hashes = [str(item["scenario_hash"]) for item in scenarios]
    previous_hashes = [str(item["scenario_hash"]) for item in previous_manifest]
    if current_hashes != previous_hashes:
        raise ValueError("regenerated scenarios do not match the cached paired audit")

    output_root = Path(output_dir)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    _prepare_root(output_root, force=force)
    _prepare_root(logs_root, force=force)
    _prepare_root(plots_root, force=force)
    _copy_cached_roles(previous_root, output_root)

    scenario_manifest = [
        {
            "episode_index": int(item["episode_index"]),
            "seed": int(item["seed"]),
            "scenario_hash": str(item["scenario_hash"]),
            "start_distance_m": float(item["start_distance_m"]),
            "curriculum_used": False,
        }
        for item in scenarios
    ]
    _write_json(output_root / "paired_scenario_manifest.json", _jsonable(scenario_manifest))
    _write_json(logs_root / "paired_scenario_manifest.json", _jsonable(scenario_manifest))

    environment = _make_environment(
        baseline_config,
        _scale_seed(baseline_config.seed, 1, "test"),
        metric_scales=baseline_config.metric_scales,
    )
    action_set_audit = _action_set_audit(baseline_config, environment)
    action_set_audit["paired_scenario_hash_count"] = len(set(current_hashes))
    action_set_audit["paired_start_distance_m"] = [float(item["start_distance_m"]) for item in scenarios]

    role_root = output_root / ACTOR_CONSTRAINED_ORACLE
    scale_root = role_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    constrained_rows: list[dict[str, Any]] = []
    for item in scenarios:
        if progress:
            print(
                f"[actor-constrained-oracle] episode {item['episode_index']}/{len(scenarios)}",
                flush=True,
            )
        summary, rows = _run_constrained_episode(
            config=baseline_config,
            scale_root=scale_root,
            scenario=item["scenario"],
            scenario_hash=str(item["scenario_hash"]),
            episode_index=int(item["episode_index"]),
            seed=int(item["seed"]),
            serial=500000 + int(item["episode_index"]),
        )
        summaries.append(summary)
        constrained_rows.extend(rows)
    constrained_report = _write_constrained_role_outputs(
        config=baseline_config,
        role_root=role_root,
        summaries=summaries,
        rows=constrained_rows,
    )

    cached_rows = _read_csv(previous_logs / "step_metrics.csv")
    cached_episode_rows = _read_csv(previous_logs / "episode_metrics.csv")
    all_rows = cached_rows + constrained_rows
    combined_episode_rows = cached_episode_rows + constrained_report["feasibility_episode_metrics"]
    _write_rows_csv(logs_root / "combined_step_metrics.csv", all_rows)
    _write_rows_csv(logs_root / "combined_episode_metrics.csv", combined_episode_rows)
    _write_rows_csv(logs_root / "actor_constraint_step_metrics.csv", constrained_rows)
    _write_rows_csv(logs_root / "actor_constraint_episode_metrics.csv", constrained_report["feasibility_episode_metrics"])
    action_amplitude_distance_summary = _action_amplitude_distance_summary(all_rows)
    _write_rows_csv(
        logs_root / "action_amplitude_distance_summary.csv",
        action_amplitude_distance_summary,
    )

    constraint_aggregate = _constraint_aggregate(constrained_rows)
    constraint_bins = _constraint_bin_summary(constrained_rows)
    _write_rows_csv(logs_root / "actor_constraint_aggregate.csv", [constraint_aggregate])
    _write_rows_csv(logs_root / "actor_constraint_distance_bin_summary.csv", constraint_bins)
    combined_bins = list(previous_summary.get("distance_bin_summary", [])) + constraint_bins
    _write_rows_csv(logs_root / "distance_bin_summary.csv", combined_bins)

    aggregates = {
        role: dict(previous_summary["aggregate_metrics"][role]) for role in OLD_POLICY_NAMES
    }
    constrained_aggregates = _aggregate_summaries(
        constrained_report["feasibility_episode_metrics"], ACTOR_CONSTRAINED_ORACLE
    )
    aggregates[ACTOR_CONSTRAINED_ORACLE] = constrained_aggregates
    action_set_audit["normal_test_goal_direction_reachable_radial_summary_mps"] = {
        "mean": constraint_aggregate["mean_actor_direction_reachable_radial_norm_mps"],
        "min": constraint_aggregate["minimum_actor_direction_reachable_radial_norm_mps"],
        "max": constraint_aggregate["maximum_actor_direction_reachable_radial_norm_mps"],
    }
    _write_json(logs_root / "action_set_audit.json", _jsonable(action_set_audit))
    _write_json(output_root / "action_set_audit.json", _jsonable(action_set_audit))

    reports: dict[str, dict[str, Any]] = {}
    for role in OLD_POLICY_NAMES:
        scale = output_root / role / "scale_1"
        reports[role] = {
            "role": role,
            "policy_label": POLICY_LABELS[role],
            "episodes": int(previous_summary["policies"][role].get("episodes", episodes)),
            "schema_validation": validate_episode_benchmark(scale),
            "cached_from": str((previous_root / role / "scale_1").resolve()),
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
        }
    reports[ACTOR_CONSTRAINED_ORACLE] = constrained_report
    _write_root_manifest(output_root, reports)

    config_audit = dict(previous_summary.get("configuration_audit", {}))
    config_audit["actor_action_set_audit"] = action_set_audit
    config_audit["cached_four_policy_audit_reused"] = True
    config_audit["paired_scenario_audit"] = {
        "episodes": len(scenarios),
        "same_scenario_passed_to_all_policies": True,
        "same_seed_indices": True,
        "all_curriculum_used_flags_false": True,
        "cached_hashes_equal_regenerated_hashes": current_hashes == previous_hashes,
        "unique_scenario_hash_count": len(set(current_hashes)),
    }
    _write_json(logs_root / "configuration_audit.json", _jsonable(config_audit))
    _write_json(output_root / "configuration_audit.json", _jsonable(config_audit))

    plot_paths = _plot_outputs(all_rows, aggregates, plots_root)
    schema_validation = {
        role: report.get("schema_validation", {}) for role, report in reports.items()
    }
    overall_schema_ok = all(bool(report.get("ok", False)) for report in schema_validation.values())
    summary = {
        "dataset_id": "single_uav_actor_constrained_oracle_feasibility",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "paired_design": {
            "unit": "normal far-distance test episode",
            "episodes": len(scenarios),
            "same_scenario_passed_to_all_policies": True,
            "same_seed_indices": True,
            "curriculum_used": False,
            "cached_four_policy_audit_reused": True,
            "scenario_hashes_recorded": True,
        },
        "configuration_audit": config_audit,
        "policies": reports,
        "aggregate_metrics": aggregates,
        "actor_constraint_aggregate": constraint_aggregate,
        "distance_bin_summary": combined_bins,
        "actor_constraint_distance_bin_summary": constraint_bins,
        "action_amplitude_distance_summary": action_amplitude_distance_summary,
        "action_set_audit": action_set_audit,
        "read_only_integrity": {
            "cached_checkpoint_files_unchanged": True,
            "cached_policy_states_unchanged": True,
            "no_training_update_call": True,
            "no_transition_added": True,
        },
        "schema_validation": schema_validation,
        "overall_schema_ok": overall_schema_ok,
        "plots": plot_paths,
        "outputs": {
            "data": str(output_root.resolve()),
            "logs": str(logs_root.resolve()),
            "plots": str(plots_root.resolve()),
        },
        "claim_boundary": "read-only action-set feasibility diagnostic; no RL training result",
    }
    _write_json(output_root / "summary.json", _jsonable(summary))
    _write_json(logs_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "ACTOR_CONSTRAINED_ORACLE",
    "POLICY_NAMES",
    "POLICY_LABELS",
    "actor_component_limit",
    "actor_reachable_radial_norm",
    "actor_action_transform",
    "ActorConstrainedEstimatedStateOracleController",
    "evaluate_actor_constrained_oracle_feasibility",
]
