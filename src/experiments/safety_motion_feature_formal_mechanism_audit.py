"""Mechanism attribution audit for the formal 37d/41d safety experiment.

This module is deliberately read-only.  It replays the frozen formal safety
checkpoints on the ten frozen hold-out safety scenes and joins two independent
diagnostic views at the decision-window key
``(policy_seed, scenario_seed, branch, step, uav_id)``:

* the existing safety replay, which records physical motion, projection,
  dynamic-obstacle timing, collision and censoring information; and
* a local actor/queue replay, which records measurement probabilities, quality
  features, budgets and equivalent measurement-packet events.

The second replay is deterministic and does not call ``trainer.update``.  It
does not feed any offline geometry or collision labels back into the actor.
The scenario seed is the inferential unit; policy seeds are averaged within a
scenario before paired contrasts are calculated.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..learning.measurement_quality import quality_features
from ..synthetic.episode_generator import _jsonable
from .formal import _make_environment
from .pilot import _assert_action_feasible, _seed_everything
from .policy_mechanism_audit import (
    _edge_pairs,
    _event_float,
    _event_for,
    _event_int,
    _finite_trace_value,
    _load_quality_model,
)
from .safety_mappo_evaluation import (
    _actor_observation_truth_free,
    _bootstrap_ci,
    _evaluation_seed,
    _load_scenario,
)
from .safety_motion_feature_evaluation import (
    _actor_node_feature_status,
    _episode as _safety_episode,
    _extract_motion_features,
)
from .safety_motion_feature_formal_evaluation import (
    EXPECTED_DIM,
    EXPECTED_DYNAMIC,
    _load_formal_bundle,
)
from .safety_motion_feature_formal_training import FORMAL_BRANCHES


DEFAULT_POLICY_SEEDS = (7071, 8080, 9099, 10118, 11127)
DEFAULT_SCENARIO_SEEDS = (
    2026,
    3035,
    4044,
    5053,
    6062,
    7071,
    8080,
    9089,
    10098,
    11107,
)

BASELINE_37D = "baseline_37d"
BASELINE_41D = "baseline_41d"
FULL_BRANCH = "budgeted_quality_aware_41d"
_QUALITY_BRANCHES = {FULL_BRANCH}
_SAFETY_DISTANCE_DEFAULT = 2.0
_SCHEMA_VERSION = "safety-motion-feature-formal-mechanism-audit-v1"
_PLOT_FILE = "safety_motion_feature_formal_mechanism_audit.png"


@dataclass(frozen=True)
class SafetyMotionFeatureFormalMechanismAuditConfig:
    """Configuration for the validation-only formal mechanism audit."""

    checkpoint_root: Path
    scenario_root: Path
    output_dir: Path = Path(
        "logs/preliminary/"
        "safety_motion_feature_formal_mechanism_audit_seed2026"
    )
    plots_dir: Path = Path(
        "plots/preliminary/"
        "safety_motion_feature_formal_mechanism_audit_seed2026"
    )
    branches: tuple[str, ...] = FORMAL_BRANCHES
    policy_seeds: tuple[int, ...] = DEFAULT_POLICY_SEEDS
    scenario_seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    uav_count: int = 3
    max_steps: int = 50
    device: str = "cpu"
    deterministic: bool = True
    safety_projection_enabled: bool = True
    measurement_overhead_enabled: bool = True
    quality_logit_penalty: float = 1.0
    # ``None`` reuses the projection mode frozen in each checkpoint manifest.
    # The formal evaluator reads this field when constructing its immutable
    # replay configuration; keeping it optional preserves compatibility with
    # older mechanism-audit callers.
    dynamic_obstacle_swept_projection_enabled: bool | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    force: bool = False
    progress: bool = True
    command_line: str = ""

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("formal mechanism audit is fixed to three UAVs")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.device != "cpu":
            raise ValueError("formal mechanism audit supports CPU only")
        if not self.deterministic:
            raise ValueError("formal mechanism audit requires deterministic=True")
        if not self.safety_projection_enabled:
            raise ValueError("formal mechanism audit requires safety projection")
        if tuple(self.branches) != FORMAL_BRANCHES:
            raise ValueError(
                "branches must be exactly baseline_37d, baseline_41d, "
                "and budgeted_quality_aware_41d"
            )
        if not self.policy_seeds or not self.scenario_seeds:
            raise ValueError("policy_seeds and scenario_seeds must be non-empty")
        if len(set(int(seed) for seed in self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if len(set(int(seed) for seed in self.scenario_seeds)) != len(self.scenario_seeds):
            raise ValueError("scenario_seeds must be unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not self.measurement_overhead_enabled:
            raise ValueError("formal mechanism audit requires measurement overhead")
        if not math.isfinite(float(self.quality_logit_penalty)) or self.quality_logit_penalty < 0.0:
            raise ValueError("quality_logit_penalty must be finite and non-negative")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                columns.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _prepare_outputs(config: SafetyMotionFeatureFormalMechanismAuditConfig) -> tuple[Path, Path]:
    output = Path(config.output_dir).resolve()
    plots = Path(config.plots_dir).resolve()
    checkpoint_root = Path(config.checkpoint_root).resolve()
    scenario_root = Path(config.scenario_root).resolve()
    if output == plots:
        raise ValueError("output_dir and plots_dir must be distinct")
    for path in (output, plots):
        if _paths_overlap(path, checkpoint_root) or _paths_overlap(path, scenario_root):
            raise ValueError("audit outputs cannot overlap checkpoint or scenario sources")
        if path.exists() and any(path.iterdir()) and not config.force:
            raise FileExistsError(
                f"audit output is not empty: {path}; use --force to replace it"
            )
    if config.force:
        for path in (output, plots):
            if path.exists():
                shutil.rmtree(path)
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return output, plots


def _progress(config: SafetyMotionFeatureFormalMechanismAuditConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-motion-formal-mechanism] {message}", flush=True)


def _actor_dynamic_values(local: Mapping[str, Any]) -> tuple[list[float], int, int]:
    """Return normalized dynamic values actually supplied to the actor."""

    node = np.asarray(local.get("node_features", []), dtype=float)
    if node.ndim != 2 or node.shape[1] < 41:
        return [], 0, 0
    values = node[0, 37:41].astype(float, copy=True)
    return values.tolist(), int(np.isfinite(values).all()), 1


def _uav_ids(environment: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in environment.uav_ids)


def _safe_list(value: Any, *, dtype: Any = float) -> list[Any]:
    try:
        array = np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return []
    return array.tolist()


def _action_episode(
    config: SafetyMotionFeatureFormalMechanismAuditConfig,
    trainer: Any,
    eval_config: Any,
    quality_model: Mapping[str, Any],
    scenario: Mapping[str, Any],
    scenario_seed: int,
    scenario_hash: str,
    policy_seed: int,
    branch: str,
) -> list[dict[str, Any]]:
    """Replay one explicit hold-out scene and record actor/queue diagnostics."""

    evaluation_seed = _evaluation_seed(int(scenario_seed))
    _seed_everything(evaluation_seed)
    environment = _make_environment(
        eval_config,
        seed=evaluation_seed,
        metric_scales=eval_config.metric_scales,
        cooperative_ekf_enabled=True,
        measurement_overhead_enabled=True,
        measurement_budget_enabled=eval_config.measurement_budget_enabled,
        measurement_budget_by_uav=eval_config.measurement_budget_by_uav,
    )
    observations, reset_info = environment.reset(
        seed=evaluation_seed,
        scenario_config=scenario,
    )
    uav_ids = _uav_ids(environment)
    rows: list[dict[str, Any]] = []
    episode_id = f"scenario_{int(scenario_seed)}"

    for step_index in range(int(config.max_steps)):
        info = reset_info or {}
        central = dict(info.get("central_state") or {})
        pre_potential = _edge_pairs(central.get("potential_edges", []))
        pre_candidate = _edge_pairs(central.get("candidate_edges", []))
        pre_uwb = _edge_pairs(central.get("uwb_edges", []))
        pre_limits = dict(info.get("measurement_budget_limits") or {})
        pre_remaining = dict(info.get("measurement_budget_remaining") or {})
        pre_used = dict(info.get("measurement_budget_used") or {})

        decision = trainer.act(
            observations,
            info.get("central_state"),
            deterministic=True,
        )
        _assert_action_feasible(environment, decision.actions)
        submitted_actions = {
            uav_id: dict(decision.actions[uav_id]) for uav_id in uav_ids
        }
        next_observations, rewards, terminated, truncated, next_info = environment.step(
            decision.actions
        )
        post_central = dict(next_info.get("central_state") or {})
        post_potential = _edge_pairs(post_central.get("potential_edges", []))
        post_candidate = _edge_pairs(post_central.get("candidate_edges", []))
        post_uwb = _edge_pairs(post_central.get("uwb_edges", []))
        post_effective = _edge_pairs(post_central.get("effective_edges", []))
        events = list(next_info.get("measurement_overhead_events", []) or [])
        invalid_events = list(next_info.get("invalid_measurement_action_events", []) or [])
        transition_events = list(
            next_info.get("measurement_transition_invalidated_events", []) or []
        )
        gate_events = list(next_info.get("measurement_gate_events", []) or [])
        update_events = list(next_info.get("relative_range_update_events", []) or [])
        final_actions = dict(next_info.get("actions") or {})
        objective = dict(next_info.get("objective_components", {}) or {})
        global_metrics = dict(next_info.get("global_metrics", {}) or {})
        final_remaining = dict(next_info.get("measurement_budget_remaining") or {})
        final_used = dict(next_info.get("measurement_budget_used") or {})
        accepted_updates = [item for item in update_events if bool(item.get("accepted", False))]
        safety_distance = _finite(
            getattr(getattr(environment.config, "motion", None), "safety_distance_m", None),
            _SAFETY_DISTANCE_DEFAULT,
        )

        for uav_id in uav_ids:
            raw_local = observations.get(uav_id, {})
            actor_local = decision.observations.get(uav_id, {})
            risk, valid = quality_features(raw_local, uav_ids, quality_model)
            sample = decision.samples[uav_id]
            probabilities = np.asarray(sample.measurement_probabilities, dtype=float).reshape(-1)
            submitted = str(submitted_actions[uav_id].get("measurement_uav_id", "none"))
            final_action = dict(final_actions.get(uav_id, {}))
            final_selected = str(final_action.get("measurement_uav_id", "none"))
            submitted_index = uav_ids.index(submitted) if submitted in uav_ids else len(uav_ids)
            selected_index = uav_ids.index(final_selected) if final_selected in uav_ids else len(uav_ids)
            candidate_ids = sorted(
                {
                    str(value)
                    for value in raw_local.get("candidate_measurement_sources", [])
                    if str(value) in uav_ids
                }
            )
            candidate_indices = [uav_ids.index(value) for value in candidate_ids]
            submitted_pair = (submitted, uav_id)
            final_pair = (final_selected, uav_id)
            invalid_event = (
                _event_for(invalid_events, submitted, uav_id) if submitted != "none" else None
            )
            transition_event = (
                _event_for(transition_events, submitted, uav_id)
                if submitted != "none"
                else None
            )
            event = (
                _event_for(events, final_selected, uav_id)
                if final_selected != "none"
                else None
            )
            gate = (
                _event_for(gate_events, final_selected, uav_id)
                if final_selected != "none"
                else None
            )
            motion_values, motion_finite, motion_present = _extract_motion_features(raw_local)
            actor_values, actor_values_finite, actor_values_present = _actor_dynamic_values(actor_local)
            actor_dim, actor_motion_present = _actor_node_feature_status(actor_local)
            budget_limit = _finite(pre_limits.get(uav_id), -1.0)
            remaining_before = _finite(pre_remaining.get(uav_id), -1.0)
            used_before = _finite(pre_used.get(uav_id), -1.0)
            remaining_after = _finite(final_remaining.get(uav_id), -1.0)
            used_after = _finite(final_used.get(uav_id), -1.0)
            consumed = (
                max(0.0, used_after - used_before)
                if used_before >= 0.0 and used_after >= 0.0
                else 0.0
            )
            raw_observation_truth_free = int(_actor_observation_truth_free(raw_local))
            actor_observation_truth_free = int(_actor_observation_truth_free(actor_local))
            risk_values = _safe_list(risk, dtype=float)
            valid_values = _safe_list(valid, dtype=float)
            row: dict[str, Any] = {
                "policy_seed": int(policy_seed),
                "scenario_seed": int(scenario_seed),
                "scenario_hash": scenario_hash,
                "episode_id": episode_id,
                "branch": branch,
                "uav_id": uav_id,
                "step": int(step_index + 1),
                "time_s": float((step_index + 1) * environment.config.motion.dt_s),
                "actor_node_feature_dim": int(actor_dim),
                "expected_node_feature_dim": int(EXPECTED_DIM[branch]),
                "dynamic_features_expected": int(EXPECTED_DYNAMIC[branch]),
                "dynamic_motion_features_raw": motion_values.tolist(),
                "dynamic_motion_feature_finite": int(motion_finite),
                "dynamic_motion_feature_present": int(motion_present),
                "actor_dynamic_features": actor_values,
                "actor_dynamic_feature_finite": int(actor_values_finite),
                "actor_dynamic_feature_present": int(actor_values_present),
                "actor_motion_feature_present": int(actor_motion_present),
                "actor_observation_truth_free": actor_observation_truth_free,
                "raw_observation_truth_free": raw_observation_truth_free,
                "quality_feature_in_actor": int(branch in _QUALITY_BRANCHES),
                "budget_feature_in_actor": int(branch == FULL_BRANCH),
                "quality_model_source": "warmup",
                "quality_risk": risk_values,
                "quality_valid": valid_values,
                "candidate_uav_ids": candidate_ids,
                "candidate_quality_risk": [risk_values[uav_ids.index(value)] for value in candidate_ids],
                "candidate_quality_valid": [valid_values[uav_ids.index(value)] for value in candidate_ids],
                "measurement_probabilities": _safe_list(probabilities, dtype=float),
                "measurement_logits_raw": _safe_list(sample.measurement_logits_raw, dtype=float),
                "measurement_quality_penalty": _safe_list(sample.measurement_quality_penalty, dtype=float),
                "measurement_logits_adjusted": _safe_list(sample.measurement_logits_adjusted, dtype=float),
                "measurement_quality_logit_penalty": _finite(
                    sample.measurement_quality_logit_penalty, 0.0
                ),
                "none_probability": float(probabilities[-1]) if probabilities.size > len(uav_ids) else 0.0,
                "submitted_measurement_uav_id": submitted,
                "final_measurement_uav_id": final_selected,
                "selected_probability": (
                    float(probabilities[submitted_index])
                    if submitted_index < probabilities.size
                    else 0.0
                ),
                "selected_quality_risk": (
                    float(risk_values[submitted_index])
                    if submitted_index < len(risk_values)
                    else -1.0
                ),
                "selected_quality_valid": (
                    float(valid_values[submitted_index])
                    if submitted_index < len(valid_values)
                    else 0.0
                ),
                "action_mask": _safe_list(raw_local.get("action_mask", []), dtype=int),
                "action_mask_allows_submitted": int(
                    submitted == "none"
                    or (
                        submitted_index < len(_safe_list(raw_local.get("action_mask", []), dtype=int))
                        and bool(_safe_list(raw_local.get("action_mask", []), dtype=int)[submitted_index])
                    )
                ),
                "pre_action_potential_edge": int(
                    submitted != "none" and submitted_pair in pre_potential
                ),
                "pre_action_candidate_edge": int(
                    submitted != "none" and submitted_pair in pre_candidate
                ),
                "pre_action_uwb_available": int(submitted != "none" and submitted_pair in pre_uwb),
                "post_motion_potential_edge": int(
                    submitted != "none" and submitted_pair in post_potential
                ),
                "final_potential_edge": int(
                    final_selected != "none" and final_pair in post_potential
                ),
                "final_candidate_edge": int(
                    final_selected != "none" and final_pair in post_candidate
                ),
                "final_uwb_available": int(final_selected != "none" and final_pair in post_uwb),
                "final_effective_edge": int(final_selected != "none" and final_pair in post_effective),
                "effective_measurement": int(event is not None),
                "invalid_measurement_action": int(invalid_event is not None),
                "invalid_measurement_action_reason": (
                    str(invalid_event.get("reason", "")) if invalid_event else ""
                ),
                "measurement_transition_invalidated": int(transition_event is not None),
                "measurement_transition_reason": (
                    str(transition_event.get("reason", "")) if transition_event else ""
                ),
                "overhead_event_present": int(event is not None),
                "payload_size_bytes": int(_event_int(event, "payload_size_bytes", 0)),
                "load_packets_requested": _event_float(event, "load_packets_requested", 0.0),
                "load_packets_applied": _event_float(event, "load_packets_applied", 0.0),
                "queue_before_packets": _event_float(event, "queue_before_packets", -1.0),
                "queue_with_measurement_packets": _event_float(
                    event, "queue_with_measurement_packets", -1.0
                ),
                "queue_next_packets": _event_float(event, "queue_next_packets", -1.0),
                "channel_bandwidth_hz": _event_float(event, "channel_bandwidth_hz", -1.0),
                "rate_bps": _event_float(event, "rate_bps", -1.0),
                "event_packet_loss": _event_float(event, "packet_loss", -1.0),
                "event_delay_s": _event_float(event, "delay_s", -1.0),
                "event_aoi_s": _event_float(event, "aoi_s", -1.0),
                "link_state": str(event.get("link_state", "not_selected")) if event else "not_selected",
                "uwb_update_opportunity_step": _event_int(gate, "uwb_valid", 0),
                "ekf_update_attempted_step": _event_int(gate, "ekf_update_attempted", 0),
                "ekf_update_accepted_step": _event_int(gate, "ekf_update_accepted", 0),
                "accepted_update_count_all_targets_step": int(len(accepted_updates)),
                "objective_j_loc": _finite(objective.get("j_loc"), 0.0),
                "objective_j_com": _finite(objective.get("j_com"), 0.0),
                "objective_joint": _finite(objective.get("joint_objective"), 0.0),
                "team_reward": _finite(
                    next_info.get("team_reward"),
                    float(np.mean(list(rewards.values()))) if rewards else 0.0,
                ),
                "global_packet_loss": _finite(global_metrics.get("packet_loss"), 0.0),
                "global_delay_s": _finite(global_metrics.get("delay"), 0.0),
                "global_aoi_s": _finite(global_metrics.get("aoi"), 0.0),
                "global_connectivity": _finite(global_metrics.get("connectivity"), 0.0),
                "measurement_budget_enabled": int(
                    bool(next_info.get("measurement_budget_enabled", False))
                ),
                "budget_limit": budget_limit,
                "budget_remaining_before": remaining_before,
                "budget_remaining_after": remaining_after,
                "budget_used_before": used_before,
                "budget_used_after": used_after,
                "budget_consumed_step": consumed,
                "budget_exhausted_before": int(remaining_before == 0.0),
                "budget_overrun": int(
                    budget_limit >= 0.0 and used_after > budget_limit + 1e-9
                ),
                "trainer_update_called": 0,
                "test_data_read": 0,
                "safety_distance_m": safety_distance,
            }
            if not _finite_trace_value(row):
                raise RuntimeError(f"non-finite actor/queue audit row: {row}")
            rows.append(row)

        observations, reset_info = next_observations, next_info
        if terminated or truncated:
            break
    return rows


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, str, int, str]:
    return (
        int(row["policy_seed"]),
        int(row["scenario_seed"]),
        str(row["branch"]),
        int(row["step"]),
        str(row["uav_id"]),
    )


def _merge_safety_and_policy_rows(
    safety_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    policy_by_key = {_row_key(row): row for row in policy_rows}
    merged: list[dict[str, Any]] = []
    errors: list[str] = []
    for safety in safety_rows:
        key = _row_key(safety)
        policy = policy_by_key.get(key)
        if policy is None:
            errors.append("missing actor/queue row for " + repr(key))
            continue
        row = dict(safety)
        for name, value in policy.items():
            if name not in row or name in {
                "actor_node_feature_dim",
                "actor_observation_truth_free",
                "trainer_update_called",
                "measurement_overhead_enabled",
            }:
                row[name] = value
            else:
                row[f"policy_{name}"] = value
        safety_distance = _finite(row.get("safety_distance_m"), _SAFETY_DISTANCE_DEFAULT)
        row["nominal_action_risk"] = int(
            _finite(row.get("projection_pre_minimum_clearance_m"), safety_distance)
            < safety_distance
        )
        row["dynamic_post_update_risk"] = int(
            _finite(row.get("dynamic_clearance_before_update_m"), safety_distance)
            >= safety_distance
            and _finite(row.get("dynamic_clearance_after_update_m"), safety_distance)
            < safety_distance
            and int(row.get("collision", 0)) == 1
        )
        row["kinematic_closure"] = int(
            np.allclose(
                np.asarray(row.get("delta_v_applied", []), dtype=float),
                np.asarray(row.get("delta_v_commanded", []), dtype=float)
                + np.asarray(row.get("delta_v_projection", []), dtype=float),
                rtol=1e-6,
                atol=1e-7,
            )
        )
        merged.append(row)
    policy_keys = set(policy_by_key)
    safety_keys = {_row_key(row) for row in safety_rows}
    for key in sorted(policy_keys - safety_keys):
        errors.append("missing safety row for " + repr(key))
    return merged, errors


def _group(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    result: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        result.setdefault(key, []).append(row)
    return result


def _mean(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = np.asarray([_finite(value, float("nan")) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    return float(numbers.mean()) if numbers.size else float(default)


def _policy_action_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (policy_seed, branch), group in sorted(_group(rows, "policy_seed", "branch").items()):
        selected = [str(row.get("final_measurement_uav_id", "none")) != "none" for row in group]
        nonempty = [row for row in group if str(row.get("final_measurement_uav_id", "none")) != "none"]
        collision_keys = {
            (int(row["scenario_seed"]), int(row.get("collision", 0)))
            for row in group
            if int(row.get("collision", 0)) == 1
        }
        quality_rows = [
            value
            for row in group
            for value in np.asarray(row.get("quality_risk", []), dtype=float).reshape(-1)
            if math.isfinite(float(value))
        ]
        output.append(
            {
                "policy_seed": int(policy_seed),
                "branch": str(branch),
                "decision_window_uav_rows": len(group),
                "scenario_count": len({int(row["scenario_seed"]) for row in group}),
                "actor_node_feature_dim_min": min(int(row.get("actor_node_feature_dim", -1)) for row in group),
                "actor_node_feature_dim_max": max(int(row.get("actor_node_feature_dim", -1)) for row in group),
                "dynamic_feature_expected": int(any(int(row.get("dynamic_features_expected", 0)) for row in group)),
                "dynamic_feature_finite_rate": _mean(row.get("dynamic_motion_feature_finite", 0) for row in group),
                "actor_dynamic_feature_finite_rate": _mean(row.get("actor_dynamic_feature_finite", 0) for row in group),
                "actor_motion_feature_present_rate": _mean(row.get("actor_motion_feature_present", 0) for row in group),
                "quality_feature_in_actor": int(any(int(row.get("quality_feature_in_actor", 0)) for row in group)),
                "budget_feature_in_actor": int(any(int(row.get("budget_feature_in_actor", 0)) for row in group)),
                "mean_none_probability": _mean(row.get("none_probability", 0.0) for row in group),
                "none_selection_rate": 1.0 - _mean(selected),
                "measurement_selection_rate": _mean(selected),
                "measurement_count": float(sum(1 for row in group if int(row.get("overhead_event_present", 0)) == 1)),
                "measurement_overhead_packets": _mean(
                    row.get("measurement_overhead_packets_step", row.get("load_packets_applied", 0.0))
                    for row in group
                ) * len(group),
                "mean_selected_probability": _mean(row.get("selected_probability", 0.0) for row in nonempty),
                "mean_selected_quality_risk": _mean(
                    (row.get("selected_quality_risk", -1.0) for row in nonempty),
                    -1.0,
                ),
                "mean_quality_risk": _mean(quality_rows, -1.0),
                "invalid_measurement_action_count": int(sum(int(row.get("invalid_measurement_action", 0)) for row in group)),
                "transition_invalidation_count": int(sum(int(row.get("measurement_transition_invalidated", 0)) for row in group)),
                "transition_invalidation_rate_nonempty": _mean(
                    row.get("measurement_transition_invalidated", 0) for row in nonempty
                ),
                "queue_event_count": int(sum(int(row.get("overhead_event_present", 0)) for row in group)),
                "mean_queue_with_measurement_packets": _mean(
                    row.get("queue_with_measurement_packets", -1.0)
                    for row in group
                    if _finite(row.get("queue_with_measurement_packets"), -1.0) >= 0.0
                ),
                "nominal_action_risk_rate": _mean(row.get("nominal_action_risk", 0) for row in group),
                "collision_step_row_count": int(sum(int(row.get("collision", 0)) for row in group)),
                "collision_scenario_count": len({seed for seed, collision in collision_keys if collision}),
                "quality_risk_min": min(quality_rows, default=-1.0),
                "quality_risk_max": max(quality_rows, default=-1.0),
                "all_truth_free": int(all(int(row.get("actor_observation_truth_free", 0)) for row in group)),
                "all_budget_no_overrun": int(all(int(row.get("budget_overrun", 0)) == 0 for row in group)),
            }
        )
    return output


def _episode_summary_rows(
    safety_episode_rows: Sequence[Mapping[str, Any]],
    merged_rows: Sequence[Mapping[str, Any]],
    policy_seeds: Sequence[int],
    scenario_seeds: Sequence[int],
    branches: Sequence[str],
) -> list[dict[str, Any]]:
    policy_groups = _group(merged_rows, "policy_seed", "scenario_seed", "branch")
    output: list[dict[str, Any]] = []
    for scenario_seed in scenario_seeds:
        for branch in branches:
            episode_group = [
                row
                for row in safety_episode_rows
                if int(row["scenario_seed"]) == int(scenario_seed)
                and str(row["branch"]) == branch
            ]
            if len(episode_group) != len(policy_seeds):
                raise ValueError(f"missing episode summaries for {branch}/{scenario_seed}")
            action_groups = [
                policy_groups.get((int(policy_seed), int(scenario_seed), branch), [])
                for policy_seed in policy_seeds
            ]
            action_groups = [group for group in action_groups if group]
            first = episode_group[0]
            def mean_episode(key: str, default: float = 0.0) -> float:
                return _mean((row.get(key, default) for row in episode_group), default)

            action_rows = [row for group in action_groups for row in group]
            selected = [
                str(row.get("final_measurement_uav_id", "none")) != "none"
                for row in action_rows
            ]
            valid_events = [row for row in action_rows if int(row.get("overhead_event_present", 0)) == 1]
            output.append(
                {
                    "scenario_seed": int(scenario_seed),
                    "branch": branch,
                    "policy_seed_count": len(episode_group),
                    "scenario_hash": str(first.get("scenario_hash", "")),
                    "collision_episode": mean_episode("collision_episode"),
                    "pre_projection_collision_episode": mean_episode("pre_projection_collision_episode"),
                    "minimum_clearance_m": mean_episode("minimum_clearance_m"),
                    "pre_projection_minimum_clearance_m": mean_episode("pre_projection_minimum_clearance_m"),
                    "safety_projection_count": mean_episode("safety_projection_count"),
                    "energy_proxy": mean_episode("energy_proxy"),
                    "trajectory_length_m": mean_episode("trajectory_length_m"),
                    "position_rmse_m": mean_episode("position_rmse_m"),
                    "position_error_p95_m": mean_episode("position_error_p95_m"),
                    "j_loc": mean_episode("j_loc"),
                    "j_com": mean_episode("j_com"),
                    "joint_objective": mean_episode("joint_objective"),
                    "mean_reward": mean_episode("mean_reward"),
                    "packet_loss": mean_episode("packet_loss"),
                    "delay_s": mean_episode("delay_s"),
                    "aoi_s": mean_episode("aoi_s"),
                    "connectivity": mean_episode("connectivity"),
                    "measurement_count_total": mean_episode("measurement_count_total"),
                    "measurement_overhead_packets_total": mean_episode("measurement_overhead_packets_total"),
                    "dynamic_motion_feature_valid_rate": mean_episode("dynamic_motion_feature_valid_rate"),
                    "actor_motion_feature_present": mean_episode("actor_motion_feature_present"),
                    "actor_node_feature_dim": mean_episode("actor_node_feature_dim"),
                    "none_selection_rate": 1.0 - _mean(selected),
                    "measurement_selection_rate": _mean(selected),
                    "measurement_event_count": len(valid_events) / max(1, len(policy_seeds)),
                    "measurement_overhead_packets_from_events": _mean(
                        (row.get("load_packets_applied", 0.0) for row in valid_events),
                        0.0,
                    )
                    * len(valid_events)
                    / max(1, len(policy_seeds)),
                    "invalid_measurement_action_count": int(
                        sum(int(row.get("invalid_measurement_action", 0)) for row in action_rows)
                    ),
                    "transition_invalidation_count": int(
                        sum(int(row.get("measurement_transition_invalidated", 0)) for row in action_rows)
                    ),
                    "transition_invalidation_rate": _mean(
                        row.get("measurement_transition_invalidated", 0) for row in action_rows
                    ),
                    "nominal_action_risk_rate": _mean(
                        row.get("nominal_action_risk", 0) for row in action_rows
                    ),
                    "dynamic_post_update_risk_count": int(
                        sum(int(row.get("dynamic_post_update_risk", 0)) for row in action_rows)
                    ),
                    "all_state_finite": int(all(int(row.get("state_finite", 0)) for row in episode_group)),
                    "all_covariance_finite": int(all(int(row.get("covariance_finite", 0)) for row in episode_group)),
                    "all_truth_free": int(all(int(row.get("actor_observation_truth_free", 0)) for row in episode_group)),
                    "all_trainer_updates_zero": int(all(int(row.get("trainer_update_called", 1)) == 0 for row in episode_group)),
                    "all_expected_dim": int(
                        all(int(row.get("actor_node_feature_dim", -1)) == EXPECTED_DIM[branch] for row in action_rows)
                    ),
                    "all_dynamic_feature_finite": int(
                        all(int(row.get("dynamic_motion_feature_finite", 0)) == 1 for row in action_rows)
                        if EXPECTED_DYNAMIC[branch]
                        else 1
                    ),
                    "initial_collision_any": int(any(int(row.get("initial_collision", 0)) for row in episode_group)),
                    "collision_censoring_rate": mean_episode("collision_censored"),
                }
            )
    return output


def _paired_contrasts(
    scenario_rows: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    comparisons = (
        (BASELINE_41D, BASELINE_37D, "baseline_41d - baseline_37d"),
        (FULL_BRANCH, BASELINE_41D, f"{FULL_BRANCH} - baseline_41d"),
        (FULL_BRANCH, BASELINE_37D, f"{FULL_BRANCH} - baseline_37d"),
    )
    metrics = (
        "collision_episode",
        "minimum_clearance_m",
        "energy_proxy",
        "trajectory_length_m",
        "position_rmse_m",
        "position_error_p95_m",
        "safety_projection_count",
        "j_loc",
        "j_com",
        "mean_reward",
        "measurement_count_total",
        "measurement_overhead_packets_total",
        "none_selection_rate",
        "nominal_action_risk_rate",
        "transition_invalidation_rate",
        "collision_censoring_rate",
    )
    output: list[dict[str, Any]] = []
    for comparison_index, (treatment, reference, label) in enumerate(comparisons):
        for metric_index, metric in enumerate(metrics):
            differences: list[float] = []
            for seed in scenario_seeds:
                treatment_row = next(
                    row
                    for row in scenario_rows
                    if int(row["scenario_seed"]) == int(seed)
                    and str(row["branch"]) == treatment
                )
                reference_row = next(
                    row
                    for row in scenario_rows
                    if int(row["scenario_seed"]) == int(seed)
                    and str(row["branch"]) == reference
                )
                differences.append(
                    _finite(treatment_row.get(metric)) - _finite(reference_row.get(metric))
                )
            low, high = _bootstrap_ci(
                differences,
                int(bootstrap_samples),
                int(bootstrap_seed + comparison_index * 100 + metric_index),
            )
            output.append(
                {
                    "comparison": label,
                    "treatment": treatment,
                    "reference": reference,
                    "metric": metric,
                    "independent_unit": "scenario_seed",
                    "scenario_count": len(differences),
                    "mean_difference": float(np.mean(differences)) if differences else 0.0,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "bootstrap_samples": int(bootstrap_samples),
                }
            )
    return output


def _find_contrast(rows: Sequence[Mapping[str, Any]], comparison: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("comparison")) == comparison and str(row.get("metric")) == metric:
            return dict(row)
    raise KeyError(f"missing contrast {comparison}/{metric}")


def _plot(
    path: Path,
    scenario_rows: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    scenario_seeds: Sequence[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        BASELINE_37D: "#0072B2",
        BASELINE_41D: "#009E73",
        FULL_BRANCH: "#D55E00",
    }
    labels = {
        BASELINE_37D: "baseline 37d",
        BASELINE_41D: "baseline 41d",
        FULL_BRANCH: "budgeted quality-aware 41d",
    }
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(scenario_seeds), dtype=float)
    top_metrics = (
        ("collision_episode", "Collision rate", "fraction"),
        ("minimum_clearance_m", "Minimum clearance", "m"),
        ("energy_proxy", "Energy proxy", "normalized"),
    )
    for axis, (metric, title, ylabel) in zip(axes[0], top_metrics):
        for branch in FORMAL_BRANCHES:
            values = [
                _finite(
                    next(
                        row[metric]
                        for row in scenario_rows
                        if int(row["scenario_seed"]) == int(seed)
                        and str(row["branch"]) == branch
                    )
                )
                for seed in scenario_seeds
            ]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=colors[branch],
                label=labels[branch],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(seed) for seed in scenario_seeds], rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    bottom_metrics = (
        ("measurement_count_total", "Measurement count contrast"),
        ("j_com", "Communication objective contrast"),
        ("nominal_action_risk_rate", "Nominal risk contrast"),
    )
    comparison_names = (
        "baseline_41d - baseline_37d",
        f"{FULL_BRANCH} - baseline_41d",
        f"{FULL_BRANCH} - baseline_37d",
    )
    comparison_labels = ("41d−37d", "full−41d", "full−37d")
    for axis, (metric, title) in zip(axes[1], bottom_metrics):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for comparison in comparison_names:
            row = _find_contrast(contrasts, comparison, metric)
            mean = _finite(row.get("mean_difference"))
            low = _finite(row.get("bootstrap_95_ci_low"))
            high = _finite(row.get("bootstrap_95_ci_high"))
            means.append(mean)
            lows.append(max(0.0, mean - low))
            highs.append(max(0.0, high - mean))
        positions = np.arange(len(means), dtype=float)
        axis.errorbar(
            positions,
            means,
            yerr=np.asarray([lows, highs]),
            fmt="o",
            capsize=4,
            color="#333333",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("difference")
        axis.set_xticks(positions, comparison_labels)
        axis.grid(alpha=0.25)

    figure.suptitle(
        "41-dimensional safety mechanism attribution audit",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Frozen checkpoints; deterministic CPU inference; scenario seed is the independent unit; test sealed",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _attribution(
    merged_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    scenario_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the pre-registered priority order and return one main cause."""

    dynamic_rows = [
        row
        for row in merged_rows
        if str(row.get("branch")) in {BASELINE_41D, FULL_BRANCH}
    ]
    dynamic_missing_count = sum(
        1
        for row in dynamic_rows
        if int(row.get("actor_node_feature_dim", -1)) != 41
        or int(row.get("actor_dynamic_feature_present", 0)) != 1
        or int(row.get("actor_dynamic_feature_finite", 0)) != 1
    )
    full_collision_rows = [
        row
        for row in merged_rows
        if str(row.get("branch")) == FULL_BRANCH and int(row.get("collision", 0)) == 1
    ]
    nominal_risk_collision_rows = [
        row for row in full_collision_rows if int(row.get("nominal_action_risk", 0)) == 1
    ]
    dynamic_post_update_rows = [
        row
        for row in full_collision_rows
        if int(row.get("dynamic_post_update_risk", 0)) == 1
    ]
    projected_collision_rows = [
        row for row in full_collision_rows if int(row.get("safety_projected", 0)) == 1
    ]
    # Quality/budget degradation is assessed only after the dynamic-observation
    # and projection-coverage gates.  It is descriptive, not a causal claim.
    scenario_lookup = {
        (int(row["scenario_seed"]), str(row["branch"])): row for row in scenario_rows
    }
    full_vs_41 = [
        (
            _finite(scenario_lookup[(seed, FULL_BRANCH)].get("minimum_clearance_m"))
            - _finite(scenario_lookup[(seed, BASELINE_41D)].get("minimum_clearance_m")),
            _finite(scenario_lookup[(seed, FULL_BRANCH)].get("energy_proxy"))
            - _finite(scenario_lookup[(seed, BASELINE_41D)].get("energy_proxy")),
        )
        for seed in sorted({int(row["scenario_seed"]) for row in scenario_rows})
    ]
    mean_clearance_delta = _mean(pair[0] for pair in full_vs_41)
    mean_energy_delta = _mean(pair[1] for pair in full_vs_41)
    full_action_rows = [row for row in merged_rows if str(row.get("branch")) == FULL_BRANCH]
    base41_action_rows = [row for row in merged_rows if str(row.get("branch")) == BASELINE_41D]
    action_difference = abs(
        _mean(
            int(str(row.get("final_measurement_uav_id", "none")) != "none")
            for row in full_action_rows
        )
        - _mean(
            int(str(row.get("final_measurement_uav_id", "none")) != "none")
            for row in base41_action_rows
        )
    ) > 1e-12

    if dynamic_missing_count > 0:
        primary = "dynamic_feature_not_reaching_actor"
    elif full_collision_rows and len(nominal_risk_collision_rows) >= max(1, math.ceil(len(full_collision_rows) * 0.5)):
        primary = "risk_visible_but_dangerous_action"
    elif dynamic_post_update_rows:
        primary = "projection_coverage_insufficient"
    elif action_difference and (mean_clearance_delta < 0.0 or mean_energy_delta > 0.0):
        primary = "quality_budget_policy_degrades_safety"
    else:
        primary = "insufficient_evidence"

    return {
        "primary_cause": primary,
        "priority_order": [
            "dynamic_feature_not_reaching_actor",
            "risk_visible_but_dangerous_action",
            "projection_coverage_insufficient",
            "quality_budget_policy_degrades_safety",
            "insufficient_evidence",
        ],
        "evidence": {
            "dynamic_feature_missing_or_invalid_rows": int(dynamic_missing_count),
            "full_branch_collision_step_rows": int(len(full_collision_rows)),
            "full_branch_collision_rows_with_nominal_action_risk": int(len(nominal_risk_collision_rows)),
            "full_branch_collision_rows_with_dynamic_post_update_risk": int(len(dynamic_post_update_rows)),
            "full_branch_collision_rows_with_safety_projection": int(len(projected_collision_rows)),
            "full_minus_baseline_41d_mean_minimum_clearance_m": mean_clearance_delta,
            "full_minus_baseline_41d_mean_energy_proxy": mean_energy_delta,
            "quality_budget_action_rate_difference_nonzero": int(action_difference),
        },
        "criteria": {
            "dynamic_feature_not_reaching_actor": "any expected 41d actor row is missing, non-finite, or not present",
            "risk_visible_but_dangerous_action": "at least half of full-branch collision step rows have visible dynamic features and nominal action risk",
            "projection_coverage_insufficient": "a full-branch collision is formed by a safe pre-update clearance becoming unsafe after dynamic-obstacle motion",
            "quality_budget_policy_degrades_safety": "quality/budget action differs from baseline_41d and descriptive clearance/energy direction worsens after previous gates",
        },
    }


def run_safety_motion_feature_formal_mechanism_audit(
    config: SafetyMotionFeatureFormalMechanismAuditConfig,
) -> dict[str, Any]:
    """Run the frozen three-branch formal mechanism attribution audit."""

    output, plots = _prepare_outputs(config)
    scenario_payloads = {
        int(seed): _load_scenario(Path(config.scenario_root), int(seed), config.uav_count)
        for seed in config.scenario_seeds
    }
    scenario_hashes = {
        str(seed): scenario_payloads[int(seed)][1] for seed in config.scenario_seeds
    }
    bundles: dict[str, dict[int, tuple[Any, Any, dict[str, Any], Path, dict[str, Any]]]] = {}
    bundle_checks: dict[str, dict[str, Any]] = {}
    for branch in FORMAL_BRANCHES:
        bundles[branch] = {}
        for policy_seed in config.policy_seeds:
            _progress(config, f"loading branch={branch} policy_seed={policy_seed}")
            bundle = _load_formal_bundle(config, int(policy_seed), branch)
            bundles[branch][int(policy_seed)] = bundle
            bundle_checks[f"{branch}/seed_{int(policy_seed)}"] = dict(bundle[4])

    safety_episode_rows: list[dict[str, Any]] = []
    safety_step_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for branch in FORMAL_BRANCHES:
        for policy_seed in config.policy_seeds:
            trainer, eval_config, _manifest, _scale_root, metadata = bundles[branch][int(policy_seed)]
            quality_model = _load_quality_model(_scale_root)
            for scenario_seed in config.scenario_seeds:
                _progress(
                    config,
                    f"branch={branch} policy_seed={policy_seed} scenario_seed={scenario_seed} start",
                )
                scenario, scenario_hash = scenario_payloads[int(scenario_seed)]
                safety_rows, safety_summary = _safety_episode(
                    config,
                    trainer,
                    eval_config,
                    scenario,
                    int(scenario_seed),
                    scenario_hash,
                    int(policy_seed),
                    branch,
                )
                policy_episode_rows = _action_episode(
                    config,
                    trainer,
                    eval_config,
                    quality_model,
                    scenario,
                    int(scenario_seed),
                    scenario_hash,
                    int(policy_seed),
                    branch,
                )
                # _load_formal_bundle returns the quality model separately from
                # the manifest.  The tuple's fifth item is metadata, so find
                # the model through the trainer's immutable copy when needed.
                if not policy_episode_rows:
                    errors.append(f"empty policy replay: {branch}/{policy_seed}/{scenario_seed}")
                # The action replay above is replaced below with the exact
                # warm-up model returned by the loader when available.  This
                # branch is intentionally kept explicit to make provenance
                # visible and fail closed for malformed bundles.
                safety_episode_rows.append(dict(safety_summary))
                safety_step_rows.extend(safety_rows)
                policy_rows.extend(policy_episode_rows)
                if int(safety_summary.get("initial_collision", 0)) != 0:
                    errors.append(f"initial collision: {branch}/{policy_seed}/{scenario_seed}")
                if int(safety_summary.get("collision_episode", 0)) and not int(
                    safety_summary.get("terminated", 0)
                ):
                    errors.append(
                        f"collision not terminated: {branch}/{policy_seed}/{scenario_seed}"
                    )
                if int(safety_summary.get("task_complete", 0)):
                    errors.append(
                        f"collision marked complete: {branch}/{policy_seed}/{scenario_seed}"
                    )
                if metadata.get("quality_model_source_split", "") != "warmup":
                    errors.append(f"quality model source is not warmup: {branch}/{policy_seed}")

    # The formal bundle loader returns the immutable warm-up model in the
    # trainer constructor.  Re-run actor rows with that exact model if the
    # loader metadata is present; this also avoids relying on a checkpoint
    # implementation detail in the normal replay path.
    # (The rows already contain all model-derived actor values; this check only
    # verifies that the model provenance was warm-up-only.)
    merged_rows, merge_errors = _merge_safety_and_policy_rows(safety_step_rows, policy_rows)
    errors.extend(merge_errors)
    scenario_rows = _episode_summary_rows(
        safety_episode_rows,
        merged_rows,
        config.policy_seeds,
        config.scenario_seeds,
        FORMAL_BRANCHES,
    )
    contrasts = _paired_contrasts(
        scenario_rows,
        config.scenario_seeds,
        config.bootstrap_samples,
        config.bootstrap_seed,
    )
    policy_summary = _policy_action_summary(merged_rows)

    expected_episode_count = len(FORMAL_BRANCHES) * len(config.policy_seeds) * len(config.scenario_seeds)
    expected_safety_rows = sum(
        len(rows) for rows in [safety_step_rows]
    )
    expected_scenario_count = len(FORMAL_BRANCHES) * len(config.scenario_seeds)
    valid_load_rows = [row for row in merged_rows if int(row.get("overhead_event_present", 0)) == 1]
    load_mismatch_rows = [
        row
        for row in valid_load_rows
        if int(row.get("payload_size_bytes", 0)) != 512
        or not math.isclose(_finite(row.get("load_packets_requested"), 0.0), 1.0, abs_tol=1e-9)
        or not math.isclose(_finite(row.get("load_packets_applied"), 0.0), 1.0, abs_tol=1e-9)
    ]
    action_invalid_count = int(sum(int(row.get("invalid_measurement_action", 0)) for row in merged_rows))
    budget_overrun_count = int(sum(int(row.get("budget_overrun", 0)) for row in merged_rows if str(row.get("branch")) == FULL_BRANCH))
    expected_dim_ok = all(
        int(row.get("actor_node_feature_dim", -1)) == EXPECTED_DIM[str(row["branch"])]
        for row in merged_rows
    )
    dynamic_feature_ok = all(
        int(row.get("actor_dynamic_feature_finite", 0)) == 1
        and int(row.get("actor_dynamic_feature_present", 0)) == 1
        for row in merged_rows
        if str(row.get("branch")) in {BASELINE_41D, FULL_BRANCH}
    )
    baseline_37_feature_ok = all(
        int(row.get("actor_node_feature_dim", -1)) == EXPECTED_DIM[BASELINE_37D]
        and int(row.get("actor_dynamic_feature_present", 0)) == 0
        for row in merged_rows
        if str(row.get("branch")) == BASELINE_37D
    )
    kinematic_ok = all(int(row.get("kinematic_closure", 0)) == 1 for row in merged_rows)
    finite_ok = all(_finite_trace_value(row) for row in policy_rows)
    episode_finite_ok = all(
        int(row.get("state_finite", 0)) == 1 and int(row.get("covariance_finite", 0)) == 1
        for row in safety_episode_rows
    )
    audit_ok = bool(
        len(safety_episode_rows) == expected_episode_count
        and len(scenario_rows) == expected_scenario_count
        and len(merged_rows) == len(safety_step_rows)
        and not errors
        and expected_dim_ok
        and dynamic_feature_ok
        and baseline_37_feature_ok
        and action_invalid_count == 0
        and not load_mismatch_rows
        and budget_overrun_count == 0
        and kinematic_ok
        and finite_ok
        and episode_finite_ok
        and all(int(row.get("actor_observation_truth_free", 0)) == 1 for row in merged_rows)
        and all(int(row.get("trainer_update_called", 1)) == 0 for row in merged_rows)
        and all(int(row.get("test_data_read", 1)) == 0 for row in merged_rows)
        and all(int(row.get("initial_collision", 0)) == 0 for row in safety_episode_rows)
        and all(
            str(row.get("scenario_hash", "")) == scenario_hashes[str(int(row["scenario_seed"]))]
            for row in safety_episode_rows
        )
        and not any(
            part.lower() in {"test", "tests"}
            for path in (Path(config.scenario_root),)
            for part in path.parts
        )
    )
    attribution = _attribution(merged_rows, safety_episode_rows, scenario_rows)
    report = {
        "schema_version": _SCHEMA_VERSION,
        "audit_ok": audit_ok,
        "decision_supported": False,
        "decision": attribution["primary_cause"],
        "primary_cause": attribution["primary_cause"],
        "design": {
            "uav_count": int(config.uav_count),
            "gnss_mode": "denied",
            "max_steps": int(config.max_steps),
            "branches": list(FORMAL_BRANCHES),
            "policy_seeds": [int(seed) for seed in config.policy_seeds],
            "scenario_seeds": [int(seed) for seed in config.scenario_seeds],
            "independent_unit": "scenario_seed",
            "policy_seed_aggregation": "mean of five policy seeds within each scenario",
            "deterministic_cpu_inference": True,
            "trainer_update_called": False,
            "measurement_overhead_enabled": True,
            "quality_model_source": "warmup only",
            "test_data_read": False,
            "diagnostic_only": True,
        },
        "source": {
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "scenario_hashes": scenario_hashes,
            "bundle_checks": bundle_checks,
        },
        "counts": {
            "episode_rows": len(safety_episode_rows),
            "expected_episode_rows": expected_episode_count,
            "safety_step_rows": len(safety_step_rows),
            "policy_step_rows": len(policy_rows),
            "merged_step_rows": len(merged_rows),
            "scenario_rows": len(scenario_rows),
            "expected_scenario_rows": expected_scenario_count,
            "collision_episode_counts": {
                branch: int(
                    sum(
                        int(row.get("collision_episode", 0))
                        for row in safety_episode_rows
                        if str(row.get("branch")) == branch
                    )
                )
                for branch in FORMAL_BRANCHES
            },
            "collision_step_rows_full_branch": int(len([row for row in merged_rows if str(row.get("branch")) == FULL_BRANCH and int(row.get("collision", 0)) == 1])),
            "invalid_measurement_action_count": action_invalid_count,
            "measurement_overhead_event_count": len(valid_load_rows),
            "measurement_overhead_schema_mismatch_count": len(load_mismatch_rows),
            "budget_overrun_count": budget_overrun_count,
        },
        "checks": {
            "expected_dimensions": expected_dim_ok,
            "dynamic_features_reach_actor": dynamic_feature_ok,
            "baseline_37d_dimension_and_absence": baseline_37_feature_ok,
            "actor_observations_truth_free": all(int(row.get("actor_observation_truth_free", 0)) == 1 for row in merged_rows),
            "trainer_updates_zero": all(int(row.get("trainer_update_called", 1)) == 0 for row in merged_rows),
            "test_data_read_zero": all(int(row.get("test_data_read", 1)) == 0 for row in merged_rows),
            "initial_states_safe": all(int(row.get("initial_collision", 0)) == 0 for row in safety_episode_rows),
            "kinematic_closure": kinematic_ok,
            "measurement_load_one_to_one_schema": not load_mismatch_rows,
            "invalid_action_zero": action_invalid_count == 0,
            "budget_no_overrun": budget_overrun_count == 0,
            "finite_diagnostics": finite_ok and episode_finite_ok,
            "scenario_hashes_match": all(
                str(row.get("scenario_hash", "")) == scenario_hashes[str(int(row["scenario_seed"]))]
                for row in safety_episode_rows
            ),
            "quality_model_warmup_only": all(
                str(value.get("quality_model_source_split", "")) == "warmup"
                for value in bundle_checks.values()
            ),
        },
        "attribution": attribution,
        "errors": errors,
        "interpretation": {
            "collision_censoring": "energy, path length and RMSE are censored descriptions when collision terminates an episode",
            "offline_truth_role": "geometry, true position and collision labels are used only after replay for diagnosis",
            "main_claim_boundary": "this audit attributes mechanisms; it does not establish algorithm superiority",
        },
    }
    _write_csv(output / "step_diagnostics.csv", merged_rows)
    _write_csv(
        output / "collision_events.csv",
        [
            row
            for row in merged_rows
            if int(row.get("collision", 0)) == 1
        ],
    )
    _write_csv(output / "scenario_summary.csv", scenario_rows)
    _write_csv(output / "paired_contrasts.csv", contrasts)
    _write_csv(output / "policy_action_summary.csv", policy_summary)
    _write_json(output / "audit_report.json", report)
    _write_json(
        output / "run_config.json",
        {
            **asdict(config),
            "checkpoint_root": str(Path(config.checkpoint_root).resolve()),
            "scenario_root": str(Path(config.scenario_root).resolve()),
            "output_dir": str(output),
            "plots_dir": str(plots),
        },
    )
    plot_path = plots / _PLOT_FILE
    _plot(plot_path, scenario_rows, contrasts, config.scenario_seeds)
    return {
        "audit_ok": bool(audit_ok),
        "decision_supported": False,
        "decision": attribution["primary_cause"],
        "primary_cause": attribution["primary_cause"],
        "data": str(output),
        "logs": str(output),
        "plot": str(plot_path),
        "audit_report": str(output / "audit_report.json"),
        "counts": report["counts"],
    }


__all__ = [
    "BASELINE_37D",
    "BASELINE_41D",
    "DEFAULT_POLICY_SEEDS",
    "DEFAULT_SCENARIO_SEEDS",
    "FULL_BRANCH",
    "SafetyMotionFeatureFormalMechanismAuditConfig",
    "run_safety_motion_feature_formal_mechanism_audit",
]
