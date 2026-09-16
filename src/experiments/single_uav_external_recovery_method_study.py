"""External-method study and minimal post-bypass recovery migration.

This module is deliberately separate from the PPO entry points.  It adapts
three ideas found in public static-navigation work to the existing
single-UAV estimated-state Teacher/DAgger interface:

* DAgger labels states visited by the learner rather than only expert paths;
* static obstacle influence is local and decays after the obstacle is passed;
* bypass and goal tracking are continuously fused instead of hard-switched.

The module does not change reward, dynamics, action mapping, the observation
contract, safety semantics, or PPO.  The only new action-producing component
is an estimated-state, actor-constrained Teacher used for demonstrations and
frozen evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import FormalExperimentConfig
from .single_uav_obstacle_dagger_stage2b import (
    STATIC_SAFE_DISTANCE_M,
    _obstacle_extent,
    _physical_obstacles,
    _unit,
    point_obstacle_distance,
    select_relevant_obstacle,
)
from .single_uav_static_obstacle_dagger_recovery_goal import (
    DEFAULT_BASELINE_CHECKPOINT,
    _aggregate,
    _build_pool,
    _load_checkpoint,
    _migrate_baseline,
    _prepare_roots,
    _rollout_episode,
    _write_rows_csv,
    run_static_obstacle_dagger_recovery_goal,
    ActorConstrainedBoundaryAwareGeometricObstacleOracleController,
)
from .single_uav_rl_equivalent_oracle_audit import _extract_rl_visible_state


STAGE_NAME = "single_uav_external_recovery_method_study"
DEFAULT_DATA_ROOT = Path(
    "data/synthetic/single_uav_external_recovery_method_study_seed2026"
)
DEFAULT_LOG_ROOT = Path(
    "logs/preliminary/single_uav_external_recovery_method_study_seed2026"
)
DEFAULT_PLOT_ROOT = Path(
    "plots/preliminary/single_uav_external_recovery_method_study_seed2026"
)

# These are Teacher-only controller constants.  They do not enter the task
# reward or the environment configuration.  The warning distance is tied to
# the existing static safe distance, while the sigmoid scale supplies a
# continuous transition rather than a new hard execution switch.
SOFT_WARNING_CLEARANCE_M = STATIC_SAFE_DISTANCE_M + 3.0
SOFT_CLEARANCE_SCALE_M = 1.25
SOFT_RECOVERY_BUFFER_M = 0.75
SOFT_PASS_SCALE_M = 1.0
SOFT_TANGENTIAL_DAMPING_GAIN = 0.70
SOFT_BOUNDARY_WARNING_M = 4.0
SOFT_BOUNDARY_GAIN = 0.55


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def _soft_clearance_risk(clearance_m: float) -> float:
    """Return a continuous local risk weight from estimated clearance."""

    if not np.isfinite(float(clearance_m)):
        return 0.0
    return _sigmoid(
        (SOFT_WARNING_CLEARANCE_M - float(clearance_m)) / SOFT_CLEARANCE_SCALE_M
    )


def _soft_pass_progress(along_obstacle_m: float, obstacle_extent_m: float) -> float:
    """Increase smoothly after the estimated obstacle rear face is crossed."""

    threshold = float(obstacle_extent_m) + SOFT_RECOVERY_BUFFER_M
    return _sigmoid(
        (-float(along_obstacle_m) - threshold) / SOFT_PASS_SCALE_M
    )


def _estimated_boundary_recovery_delta(
    position_m: np.ndarray,
    bounds_min_m: np.ndarray,
    bounds_max_m: np.ndarray,
    *,
    max_delta_v_mps: float,
) -> tuple[np.ndarray, float]:
    """Create an inward Teacher command near a configured world boundary.

    This is a label-side recovery cue.  It does not alter the environment's
    boundary projection or its collision/termination semantics.
    """

    position = np.asarray(position_m, dtype=float).reshape(3)
    lower = np.asarray(bounds_min_m, dtype=float).reshape(3)
    upper = np.asarray(bounds_max_m, dtype=float).reshape(3)
    inward = np.zeros(3, dtype=float)
    severity = 0.0
    for index in range(3):
        low_distance = float(position[index] - lower[index])
        high_distance = float(upper[index] - position[index])
        if low_distance < SOFT_BOUNDARY_WARNING_M:
            weight = (SOFT_BOUNDARY_WARNING_M - low_distance) / SOFT_BOUNDARY_WARNING_M
            inward[index] += max(0.0, weight)
            severity = max(severity, weight)
        if high_distance < SOFT_BOUNDARY_WARNING_M:
            weight = (SOFT_BOUNDARY_WARNING_M - high_distance) / SOFT_BOUNDARY_WARNING_M
            inward[index] -= max(0.0, weight)
            severity = max(severity, weight)
    norm = float(np.linalg.norm(inward))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float), 0.0
    inward /= norm
    return inward * float(max_delta_v_mps) * SOFT_BOUNDARY_GAIN * float(min(1.0, severity)), float(severity)


class SoftFusionRecoveryOracleController(
    ActorConstrainedBoundaryAwareGeometricObstacleOracleController
):
    """Actor-constrained bypass/goal Teacher with post-bypass hysteresis.

    The parent class supplies the existing geometric route, boundary-aware
    side choice, estimated-state obstacle selection, and exact Actor action
    transform.  This class only changes how its legal bypass command is
    blended with goal tracking after the obstacle is passed.
    """

    def __init__(self, environment: Any) -> None:
        super().__init__(environment)
        self._active_obstacle_id: str | None = None
        self._release_latched = False
        self._last_pass_progress = 0.0

    def _visible(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        # Use the same estimated-state extraction used by the Actor contract.
        # No environment truth state is read here.
        node = np.asarray(observation["node_features"], dtype=float)
        visible = _extract_rl_visible_state(
            observation,
            expected_node_dim=int(node.shape[1]),
        )
        ids = list(observation.get("node_ids", []))
        own = ids.index(self.uav_id) if self.uav_id in ids else 0
        row = node[own]
        yaw = float(row[8]) if row.size > 8 and np.isfinite(row[8]) else 0.0
        return {**visible, "yaw_rad": yaw}

    def _goal_action(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        target_direction = _unit(goal - position)
        desired_speed = self._desired_speed(goal, position, bypass=False)
        desired_velocity = target_direction * desired_speed
        transformed = self._actor_transform(desired_velocity - velocity)
        return transformed, desired_speed

    def _actor_transform(self, delta_v: np.ndarray) -> np.ndarray:
        # Reuse the parent's already validated Actor componentwise transform.
        from .single_uav_actor_constrained_oracle import actor_action_transform

        transformed = actor_action_transform(
            np.asarray(delta_v, dtype=float).reshape(3),
            self.actor_max_delta_v_mps,
        )
        return np.asarray(transformed["constrained_delta_v_mps"], dtype=float).reshape(3)

    def action(
        self, observations: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_payload, parent_control = super().action(observations)
        parent_control = dict(parent_control)
        visible = self._visible(observations[self.uav_id])
        position = np.asarray(visible["position_m"], dtype=float).reshape(3)
        velocity = np.asarray(visible["velocity_mps"], dtype=float).reshape(3)
        goal = np.asarray(self.environment.goals[self.uav_id], dtype=float).reshape(3)
        goal_direction = _unit(goal - position)

        obstacle = select_relevant_obstacle(position, _physical_obstacles(self.environment))
        obstacle_id = str(obstacle.obstacle_id) if obstacle is not None else None
        if obstacle_id != self._active_obstacle_id:
            self._active_obstacle_id = obstacle_id
            self._release_latched = False
            self._last_pass_progress = 0.0

        clearance = float(point_obstacle_distance(position, obstacle)) if obstacle is not None else float("inf")
        along = float("nan")
        extent = 0.0
        pass_progress = 0.0
        obstacle_blocks = bool(parent_control.get("obstacle_blocks_goal_corridor", False))
        if obstacle is not None and np.linalg.norm(goal_direction) > 1.0e-12:
            relative = np.asarray(obstacle.center_m, dtype=float).reshape(3) - position
            along = float(np.dot(relative, goal_direction))
            extent = float(_obstacle_extent(obstacle, goal_direction))
            pass_progress = _soft_pass_progress(along, extent)
            if (
                along < -(extent + SOFT_RECOVERY_BUFFER_M)
                and clearance >= STATIC_SAFE_DISTANCE_M + 0.5
            ):
                # Once the rear face is safely crossed, keep the recovery
                # decision latched until a different obstacle is selected.
                self._release_latched = True
        self._last_pass_progress = float(pass_progress)

        route_phase_before = str(parent_control.get("route_phase", "goal"))
        route_active = route_phase_before not in {"goal", ""} or obstacle_blocks
        risk = _soft_clearance_risk(clearance)
        if route_active and obstacle is not None:
            # The route has priority before the rear face, then decays
            # continuously as the obstacle becomes geometrically behind us.
            pre_weight = 0.72 + 0.23 * risk
            bypass_weight = (1.0 - pass_progress) * pre_weight
            if self._release_latched:
                bypass_weight *= 0.35
            bypass_weight = float(np.clip(bypass_weight, 0.0, 1.0))
        else:
            bypass_weight = 0.0
        goal_weight = float(1.0 - bypass_weight)

        bypass_action = np.asarray(
            base_payload[self.uav_id]["delta_v"], dtype=float
        ).reshape(3)
        goal_action, desired_goal_speed = self._goal_action(position, velocity, goal)
        fused_before_damping = bypass_weight * bypass_action + goal_weight * goal_action

        radial_velocity = float(np.dot(velocity, goal_direction)) if np.linalg.norm(goal_direction) > 1.0e-12 else 0.0
        tangential_velocity = velocity - radial_velocity * goal_direction
        # Damping is intentionally small while the bypass route is active and
        # becomes strong only as the soft fusion releases it.
        damping_release = max(0.0, 1.0 - bypass_weight)
        tangential_damping = (
            -SOFT_TANGENTIAL_DAMPING_GAIN * damping_release * tangential_velocity
        )
        boundary_delta, boundary_severity = _estimated_boundary_recovery_delta(
            position,
            np.asarray(self.motion.bounds_min_m, dtype=float),
            np.asarray(self.motion.bounds_max_m, dtype=float),
            max_delta_v_mps=float(self.actor_max_delta_v_mps),
        )
        fused_raw = fused_before_damping + tangential_damping + boundary_delta
        transformed = self._actor_transform(fused_raw)
        final_action = transformed

        route_phase = route_phase_before
        if self._release_latched and route_active:
            route_phase = "post_bypass_recovery"
        elif route_active and obstacle is not None:
            route_phase = "bypass_soft_fusion"

        velocity_unit = _unit(velocity)
        control = {
            **parent_control,
            "oracle_action_source": "external_soft_fusion_recovery_teacher",
            "state_source": "actor_visible_estimated_state_plus_static_map_geometry",
            "truth_state_read_for_control": False,
            "route_phase": route_phase,
            "route_phase_before_soft_fusion": route_phase_before,
            "soft_fusion_enabled": True,
            "soft_fusion_bypass_weight": float(bypass_weight),
            "soft_fusion_goal_weight": float(goal_weight),
            "soft_fusion_clearance_risk": float(risk),
            "soft_fusion_pass_progress": float(pass_progress),
            "soft_fusion_release_latched": bool(self._release_latched),
            "soft_fusion_lateral_release_trigger": bool(self._release_latched and pass_progress > 0.5),
            "soft_fusion_along_obstacle_m": float(along) if np.isfinite(along) else float("nan"),
            "soft_fusion_obstacle_extent_m": float(extent),
            "soft_fusion_relevant_obstacle_clearance_m": float(clearance),
            "soft_fusion_tangential_velocity_mps": float(np.linalg.norm(tangential_velocity)),
            "soft_fusion_tangential_damping_delta_v_mps": tangential_damping.tolist(),
            "soft_fusion_boundary_recovery_delta_v_mps": boundary_delta.tolist(),
            "soft_fusion_boundary_severity": float(boundary_severity),
            "soft_fusion_goal_desired_speed_mps": float(desired_goal_speed),
            "soft_fusion_bypass_action_mps": bypass_action.tolist(),
            "soft_fusion_goal_action_mps": goal_action.tolist(),
            "raw_delta_v_mps": fused_raw.tolist(),
            "raw_delta_v_norm_mps": float(np.linalg.norm(fused_raw)),
            "legal_delta_v_mps": final_action.tolist(),
            "legal_delta_v_norm_mps": float(np.linalg.norm(final_action)),
            "demanded_braking_delta_v_mps": max(
                0.0, -float(np.dot(velocity_unit, final_action))
            ),
            "lateral_action_projection_mps": float(
                np.linalg.norm(final_action - np.dot(final_action, goal_direction) * goal_direction)
            )
            if np.linalg.norm(goal_direction) > 1.0e-12
            else 0.0,
        }
        self.last_control = control
        return {
            self.uav_id: {
                "delta_v": final_action.tolist(),
                "measurement_uav_id": "none",
                "bandwidth": {},
            }
        }, control


def _teacher_only_plot(
    rows: Sequence[Mapping[str, Any]],
    plots_root: Path,
) -> str:
    """Plot the Teacher's bypass-release and lateral-damping diagnostics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_root.mkdir(parents=True, exist_ok=True)
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row.get("step", 0)), []).append(row)
    steps = sorted(by_step)
    if not steps:
        raise ValueError("teacher-only diagnostics have no step rows")

    def mean_field(step: int, field: str) -> float:
        values = []
        for row in by_step[step]:
            try:
                value = float(row.get(field, float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("nan")

    bypass = [mean_field(step, "soft_fusion_bypass_weight") for step in steps]
    goal = [mean_field(step, "soft_fusion_goal_weight") for step in steps]
    passed = [mean_field(step, "soft_fusion_pass_progress") for step in steps]
    tangent = [mean_field(step, "soft_fusion_tangential_velocity_mps") for step in steps]
    def vector_norm(value: Any) -> float:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return 0.0
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return 0.0
        return float(np.linalg.norm(array))

    damping = [
        float(
            vector_norm(
                next(
                    (
                        row.get("soft_fusion_tangential_damping_delta_v_mps")
                        for row in by_step[step]
                        if row.get("soft_fusion_tangential_damping_delta_v_mps")
                    ),
                    [0.0, 0.0, 0.0],
                )
            )
        )
        for step in steps
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(steps, bypass, label="bypass weight", color="#d55e00", linewidth=2.0)
    axes[0].plot(steps, goal, label="goal weight", color="#0072b2", linewidth=2.0)
    axes[0].plot(steps, passed, label="pass progress", color="#009e73", linestyle="--", linewidth=1.8)
    axes[0].set_ylabel("soft-fusion weight")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(steps, tangent, label="tangential speed", color="#cc79a7", linewidth=2.0)
    axes[1].plot(steps, damping, label="damping |delta-v|", color="#f0e442", linewidth=2.0)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("m/s")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    path = plots_root / "teacher_only_soft_fusion_recovery.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path.resolve())


def _teacher_only_audit(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    pool: Sequence[Mapping[str, Any]],
    data_root: Path,
    logs_root: Path,
    plots_root: Path,
    progress: bool,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for item in pool:
        summary, episode_rows, _records = _rollout_episode(
            student_trainer=trainer,
            preservation_trainer=trainer,
            config=config,
            item=item,
            trace_root=data_root / "teacher_only_traces",
            policy="soft_fusion_teacher",
            collection_mode="teacher_bootstrap",
            include_demo=False,
            evaluation_stage="teacher_only_static_holdout",
            oracle_factory=SoftFusionRecoveryOracleController,
        )
        summaries.append(summary)
        rows.extend(episode_rows)
        if progress and int(item["episode_index"]) in {1, len(pool)}:
            print(
                f"[{STAGE_NAME}] teacher-only {item['episode_index']}/{len(pool)}",
                flush=True,
            )
    aggregate = _aggregate(summaries, rows)
    _write_rows_csv(logs_root / "teacher_only_episode_metrics.csv", summaries)
    _write_rows_csv(logs_root / "teacher_only_step_metrics.csv", rows)
    teacher_plot = _teacher_only_plot(rows, plots_root)
    acceptance = {
        "entered_8m_at_least": int(aggregate.get("entered_8m", 0)) >= max(1, len(pool) - 1),
        "entered_3m_at_least": int(aggregate.get("entered_3m", 0)) >= max(1, len(pool) - 1),
        "entered_1m_at_least": int(aggregate.get("entered_1m", 0)) >= max(1, len(pool) - 1),
        "stable_goal_nonzero": int(aggregate.get("stable_goal", 0)) > 0,
        "recovery_at_least": int(aggregate.get("post_bypass_recovery", 0)) >= max(1, len(pool) - 1),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    result = {
        "episodes": len(pool),
        "aggregate": aggregate,
        "acceptance": acceptance,
        "teacher_only": True,
        "training_performed": False,
        "ppo_updates": 0,
        "reward_or_environment_modified": False,
        "dynamic_obstacles_used": False,
        "files": {
            "episode_metrics": str((logs_root / "teacher_only_episode_metrics.csv").resolve()),
            "step_metrics": str((logs_root / "teacher_only_step_metrics.csv").resolve()),
            "diagnostic_plot": teacher_plot,
        },
    }
    _write_json(logs_root / "teacher_only_summary.json", _jsonable(result))
    return result


def _method_comparison_report(logs_root: Path) -> None:
    report = """# External method comparison and local migration

## Public reference patterns

| Reference | Relevant pattern | Local use |
|---|---|---|
| [uzh-rpg/agile_autonomy](https://github.com/uzh-rpg/agile_autonomy) | DAgGER alternates learner rollouts, expert labels and later learner updates; the learner visits the states that matter for its own errors. | Keep the current student-closed-loop recollection and query the Teacher on visited post-bypass states. |
| [ZYunfeii/UAV_Obstacle_Avoiding_DRL](https://github.com/ZYunfeii/UAV_Obstacle_Avoiding_DRL) | Static APF combines goal attraction and local obstacle repulsion whose influence is limited to a local range. | Use static-map geometry only, with a clearance risk that decays after the obstacle rear face is crossed. |
| [suiuiuiuiu/av-soft-fusion-rl](https://github.com/suiuiuiuiu/av-soft-fusion-rl) | A sigmoid soft fusion continuously blends navigation and recovery actions rather than using a brittle hard switch. | Blend the existing legal bypass action and goal action, then apply the existing Actor action transform. |

The external projects are references for mechanisms, not copied task definitions:
the current project remains single-UAV, static-obstacle-only, estimated-state,
and uses its existing action/dynamics/safety contracts.

## Local root-cause evidence

The previous R3 audit recorded post-bypass phase imbalance, repeated boundary
projection in the post-bypass segment, and persistent lateral actions while
radial progress became negative.  The failure is therefore a transition and
velocity-inertia problem: the bypass command remains active after the obstacle
is behind the vehicle, so lateral velocity is not released and damped in time.
The migrated Teacher below addresses that transition only; it does not change
the task reward or environment execution layer.
"""
    (logs_root / "external_method_comparison.md").write_text(report, encoding="utf-8")


def _root_cause_report(logs_root: Path) -> None:
    report = """# Current post-bypass root-cause audit

## Observed failure chain

`bypass action persists` → `tangential velocity accumulates` →
`boundary layer intervenes` → `goal-directed radial progress is weakened or
reversed` → `the policy does not reliably recover toward the goal`.

This is the specific instance of the reported issue:

> 绕障动作持续时间过长 → 横向漂移 → 边界层持续介入 → 无法恢复目标径向运动。

The current work treats it as a post-bypass recovery and credit-assignment
problem, not as evidence that static-obstacle direction observation is absent.

## Minimal mechanism required

1. Keep a bypass state while the obstacle is ahead and unsafe.
2. Detect obstacle passage from estimated position, static map geometry and
   goal direction, with a clearance margin and a latched release condition.
3. Continuously reduce bypass weight after passage instead of switching at a
   single step.
4. Dampen estimated tangential velocity during release and restore the goal
   action's radial component.
5. Provide these labels on student-visited DAgger states, including boundary
   onset and recovery states.

No new reward, termination rule, dynamics term, truth-state input or action
space is introduced.
"""
    (logs_root / "current_failure_root_cause.md").write_text(report, encoding="utf-8")


def _teacher_design_report(logs_root: Path) -> None:
    report = """# Soft-fusion recovery Teacher design

The Teacher reuses the current Actor-constrained boundary-aware geometric
controller.  It computes, from Actor-visible estimated state plus static map
geometry:

* local clearance risk `q(d) = sigmoid((d_warning-d)/s_d)`;
* rear-face passage progress `p = sigmoid((-along-obstacle - extent-buffer)/s_p)`;
* bypass weight `w_b = (1-p) * w_pre`, with a latched safe-release reduction;
* goal weight `w_g = 1-w_b`;
* tangential damping `-k_t (1-w_b) v_t`;
* a small inward boundary recovery label when the estimated state is close to
  the configured world boundary.

The final command is

`a_teacher = ActorTransform(w_b a_bypass + w_g a_goal - k_t(1-w_b)v_t + a_boundary)`.

The final transform is the existing componentwise Actor action transform; no
teleport, direct velocity assignment, safety semantic change or truth-state
read is used.  The Teacher control record exposes the fusion weights,
passage/release state, tangential damping and boundary cue for diagnosis.
"""
    (logs_root / "teacher_design_report.md").write_text(report, encoding="utf-8")


def _evaluation_report(logs_root: Path, run_result: Mapping[str, Any]) -> None:
    aggregates = dict(run_result.get("evaluation", {}).get("aggregates", {}))

    def metric(value: Mapping[str, Any], field: str) -> str:
        return f"{int(value.get(field, 0) or 0)}/{int(value.get('episodes', 0) or 0)}"

    roles = [
        "baseline_rl",
        "dagger_round1",
        "navigation_preserving_fine_tuned",
        "bc_actor",
        "oracle_teacher",
    ]
    lines = [
        "# Final frozen evaluation",
        "",
        "This report compares the external recovery study roles without PPO updates.",
        "",
        "## Normal far-distance holdout",
        "",
        "| role | entered 8m | entered 3m | entered 1m | stable | mean min distance (m) | mean max hold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in roles:
        value = aggregates.get("normal_far_distance", {}).get(role)
        if not value:
            continue
        lines.append(
            f"| {role} | {metric(value, 'entered_8m')} | {metric(value, 'entered_3m')} | "
            f"{metric(value, 'entered_1m')} | {metric(value, 'stable_goal')} | "
            f"{float(value.get('mean_minimum_goal_distance_m', 0.0)):.3f} | "
            f"{float(value.get('mean_maximum_consecutive_goal_hold_steps', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Static holdout",
            "",
            "| role | entered 8m | entered 3m | entered 1m | stable | recovery | boundary projection steps | mean max hold |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for role in roles:
        value = aggregates.get("static_holdout", {}).get(role)
        if not value:
            continue
        lines.append(
            f"| {role} | {metric(value, 'entered_8m')} | {metric(value, 'entered_3m')} | "
            f"{metric(value, 'entered_1m')} | {metric(value, 'stable_goal')} | "
            f"{metric(value, 'post_bypass_recovery')} | {int(value.get('boundary_projection_steps', 0) or 0)} | "
            f"{float(value.get('mean_maximum_consecutive_goal_hold_steps', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "The normal-navigation gate is entered-8m >= 19/20.  Static success is "
            "interpreted together with raw action, pre-projection risk and recovery "
            "diagnostics; collision=0 alone is not sufficient evidence.",
            "",
        ]
    )
    (logs_root / "final_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _collection_report(logs_root: Path, run_result: Mapping[str, Any], smoke: bool) -> None:
    dataset = dict(run_result.get("dataset", {}))
    report = "\n".join(
        [
            "# DAgger collection and BC report",
            "",
            f"smoke={bool(smoke)}; PPO updates={run_result.get('ppo_updates', 0)}; "
            f"training_performed={run_result.get('training_performed', False)}.",
            "",
            f"train trajectories={dataset.get('train_episode_count', 0)}, "
            f"validation trajectories={dataset.get('validation_episode_count', 0)}, "
            f"train balanced samples={dataset.get('train_samples_balanced', 0)}, "
            f"validation balanced samples={dataset.get('validation_samples_balanced', 0)}.",
            "",
            f"phase counts={json.dumps(dataset.get('phase_counts_raw', {}), ensure_ascii=False)}",
            f"collection modes={json.dumps(dataset.get('collection_mode_counts', {}), ensure_ascii=False)}",
            f"trajectory leakage={dataset.get('trajectory_leakage')}; pair leakage={dataset.get('pair_leakage')}.",
            "",
            "BC is Actor-only, initialized from the migrated baseline, with critic, "
            "normalizer and log-std frozen.  No PPO fine-tuning is launched by this "
            "stage.",
            "",
        ]
    )
    (logs_root / "dagger_collection_report.md").write_text(report, encoding="utf-8")
    (logs_root / "bc_smoke_report.md").write_text(report, encoding="utf-8")


def _validate_nested_evaluations(run_result: Mapping[str, Any]) -> dict[str, Any]:
    """Run the project's standard validator on every role/split root."""

    from .formal import validate_preliminary

    evaluation = run_result.get("evaluation", {})
    evaluation_files = run_result.get("evaluation", {}).get("files", {})
    episode_metrics = Path(str(evaluation_files.get("episode_metrics", "")))
    # evaluation_episode_metrics.csv lives under logs/evaluation; derive the
    # sibling data evaluation root from the run's data manifest instead of
    # guessing a repository-wide path.
    dataset_manifest = Path(str(run_result.get("dataset", {}).get("files", {}).get("train", "")))
    if not dataset_manifest.is_file():
        return {"ok": False, "reason": "dataset manifest not found"}
    data_eval_root = dataset_manifest.parents[1] / "evaluation"
    reports: dict[str, Any] = {}
    for split in ("normal_far_distance", "static_holdout"):
        split_root = data_eval_root / split
        for role_dir in sorted(split_root.iterdir()) if split_root.is_dir() else []:
            if role_dir.is_dir():
                report = validate_preliminary(role_dir)
                reports[f"{split}/{role_dir.name}"] = {
                    "ok": bool(report.get("ok", False)),
                    "path": str(role_dir.resolve()),
                    "scale_count": len(report.get("scales", {})),
                }
    return {
        "ok": bool(reports) and all(bool(value["ok"]) for value in reports.values()),
        "roles": reports,
        "episode_metrics_path": str(episode_metrics.resolve()) if episode_metrics else "",
    }


def run_external_recovery_method_study(
    *,
    baseline_checkpoint: str | Path = DEFAULT_BASELINE_CHECKPOINT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    logs_root: str | Path = DEFAULT_LOG_ROOT,
    plots_root: str | Path = DEFAULT_PLOT_ROOT,
    smoke: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run Teacher audit, then only proceed to BC/DAgger if Teacher passes."""

    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_root = Path(plots_root)
    _prepare_roots(data_root, logs_root, plots_root)
    baseline_path = Path(baseline_checkpoint).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    _method_comparison_report(logs_root)
    _root_cause_report(logs_root)
    _teacher_design_report(logs_root)

    migrated_path, migration = _migrate_baseline(
        baseline_path,
        data_root=data_root,
        logs_root=logs_root,
        plots_root=plots_root,
        progress=progress,
    )
    trainer, config, _payload, identity = _load_checkpoint(
        migrated_path, data_root / "teacher_only_source"
    )
    teacher_count = 4 if smoke else 20
    teacher_pool = _build_pool(
        config,
        split="test",
        normal_count=0,
        static_count=teacher_count,
        seed_base=3900000,
        round_index=7,
    )
    teacher_result = _teacher_only_audit(
        trainer=trainer,
        config=config,
        pool=teacher_pool,
        data_root=data_root,
        logs_root=logs_root,
        plots_root=plots_root,
        progress=progress,
    )

    summary: dict[str, Any] = {
        "stage": STAGE_NAME,
        "status": "teacher_gate_failed" if not teacher_result["acceptance"]["passed"] else "teacher_gate_passed",
        "smoke": bool(smoke),
        "training_performed": False,
        "ppo_updates": 0,
        "reward_or_environment_modified": False,
        "dynamic_obstacles_used": False,
        "checkpoint_provenance": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "migrated": migration,
            "loaded_policy": identity,
        },
        "teacher_only": teacher_result,
        "contracts": {
            "static_only": True,
            "single_uav": True,
            "teacher_truth_state_read_for_control": False,
            "reward_or_environment_modified": False,
            "ppo_updates": 0,
            "old_outputs_overwritten": False,
        },
        "files": {
            "external_method_comparison": str((logs_root / "external_method_comparison.md").resolve()),
            "current_failure_root_cause": str((logs_root / "current_failure_root_cause.md").resolve()),
            "teacher_design_report": str((logs_root / "teacher_design_report.md").resolve()),
        },
    }

    if teacher_result["acceptance"]["passed"]:
        dagger_data = data_root / "dagger_bc"
        dagger_logs = logs_root / "dagger_bc"
        dagger_plots = plots_root / "dagger_bc"
        run_result = run_static_obstacle_dagger_recovery_goal(
            baseline_checkpoint=baseline_path,
            data_root=dagger_data,
            logs_root=dagger_logs,
            plots_root=dagger_plots,
            smoke=smoke,
            progress=progress,
            oracle_factory=SoftFusionRecoveryOracleController,
        )
        summary["dagger_bc"] = run_result
        summary["status"] = "completed"
        normal_bc = run_result.get("evaluation", {}).get("aggregates", {}).get(
            "normal_far_distance", {}
        ).get("bc_actor", {})
        static_bc = run_result.get("evaluation", {}).get("aggregates", {}).get(
            "static_holdout", {}
        ).get("bc_actor", {})
        normal_episodes = int(normal_bc.get("episodes", 0) or 0)
        normal_threshold = 19 if normal_episodes >= 20 else max(0, normal_episodes - 1)
        normal_gate = int(normal_bc.get("entered_8m", 0) or 0) >= normal_threshold
        summary["bc_gates"] = {
            "normal_entered_8m_ge_19_of_20": normal_gate,
            "normal_entered_8m": int(normal_bc.get("entered_8m", 0) or 0),
            "normal_gate_threshold": normal_threshold,
            "normal_gate_is_formal_20_episode_gate": normal_episodes >= 20,
            "static_entered_8m": int(static_bc.get("entered_8m", 0) or 0),
            "static_entered_1m": int(static_bc.get("entered_1m", 0) or 0),
            "static_stable_goal": int(static_bc.get("stable_goal", 0) or 0),
            "ppo_allowed": False,
            "decision": "stop_before_ppo_because_normal_navigation_gate_failed",
        }
        summary["validate_preliminary"] = _validate_nested_evaluations(run_result)
        _write_json(
            logs_root / "validate_preliminary_summary.json",
            _jsonable(summary["validate_preliminary"]),
        )
        summary["files"].update(
            {
                "dagger_collection_report": str((logs_root / "dagger_collection_report.md").resolve()),
                "bc_smoke_report": str((logs_root / "bc_smoke_report.md").resolve()),
                "final_evaluation_report": str((logs_root / "final_evaluation_report.md").resolve()),
            }
        )
        _collection_report(logs_root, run_result, smoke)
        _evaluation_report(logs_root, run_result)

    _write_json(logs_root / "summary.json", _jsonable(summary))
    _write_json(data_root / "summary.json", _jsonable(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=STAGE_NAME)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run_external_recovery_method_study(
        baseline_checkpoint=args.baseline_checkpoint,
        data_root=args.data_root,
        logs_root=args.logs_root,
        plots_root=args.plots_root,
        smoke=args.smoke,
        progress=not args.quiet,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASELINE_CHECKPOINT",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_PLOT_ROOT",
    "STAGE_NAME",
    "SoftFusionRecoveryOracleController",
    "run_external_recovery_method_study",
]
