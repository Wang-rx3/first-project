"""Minimal diagnostic-only lateral-exit braking assistance.

The helper consumes estimated-state/static-map-derived gates and the frozen Actor
command.  It has no simulator-actual-state input and does not choose a bypass side.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

import numpy as np


LATERAL_EXIT_ASSIST_CAP_MPS = 0.075
LATERAL_EXIT_THRESHOLD_MPS = 0.5
COUNTERSTEER_WINDOW_STEPS = 5
COUNTERSTEER_MIN_STEPS = 4
CLEARANCE_INCREASING_STEPS = 3
SIGN_DEADBAND_MPS = 1.0e-8
CLEARANCE_EPSILON_M = 1.0e-9


def _vector3(value: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _sign(value: float, *, deadband: float = SIGN_DEADBAND_MPS) -> int:
    if not np.isfinite(value) or abs(float(value)) <= float(deadband):
        return 0
    return 1 if value > 0.0 else -1


@dataclass(frozen=True)
class LateralExitBrakingDecision:
    enabled: bool
    eligible: bool
    active: bool
    inactive_reason: str
    encounter_step: int | None
    initial_lateral_sign: int
    minimum_clearance_m: float
    clearance_increase_run: int
    actor_sign_window: tuple[int, ...]
    countersteer_count: int
    closest_approach_gate: bool
    frozen_obstacle_gate: bool
    clearance_increasing_gate: bool
    sustained_countersteer_gate: bool
    estimated_lateral_velocity_mps: float
    actor_lateral_delta_v_mps: float
    requested_assist_lateral_delta_v_mps: float
    post_cap_assist_lateral_delta_v_mps: float
    requested_assist_delta_v_mps: np.ndarray
    applied_assist_delta_v_mps: np.ndarray
    combined_pre_transform_delta_v_mps: np.ndarray


class MinimalLateralExitBrakingAssistance:
    """Stateful pre-transform assistance for one episode.

    The treatment is deliberately fixed: no cap sweep and no direction planning.
    The initial lateral-motion sign is frozen from estimated velocity when the
    encounter is first established.
    """

    def __init__(self, *, enabled: bool, cap_mps: float = LATERAL_EXIT_ASSIST_CAP_MPS) -> None:
        if float(cap_mps) != LATERAL_EXIT_ASSIST_CAP_MPS:
            raise ValueError("lateral-exit diagnostic cap is locked to 0.075 m/s per step")
        self.enabled = bool(enabled)
        self.cap_mps = float(cap_mps)
        self.reset()

    def reset(self) -> None:
        self.encounter_step: int | None = None
        self.initial_lateral_sign = 0
        self.previous_clearance_m = float("nan")
        self.minimum_clearance_m = float("inf")
        self.clearance_increase_run = 0
        self.actor_sign_window: Deque[int] = deque(maxlen=COUNTERSTEER_WINDOW_STEPS)

    def compute(
        self,
        *,
        step: int,
        encounter_established: bool,
        frozen_obstacle_valid: bool,
        estimated_obstacle_clearance_m: float,
        estimated_velocity_mps: Iterable[float],
        actor_delta_v_mps: Iterable[float],
        encounter_lateral_unit: Iterable[float],
        goal_recovery_active: bool,
        episode_done: bool = False,
    ) -> LateralExitBrakingDecision:
        velocity = _vector3(estimated_velocity_mps, name="estimated_velocity_mps")
        actor = _vector3(actor_delta_v_mps, name="actor_delta_v_mps")
        lateral = _vector3(encounter_lateral_unit, name="encounter_lateral_unit")
        norm = float(np.linalg.norm(lateral))
        if abs(norm - 1.0) > 1.0e-8:
            raise ValueError("encounter_lateral_unit must be a unit vector")
        clearance = float(estimated_obstacle_clearance_m)
        if not np.isfinite(clearance):
            clearance = float("nan")

        estimated_lateral_velocity = float(np.dot(velocity, lateral))
        actor_lateral = float(np.dot(actor, lateral))
        if encounter_established and self.encounter_step is None:
            self.encounter_step = int(step)
            self.initial_lateral_sign = _sign(estimated_lateral_velocity)
            self.previous_clearance_m = clearance
            self.minimum_clearance_m = clearance
        elif self.encounter_step is not None and self.initial_lateral_sign == 0:
            # Fail closed until the first non-deadband estimated motion sign is
            # available; once latched, the encounter direction never changes.
            self.initial_lateral_sign = _sign(estimated_lateral_velocity)

        valid_encounter = self.encounter_step is not None and self.initial_lateral_sign != 0
        closest_gate = False
        clearance_gate = False
        sustained_gate = False
        if valid_encounter:
            if np.isfinite(clearance):
                closest_gate = bool(
                    np.isfinite(self.minimum_clearance_m)
                    and clearance > self.minimum_clearance_m + CLEARANCE_EPSILON_M
                )
                increasing = bool(
                    np.isfinite(self.previous_clearance_m)
                    and clearance > self.previous_clearance_m + CLEARANCE_EPSILON_M
                )
                self.clearance_increase_run = self.clearance_increase_run + 1 if increasing else 0
                self.minimum_clearance_m = min(self.minimum_clearance_m, clearance)
                self.previous_clearance_m = clearance
            else:
                self.clearance_increase_run = 0
            clearance_gate = self.clearance_increase_run >= CLEARANCE_INCREASING_STEPS
            self.actor_sign_window.append(_sign(actor_lateral))
            opposite = -self.initial_lateral_sign
            sustained_gate = bool(
                len(self.actor_sign_window) == COUNTERSTEER_WINDOW_STEPS
                and sum(value == opposite for value in self.actor_sign_window)
                >= COUNTERSTEER_MIN_STEPS
            )

        velocity_still_original = bool(
            valid_encounter
            and _sign(estimated_lateral_velocity) == self.initial_lateral_sign
        )
        gates = {
            "episode_done": not bool(episode_done),
            "valid_encounter": valid_encounter,
            "frozen_obstacle_valid": bool(frozen_obstacle_valid),
            "past_closest_approach": closest_gate,
            "clearance_increasing": clearance_gate,
            "sustained_countersteer": sustained_gate,
            "estimated_velocity_original_direction": velocity_still_original,
            "goal_recovery_inactive": not bool(goal_recovery_active),
        }
        eligible = all(gates.values())
        countersteer_count = (
            sum(value == -self.initial_lateral_sign for value in self.actor_sign_window)
            if self.initial_lateral_sign else 0
        )
        if episode_done:
            reason = "episode_done"
        elif not valid_encounter:
            reason = "encounter_not_established"
        elif not frozen_obstacle_valid:
            reason = "frozen_obstacle_unavailable_or_mismatch"
        elif not closest_gate:
            reason = "not_past_closest_approach"
        elif not clearance_gate:
            reason = "clearance_not_sustained_increasing"
        elif not sustained_gate:
            reason = "countersteer_not_sustained"
        elif not velocity_still_original:
            reason = "estimated_lateral_velocity_crossed_zero"
        elif goal_recovery_active:
            reason = "goal_recovery_active"
        elif not self.enabled:
            reason = "disabled_off_condition"
        else:
            reason = "active"

        requested_lateral = (
            -float(_sign(estimated_lateral_velocity))
            * min(self.cap_mps, abs(estimated_lateral_velocity))
            if eligible and self.enabled else 0.0
        )
        requested = requested_lateral * lateral
        active = bool(self.enabled and eligible and abs(requested_lateral) > 0.0)
        applied_lateral = requested_lateral if active else 0.0
        applied = applied_lateral * lateral
        combined = actor + applied
        return LateralExitBrakingDecision(
            enabled=self.enabled,
            eligible=eligible,
            active=active,
            inactive_reason=reason,
            encounter_step=self.encounter_step,
            initial_lateral_sign=int(self.initial_lateral_sign),
            minimum_clearance_m=float(self.minimum_clearance_m),
            clearance_increase_run=int(self.clearance_increase_run),
            actor_sign_window=tuple(self.actor_sign_window),
            countersteer_count=int(countersteer_count),
            closest_approach_gate=closest_gate,
            frozen_obstacle_gate=bool(frozen_obstacle_valid),
            clearance_increasing_gate=clearance_gate,
            sustained_countersteer_gate=sustained_gate,
            estimated_lateral_velocity_mps=estimated_lateral_velocity,
            actor_lateral_delta_v_mps=actor_lateral,
            requested_assist_lateral_delta_v_mps=requested_lateral,
            post_cap_assist_lateral_delta_v_mps=applied_lateral,
            requested_assist_delta_v_mps=requested,
            applied_assist_delta_v_mps=applied,
            combined_pre_transform_delta_v_mps=combined,
        )


__all__ = [
    "CLEARANCE_INCREASING_STEPS",
    "COUNTERSTEER_MIN_STEPS",
    "COUNTERSTEER_WINDOW_STEPS",
    "LATERAL_EXIT_ASSIST_CAP_MPS",
    "LATERAL_EXIT_THRESHOLD_MPS",
    "LateralExitBrakingDecision",
    "MinimalLateralExitBrakingAssistance",
]
