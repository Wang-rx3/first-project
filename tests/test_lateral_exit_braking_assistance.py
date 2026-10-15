"""Pure tests for the diagnostic-only lateral-exit assistance helper."""

from __future__ import annotations

import unittest

import numpy as np

from src.control.lateral_exit_braking_assistance import (
    LATERAL_EXIT_ASSIST_CAP_MPS,
    MinimalLateralExitBrakingAssistance,
)


LATERAL = np.array([0.0, 1.0, 0.0])


def decide(
    helper: MinimalLateralExitBrakingAssistance,
    step: int,
    *,
    clearance: float,
    actor_lateral: float = -0.1,
    velocity_lateral: float = 1.0,
    encounter: bool = True,
    recovery: bool = False,
):
    return helper.compute(
        step=step,
        encounter_established=encounter,
        frozen_obstacle_valid=True,
        estimated_obstacle_clearance_m=clearance,
        estimated_velocity_mps=[0.0, velocity_lateral, 0.0],
        actor_delta_v_mps=[0.0, actor_lateral, 0.0],
        encounter_lateral_unit=LATERAL,
        goal_recovery_active=recovery,
    )


class LateralExitBrakingAssistanceTests(unittest.TestCase):
    def test_locked_cap_rejects_sweep_value(self) -> None:
        with self.assertRaises(ValueError):
            MinimalLateralExitBrakingAssistance(enabled=True, cap_mps=0.05)

    def test_requires_all_gates_and_uses_four_of_five_countersteer(self) -> None:
        helper = MinimalLateralExitBrakingAssistance(enabled=True)
        decisions = [
            decide(helper, index, clearance=value, actor_lateral=actor)
            for index, (value, actor) in enumerate(
                [
                    (3.0, -0.1), (2.0, 0.1), (2.1, 0.1),
                    (2.2, -0.1), (2.3, -0.1), (2.4, -0.1), (2.5, -0.1),
                ]
            )
        ]
        self.assertFalse(any(item.active for item in decisions[:-1]))
        self.assertTrue(decisions[-1].active)

    def test_formula_is_lateral_only_and_capped(self) -> None:
        helper = MinimalLateralExitBrakingAssistance(enabled=True)
        decision = None
        for step, clearance in enumerate([3.0, 2.0, 2.1, 2.2, 2.3, 2.4]):
            decision = decide(helper, step, clearance=clearance, velocity_lateral=1.5)
        assert decision is not None
        self.assertAlmostEqual(decision.post_cap_assist_lateral_delta_v_mps, -0.075)
        np.testing.assert_allclose(decision.applied_assist_delta_v_mps, [0.0, -0.075, 0.0])
        self.assertEqual(LATERAL_EXIT_ASSIST_CAP_MPS, 0.075)

    def test_off_computes_eligibility_but_applies_exact_zero(self) -> None:
        helper = MinimalLateralExitBrakingAssistance(enabled=False)
        decision = None
        for step, clearance in enumerate([3.0, 2.0, 2.1, 2.2, 2.3, 2.4]):
            decision = decide(helper, step, clearance=clearance)
        assert decision is not None
        self.assertTrue(decision.eligible)
        self.assertFalse(decision.active)
        self.assertEqual(decision.inactive_reason, "disabled_off_condition")
        self.assertEqual(decision.requested_assist_lateral_delta_v_mps, 0.0)
        np.testing.assert_array_equal(decision.applied_assist_delta_v_mps, np.zeros(3))

    def test_velocity_zero_crossing_immediately_disables(self) -> None:
        helper = MinimalLateralExitBrakingAssistance(enabled=True)
        for step, clearance in enumerate([3.0, 2.0, 2.1, 2.2, 2.3, 2.4]):
            decision = decide(helper, step, clearance=clearance)
        self.assertTrue(decision.active)
        stopped = decide(helper, 6, clearance=2.5, velocity_lateral=-0.01)
        self.assertFalse(stopped.active)
        self.assertEqual(stopped.inactive_reason, "estimated_lateral_velocity_crossed_zero")

    def test_countersteer_or_clearance_loss_immediately_disables(self) -> None:
        helper = MinimalLateralExitBrakingAssistance(enabled=True)
        for step, clearance in enumerate([3.0, 2.0, 2.1, 2.2, 2.3, 2.4]):
            decision = decide(helper, step, clearance=clearance)
        self.assertTrue(decision.active)
        lost_clearance = decide(helper, 6, clearance=2.39)
        self.assertFalse(lost_clearance.active)
        self.assertFalse(lost_clearance.clearance_increasing_gate)

    def test_goal_recovery_interlock_disables(self) -> None:
        helper = MinimalLateralExitBrakingAssistance(enabled=True)
        for step, clearance in enumerate([3.0, 2.0, 2.1, 2.2, 2.3, 2.4]):
            decision = decide(helper, step, clearance=clearance)
        self.assertTrue(decision.active)
        blocked = decide(helper, 6, clearance=2.5, recovery=True)
        self.assertFalse(blocked.active)
        self.assertEqual(blocked.inactive_reason, "goal_recovery_active")

    def test_helper_has_no_actual_state_parameter(self) -> None:
        names = MinimalLateralExitBrakingAssistance.compute.__annotations__
        self.assertNotIn("actual_position_m", names)
        self.assertNotIn("actual_velocity_mps", names)


if __name__ == "__main__":
    unittest.main()
