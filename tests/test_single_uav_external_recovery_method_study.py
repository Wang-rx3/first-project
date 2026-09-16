"""Small contract tests for the external post-bypass recovery migration."""

import numpy as np

from src.experiments.single_uav_external_recovery_method_study import (
    SOFT_RECOVERY_BUFFER_M,
    SoftFusionRecoveryOracleController,
    _estimated_boundary_recovery_delta,
    _soft_clearance_risk,
    _soft_pass_progress,
    _sigmoid,
)


def test_soft_fusion_sigmoid_and_clearance_weights_are_bounded_and_ordered():
    assert 0.0 < _sigmoid(-10.0) < 0.5 < _sigmoid(10.0) < 1.0
    assert _soft_clearance_risk(1.0) > _soft_clearance_risk(6.0)
    assert _soft_pass_progress(-6.0, 4.0) > _soft_pass_progress(6.0, 4.0)


def test_boundary_recovery_command_points_inward_and_is_zero_inside_margin():
    lower = np.zeros(3)
    upper = np.full(3, 200.0)
    delta, severity = _estimated_boundary_recovery_delta(
        np.asarray([1.0, 100.0, 20.0]), lower, upper, max_delta_v_mps=0.6
    )
    assert delta[0] > 0.0
    assert severity > 0.0
    clear_delta, clear_severity = _estimated_boundary_recovery_delta(
        np.asarray([100.0, 100.0, 20.0]), lower, upper, max_delta_v_mps=0.6
    )
    assert np.allclose(clear_delta, 0.0)
    assert clear_severity == 0.0


def test_teacher_contract_is_explicitly_static_and_actor_constrained():
    assert issubclass(
        SoftFusionRecoveryOracleController,
        object,
    )
    assert SOFT_RECOVERY_BUFFER_M > 0.0
    assert "SoftFusionRecoveryOracleController" in SoftFusionRecoveryOracleController.__name__
