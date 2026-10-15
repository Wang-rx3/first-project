import copy
import numpy as np

from src.analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
from src.control.fixed_navigation_execution import compose_fixed_navigation_actions


def test_unused_truth_annotations_cannot_change_bias_acquisition():
    left=StartupLocalEKF(np.arange(15)*.1);right=StartupLocalEKF(np.arange(15)*.1)
    observable={"accel":np.array([.12,-.2,.08]),"gyro":np.array([.01,.03,-.04])}
    alternate={**observable,"truth_position":[1000,1000,1000],"bias_accel":[20,20,20],"bias_gyro":[-50,-50,-50]}
    left.predict(observable,.2);right.predict(alternate,.2)
    np.testing.assert_array_equal(left.state,right.state)
    np.testing.assert_array_equal(left.covariance,right.covariance)
    assert not np.allclose(left.state[9:15],np.zeros(6))


def test_controller_ignores_truth_fields_and_uses_local_estimate():
    state=np.zeros(15);state[:3]=[40,40,22];state[3:6]=[.1,.2,.3]
    first={"uav_00":{"estimate":state,"goal":np.array([75,40,22]),"truth_position":[0,0,0]}}
    second=copy.deepcopy(first);second["uav_00"]["truth_position"]=[200,200,60]
    comm={"uav_00":{"measurement_uav_id":"none","bandwidth":{}}}
    a,_=compose_fixed_navigation_actions(first,comm);b,_=compose_fixed_navigation_actions(second,comm)
    np.testing.assert_array_equal(a["uav_00"]["delta_v"],b["uav_00"]["delta_v"])


def test_stationary_fixture_is_required_by_harness_before_candidate():
    from src.experiments.local_ekf_startup_validation_v8 import episode
    from src.analysis.local_ekf_startup_diagnostic_v8 import CONFIG
    from src.experiments.partial_gnss_baseline_v7 import old
    spec=old.read_json(CONFIG);case=copy.deepcopy(spec["original_cases"][0])
    case["scenario"]["initial_velocities"]["uav_00"]=[1,0,0]
    # Constructor/reset is reversible and no physical step is reached.
    import pytest
    with pytest.raises(ValueError,match="stationary startup fixture"):
        episode(case,spec,"stationary_bias_acquisition")


def test_existing_batch_cap_rejects_before_any_new_physical_step(monkeypatch):
    import pytest
    from src.experiments import local_ekf_startup_validation_v8 as validation
    from src.analysis.local_ekf_startup_diagnostic_v8 import BASE
    target=BASE/"UNIT_BUDGET_GUARD_001"
    assert not target.exists()
    def forbidden(*args,**kwargs):
        raise AssertionError("physical episode was called")
    monkeypatch.setattr(validation,"episode",forbidden)
    with pytest.raises(RuntimeError,match="whole invocation"):
        validation.run("UNIT_BUDGET_GUARD_001","native",batch_id="v8_research_001")
    assert not target.exists()
