import numpy as np
from src.analysis.partial_gnss_ci_diagnostic_v9 import endpoint_derivative


def test_objective_endpoint_matches_finite_difference():
    p = np.diag([.3, 2., 4.])
    h = np.array([.6, .8, 0.])
    variance = .4
    def trace(w):
        return np.trace(np.linalg.inv(w*np.linalg.inv(p)+(1-w)*np.outer(h, h)/variance))
    numeric = (trace(1)-trace(1-1e-6))/1e-6
    np.testing.assert_allclose(endpoint_derivative(p, h, variance), numeric, atol=1e-4)


def test_isotropic_endpoint_condition_includes_unmeasured_axes():
    h = [1., 0., 0.]
    assert endpoint_derivative(np.eye(3), h, .4) < 0
    assert endpoint_derivative(np.eye(3), h, .2) > 0
