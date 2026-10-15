import numpy as np
from src.analysis.partial_gnss_ci_resolution_v9 import objective_values


def test_scalar_trace_matches_full_information_matrix_inversion():
    p = np.array([[2., .1, -.2], [.1, 1., .3], [-.2, .3, 3.]])
    h = np.array([.6, .8, 0.])
    weights = np.arange(1, 21)/20
    expected = [np.trace(np.linalg.inv(w*np.linalg.inv(p)+(1-w)*np.outer(h, h)/.45)) for w in weights]
    np.testing.assert_allclose(objective_values(p, h, .45, weights), expected, rtol=1e-12, atol=1e-12)
