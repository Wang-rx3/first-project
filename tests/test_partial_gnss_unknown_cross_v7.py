import numpy as np

from src.analysis.partial_gnss_fusion_v7 import conditional_linear_error_bound


def test_scalar_partial_information_bound_covers_admissible_cross_covariances():
    rng=np.random.default_rng(72263)
    for _ in range(30):
        factor=rng.normal(size=(15,15))
        prior=factor@factor.T+np.eye(15)
        H=np.zeros((1,15)); H[0,:3]=rng.normal(size=3); H/=np.linalg.norm(H)
        q=.2; measurement_noise=.08**2; pseudo=q+measurement_noise; w=.7
        posterior=np.linalg.solve(w*np.linalg.solve(prior,np.eye(15))+(1-w)*H.T@H/pseudo,np.eye(15))
        K=posterior@H.T*(1-w)/pseudo
        A=np.eye(15)-K@H
        bound=conditional_linear_error_bound(prior,H,w,K,pseudo)
        np.testing.assert_allclose(bound,posterior,rtol=1e-11,atol=1e-11)
        vector=rng.normal(size=15); vector/=np.linalg.norm(vector)
        cross=np.linalg.cholesky(prior)@vector[:,None]*np.sqrt(q)*.99
        joint=np.block([[prior,cross],[cross.T,np.array([[pseudo]])]])
        assert np.linalg.eigvalsh(joint)[0]>=-1e-10
        for sign in (-1,1):
            actual=A@prior@A.T+K@K.T*pseudo+sign*(A@cross@K.T+K@cross.T@A.T)
            assert np.linalg.eigvalsh(bound-actual)[0]>=-1e-9
