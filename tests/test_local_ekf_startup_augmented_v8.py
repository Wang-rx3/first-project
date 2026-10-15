"""Independent 30-state reference, conditioning BOTH sides of the clone."""
import numpy as np
import pytest

from src.analysis.local_ekf_startup_filter_v8 import StartupLocalEKF
from src.simulation.ekf import LocalEKF


@pytest.mark.parametrize("seed",range(30))
def test_real_augmented_bias_then_vo_joseph_reference(seed):
    rng=np.random.default_rng(8000+seed)
    prior=np.zeros((15,15));a=rng.normal(size=(9,9));b=rng.normal(size=(6,6))
    prior[:9,:9]=a@a.T+np.eye(9);prior[9:,9:]=b@b.T+np.eye(6)
    state=rng.normal(size=15);state[3:6]*=.01
    candidate=StartupLocalEKF(state.copy());candidate.covariance=prior.copy()
    observation={"accel":rng.normal(size=3)*.05,"gyro":rng.normal(size=3)*.02}
    temp=LocalEKF(state.copy());temp.covariance=np.zeros((15,15));temp.predict(observation,.2)
    Q=temp.covariance;F=np.eye(15);F[:3,3:6]=np.eye(3)*.2
    predicted=state.copy();predicted[:3]+=state[3:6]*.2
    mean=np.r_[predicted,state];P=np.block([[F@prior@F.T+Q,F@prior],[prior@F.T,prior]])
    Hbias=np.zeros((6,30));Hbias[:,9:15]=np.eye(6)
    Rbias=np.diag([.035**2]*3+[.01**2]*3);z=np.r_[observation["accel"],observation["gyro"]]
    S=Hbias@P@Hbias.T+Rbias;K=P@Hbias.T@np.linalg.inv(S)
    mean+=K@(z-Hbias@mean);A=np.eye(30)-K@Hbias
    P=A@P@A.T+K@Rbias@K.T
    candidate.predict(observation,.2)
    np.testing.assert_allclose(candidate.state,mean[:15],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(candidate.covariance,P[:15,:15],rtol=1e-11,atol=1e-11)
    assert np.linalg.eigvalsh(P)[0]>=-1e-10
    # Past bias mean/cov and cross blocks really changed; do not reconstruct
    # an invalid augmented matrix from an untouched past bias cross block.
    assert not np.allclose(P[15+9:,15+9:],prior[9:,9:])
    H=np.zeros((3,30));H[:,:3]=np.eye(3);H[:,15:18]=-np.eye(3)
    Rvo=np.eye(3)*.015**2;increment=rng.normal(size=3)*.004
    Svo=H@P@H.T+Rvo;Kvo=P@H.T@np.linalg.inv(Svo)
    selector=np.zeros((3,15));selector[:,:3]=np.eye(3)
    old_cross=prior@F.T
    nativeS=selector@(candidate.covariance+prior-old_cross-old_cross.T)@selector.T+Rvo
    nativeK=(candidate.covariance@selector.T-old_cross.T@selector.T)@np.linalg.inv(nativeS)
    np.testing.assert_allclose(nativeS,Svo,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(nativeK,Kvo[:15],rtol=1e-11,atol=1e-11)
    mean+=Kvo@(increment-H@mean);A=np.eye(30)-Kvo@H
    P=A@P@A.T+Kvo@Rvo@Kvo.T
    context=candidate.last_prediction_context
    assert candidate.update_visual_delta(increment,Rvo,previous_state=context["previous_state"],
        previous_covariance=context["previous_covariance"],transition_matrix=context["transition_matrix"],dt_s=.2)
    np.testing.assert_allclose(candidate.state,mean[:15],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(candidate.covariance,P[:15,:15],rtol=1e-10,atol=1e-10)
    assert np.linalg.eigvalsh(P)[0]>=-1e-10
