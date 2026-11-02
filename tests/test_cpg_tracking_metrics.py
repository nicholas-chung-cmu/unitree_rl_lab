"""Velocity reporting must not average the kick pause into travel speed."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/cpg_tracking_metrics.py'
spec=importlib.util.spec_from_file_location('tracking_metrics_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_pause_and_reset_never_contribute_to_velocity_statistics():
    x={'travel_active':np.array([[1],[0],[0],[1],[1]],bool),
       'ready':np.ones((5,1),bool),'reset':np.array([[0],[0],[0],[0],[1]],bool),
       'tracking_velocity':np.tile([.1,0,0],(5,1,1)),
       'operator_velocity':np.tile([.1,0,0],(5,1,1)),
       'body_velocity_local':np.array([[[.1,0,0]],[[10,10,0]],[[-10,-10,0]],[[.1,0,0]],[[10,10,0]]]),
       'angular_velocity':np.zeros((5,1,3))}
    result=module.tracking_metrics(x)
    assert result['active_samples']==2 and result['excluded_samples']==3
    assert result['xy_rmse_m_s']==0
    assert abs(result['nonzero_directional_gain']-1)<1e-12
    case=next(iter(result['cases'].values()))
    np.testing.assert_allclose(case['actual_mean'],[.1,0,0])
