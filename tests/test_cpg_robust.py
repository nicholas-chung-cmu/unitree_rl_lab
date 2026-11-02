import importlib
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy.cpg.runtime import load_control_modules
load_control_modules()
S=importlib.import_module('_go2_control.cpg.sensing')
M=importlib.import_module('_go2_control.cpg.symmetry')


def test_sensor_delay_and_partial_reset_never_reuse_old_episode():
    pipe=S.SensorPipeline(2,'cpu',{'delay_ticks':1})
    pipe.lag[:]=1
    def update(value):return pipe.update(torch.full((2,12),float(value)),torch.zeros(2,12),torch.zeros(2,3),torch.tensor([[0.,0.,-1.]]).repeat(2,1),torch.ones(2,4)*40)[0]
    assert (update(1)==1).all()
    assert (update(2)==1).all()
    pipe.reset(torch.tensor([0]));pipe.lag[:]=1
    q=update(3)
    assert (q[0]==3).all() and (q[1]==2).all()


def test_sensor_corruption_does_not_modify_physical_truth():
    torch.manual_seed(1);pipe=S.SensorPipeline(8,'cpu',{'q_noise':.002,'force_bias':2,'force_scale':.1})
    q=torch.zeros(8,12);g=torch.tensor([[0.,0.,-1.]]).repeat(8,1);f=torch.ones(8,4)*40
    out=pipe.update(q,q.clone(),torch.zeros(8,3),g,f)
    assert (q==0).all() and (f==40).all()
    assert out[0].abs().max()<=.002 and out[0].abs().max()>0
    torch.testing.assert_close(out[3].norm(dim=-1),torch.ones(8))


def test_81_mirror_and_zero_input_warm_start():
    torch.manual_seed(4)
    a=torch.nn.Sequential(torch.nn.Linear(75,16),torch.nn.ELU(),torch.nn.Linear(16,12))
    b=torch.nn.Sequential(torch.nn.Linear(81,16),torch.nn.ELU(),torch.nn.Linear(16,12))
    with torch.no_grad():
        b[0].weight.zero_();b[0].weight[:,:75]=a[0].weight;b[0].bias.copy_(a[0].bias);b[2].load_state_dict(a[2].state_dict())
    x=torch.randn(30,81)
    torch.testing.assert_close(M.mirror_observation(M.mirror_observation(x)),x)
    aa,bb=M.SagittalActor(a),M.SagittalActor(b)
    torch.testing.assert_close(aa(x[:,:75]),bb(x),atol=1e-7,rtol=1e-6)
    y=M.mirror_observation(x)
    torch.testing.assert_close(y[:,75],x[:,75]);torch.testing.assert_close(y[:,76],-x[:,76])
    torch.testing.assert_close(y[:,77:81],x[:,[78,77,80,79]])


def test_extra_feedback_uses_support_kinematics_not_truth_velocity():
    from types import SimpleNamespace
    core,control=load_control_modules()
    q=core.ik(core.nominal_feet(1,'cpu',core.GaitConfig()))
    basis=torch.eye(3)
    jac=torch.stack([core.foot_velocity(q,basis[i].expand_as(q)) for i in range(3)],dim=-1)
    desired=torch.tensor([-.1,-.05,0.]).expand(1,4,3)
    qd=torch.linalg.solve(jac,desired.unsqueeze(-1)).squeeze(-1)
    qd[:,0]=100  # airborne foot must not influence support-derived body velocity
    c=object.__new__(control.CPGController);c._measured_data=None;c.joint_ids=list(range(12))
    c.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q.flatten(1),joint_vel=qd.flatten(1),root_ang_vel_b=torch.zeros(1,3),root_lin_vel_b=torch.full((1,3),999.)))
    c.events=SimpleNamespace(contact=torch.tensor([[False,True,True,True]]))
    torch.testing.assert_close(c.measured_features,torch.tensor([[.1,.05,0,1,1,1.]]),atol=1e-6,rtol=1e-5)
