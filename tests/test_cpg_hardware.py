import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import pytest
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from deploy.cpg.safety import CommandEnvelope,ForceCalibration,SensorGate,JsonLineReader
from deploy.cpg.runtime import core
spec=importlib.util.spec_from_file_location('hardware',R/'scripts/cpg_quad_hardware.py');hw=importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)

def test_restricted_commands_and_legacy_default():
    e=CommandEnvelope(dict(command_min=[0,0,-.15],command_max=[.12,0,.15]))
    assert e.apply([-1,1,-1])==[0,0,-.15]
    assert e.apply([1,-1,1])==[.12,0,.15]
    assert e.apply([0,0,0])==[0,0,0]
    assert CommandEnvelope({}).apply([-1,1,1])==[-.05,.04,.1]
    for v in ([0,0],[0,float('nan'),0],[float('inf'),0,0]):
        with pytest.raises(ValueError):e.apply(v)

def test_per_foot_calibration_and_gate():
    f=ForceCalibration(dict(order='FL,RL,FR,RR',measured=True,scale=[1,2,3,4],offset=[10,20,30,40]),True)
    assert f.apply([12,22,32,42])==[2,4,6,8]
    assert f.apply([0,0,0,0])==[0,0,0,0]
    cfg=json.loads((R/'deploy/cpg/hardware_profile.template.json').read_text())
    sel=dict(onnx_sha256='abc',command_min=[0,0,-.15],command_max=[.12,0,.15])
    hw.validate_profile(cfg,sel,False)
    with pytest.raises(ValueError,match='hardware_enabled'):hw.validate_profile(cfg,sel,True)
    cfg.update(hardware_enabled=True,contact_calibrated=True,timing_verified=True,policy_sha256='abc')
    with pytest.raises(ValueError,match='measured'):hw.validate_profile(cfg,sel,True)
    cfg['force_calibration']['measured']=True
    hw.validate_profile(cfg,sel,True)
    cfg['policy_sha256']='wrong'
    with pytest.raises(ValueError,match='hash'):hw.validate_profile(cfg,sel,True)

def test_sensor_source_stamps_must_advance_and_agree():
    gate=SensorGate();samples={k:(10.,20.) for k in ['joints','imu','contact']}
    gate.check(samples,10.,20.)
    for reason,s,r,w in [('old',samples,10.1,20.1),('future',{k:(11.,20.) for k in samples},10,20),('skew',{**samples,'imu':(9.98,20)},10,20)]:
        with pytest.raises(ValueError):gate.check(s,r,w)
    gate=SensorGate();gate.check(samples,10.,20.)
    with pytest.raises(ValueError,match='Regressing'):gate.check({k:(9.999,20.01) for k in samples},10.01,20.01)

def test_worker_partial_line_timeout_and_eof():
    rd,wr=os.pipe()
    with os.fdopen(rd,'rb',buffering=0) as stream:
        reader=JsonLineReader(stream);os.write(wr,b'{"ready":')
        start=time.monotonic()
        with pytest.raises(TimeoutError):reader.read(.01)
        assert time.monotonic()-start<.1
        os.write(wr,b'true}\n');assert reader.read(.1)=={'ready':True}
        os.close(wr)
        with pytest.raises(EOFError):reader.read(.1)

def test_restricted_training_sampler_has_no_lateral_or_backward():
    x=torch.randn(1000,3);c=core.GaitConfig(training_backward_fraction=0,training_lateral_scale=0,training_turn_fraction=.25)
    out=core.training_direction_bias(x,c)
    assert (out[:,0]>=0).all() and (out[:,1]==0).all()
    assert (out[:,2]>0).any() and (out[:,2]<0).any()

def test_support_margin_does_not_reward_planted_swing_leg():
    path=R/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/dance_adapter.py'
    fn=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='dance_reward')
    ns=dict(torch=torch);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    # Nontrivial sensor permutation catches asset-body vs sensor-body indexing.
    forces=torch.zeros(1,6,3);ids=[4,1,5,2];forces[:,ids,2]=20
    selected=torch.tensor([[True,False,False,False]])
    c=NS(sync=lambda:None,sensor=NS(data=NS(net_forces_w=forces)),sensor_ids=ids,foot_ids=[10,11,12,13],ready=torch.tensor([True]),dance=NS(kind=torch.tensor([2]),KICK=2,HOLD=3,STEP=1,selected=selected,raised=torch.tensor([-1])))
    env=NS(command_manager=NS(get_term=lambda _:c))
    reward=ns['dance_reward']
    assert reward(env,'support_margin').item()==0
    forces[:,4,2]=0 # selected kicking leg: never penalized for being airborne
    assert reward(env,'support_margin').item()==0
    forces[:,1,2]=0
    assert reward(env,'support_margin').item()==pytest.approx(1/3)
    c.dance.kind[:]=1;c.dance.raised[:]=0
    assert reward(env,'support_margin').item()==pytest.approx(1/3)
    c.dance.raised[:]=-1
    assert reward(env,'support_margin').item()==0


def test_coupled_command_cap_is_float_and_idempotent():
    e=CommandEnvelope(dict(command_min=[0,0,-.1],command_max=[.1,0,.1],forward_at_max_yaw=.08))
    for command,expected in [([1,0,1],[.08,0,.1]),([.1,0,.05],[.09,0,.05]),([.02,0,-1],[.02,0,-.1]),([-1,1,0],[0,0,0])]:
        out=e.apply(command)
        assert out==pytest.approx(expected)
        assert e.apply(out)==pytest.approx(out)
        assert all(type(v) is float for v in out) # generated ROS doubles require floats


def test_contact_recording_needs_advancing_ticks_and_distinct_channels(tmp_path):
    spec=importlib.util.spec_from_file_location('calibration',R/'scripts/cpg_contact_calibration.py')
    calibration=importlib.util.module_from_spec(spec);spec.loader.exec_module(calibration)
    path=tmp_path/'raw.jsonl'
    rows=[dict(tick=(0xffffff00+i*2)&0xffffffff,foot_force_sdk=[10,20,30,40]) for i in range(300)]
    path.write_text('\n'.join(map(json.dumps,rows)))
    _,force=calibration.load(path)
    assert force[0].tolist()==[20,40,10,30]
    rows[200]['tick']=rows[199]['tick']-2
    path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='Regressing'):calibration.load(path)
    path.write_text('\n'.join(json.dumps(rows[0]) for _ in range(300)))
    with pytest.raises(ValueError,match='advancing'):calibration.load(path)


def test_v2_audit_detects_support_loss_after_old_window(tmp_path):
    import subprocess,numpy as np
    t=np.arange(-.2,.121,.02);contact=np.ones((len(t),1,4),bool);sole=np.zeros_like(contact,dtype=float)
    swing=(t>=-.16-1e-8)&(t<-.001)
    for leg in (0,3):contact[swing,0,leg]=False;sole[swing,0,leg]=.03
    late=(t>=.06-1e-8)&(t<=.08+1e-8);contact[late,0,1]=False
    np.savez(tmp_path/'traces.npz',time=t[:,None],contact=contact,sole=sole,ready=np.ones((len(t),1),bool),groups=np.array([0]))
    for script in ('analyze_cpg_contacts.py','analyze_cpg_contacts_v2.py'):
        subprocess.run([sys.executable,str(R/'scripts'/script),str(tmp_path),'--dance','--swing','.2','--tolerance','.06'],check=True,capture_output=True)
    old=json.loads((tmp_path/'contact_audit.json').read_text())['cases']['0']
    new=json.loads((tmp_path/'contact_audit_v2.json').read_text())['cases']['0']
    assert old['expected']==new['expected']==1
    assert old['correct']==1 and old['support_loss']==0
    assert new['correct']==0 and new['support_loss']==1


def test_control_clock_rejects_accumulated_slowness_without_skipping_phase():
    from deploy.cpg.safety import ControlClockGate
    clock=ControlClockGate()
    for tick in range(100):
        clock.check(100+tick*.02+(.003 if tick%2 else 0),tick)
    slow=ControlClockGate()
    # Each 22ms tick is individually within the interval guard, but sustained
    # slowness would change tempo; cumulative phase error must catch it.
    with pytest.raises(ValueError,match='Accumulated control clock drift'):
        for tick in range(30):slow.check(100+tick*.022,tick)
