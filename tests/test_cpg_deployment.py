"""Deployment contracts and parity against the immutable pre-port controller."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.cpg.runtime import DanceRuntime, SensorFrame, core, control, ROOT, load_selection
from deploy.cpg.inputs import stick_velocity, WirelessInput

SESSION = ROOT/'logs/conga_cpg_session_20261101'
SELECTION_PATH = ROOT/'exports/go2_cpg_quad_sdk_candidate/assets/selection.json'
SELECTION = load_selection(SELECTION_PATH) if SELECTION_PATH.exists() else None


@pytest.fixture
def runtime():
    if SELECTION is None:pytest.skip('Selected policy artifacts are not installed')
    torch.set_num_threads(1)
    return DanceRuntime(SELECTION['deploy_metadata'], SELECTION['onnx_export'])


def frame(runtime, time=0.):
    q = np.empty(12)
    q[runtime.map] = runtime.defaults
    return SensorFrame(time, q, np.zeros(12), np.array([1.,0,0,0]), np.zeros(3), np.full(4, 40.))


def test_unitree_mapping_neutral_and_bounds():
    np.testing.assert_array_equal(stick_velocity(.03,-.03,.07), [0,0,0])
    np.testing.assert_allclose(stick_velocity(-1,1,-1), [.05,.04,.1])
    np.testing.assert_allclose(stick_velocity(1,-1,1), [-.05,-.04,-.1])
    with pytest.raises(ValueError): stick_velocity(float('nan'),0,0)
    np.testing.assert_allclose(stick_velocity(-1,1,-1,limits=(.08,.06,.15)), [.08,.06,.15])
    for limits in ((-.1,.04,.1),(.1,float('nan'),.1),(.1,.1)):
        with pytest.raises(ValueError):stick_velocity(0,1,0,limits=limits)


def test_wireless_timeout_reconnect_and_stop(tmp_path):
    path = tmp_path/'input.json'
    source = WirelessInput(path)
    np.testing.assert_array_equal(source.read(now=10), [0,0,0])
    path.write_text(json.dumps(dict(monotonic_seconds=10,lx=0,ly=1,rx=0,keys=0)))
    assert source.read(now=10.1)[0] == .05
    np.testing.assert_array_equal(source.read(now=10.3), [0,0,0])
    path.write_text(json.dumps(dict(monotonic_seconds=10.4,lx=0,ly=1,rx=0,keys=512)))
    source.read(now=10.5)
    assert source.stop
    path.write_text('{}')
    source.read(now=11)
    assert source.stop


def test_sensor_fault_latches_and_does_not_advance_clock(runtime):
    runtime.observe(frame(runtime), [0,0,0])
    clock = runtime.controller.time.clone()
    with pytest.raises(ValueError, match='Stale'):
        runtime.observe(frame(runtime), [0,0,0])
    torch.testing.assert_close(runtime.controller.time, clock)
    with pytest.raises(RuntimeError, match='latched'):
        runtime.observe(frame(runtime,.02), [0,0,0])


def test_observation_joint_mapping_and_bounded_action(runtime):
    sample = frame(runtime)
    sample.qd = np.arange(12)*.1
    observation = runtime.observe(sample, [0,0,0])
    assert observation.shape == (1,75)
    np.testing.assert_allclose(observation[0,18:30], sample.qd[runtime.map]*.05, atol=1e-7)
    target = runtime.act(observation)
    np.testing.assert_allclose(target[runtime.map], runtime.controller.q_ref[0], atol=1e-7)
    runtime.controller.ready[:] = True
    runtime.policy = SimpleNamespace(run=lambda *args: [np.full((1,12), 1e6, np.float32)])
    target = runtime.act(observation)[runtime.map]
    assert np.max(np.abs(target-runtime.controller.q_ref.numpy()[0])) <= .120001
    assert (target <= runtime.limits[:,1]).all() and (target >= runtime.limits[:,0]).all()


def test_zero_command_keeps_clock_running(runtime):
    for i in range(80):
        runtime.observe(frame(runtime, i*.02), [0,0,0])
    assert runtime.controller.ready.item()
    assert runtime.controller.time.item() > -runtime.controller.cfg.gait.beat


def test_walking_runtime_uses_shared_walking_controller():
    path=ROOT/'exports/go2_cpg_demo_walk_20261101/assets/selection.json'
    if not path.exists():pytest.skip('Walking checkpoint unavailable')
    selected=load_selection(path)
    walk=DanceRuntime(selected['deploy_metadata'],selected['onnx_export'])
    assert type(walk.controller) is control.CPGController
    for i in range(80):
        observation=walk.observe(frame(walk,i*.02),[.1,0,0])
        target=walk.act(observation)
        assert observation.shape==(1,75) and np.isfinite(target).all()
    assert walk.controller.ready.item()
    assert not hasattr(walk.controller,'dance')


def legacy_class():
    # Execute only archived class definitions; no Isaac imports or source edits.
    class Term:
        def __init__(self,cfg,env): self.cfg,self._env=cfg,env
        @property
        def num_envs(self): return self._env.num_envs
        @property
        def device(self): return self._env.device
    namespace = {**vars(core), 'math':math, 'torch':torch, 'CommandTerm':Term,
                 'quat_from_euler_xyz':control.quat_from_euler_xyz,
                 'quat_apply_inverse':control.quat_apply_inverse,
                 'DanceReference':control.DanceReference,'support_forces':control.support_forces}
    directory = SESSION/'source_final/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg'
    for file, name in [('adapter.py','CPGCommand'),('dance_adapter.py','CPGDanceCommand')]:
        node = next(n for n in ast.parse((directory/file).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]), str(directory/file),'exec'), namespace)
    return namespace['CPGDanceCommand']


def test_shared_controller_matches_archived_controller(runtime):
    path=ROOT/'logs/cpg_mujoco_20261101/final_zero_FL/traces.npz'
    if not path.exists():pytest.skip('Run the MuJoCo evaluation before the recorded parity test')
    traces = dict(np.load(path))
    env = copy.deepcopy(runtime.env)
    old = legacy_class()(runtime.controller.cfg, env)
    old.set_velocity([0,0,0]);old.reset()
    for i in range(len(traces['time'])):
        sample = SensorFrame(float(traces['sensor_time'][i,0]), traces['sensor_q'][i,0], traces['sensor_qd'][i,0],
                             traces['sensor_quaternion'][i,0], traces['angular_velocity'][i,0], traces['sensor_force'][i,0])
        runtime.observe(sample, traces['operator_velocity'][i,0])
        env.common_step_counter = runtime.env.common_step_counter
        env.scene['robot'].data = copy.deepcopy(runtime.robot.data)
        env.scene['contact_forces'].data = copy.deepcopy(runtime.sensor.data)
        old.set_velocity(traces['operator_velocity'][i,0]);old._update_command()
        for name in ('q_ref','command','time','active','recovery','ready','kick_success','kick_failure'):
            torch.testing.assert_close(getattr(old,name),getattr(runtime.controller,name),rtol=0,atol=0)
        for name in old.last_records:
            torch.testing.assert_close(old.last_records[name],runtime.controller.last_records[name],rtol=0,atol=0)


def test_onnx_matches_exported_equivariant_jit(runtime):
    path=ROOT/'logs/cpg_mujoco_20261101/final_zero_FL/traces.npz'
    if not path.exists():pytest.skip('Run the MuJoCo evaluation before the recorded parity test')
    traces = dict(np.load(path))
    policy = torch.jit.load(SELECTION['jit_export'],map_location='cpu').eval()
    with torch.inference_mode():
        for observation in traces['observation'][::31]:
            actual = runtime.policy.run(None,{runtime.input_name:observation})[0]
            expected = policy(torch.from_numpy(observation)).numpy()
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)


def test_mujoco_velocity_is_at_the_foot_origin():
    import mujoco
    path=ROOT/'logs/cpg_mujoco_20261101/final_zero_FL/scene.xml'
    if not path.exists():pytest.skip('MuJoCo model is not installed')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    data.qvel[:]=np.linspace(-.3,.4,model.nv)
    mujoco.mj_forward(model,data)
    for leg in ('FL','FR','RL','RR'):
        body=model.body(leg+'_foot').id
        jac=np.zeros((3,model.nv));angular=jac.copy();velocity=np.zeros(6)
        mujoco.mj_jacBody(model,data,jac,angular,body)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,body,velocity,0)
        np.testing.assert_allclose(velocity[3:],jac@data.qvel,atol=1e-12)


def test_quad_sdk_permutation_and_reject_unknown_names():
    from deploy.cpg.quad_sdk import decode_state,encode_command,QUAD_NAMES,QUAD_TO_SDK
    joints=SimpleNamespace(name=list(QUAD_NAMES)[::-1],position=list(range(12))[::-1],velocity=list(range(12))[::-1])
    imu=SimpleNamespace(orientation=SimpleNamespace(w=1.,x=0.,y=0.,z=0.),angular_velocity=SimpleNamespace(x=0.,y=0.,z=0.))
    contact=SimpleNamespace(foot_force_raw=[10,20,30,40])
    sensor=decode_state(joints,imu,contact,0.)
    assert [sensor['q'][i] for i in QUAD_TO_SDK]==list(range(12))
    assert sensor['foot_force']==[10,30,20,40]
    message=encode_command(sensor['q'],[25]*12,[.5]*12,
                           lambda:SimpleNamespace(leg_commands=[]),lambda:SimpleNamespace(motor_commands=[]),SimpleNamespace)
    assert [m.pos_setpoint for leg in message.leg_commands for m in leg.motor_commands]==list(range(12))
    assert all(m.torque_ff==0 for leg in message.leg_commands for m in leg.motor_commands)
    joints.name[0]='unknown'
    with pytest.raises(ValueError):decode_state(joints,imu,contact,0.)


def test_training_feedback_matches_deployment_estimator(runtime):
    from deploy.cpg.runtime import FlatGroundEstimator,SDK_NAMES,rotation
    rng=np.random.default_rng(9102)
    for count in range(5):
        sample=frame(runtime)
        sample.q+=rng.uniform(-.02,.02,12)
        sample.qd=rng.uniform(-.2,.2,12)
        sample.gyro=rng.uniform(-.1,.1,3)
        sample.foot_force[:]=0;sample.foot_force[:count]=40
        estimator=FlatGroundEstimator()
        positions,speeds,gravity,_=estimator.update(sample)
        order=[SDK_NAMES.index(f'{leg}_{joint}_joint') for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
        sole,velocity,height=core.estimated_foot_feedback(torch.tensor(sample.q[order],dtype=torch.float32).reshape(1,4,3),
            torch.tensor(sample.qd[order],dtype=torch.float32).reshape(1,4,3),torch.tensor(sample.gyro,dtype=torch.float32)[None],
            torch.tensor(gravity,dtype=torch.float32)[None],torch.tensor(estimator.contact)[None],torch.tensor([.322]))
        np.testing.assert_allclose(sole[0].numpy(),positions[:,2]-.022,atol=1e-7)
        np.testing.assert_allclose(velocity[0].numpy()@rotation(sample.quaternion).T,speeds,atol=1e-7)


def test_deliberate_stand_freezes_after_confirmed_support_but_sensors_still_checked(runtime):
    for i in range(90):
        o=runtime.observe(frame(runtime,i*.02),[0,0,0]);runtime.act(o)
    assert not runtime.stand_requested and not runtime.standing
    runtime.request_stand()
    for i in range(90,490):
        o=runtime.observe(frame(runtime,i*.02),[.1,0,.1]);runtime.act(o)
        if runtime.standing:break
    assert runtime.standing
    target=runtime.last_target.copy();clock=runtime.controller.time.clone()
    for j in range(i+1,i+11):
        o=runtime.observe(frame(runtime,j*.02),[.1,0,.1])
        np.testing.assert_array_equal(runtime.act(o),target)
        torch.testing.assert_close(runtime.controller.time,clock)
    with pytest.raises(ValueError,match='Stale'):runtime.observe(frame(runtime,j*.02),[0,0,0])
    with pytest.raises(RuntimeError,match='latched'):runtime.observe(frame(runtime,(j+1)*.02),[0,0,0])
