import importlib.util
from pathlib import Path
import sys
import torch

p=Path(__file__).resolve().parents[1]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/core.py'
spec=importlib.util.spec_from_file_location('cpg_core',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def test_ik_roundtrip_and_real_limits():
    cfg=m.GaitConfig();t=torch.arange(0,8,.002)
    for v in [(0.,0.,0.),(.15,0.,0.),(-.15,.05,.25),(0.,0.,-.25)]:
        feet,_,_=m.walking_reference(t,torch.tensor(v).repeat(len(t),1),cfg)
        q=m.ik(feet)
        torch.testing.assert_close(m.fk(q),feet,atol=1e-6,rtol=1e-5)
        assert q[:,:,0].abs().max()<.9
        assert q[:,:,1].min()>-.4 and q[:,:,1].max()<3.3
        assert q[:,:,2].min()>-2.6 and q[:,:,2].max()<-.95


def test_clock_pair_sequence_and_marching():
    cfg=m.GaitConfig();t=torch.tensor([.5,1.2,1.84,2.5])
    _,_,pair,swing=m.schedule(t,cfg)
    assert pair.tolist()==[[True,False,False,True],[False,True,True,False]]*2
    assert swing.all()
    refs=[]
    for speed in [0.,.15]:
        refs.append(m.walking_reference(t,torch.tensor([speed,0.,0.]).repeat(4,1),cfg))
    torch.testing.assert_close(refs[0][1],refs[1][1])
    assert (refs[0][1].max(-1).values>.02).all()


def test_integrated_walking_matches_constant_command_reference():
    cfg=m.GaitConfig(swing_seconds=.2,clearance=.06)
    ref=m.WalkingReference(4,'cpu',cfg)
    velocity=torch.tensor([[0.,0.,0.],[.2,0.,0.],[-.1,.1,.3],[0.,-.2,-.3]])
    ready=torch.ones(4,dtype=torch.bool)
    for tick in range(1,501):
        time=torch.full((4,),tick*.02)
        actual=ref.update(time,velocity,ready)
        expected=m.walking_reference(time,velocity,cfg)
        torch.testing.assert_close(actual[0],expected[0],atol=3e-6,rtol=1e-5)
        torch.testing.assert_close(actual[1],expected[1])
        assert torch.equal(actual[2],expected[2])


def test_command_reversal_preserves_airborne_goal_and_planted_continuity():
    import copy
    cfg=m.GaitConfig(swing_seconds=.2,clearance=.06)
    ref=m.WalkingReference(1,'cpu',cfg)
    old=torch.tensor([[.15,0.,0.]])
    new=-old
    ready=torch.ones(1,dtype=torch.bool)
    for tick in range(1,27):ref.update(torch.tensor([tick*.02]),old,ready)
    unchanged=copy.deepcopy(ref)
    previous=ref.feet.clone()
    for tick in range(27,34):
        time=torch.tensor([tick*.02])
        actual,_,swing=ref.update(time,new,ready)
        control,_,_=unchanged.update(time,old,ready)
        torch.testing.assert_close(actual[swing],control[swing],atol=1e-7,rtol=0)
        planted=~swing
        torch.testing.assert_close((actual-previous)[planted][:,:2],(-new[:,:2]*.02).expand(int(planted.sum()),2),atol=1e-6,rtol=0)
        previous=actual


def test_integrated_walking_reset_reinitializes_only_requested_environments():
    cfg=m.GaitConfig(swing_seconds=.2)
    ref=m.WalkingReference(2,'cpu',cfg)
    velocity=torch.tensor([[.1,0.,0.],[0.,.1,.2]])
    ready=torch.ones(2,dtype=torch.bool)
    ref.update(torch.tensor([.02,.02]),velocity,ready)
    ref.reset(torch.tensor([0]))
    assert ref.initialized.tolist()==[False,True]
    actual,_,_=ref.update(torch.tensor([.02,.04]),-velocity,ready)
    expected,_,_=m.walking_reference(torch.tensor([.02,.04]),-velocity,cfg)
    torch.testing.assert_close(actual[0],expected[0])


def test_operator_command_acceleration_bounds_and_disabled_compatibility():
    current=torch.zeros(2,3)
    target=torch.tensor([[.2,.2,.3],[-.2,0.,-.3]])
    cfg=m.GaitConfig(linear_acceleration_limit=.075,yaw_acceleration_limit=.15)
    updated=m.advance_velocity(current,target,.02,cfg)
    torch.testing.assert_close(updated[:,:2].norm(dim=-1),torch.full((2,),.075*.02))
    torch.testing.assert_close(updated[:,2],torch.tensor([.15*.02,-.15*.02]))
    torch.testing.assert_close(m.advance_velocity(current,target,.02,m.GaitConfig()),target)


def test_ramped_integrated_reversals_remain_reachable_with_joint_limits():
    cfg=m.GaitConfig(swing_seconds=.2,clearance=.06,linear_acceleration_limit=.075,yaw_acceleration_limit=.15)
    ref=m.WalkingReference(8,'cpu',cfg)
    commands=torch.tensor([[0.,0.,0.],[.2,0.,0.],[-.2,0.,0.],[0.,.2,0.],[0.,-.2,0.],
                           [0.,0.,.3],[.14,.14,.3],[-.14,-.14,-.3]])
    velocity=commands.clone()
    # Actual soft limits: URDF limits with the configured0.9 range factor.
    lower=torch.tensor([[-1.0472,-1.5708,-2.7227]]*2+[[-1.0472,-.5236,-2.7227]]*2)
    upper=torch.tensor([[1.0472,3.4907,-.83776]]*2+[[1.0472,4.5379,-.83776]]*2)
    middle=(upper+lower)/2;half=.9*(upper-lower)/2
    for tick in range(1,3101):
        time=tick*.02
        target=commands[(torch.arange(8)+int((time+1.04)/4.1))%8]
        velocity=m.advance_velocity(velocity,target,.02,cfg)
        feet,_,_=ref.update(torch.full((8,),time),velocity,torch.ones(8,dtype=torch.bool))
        q=m.ik(feet)
        assert (q>=middle-half).all() and (q<=middle+half).all()
        torch.testing.assert_close(m.fk(q),feet,atol=1e-6,rtol=1e-5)


def test_reference_continuous_position_and_velocity_at_boundaries():
    cfg=m.GaitConfig()
    for center in [cfg.beat-cfg.swing_seconds,cfg.beat,cfg.cycle-cfg.swing_seconds,cfg.cycle]:
        t=torch.tensor([center-2e-4,center-1e-4,center,center+1e-4,center+2e-4],dtype=torch.float64)
        feet,_,_=m.walking_reference(t,torch.tensor([.15,.04,.2],dtype=torch.float64).repeat(5,1),cfg)
        vel=torch.diff(feet,dim=0)/1e-4
        assert torch.diff(vel,dim=0).abs().max()<.02


def test_contact_onset_confirmation_and_hysteresis():
    c=m.ContactEvents(1,'cpu');t=torch.tensor([1.])
    td,lo,onset=c.update(torch.ones(1,4)*10,t,.02);assert not td.any()
    td,_,onset=c.update(torch.ones(1,4)*10,t+.02,.02)
    assert td.all();torch.testing.assert_close(onset,torch.ones(1,4))
    _,lo,_=c.update(torch.ones(1,4)*3,t+.04,.02);assert not lo.any()
    _,lo,_=c.update(torch.ones(1,4),t+.06,.02);assert lo.all()


def test_inclusive_timing_window_tolerates_only_numeric_roundoff():
    error=torch.tensor([-.06000125,.06000125,-.0601,.0601,0.])
    assert m.timing_valid(error,.06).tolist()==[True,True,False,False,True]


def test_kick_endpoint_remains_raised_and_mirrors():
    cfg=m.GaitConfig();start=m.nominal_feet(2,'cpu',cfg);start[:,:,2]=0
    leg=torch.tensor([0,1]);target=m.kick_reference(start,start[:,:,1],torch.zeros(2),leg,torch.ones(2),cfg)
    assert target[0,0,2]==.05 and target[1,1,2]==.05
    torch.testing.assert_close(target[0,0,1]-start[0,0,1],torch.tensor(.08))
    torch.testing.assert_close(target[1,1,1]-start[1,1,1],torch.tensor(-.08))
    torch.testing.assert_close(target,m.kick_reference(start,start[:,:,1],torch.zeros(2),leg,torch.ones(2)*2,cfg))


def test_dance_touchdown_schedule_and_raised_transition():
    import importlib.util
    import types
    package=types.ModuleType('cpg_test_package');package.__path__=[]
    sys.modules['cpg_test_package']=package
    sys.modules['cpg_test_package.core']=m
    path=Path(__file__).resolve().parents[1]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/scheduler.py'
    spec=importlib.util.spec_from_file_location('cpg_test_package.scheduler',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    scheduler=module.DanceScheduler(m.GaitConfig())
    first=scheduler.phrase(0);second=scheduler.phrase(1)
    assert [round(x.target,6) for x in first[:3]]==[0.,.666667,1.333333]
    assert abs(first[3].start-5/3)<1e-9
    assert first[3].pair==(1,) and second[3].pair==(0,)
    assert first[3].pair[0] in second[0].pair and second[0].from_kick
    assert abs(scheduler.static_hold_before_next_step-.45)<1e-9
    assert first[0].start<0  # pre-roll, not a command issued on its touchdown beat
    # A finished trajectory without measured success must block the next dispatch.
    assert scheduler.dispatch(-.30,[True]*4).kind=='STEP'
    assert scheduler.dispatch(2/3-.30,[True]*4) is None


def test_load_compensation_matches_numerical_jacobian():
    q=m.ik(m.nominal_feet(3,'cpu',m.GaitConfig())).double()
    force=torch.randn(3,4,3,dtype=torch.double)*30
    expected=[]
    eps=1e-6
    for j in range(3):
        qp=q.clone();qm=q.clone();qp[:,:,j]+=eps;qm[:,:,j]-=eps
        derivative=(m.fk(qp)-m.fk(qm))/(2*eps)
        expected.append(-(derivative*force).sum(-1)/25)
    torch.testing.assert_close(m.gravity_position_offset(q,force,25),torch.stack(expected,-1),atol=1e-7,rtol=1e-6)


def test_discrete_dance_kick_hold_and_postkick_landing():
    import types
    package=types.ModuleType('cpg_dance_test');package.__path__=[]
    sys.modules['cpg_dance_test']=package;sys.modules['cpg_dance_test.core']=m
    path=Path(__file__).resolve().parents[1]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/dance.py'
    spec=importlib.util.spec_from_file_location('cpg_dance_test.dance',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    cfg=m.GaitConfig(swing_seconds=.2,clearance=.06)
    d=module.DanceReference(1,'cpu',cfg)
    contacts=torch.ones(1,4,dtype=torch.bool);success=torch.zeros(1,dtype=torch.bool)
    events=[];held=[];entry=None
    for _ in range(2200):
        feet,clear=d.update(.002,torch.ones(1,dtype=torch.bool),contacts,success,torch.zeros_like(success),torch.zeros(1,3))
        contacts=clear<1e-6
        success=((d.kind==d.STEP)&(d.phase>=1)&contacts.all(-1))|((d.kind==d.KICK)&(d.phase>=1))
        if d.dispatched[0]:
            events.append((int(d.kind[0]),float(d.time[0]),d.selected[0].tolist()))
            if entry is None and d.time[0]>2 and d.kind[0]==d.STEP:entry=feet[0,1].clone()
        if d.kind[0]==d.HOLD:held.append(feet[0,1].clone())
    assert [x[0] for x in events[:5]]==[1,1,1,2,1]
    assert abs(events[3][1]-5/3)<.003
    assert len(held)>200 and all(abs(float(x[2])+cfg.leg_height-cfg.kick_height)<1e-5 for x in held)
    assert entry is not None and abs(float(entry[2])+cfg.leg_height-cfg.kick_height)<1e-4
    assert events[4][2]==[False,True,True,False]


def test_encoder_velocity_jacobian():
    q=m.ik(m.nominal_feet(5,'cpu',m.GaitConfig())).double()
    qd=torch.randn_like(q);eps=1e-6
    numerical=(m.fk(q+eps*qd)-m.fk(q-eps*qd))/(2*eps)
    torch.testing.assert_close(m.foot_velocity(q,qd),numerical,atol=1e-7,rtol=1e-6)


def test_balanced_preroll_never_dispatches_expired_kick():
    module=sys.modules.get('cpg_dance_test.dance')
    if module is None:
        test_discrete_dance_kick_hold_and_postkick_landing()
        module=sys.modules['cpg_dance_test.dance']
    d=module.DanceReference(2,'cpu',m.GaitConfig(swing_seconds=.2,balanced_phrases=True))
    first=[None,None]
    for _ in range(40):
        d.update(.02,torch.ones(2,dtype=torch.bool),torch.ones(2,4,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),torch.zeros(2,3))
        for i in range(2):
            if d.dispatched[i] and first[i] is None:first[i]=(int(d.kind[i]),float(d.time[i]))
    assert first[0][0]==d.STEP and first[1][0]==d.STEP
    assert abs(first[0][1]+.2)<.021
    assert abs(first[1][1]-(4*d.cfg.beat-.2))<.021


def test_sagittal_observation_action_reflection():
    path=Path(__file__).resolve().parents[1]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/symmetry.py'
    spec=importlib.util.spec_from_file_location('cpg_symmetry_test',path)
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    for dim in (75,90):
        x=torch.randn(7,dim)
        torch.testing.assert_close(s.mirror_observation(s.mirror_observation(x)),x)
        start=42 if dim==75 else 57
        torch.testing.assert_close(s.mirror_observation(x)[:,start+14:start+16],x[:,start+14:start+16])
    q=m.ik(m.nominal_feet(7,'cpu',m.GaitConfig()))+torch.randn(7,4,3)*.05
    flat=q.transpose(1,2).flatten(1)
    mirrored=s.mirror_joints(flat).reshape(-1,3,4).transpose(1,2)
    feet=m.fk(q)[:,[1,0,3,2]].clone();feet[:,:,1]*=-1
    torch.testing.assert_close(m.fk(mirrored),feet)
    from tensordict import TensorDict
    obs=TensorDict({'policy':torch.randn(7,75),'critic':torch.randn(7,90)},batch_size=[7])
    actions=torch.randn(7,12);aug,act=s.augment(obs=obs,actions=actions)
    assert aug.batch_size==torch.Size([14]) and act.shape==(14,12)
    torch.testing.assert_close(act[:7],actions)


def test_reflection_polar_axial_and_kick_semantics():
    path=Path(__file__).resolve().parents[1]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/symmetry.py'
    spec=importlib.util.spec_from_file_location('cpg_symmetry_semantics',path)
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    x=torch.zeros(1,75);x[0,:6]=torch.arange(1,7.)
    x[0,58:62]=torch.tensor([1.,0,0,1.]);x[0,62:65]=torch.tensor([.1,.2,.3])
    x[0,70:74]=torch.tensor([0.,1,1,0])
    y=s.mirror_observation(x)
    torch.testing.assert_close(y[0,:6],torch.tensor([-1.,2,-3,4,-5,6]))
    torch.testing.assert_close(y[0,58:62],torch.tensor([0.,1,1,0]))
    torch.testing.assert_close(y[0,62:65],torch.tensor([.1,-.2,-.3]))
    torch.testing.assert_close(y[0,70:74],torch.tensor([0.,1,0,1]))


def test_recovery_strategies_preserve_clock_and_require_support():
    module=sys.modules.get('cpg_dance_test.dance')
    if module is None:
        test_discrete_dance_kick_hold_and_postkick_landing()
        module=sys.modules['cpg_dance_test.dance']
    for strategy in ('land_and_rejoin','land_and_stop','hold_and_stop'):
        cfg=m.GaitConfig(swing_seconds=.2,recovery_strategy=strategy)
        d=module.DanceReference(1,'cpu',cfg)
        d.time[:]=.2;d.kind[:]=d.HOLD;d.raised[:]=0
        d.feet[:,0,2]+=.05
        start=d.feet.clone();contacts=torch.tensor([[False,True,True,True]])
        ready=torch.ones(1,dtype=torch.bool);zero=torch.zeros_like(ready)
        for i in range(20):
            d.update(.02,ready,contacts,zero,ready if i==0 else zero,torch.zeros(1,3))
            assert d.kind[0]==d.RECOVERY
        assert float(d.time[0])>.59  # recovery never resets or pauses music
        if strategy=='hold_and_stop':torch.testing.assert_close(d.feet,start)
        else:torch.testing.assert_close(d.feet,m.nominal_feet(1,'cpu',cfg))
        d.update(.02,ready,torch.ones_like(contacts),zero,zero,torch.zeros(1,3))
        assert int(d.kind[0])==(d.SUPPORT if strategy=='land_and_rejoin' else d.RECOVERY)


def test_mixed_hold_curriculum_is_bilateral_and_preserves_dance_dispatch():
    module=sys.modules.get('cpg_dance_test.dance')
    if module is None:
        test_discrete_dance_kick_hold_and_postkick_landing()
        module=sys.modules['cpg_dance_test.dance']
    cfg=m.GaitConfig(swing_seconds=.2,extended_hold_fraction=.25)
    d=module.DanceReference(200,'cpu',cfg)
    assert d.extended_hold.sum()==50
    assert torch.equal(d.extended_hold[::2],d.extended_hold[1::2])
    d.time[:]=4*cfg.beat-cfg.swing_seconds-.01
    d.kind[:]=d.HOLD;d.raised[:]=1;d.event_id[:]=3
    contacts=torch.ones(200,4,dtype=torch.bool);contacts[:,1]=False
    ready=torch.ones(200,dtype=torch.bool);zero=torch.zeros_like(ready)
    d.update(.02,ready,contacts,zero,zero,torch.zeros(200,3))
    assert (d.kind[d.extended_hold]==d.HOLD).all()
    assert (d.kind[~d.extended_hold]==d.STEP).all()


def test_planted_support_shift_preserves_kick_endpoint_and_long_hold():
    module=sys.modules.get('cpg_dance_test.dance')
    if module is None:
        test_discrete_dance_kick_hold_and_postkick_landing()
        module=sys.modules['cpg_dance_test.dance']
    cfg=m.GaitConfig(swing_seconds=.2,mode='kick_hold',balanced_phrases=True,
                     kick_shift_back=.03,kick_shift_side=.015)
    d=module.DanceReference(2,'cpu',cfg)
    contacts=torch.ones(2,4,dtype=torch.bool);success=torch.zeros(2,dtype=torch.bool)
    held=torch.zeros(2,dtype=torch.long);shift_at_kick=None
    velocity=torch.tensor([[.05,0.,.1],[-.05,.02,-.1]])
    for _ in range(500):
        feet,clear=d.update(.02,torch.ones(2,dtype=torch.bool),contacts,success,
                            torch.zeros_like(success),velocity)
        contacts=clear<1e-6
        success=((d.kind==d.STEP)&(d.phase>=1)&contacts.all(-1))|((d.kind==d.KICK)&(d.phase>=1))
        if (d.kind==d.KICK).all() and shift_at_kick is None:
            shift_at_kick=d.support_shift.clone()
            torch.testing.assert_close(shift_at_kick[:,0],torch.full((2,),-.03))
        for env in range(2):
            if d.kind[env]!=d.HOLD:continue
            held[env]+=1;leg=int(d.raised[env])
            target=torch.tensor([.1934,.222 if leg==0 else -.222,-.25])
            torch.testing.assert_close(feet[env,leg],target,atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(d.endpoint_body[env,leg],target,atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(d.support_shift[env],shift_at_kick[env])
        torch.testing.assert_close(m.fk(m.ik(feet)),feet,atol=2e-6,rtol=1e-5)
    assert (held>200).all()


def test_kick_endpoint_excludes_travel_and_yaw_stride():
    module=sys.modules.get('cpg_dance_test.dance')
    if module is None:
        test_discrete_dance_kick_hold_and_postkick_landing()
        module=sys.modules['cpg_dance_test.dance']
    d=module.DanceReference(4,'cpu',m.GaitConfig(swing_seconds=.2))
    d.time[:]=5/3-.01
    velocity=torch.tensor([[.1,0.,0.],[-.1,0.,0.],[0.,0.,.2],[0.,0.,-.2]])
    d.update(.02,torch.ones(4,dtype=torch.bool),torch.ones(4,4,dtype=torch.bool),
             torch.zeros(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool),velocity)
    assert (d.kind==d.KICK).all()
    torch.testing.assert_close(d.endpoint_body[:,1],torch.tensor([.1934,-.222,-.25]).expand(4,3))


def test_equivariant_actor_gradient_and_export():
    path=Path(__file__).resolve().parents[1]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg/symmetry.py'
    spec=importlib.util.spec_from_file_location('cpg_symmetric_actor_test',path)
    s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
    base=torch.nn.Sequential(torch.nn.Linear(75,32),torch.nn.ELU(),torch.nn.Linear(32,12))
    model=s.SagittalActor(base);x=torch.randn(8,75)
    torch.testing.assert_close(model(s.mirror_observation(x)),s.mirror_joints(model(x)))
    loss=model(x).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters())
    compiled=torch.jit.script(model)
    torch.testing.assert_close(compiled(x),model(x))


def test_balanced_preroll_goals_are_mirrored_and_reset_without_stale_time():
    module=sys.modules.get('cpg_dance_test.dance')
    if module is None:
        test_discrete_dance_kick_hold_and_postkick_landing()
        module=sys.modules['cpg_dance_test.dance']
    cfg=m.GaitConfig(swing_seconds=.2,balanced_phrases=True)
    d=module.DanceReference(2,'cpu',cfg)
    for _ in range(2):
        torch.testing.assert_close(d.target_time-d.time,torch.full((2,),cfg.beat))
        phase=2*torch.pi*((d.initial_front+1)*cfg.beat+d.time-d.target_time)/cfg.cycle
        torch.testing.assert_close(phase.cos(),torch.tensor([1.,-1.]),atol=1e-6,rtol=1e-5)
        assert d.initial_front.tolist()==[0,1]
        d.target_time[:]=100.;d.start_time[:]=99.;d.phase[:]=1.
        d.reset(torch.arange(2))
