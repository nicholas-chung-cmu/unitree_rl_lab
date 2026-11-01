"""One bounded CPG physics, learning or evaluation child; supervised externally."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/unitree_rl_lab'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--steps',type=int,default=1100)
p.add_argument('--iterations',type=int,default=0)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--settings',type=Path)
p.add_argument('--seed',type=int,default=9041)
p.add_argument('--video',action='store_true')
p.add_argument('--task',default='Unitree-Go2-CPG-Walk')
p.add_argument('--component-tests',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.video:a.enable_cameras=True
app=AppLauncher(a).app
import gymnasium as gym
import torch,numpy as np
import signal
stop_requested=False
def request_stop(*_):
    global stop_requested
    stop_requested=True
signal.signal(signal.SIGUSR1,request_stop)
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,export_policy_as_jit,export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply_inverse
import unitree_rl_lab.tasks
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
from unitree_rl_lab.tasks.locomotion.cpg.core import fk

def write(name,data):
    (a.output/name).write_text(json.dumps(data,indent=2)+'\n')

def main():
    a.output.mkdir(parents=True,exist_ok=True)
    write('status.json',{'status':'running','pid':os.getpid()})
    if a.component_tests:
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
        import pytest
        result=pytest.main(['tests/test_cpg_core.py','tests/test_cpg_adapter.py','tests/test_hierarchical_skills.py','tests/test_hierarchical_adapter.py',
                           'tests/test_conga_reference.py','tests/test_conga_mdp.py','-q','-p','no:cacheprovider'])
        if result:raise RuntimeError(f'Component tests failed: {result}')
    for suffix in ('Walk','Timing','Travel','Dance','Kick-Hold','Bounce'):
        candidate=load_cfg_from_registry('Unitree-Go2-CPG-'+suffix,'env_cfg_entry_point')
        candidate.validate()
        assert candidate.observations.policy.velocity_commands is None
        assert not hasattr(candidate.observations.policy,'base_lin_vel')
    cfg=load_cfg_from_registry(a.task,'env_cfg_entry_point')
    agent=load_cfg_from_registry(a.task,'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;agent.seed=a.seed
    cfg.sim.device=a.device or 'cuda:0';agent.device=cfg.sim.device
    settings=json.loads(a.settings.read_text()) if a.settings else {}
    cfg.equivariant_actor=settings.get('equivariant_actor',False)
    for k,v in settings.get('gait',{}).items():setattr(cfg.commands.cpg.gait,k,v)
    cfg.commands.cpg.gait.__post_init__()
    for k,v in settings.get('rewards',{}).items():getattr(cfg.rewards,k).weight=v
    for k,v in settings.get('ppo',{}).items():setattr(agent.algorithm,k,v)
    for k,v in settings.get('actuator',{}).items():setattr(cfg.scene.robot.actuators['GO2HV'],k,v)
    if settings.get('symmetry_mirror_loss',0):
        if a.task not in ('Unitree-Go2-CPG-Dance','Unitree-Go2-CPG-Kick-Hold','Unitree-Go2-CPG-Bounce'):
            raise ValueError('Musical-phase reflection is qualified for the discrete dance task only')
        from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
        agent.algorithm.symmetry_cfg=RslRlSymmetryCfg(use_data_augmentation=False,use_mirror_loss=True,
            mirror_loss_coeff=settings['symmetry_mirror_loss'],
            data_augmentation_func='unitree_rl_lab.tasks.locomotion.cpg.symmetry:augment')
    if not a.iterations:cfg.observations.policy.enable_corruption=settings.get('observation_noise',False)
    cfg.episode_length_s=settings.get('episode_length_s',cfg.episode_length_s)
    if a.video:
        cfg.viewer.resolution=(960,720);cfg.viewer.eye=(1.7,1.7,1.1);cfg.viewer.lookat=(0.,0.,.25)
        cfg.viewer.eye=tuple(settings.get('camera_eye',cfg.viewer.eye))
        cfg.viewer.lookat=tuple(settings.get('camera_lookat',cfg.viewer.lookat))
        if settings.get('camera_follow',False):
            cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot'
    cfg.validate()
    env=gym.make(a.task,cfg=cfg,render_mode='rgb_array' if a.video else None)
    raw=env.unwrapped;c=raw.command_manager.get_term('cpg')
    if a.video:
        env=gym.wrappers.RecordVideo(env,video_folder=str(a.output/'video'),step_trigger=lambda s:s==0,video_length=a.steps,disable_logger=True)
    wrapped=RslRlVecEnvWrapper(env)
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(a.output/'training') if a.iterations else None,device=agent.device)
    def wrap_actor():
        from unitree_rl_lab.tasks.locomotion.cpg.symmetry import SagittalActor
        if not isinstance(runner.alg.policy.actor_obs_normalizer,torch.nn.Identity):
            raise ValueError('Equivariant actor requires unnormalized physical observations')
        runner.alg.policy.actor=SagittalActor(runner.alg.policy.actor).to(raw.device)
    source_equivariant=False
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        source_equivariant=any(k.startswith('actor.base.') for k in state)
        del state
        if source_equivariant and not cfg.equivariant_actor:
            raise ValueError('This checkpoint requires equivariant_actor in its settings')
    if source_equivariant:wrap_actor()
    if a.checkpoint:
        runner.load(str(a.checkpoint))
        if a.iterations:
            runner.current_learning_iteration+=1
            runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
    if cfg.equivariant_actor and not source_equivariant:wrap_actor()
    dump_yaml(str(a.output/'env.yaml'),cfg);dump_yaml(str(a.output/'agent.yaml'),agent)
    export_deploy_cfg(raw,str(a.output))
    obs=wrapped.get_observations()
    assert obs['policy'].shape[-1]==75,obs['policy'].shape
    assert wrapped.num_actions==12
    start=time.monotonic()
    if a.iterations:
        class BudgetStop(Exception):pass
        original_log=runner.log
        first_iteration=runner.current_learning_iteration
        def bounded_log(locs,*args,**kwargs):
            original_log(locs,*args,**kwargs)
            if stop_requested or time.time()>=float(os.environ.get('CPG_DEADLINE','inf')):
                runner.save(str(a.output/'training'/f'model_{runner.current_learning_iteration}.pt'))
                raise BudgetStop()
        runner.log=bounded_log
        stopped=False
        try:runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
        except BudgetStop:stopped=True
        actual=runner.current_learning_iteration-first_iteration+1
        write('result.json',{'iterations':actual,'elapsed':time.monotonic()-start,'samples':actual*a.num_envs*24,'last_index':runner.current_learning_iteration,'budget_stop':stopped})
    else:
        cases=torch.tensor(settings.get('evaluation_commands',[[0,0,0],[.1,0,0],[-.1,0,0],[0,.08,0],[0,-.08,0],[0,0,.15],[0,0,-.15],[.08,0,.15]]),device=raw.device)
        group=torch.arange(a.num_envs,device=raw.device)%len(cases)
        c.set_velocity(cases[group])
        command_period=settings.get('evaluation_command_period_s',0.)
        if command_period and command_period<raw.step_dt:
            raise ValueError('Evaluation command period must be at least one control step')
        command_slot=0
        policy=runner.get_inference_policy(device=raw.device) if a.checkpoint else None
        body_masses=c.robot.root_physx_view.get_masses().to(raw.device)
        records=[];falls=torch.zeros(a.num_envs,device=raw.device);resets=falls.clone()
        totals={k:torch.zeros(a.num_envs,device=raw.device) for k in c.metrics}
        previous={k:v.clone() for k,v in c.metrics.items()}
        maxik=0.;maxfk=0.;minheight=1.;nan=False
        for step in range(a.steps):
            with torch.inference_mode():
                slot=int(step*raw.step_dt/command_period) if command_period else 0
                if slot!=command_slot:
                    c.set_velocity(cases[(group+slot)%len(cases)])
                    command_slot=slot
                action=policy(obs) if policy else torch.zeros(a.num_envs,12,device=raw.device)
                obs,reward,done,info=wrapped.step(action)
                assert all(torch.isfinite(x).all() for x in obs.values()) and torch.isfinite(reward).all() and torch.isfinite(action).all()
                falls+=raw.reset_terminated.float();resets+=done
                # Pre-reset event records survive command reset. Count attempts from completed events + skips below.
                rec=c.last_records
                slip=(raw.scene['robot'].data.body_lin_vel_w[:,c.foot_ids,:2].norm(dim=-1)*rec['contact']).sum(-1)/rec['contact'].sum(-1).clamp_min(1)
                height=raw.scene['robot'].data.root_pos_w[:,2]-raw.scene.env_origins[:,2]
                records.append({k:v.detach().cpu().numpy() for k,v in rec.items()})
                records[-1]['slip']=slip.cpu().numpy();records[-1]['height']=height.cpu().numpy()
                records[-1]['fall']=raw.reset_terminated.cpu().numpy().copy()
                records[-1]['reset']=done.cpu().numpy().copy()
                records[-1]['gravity']=c.robot.data.projected_gravity_b.cpu().numpy().copy()
                records[-1]['body_velocity']=c.robot.data.root_lin_vel_w.cpu().numpy().copy()
                records[-1]['body_velocity_local']=c.robot.data.root_lin_vel_b.cpu().numpy().copy()
                records[-1]['target_velocity']=getattr(c,'travel_velocity',c.velocity).cpu().numpy().copy()
                records[-1]['operator_velocity']=c.requested_velocity.cpu().numpy().copy()
                records[-1]['planned_velocity']=c.velocity.cpu().numpy().copy()
                records[-1]['foot_velocity']=c.robot.data.body_lin_vel_w[:,c.foot_ids].cpu().numpy().copy()
                relative=c.robot.data.body_pos_w[:,c.foot_ids]-c.robot.data.root_pos_w[:,None,:]
                feet_body=quat_apply_inverse(c.robot.data.root_quat_w[:,None,:].expand(-1,4,-1).reshape(-1,4),relative.reshape(-1,3)).reshape(-1,4,3)
                records[-1]['feet_body']=feet_body.cpu().numpy().copy()
                com_world=(c.robot.data.body_com_pos_w*body_masses[:,:,None]).sum(1)/body_masses.sum(-1,keepdim=True)
                records[-1]['com_body']=quat_apply_inverse(c.robot.data.root_quat_w,com_world-c.robot.data.root_pos_w).cpu().numpy().copy()
                records[-1]['com_world']=(com_world-raw.scene.env_origins).cpu().numpy().copy()
                records[-1]['feet_world']=(c.robot.data.body_pos_w[:,c.foot_ids]-raw.scene.env_origins[:,None,:]).cpu().numpy().copy()
                valid=~done.bool()
                if valid.any():maxfk=max(maxfk,float((fk(c.robot.data.joint_pos[:,c.joint_ids].reshape(-1,4,3))-feet_body)[valid].norm(dim=-1).max()))
                records[-1]['angular_velocity']=c.robot.data.root_ang_vel_b.cpu().numpy().copy()
                records[-1]['joints']=c.robot.data.joint_pos[:,c.joint_ids].cpu().numpy().copy()
                records[-1]['reference_joints']=c.q_ref[:,c.joint_ids].cpu().numpy().copy()
                maxik=max(maxik,float(c.ik_error.max()));minheight=min(minheight,float(height.min()))
                for k,v in c.metrics.items():
                    delta=torch.where(done.bool(),torch.zeros_like(v),v-previous[k])
                    totals[k]+=delta.clamp_min(0);previous[k]=v.clone()
        data={k:np.stack([x[k] for x in records]) for k in records[0]}
        np.savez_compressed(a.output/'traces.npz',**data,groups=group.cpu().numpy())
        summary={'checkpoint':str(a.checkpoint) if a.checkpoint else None,'steps':a.steps,'num_envs':a.num_envs,
                 'seconds':a.steps*raw.step_dt,'elapsed':time.monotonic()-start,'max_ik_error_m':maxik,
                 'max_physical_fk_error_m':maxfk,'minimum_body_height_m':minheight,'falls':int(falls.sum()),'resets':int(resets.sum()),
                 'stable_fraction':float((falls==0).float().mean()),'observation_action_dimensions':[75,90,12],'cases':{}}
        summary['evaluation_command_period_s']=command_period
        summary['command_cases']=cases.cpu().tolist()
        for i,v in enumerate(cases.cpu().tolist()):
            mask=group.cpu().numpy()==i
            if not mask.any():continue
            done_steps=data['finish'][:,mask];correct=data['correct'][:,mask]
            count=int(done_steps.sum());success=int(correct.sum())
            touch=data['touch_error'][:,mask][data['touchdown'][:,mask]]
            label=f'command rotation offset {i}' if command_period else str(v)
            summary['cases'][label]={'completed_attempts':count,'correct':success,
                'correct_fraction':success/max(count,1),'skipped':int(totals['skipped'][group==i].sum()),
                'stable_fraction':float((falls[group==i]==0).float().mean()),
                'median_touchdown_error_s':float(np.median(np.abs(touch))) if len(touch) else None,
                'mean_slip_m_s':float(data['slip'][:,mask][data['ready'][:,mask]].mean()) if data['ready'][:,mask].any() else None,
                'mean_body_velocity':data['body_velocity_local'][:,mask][data['ready'][:,mask]].mean(0).tolist() if data['ready'][:,mask].any() else None,
                'mean_yaw_rate':float(data['angular_velocity'][:,mask,2][data['ready'][:,mask]].mean()) if data['ready'][:,mask].any() else None,
                'mean_target_velocity':data['target_velocity'][:,mask][data['ready'][:,mask]].mean(0).tolist() if data['ready'][:,mask].any() else None,
                'velocity_rmse_m_s':float(np.sqrt(np.mean(np.sum((data['body_velocity_local'][:,mask,:2][data['ready'][:,mask]]-data['target_velocity'][:,mask,:2][data['ready'][:,mask]])**2,axis=-1)))) if data['ready'][:,mask].any() else None,
                'yaw_rmse_rad_s':float(np.sqrt(np.mean((data['angular_velocity'][:,mask,2][data['ready'][:,mask]]-data['target_velocity'][:,mask,2][data['ready'][:,mask]])**2))) if data['ready'][:,mask].any() else None,
                'paired_lifts':int(data['paired_lift'][:,mask].sum()),
                'FL_RR_correct':int((correct & data['pair'][:,mask,0]).sum()),
                'FR_RL_correct':int((correct & data['pair'][:,mask,1]).sum())}
        if 'kick_success' in data:
            summary['dance']={}
            for leg,name in enumerate(('FL','FR')):
                selected=data['kick_leg']==leg
                starts=data['kick_active']&~np.vstack((np.zeros((1,a.num_envs),dtype=bool),data['kick_active'][:-1]))
                durations=[]
                for env_index in range(a.num_envs):
                    valid=data['hold_valid'][:,env_index]&selected[:,env_index]
                    edges=np.diff(np.r_[False,valid,False].astype(int))
                    durations.extend(((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*.02).tolist())
                summary['dance'][name]={'kick_attempts':int((starts&selected).sum()),
                    'kick_successes':int((data['kick_success']&selected).sum()),
                    'kick_failures':int((data['kick_failure']&selected).sum()),
                    'longest_continuous_valid_hold_s':max(durations,default=0.),
                    'holds_over_075_s':sum(x>=.75 for x in durations),
                    'postkick_successes':int((data['postkick_success']&selected).sum()) if 'postkick_success' in data else None,
                    'median_endpoint_delay_s':float(np.median(data['kick_endpoint_delay'][data['kick_success']&selected])) if 'kick_endpoint_delay' in data and (data['kick_success']&selected).any() else None}
        write('result.json',summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,1,figsize=(13,7),sharex=True)
        for j,ax in enumerate(axes):
            ax.step(np.arange(a.steps)*.02,data['expected'][:,0,j],label='expected',where='post')
            ax.step(np.arange(a.steps)*.02,data['contact'][:,0,j]*.8,label='measured',where='post')
            ax.set_ylabel(('FL','FR','RL','RR')[j]);ax.set_ylim(-.1,1.1)
        axes[0].legend();axes[-1].set_xlabel('Physical time (s), first environment: '+a.task)
        fig.tight_layout();fig.savefig(a.output/'contacts.png');plt.close(fig)
        normalizer=getattr(runner.alg.policy,'actor_obs_normalizer',None)
        export_policy_as_jit(runner.alg.policy,normalizer=normalizer,path=str(a.output/'export'),filename='policy.pt')
        export_policy_as_onnx(runner.alg.policy,normalizer=normalizer,path=str(a.output/'export'),filename='policy.onnx')
        reloaded=torch.jit.load(str(a.output/'export/policy.pt'),map_location=raw.device)
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(obs['policy']),runner.get_inference_policy(device=raw.device)(obs))
    write('status.json',{'status':'complete'})
    env.close()

try:main()
except Exception as e:
    a.output.mkdir(parents=True,exist_ok=True)
    write('status.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc()})
    traceback.print_exc()
finally:
    # Isaac Sim can hang during shutdown even after a Python exception. Bound cleanup.
    import threading
    timer=threading.Timer(15.,lambda:os._exit(0 if (a.output/'status.json').exists() and json.loads((a.output/'status.json').read_text()).get('status')=='complete' else 1))
    timer.daemon=True;timer.start()
    app.close()
