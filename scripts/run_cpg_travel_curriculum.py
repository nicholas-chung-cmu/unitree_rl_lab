"""Travel-first curriculum with sequential Isaac/MuJoCo review gates.

Separate session and report; never reopens or overwrites the closed overnight run.
"""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);a=p.parse_args()
    os.chdir(ROOT);session=a.session.resolve();meta=json.loads((session/'session.json').read_text())
    if meta['status'] in ('complete','complete_walk_only','review_stop','failed'):raise RuntimeError('Closed session cannot be restarted')
    deadline=meta['deadline_epoch']
    if not 0<deadline-meta['started_epoch']<=14400:raise ValueError('Invalid wall-clock budget')
    lock=open(ROOT/'logs/cpg_gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session_lock=open(session/'runner.lock','w');fcntl.flock(session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'running','updates':0,'jobs':[],'gates':[],'deadline_epoch':deadline,'num_envs':4096}
    source=(ROOT/meta['source']).resolve()
    train=json.loads((session/'train_initial.json').read_text())
    evaluation=json.loads((session/'eval_initial.json').read_text())
    def write(path,value):
        path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    def status():write(session/'runner_status.json',state)
    snapshot=session/'source';snapshot.mkdir(exist_ok=False)
    for name in ['source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg','deploy/cpg']:
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['scripts/cpg_experiment.py','scripts/cpg_mujoco.py','scripts/cpg_tracking_metrics.py','scripts/run_cpg_travel_curriculum.py',
                 'scripts/analyze_cpg_contacts.py','scripts/audit_cpg_kicks.py','scripts/cpg_worker.py','scripts/cpg_quad_ros2.py','AGENTS.md']:
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(session/'source_manifest.json',{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()})

    def run(name,cmd,timeout=300,allow_fault=False):
        if time.time()+timeout>deadline-120:raise TimeoutError('Insufficient session time for bounded next job')
        job={'name':name,'command':cmd,'start':time.time()};state['jobs'].append(job);state['current']=name;status()
        with open(session/(name+'.log'),'w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                pass_fds=(lock.fileno(),session_lock.fileno()),env={**os.environ,'PYTHONUNBUFFERED':'1','CPG_DEADLINE':str(deadline-600)})
            job['pid']=child.pid;status()
            try:child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGUSR1)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError(name+' exceeded job budget')
        job.update(returncode=child.returncode,end=time.time());status()
        if child.returncode and not allow_fault:raise RuntimeError(name+' failed; inspect log')

    def isaac(name,checkpoint,settings,iterations=0,envs=32,steps=1600):
        config=session/(name+'_settings.json');write(config,settings)
        output=session/name
        if output.exists():raise RuntimeError('Refusing to overwrite '+str(output))
        cmd=[PY,'scripts/cpg_experiment.py','--headless','--task',settings.get('task','Unitree-Go2-CPG-Travel'),'--checkpoint',str(checkpoint),
             '--settings',str(config),'--output',str(output),'--num_envs',str(envs),'--seed','9101' if not iterations else '81',
             '--iterations',str(iterations),'--steps',str(steps)]
        if name=='walk_baseline_isaac' or (name=='dance_transfer_baseline_isaac' and meta.get('component_tests')):cmd.append('--component-tests')
        run(name,cmd,timeout=max(300,iterations*2+180))
        if json.loads((output/'status.json').read_text())['status']!='complete':raise RuntimeError(name+' reported failure')
        result=json.loads((output/'result.json').read_text())
        if iterations:
            if result['iterations']!=iterations or result.get('budget_stop'):raise RuntimeError('Incomplete training segment')
            return output/'training'/f"model_{result['last_index']}.pt"
        return output

    def audit(output):
        trace=dict(np.load(output/'traces.npz'));dance='kick_active' in trace
        scripts=[('analyze_cpg_contacts.py',(['--dance'] if dance else [])+['--swing','.2','--tolerance','.06'])]
        if dance:scripts.append(('audit_cpg_kicks.py',[]))
        for script,args in scripts:
            run(output.name+'_'+script,[PY,'scripts/'+script,str(output),*args],timeout=60)
        contacts=json.loads((output/'contact_audit.json').read_text())['cases']
        kicks=json.loads((output/'kick_audit.json').read_text())['legs'] if dance else {}
        expected=sum(v['expected'] for v in contacts.values());correct=sum(v['correct'] for v in contacts.values())
        kexp=sum(v['expected'] for v in kicks.values());kcorrect=sum(v['qualified_chain'] for v in kicks.values())
        result=json.loads((output/'result.json').read_text())
        track=result.get('travel_only_tracking') or json.loads((output/'tracking_audit.json').read_text())
        return {'steps':[correct,expected],'step_rate':correct/max(1,expected),'kicks':[kcorrect,kexp],
                'kick_rate':kcorrect/max(1,kexp),'falls':result['falls'],'tracking':track,
                'net_displacement_xyz':(trace['body_position'][-1]-trace['body_position'][0]).mean(0).tolist(),
                'reset_samples':int(trace['reset'].sum()),'kick_legs':kicks}

    def gate(label,checkpoint):
        output=isaac(label+'_isaac',checkpoint,evaluation)
        isaac_result=audit(output)
        selection={'checkpoint':str(checkpoint),'deploy_metadata':str(output/'params/deploy.yaml'),
                   'onnx_export':str(output/'export/policy.onnx'),'jit_export':str(output/'export/policy.pt'),
                   'settings':str(session/(label+'_isaac_settings.json')),'hardware_compatible':False,
                   'joystick_limits':evaluation.get('joystick_limits',[.12,.08,.15]),
                   'onnx_sha256':hashlib.sha256((output/'export/policy.onnx').read_bytes()).hexdigest()}
        selected=session/(label+'_selection.json');write(selected,selection)
        cases=evaluation['evaluation_commands'];mj=[]
        for i,velocity in enumerate(cases):
            name=f'{label}_mj_{i}';out=session/name
            run(name,[PY,'scripts/cpg_mujoco.py','--selection',str(selected),'--output',str(out),'--seconds','22',
                      '--input','fixed','--velocity',*[str(v) for v in velocity],'--first-front','FL' if i%2==0 else 'FR'],timeout=60,allow_fault=True)
            r=json.loads((out/'result.json').read_text())
            if r['status']=='fault':
                mj.append({'command':velocity,'falls':max(1,r['falls']),'step_rate':0.,'kick_rate':0.,'fault':r['fault']});continue
            mj.append({'command':velocity,**audit(out)})
        safe=[x for x in mj if 'tracking' in x]
        total_steps=sum(x['steps'][1] for x in safe);total_correct=sum(x['steps'][0] for x in safe)
        total_kicks=sum(x['kicks'][1] for x in safe);total_kcorrect=sum(x['kicks'][0] for x in safe)
        def average(key):
            values=[x['tracking'][key] for x in safe if x['tracking'][key] is not None]
            return sum(values)/len(values) if values else None
        result={'label':label,'updates':state['updates'],'checkpoint':str(checkpoint),'selection':str(selected),
                'isaac':isaac_result,'mujoco_cases':mj,'mujoco':{'steps':[total_correct,total_steps],
                'step_rate':total_correct/max(1,total_steps),'kicks':[total_kcorrect,total_kicks],
                'kick_rate':total_kcorrect/max(1,total_kicks),'falls':sum(x['falls'] for x in mj),
                'xy_bias':average('mean_command_xy_bias_m_s'),'yaw_bias':average('mean_command_yaw_bias_rad_s'),
                'directional_gain':average('nonzero_directional_gain'),
                'kick_legs':{leg:[sum(x.get('kick_legs',{}).get(leg,{}).get('qualified_chain',0) for x in safe),
                                  sum(x.get('kick_legs',{}).get(leg,{}).get('expected',0) for x in safe)] for leg in ('FL','FR')}}}
        state['gates'].append(result);write(session/(label+'_gate.json'),result);status();return result

    def safe(g):
        return g['isaac']['falls']==0 and g['mujoco']['falls']==0 and g['isaac']['step_rate']>=.95 and g['mujoco']['step_rate']>=.95
    def forward(g):
        return g['mujoco_cases'][1].get('net_displacement_xyz',[0])[0]
    def travel_ok(g):
        if not safe(g):return False
        cases=g['mujoco_cases']
        for i,axis,minimum in [(1,0,.06),(2,0,.035),(3,1,.025),(4,1,.025)]:
            stats=list(cases[i]['tracking']['cases'].values())
            if not stats:return False
            speed=stats[0]['actual_mean'][axis]
            if speed*np.sign(cases[i]['command'][axis])<minimum:return False
        return forward(g)>=.8
    def dance_ok(g):
        legs=g['isaac'].get('kick_legs',{})
        bilateral=all(legs.get(leg,{}).get('qualified_fraction',0)>=.8 for leg in ('FL','FR'))
        mj=g['mujoco'].get('kick_legs',{})
        bilateral &= all(mj.get(leg,[0,1])[0]/max(1,mj.get(leg,[0,1])[1])>=.7 for leg in ('FL','FR'))
        return travel_ok(g) and bilateral
    def assessment(g):
        return dict(safe=safe(g),travel_qualified=travel_ok(g),dance_qualified=dance_ok(g),forward_metres=forward(g))
    def eval_config(training):
        cfg=json.loads(json.dumps(training));cfg.update(episode_length_s=60,transfer_randomization=False,observation_noise=True)
        cfg['actuator'].update(min_delay=0,max_delay=0)
        return cfg
    def learn(name,checkpoint,settings,amount):
        if state['updates']+amount>meta['max_updates']:raise RuntimeError('Update budget exceeded')
        result=isaac(name,checkpoint,settings,iterations=amount,envs=4096)
        state['updates']+=amount;status();return result
    selected_walk=None;selected_dance=None
    try:
        if meta.get('existing_walk_gate'):
            baseline=json.loads(Path(meta['existing_walk_gate']).read_text())
            assert Path(baseline['checkpoint']).resolve()==source
            cached=json.loads(Path(json.loads(Path(baseline['selection']).read_text())['settings']).read_text())
            assert cached==evaluation,'Cached walking evaluation settings differ'
            state['gates'].append(baseline);state['reused_walk_gate']=meta['existing_walk_gate'];status()
        else:baseline=gate('walk_baseline',source)
        baseline['assessment']=assessment(baseline)
        best_walk=baseline;selected_walk=baseline
        if baseline['isaac']['falls'] or baseline['mujoco']['falls']:
            raise RuntimeError('Walking baseline requires physical review before training')
        for amount in ([] if meta.get('skip_walk_updates') else [100,200]):
            source=learn(f'walk_train_{state["updates"]+amount}',source,train,amount)
            current=gate(f'walk_gate_{state["updates"]}',source);current['assessment']=assessment(current)
            if safe(current) and (not safe(best_walk) or forward(current)>forward(best_walk)*.95):best_walk=current
            selected_walk=best_walk;state['selected_walk']=best_walk['selection'];status()
            if travel_ok(current):best_walk=current;selected_walk=current;break
            if current['mujoco']['falls'] or current['isaac']['step_rate']<baseline['isaac']['step_rate']-.1:
                state['review_reason']='Walking retention regressed';break
        write(session/'selected_walk.json',json.loads(Path(selected_walk['selection']).read_text()))
        state['selected_walk']=selected_walk['selection']
        if travel_ok(best_walk):
            # Preserve ordinary walking with short rehearsal segments between
            # dance updates, rather than training only the final choreography.
            dance=json.loads((session/'train_dance.json').read_text())
            evaluation=eval_config(dance);source=Path(meta.get('dance_source',best_walk['checkpoint']))
            initial=gate('dance_transfer_baseline',source);initial['assessment']=assessment(initial)
            best_dance=initial if dance_ok(initial) else None
            if (initial['mujoco']['falls'] or initial['isaac']['falls']>1) and not meta.get('initial_policy_may_fail_kicks',False):
                state['review_reason']='Dance transfer is physically unstable; retain qualified walking'
            else:
                for index,amount in enumerate(meta.get('dance_amounts',[200,300,500,400])):
                    source=learn(f'dance_train_{index}',source,dance,amount)
                    source=learn(f'walk_rehearsal_{index}',source,train,meta.get('rehearsal_updates',50))
                    current=gate(f'dance_gate_{state["updates"]}',source);current['assessment']=assessment(current)
                    if dance_ok(current) and (best_dance is None or min(current['isaac']['kick_rate'],current['mujoco']['kick_rate'])>min(best_dance['isaac']['kick_rate'],best_dance['mujoco']['kick_rate'])):best_dance=current
                    state['dance_assessment']=assessment(current);status()
                    if current['mujoco']['falls'] or current['isaac']['falls']>1:
                        state['review_reason']='Physical fall review in dance stage';break
                    if dance_ok(current):break
                    if index>=1 and current['isaac']['step_rate']<.8:
                        state['review_reason']='Dance stepping unreliable after two gates';break
                if best_dance:
                    selected_dance=best_dance
                    # Bounce is only attempted after both simulators retain
                    # travel, steps and strict kick holds.
                    if meta.get('enable_bounce',True) and state['updates']+100<=meta['max_updates']:
                        bounce=json.loads(json.dumps(dance));bounce['gait']['bounce_amplitude']=.0075;bounce['task']='Unitree-Go2-CPG-Bounce'
                        source=learn('bounce_train',Path(best_dance['checkpoint']),bounce,100)
                        evaluation=eval_config(bounce)
                        bounced=gate('bounce_gate',source);bounced['assessment']=assessment(bounced)
                        if dance_ok(bounced):selected_dance=bounced
                    write(session/'selected_dance.json',json.loads(Path(selected_dance['selection']).read_text()))
        else:state['review_reason']='Walking travel gate not achieved; no kick advancement'
        chosen=selected_dance or selected_walk
        write(session/'selected.json',json.loads(Path(chosen['selection']).read_text()))
        state['selected']=chosen['selection'];state['selected_mode']='dance' if selected_dance else 'walk'
        evaluation=json.loads(Path(json.loads(Path(chosen['selection']).read_text())['settings']).read_text())
        evaluation['evaluation_command_period_s']=6.
        state['dynamic']=audit(isaac('selected_dynamic',Path(chosen['checkpoint']),evaluation,steps=2100))
        for name,mode,velocity in [('selected_forward_video','fixed',['.12','0','0']),('selected_commands_video','scripted',[])]:
            cmd=[PY,'scripts/cpg_mujoco.py','--selection',str(session/'selected.json'),'--output',str(session/name),'--seconds','30' if mode=='fixed' else '40','--input',mode,'--video']
            if velocity:cmd+=['--velocity',*velocity]
            run(name,cmd,timeout=300,allow_fault=True)
            state[name]=audit(session/name)
        state['status']='complete' if selected_dance else 'complete_walk_only'
        status()
    except Exception as error:
        state['status']='failed';state['error']=repr(error);status();raise
    finally:
        state['ended_epoch']=time.time();status()
        meta['status']=state['status'];meta['updates']=state['updates'];meta['ended_epoch']=state['ended_epoch'];write(session/'session.json',meta)


if __name__=='__main__':main()
