"""Bounded fine-tuning with sequential Isaac/MuJoCo review gates.

Separate session and report; never reopens or overwrites the closed overnight run.
"""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);a=p.parse_args()
    os.chdir(ROOT);session=a.session.resolve();meta=json.loads((session/'session.json').read_text())
    if meta['status'] in ('complete','review_stop','failed'):raise RuntimeError('Closed session cannot be restarted')
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
    for name in ['scripts/cpg_experiment.py','scripts/cpg_mujoco.py','scripts/cpg_tracking_metrics.py','scripts/run_cpg_finetune.py',
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
        cmd=[PY,'scripts/cpg_experiment.py','--headless','--task','Unitree-Go2-CPG-Bounce','--checkpoint',str(checkpoint),
             '--settings',str(config),'--output',str(output),'--num_envs',str(envs),'--seed','9101' if not iterations else '71',
             '--iterations',str(iterations),'--steps',str(steps)]
        run(name,cmd,timeout=max(300,iterations*2+180))
        if json.loads((output/'status.json').read_text())['status']!='complete':raise RuntimeError(name+' reported failure')
        result=json.loads((output/'result.json').read_text())
        if iterations:
            if result['iterations']!=iterations or result.get('budget_stop'):raise RuntimeError('Incomplete training segment')
            return output/'training'/f"model_{result['last_index']}.pt"
        return output

    def audit(output):
        for script,args in [('analyze_cpg_contacts.py',['--dance','--swing','.2','--tolerance','.06']),('audit_cpg_kicks.py',[])]:
            run(output.name+'_'+script,[PY,'scripts/'+script,str(output),*args],timeout=60)
        contacts=json.loads((output/'contact_audit.json').read_text())['cases']
        kicks=json.loads((output/'kick_audit.json').read_text())['legs']
        expected=sum(v['expected'] for v in contacts.values());correct=sum(v['correct'] for v in contacts.values())
        kexp=sum(v['expected'] for v in kicks.values());kcorrect=sum(v['qualified_chain'] for v in kicks.values())
        result=json.loads((output/'result.json').read_text())
        track=result.get('travel_only_tracking') or json.loads((output/'tracking_audit.json').read_text())
        return {'steps':[correct,expected],'step_rate':correct/max(1,expected),'kicks':[kcorrect,kexp],
                'kick_rate':kcorrect/max(1,kexp),'falls':result['falls'],'tracking':track}

    def gate(label,checkpoint):
        output=isaac(label+'_isaac',checkpoint,evaluation)
        isaac_result=audit(output)
        selection={'checkpoint':str(checkpoint),'deploy_metadata':str(output/'params/deploy.yaml'),
                   'onnx_export':str(output/'export/policy.onnx'),'jit_export':str(output/'export/policy.pt'),
                   'settings':str(session/(label+'_isaac_settings.json')),'hardware_compatible':False,
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
                'directional_gain':average('nonzero_directional_gain')}}
        state['gates'].append(result);write(session/(label+'_gate.json'),result);status();return result

    def eligible(g):
        return g['isaac']['falls']==0 and g['mujoco']['falls']==0 and g['isaac']['step_rate']>=.95 and g['mujoco']['step_rate']>=.90
    def score(g):
        m=g['mujoco'];return 4*(m['xy_bias'] or 0)+.3*(m['yaw_bias'] or 0)+.15*(1-m['kick_rate'])+.3*(1-m['step_rate'])

    try:
        baseline=gate('baseline',source);best=baseline
        if baseline['isaac']['falls']>1 or baseline['mujoco']['falls']:raise RuntimeError('Baseline port preflight is unstable')
        for amount in [100,200,400,500]:
            name=f'train_{state["updates"]+amount}'
            source=isaac(name,source,train,iterations=amount,envs=4096)
            state['updates']+=amount;status()
            evaluation['gait']['max_speed']=train['gait']['max_speed']
            evaluation['gait']['max_yaw']=train['gait']['max_yaw']
            current=gate(f'gate_{state["updates"]}',source)
            if eligible(current) and (not eligible(best) or score(current)<score(best)):best=current
            state['best_selection']=best['selection'];status()
            if current['isaac']['falls']/32>.05 or current['mujoco']['falls']:
                state['status']='review_stop';state['reason']='Physical fall review threshold';break
            if current['isaac']['step_rate']<baseline['isaac']['step_rate']-.10 or current['mujoco']['step_rate']<baseline['mujoco']['step_rate']-.15:
                state['status']='review_stop';state['reason']='Step retention regressed beyond review threshold';break
            # Increase the training range only after contact and tracking gates.
            # Fixed low-speed evaluation remains identical for valid comparisons.
            m=current['mujoco'];old_range=train['gait']['max_speed']
            if state['updates']>=300 and current['isaac']['step_rate']>=.97 and m['step_rate']>=.95 and m['kick_rate']>=.5 and (m['directional_gain'] or 0)>=.75:
                train['gait']['max_speed']=min(.15,.12 if old_range<.12 else .15)
                train['gait']['max_yaw']=.20
            write(session/f'next_settings_after_{state["updates"]}.json',train)
        else:state['status']='complete'
        state['best_selection']=best['selection'];state['best_gate']=best['label'];status()
        # Independently evaluate changing commands with selected checkpoint.
        selected=Path(best['selection']);checkpoint=Path(best['checkpoint'])
        dynamic=json.loads(Path(json.loads(selected.read_text())['settings']).read_text());dynamic['evaluation_command_period_s']=4.1
        dyn=isaac('selected_dynamic_isaac',checkpoint,dynamic,steps=2100)
        state['selected_dynamic_isaac']=audit(dyn)
        for name,input_mode,seconds in [('selected_zero_video','zero',22),('selected_commands_video','scripted',40)]:
            output=session/name
            run(name,[PY,'scripts/cpg_mujoco.py','--selection',str(selected),'--output',str(output),
                      '--input',input_mode,'--seconds',str(seconds),'--video'],timeout=360,allow_fault=True)
            if (output/'result.json').exists():audit(output)
        write(session/'selected.json',json.loads(selected.read_text()))
        status()
    except Exception as error:
        state['status']='failed';state['error']=repr(error);status();raise
    finally:
        state['ended_epoch']=time.time();status()
        meta['status']=state['status'];meta['updates']=state['updates'];meta['ended_epoch']=state['ended_epoch'];write(session/'session.json',meta)


if __name__=='__main__':main()
