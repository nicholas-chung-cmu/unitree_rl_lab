"""Persistent sequential CPG jobs with the original, immutable four-hour deadline.

Plan jobs are immutable once launched. Reinvocation skips completed jobs and refuses
failed jobs; a corrected experiment needs a new job name. No automatic crash retry.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--session',type=Path,default=Path('logs/conga_cpg_session_20261101'))
a=p.parse_args();os.chdir(ROOT)
SESSION=a.session.resolve()
meta=json.loads((SESSION/'session.json').read_text());deadline=meta['deadline_epoch']
if meta.get('status')=='complete':
    raise RuntimeError('This session is closed; use a separate authorized session for new experiments')
if not 0<deadline-meta['started_epoch']<=4*3600:
    raise ValueError('A CPG session must preserve its recorded start and <=four-hour deadline')
gpu_lock=open(ROOT/'logs/cpg_gpu.lock','w')
fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock=open(SESSION/'runner.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads(a.plan.read_text())
state={'status':'running','deadline_epoch':deadline,'jobs':[],'plan':str(a.plan)}

def write_state():
    tmp=SESSION/'runner_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(SESSION/'runner_status.json')

def checkpoint(job):
    files=list((SESSION/job/'training').glob('model_*.pt'))
    if not files:raise RuntimeError(f'No checkpoint in {job}')
    return max(files,key=lambda x:int(x.stem.split('_')[-1]))

def report():
    lines=['# CPG autonomous session','',f"Original deadline: {meta['deadline_utc']}. Runner state: {state['status']}.",
           '', 'This incremental report is updated after each job; final analysis follows once evaluation is complete.',
           '', '| Job | Status | Duration | Evidence |','|---|---|---|---|']
    for job in plan['jobs']:
        out=SESSION/job['name'];status=out/'status.json';result=out/'result.json'
        if not status.exists():continue
        s=json.loads(status.read_text());r=json.loads(result.read_text()) if result.exists() else {}
        evidence=f"{r.get('iterations',0)} updates" if job.get('iterations') else f"stable={r.get('stable_fraction')}, falls={r.get('falls')}"
        lines.append(f"| {job['name']} | {s['status']} | {r.get('elapsed',0):.1f} s | {evidence} |")
    (ROOT/'reports/conga_cpg_overnight.md').write_text('\n'.join(lines)+'\n')

try:
    for job in plan['jobs']:
        out=SESSION/job['name'];status=out/'status.json'
        if status.exists():
            old=json.loads(status.read_text())
            if old['status']=='complete':continue
            raise RuntimeError(f"Refusing implicit retry of {job['name']}: {old['status']}")
        remaining=deadline-time.time()
        reserve=600 if job.get('iterations') else 60
        if remaining<reserve+120:
            state['status']='budget_stop';break
        out.mkdir(parents=True)
        source=out/'source';source.mkdir()
        for rel in ('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg',
                    'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/cpg_env_cfg.py',
                    'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/cpg_ppo_cfg.py',
                    'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py',
                    'source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py',
                    'scripts/cpg_experiment.py','scripts/run_cpg_session.py',
                    'scripts/analyze_cpg_contacts.py','scripts/audit_cpg_kicks.py',
                    'scripts/audit_cpg_hold.py','scripts/audit_cpg_bounce.py','scripts/calibrate_cpg_com.py',
                    'tests/test_cpg_core.py','tests/test_cpg_adapter.py'):
            src=ROOT/rel
            if src.is_dir():shutil.copytree(src,source/src.name,ignore=shutil.ignore_patterns('__pycache__'))
            else:shutil.copy2(src,source/src.name)
        (out/'git_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
        (out/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
        (out/'job.json').write_text(json.dumps(job,indent=2))
        settings=out/'settings.json';settings.write_text(json.dumps(job.get('settings',{}),indent=2))
        cmd=[PY,'scripts/cpg_experiment.py','--headless','--output',str(out),'--settings',str(settings),
             '--num_envs',str(job.get('num_envs',64)),'--seed',str(job.get('seed',9041))]
        for name in ('iterations','steps'):
            if name in job:cmd+=['--'+name,str(job[name])]
        if job.get('resume_job'):cmd+=['--checkpoint',str(checkpoint(job['resume_job']))]
        if job.get('checkpoint'):cmd+=['--checkpoint',job['checkpoint']]
        if job.get('video'):cmd+=['--video']
        if job.get('task'):cmd+=['--task',job['task']]
        if job.get('component_tests'):cmd+=['--component-tests']
        record={'job':job['name'],'started_epoch':time.time(),'command':cmd}
        state['jobs'].append(record);state['current']=job['name'];write_state()
        with open(out/'process.log','w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                  pass_fds=(gpu_lock.fileno(),lock.fileno()),
                                  env={**os.environ,'PYTHONUNBUFFERED':'1','CPG_DEADLINE':str(deadline-reserve)})
            record['pid']=proc.pid;write_state()
            until=min(deadline-reserve,time.time()+job.get('timeout_seconds',3600 if job.get('iterations') else 300))
            signaled=False
            while proc.poll() is None:
                if time.time()>until-30 and not signaled:
                    os.killpg(proc.pid,signal.SIGUSR1);signaled=True
                if time.time()>until:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(2)
            record['returncode']=proc.returncode;record['ended_epoch']=time.time();write_state()
        if not status.exists() or json.loads(status.read_text())['status']!='complete':
            raise RuntimeError(f"Child {job['name']} failed or hit timeout; inspect its status/process.log")
        report()
    else:state['status']='complete'
except Exception as exc:
    state['status']='stopped';state['error']=repr(exc)
finally:
    write_state();report()
