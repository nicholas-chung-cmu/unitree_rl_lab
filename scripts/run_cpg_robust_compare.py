"""Bounded matched 75/81 robustness comparison. No hardware commands or promotion."""
from pathlib import Path
import os,sys,json,time,subprocess,signal,hashlib,fcntl,shutil
R=Path(__file__).resolve().parents[1];PY='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'
S=R/'logs/cpg_robust_compare_20261101';os.chdir(R)

def write(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def main():
 if (S/'runner_status.json').exists():raise RuntimeError('Session already started; do not overwrite')
 lock=open(R/'logs/cpg_gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 deadline=time.time()+7200
 state={'status':'running','started_epoch':time.time(),'deadline_epoch':deadline,'updates':{'75':0,'81':0},'num_envs':4096,'jobs':[],'gates':[],'hardware':False}
 source=R/'logs/cpg_travel_curriculum_20261101/hold_transfer/walk_rehearsal_1/training/model_2499.pt'
 state['original_checkpoint_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
 snapshot=S/'source';snapshot.mkdir()
 for name in ['source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg','deploy/cpg']:
  shutil.copytree(R/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
 for name in ['scripts/cpg_experiment.py','scripts/cpg_mujoco.py','scripts/run_cpg_robust_compare.py','scripts/audit_cpg_kicks.py','scripts/analyze_cpg_contacts.py','scripts/cpg_tracking_metrics.py','source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py','source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/cpg_env_cfg.py','source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py']:
  p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
 write(S/'source_manifest.json',{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()})
 def status():write(S/'runner_status.json',state)
 def run(name,cmd,timeout=300,allow_fault=False):
  if time.time()+timeout>deadline-60:raise TimeoutError('Insufficient remaining experiment budget')
  job={'name':name,'start':time.time(),'command':cmd};state['jobs'].append(job);state['current']=name;status()
  print('START',name,flush=True)
  with (S/(name+'.log')).open('w') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),),env={**os.environ,'PYTHONUNBUFFERED':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','CPG_DEADLINE':str(deadline-60)})
   job['pid']=p.pid;status()
   try:p.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGUSR1)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise TimeoutError(name)
  job.update(end=time.time(),returncode=p.returncode);status();print('END',name,p.returncode,flush=True)
  if p.returncode and not allow_fault:raise RuntimeError(name+' failed')
 def isaac(name,ckpt,cfg,updates=0,steps=2000,envs=32,seed=9101):
  config=S/(name+'_settings.json');write(config,cfg)
  cmd=[PY,'scripts/cpg_experiment.py','--headless','--task','Unitree-Go2-CPG-Dance','--settings',str(config),'--checkpoint',str(ckpt),'--output',str(S/name),'--num_envs',str(4096 if updates else envs),'--seed',str(seed),'--iterations',str(updates),'--steps',str(steps)]
  run(name,cmd,timeout=max(300,updates*2+120))
  result=json.loads((S/name/'result.json').read_text())
  if updates:
   if result['iterations']!=updates or result['budget_stop']:raise RuntimeError('Incomplete training segment')
   return S/name/'training'/f"model_{result['last_index']}.pt"
  return S/name
 def audit(out):
  for script,args in [('analyze_cpg_contacts.py',['--dance','--swing','.2','--tolerance','.06']),('audit_cpg_kicks.py',[])]:
   run(out.name+'_'+script,[PY,'scripts/'+script,str(out),*args],timeout=30)
  r=json.loads((out/'result.json').read_text());k=json.loads((out/'kick_audit.json').read_text())['legs'];ct=json.loads((out/'contact_audit.json').read_text())['cases']
  return {'status':r.get('status','complete'),'fault':r.get('fault'),'falls':r['falls'],'steps':[sum(v['correct'] for v in ct.values()),sum(v['expected'] for v in ct.values())],'kicks':[sum(v['qualified_chain'] for v in k.values()),sum(v['expected'] for v in k.values())],'legs':k,'tracking':r.get('travel_only_tracking') or json.loads((out/'tracking_audit.json').read_text())}
 def gate(label,ckpt,obs,fresh=False):
  cfg=json.loads((S/f'eval{obs}.json').read_text());cfg['fresh_optimizer']=fresh
  out=isaac(label+'_isaac',ckpt,cfg);a=audit(out)
  selection={'checkpoint':str(ckpt),'deploy_metadata':str(out/'params/deploy.yaml'),'onnx_export':str(out/'export/policy.onnx'),'jit_export':str(out/'export/policy.pt'),'settings':str(S/(label+'_isaac_settings.json')),'hardware_compatible':False,'joystick_limits':[.12,.06,.1],'onnx_sha256':hashlib.sha256((out/'export/policy.onnx').read_bytes()).hexdigest(),'observation_dim':obs}
  sp=S/(label+'_selection.json');write(sp,selection);cases=[]
  for i,command in enumerate(cfg['evaluation_commands']):
   dest=S/(label+'_mj_'+str(i));run(dest.name,[PY,'scripts/cpg_mujoco.py','--selection',str(sp),'--output',str(dest),'--seconds','40','--input','fixed','--velocity',*map(str,command),'--first-front','FL' if i%2==0 else 'FR'],timeout=60,allow_fault=True)
   if not (dest/'result.json').exists():raise RuntimeError('Missing MuJoCo output '+dest.name)
   cases.append({'command':command,**audit(dest)})
  result={'label':label,'obs':obs,'selection':str(sp),'checkpoint':str(ckpt),'isaac':a,'mujoco':cases}
  state['gates'].append(result);write(S/(label+'_gate.json'),result);status();return result
 try:
  gate('baseline',source,75,fresh=True)
  checkpoints={75:source,81:source}
  for stage,updates in [(1,250),(2,750)]:
   for obs in (75,81):
    cfg=json.loads((S/f'train{obs}_stage{stage}.json').read_text())
    name=f'obs{obs}_train{stage}'
    checkpoints[obs]=isaac(name,checkpoints[obs],cfg,updates=updates,seed=80+stage)
    state['updates'][str(obs)]+=updates;status()
    gate(f'obs{obs}_{state["updates"][str(obs)]}',checkpoints[obs],obs)
  for obs in (75,81):
   cfg=json.loads((S/f'train{obs}_stage2.json').read_text());cfg.update(episode_length_s=90,evaluation_command_period_s=6)
   out=isaac(f'obs{obs}_stress',checkpoints[obs],cfg,steps=3000,seed=9217)
   write(S/f'obs{obs}_stress_audit.json',audit(out))
  assert hashlib.sha256(source.read_bytes()).hexdigest()==state['original_checkpoint_sha256']
  state['status']='complete';state['ended_epoch']=time.time();status()
 except Exception as e:
  state['status']='failed';state['error']=repr(e);state['ended_epoch']=time.time();status();raise
 print('DONE',flush=True)

if __name__=='__main__':main()
