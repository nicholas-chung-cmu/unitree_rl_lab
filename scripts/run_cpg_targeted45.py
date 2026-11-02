"""Bounded targeted robustness fine-tune; checkpoints and immutable evaluations."""
from pathlib import Path
import argparse,json,time,os,signal,subprocess,hashlib,fcntl,shutil
R=Path('/home/rml2/Documents/nicholas_chung/go2-conga/unitree_rl_lab');S=R/'logs/cpg_targeted45_20261102';PY='/home/rml2/anaconda3/envs/nick_sillywalk/bin/python'
p=argparse.ArgumentParser();p.add_argument('--branch',default='precision');p.add_argument('--settings',type=Path,default=S/'train.json');p.add_argument('--checkpoint',type=Path,default=R/'logs/cpg_robust_compare_20261101/obs75_train1/training/model_249.pt');p.add_argument('--updates',type=int,nargs='+',default=[150,150,300]);a=p.parse_args();os.chdir(R)
statepath=S/(a.branch+'_runner.json')
if statepath.exists():raise RuntimeError('Refusing to overwrite an existing branch')
deadline=json.loads((S/'authorization.json').read_text())['deadline_epoch']
lock=open(R/'logs/cpg_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'status':'running','branch':a.branch,'start':time.time(),'deadline':deadline,'source':str(a.checkpoint),'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'updates':0,'jobs':[],'gates':[],'hardware':False}
def write(p,x):
 tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def save():write(statepath,state)
def run(name,cmd,timeout=300,allow_fault=False):
 if time.time()+timeout>deadline-240:raise TimeoutError('Reserve final four minutes for handoff/cleanup')
 job={'name':name,'command':cmd,'start':time.time()};state['jobs'].append(job);state['current']=name;save();print('START',name,flush=True)
 with (S/(name+'.log')).open('w') as f:
  q=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),),env={**os.environ,'CPG_DEADLINE':str(deadline-300),'PYTHONUNBUFFERED':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})
  job['pid']=q.pid;save()
  try:q.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(q.pid,signal.SIGUSR1)
   try:q.wait(timeout=15)
   except subprocess.TimeoutExpired:
    os.killpg(q.pid,signal.SIGTERM)
    try:q.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(q.pid,signal.SIGKILL);q.wait()
   raise TimeoutError(name)
 job.update(returncode=q.returncode,end=time.time());save();print('END',name,q.returncode,flush=True)
 if q.returncode and not allow_fault:raise RuntimeError(name+' failed')
def audit(d):
 for script,args in [('analyze_cpg_contacts.py',['--dance','--swing','.2','--tolerance','.06']),('audit_cpg_kicks.py',[])]:run(d.name+'_'+script,[PY,'scripts/'+script,str(d),*args],30)
 r=json.loads((d/'result.json').read_text());k=json.loads((d/'kick_audit.json').read_text())['legs'];c=json.loads((d/'contact_audit.json').read_text())['cases']
 import numpy as np
 x=np.load(d/'traces.npz');position=x['body_position'][:,0];neutral_net=(position[-1,:2]-position[0,:2]).tolist()
 return {'status':r.get('status','complete'),'fault':r.get('fault'),'falls':r['falls'],'steps':[sum(v['correct'] for v in c.values()),sum(v['expected'] for v in c.values())],'kicks':[sum(v['qualified_chain'] for v in k.values()),sum(v['expected'] for v in k.values())],'net_xy':neutral_net,'tracking':r.get('travel_only_tracking') or json.loads((d/'tracking_audit.json').read_text())}
def gate(label,checkpoint):
 cfg=json.loads((S/'eval.json').read_text());cfg['gait']=json.loads(a.settings.read_text())['gait'];config=S/(label+'_eval_settings.json');write(config,cfg);d=S/(label+'_isaac')
 run(d.name,[PY,'scripts/cpg_experiment.py','--headless','--task','Unitree-Go2-CPG-Dance','--settings',str(config),'--checkpoint',str(checkpoint),'--output',str(d),'--num_envs','32','--seed','9101','--iterations','0','--steps','2000'],180)
 isaac=audit(d);sel={'checkpoint':str(checkpoint),'deploy_metadata':str(d/'params/deploy.yaml'),'onnx_export':str(d/'export/policy.onnx'),'jit_export':str(d/'export/policy.pt'),'settings':str(config),'hardware_compatible':False,'joystick_limits':[.12,.06,.1],'onnx_sha256':hashlib.sha256((d/'export/policy.onnx').read_bytes()).hexdigest(),'observation_dim':75,'actuator_profile':'go2hv_mujoco_v1'}
 sp=S/(label+'_selection.json');write(sp,sel);cases=[]
 for i,v in enumerate(cfg['evaluation_commands']):
  d=S/(label+'_mj_'+str(i));run(d.name,[PY,'scripts/cpg_mujoco.py','--selection',str(sp),'--output',str(d),'--seconds','40','--input','fixed','--velocity',*map(str,v),'--first-front','FL' if i%2==0 else 'FR'],60,True)
  cases.append({'command':v,**audit(d)})
 result={'label':label,'selection':str(sp),'checkpoint':str(checkpoint),'isaac':isaac,'mujoco':cases};write(S/(label+'_gate.json'),result);state['gates'].append(result);save()
try:
 snapshot=S/(a.branch+'_source');snapshot.mkdir()
 for name in ['source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/cpg','deploy/cpg']:
  shutil.copytree(R/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
 for name in ['scripts/cpg_experiment.py','scripts/cpg_mujoco.py','scripts/audit_cpg_kicks.py','scripts/analyze_cpg_contacts.py']:
  p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
 shutil.copy2(__file__,snapshot/'runner.py');write(S/(a.branch+'_source_manifest.json'),{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*') if p.is_file()})
 checkpoint=a.checkpoint
 for idx,updates in enumerate(a.updates):
  name=f'{a.branch}_train{idx+1}';cfg=json.loads(a.settings.read_text());cfg['fresh_optimizer']=idx==0;settings=S/(name+'_settings.json');write(settings,cfg);d=S/name
  run(name,[PY,'scripts/cpg_experiment.py','--headless','--task','Unitree-Go2-CPG-Dance','--settings',str(settings),'--checkpoint',str(checkpoint),'--output',str(d),'--num_envs','4096','--iterations',str(updates),'--seed',str(101+idx)],updates*2+100)
  r=json.loads((d/'result.json').read_text())
  if r['iterations']!=updates or r['budget_stop']:raise RuntimeError('Training segment incomplete')
  checkpoint=d/'training'/f"model_{r['last_index']}.pt";state['updates']+=updates;save();gate(f'{a.branch}_{state["updates"]}',checkpoint)
 assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==state['source_sha256']
 state.update(status='complete',end=time.time());save()
except Exception as e:
 state.update(status='failed',error=repr(e),end=time.time());save();raise
print('COMPLETE',a.branch,flush=True)
