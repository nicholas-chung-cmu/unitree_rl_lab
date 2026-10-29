"""Score exact introductory kick rewards and measure saved kicks without learning."""
import argparse,ast,csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
parser=argparse.ArgumentParser();parser.add_argument('--rollout',type=Path);parser.add_argument('--refined',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
root=Path('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
refmod=module('kick_intro_ref',root/'conga_reference.py');scoremod=module('kick_intro_score',root/'kick_scores.py')
cfg=refmod.CongaReferenceCfg(kick_lateral=.02,kick_height=.02,kick_scale=1.,bounce_scale=0.)
t=torch.arange(0,4,.005,dtype=torch.float64);ref=refmod.generate_kick_only_reference(t,cfg);n=len(t);rows=torch.arange(n);leg=ref.kick_leg
nominal=torch.tensor([cfg.nominal_front_y,-cfg.nominal_front_y,.18,-.18],dtype=torch.float64).expand(n,-1).clone()
weights=torch.tensor([2.,2.,2. if args.refined else .5,-1.])
out={'weights':weights.tolist(),'target_lateral_m':.02,'target_clearance_m':.02,'synthetic':{}}
for name,fraction,bad_support in [('target',1.,False),('half',.5,False),('quarter',.25,False),('standing',0.,False),('tiny_departure',.05,False),('zero_clearance_contact_flicker',0.,False),('target_plus_wrong_foot_lift',1.,True),('standing_plus_wrong_foot_lift',0.,True)]:
    y=nominal.clone();h=ref.foot_height*fraction
    y[rows,leg]=nominal[rows,leg]+fraction*(ref.kick_y-nominal[rows,leg])
    c=torch.ones(n,4,dtype=torch.bool)
    if fraction>0 or name=='zero_clearance_contact_flicker':c[rows,leg]=ref.contact[rows,leg]>=.5
    if bad_support:c[rows,(leg+1)%4]=ref.kick_weight<.5
    values=scoremod.components(y,h,c,ref,height_aware=args.refined)*weights
    out['synthetic'][name]={'components_mean':values.mean(0).tolist(),'combined_rate':float(values.sum(-1).mean())}
s=out['synthetic'];assert s['target']['combined_rate']>s['half']['combined_rate']>s['quarter']['combined_rate']>s['standing']['combined_rate']
assert s['target']['combined_rate']>s['target_plus_wrong_foot_lift']['combined_rate']
assert s['standing']['combined_rate']>s['standing_plus_wrong_foot_lift']['combined_rate']
if args.refined:
    assert s['quarter']['combined_rate']>s['tiny_departure']['combined_rate']
    assert s['standing']['combined_rate']>=s['zero_clearance_contact_flicker']['combined_rate']
output=Path('logs/conga_kick_refine_scoring' if args.refined else 'logs/conga_kick_intro_scoring');output.mkdir(parents=True,exist_ok=True)
if args.rollout:
    data=list(csv.DictReader(args.rollout.open()));valid=[r for r in data if r['episode_time_s'] and float(r['episode_time_s'])>=2]
    active=[r for r in valid if r['kick_active']=='True'];report={}
    for leg,name in [(0,'FL'),(1,'FR')]:
        samples=[r for r in active if int(r['kick_leg'])==leg]
        if not samples:report[name]={'windows':0,'qualified':0};continue
        time=torch.tensor([float(r['episode_time_s'])+float(r.get('phase_offset_s',0)) for r in samples],dtype=torch.float64)
        q=refmod.generate_kick_only_reference(time,cfg)
        error=np.array([float(r['kick_y_error_m']) for r in samples]);he=np.array([float(r['kick_height_error_m']) for r in samples])
        y=q.kick_y.numpy()+error;outward=(1 if leg==0 else -1)*y-cfg.nominal_front_y
        height=np.array([ast.literal_eval(r['foot_clearance_FL_FR_RL_RR_m'])[leg] for r in samples])
        contact=np.array([ast.literal_eval(r['actual_contact_FL_FR_RL_RR']) for r in samples],bool)
        support=np.delete(contact,leg,axis=1).all(-1)
        groups={}
        for i,r in enumerate(samples):groups.setdefault((r['env'],int(r['step'])//100),[]).append(i)
        qualified=0;liftoff=0;landings=0;peaks=[]
        for ids in groups.values():
            qualified+=bool(((outward[ids]>=.01)&(height[ids]>=.01)&support[ids]&~contact[ids,leg]).any())
            liftoff+=bool((~contact[ids,leg]).any());peaks.append([float(outward[ids].max()),float(height[ids].max())])
            first=samples[ids[0]];event_step=int(first['step']);end=float(first['episode_time_s'])-float(first['episode_time_s'])%2+1.5
            segment=[r for r in valid if r['env']==first['env'] and abs(int(r['step'])-event_step)<35]
            for a,b in zip(segment[:-1],segment[1:]):
                if int(b['step'])!=int(a['step'])+1 or float(b['episode_time_s'])<=float(a['episode_time_s']):continue
                if abs(float(b['episode_time_s'])-(end-.02))>.080001:continue
                if not ast.literal_eval(a['actual_contact_FL_FR_RL_RR'])[leg] and ast.literal_eval(b['actual_contact_FL_FR_RL_RR'])[leg]:landings+=1;break
        report[name]={'windows':len(groups),'qualified':qualified,'qualification':'same sample >=1 cm outward, >=1 cm clearance, requested foot airborne, other three in contact; diagnostic, not final-kick acceptance','windows_with_liftoff':liftoff,'timely_touchdowns':landings,'lateral_mae_m':float(abs(error).mean()),'height_mae_m':float(abs(he).mean()),'three_support_fraction':float(support.mean()),'airborne_fraction':float((~contact[:,leg]).mean()),'mean_peak_outward_m':float(np.mean(peaks,axis=0)[0]),'mean_peak_clearance_m':float(np.mean(peaks,axis=0)[1])}
    if valid:
        times=torch.tensor([float(r['episode_time_s'])+float(r.get('phase_offset_s',0)) for r in valid],dtype=torch.float64)
        q=refmod.generate_kick_only_reference(times,cfg);idx=torch.arange(len(valid))
        y=torch.zeros(len(valid),4,dtype=torch.float64)
        y[idx,q.kick_leg]=q.kick_y+torch.tensor([float(r['kick_y_error_m']) for r in valid],dtype=torch.float64)
        h=torch.tensor([ast.literal_eval(r['foot_clearance_FL_FR_RL_RR_m']) for r in valid],dtype=torch.float64)
        c=torch.tensor([ast.literal_eval(r['actual_contact_FL_FR_RL_RR']) for r in valid],dtype=torch.bool)
        replay=scoremod.components(y,h,c,q,height_aware=args.refined)*weights
        out['replay_kick_component_rates']=replay.mean(0).tolist()
        out['replay_combined_rate']=float(replay.sum(-1).mean())
    quiet=[r for r in valid if r['kick_active']=='False']
    out['outside_kick_four_contact_fraction']=float(np.mean([all(ast.literal_eval(r['actual_contact_FL_FR_RL_RR'])) for r in quiet])) if quiet else None
    out['xy_drift_mps']=float(np.mean([float(r['velocity_xy_error_mps']) for r in valid])) if valid else None
    out['slip_mps']=float(np.mean([float(r['slip_mps']) for r in valid])) if valid else None
    out.update(legs=report,falls=sum(r['fall']=='True' for r in data),timeouts=sum(not r['episode_time_s'] and r['fall']=='False' for r in data),qualified_total=sum(r['qualified'] for r in report.values()),windows_total=sum(r['windows'] for r in report.values()))
    destination=args.output or args.rollout.parent
    destination.mkdir(exist_ok=True,parents=True)
    (destination/'cadence_metrics.json').write_text(json.dumps(out,indent=2)+'\n')
(output/'scores.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
