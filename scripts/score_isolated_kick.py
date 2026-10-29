"""Isolated-kick reference scoring and optional saved-policy measurement audit."""
import argparse,ast,csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
parser=argparse.ArgumentParser();parser.add_argument('--rollout',type=Path);args=parser.parse_args()
p=Path('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/conga_reference.py')
spec=importlib.util.spec_from_file_location('kick_ref',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cfg=m.CongaReferenceCfg(kick_scale=1.,bounce_scale=0.)
t=torch.arange(0,4,.005,dtype=torch.float64);r=m.generate_kick_only_reference(t,cfg)
rows=torch.arange(len(t));mask=torch.arange(4)[None,:]==r.kick_leg[:,None]
nominal=torch.where(r.kick_leg==0,cfg.nominal_front_y,-cfg.nominal_front_y)
out={'reference':{'kick_duration_s':cfg.kick_duration,'full_lateral_m':cfg.kick_lateral,'full_clearance_m':cfg.kick_height,'first_leg':'FR','second_leg':'FL','walking_swings':False,'bounce':False},'synthetic':{}}
for name,fraction in [('target',1.),('half_extension',.5),('quarter_extension',.25),('standing',0.)]:
    y=nominal+fraction*(r.kick_y-nominal)
    h=fraction*r.foot_height[rows,r.kick_leg]
    contact=torch.ones_like(r.contact)
    if fraction>0:contact=torch.where(mask & (r.contact<.5),0.,contact)
    reward_contact=1-((2*r.contact-1).square()*(contact-r.contact).square()).mean(-1)
    reward_kick=r.kick_weight*torch.exp(-((y-r.kick_y)/.035).square()-((h-r.foot_height[rows,r.kick_leg])/.025).square())
    out['synthetic'][name]={'contact_rate':float(reward_contact.mean()),'kick_rate':float(reward_kick.mean()),'combined_rate':float((reward_contact+reward_kick).mean()),'kick_window_tracking_quality':float(reward_kick.sum()/r.kick_weight.sum())}
assert out['synthetic']['target']['combined_rate']>out['synthetic']['half_extension']['combined_rate']>out['synthetic']['quarter_extension']['combined_rate']>out['synthetic']['standing']['combined_rate']
# Endpoint/peak derivatives and both-leg symmetry of the sin-squared path.
for times in [[1.25,1.375,1.5],[3.25,3.375,3.5]]:
    x=torch.tensor(times,dtype=torch.float64,requires_grad=True);q=m.generate_kick_only_reference(x,cfg)
    dy=torch.autograd.grad(q.kick_y.sum(),x,retain_graph=True)[0]
    dz=torch.autograd.grad(q.foot_height.sum(),x)[0]
    assert dy.abs().max()<1e-12 and dz.abs().max()<1e-12
out['endpoint_velocity_check']='passed at start, peak and end for both legs'
if args.rollout:
    data=list(csv.DictReader(args.rollout.open()))
    valid=[r for r in data if r['episode_time_s'] and float(r['episode_time_s'])>=2]
    active=[r for r in valid if r['kick_active']=='True']
    report={'rows':len(data),'falls':sum(r['fall']=='True' for r in data),'legs':{}}
    for leg,name in [(0,'FL'),(1,'FR')]:
        samples=[r for r in active if int(r['kick_leg'])==leg]
        contacts=np.array([ast.literal_eval(r['actual_contact_FL_FR_RL_RR']) for r in samples],bool)
        heights=np.array([ast.literal_eval(r['foot_clearance_FL_FR_RL_RR_m'])[leg] for r in samples])
        lateral_error=np.array([float(r['kick_y_error_m']) for r in samples]);height_error=np.array([float(r['kick_height_error_m']) for r in samples])
        support=np.delete(contacts,leg,axis=1).all(-1)
        event_groups={}
        for s in samples:
            event_groups.setdefault((s['env'],int(s['step'])//100),[]).append(s)
        completed=0
        for event in event_groups.values():
            c=[ast.literal_eval(s['actual_contact_FL_FR_RL_RR'])[leg] for s in event]
            completed+=not all(c)
        times=torch.tensor([float(s['episode_time_s']) for s in samples],dtype=torch.float64)
        q=m.generate_kick_only_reference(times,cfg)
        actual_y=q.kick_y.numpy()+lateral_error
        sign=1 if leg==0 else -1
        outward=sign*actual_y-cfg.nominal_front_y
        landings=0
        for group in event_groups.values():
            first=group[0];event_step=int(first['step'])
            desired_end=float(first['episode_time_s'])-float(first['episode_time_s'])%2+1.5
            group_all=[s for s in valid if s['env']==first['env'] and abs(int(s['step'])-event_step)<35]
            for a,b in zip(group_all[:-1],group_all[1:]):
                if int(b['step'])-int(a['step'])!=1 or float(b['episode_time_s'])<=float(a['episode_time_s']):continue
                if abs(float(b['episode_time_s'])-(desired_end-.02))>.080001:continue
                if not ast.literal_eval(a['actual_contact_FL_FR_RL_RR'])[leg] and ast.literal_eval(b['actual_contact_FL_FR_RL_RR'])[leg]:
                    landings+=1;break
        report['legs'][name]={'peak_outward_extension_m':float(outward.max()),'timely_touchdowns':landings,'missing_touchdowns':len(event_groups)-landings,'kick_samples':len(samples),'scheduled_kick_windows':len(event_groups),'windows_with_any_liftoff':completed,'lateral_mae_m':float(abs(lateral_error).mean()),'height_mae_m':float(abs(height_error).mean()),'kick_foot_airborne_fraction':float((~contacts[:,leg]).mean()),'three_support_fraction':float(support.mean()),'peak_observed_clearance_m':float(heights.max()),'mean_observed_clearance_m':float(heights.mean())}
    out['policy_probe']=report
outdir=Path('logs/conga_kick_isolation_scoring');outdir.mkdir(exist_ok=True,parents=True)
(outdir/'scores.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
