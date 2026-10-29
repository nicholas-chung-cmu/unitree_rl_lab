"""Audit the normalized paired scheme without changing or training a policy."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
mdp=root/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp'
def load(name):
    spec=importlib.util.spec_from_file_location(name,mdp/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
refmod=load('conga_reference');score=load('cadence_pair_scores')
torch.set_num_threads(2)
weights=torch.tensor([.1,.5,.5],dtype=torch.float64)
out={'weights':weights.tolist(),'schedules':{},'rollouts':[]}
for duration in [.36,.2]:
    cfg=refmod.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,
        cadence_swing_start=.22-duration/2,cadence_swing_end=.22+duration/2)
    t=torch.arange(200,dtype=torch.float64)*.02
    ref=refmod.generate_reference(t,cfg)
    c=(ref.contact>=.5-1e-6).double();h=ref.foot_height
    norms=score.raw_scores(ref.contact,h,c,h).mean(0)
    def evaluate(times,contacts,heights):
        r=refmod.generate_reference(times,cfg)
        return score.normalized_scores(score.raw_scores(r.contact,r.foot_height,contacts,heights),norms)
    patterns={'reference':(c,h),'planted':(torch.ones_like(c),torch.zeros_like(h)),
              'wrong_diagonal':(c[:,[1,0,3,2]],h[:,[1,0,3,2]]),
              'flight':(torch.zeros_like(c),h)}
    for foot in range(4):
        cs=torch.ones_like(c);hs=torch.zeros_like(h);cs[:,foot]=c[:,foot];hs[:,foot]=h[:,foot]
        patterns['only_'+['FL','FR','RL','RR'][foot]]=(cs,hs)
    for offset in [0,.25,.5]:
        r=refmod.generate_reference(2*t+offset,cfg)
        patterns['double_frequency_'+str(offset)]=((r.contact>=.5).double(),r.foot_height)
    syn={}
    for name,(cs,hs) in patterns.items():
        values=evaluate(t,cs,hs)
        syn[name]={'components':values.mean(0).tolist(),'weighted_mean':float((values@weights).mean()),
                   'peak_rate':float((values@weights).max())}
    torch.testing.assert_close(evaluate(t,c,h).mean(0),torch.ones(3,dtype=torch.float64))
    assert syn['reference']['weighted_mean']>syn['planted']['weighted_mean']
    for name in ['planted','only_FL','only_FR','only_RL','only_RR','flight','wrong_diagonal']:
        assert syn[name]['components'][1]<1e-10
    out['schedules'][str(duration)]={'normalizers':norms.tolist(),'synthetic':syn}
    if duration!=.2:continue
    for directory in ['logs/conga_cadence_reward_audit','logs/conga_cadence_eval_final']:
        meta=json.loads((root/directory/'summary.json').read_text())
        data=np.genfromtxt(root/directory/'rollout.csv',delimiter=',',names=True)
        rows=data[(data['done']==0)&(data['episode_time_s']>=2)]
        times=torch.tensor(rows['episode_time_s'],dtype=torch.float64)
        cs=torch.tensor(np.stack([rows['contact_'+f] for f in ['FL','FR','RL','RR']],-1))
        hs=torch.tensor(np.stack([rows['clearance_'+f] for f in ['FL','FR','RL','RR']],-1))
        vals=evaluate(times,cs,hs)
        cases=[]
        for case in meta['cases']:
            v=vals[rows['env']==case['case']]
            cases.append({'command':case['command'],'components':v.mean(0).tolist(),
                'weighted_mean':float((v@weights).mean()),'paired_nonzero_fraction':float((v[:,1]>1e-6).double().mean())})
        out['rollouts'].append({'directory':directory,'checkpoint':meta['checkpoint'],'cases':cases})
p=root/'logs/conga_paired_scheme';p.mkdir(exist_ok=True)
(p/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
