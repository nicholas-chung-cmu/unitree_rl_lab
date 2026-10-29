"""Exercise the exact runtime duration-cost core on synthetic and saved contacts."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
p=Path('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/duration_scores.py')
spec=importlib.util.spec_from_file_location('duration_scores',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def score(contacts,active=None,reset=None):
    tracker=m.DurationState(1,'cpu'); values=[]
    for j,c in enumerate(contacts):
        a=True if active is None else bool(active[j])
        r=j==0 if reset is None else bool(reset[j])
        v=tracker.step(torch.tensor(c)[None].bool(),torch.tensor([a]),torch.tensor([r]),.02)
        assert torch.isfinite(v).all() and (v>=0).all() and (v<=1).all()
        if j>=100: values.append(v[0].tolist())
    return np.mean(values,axis=0).tolist()

t=np.arange(2000)*.02
patterns={}
for name,period,swing in [('target',1.,.20),('fast',.5,.20),('long_swing',1.,.40)]:
    c=np.ones((len(t),4),bool)
    for ids,offset in [([0,3],0),([1,2],period/2)]:
        c[:,ids]=~((((t-offset)%period)<swing)[:,None])
    patterns[name]=score(c)
patterns['standing_active']=score(np.ones((len(t),4),bool))
patterns['standing_neutral']=score(np.ones((len(t),4),bool),np.zeros(len(t),bool))
assert max(patterns['target'])<.001
assert patterns['standing_neutral']==[0.,0.]
assert patterns['standing_active'][1]>.9
assert sum(patterns['target'])<sum(patterns['fast'])
# Deactivation clears completed errors; a new partial interval cannot receive short-duration cost.
st=m.DurationState(1,'cpu')
for _ in range(200):st.step(torch.ones(1,4,dtype=torch.bool),torch.tensor([True]),torch.tensor([False]),.02)
assert st.step(torch.ones(1,4,dtype=torch.bool),torch.tensor([False]),torch.tensor([False]),.02).sum()==0
assert st.step(torch.zeros(1,4,dtype=torch.bool),torch.tensor([True]),torch.tensor([False]),.02).sum()==0
out={'synthetic_normalized_costs_air_stance':patterns,'weighted_rates':{k:-.05*sum(v) for k,v in patterns.items()},'replays':{}}
for directory in ['logs/conga_basic_eval_final','logs/conga_march_4096_experiment/eval_1000','logs/conga_march_balanced_4096_experiment/eval_1000','logs/conga_history_1024_experiment/eval_1200']:
    path=Path(directory)/'rollout.csv'
    if not path.exists():continue
    d=np.genfromtxt(path,delimiter=',',names=True);cases={}
    for i in np.unique(d['env']):
        x=d[d['env']==i];valid=x['done']==0;x=x[valid]
        c=np.stack([x['contact_'+f] for f in ['FL','FR','RL','RR']],axis=-1)
        reset=np.r_[True,np.diff(x['episode_time_s'])<0]
        # Counterfactual active command for comparing observed contact patterns.
        cases[str(int(i))]=score(c,reset=reset)
    out['replays'][directory]=cases
p=Path('logs/conga_duration_scoring');p.mkdir(exist_ok=True,parents=True)
(p/'scores.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
