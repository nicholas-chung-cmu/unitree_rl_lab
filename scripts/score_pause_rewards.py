"""Offline pause shaping weight grid; leaves all runtime tasks unchanged."""
import json
from pathlib import Path
import numpy as np
import torch
from score_liftoff_history import history_gate, load, ROOT, FEET


def pause_gate(t):
    # The pause runs from .32 to .62 modulo .5. Soften its first/last 20 ms
    # without penalizing any part of the .12–.32 swing window.
    progress=(np.asarray(t)-.32)% .5
    return np.clip(progress/.02,0,1)*np.clip((.30-progress)/.02,0,1)


def main():
    torch.set_num_threads(2)
    refmod=load('conga_reference');scores=load('cadence_pair_scores')
    cfg=refmod.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,
                               cadence_swing_start=.12,cadence_swing_end=.32)
    def reference(t):return refmod.generate_reference(torch.tensor(t,dtype=torch.float64),cfg)
    rt=np.arange(200)*.02;ref=reference(rt)
    norm=scores.raw_scores(ref.contact,ref.foot_height,(ref.contact>=.5-1e-6).double(),ref.foot_height).mean(0)
    pause_norm=pause_gate(rt).mean()
    grid=[(support,penalty) for support in (0.,.05,.1) for penalty in (.1,.25,.5)]
    def evaluate(t,c,h,done=None):
        r=reference(t)
        base=scores.normalized_scores(scores.raw_scores(r.contact,r.foot_height,torch.tensor(c,dtype=torch.float64),torch.tensor(h,dtype=torch.float64)),norm).numpy()
        eligible,_=history_gate(t,c,done)
        pair=np.where(r.foot_height[:,0].numpy()>=r.foot_height[:,1].numpy(),0,1)
        current=.1*base[:,1]*eligible[np.arange(len(t)),pair]+.25*base[:,2]
        gate=pause_gate(t)/pause_norm
        support=gate*(c>.5).all(-1)
        missing=gate*(c<=.5).mean(-1)
        valid=(t>=2)&(np.ones(len(t),bool) if done is None else ~done.astype(bool))
        components={k:float(v[valid].mean()) for k,v in
                    [('current',current),('pause_support',support),('pause_missing',missing)]}
        return {**components,'scores':{f'support_{a}_penalty_{b}':components['current']+a*components['pause_support']-b*components['pause_missing'] for a,b in grid}}
    t=np.arange(1000)*.02;r=reference(t)
    c=(r.contact.numpy()>=.5-1e-6).astype(float);h=r.foot_height.numpy()
    patterns={'correct':(c,h),'standing':(np.ones_like(c),np.zeros_like(h)),
              'wrong_diagonal':(c[:,[1,0,3,2]],h[:,[1,0,3,2]])}
    for p,feet in enumerate(([0,3],[1,2])):
        cc=np.ones_like(c);hh=np.zeros_like(h)
        for f in feet:cc[:,f]=(c>.5).all(-1);hh[:,f]=h.max(-1)
        patterns[f'repeated_pair_{p}']=(cc,hh)
    for shift in (0,.125,.25,.375,.5):
        rr=reference(2*t+shift)
        patterns[f'double_frequency_{shift}']=((rr.contact.numpy()>=.5-1e-6).astype(float),rr.foot_height.numpy())
    for shift in (-.02,.02):
        rr=reference(t-shift)
        patterns[f'correct_shift_{shift}']=((rr.contact.numpy()>=.5-1e-6).astype(float),rr.foot_height.numpy())
    synthetic={k:evaluate(t,cs,hs) for k,(cs,hs) in patterns.items()}
    assert synthetic['correct']['pause_missing']<1e-8
    assert synthetic['standing']['pause_missing']==0
    for key in synthetic['correct']['scores']:
        assert synthetic['correct']['scores'][key]>synthetic['standing']['scores'][key]
        assert synthetic['correct']['scores'][key]>max(v['scores'][key] for k,v in synthetic.items() if k.startswith('double_frequency'))
    rollouts=[]
    for directory in ['logs/conga_early_1024_experiment/eval_1000','logs/conga_history_1024_experiment/eval_1200']:
        data=np.genfromtxt(ROOT/directory/'rollout.csv',delimiter=',',names=True)
        meta=json.loads((ROOT/directory/'summary.json').read_text())
        cases=[]
        for case in meta['cases']:
            rows=data[data['env']==case['case']]
            contacts=np.stack([rows['contact_'+f] for f in FEET],-1)
            heights=np.stack([rows['clearance_'+f] for f in FEET],-1)
            result=evaluate(rows['episode_time_s'],contacts,heights,rows['done'])
            valid=(rows['episode_time_s']>=2)&(rows['done']==0)
            # Actual audited full reward, with ONLY added pause terms estimated
            # from post-step observations (one control tick alignment uncertainty).
            actual=sum(rows[n][valid].mean() for n in rows.dtype.names if n.startswith('reward_rate_'))
            result['logged_full_reward_rate']=float(actual)
            result['estimated_full_scores']={f'support_{a}_penalty_{b}':float(actual+a*result['pause_support']-b*result['pause_missing']) for a,b in grid}
            cases.append({'case':case['case'],'command':case['command'],**result})
        rollouts.append({'directory':directory,'cases':cases})
    out={'definition':{'swing_s':.2,'pause_s':.3,'pause_edge_ramp_s':.02,
         'pause_normalizer':float(pause_norm),'phase_weight':.25,'history_pair_weight':.1,
         'missing_contact_cost':'fraction of four feet not in contact, averaged over fixed reference pause budget',
         'units':'mean reward rate per simulation second; not episode return'},
         'synthetic':synthetic,'rollouts':rollouts}
    dest=ROOT/'logs/conga_pause_scoring';dest.mkdir(exist_ok=True)
    (dest/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
    key='support_0.05_penalty_0.25'
    for k,v in synthetic.items():print(k,'current',round(v['current'],4),'new',round(v['scores'][key],4),'missing',round(v['pause_missing'],4))
    for case in rollouts[-1]['cases'][:8]:
        print(case['command'],'current',round(case['current'],4),'new',round(case['scores'][key],4),'missing',round(case['pause_missing'],4))


if __name__=='__main__':main()
