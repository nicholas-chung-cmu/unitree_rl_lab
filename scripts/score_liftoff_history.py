"""Offline last-diagonal gate on the existing paired-clearance reward."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FEET=['FL','FR','RL','RR']
PAIRS=([0,3],[1,2])


def history_gate(t, contact, done=None, debounce=2, together=.04):
    """Causal 50 Hz detector. First pair seeds history; no event bonus.

    Both feet must transition off within `together` seconds, remain off for
    `debounce` samples, and have touched down together before rearming. Flight
    is ambiguous and seeds neither pair. Eligibility is latched for the swing.
    History uses detections independent of phase, clearance or reward value.
    """
    done=np.zeros(len(t),bool) if done is None else done.astype(bool)
    gate=np.zeros((len(t),2),bool);events=[]
    last=None;prev=None;off=np.full(4,np.nan)
    air=np.zeros(2,int);ground=np.zeros(2,int);armed=np.zeros(2,bool);eligible=np.zeros(2,bool)
    for i in range(len(t)):
        if done[i] or i==0 or done[i-1] or t[i]<=t[i-1] or t[i]-t[i-1]>.021:
            last=None;prev=None;off[:]=np.nan
            air[:]=0;ground[:]=0;armed[:]=False;eligible[:]=False
        if done[i]:continue
        c=contact[i]>.5
        if prev is not None:
            off[prev & ~c]=t[i]
        off[c]=np.nan
        candidates=[]
        for p,feet in enumerate(PAIRS):
            both_air=(~c[feet]).all();both_ground=c[feet].all()
            air[p]=air[p]+1 if both_air else 0
            ground[p]=ground[p]+1 if both_ground else 0
            if ground[p]>=debounce:
                armed[p]=True;eligible[p]=False
            if armed[p] and air[p]>=debounce:
                # Disarm even a rejected asynchronous takeoff until touchdown.
                armed[p]=False
                if np.isfinite(off[feet]).all() and np.ptp(off[feet])<=together+1e-5:
                    candidates.append(p)
        if len(candidates)==1 and c.any():
            p=candidates[0]
            eligible[p]=last is not None and last!=p
            events.append({'index':i,'pair':p,'eligible':bool(eligible[p]),'repeated':last==p})
            last=p
        elif candidates:
            eligible[candidates]=False
        for p,feet in enumerate(PAIRS):
            gate[i,p]=eligible[p] and (~c[feet]).all()
        prev=c
    return gate,events


def load(name):
    path=ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def main():
    torch.set_num_threads(2)
    refmod=load('conga_reference');scores=load('cadence_pair_scores')
    cfg=refmod.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,
                               cadence_swing_start=.12,cadence_swing_end=.32)
    def reference(t):
        return refmod.generate_reference(torch.tensor(t,dtype=torch.float64),cfg)
    ideal=reference(np.arange(200)*.02)
    norm=scores.raw_scores(ideal.contact,ideal.foot_height,(ideal.contact>=.5-1e-6).double(),ideal.foot_height).mean(0)
    def evaluate(t,c,h,done=None,debounce=2,together=.04):
        ref=reference(t)
        raw=scores.raw_scores(ref.contact,ref.foot_height,torch.tensor(c,dtype=torch.float64),torch.tensor(h,dtype=torch.float64))
        old=scores.normalized_scores(raw,norm)[:,1].numpy()
        gate,events=history_gate(t,c,done,debounce,together)
        pair=np.where(ref.foot_height[:,0].numpy()>=ref.foot_height[:,1].numpy(),0,1)
        new=old*gate[np.arange(len(t)),pair]
        valid=(t>=2)&(np.ones(len(t),bool) if done is None else ~done.astype(bool))
        scored_events=[e for e in events if valid[e['index']]]
        return {'samples':int(valid.sum()),'old_pair_mean':float(old[valid].mean()),
                'history_pair_mean':float(new[valid].mean()),
                'old_nonzero_fraction':float((old[valid]>1e-6).mean()),
                'history_nonzero_fraction':float((new[valid]>1e-6).mean()),
                'detected_liftoffs':len(scored_events),
                'eligible_liftoffs':sum(e['eligible'] for e in scored_events)}
    t=np.arange(1000)*.02;r=reference(t)
    c=(r.contact.numpy()>=.5-1e-6).astype(float);h=r.foot_height.numpy()
    patterns={'correct':(c,h),'standing':(np.ones_like(c),np.zeros_like(h)),
              'wrong_phase':(c[:,[1,0,3,2]],h[:,[1,0,3,2]]),
              'flight':(np.zeros_like(c),h)}
    for p,feet in enumerate(PAIRS):
        cc=np.ones_like(c);hh=np.zeros_like(h)
        for f in feet:cc[:,f]=(c>.5).all(-1);hh[:,f]=h.max(-1)
        patterns[f'repeated_pair_{p}']=(cc,hh)
    for offset in (0,.125,.25,.375,.5):
        rr=reference(2*t+offset)
        patterns[f'double_frequency_{offset}']=((rr.contact.numpy()>=.5-1e-6).astype(float),rr.foot_height.numpy())
    cc=np.ones_like(c);hh=np.zeros_like(h);cc[::2]=c[::2];hh[::2]=h[::2]
    patterns['chatter']=(cc,hh)
    synthetic={name:evaluate(t,cs,hs) for name,(cs,hs) in patterns.items()}
    assert synthetic['correct']['history_pair_mean']>.95
    for name in ('standing','flight','repeated_pair_0','repeated_pair_1','chatter'):
        assert synthetic[name]['history_pair_mean']==0
    # A,A,B,A must yield ineligible,ineligible,eligible,eligible, regardless of
    # whether any event would earn a phase/clearance reward.
    ct=np.ones((100,4));tt=np.arange(100)*.02
    for start,p in zip((5,30,55,80),(0,0,1,0)):ct[start:start+5,PAIRS[p]]=0
    _,events=history_gate(tt,ct)
    assert [e['eligible'] for e in events]==[False,False,True,True]
    reset=np.zeros(100,bool);reset[50]=True
    _,events=history_gate(tt,ct,reset)
    assert [e['eligible'] for e in events]==[False,False,False,True]
    rollouts=[]
    directories=json.loads((ROOT/'logs/conga_alternation_scoring/scores.json').read_text())['rollouts']
    for source in directories:
        directory=source['directory']
        data=np.genfromtxt(ROOT/directory/'rollout.csv',delimiter=',',names=True)
        meta=json.loads((ROOT/directory/'summary.json').read_text())
        cases=[]
        for case in meta['cases']:
            rows=data[data['env']==case['case']]
            cs=np.stack([rows['contact_'+f] for f in FEET],-1)
            hs=np.stack([rows['clearance_'+f] for f in FEET],-1)
            args=(rows['episode_time_s'],cs,hs,rows['done'])
            result=evaluate(*args)
            result['sensitivity']={f'{debounce*20}ms_{together*1000:.0f}ms':evaluate(*args,debounce,together)
                for debounce in (2,3) for together in (.02,.04,.06)}
            cases.append({'case':case['case'],'command':case['command'],**result})
        rollouts.append({'directory':directory,'cases':cases})
    out={'definition':{'dt':.02,'debounce_samples':2,'liftoff_separation_s':.04,
         'normalization':'unchanged paired-reference normalizer; gate only removes credit',
         'candidate_weight':.1,'warmup_s':2},'synthetic':synthetic,'rollouts':rollouts}
    dest=ROOT/'logs/conga_liftoff_history_scoring';dest.mkdir(exist_ok=True)
    (dest/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,result in synthetic.items():print(name,result)
    for run in rollouts:
        cases=run['cases'];n=sum(c['samples'] for c in cases)
        print(run['directory'],{k:sum(c[k]*c['samples'] for c in cases)/n for k in
              ('old_pair_mean','history_pair_mean','old_nonzero_fraction','history_nonzero_fraction')})


if __name__=='__main__':main()
