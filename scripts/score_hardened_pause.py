"""Audit the actual runtime pause/repetition implementation without Isaac Sim."""
import json
import sys
import numpy as np
import torch
from score_liftoff_history import ROOT, FEET, load


def main():
    march_first='--march-first' in sys.argv
    torch.set_num_threads(2)
    refs=load('conga_reference');scores=load('cadence_pair_scores');pause=load('pause_rewards')
    cfg=refs.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,cadence_swing_start=.12,cadence_swing_end=.32)
    def ref(t):return refs.generate_reference(torch.tensor(t,dtype=torch.float64),cfg)
    rt=np.arange(200)*.02;r=ref(rt)
    norm=scores.raw_scores(r.contact,r.foot_height,(r.contact>=.5-1e-6).double(),r.foot_height).mean(0)
    pn=pause.pause_weight(torch.tensor(rt)).mean()
    march=load('march_rewards')
    mn=march.raw_terms(r.contact,r.foot_height,(r.contact>=.5-1e-6).double(),r.foot_height).mean(0)[[0,2]]
    def evaluate(t,c,h,done=None):
        # time, env, foot. Preserve per-environment state and asynchronous resets.
        n=c.shape[1];state=pause.StepHistory(n,'cpu')
        times=torch.tensor(t);contacts=torch.tensor(c).bool();heights=torch.tensor(h)
        gates=[];events=[]
        for i in range(len(t)):
            gate,event=state.update(contacts[i],heights[i],torch.round(times[i]/.02).long())
            gates.append(gate.clone());events.append(event.clone())
        gates=torch.stack(gates);events=torch.stack(events)
        r=ref(t.reshape(-1));flatc=contacts.reshape(-1,4).double();flath=heights.reshape(-1,4)
        base=scores.normalized_scores(scores.raw_scores(r.contact,r.foot_height,flatc,flath),norm)
        pair=torch.where(r.foot_height[:,0]>=r.foot_height[:,1],0,1)
        eligibility=gates.reshape(-1,2).gather(1,pair[:,None]).squeeze(1)
        phase=(.25*base[:,2]).reshape(-1,n)
        quality=(.1*base[:,1]*eligibility).reshape(-1,n)
        missing=.25*pause.pause_weight(times)/pn*(~contacts).double().mean(-1)
        repeat=.025*events/.02
        individual=torch.zeros_like(phase);missed_swing=torch.zeros_like(phase)
        if march_first:
            phase=base[:,2].reshape(-1,n)
            quality=(.25*base[:,1]).reshape(-1,n)
            raw=march.raw_terms(r.contact,r.foot_height,flatc,flath)
            individual=(.25*raw[:,0]/mn[0]).reshape(-1,n)
            missed_swing=(.5*raw[:,1]/mn[1]).reshape(-1,n)
        valid=times>=2
        if done is not None:valid &= ~torch.tensor(done).bool()
        results=[]
        for j in range(n):
            mask=valid[:,j]
            results.append({name:float(value[:,j][mask].mean()) for name,value in
                [('phase',phase),('paired',quality),('individual_clearance',individual),('missed_swing_cost',missed_swing),('pause_cost',missing),('repeat_cost',repeat),('total',phase+quality+individual-missed_swing-missing-repeat)]})
        return results
    t=np.arange(1000)*.02;r=ref(t);c=(r.contact.numpy()>=.5-1e-6).astype(float);h=r.foot_height.numpy()
    patterns={'correct':(c,h),'standing':(np.ones_like(c),np.zeros_like(h))}
    for p,feet in enumerate(([0,3],[1,2])):
        cc=np.ones_like(c);hh=np.zeros_like(h)
        for f in feet:cc[:,f]=(c>.5).all(-1);hh[:,f]=h.max(-1)
        patterns[f'repeat_{p}']=(cc,hh)
        for delay in (3,10):
            sc=cc.copy();sh=hh.copy();f=feet[1]
            sc[:,f]=np.r_[np.ones(delay),cc[:-delay,f]];sh[:,f]=np.r_[np.zeros(delay),hh[:-delay,f]]
            patterns[f'stagger_{p}_{delay*20}ms']=(sc,sh)
    for foot in range(4):
        cc=np.ones_like(c);hh=np.zeros_like(h);cc[:,foot]=c[:,foot];hh[:,foot]=h[:,foot]
        patterns['single_'+FEET[foot]]=(cc,hh)
    cc=np.ones_like(c);hh=np.zeros_like(h);cc[::2]=c[::2];hh[::2]=h[::2]
    patterns['chatter']=(cc,hh)
    # Opposite low-clearance wiggles must not clear a repeatedly used pair.
    cc,hh=patterns['repeat_0'];cc=cc.copy();hh=hh.copy()
    low=(((t+.25)%.5)>.16)&(((t+.25)%.5)<.28)
    cc[low,1]=0;cc[low,2]=0;hh[low,1]=.001;hh[low,2]=.001
    patterns['opposite_1mm_wiggle']=(cc,hh)
    for offset in (0,.125,.25,.375,.5):
        rr=ref(2*t+offset)
        patterns[f'fast_{offset}']=((rr.contact.numpy()>=.5-1e-6).astype(float),rr.foot_height.numpy())
    values=evaluate(np.tile(t[:,None],(1,len(patterns))),np.stack([x[0] for x in patterns.values()],1),np.stack([x[1] for x in patterns.values()],1))
    synthetic=dict(zip(patterns,values))
    assert synthetic['correct']['repeat_cost']==0 and synthetic['correct']['pause_cost']<1e-8
    assert synthetic['repeat_0']['repeat_cost']>=.099 and synthetic['repeat_1']['repeat_cost']>=.099
    assert synthetic['chatter']['repeat_cost']>0
    assert synthetic['opposite_1mm_wiggle']['repeat_cost']>=.099
    assert all(v['repeat_cost']>0 for k,v in synthetic.items() if k.startswith(('stagger','single')))
    assert synthetic['correct']['total']>max(v['total'] for k,v in synthetic.items() if k!='correct')
    if march_first:
        assert abs(synthetic['correct']['total']-1.5)<1e-6
        assert synthetic['standing']['total']<0
        # Dense progress survives before full pair completion or alternation.
        assert synthetic['single_FL']['individual_clearance']>0
        assert synthetic['single_FL']['total']>synthetic['standing']['total']
    directory=ROOT/('logs/conga_pause_4096_experiment/eval_2200' if march_first else 'logs/conga_history_1024_experiment/eval_1200')
    data=np.genfromtxt(directory/'rollout.csv',delimiter=',',names=True)
    meta=json.loads((directory/'summary.json').read_text())
    rows=[data[data['env']==case['case']] for case in meta['cases']]
    result=evaluate(np.stack([v['episode_time_s'] for v in rows],1),
        np.stack([np.stack([v['contact_'+f] for f in FEET],-1) for v in rows],1),
        np.stack([np.stack([v['clearance_'+f] for f in FEET],-1) for v in rows],1),
        np.stack([v['done'] for v in rows],1))
    out={'synthetic':synthetic,'rollout':[{'command':case['command'],**v} for case,v in zip(meta['cases'],result)],
         'note':'Cadence-only mean reward rates; actual runtime core. No physical/velocity scores for synthetic motions.'}
    out['task']='Unitree-Go2-Conga-March' if march_first else 'Unitree-Go2-Conga-Pause'
    out['rollout_source']=str(directory)
    if march_first:
        out['sensitivity']={name:{f'phase_{phase}_miss_{miss}':
            v['total']+(phase-1)*v['phase']-(miss/.5-1)*v['missed_swing_cost']
            for phase in (.5,1.,1.5) for miss in (.25,.5,1.)} for name,v in synthetic.items()}
    dest=ROOT/('logs/conga_march_scoring' if march_first else 'logs/conga_hardened_pause_scoring');dest.mkdir(exist_ok=True)
    (dest/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
