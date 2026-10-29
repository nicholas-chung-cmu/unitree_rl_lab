"""Offline event-cost audit combined with the proposed scheduled-pause cost."""
import json
import numpy as np
import torch
from score_liftoff_history import ROOT, FEET, history_gate, load
from score_pause_rewards import pause_gate


def main():
    torch.set_num_threads(2)
    refs=load('conga_reference');scores=load('cadence_pair_scores')
    cfg=refs.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,
                              cadence_swing_start=.12,cadence_swing_end=.32)
    def ref(t):return refs.generate_reference(torch.tensor(t,dtype=torch.float64),cfg)
    rt=np.arange(200)*.02;r=ref(rt)
    norm=scores.raw_scores(r.contact,r.foot_height,(r.contact>=.5-1e-6).double(),r.foot_height).mean(0)
    pn=pause_gate(rt).mean();costs=(.01,.025,.05,.1)
    def evaluate(t,c,h,done=None):
        r=ref(t)
        raw=scores.normalized_scores(scores.raw_scores(r.contact,r.foot_height,torch.tensor(c,dtype=torch.float64),torch.tensor(h,dtype=torch.float64)),norm).numpy()
        gate,events=history_gate(t,c,done)
        pair=np.where(r.foot_height[:,0].numpy()>=r.foot_height[:,1].numpy(),0,1)
        cadence=.25*raw[:,2]+.1*raw[:,1]*gate[np.arange(len(t)),pair]
        missing=pause_gate(t)/pn*(c<=.5).mean(-1)
        valid=(t>=2)&(np.ones(len(t),bool) if done is None else ~done.astype(bool))
        seconds=valid.sum()*.02
        repeated=sum(e['repeated'] for e in events if valid[e['index']])
        frequency=repeated/seconds
        current=float(cadence[valid].mean());pause=float(.25*missing[valid].mean())
        return {'current':current,'pause_cost_rate':pause,'pause_only':current-pause,
                'repeated_liftoffs':int(repeated),'scored_seconds':seconds,'repeats_per_second':frequency,
                'combined':{str(cost):current-pause-cost*frequency for cost in costs}}
    t=np.arange(1000)*.02;r=ref(t);c=(r.contact.numpy()>=.5-1e-6).astype(float);h=r.foot_height.numpy()
    patterns={'correct':(c,h),'standing':(np.ones_like(c),np.zeros_like(h)),
              'wrong_phase':(c[:,[1,0,3,2]],h[:,[1,0,3,2]]),
              'flight':(np.zeros_like(c),h)}
    for p,feet in enumerate(([0,3],[1,2])):
        cc=np.ones_like(c);hh=np.zeros_like(h)
        for f in feet:cc[:,f]=(c>.5).all(-1);hh[:,f]=h.max(-1)
        patterns[f'repeat_pair_{p}']=(cc,hh)
        # A deliberately staggered repeated diagonal avoids the <=40 ms detector.
        cc2=cc.copy();hh2=hh.copy();f=feet[1]
        cc2[:,f]=np.r_[np.ones(3),cc[:-3,f]]
        hh2[:,f]=np.r_[np.zeros(3),hh[:-3,f]]
        patterns[f'staggered_repeat_{p}']=(cc2,hh2)
    for f in range(4):
        cc=np.ones_like(c);hh=np.zeros_like(h);cc[:,f]=c[:,f];hh[:,f]=h[:,f]
        patterns[f'single_{FEET[f]}']=(cc,hh)
    for offset in (0,.125,.25,.375,.5):
        r=ref(2*t+offset)
        patterns[f'double_frequency_{offset}']=((r.contact.numpy()>=.5-1e-6).astype(float),r.foot_height.numpy())
    cc=np.ones_like(c);hh=np.zeros_like(h);cc[::2]=c[::2];hh[::2]=h[::2]
    patterns['chatter']=(cc,hh)
    synthetic={n:evaluate(t,cs,hs) for n,(cs,hs) in patterns.items()}
    assert synthetic['correct']['repeated_liftoffs']==0
    assert synthetic['repeat_pair_0']['repeats_per_second']==2
    assert synthetic['repeat_pair_1']['repeats_per_second']==2
    assert all(synthetic['correct']['combined'][str(cost)]>max(v['combined'][str(cost)] for k,v in synthetic.items() if k!='correct') for cost in costs)
    # Verify first-event exemption, repeated detection and per-episode reset.
    ct=np.ones((150,4));tt=np.arange(150)*.02
    for start in (5,30,80,105):ct[start:start+5,[0,3]]=0
    reset=np.zeros(150,bool);reset[70]=True
    _,ev=history_gate(tt,ct,reset)
    assert [e['repeated'] for e in ev]==[False,True,False,True]
    directory='logs/conga_history_1024_experiment/eval_1200'
    data=np.genfromtxt(ROOT/directory/'rollout.csv',delimiter=',',names=True)
    meta=json.loads((ROOT/directory/'summary.json').read_text());cases=[]
    for case in meta['cases']:
        rows=data[data['env']==case['case']]
        cs=np.stack([rows['contact_'+f] for f in FEET],-1);hs=np.stack([rows['clearance_'+f] for f in FEET],-1)
        result=evaluate(rows['episode_time_s'],cs,hs,rows['done'])
        valid=(rows['episode_time_s']>=2)&(rows['done']==0)
        actual=float(sum(rows[n][valid].mean() for n in rows.dtype.names if n.startswith('reward_rate_')))
        result['logged_full_rate']=actual
        result['estimated_full_rates']={str(cost):actual-result['pause_cost_rate']-cost*result['repeats_per_second'] for cost in costs}
        cases.append({'case':case['case'],'command':case['command'],**result})
    output={'definition':{'phase_weight':.25,'history_pair_weight':.1,'pause_penalty':.25,
            'repeat_cost_per_event':costs,'dt':.02,'no_added_support_bonus':True,
            'units':'cadence scores are mean reward per second; event costs are integrated per liftoff, NOT manager weights'},
            'synthetic':synthetic,'rollout':{'directory':directory,'cases':cases}}
    dest=ROOT/'logs/conga_repeat_penalty_scoring';dest.mkdir(exist_ok=True)
    (dest/'scores.json').write_text(json.dumps(output,indent=2)+'\n')
    for name,v in synthetic.items():print(name,'pause',round(v['pause_only'],4),'repeat Hz',v['repeats_per_second'],'combined', {k:round(x,4) for k,x in v['combined'].items()})
    for case in cases[:8]:print(case['command'],'repeat Hz',case['repeats_per_second'],'scores',case['combined'])


if __name__=='__main__':main()
