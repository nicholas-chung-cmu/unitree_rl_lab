"""Reward permutation/phase audit and balanced-cycle counterfactual scoring."""
import json
import numpy as np
import torch
from score_liftoff_history import ROOT, load, FEET


def main():
    torch.set_num_threads(2)
    refs=load('conga_reference');pair=load('cadence_pair_scores');march=load('march_rewards');pause=load('pause_rewards')
    cfg=refs.CongaReferenceCfg(kick_enabled=False,kick_scale=0,bounce_scale=0,cadence_swing_start=.12,cadence_swing_end=.32)
    def reference(t):return refs.generate_reference(torch.tensor(t,dtype=torch.float64),cfg)
    rt=np.arange(200)*.02;r=reference(rt);ideal=(r.contact>=.5-1e-6).double()
    pn=pair.raw_scores(r.contact,r.foot_height,ideal,r.foot_height).mean(0)
    mn=march.raw_terms(r.contact,r.foot_height,ideal,r.foot_height).mean(0)[[0,2]]
    pause_norm=pause.pause_weight(torch.tensor(rt)).mean()
    def evaluate(time,contacts,heights,offset=0.):
        t=torch.tensor(time,dtype=torch.float64);c=torch.tensor(contacts,dtype=torch.float64);h=torch.tensor(heights,dtype=torch.float64)
        r=reference(time+offset)
        p=pair.normalized_scores(pair.raw_scores(r.contact,r.foot_height,c,h),pn)
        m=march.raw_terms(r.contact,r.foot_height,c,h)
        state=pause.StepHistory(1,'cpu');gates=[];repeats=[]
        for i in range(len(t)):
            g,e=state.update(c[i:i+1].bool(),h[i:i+1],torch.round(t[i:i+1]/.02).long())
            which=0 if r.foot_height[i,0]>=r.foot_height[i,1] else 1
            gates.append(g[0,which]);repeats.append(e[0])
        gate=torch.stack(gates);repeat=.025*torch.stack(repeats)/.02
        components=torch.stack((p[:,2],.25*p[:,1],.25*m[:,0]/mn[0],
            -.5*m[:,1]/mn[1],-.25*pause.pause_weight(t+offset)/pause_norm*(1-c).mean(-1),-repeat),-1)
        valid=t>=2
        # Only complete one-second cycles. No credit bridges an episode reset.
        breaks=np.r_[0,np.flatnonzero(np.diff(time)<=0)+1,len(time)]
        cycle_bonus=torch.zeros(len(t),dtype=torch.float64);cycles=[]
        for a,b in zip(breaks[:-1],breaks[1:]):
            ids=np.arange(a,b);cycle=np.floor(time[ids]+1e-5).astype(int)
            for j in np.unique(cycle):
                use=ids[cycle==j]
                if j<2 or len(use)!=50:continue
                # Slots refer to clock-order half cycles; min is pair symmetric.
                qa=float(p[use[:25],1].mean());qb=float(p[use[25:],1].mean())
                value=.25*min(qa,qb)
                cycle_bonus[use]=value
                cycles.append({'a':qa,'b':qb,'bonus':value})
        new=components.sum(-1)-components[:,1]+components[:,1]*gate+cycle_bonus
        return {'components':components[valid].mean(0).tolist(),
                'current':float(components.sum(-1)[valid].mean()),
                'restored_gate':float((components.sum(-1)-components[:,1]+components[:,1]*gate)[valid].mean()),
                'balanced_bonus':float(cycle_bonus[valid].mean()),'proposed':float(new[valid].mean()),
                'positive_cycles':sum(v['bonus']>1e-6 for v in cycles),'cycles':len(cycles)},components
    t=np.arange(2100)*.02;r=reference(t);c=(r.contact.numpy()>=.5-1e-6).astype(float);h=r.foot_height.numpy()
    cases={}
    for fraction in (0.,.25,.5,1.):
        cc=c.copy();hh=h.copy()
        # Keep the correctly scheduled A swing; replace selected B opportunities
        # with another A swing, reproducing the repeated-pair failure.
        for i in range(len(t)):
            beat=int(np.floor((t[i]+1e-5)/.5))
            cycle=beat//2
            keep=(cycle%4)<round(4*fraction)
            if beat%2 and not keep:
                cc[i]=cc[i,[1,0,3,2]];hh[i]=hh[i,[1,0,3,2]]
        values,raw=evaluate(t,cc,hh)
        mirrored,mirror_raw=evaluate(t,cc[:,[1,0,3,2]],hh[:,[1,0,3,2]],.5)
        torch.testing.assert_close(raw,mirror_raw,atol=2e-10,rtol=2e-10)
        cases[str(fraction)]={**values,'mirror_max_error':float((raw-mirror_raw).abs().max())}
    # Repeating the correct A on its own opportunities, remaining planted for B,
    # is another asymmetry that cannot be fixed by penalizing only extra steps.
    cc=c.copy();hh=h.copy();b=(np.floor((t+1e-5)/.5).astype(int)%2)==1
    cc[b]=1;hh[b]=0
    cases['A_only_on_own_beat']=evaluate(t,cc,hh)[0]
    cases['standing']=evaluate(t,np.ones_like(c),np.zeros_like(h))[0]
    d=np.genfromtxt(ROOT/'logs/conga_march_4096_experiment/eval_1000/rollout.csv',delimiter=',',names=True)
    actual=[]
    for env in range(16):
        rows=d[(d['env']==env)&(d['done']==0)]
        cc=np.stack([rows['contact_'+f] for f in FEET],-1);hh=np.stack([rows['clearance_'+f] for f in FEET],-1)
        # Terminal rows contain NaN physical values; exclude them. The next
        # episode's time drop still resets history, never bridging episodes.
        values,raw=evaluate(rows['episode_time_s'],cc,hh)
        _,mirror_raw=evaluate(rows['episode_time_s'],cc[:,[1,0,3,2]],hh[:,[1,0,3,2]],.5)
        torch.testing.assert_close(raw,mirror_raw,atol=2e-5,rtol=2e-5)
        actual.append({'env':env,**values,'mirror_max_error':float((raw-mirror_raw).abs().max())})
    out={'components':['phase','paired','individual','missed_swing','pause','repeat'],
         'candidate':'Restore pair history eligibility; add .25 * min(per-half-cycle normalized paired quality) once per full 1 s cycle. Scores shown as mean rate; implementation would pay a delayed cycle bonus.',
         'synthetic':cases,'actual':actual,
         'initial_phase_enumeration':{'possible_initial_episode_steps':1000,'even_beat_steps':500,'odd_beat_steps':500},
         'limitations':'Contact reward symmetry only; no simulation of synthetic motion or claim about torque/action costs. No new configuration or training.'}
    dest=ROOT/'logs/conga_march_phase_audit';dest.mkdir(exist_ok=True)
    (dest/'reward_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
