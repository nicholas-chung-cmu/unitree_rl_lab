"""Offline test of instantaneous versus rolling mean velocity-error rewards.

No simulator, task mutation or learning. Recorded cases have fixed commands.
Body-frame error averaging is a candidate score, not a displacement guarantee
when turning. Synthetic velocity traces are not physically validated gaits.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('logs/conga_velocity_average_scoring');OUT.mkdir(parents=True,exist_ok=True)
DT=.02
WINDOWS=(.5,1.,2.)

def rolling(x,n):
    cs=np.concatenate((np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)))
    return (cs[n:]-cs[:-n])/n

def scores(error):
    # Same samples in all variants: discard full two-second warmup.
    start=100
    instant=1.5*np.exp(-np.square(error).sum(-1)/.25)
    result={'instant':float(instant[start:].mean())}
    for w in WINDOWS:
        n=round(w/DT);avg=rolling(error,n)
        result[f'average_{w:g}s']=float((1.5*np.exp(-np.square(avg).sum(-1)/.25))[start-n+1:].mean())
        result[f'blend_{w:g}s']=.8*result[f'average_{w:g}s']+.2*result['instant']
    result['mean_error_xy']=error[start:].mean(0).tolist()
    result['rms_error_mps']=float(np.sqrt(np.square(error[start:]).sum(-1).mean()))
    result['fluctuation_rms_mps']=float(np.sqrt(np.square(error[start:]-error[start:].mean(0)).sum(-1).mean()))
    return result

sources={
    'gentle':'logs/conga_duration_4096_experiment/eval_1000',
    'strong':'logs/conga_duration_strong_4096_experiment/eval_1000',
    'basic':'logs/conga_basic_eval_final',
    'planted':'logs/conga_march_balanced_4096_experiment/eval_1000',
}
recorded={}
for label,directory in sources.items():
    directory=Path(directory);meta=json.loads((directory/'summary.json').read_text())
    data=np.genfromtxt(directory/'rollout.csv',delimiter=',',names=True)
    cases=[]
    for case in meta['cases']:
        rows=data[data['env']==case['case']]
        # Segment before removing terminal rows so no window bridges resets.
        boundaries=np.r_[0,np.flatnonzero(np.diff(rows['episode_time_s'])<=0)+1,len(rows)]
        segments=[];counts=[];audits=[]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            r=rows[a:b];r=r[r['done']==0]
            if len(r)<=100:continue
            assert np.all(np.diff(r['step'])==1)
            v=np.column_stack((r['vx'],r['vy']));cmd=np.array(case['command'][:2]);err=v-cmd
            result=scores(err)
            if 'reward_rate_track_lin_vel_xy' in r.dtype.names:
                expected=1.5*np.exp(-np.square(err).sum(-1)/.25)
                audits.append(float(np.max(abs(expected-r['reward_rate_track_lin_vel_xy']))))
            segments.append(result);counts.append(len(r)-100)
        if not segments:continue
        merged={k:np.average([x[k] for x in segments],axis=0,weights=counts).tolist() for k in segments[0]}
        merged.update(case=case['case'],command=case['command'],samples=sum(counts),
                      max_logged_reward_difference=max(audits) if audits else None)
        cases.append(merged)
    recorded[label]=cases

# Four-second repeats avoid biased phase occupancy in finite traces.
t=np.arange(1100)*DT
synthetic={}
traces={
    'constant_target':np.full_like(t,.4),
    'standing':np.zeros_like(t),
    'constant_underspeed':np.full_like(t,.2),
    '1Hz_small_ripple':.4+.1*np.sin(2*np.pi*t),
    '2Hz_small_ripple':.4+.1*np.sin(4*np.pi*t),
    '1Hz_large_ripple':.4+.3*np.sin(2*np.pi*t),
    '2Hz_large_ripple':.4+.3*np.sin(4*np.pi*t),
    'full_stop_go':np.where((np.arange(len(t))%25)<10,1.,0.),
    'back_and_forth':.4+.8*np.sin(2*np.pi*t),
}
for name,v in traces.items():
    synthetic[name]=scores(np.column_stack((v-.4,np.zeros_like(v))))
    synthetic[name]['velocity_range']=[float(v.min()),float(v.max())]
assert abs(synthetic['1Hz_small_ripple']['instant']-synthetic['2Hz_small_ripple']['instant'])<1e-9
assert abs(synthetic['constant_underspeed']['instant']-synthetic['constant_underspeed']['average_2s'])<1e-9
assert abs(synthetic['standing']['instant']-synthetic['standing']['average_2s'])<1e-9
for name in ('constant_target','1Hz_small_ripple','1Hz_large_ripple','full_stop_go','back_and_forth'):
    assert abs(synthetic[name]['average_2s']-1.5)<1e-8
for cases in recorded.values():
    for c in cases:
        if c['max_logged_reward_difference'] is not None:
            assert c['max_logged_reward_difference']<1e-4
out={'dt':DT,'xy_weight':1.5,'std':.5,'windows_s':WINDOWS,'recorded':recorded,'synthetic':synthetic,
     'limitations':['Fixed-command replay only; no learned desired slow gait available.',
                    'Reward preference does not establish causal PPO response.',
                    'Body-frame mean error can hide excursions and turning displacement error.',
                    'Synthetic full-stop motion is not a physical feasibility claim.']}
(OUT/'scores.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4))
x=np.arange(8)
for label in ['gentle','strong']:
    c=recorded[label][:8]
    axes[0].plot(x,[v['average_2s']-v['instant'] for v in c],marker='o',label=label)
axes[0].set(xticks=x,xticklabels=['stop','fwd','back','left','right','yaw+','yaw-','mixed'],ylabel='Reward gain / second',title='Actual rollout: 2 s averaging gain')
axes[0].legend();axes[0].grid(alpha=.2)
names=['constant_target','1Hz_small_ripple','1Hz_large_ripple','full_stop_go','back_and_forth','standing']
x=np.arange(len(names));axes[1].bar(x-.15,[synthetic[n]['instant'] for n in names],.3,label='instant')
axes[1].bar(x+.15,[synthetic[n]['average_2s'] for n in names],.3,label='2 s average')
axes[1].set(xticks=x,xticklabels=['steady','small ripple','large ripple','stop/go','reversal','standing'],ylabel='XY reward / second',title='Synthetic traces: equal average speed except standing')
axes[1].tick_params(axis='x',rotation=25);axes[1].legend()
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160)
print('RECORDED')
for label,cases in recorded.items():
    for c in cases[:8]:print(label,c['case'],c['command'],'instant',round(c['instant'],5),'mean2',round(c['average_2s'],5),'gain',round(c['average_2s']-c['instant'],5),'ripple',round(c['fluctuation_rms_mps'],4))
print('SYNTHETIC')
for name,s in synthetic.items():print(name,{k:round(s[k],5) for k in ['instant','average_0.5s','average_1s','average_2s','blend_2s']})
