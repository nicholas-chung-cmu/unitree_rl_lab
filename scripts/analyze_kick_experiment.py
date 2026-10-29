"""Summarize the two bounded small-kick learning blocks and evaluation gates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path('logs/conga_kick_intro_4096_experiment');state=json.loads((root/'status.json').read_text())
out=root/'analysis';out.mkdir(exist_ok=True)
curves={}
for gate in state['gates']:
    for run in gate.get('runs',[gate['run']]):
        e=EventAccumulator(run,size_guidance={'scalars':0}).Reload()
        for tag in e.Tags()['scalars']:
            for p in e.Scalars(tag):curves.setdefault(tag,{})[p.step]=p.value
metrics=[]
for g in state['gates']:
    p=Path(g['evaluation'])/'cadence_metrics.json';d=json.loads(p.read_text());metrics.append({'iteration':g['iteration'],**d})
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,tags,scale,title in [
    (axes[0,0],['Train/mean_episode_length'],.02,'Training episode duration (s)'),
    (axes[0,1],['Episode_Reward/kick_height','Episode_Reward/kick_lateral','Episode_Reward/kick_airborne','Episode_Reward/kick_support'],1,'Logged kick reward contributions'),
    (axes[1,0],['Train/mean_reward'],1,'Training return (reward changes at +500)')]:
    for tag in tags:
        pts=sorted(curves.get(tag,{}).items())
        if pts:ax.plot([i-999 for i,v in pts],[v*scale for i,v in pts],label=tag.split('/')[-1])
    ax.axvline(500,color='gray',ls='--');ax.set(title=title,xlabel='Additional kick PPO iterations');ax.grid(alpha=.2);ax.legend(fontsize=8)
for leg in ['FL','FR']:
    axes[1,1].plot([m['iteration']-1000 for m in metrics],
                  [m['legs'][leg]['qualified']/max(1,m['legs'][leg]['windows']) for m in metrics],marker='o',label=leg)
axes[1,1].set(title='Qualified 2 cm diagnostic windows',xlabel='Additional kick PPO iterations',ylabel='Fraction',ylim=(-.05,1.05));axes[1,1].legend();axes[1,1].grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'kick_training_curves.png',dpi=160)
summary={'status':state['status'],'final_checkpoint':state['checkpoint'],'evaluations':metrics,
         'training':{k:{'last50':sum(v for _,v in sorted(p.items())[-50:])/len(sorted(p.items())[-50:])} for k,p in curves.items() if p},
         'note':'First500 uses lift .5; continuation uses lift2 with measured-clearance gating. Qualified windows are half-amplitude progress diagnostics, not full final-kick acceptance.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(out)
