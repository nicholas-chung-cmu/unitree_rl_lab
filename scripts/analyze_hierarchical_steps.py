"""Analyze completed STEP trials, including repeated contact and real clearance."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('evaluation', type=Path)
args=parser.parse_args()
streams={}
trials=[]
for line in (args.evaluation/'trace.jsonl').open():
    r=json.loads(line)
    i=r['env']
    if r['kind']!=0:
        streams[i]=[]
        continue
    streams.setdefault(i,[]).append(r)
    if not r['event']:
        continue
    samples=streams.pop(i)
    selected=np.array(r['selected'],dtype=bool)
    heights=np.array([s['foot_clearance'] for s in samples])
    contacts=np.array([s['contacts'] for s in samples],dtype=bool)
    airborne=(heights>.008)&~contacts
    excursions=[]
    for foot in range(4):
        count=length=0
        for air in list(airborne[:,foot])+[False]:
            if air:length+=1
            else:
                count+=int(length>=2)
                length=0
        excursions.append(count)
    exact_contacts=all(r['touchdowns'][j]==1 for j in range(4) if selected[j])
    clean=(r['status']==1 and r['support_departures']==0 and exact_contacts)
    trials.append(dict(pair='FL_RR' if selected[0] else 'FR_RL', success=r['status']==1,
        clean_contact_trial=clean, support_departures=r['support_departures'],
        selected_contact_counts=[r['touchdowns'][j] for j in range(4) if selected[j]],
        selected_air_excursions=[excursions[j] for j in range(4) if selected[j]],
        support_air_excursions=sum(excursions[j] for j in range(4) if not selected[j]),
        max_support_clearance=float(heights[:,~selected].max()),
        duration=r['duration'], slip=r['slip'], foot_error=r['foot_error'],body_error=r['body_error']))
result={}
for pair in ['FL_RR','FR_RL']:
    rows=[r for r in trials if r['pair']==pair]
    result[pair]=dict(trials=len(rows),successes=sum(r['success'] for r in rows),
        clean_contact_trials=sum(r['clean_contact_trial'] for r in rows),
        trials_with_support_departure=sum(r['support_departures']>0 for r in rows),
        trials_with_support_air_excursion=sum(r['support_air_excursions']>0 for r in rows),
        trials_with_repeated_selected_air_excursion=sum(max(r['selected_air_excursions'])>1 for r in rows),
        trials_with_repeated_selected_contact=sum(max(r['selected_contact_counts'])>1 for r in rows),
        **{f'mean_{k}':float(np.mean([r[k] for r in rows])) if rows else None for k in ['duration','slip','foot_error','body_error']})
(args.evaluation/'contact_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
