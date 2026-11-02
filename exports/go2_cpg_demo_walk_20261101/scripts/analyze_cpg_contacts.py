"""Independent expected-opportunity audit of saved CPG physics traces.

Counts all scheduled swings, including those suppressed by the safety scheduler.
A passing pair needs one verified >8mm/40ms swing per selected foot, confirmed
landing in ±tolerance, and no >=40ms opposite-support contact loss. Raw contacts
use the simulator adapter's 5N/2N hysteresis. Sampling resolution is20ms.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--swing',type=float,default=.30)
p.add_argument('--tolerance',type=float,default=.10)
p.add_argument('--dance',action='store_true')
a=p.parse_args();x=dict(np.load(a.directory/'traces.npz'));dt=.02;beat=2/3

def runs(mask):
    z=np.r_[False,mask,False].astype(int);return list(zip(np.flatnonzero(np.diff(z)==1),np.flatnonzero(np.diff(z)==-1)))

cases={}
for env in range(x['time'].shape[1]):
    group=str(int(x['groups'][env]));out=cases.setdefault(group,{'expected':0,'correct':0,'paired_lift':0,'support_loss':0,'repeat_swing':0,'FL_RR_expected':0,'FR_RL_expected':0,'FL_RR_correct':0,'FR_RL_correct':0,'touch_errors':[],'lift_errors':[],'air_durations':[]})
    time=x['time'][:,env]
    boundaries=np.r_[0,np.flatnonzero(np.diff(time)<-.1)+1,len(time)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        clock=time[begin:end];ready=x['ready'][begin:end,env]
        if not ready.any():continue
        if a.dance:
            opportunities=[(phrase*3+slot,phrase*4*beat+slot*beat) for phrase in range(max(0,int(clock.min()/(4*beat))),int(clock.max()/(4*beat))+1) for slot in range(3)]
        else:opportunities=[(k,(k+1)*beat) for k in range(int(clock.max()/beat)+1)]
        for k,target in opportunities:
            start=target-a.swing
            if start<clock.min()-1e-5:continue
            if clock.max()<target+a.tolerance:continue
            indices=np.flatnonzero((clock>=start-1e-5)&(clock<=target+a.tolerance+1e-5)&ready)
            if not len(indices):continue
            idx=indices+begin;t=time[idx];contact=x['contact'][idx,env];sole=x['sole'][idx,env]
            pair=(k%2,3-k%2);support=tuple(j for j in range(4) if j not in pair)
            label='FL_RR' if k%2==0 else 'FR_RL'
            out['expected']+=1;out[label+'_expected']+=1
            bad_support=any(any((stop-lo)*dt>=.04-1e-6 for lo,stop in runs(~contact[:,j])) for j in support)
            out['support_loss']+=bad_support
            lifted=[];landings=[];repeated=False
            for j in pair:
                verified=[]
                for lo,stop in runs(~contact[:,j]):
                    if (stop-lo)*dt>=.04-1e-6 and (sole[lo+1:stop,j]>.008).any():verified.append((lo,stop))
                lifted.append(bool(verified));repeated|=len(verified)>1
                if verified:
                    lo,stop=verified[0];out['lift_errors'].append(float(t[lo]-start));out['air_durations'].append((stop-lo)*dt)
                    # Confirmation may occur one sample after the allowed onset
                    # window. Judge timing by onset, not confirmation latency.
                    global_stop=int(idx[0])+stop
                    landed=global_stop+1<end and x['contact'][global_stop:global_stop+2,env,j].all()
                    error=float(time[global_stop]-target) if landed else None
                    if error is not None:out['touch_errors'].append(error)
                    landings.append(landed and abs(error)<=a.tolerance+1e-5)
                else:landings.append(False)
            out['paired_lift']+=all(lifted);out['repeat_swing']+=repeated
            correct=all(lifted) and all(landings) and not bad_support and not repeated
            out['correct']+=correct;out[label+'_correct']+=correct
for out in cases.values():
    out['correct_fraction']=out['correct']/max(1,out['expected'])
    out['paired_lift_fraction']=out['paired_lift']/max(1,out['expected'])
    for field in ('touch_errors','lift_errors','air_durations'):
        values=out.pop(field)
        out[field+'_count']=len(values)
        out[field+'_median_abs']=float(np.median(np.abs(values))) if values else None
result={'method':__doc__,'horizon_seconds':x['time'].shape[0]*dt,'cases':cases}
(a.directory/'contact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
