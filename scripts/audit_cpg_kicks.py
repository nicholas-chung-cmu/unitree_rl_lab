"""Audit kick endpoints and continuous holds from recorded physical measurements.

Strict endpoint: <=2cm planar error from nominal front-foot XY plus8cm outward,
5cm sole clearance +/-1.5cm, >=40ms selected air and support contact, foot speed
<.1m/s, tilt<.2rad and body angular speed<.4rad/s. Report timing independently:
reference extension is.25s; strict arrival permits60ms measured tolerance.
A steady hold needs these same conditions for >=80% of the remaining scheduled
hold interval, and no confirmed selected-foot touchdown. Musical success also
requires the next STEP to complete directly from the raised leg.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
p.add_argument('--swing',type=float,default=.2);p.add_argument('--timing',type=float,default=.06)
p.add_argument('--group',type=int,help='Audit only one evaluation-command group')
a=p.parse_args();x=dict(np.load(a.directory/'traces.npz'));beat=2/3;dt=.02
out={leg:{'expected':0,'relaxed_endpoint':0,'timely_endpoint':0,'steady_hold':0,'direct_postkick_step':0,'qualified_chain':0,'delays':[],'hold_valid_fractions':[]} for leg in ('FL','FR')}
q=x['joints'].reshape(*x['joints'].shape[:2],4,3)
if 'feet_body' in x:feet=x['feet_body']
else:
 h,t,k=q[...,0],q[...,1],q[...,2];z=-.213*(np.cos(t)+np.cos(t+k))
 feet=np.stack((-.213*(np.sin(t)+np.sin(t+k))+np.array([.1934,.1934,-.1934,-.1934]),
                np.cos(h)*np.array([.0955,-.0955,.0955,-.0955])-np.sin(h)*z+np.array([.0465,-.0465,.0465,-.0465]),
                np.sin(h)*np.array([.0955,-.0955,.0955,-.0955])+np.cos(h)*z),-1)
contact=x['contact'];confirmed=contact&np.concatenate((np.zeros_like(contact[:1]),contact[:-1]),0)
air=~contact&np.concatenate((np.zeros_like(contact[:1]),~contact[:-1]),0)
for env in range(x['time'].shape[1]):
 if a.group is not None and int(x['groups'][env])!=a.group:continue
 time=x['time'][:,env];bounds=np.r_[0,np.flatnonzero(np.diff(time)<-.1)+1,len(time)]
 for begin,end in zip(bounds[:-1],bounds[1:]):
  clock=time[begin:end]
  for phrase in range(max(0,int(clock.min()/(4*beat))),int(clock.max()/(4*beat))+1):
   start=phrase*4*beat+2.5*beat;target=start+.25
   next_land=(phrase+1)*4*beat;hold_end=next_land-a.swing
   if start<clock.min() or clock.max()<next_land+a.timing:continue
   leg=(phrase+1)%2;label=('FL','FR')[leg];r=out[label];r['expected']+=1
   idx=np.arange(begin,end);window=(clock>=target)&(clock<=min(target+.2,hold_end))
   target_xy=np.array([.1934,.222 if leg==0 else -.222])
   planar=np.linalg.norm(feet[begin:end,env,leg,:2]-target_xy,axis=-1)
   height=np.abs(x['sole'][begin:end,env,leg]-.05)
   speed=np.linalg.norm(x['foot_velocity'][begin:end,env,leg],axis=-1) if 'foot_velocity' in x else x['kick_speed'][begin:end,env]
   support=confirmed[begin:end,env][:,[j for j in range(4) if j!=leg]].all(-1)
   tilt=np.arccos(np.clip(-x['gravity'][begin:end,env,2],-1,1))
   angular=np.linalg.norm(x['angular_velocity'][begin:end,env],axis=-1)
   valid=(planar<.02)&(height<.015)&(speed<.1)&support&air[begin:end,env,leg]&(tilt<.2)&(angular<.4)
   if 'reset' in x:valid &= ~x['reset'][begin:end,env].astype(bool)
   endpoints=np.flatnonzero(window&valid)
   if not len(endpoints):continue
   first=endpoints[0];delay=float(clock[first]-target);r['delays'].append(delay);r['relaxed_endpoint']+=1
   timely=delay<=a.timing+1e-5;r['timely_endpoint']+=int(timely)
   hold=(clock>=clock[first])&(clock<hold_end)
   fraction=float(valid[hold].mean()) if hold.any() else 0.;r['hold_valid_fractions'].append(fraction)
   steady=fraction>=.8 and not confirmed[begin:end,env,leg][hold].any()
   r['steady_hold']+=int(steady)
   landing=(clock>=next_land-.02)&(clock<=next_land+a.timing+.04)
   direct=bool((x['postkick_success'][begin:end,env]&landing).any()) if 'postkick_success' in x else False
   r['direct_postkick_step']+=int(direct);r['qualified_chain']+=int(timely and steady and direct)
for r in out.values():
 r['median_endpoint_delay_s']=float(np.median(r.pop('delays'))) if r['delays'] else None
 fractions=r.pop('hold_valid_fractions');r['median_hold_valid_fraction']=float(np.median(fractions)) if fractions else None
 r['qualified_fraction']=r['qualified_chain']/max(1,r['expected'])
result={'method':__doc__,'group':a.group,'legs':out}
filename='kick_audit.json' if a.group is None else f'kick_audit_group{a.group}.json'
(a.directory/filename).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
