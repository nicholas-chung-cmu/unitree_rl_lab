"""Offline scoring of installed lift/height terms; not a physics feasibility test."""
import importlib.util
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shaping',ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/skills/shaping.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=torch.arange(1,51)*.02
blend=6*p**5-15*p**4+10*p**3
ref=torch.stack((.08*blend,torch.zeros_like(p),.04*torch.sin(torch.pi*p).square()),-1)
result={}
for name in ['reference','grounded_slide','half_height','one_foot','double_height','crouched_slide']:
 feet=ref[:,None,:].expand(-1,2,-1).clone()
 if name in ['grounded_slide','crouched_slide']:feet[:,:,2]=0
 if name=='half_height':feet[:,:,2]*=.5
 if name=='one_foot':feet[:,1,2]=0
 if name=='double_height':feet[:,:,2]*=2
 h=feet[:,:,2];target=ref[:,None,2]
 old=1.5*torch.exp(-(feet-ref[:,None,:]).square().sum(-1)/.0025).mean(-1)
 placement=1.5*(torch.exp(-(feet-ref[:,None,:]).square().sum(-1)/.0025)*m.swing_placement_gate(h,target)).mean(-1)
 lift=m.clearance_progress(h,target).mean(-1)
 body=torch.full_like(p,.22 if name=='crouched_slide' else .32)
 cost=-.1*m.body_height_deviation(body,torch.full_like(p,.32))
 result[name]={key:float(val.sum()*.02) for key,val in dict(old_placement=old,new_placement=placement,clearance=lift,body_height=cost,new_total=placement+lift+cost).items()}
# Verify basic ordering and absence of an incentive for excess height.
assert result['reference']['new_total']>result['half_height']['new_total']>result['grounded_slide']['new_total']
assert result['reference']['new_total']>result['double_height']['new_total']
assert result['crouched_slide']['new_total']<result['grounded_slide']['new_total']
out=ROOT/'logs/hierarchical_clearance_height_scoring';out.mkdir(exist_ok=True)
(out/'scores.json').write_text(json.dumps(dict(scores=result,note='One-second synthetic traces; changed terms only. Other rewards, actuator costs and early termination excluded; not learned feasibility.'),indent=2)+'\n')
print(json.dumps(result,indent=2))
