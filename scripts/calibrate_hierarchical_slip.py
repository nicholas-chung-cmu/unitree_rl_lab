"""Equal-duration synthetic reward audit; not a physics/learnability proof."""
import importlib.util
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shaping',ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/skills/shaping.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=torch.arange(1,501,dtype=torch.float64)*.002
blend=6*p**5-15*p**4+10*p**3
out={}
for distance in [.04,.08,.12]:
 ref=torch.stack((distance*blend,torch.zeros_like(p),.04*torch.sin(torch.pi*p).square()),-1)
 velocity=distance*(30*p**4-60*p**3+30*p**2)
 cases={}
 for name in ['reference','grounded_slide','body_shift_planted','standing']:
  feet=ref[:,None,:].expand(-1,2,-1).clone()
  if name!='reference':feet[:,:,2]=0
  if name in ['body_shift_planted','standing']:feet[:,:,0]=0
  height=feet[:,:,2];target=ref[:,None,2]
  placement=1.5*(torch.exp(-(feet-ref[:,None,:]).square().sum(-1)/.0025)*m.swing_placement_gate(height,target)).mean(-1)
  clearance=m.clearance_progress(height,target).mean(-1)
  displacement=torch.ones_like(p) if name!='standing' else torch.exp(-(distance*blend).square()/.0064)
  air=(height>.002).double()
  contact=.5*(2+(1-(air-(target/.008).clamp(0,1)).abs()).sum(-1))/4
  # Two support feet stationary; selected feet track XY or stay planted.
  grounded_speed=((1-air)*velocity[:,None]).sum(-1) if name in ['reference','grounded_slide'] else torch.zeros_like(p)
  dense=placement+clearance+displacement+.5+.3+contact
  cases[name]={str(weight):float(((dense-weight*grounded_speed)*.002).sum()) for weight in [.1,2,4,8,12]}
 out[str(distance)]=cases
# Choose the smallest tested coefficient >=current that removes the slide-over-
# planted-body-shift advantage while keeping reference highest in every case.
chosen=None
for weight in [2,4,8,12]:
 k=str(weight)
 if all(c['grounded_slide'][k]<=min(c['body_shift_planted'][k],c['standing'][k]) and c['reference'][k]>max(c[n][k] for n in ['grounded_slide','body_shift_planted','standing']) for c in out.values()):
  chosen=weight;break
result=dict(selected_magnitude=chosen,scores=out,limitations='Synthetic 1s traces; 2mm height contact proxy, not actual force history; excludes physical actuator costs, event rewards and early termination. Confirms incentive ordering only.')
pth=ROOT/'logs/hierarchical_long_calibration';pth.mkdir(exist_ok=True)
(pth/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
