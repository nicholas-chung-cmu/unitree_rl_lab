"""Read-only Stage A reward/exploration audit; no simulator or reward mutation."""
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'logs/hierarchical_stage_a_20261029_153221'
TRAIN = ROOT/'logs/rsl_rl/go2_hierarchical/2026-10-29_15-32-24_stage_a_20261029_153221'
OUT = ROOT/'logs/hierarchical_stage_a_reward_audit'
OUT.mkdir(exist_ok=True)
events = EventAccumulator(str(TRAIN), size_guidance={'scalars':0}).Reload()
training = {}
for name in ['Policy/mean_noise_std', 'Metrics/skill/successes', 'Episode_Reward/success',
             'Episode_Reward/failure', 'Train/mean_reward', 'Episode_Termination/skill_failure']:
    rows = events.Scalars(name)
    training[name] = dict(first=rows[0].value, last=rows[-1].value,
                         maximum=max(r.value for r in rows),
                         last50_mean=float(np.mean([r.value for r in rows[-50:]])))
rows = [json.loads(line) for line in (RUN/'eval_199/trace.jsonl').open()]
step = [r for r in rows if r['kind']==0]
terminal = [r for r in step if r['event']]
height = np.array([r['foot_clearance'] for r in step])
selected = np.array([r['selected'] for r in step], dtype=bool)
contacts = np.array([r['contacts'] for r in step], dtype=bool)
observed = dict(step_samples=len(step), terminal_trials=len(terminal),
                trials_with_registered_lift=sum(any(r['lifted']) for r in terminal),
                selected_contact_fraction=float(contacts[selected].mean()),
                selected_clearance_p50_p95_max=np.quantile(height[selected], [.5,.95,1]).tolist())
# Synthetic one-second STEP at .08 m displacement; initial soles exactly z=0.
# Compare equal durations; no success/failure events or physical penalties.
# This intentionally does not claim any trajectory is dynamically feasible.
p=np.arange(1,51)*.02
blend=6*p**5-15*p**4+10*p**3
ref=np.stack((.08*blend, np.zeros_like(p), .04*np.sin(np.pi*p)**2),-1)
air=(p>.15)&(p<.7)
scores={}
for name in ['reference','planted_body_tracks','one_selected_foot','double_height']:
    feet=np.broadcast_to(ref[:,None,:],(50,2,3)).copy()
    if name=='planted_body_tracks': feet[:]=0
    if name=='one_selected_foot': feet[:,1]=0
    if name=='double_height': feet[:,:,2]*=2
    placement=np.exp(-((feet-ref[:,None,:])**2).sum(-1)/.0025).mean(-1)
    # Synthetic force proxy: soles >2 mm are airborne, including outside the
    # rewarded contact window. This exposes the contact/height reference conflict.
    actual_air=feet[:,:,2]>.002
    contact=(2+(actual_air==air[:,None]).sum(-1))/4
    base=1.5*placement+1+.5+.3+.5*contact
    # Candidate: signed improvement over planted height error, normalized by
    # 4 cm, bounded [-1,1], selected feet only, during scheduled swing.
    lift=((ref[:,None,2]-abs(feet[:,:,2]-ref[:,None,2]))/.04).clip(-1,1).mean(-1)*air
    scores[name]=dict(placement_mean=float(placement.mean()),contact_mean=float(contact.mean()),
                     dense_reward_integral=float(base.sum()*.02),
                     candidate_lift_integral=float(lift.sum()*.02),
                     with_candidate_weight_05=float((base+.5*lift).sum()*.02),
                     with_candidate_weight_1=float((base+lift).sum()*.02))
result=dict(training=training, observed=observed, synthetic=scores,
            contact_reference_conflict=dict(contact_air_window=[.15,.7],
                reference_clearance_at_contact_return_m=float(.04*np.sin(np.pi*.7)**2),
                reference_above_8mm_phase_window=[float(np.arcsin(np.sqrt(.008/.04))/np.pi), float(1-np.arcsin(np.sqrt(.008/.04))/np.pi)]),
            interpretation='Equal-duration synthetic task rewards only; excludes event bonuses, initial support acquisition, actuator costs and physical feasibility. Candidate is NOT installed.',
            candidate='Selected-foot linear height-error improvement over planted baseline during .15<p<.7, normalized by .04 m; compare +.5/+1. No contact gate, no history bonus.')
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
