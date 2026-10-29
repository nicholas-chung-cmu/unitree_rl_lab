"""Offline comparison of contact weights and diagonal-completion rewards.

Only reads recorded rollouts. Synthetic foot patterns do not establish physical
feasibility or predict whole-body returns. Nothing is registered as a task reward.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import math

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rollouts", nargs="+", type=Path, default=[Path("logs/conga_cadence_reward_audit"), Path("logs/conga_cadence_eval_final")])
parser.add_argument("--output", type=Path, default=Path("logs/conga_cadence_candidate_comparison"))
parser.add_argument("--swing-duration", type=float, default=.36, choices=[.20, .36])
parser.add_argument("--swing-center", type=float, default=.22)
args = parser.parse_args()
if not args.swing_duration/2 <= args.swing_center <= .5-args.swing_duration/2:
    parser.error("Swing window must fit inside a half-second beat")
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
path = Path(__file__).resolve().parents[1] / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/conga_reference.py"
spec = importlib.util.spec_from_file_location("candidate_ref", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
cfg = module.CongaReferenceCfg(kick_enabled=False, kick_scale=0, bounce_scale=0)


def reference(t):
    # Offline reference variant only: the actual task/reference files are untouched.
    ref = module.generate_reference(t, cfg)
    contact = torch.ones((len(t), 4), dtype=t.dtype)
    height = torch.zeros_like(contact)
    for beat in range(8):
        start = beat*.5 + args.swing_center - args.swing_duration/2
        end = start + args.swing_duration
        occupancy, pulse = module.swing_window(t.remainder(4), start, end, cfg.contact_ramp)
        pair = torch.tensor([1.,0.,0.,1.] if beat%2 == 0 else [0.,1.,1.,0.], dtype=t.dtype)
        contact -= occupancy[:,None]*pair
        height += .03*pulse[:,None]*pair
    ref.contact = contact
    ref.foot_height = height
    return ref


def scores(t, contact, height):
    ref = reference(t)
    match = 1 - (2 * ref.contact - 1).square() * (contact-ref.contact).square()
    emphasis = (ref.foot_height / cfg.swing_height).square()
    lift = (height / ref.foot_height.clamp_min(.015)).clamp(0, 1)
    q = lift * (1-contact) * torch.exp(-((height-ref.foot_height)/.015).square())
    old_clearance = (emphasis*q).sum(-1) / emphasis.sum(-1).clamp_min(1)
    a = ref.foot_height[:, 0] >= ref.foot_height[:, 1]
    swing_min = torch.where(a, torch.minimum(q[:, 0], q[:, 3]), torch.minimum(q[:, 1], q[:, 2]))
    stance = torch.where(a, contact[:, 1]*contact[:, 2], contact[:, 0]*contact[:, 3])
    activity = emphasis.sum(-1) / emphasis.sum(-1).clamp_min(1)
    paired = activity * swing_min * stance
    return {"mean_contact": match.mean(-1).numpy(), "pattern_contact": match.prod(-1).numpy(),
            "independent_clearance": old_clearance.numpy(), "paired_clearance": paired.numpy(),
            "blended_clearance": (.5*old_clearance + .5*paired).numpy()}


def summarize(s):
    return {k: {"mean": float(v.mean()), "positive_fraction": float((v>1e-6).mean()),
                "p95": float(np.quantile(v, .95))} for k,v in s.items()}


def combinations(s):
    out = {}
    for weight in (.25, .5, 1., 2.):
        out[f"mean_contact_{weight:g}+independent_0.5"] = weight*s['mean_contact'] + .5*s['independent_clearance']
        out[f"mean_contact_{weight:g}+paired_0.5"] = weight*s['mean_contact'] + .5*s['paired_clearance']
        out[f"pattern_contact_{weight:g}+paired_0.5"] = weight*s['pattern_contact'] + .5*s['paired_clearance']
    out['mean_contact_0.25+blended_0.5'] = .25*s['mean_contact'] + .5*s['blended_clearance']
    return out


# Equal sampling over complete four-second cycles, avoiding episode boundaries.
t = torch.arange(0, 4, .02, dtype=torch.float64)
ref = reference(t)
if args.swing_duration == .36 and args.swing_center == .22:
    original = module.generate_reference(t, cfg)
    torch.testing.assert_close(ref.contact, original.contact)
    torch.testing.assert_close(ref.foot_height, original.foot_height)
ideal_c = (ref.contact >= .5).double()
patterns = {"planted": (torch.ones_like(ideal_c), torch.zeros_like(ref.foot_height)),
            "reference": (ideal_c, ref.foot_height),
            "wrong_diagonal": (ideal_c[:, [1,0,3,2]], ref.foot_height[:, [1,0,3,2]]),
            "all_unloaded": (torch.zeros_like(ideal_c), ref.foot_height)}
for foot in range(4):
    c = torch.ones_like(ideal_c); h = torch.zeros_like(ref.foot_height)
    c[:, foot] = ideal_c[:, foot]; h[:, foot] = ref.foot_height[:, foot]
    patterns[f"only_{['FL','FR','RL','RR'][foot]}"] = (c, h)
# One of the prescribed feet per beat, alternating diagonals but never completing a pair.
single_c = torch.ones_like(ideal_c); single_h = torch.zeros_like(ref.foot_height)
single_c[:, :2] = ideal_c[:, :2]; single_h[:, :2] = ref.foot_height[:, :2]
patterns['one_foot_each_beat'] = (single_c, single_h)
for offset in (0, .25, .5):
    fast = reference(t*2 + offset)
    patterns[f"double_frequency_offset_{offset:g}"] = ((fast.contact >= .5).double(), fast.foot_height)
synthetic = {}
raw_synthetic = {}
for name,(c,h) in patterns.items():
    raw = scores(t,c,h); raw_synthetic[name] = raw
    synthetic[name] = {"scores": summarize(raw), "combinations": summarize(combinations(raw))}
assert synthetic['reference']['scores']['paired_clearance']['mean'] > .1
for name in ['planted','wrong_diagonal','all_unloaded','only_FL','only_FR','only_RL','only_RR','one_foot_each_beat']:
    assert np.max(raw_synthetic[name]['paired_clearance']) < 1e-12, name
for raw in raw_synthetic.values():
    assert np.all(raw['paired_clearance'] <= raw['independent_clearance'] + 1e-12)
    assert np.all((raw['pattern_contact'] >= -1e-12) & (raw['pattern_contact'] <= 1+1e-12))

report = {"schedule": {"swing_duration_s": args.swing_duration, "swing_center_s": args.swing_center,
                       "four_support_window_s": .5-args.swing_duration, "soft_ramp_s": cfg.contact_ramp,
                       "reference_peak_vertical_speed_mps": .03*math.pi/args.swing_duration,
                       "reference_peak_vertical_acceleration_mps2": .06*math.pi**2/args.swing_duration**2},
          "synthetic": synthetic, "rollouts": [],
          "definitions": {"mean_contact": "mean of per-foot soft contact-match scores (current)",
                          "pattern_contact": "product of four per-foot soft contact-match scores",
                          "paired_clearance": "reference swing activity * minimum quality of scheduled diagonal feet * both opposite support contacts",
                          "blended_clearance": "half independent clearance plus half paired clearance"},
          "caveat": __doc__}
retired = ['joint_pos','feet_air_time','air_time_variance','conga_contact','conga_clearance']
for directory in args.rollouts:
    meta = json.loads((directory/'summary.json').read_text())
    data = np.genfromtxt(directory/'rollout.csv', delimiter=',', names=True)
    rows = data[(data['done']==0) & (data['episode_time_s']>=2)]
    clock = torch.tensor(rows['episode_time_s'], dtype=torch.float64)
    c = torch.tensor(np.stack([rows[f'contact_{f}'] for f in ['FL','FR','RL','RR']],-1))
    h = torch.tensor(np.stack([rows[f'clearance_{f}'] for f in ['FL','FR','RL','RR']],-1))
    raw = scores(clock,c,h); variants = combinations(raw)
    base = sum(rows[n] for n in rows.dtype.names if n.startswith('reward_rate_') and n.removeprefix('reward_rate_') not in retired)
    result = {"directory": str(directory), "checkpoint": meta['checkpoint'], "sha256": meta['sha256'], "cases": []}
    for case in meta['cases']:
        mask = rows['env']==case['case']
        measured = {k:v[mask] for k,v in raw.items()}
        candidate = {k:v[mask] for k,v in variants.items()}
        result['cases'].append({"case": case['case'], "command": case['command'], "samples": int(mask.sum()),
            "scores": summarize(measured), "combinations": summarize(candidate),
            "retained_actual_reward_rate": float(base[mask].mean()),
            "actual_total_by_candidate": {k: float((base[mask]+v).mean()) for k,v in candidate.items()}})
    result['main_eight'] = {"scores": summarize({k:v[rows['env']<8] for k,v in raw.items()}),
                            "combinations": summarize({k:v[rows['env']<8] for k,v in variants.items()})}
    report['rollouts'].append(result)

(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes = plt.subplots(1,2,figsize=(13,5))
weights = [.25,.5,1.,2.]
for name in ['planted','only_RL','one_foot_each_beat','reference','double_frequency_offset_0']:
    axes[0].plot(weights,[synthetic[name]['combinations'][f'mean_contact_{w:g}+independent_0.5']['mean'] for w in weights],marker='o',label=name)
final = report['rollouts'][-1]['cases'][:8]
x = np.arange(8)
axes[1].bar(x-.17,[c['scores']['independent_clearance']['mean'] for c in final],.34,label='Current independent clearance')
axes[1].bar(x+.17,[c['scores']['paired_clearance']['mean'] for c in final],.34,label='Strict paired clearance')
axes[0].set(xlabel='Contact weight (clearance fixed at 0.5)',ylabel='Mean synthetic phase-reward rate',title='Stronger contact: reference-only benchmarks')
axes[1].set(xticks=x,xticklabels=['zero','forward','backward','left','right','yaw+','yaw-','turn'],ylabel='Unweighted score',title='Recorded final cadence policy')
for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
print(json.dumps({'synthetic_scores':{k:v['scores'] for k,v in synthetic.items()},
                  'measured':[{'directory':r['directory'],'main_eight':r['main_eight']} for r in report['rollouts']]},indent=2))
