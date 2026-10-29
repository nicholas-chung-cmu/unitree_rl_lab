"""Measure fixed-clock contact timing, clearance, tracking and falls from a rollout."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--paired", action="store_true", help="Score the revised 0.12–0.32 s swing schedule")
args = parser.parse_args()
path = Path(__file__).resolve().parents[1] / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/conga_reference.py"
spec = importlib.util.spec_from_file_location("cadence_ref", path)
refmod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refmod
spec.loader.exec_module(refmod)
cfg = refmod.CongaReferenceCfg(kick_enabled=False, kick_scale=0, bounce_scale=0, max_travel_speed=.5)
if args.paired:
    cfg.cadence_swing_start, cfg.cadence_swing_end = .12, .32
meta = json.loads((args.directory / "summary.json").read_text())
data = np.genfromtxt(args.directory / "rollout.csv", delimiter=",", names=True)
feet = ["FL", "FR", "RL", "RR"]
results = []
for case in meta["cases"]:
    rows = data[data["env"] == case["case"]]
    t = torch.tensor(rows["episode_time_s"], dtype=torch.float64)
    offset = torch.tensor(rows['phase_offset_s'],dtype=torch.float64) if 'phase_offset_s' in rows.dtype.names else meta.get('phase_offset_s',0.)
    ref = refmod.generate_reference(t+offset, cfg)
    contacts = np.stack([rows[f"contact_{f}"] for f in feet], -1)
    target = ref.contact.numpy()
    valid = (rows["done"] == 0) & (rows["episode_time_s"] >= 2)
    continuous = valid[1:] & valid[:-1] & (np.diff(rows["episode_time_s"]) > 0)
    force_cross = (contacts[1:] > .5) & (contacts[:-1] <= .5) & continuous[:, None]
    target_cross = (target[1:] >= .5) & (target[:-1] < .5) & continuous[:, None]
    abs_t = rows["step"][1:] * .02
    # Keep target events whose complete +/-80 ms window is inside scored data.
    target_cross &= ((rows["episode_time_s"][1:] >= 2.08) &
                     (rows["episode_time_s"][1:] <= 19.9))[:, None]
    missing, extra, matches, errors = 0, 0, 0, []
    for foot in range(4):
        actual, expected = abs_t[force_cross[:, foot]], abs_t[target_cross[:, foot]]
        pairs = sorted((abs(a-b), i, j) for i,a in enumerate(actual) for j,b in enumerate(expected) if abs(a-b) <= .080001)
        used_a, used_b = set(), set()
        for error, i, j in pairs:
            if i not in used_a and j not in used_b:
                used_a.add(i); used_b.add(j); errors.append(error)
        missing += len(expected) - len(used_b)
        extra += len(actual) - len(used_a)
        matches += len(used_b)
    contact_score = 1 - (((2 * target - 1)**2) * (contacts - target)**2).mean(-1)
    result = {**case, "contact_score": float(contact_score[valid].mean()) if valid.any() else None,
              "four_foot_fraction": float((contacts[valid].sum(-1) == 4).mean()) if valid.any() else None,
              "scheduled_pair_fraction": float((((contacts[:, 0] == 0) & (contacts[:, 3] == 0) & (contacts[:, 1] == 1) & (contacts[:, 2] == 1) & (target[:, 0] < .5)) | ((contacts[:, 1] == 0) & (contacts[:, 2] == 0) & (contacts[:, 0] == 1) & (contacts[:, 3] == 1) & (target[:, 1] < .5)))[valid].mean()) if valid.any() else None,
              "force_hz": (force_cross.sum(0) / max(.02, continuous.sum() * .02)).tolist(),
              "target_events": int(target_cross.sum()), "matched_within_80ms": matches,
              "missing_events": missing, "extra_events": extra,
              "matched_timing_error_s": float(np.mean(errors)) if errors else None}
    # Summarize complete contact intervals only, never across resets/warmup.
    masks={f'{f}_airborne':contacts[:,i]<.5 for i,f in enumerate(feet)}
    masks.update({f'{f}_grounded':contacts[:,i]>.5 for i,f in enumerate(feet)})
    masks.update({'FL_RR_airborne':(contacts[:,[0,3]]<.5).all(-1),
                  'FR_RL_airborne':(contacts[:,[1,2]]<.5).all(-1),
                  'four_foot_support':(contacts>.5).all(-1)})
    boundaries=np.r_[0,np.flatnonzero(~continuous)+1,len(rows)]
    durations={name:[] for name in masks}
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if b-a<3 or not valid[a:b].all():continue
        for name,mask in masks.items():
            edges=np.diff(np.r_[False,mask[a:b],False].astype(int))
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                if start>0 and end<b-a:
                    durations[name].append((end-start)*.02)
    result['contact_intervals']={name:{'count':len(values),
        'median_s':float(np.median(values)) if values else None,
        'p10_s':float(np.percentile(values,10)) if values else None,
        'p90_s':float(np.percentile(values,90)) if values else None}
        for name,values in durations.items()}
    result['duration_target_fraction']={name:float(np.mean((np.array(values)>=low)&(np.array(values)<=high))) if values else None
        for name,values in durations.items()
        for low,high in [((.70,.90) if name.endswith('_grounded') else (.16,.24))]
        if name in [f'{f}_{kind}' for f in feet for kind in ('grounded','airborne')]}
    result['reward_rates']={name.removeprefix('reward_rate_'):float(rows[name][valid].mean())
        for name in rows.dtype.names if name.startswith('reward_rate_') and valid.any()}
    if "clearance_FL" in rows.dtype.names and valid.any():
        height = np.stack([rows[f"clearance_{f}"] for f in feet], -1)
        target_height = ref.foot_height.numpy()
        emphasis = (target_height / .03)**2
        quality = np.clip(height / np.maximum(target_height, .015), 0, 1) * (1-contacts) * np.exp(-((height-target_height)/.015)**2)
        clearance_score = (emphasis * quality).sum(-1) / np.maximum(1, emphasis.sum(-1))
        result["clearance_score"] = float(clearance_score[valid].mean())
        result["swing_height_error_m"] = float((emphasis[valid] * abs(height[valid]-target_height[valid])).sum() / emphasis[valid].sum())
    results.append(result)
output = {"checkpoint": meta["checkpoint"], "timing_tolerance_s": .08,
          "swing_window": [cfg.cadence_swing_start, cfg.cadence_swing_end],
          "duration_task": meta.get("task", "").startswith("Unitree-Go2-Conga-Duration"),
          "phase_acceptance_required": not meta.get("task", "").startswith("Unitree-Go2-Conga-Duration"),
          "note": "For Duration, fixed-clock comparisons are descriptive, not requirements; neutral standing is allowed. Force crossings, not sensor first-contact events. Timing misses/excess are tolerance-based diagnostics, not rewards.",
          "falls": sum(c["falls"] for c in results), "timeouts": sum(c["timeouts"] for c in results),
          "cases": results}
(args.directory / "cadence_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
print('falls', output['falls'], 'timeouts', output['timeouts'])
for c in results[:8]:
    print(c['command'], 'v', [round(v, 3) for v in c.get('actual_velocity', [])],
          'contact', round(c['contact_score'], 3) if c['contact_score'] is not None else None,
          'clearance', round(c.get('clearance_score', 0), 3), 'hz', [round(v, 2) for v in c['force_hz']],
          'missing/expected', c['missing_events'], '/', c['target_events'], 'extra', c['extra_events'])
