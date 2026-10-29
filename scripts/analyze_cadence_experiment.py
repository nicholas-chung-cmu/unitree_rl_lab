"""Combine segmented cadence training logs and independently scored evaluations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", nargs="+", type=Path, required=True)
parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--latest-overlaps", action="store_true", help="For checkpoint retries, later run arguments replace repeated iteration samples")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
scalars = {}
for run in args.runs:
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    for tag in events.Tags()["scalars"]:
        if tag.endswith("/time"):
            continue  # wall-time axes restart per process; merge iteration axes only
        target = scalars.setdefault(tag, {})
        for point in events.Scalars(tag):
            if point.step in target and not args.latest_overlaps:
                raise ValueError(f"Overlapping stage iteration {point.step} in {tag}")
            target[point.step] = point.value
training = {}
for tag, points in scalars.items():
    values = np.array([v for _, v in sorted(points.items())])
    training[tag] = {"count": len(values), "first20": float(values[:20].mean()),
                     "last20": float(values[-20:].mean()), "last": float(values[-1])}
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
groups = [(["Train/mean_reward"], 1, "Total reward (changes with fade)"),
          (["Train/mean_episode_length"], .02, "Mean episode duration (s)"),
          (["Cadence/stock_reward_fraction", "Early/weight_cadence_phase", "Early/weight_cadence_pair"], 1, "Reward schedule"),
          (["Episode_Reward/duration_air", "Episode_Reward/duration_stance", "Episode_Reward/conga_contact", "Episode_Reward/conga_clearance", "Episode_Reward/cadence_support", "Episode_Reward/cadence_pair", "Episode_Reward/cadence_phase", "Episode_Reward/pause_missing", "Episode_Reward/repeated_departures", "Episode_Reward/march_clearance", "Episode_Reward/march_missed_swing"], 1, "Cadence reward contributions"),
          (["Episode_Termination/time_out", "Episode_Termination/bad_orientation", "Episode_Termination/base_contact"], 1, "Logged termination fractions"),
          (["Metrics/base_velocity/velocity_xy_error_mps", "Metrics/base_velocity/velocity_yaw_error_radps"], 1, "Training command errors (different units)")]
for ax, (tags, scale, title) in zip(axes.flat, groups):
    for tag in tags:
        if tag not in scalars: continue
        points = sorted(scalars[tag].items())
        ax.plot([s+1 for s,v in points], [v*scale for s,v in points], label=tag.split('/')[-1])
    for gate in (50, 100, 200): ax.axvline(gate, color='gray', alpha=.25)
    ax.set(title=title, xlabel="Completed cadence iterations")
    if ax.lines: ax.legend(fontsize=7)
    ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(args.output / "training_curves.png", dpi=150); plt.close(fig)
evaluations = []
for directory in args.evaluations:
    result = json.loads((directory / "cadence_metrics.json").read_text())
    result["directory"] = str(directory)
    cases = result["cases"]
    result["aggregates"] = {}
    for name, group in [("all", cases), ("conga_range", cases[:8]), ("initial_range", cases[8:])]:
        valid = [c for c in group if c["steady_samples"] > 0]
        count = sum(c['steady_samples'] for c in valid)
        result["aggregates"][name] = {k: sum(c[k]*c['steady_samples'] for c in valid)/count if count else None
            for k in ['xy_error_mps', 'yaw_error_radps', 'slip_mps', 'contact_score', 'clearance_score']}
        result["aggregates"][name].update({"missing": sum(c['missing_events'] for c in group),
            "expected": sum(c['target_events'] for c in group), "extra": sum(c['extra_events'] for c in group)})
    evaluations.append(result)
summary = {"runs": [str(r) for r in args.runs], "training": training, "evaluations": evaluations}
(args.output / "summary.json").write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({"training": {k:v for k,v in training.items() if k.startswith(('Train/', 'Cadence/'))},
                  "evaluations": [{"directory": r['directory'], "falls": r['falls'], "aggregates": r['aggregates']} for r in evaluations]}, indent=2))
