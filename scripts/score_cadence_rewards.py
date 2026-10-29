"""Offline cadence reward calibration on recorded stock/basic trajectories.

No simulator, reward configuration changes, or optimizer steps. Reference-only
counterfactuals replace feet while retaining other measured costs; they are not
physically validated policies or predictions of learned returns.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rollout", type=Path, required=True)
parser.add_argument("--existing", type=Path, default=Path("logs/conga_basic_eval_final"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
path = Path(__file__).resolve().parents[1] / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/conga_reference.py"
spec = importlib.util.spec_from_file_location("cadence_reference", path)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
cfg = reference.CongaReferenceCfg(kick_enabled=False, kick_scale=0, bounce_scale=0)


def terms(t, contact, height, std=0.015):
    ref = reference.generate_reference(t, cfg)
    confidence = (2 * ref.contact - 1).square()
    contact_score = 1 - (confidence * (contact - ref.contact).square()).mean(-1)
    emphasis = (ref.foot_height / cfg.swing_height).square()
    lift = (height / ref.foot_height.clamp_min(0.015)).clamp(0, 1)
    quality = lift * (1 - contact) * torch.exp(-((height - ref.foot_height) / std).square())
    clearance = (emphasis * quality).sum(-1) / emphasis.sum(-1).clamp_min(1)
    return contact_score.numpy(), clearance.numpy()


def stats(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x)),
            "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95))}


data = np.genfromtxt(args.rollout / "rollout.csv", delimiter=",", names=True)
meta = json.loads((args.rollout / "summary.json").read_text())
old = np.genfromtxt(args.existing / "rollout.csv", delimiter=",", names=True)
valid = (data["done"] == 0) & (data["episode_time_s"] >= 2)
rows = data[valid]
t = torch.tensor(rows["episode_time_s"], dtype=torch.float64)
feet = ["FL", "FR", "RL", "RR"]
contacts = torch.tensor(np.stack([rows[f"contact_{f}"] for f in feet], -1))
heights = torch.tensor(np.stack([rows[f"clearance_{f}"] for f in feet], -1))
contact, clearance = terms(t, contacts, heights)
ref = reference.generate_reference(t, cfg)
perfect_contact = (ref.contact > .5).to(torch.float64)
ideal = terms(t, perfect_contact, ref.foot_height)
planted = terms(t, torch.ones_like(contacts), torch.zeros_like(heights))
flight = terms(t, torch.zeros_like(contacts), ref.foot_height)
wrong = terms(t, perfect_contact[:, [1, 0, 3, 2]], ref.foot_height[:, [1, 0, 3, 2]])
assert np.all(planted[1] == 0)
assert np.all(ideal[0] >= planted[0] - 1e-12)
assert np.all(ideal[0] >= flight[0] - 1e-12)
# The lift factor can prefer a slight overshoot near low reference heights;
# reference-perfect feet are therefore a benchmark, not an exact score maximum.
emphasis_sum = (ref.foot_height / cfg.swing_height).square().sum(-1)
clearance_bound = (emphasis_sum / emphasis_sum.clamp_min(1)).numpy()
assert np.all(clearance_bound >= clearance - 1e-12)

rate_names = [n for n in rows.dtype.names if n.startswith("reward_rate_")]
rates = {n.removeprefix("reward_rate_"): rows[n] for n in rate_names}
removed_names = ["joint_pos", "feet_air_time", "air_time_variance"]
removed = sum(rates[n] for n in removed_names)
stock = sum(rates.values())
retained = stock - removed
tracking = rates["track_lin_vel_xy"] + rates["track_ang_vel_z"]
grid = [(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 1.0), (1.0, 0.75), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0)]
report = {"checkpoint": meta["checkpoint"], "sha256": meta["sha256"],
          "units": "reward rate before dt; multiply by 0.02 for a control-step contribution",
          "candidate": {"contact": "1 - mean((2*target_contact-1)^2 * (actual_contact-target_contact)^2)",
                        "clearance": "existing swing_quality formula; std=0.015 m, actual unloading required",
                        "target_clearance_m": cfg.swing_height},
          "cases": [], "counterfactual_warning": __doc__}
if len(old) == len(data):
    report["existing_rollout_comparison"] = {
        "max_velocity_difference": float(np.nanmax(np.abs(data["vx"] - old["vx"]))),
        "contact_agreement": float(np.mean(np.stack([
            data[f"contact_{f}"][valid] == old[f"contact_{f}"][valid] for f in feet]))),
    }
for case in meta["cases"]:
    mask = rows["env"] == case["case"]
    result = {"case": case["case"], "command": case["command"], "n": int(mask.sum()),
              "stock_reward_rates": {n: stats(v[mask]) for n, v in rates.items()},
              "stock_total": stats(stock[mask]), "retained_total": stats(retained[mask]),
              "tracking_total": stats(tracking[mask]),
              "contact": stats(contact[mask]), "clearance": stats(clearance[mask]),
              "reference_only": {n: {"contact": float(a[mask].mean()), "clearance": float(b[mask].mean())}
                                 for n, (a, b) in [("ideal", ideal), ("planted", planted), ("flight", flight), ("wrong_diagonal", wrong)]},
              "clearance_width_sensitivity": {str(std): float(terms(t[mask], contacts[mask], heights[mask], std)[1].mean())
                                              for std in (.01, .015, .02)},
              "weight_grid": []}
    for wc, wh in grid:
        added = wc * contact[mask] + wh * clearance[mask]
        result["weight_grid"].append({"contact_weight": wc, "clearance_weight": wh,
            "added_rate": stats(added), "new_total_same_trajectory": stats(retained[mask] + added),
            "reference_ideal_minus_recorded": float((wc * (ideal[0][mask] - contact[mask]) + wh * (ideal[1][mask] - clearance[mask])).mean()),
            "reference_ideal_minus_planted": float((wc * (ideal[0][mask] - planted[0][mask]) + wh * (ideal[1][mask] - planted[1][mask])).mean())})
    # A single offset can improve score without establishing frequency locking.
    offsets = np.linspace(0, 1, 51)[:-1]
    phase_scores = [float(terms(t[mask] + offset, contacts[mask], heights[mask])[0].mean()) for offset in offsets]
    result["phase_offset_scan"] = {"best_offset_s": float(offsets[np.argmax(phase_scores)]),
                                   "min_contact": min(phase_scores), "max_contact": max(phase_scores)}
    report["cases"].append(result)

report["pooled"] = {"stock_rates": {n: stats(v) for n, v in rates.items()},
                    "contact": stats(contact), "clearance": stats(clearance),
                    "tracking": stats(tracking), "stock_total": stats(stock),
                    "retained_total": stats(retained)}
(args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for index, name in [(0, "stand"), (1, "forward"), (2, "backward"), (3, "left"), (4, "right"), (5, "yaw+"), (6, "yaw-"), (7, "moving turn")]:
    c = report["cases"][index]
    axes[0].bar(index - .18, c["contact"]["mean"], width=.35, color="tab:blue", label="Contact score" if index == 0 else None)
    axes[0].bar(index + .18, c["clearance"]["mean"], width=.35, color="tab:orange", label="Clearance score" if index == 0 else None)
axes[0].set(xticks=range(8), xticklabels=["stand", "forward", "backward", "left", "right", "yaw+", "yaw-", "turn"], ylabel="Unweighted mean score")
axes[0].legend()
mask = (rows["env"] == 1) & (rows["step"] < 300) & (rows["episode_time_s"] >= 2)
for i, foot in enumerate(feet):
    axes[1].step(t[mask], i + contacts[mask, i] * .65, where="post", label=foot)
    axes[1].plot(t[mask], i + ref.contact[mask, i].numpy() * .65, "k--", alpha=.5)
axes[1].set(xlabel="Episode time (s)", ylabel="Actual / target contact", yticks=range(4), yticklabels=feet,
            title="Forward walking: actual contacts (colored), cadence targets (dashed)")
fig.tight_layout(); fig.savefig(args.output / "scores.png", dpi=150); plt.close(fig)
print(json.dumps({"pooled": report["pooled"], "existing_rollout_comparison": report.get("existing_rollout_comparison")}, indent=2))
