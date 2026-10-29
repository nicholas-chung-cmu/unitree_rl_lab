"""Plot stock training and compare fixed-command checkpoint evaluations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
events = EventAccumulator(str(args.run), size_guidance={"scalars": 0}).Reload()
training = {}
for tag in events.Tags()["scalars"]:
    points = events.Scalars(tag)
    values = np.array([p.value for p in points])
    training[tag] = {"first20": float(values[:20].mean()), "last20": float(values[-20:].mean()),
                     "last": float(values[-1]), "count": len(values)}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, tags, scale, title in zip(
    axes.flat,
    [["Train/mean_reward"], ["Train/mean_episode_length"],
     ["Episode_Termination/time_out", "Episode_Termination/bad_orientation", "Episode_Termination/base_contact"],
     ["Curriculum/lin_vel_cmd_levels"]],
    [1, 0.02, 1, 1],
    ["Episode return", "Mean episode duration (s)", "Logged termination fractions", "XY command curriculum limit (m/s)"]
):
    for tag in tags:
        if tag not in events.Tags()["scalars"]:
            continue
        values = events.Scalars(tag)
        ax.plot([p.step for p in values], [p.value * scale for p in values], label=tag.split("/")[-1])
    ax.set(title=title, xlabel="PPO iteration")
    if ax.lines:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not logged / fixed command range", ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.output / "training_curves.png", dpi=150)
plt.close(fig)
comparison = []
fig, axes = plt.subplots(len(args.evaluations), 2, figsize=(13, 3.5 * len(args.evaluations)), squeeze=False)
for index, directory in enumerate(args.evaluations):
    result = json.loads((directory / "summary.json").read_text())
    groups = {}
    for name in ("conga_range", "stock_initial"):
        cases = [r for r in result["cases"] if r["group"] == name]
        lengths = [x for r in cases for x in r["completed_lengths_s"]]
        count = sum(r["steady_samples"] for r in cases)
        groups[name] = {
            "falls": sum(r["falls"] for r in cases), "timeouts": sum(r["timeouts"] for r in cases),
            "mean_completed_episode_s": float(np.mean(lengths)) if lengths else None,
            "steady_samples": count,
            "xy_error_mps": sum(r.get("xy_error_mps", 0) * r["steady_samples"] for r in cases) / count if count else None,
            "yaw_error_radps": sum(r.get("yaw_error_radps", 0) * r["steady_samples"] for r in cases) / count if count else None,
        }
    comparison.append({"evaluation": str(directory), "checkpoint": result["checkpoint"], "groups": groups})
    data = np.genfromtxt(directory / "rollout.csv", delimiter=",", names=True)
    crossings = []
    for case in result["cases"]:
        rows = data[data["env"] == case["case"]]
        continuous = ((rows["episode_time_s"][1:] > rows["episode_time_s"][:-1])
                      & (rows["episode_time_s"][:-1] >= 2) & (rows["done"][1:] == 0)
                      & (rows["done"][:-1] == 0))
        duration = float(continuous.sum() * 0.02)
        counts = [int(((rows[f"contact_{f}"][1:] == 1) & (rows[f"contact_{f}"][:-1] == 0)
                       & continuous).sum()) for f in ("FL", "FR", "RL", "RR")]
        crossings.append({"case": case["case"], "command": case["command"],
                          "force_crossing_counts": counts, "scored_duration_s": duration,
                          "force_crossings_per_foot_hz": [n / duration for n in counts] if duration else None,
                          "sensor_events_per_foot_hz": case.get("touchdowns_per_foot_hz")})
    comparison[-1]["contact_crossings"] = crossings
    for column, case in enumerate((1, 9)):
        ax = axes[index, column]
        # First episode only: reset rows are never connected into a fake gait.
        rows = data[data["env"] == case]
        ends = np.flatnonzero(rows["done"])
        rows = rows[:ends[0]] if len(ends) else rows
        rows = rows[(rows["episode_time_s"] >= 2) & (rows["episode_time_s"] <= 6)]
        if not len(rows):
            ax.text(0.5, 0.5, "No first episode survives warmup", ha="center", transform=ax.transAxes)
        else:
            for foot, name in enumerate(("FL", "FR", "RL", "RR")):
                ax.step(rows["episode_time_s"], foot + 0.7 * rows[f"contact_{name}"], where="post", label=name)
        ax.set(title=f"{directory.name}: vx={0.4 if case == 1 else 0.1} m/s",
               xlabel="Episode time (s)", yticks=range(4), yticklabels=["FL", "FR", "RL", "RR"])
fig.tight_layout()
fig.savefig(args.output / "contacts_comparison.png", dpi=150)
plt.close(fig)
(args.output / "summary.json").write_text(json.dumps({"training": training, "comparison": comparison}, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
