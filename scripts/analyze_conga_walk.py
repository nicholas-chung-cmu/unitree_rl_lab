"""Summarize the temporary walking experiment and its fixed-command evaluation."""

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COMMANDS = [(0, 0, 0), (0.4, 0, 0), (-0.4, 0, 0), (0, 0.25, 0),
            (0, -0.25, 0), (0, 0, 1), (0, 0, -1), (0.3, 0.15, 0.5)]


def training_summary(run, output):
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    series = {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}
    summary = {}
    with (output / "training_scalars.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "iteration", "value"])
        for tag, points in series.items():
            values = np.array([p.value for p in points])
            summary[tag] = {"first20": float(values[:20].mean()), "last20": float(values[-20:].mean()),
                            "last": float(values[-1]), "count": len(values)}
            writer.writerows((tag, p.step, p.value) for p in points)
    groups = [
        ("Reward", ["Train/mean_reward"]),
        ("Episode length (control steps; 50 Hz)", ["Train/mean_episode_length"]),
        ("Velocity errors", ["Metrics/base_velocity/velocity_xy_error_mps",
                             "Metrics/base_velocity/velocity_yaw_error_radps"]),
        ("Contacts and timing", ["Metrics/base_velocity/contact_mae",
                                 "Metrics/base_velocity/touchdown_error_s"]),
        ("Touchdowns per foot per second", ["Metrics/base_velocity/touchdowns_per_foot_hz",
                                            "Metrics/base_velocity/reference_touchdowns_per_foot_hz"]),
        ("Weighted episode rewards", ["Episode_Reward/conga_contact", "Episode_Reward/conga_cadence",
                                      "Episode_Reward/conga_clearance", "Episode_Reward/conga_marching",
                                      "Episode_Reward/walk_missed_swing", "Episode_Reward/walk_nominal_height",
                                      "Episode_Reward/action_rate"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, (title, tags) in zip(axes.flat, groups):
        for tag in tags:
            points = series.get(tag, [])
            ax.plot([p.step for p in points], [p.value for p in points], label=tag.split("/")[-1])
        ax.set(title=title, xlabel="PPO iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    return summary


def evaluation_summary(path, output):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    results = []
    fig, axes = plt.subplots(4, 2, figsize=(14, 11))
    velocity_fig, velocity_axes = plt.subplots(4, 2, figsize=(14, 11))
    for env, ax in zip(sorted({int(r["env"]) for r in rows}), axes.flat):
        all_rows = [r for r in rows if int(r["env"]) == env]
        falls = sum(r["fall"] == "True" for r in all_rows)
        # Each segment is one episode; never match events across resets.
        segments, segment, completed_lengths = [], [], []
        for row in all_rows:
            if not row["episode_time_s"]:
                if segment:
                    completed_lengths.append(float(segment[-1]["episode_time_s"]) + 0.02)
                    segments.append(segment)
                    segment = []
                continue
            if segment and float(row["episode_time_s"]) < float(segment[-1]["episode_time_s"]):
                segments.append(segment)
                segment = []
            segment.append(row)
        if segment:
            segments.append(segment)
        selected = [r for s in segments for r in s if float(r["episode_time_s"]) >= 2.0]
        if not selected:
            results.append({"env": env, "command_vx_vy_yaw": COMMANDS[env % 8],
                            "falls": falls, "valid_after_warmup": 0,
                            "completed_episode_lengths_s": completed_lengths,
                            "steady_state_metrics": None})
            for empty_ax in (ax, velocity_axes.flat[env]):
                empty_ax.set_title(f"Case {env}: no episode survives the 2 s warmup")
                empty_ax.text(0.5, 0.5, "No steady-state measurements", ha="center",
                              transform=empty_ax.transAxes)
            continue
        expected_count = actual_count = matches = 0
        errors, foot_counts, duration = [], np.zeros(4), 0.0
        for segment in segments:
            segment = [r for r in segment if float(r["episode_time_s"]) >= 2.0]
            if len(segment) < 10:
                continue
            times = np.array([float(r["episode_time_s"]) for r in segment])
            actual = np.array([ast.literal_eval(r["touchdown_FL_FR_RL_RR"]) for r in segment])
            # Full tolerance windows only; discard boundary events symmetrically.
            lo, hi = times[0] + 0.08, times[-1] - 0.08
            duration += hi - lo
            for foot in range(4):
                offset = 0.38 if foot in (0, 3) else 0.88
                expected = np.arange(offset, hi + 1, 1.0)
                expected = expected[(expected >= lo) & (expected <= hi)]
                observed = times[actual[:, foot].astype(bool)]
                observed = observed[(observed >= lo) & (observed <= hi)]
                expected_count += len(expected)
                actual_count += len(observed)
                foot_counts[foot] += len(observed)
                candidates = sorted((abs(a - e), i, j) for i, e in enumerate(expected)
                                    for j, a in enumerate(observed) if abs(a - e) <= 0.080001)
                used_e, used_a = set(), set()
                for error, i, j in candidates:
                    if i not in used_e and j not in used_a:
                        used_e.add(i)
                        used_a.add(j)
                        errors.append(error)
                        matches += 1
        ref = np.array([ast.literal_eval(r["reference_contact_FL_FR_RL_RR"]) for r in selected])
        contact = np.array([ast.literal_eval(r["actual_contact_FL_FR_RL_RR"]) for r in selected])
        result = {"env": env, "command_vx_vy_yaw": COMMANDS[env % 8],
                  "falls": falls, "valid_after_warmup": len(selected),
                  "completed_episode_lengths_s": completed_lengths,
                  "mean_actual_velocity": [float(np.mean([float(r[k]) for r in selected]))
                                           for k in ("actual_vx_mps", "actual_vy_mps", "actual_yaw_radps")],
                  "xy_error_mps": float(np.mean([float(r["velocity_xy_error_mps"]) for r in selected])),
                  "yaw_error_radps": float(np.mean([float(r["velocity_yaw_error_radps"]) for r in selected])),
                  "contact_mae": float(np.abs(ref - contact).mean()),
                  "all_four_contact_fraction": float(contact.all(axis=1).mean()),
                  "slip_mps": float(np.mean([float(r["slip_mps"]) for r in selected])),
                  "expected_touchdowns": expected_count, "actual_touchdowns": actual_count,
                  "matched_touchdowns": matches, "missing_touchdowns": expected_count - matches,
                  "extra_touchdowns": actual_count - matches,
                  "matched_timing_mae_s": float(np.mean(errors)) if errors else None,
                  "touchdowns_per_foot_hz": (foot_counts / max(duration, 1e-9)).tolist()}
        results.append(result)
        velocity_ax = velocity_axes.flat[env]
        for key, color, label, command in zip(
            ("actual_vx_mps", "actual_vy_mps", "actual_yaw_radps"),
            ("C0", "C1", "C2"), ("vx (m/s)", "vy (m/s)", "yaw (rad/s)"), COMMANDS[env % 8]
        ):
            velocity_ax.plot([int(r["step"]) * 0.02 for r in selected],
                             [float(r[key]) for r in selected], color=color, label=label, linewidth=0.7)
            velocity_ax.axhline(command, color=color, linestyle=":")
        velocity_ax.set(title=f"Case {env}: target {COMMANDS[env % 8]}", xlabel="Rollout time (s)")
        velocity_ax.legend(fontsize=8)
        example = [r for r in segments[0] if 2 <= float(r["episode_time_s"]) <= 6]
        t = [float(r["episode_time_s"]) for r in example]
        for foot, name in enumerate(("FL", "FR", "RL", "RR")):
            y = [ast.literal_eval(r["actual_contact_FL_FR_RL_RR"])[foot] for r in example]
            target = [ast.literal_eval(r["reference_contact_FL_FR_RL_RR"])[foot] for r in example]
            ax.step(t, np.array(y) * 0.7 + foot, where="post", label=name)
            ax.plot(t, np.array(target) * 0.7 + foot, color="black", linestyle=":", alpha=0.5)
        ax.set(title=f"Command case {env}: actual vs dotted target", xlabel="Episode time (s)",
               yticks=range(4), yticklabels=["FL", "FR", "RL", "RR"])
    fig.tight_layout()
    fig.savefig(output / "contact_timing.png", dpi=150)
    plt.close(fig)
    velocity_fig.tight_layout()
    velocity_fig.savefig(output / "velocity_tracking.png", dpi=150)
    plt.close(velocity_fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--baseline-run", type=Path, help="Optional earlier run for survival comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"training": training_summary(args.run, args.output)}
    if args.rollout:
        result["evaluation"] = evaluation_summary(args.rollout, args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.baseline_run:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for path, label in ((args.baseline_run, "Original rewards"), (args.run, "Revised rewards")):
            events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
            for ax, tag, scale, title in zip(
                axes, ("Train/mean_episode_length", "Episode_Termination/time_out"),
                (0.02, 1.0), ("Mean training episode length (s)", "Logged timeout fraction")
            ):
                values = events.Scalars(tag)
                ax.plot([p.step for p in values], [p.value * scale for p in values], label=label)
                ax.set(title=title, xlabel="PPO iteration")
                ax.grid(alpha=0.3)
                ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "baseline_comparison.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
