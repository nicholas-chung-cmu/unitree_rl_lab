"""Strict admission audit for the overnight experiment; no learning.

Evaluate one initialization attempt per environment, balanced FL/FR. A failed
preparation is a failed preflight, not a policy landing trial. Emit result.json
before simulator shutdown; process exit status alone is not a success signal.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/unitree_rl_lab"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("A", "B", "C"), required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=9029)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trace", action="store_true", help="Record preparation diagnostics before reset")
parser.add_argument("--hold-seconds", type=float, help="Override stationary hold duration for stress probes")
parser.add_argument("--video-env", type=int, help="Record this environment; requires --enable_cameras")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
started = time.time()
(args.output/"result.json").write_text(json.dumps({"status": "running", "started": started})+"\n")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.skills.core import Skill
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    task = "Unitree-Go2-Hierarchical-"+{"A": "Step", "B": "Raised-Step", "C": "Kick"}[args.stage]
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.observations.policy.enable_corruption = False
    if args.stage == "B":
        cfg.commands.skill.raised_fraction = 1.
    if args.hold_seconds is not None:
        cfg.commands.skill.preparation.hold_seconds = args.hold_seconds
        cfg.commands.skill.preparation.timeout_seconds = max(10., 7.+args.hold_seconds)
    if args.video_env is not None:
        if not args.enable_cameras:
            raise ValueError("--video-env requires --enable_cameras")
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = args.video_env
        cfg.viewer.eye = (1.1, 1.1 if args.video_env % 2 == 0 else -1.1, .65)
        cfg.viewer.lookat = (0., 0., -.08)
        cfg.viewer.resolution = (960, 720)
    cfg.validate()
    dump_yaml(str(args.output/"env.yaml"), cfg)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video_env is not None else None).unwrapped
    writer = None
    if args.video_env is not None:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(args.output/"preparation.mp4"), fps=round(1/env.step_dt))
    try:
        obs, _ = env.reset()
        cmd = env.command_manager.get_term("skill")
        export_deploy_cfg(env, str(args.output))
        finished = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        admitted = torch.zeros_like(finished)
        rows = []
        resets = 0
        attempts = []
        targets = torch.arange(args.num_envs, device=env.device).remainder(2)
        with torch.inference_mode(), (args.output/"preparation_trace.jsonl").open("w") as trace:
            for tick in range(int((cfg.commands.skill.preparation.timeout_seconds+2.)/env.step_dt)):
                # Deterministic balanced requested diagonals/raised front legs.
                waiting = ~cmd.started
                cmd.prepare_leg[waiting] = targets[waiting]
                previous_preparing = cmd.preparing.clone()
                obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 12, device=env.device))
                if writer is not None:
                    writer.append_data(env.render())
                assert torch.isfinite(reward).all(), "Nonfinite rewards"
                assert all(torch.isfinite(value).all() for value in obs.values()), "Nonfinite observations"
                assert torch.isfinite(cmd.robot.data.joint_pos).all(), "Nonfinite joint positions"
                assert torch.isfinite(cmd.robot.data.joint_vel).all(), "Nonfinite joint velocities"
                done = terminated | truncated
                resets += int(done.sum())
                expected_kind = Skill.KICK if args.stage == "C" else Skill.STEP
                new_goal = (cmd.core.kind == expected_kind) & (cmd.core.elapsed == 0) & ~done & ~finished & ~admitted
                if args.stage == "B":
                    new_goal &= previous_preparing & ~cmd.preparing
                state = cmd.state()
                if args.trace and cmd.prepare_diagnostics:
                    diagnostic = {k:v.cpu().tolist() for k,v in cmd.prepare_diagnostics.items()}
                    for i in (~finished & previous_preparing).nonzero().flatten().tolist():
                        trace.write(json.dumps(dict(tick=tick, env=i, **{k:v[i] for k,v in diagnostic.items()}))+"\n")
                for i in new_goal.nonzero().flatten().tolist():
                    contacts = cmd.core.confirmed[i]
                    selected = cmd.core.selected[i]
                    q = cmd.robot.data.joint_pos[i]
                    limits = cmd.robot.data.joint_pos_limits[i]
                    joint_valid = bool(((q >= limits[:, 0]-.02) & (q <= limits[:, 1]+.02)).all())
                    raised = cmd.core.entry_raised[i]
                    if args.stage == "B":
                        contact_valid = bool(contacts.sum() == 3 and raised.sum() == 1 and raised[targets[i]])
                        contact_valid &= bool((~raised | selected).all())
                        contact_valid &= bool(state.feet[i, targets[i], 2]-state.ground[i] > .03)
                    else:
                        contact_valid = bool(contacts.all() and not raised.any())
                    selected_valid = bool(selected[targets[i]])
                    if args.stage != "C":
                        selected_valid &= bool(selected[3-targets[i]] and selected.sum() == 2)
                    else:
                        selected_valid &= bool(selected.sum() == 1)
                    row = dict(env=i, front_leg="FL" if int(targets[i]) == 0 else "FR",
                               entry_time_s=(tick+1)*env.step_dt, joint_valid=joint_valid,
                               contact_valid=contact_valid, selected_valid=selected_valid,
                               tilt=float(state.tilt[i]), angular_speed=float(cmd.robot.data.root_ang_vel_w[i].norm()),
                               base_speed=float(cmd.robot.data.root_lin_vel_w[i].norm()),
                               max_foot_speed=float(state.foot_velocity[i].norm(dim=-1).max()),
                               stable_hold_s=float(cmd.prepare_confirm[i]),
                               settle_time_s=float(cmd.settle_elapsed[i]),
                               settle_hold_s=float(cmd.settle_confirm[i]),
                               contacts=contacts.tolist(), selected=selected.tolist(), entry_raised=raised.tolist(),
                               sole_clearance=(state.feet[i, :, 2]-state.ground[i]).tolist(),
                               joint_positions=q.tolist(), joint_velocities=cmd.robot.data.joint_vel[i].tolist())
                    row["valid"] = joint_valid and contact_valid and selected_valid and row["tilt"] < .30 and row["angular_speed"] < 1.5
                    if args.stage == "B":
                        pcfg = cfg.commands.skill.preparation
                        row["valid"] &= (row["tilt"] < .2 and row["angular_speed"] < pcfg.max_angular_speed
                                         and row["base_speed"] < pcfg.max_body_speed
                                         and row["max_foot_speed"] < pcfg.max_foot_speed
                                         and row["stable_hold_s"] >= pcfg.hold_seconds)
                    if args.stage in ("A", "C"):
                        row["valid"] &= (row["settle_time_s"] >= 1. and row["settle_hold_s"] >= .5
                                         and row["base_speed"] < .05 and row["angular_speed"] < .2
                                         and row["max_foot_speed"] < .05 and row["tilt"] < .15)
                    rows.append(row)
                    admitted[i] = True
                    if args.stage == "B":
                        finished[i] = True
                        attempts.append(dict(env=i, outcome="admitted" if row["valid"] else "invalid_admission"))
                for i in (done & ~finished).nonzero().flatten().tolist():
                    if args.stage == "B":
                        event = cmd.last_events
                        reason = "preparation_timeout" if bool(event["preparation_failed"][i]) else "preparation_termination"
                        attempts.append(dict(env=i, outcome=reason, fall=bool(event["fall"][i]),
                                             tilt=float(event["tilt"][i]), base_height=float(event["base_height"][i])))
                    else:
                        attempts.append(dict(env=i, outcome="reset_after_valid_entry" if admitted[i] else "failed_before_entry"))
                    finished[i] = True
                if bool(finished.all()):
                    break
        for i in (~finished).nonzero().flatten().tolist():
            attempts.append(dict(env=i, outcome="initialization_timeout"))
        passed = len(rows) == args.num_envs and all(row["valid"] for row in rows)
        passed &= bool(finished.all())
        if args.stage != "B":
            passed &= all(x["outcome"] == "reset_after_valid_entry" for x in attempts)
        by_leg = {}
        for name, value in (("FL", 0), ("FR", 1)):
            relevant = [r for r in rows if r["front_leg"] == name]
            count = int((targets == value).sum())
            by_leg[name] = dict(attempts=count, admitted=len(relevant), valid=sum(r["valid"] for r in relevant))
        result = dict(status="passed" if passed else "failed", task=task, stage=args.stage,
                      seed=args.seed, num_envs=args.num_envs, duration_s=time.time()-started,
                      control_steps=tick+1, reset_events=resets, admissions=rows, attempts=attempts, by_leg=by_leg,
                      rationale="All initialization attempts must admit valid measured states; failed preparation blocks overnight training.",
                      policy_success_evaluated=False)
        (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("admissions", "attempts")}, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


try:
    main()
except BaseException:
    import traceback
    error = traceback.format_exc()
    print(error, flush=True)
    (args.output/"result.json").write_text(json.dumps({"status":"failed", "error":error, "duration_s":time.time()-started}, indent=2)+"\n")
    raise
finally:
    app.close()
