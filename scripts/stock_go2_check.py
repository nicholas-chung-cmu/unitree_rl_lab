"""Evaluate stock or conga-basic Go2 checkpoints on identical fixed commands."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-Go2-Velocity",
                    choices=["Unitree-Go2-Conga-Kick-Intro", "Unitree-Go2-Conga-Kick-Refine", "Unitree-Go2-Conga-Duration-Average", "Unitree-Go2-Conga-Duration-Strong", "Unitree-Go2-Conga-Duration", "Unitree-Go2-Velocity", "Unitree-Go2-Conga-Basic", "Unitree-Go2-Conga-Cadence-Start", "Unitree-Go2-Conga-Cadence-Paired", "Unitree-Go2-Conga-Early", "Unitree-Go2-Conga-History", "Unitree-Go2-Conga-Pause", "Unitree-Go2-Conga-March", "Unitree-Go2-Conga-March-Balanced"])
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", action="store_true", help="Record an MP4 with a camera following environment zero")
parser.add_argument("--reward-audit", action="store_true", help="Log sole heights and weighted stock reward rates without changing rewards")
parser.add_argument("--command", type=float, nargs=3, help="Use a single fixed vx, vy, yaw command")
parser.add_argument("--phase-offset", type=float, default=0., help="Evaluation-only march clock offset; does not change episode timeout")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from isaaclab_tasks.utils import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

CASES = [
    [0, 0, 0], [0.4, 0, 0], [-0.4, 0, 0], [0, 0.25, 0],
    [0, -0.25, 0], [0, 0, 1], [0, 0, -1], [0.3, 0.15, 0.5],
    [0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0],
    [0, -0.1, 0], [0, 0, 1], [0, 0, -1], [0.1, 0.1, 0.5],
]
if args.task in ("Unitree-Go2-Conga-March", "Unitree-Go2-Conga-March-Balanced"):
    if args.command is not None and any(args.command):
        raise ValueError("March-first evaluation requires zero commands")
    CASES = [[0, 0, 0] for _ in CASES]
if args.command is not None:
    CASES = [args.command]
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.seed = 42
    cfg.scene.num_envs = len(CASES)
    cfg.sim.device = args.device or "cuda:0"
    # Identical controlled evaluation for all policies; training config is untouched.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.height_scanner = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.debug_vis = False
    if args.video:
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (2.5, 2.5, 1.6)
        cfg.viewer.lookat = (0.0, 0.0, 0.25)
        cfg.viewer.resolution = (960, 720)
    cfg.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    gym_env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        gym_env = gym.wrappers.RecordVideo(
            gym_env, video_folder=str(args.output / "videos"), step_trigger=lambda step: step == 0,
            video_length=args.steps, name_prefix="go2", disable_logger=True,
        )
    env = gym_env.unwrapped
    if args.phase_offset:
        if args.task != "Unitree-Go2-Conga-March":
            raise ValueError('Phase-offset audit is restricted to the march task')
        from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import generate_reference
        term = env.command_manager.get_term('base_velocity')
        # Change only this evaluator's command instance; observations and rewards
        # share its reference. Keep physical reset state and episode age intact.
        class AuditPhaseCommand(type(term)):
            @property
            def reference(self):
                return generate_reference(self._env.episode_length_buf*self._env.step_dt+args.phase_offset,self.cfg.reference)
        term.__class__ = AuditPhaseCommand
    try:
        obs, _ = gym_env.reset()
        extra = 0 if args.task == "Unitree-Go2-Velocity" else 10
        assert obs["policy"].shape == (len(CASES), 45 + extra)
        assert obs["critic"].shape == (len(CASES), 60 + extra)
        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        wrapped = RslRlVecEnvWrapper(gym_env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        if args.task == "Unitree-Go2-Conga-Early":
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            env.common_step_counter = saved['infos']['early_stage']['control_steps']
            env.early_frozen_step = env.common_step_counter
        if args.task in ("Unitree-Go2-Conga-Cadence-Start", "Unitree-Go2-Conga-Cadence-Paired"):
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            env.common_step_counter = (saved.get("infos") or {}).get("cadence_stage", {}).get("control_steps", 0)
            env.cadence_fade_frozen_step = env.common_step_counter
        args.output.mkdir(parents=True, exist_ok=True)
        policy = runner.alg.policy
        export_policy_as_jit(policy, normalizer=getattr(policy, "actor_obs_normalizer", None),
                             path=str(args.output / "exported"), filename="policy.pt")
        actor = torch.jit.load(str(args.output / "exported/policy.pt"), map_location=env.device).eval()
        with torch.inference_mode():
            obs = env.observation_manager.compute()
            torch.testing.assert_close(actor(obs["policy"]), runner.get_inference_policy()(wrapped.get_observations()))
        export_deploy_cfg(env, str(args.output))
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        ids, names = robot.find_bodies(FEET, preserve_order=True)
        sensor_ids, sensor_names = sensor.find_bodies(FEET, preserve_order=True)
        assert names == FEET and sensor_names == FEET
        command = env.command_manager.get_term("base_velocity")
        targets = torch.tensor(CASES, device=env.device)
        rows = []
        columns = ["step", "env", "episode_time_s", "done", "fall", "timeout",
                   "vx", "vy", "yaw", "xy_error", "yaw_error", "height", "slip"]
        columns += [f"contact_{f}" for f in ("FL", "FR", "RL", "RR")]
        columns += [f"touchdown_{f}" for f in ("FL", "FR", "RL", "RR")]
        if args.reward_audit:
            columns += [f"clearance_{f}" for f in ("FL", "FR", "RL", "RR")]
            columns += [f"reward_rate_{name}" for name in env.reward_manager.active_terms]
        has_offset = hasattr(command, "phase_offset")
        if has_offset:
            columns += ["phase_offset_s"]
        lengths = [[] for _ in CASES]
        elapsed = np.zeros(len(CASES))
        with torch.inference_mode():
            for step in range(args.steps):
                command.vel_command_b[:] = targets
                command.is_standing_env[:] = False
                command.is_heading_env[:] = False
                command.time_left[:] = 1e6
                obs = env.observation_manager.compute()
                offset_before = command.phase_offset.clone() if has_offset else None
                obs, reward, terminated, timeout, _ = gym_env.step(actor(obs["policy"]))
                assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
                contact = sensor.data.net_forces_w[:, sensor_ids].norm(dim=-1) > 1
                first = sensor.compute_first_contact(env.step_dt)[:, sensor_ids]
                slip = robot.data.body_lin_vel_w[:, ids, :2].norm(dim=-1)
                velocity = robot.data.root_lin_vel_b[:, :2]
                yaw = robot.data.root_ang_vel_b[:, 2]
                done = terminated | timeout
                scalars = torch.stack([
                    env.episode_length_buf * env.step_dt, done, terminated, timeout,
                    velocity[:, 0], velocity[:, 1], yaw,
                    (velocity - targets[:, :2]).norm(dim=-1), (yaw - targets[:, 2]).abs(),
                    robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
                    (slip * contact).sum(-1) / contact.sum(-1).clamp_min(1),
                ], dim=-1)
                parts = [scalars, contact.float(), first.float()]
                if args.reward_audit:
                    clearance = robot.data.body_pos_w[:, ids, 2] - env.scene.env_origins[:, None, 2] - 0.022
                    # Isaac Lab 2.3.2 stores weight * raw_term here (before dt).
                    rates = env.reward_manager._step_reward
                    torch.testing.assert_close(rates.sum(-1) * env.step_dt, reward, atol=1e-5, rtol=1e-5)
                    parts.extend([clearance, rates])
                if has_offset:
                    parts.append(offset_before[:, None])
                data = torch.cat(parts, dim=-1).cpu().numpy()
                elapsed += env.step_dt
                for i, values in enumerate(data):
                    if values[1]:
                        lengths[i].append(float(elapsed[i]))
                        elapsed[i] = 0
                        values[4:] = np.nan  # reset measurements are not terminal state measurements
                    rows.append([step, i, *values.tolist()])
        with (args.output / "rollout.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        data = np.array(rows)
        summary = []
        for i, target in enumerate(CASES):
            group = data[data[:, 1] == i]
            valid = group[(group[:, 2] >= 2.0) & (group[:, 3] == 0)]
            result = {"case": i, "group": "custom" if args.command is not None else ("conga_range" if i < 8 else "stock_initial"),
                      "command": target, "falls": int(group[:, 4].sum()),
                      "timeouts": int(group[:, 5].sum()), "completed_lengths_s": lengths[i],
                      "steady_samples": len(valid)}
            if len(valid):
                contacts = valid[:, 13:17].astype(bool)
                two = contacts.sum(-1) == 2
                diagonal = two & ((contacts[:, 0] & contacts[:, 3]) | (contacts[:, 1] & contacts[:, 2]))
                result.update(
                    xy_error_mps=float(valid[:, 9].mean()), yaw_error_radps=float(valid[:, 10].mean()),
                    actual_velocity=valid[:, 6:9].mean(0).tolist(), height_m=float(valid[:, 11].mean()),
                    slip_mps=float(valid[:, 12].mean()), all_four_fraction=float(contacts.all(-1).mean()),
                    two_contact_fraction=float(two.mean()), diagonal_support_fraction=float(diagonal.mean()),
                    diagonal_fraction_given_two=float(diagonal.sum() / two.sum()) if two.any() else None,
                    touchdowns_per_foot_hz=(valid[:, 17:21].sum(0) / (len(valid) * env.step_dt)).tolist(),
                )
            summary.append(result)
        result = {"task": args.task, "reward_audit": args.reward_audit, "phase_offset_s":args.phase_offset,
                  "reward_rate_units": "weighted reward per simulation second, before multiplying by step_dt",
                  "step_dt": env.step_dt,
                  "checkpoint": str(args.checkpoint.resolve()),
                  "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "steps": args.steps, "seed": 42, "cases": summary}
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"EVALUATION COMPLETE: falls={sum(r['falls'] for r in summary)}, "
              f"timeouts={sum(r['timeouts'] for r in summary)}; {args.output}", flush=True)
    finally:
        gym_env.close()


try:
    main()
finally:
    app.close()
