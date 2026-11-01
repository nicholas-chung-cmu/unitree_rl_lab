"""Bounded hierarchical simulation/evaluation and optional <=2-iteration PPO smoke.

Use --scenario to evaluate one of nine separate acceptance cases. With no
checkpoint this is an infrastructure test, never a learned-skill evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/unitree_rl_lab"))
from isaaclab.app import AppLauncher

SCENARIOS = {
    "four-foot": ("Step", (0.08, 0., 0.)),
    "raised": ("Raised-Step", (0.05, 0., 0.)),
    "kick": ("Kick", (0., 0., 0.)),
    "transition": ("Transition", (0.05, 0., 0.)),
    "consecutive": ("Step", (0.08, 0., 0.)),
    "march": ("Dance", (0., 0., 0.)),
    "moving-turn": ("Dance", (0.25, 0.1, .3)),
    "in-place": ("Dance", (0., 0., .3)),
    "dance": ("Dance", (.2, 0., 0.)),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-Go2-Hierarchical-Step")
parser.add_argument("--scenario", choices=SCENARIOS)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=9041)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--stochastic", action="store_true", help="Sample the checkpoint action distribution for exploration diagnostics")
parser.add_argument("--audit-rewards", action="store_true")
parser.add_argument("--diagnostic-video", action="store_true", help="Label phases and show active STEP frames at quarter speed")
parser.add_argument("--video-foot", choices=("FL", "FR", "RL", "RR"), help="Record a close view and request the diagonal containing this foot")
parser.add_argument("--config-only", action="store_true")
parser.add_argument("--test-components", action="store_true")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--check-ppo-export", action="store_true")
parser.add_argument("--ppo-iterations", type=int, choices=(0, 1, 2), default=0)
parser.add_argument("--output", type=Path, default=Path("logs/hierarchical_check"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.skills.core import Skill, Status
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"status.json").write_text(json.dumps({"status": "running"})+"\n")
    if args.test_components:
        import pytest
        tests = Path(__file__).resolve().parents[1]/"tests"
        result = pytest.main([str(tests/"test_hierarchical_skills.py"), str(tests/"test_hierarchical_adapter.py"),
                              "-q", "-s", "-p", "no:cacheprovider"])
        if result:
            raise RuntimeError(f"Component tests failed: {result}")
    for suffix in ("Step", "Raised-Step", "Kick", "Transition", "Timing", "Dance"):
        cfg = load_cfg_from_registry(f"Unitree-Go2-Hierarchical-{suffix}", "env_cfg_entry_point")
        cfg.validate()
        assert cfg.rewards.track_lin_vel_xy is None
        assert cfg.commands.skill.skill.bpm == 90
        assert cfg.commands.skill.skill.max_speed == .5
        print(f"CONFIG PASS: {suffix}", flush=True)
    if args.config_only:
        (args.output/"status.json").write_text(json.dumps({"status": "passed", "config_only": True})+"\n")
        return
    if args.scenario:
        args.task = "Unitree-Go2-Hierarchical-"+SCENARIOS[args.scenario][0]
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.observations.policy.enable_corruption = False
    if args.scenario == "raised":
        cfg.commands.skill.raised_fraction = 1.
    if args.scenario == "consecutive":
        cfg.commands.skill.isolated_steps = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.video_foot:
        if not args.enable_cameras or args.num_envs != 1 or args.scenario != "four-foot":
            raise ValueError("--video-foot requires --enable_cameras --num_envs 1 --scenario four-foot")
        cfg.viewer.resolution = (960, 720)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video_foot else None).unwrapped
    writer = None
    last_video_failure = "none"
    if args.diagnostic_video and not args.video_foot:
        raise ValueError("--diagnostic-video requires --video-foot")
    if args.video_foot:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(args.output/(args.video_foot+".mp4")), fps=round(1/env.step_dt))
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("skill")
        if args.scenario:
            command.set_velocity(torch.tensor(SCENARIOS[args.scenario][1], device=env.device).repeat(args.num_envs, 1))
        assert obs["policy"].shape == (args.num_envs, 100), obs["policy"].shape
        assert obs["critic"].shape == (args.num_envs, 112), obs["critic"].shape
        assert env.action_manager.total_action_dim == 12
        export_deploy_cfg(env, str(args.output))
        if args.stochastic and not args.checkpoint:
            raise ValueError("--stochastic requires a checkpoint")
        policy = None
        if args.check_ppo_export or args.checkpoint or args.ppo_iterations:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
            from rsl_rl.runners import OnPolicyRunner
            runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            wrapper = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
            runner = OnPolicyRunner(wrapper, runner_cfg.to_dict(), str(args.output/"ppo"), device=env.device)
            if args.checkpoint:
                runner.load(str(args.checkpoint), load_optimizer=False)
            if args.ppo_iterations:
                runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                runner.save(str(args.output/"smoke_checkpoint.pt"))
                runner.load(str(args.output/"smoke_checkpoint.pt"))
                if args.ppo_iterations == 2:
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            network = runner.alg.policy
            normalizer = getattr(network, "actor_obs_normalizer", None)
            export_policy_as_jit(network, normalizer=normalizer, path=str(args.output/"export"), filename="policy.pt")
            export_policy_as_onnx(network, normalizer=normalizer, path=str(args.output/"export"), filename="policy.onnx")
            reloaded = torch.jit.load(str(args.output/"export/policy.pt"), map_location=env.device)
            with torch.inference_mode():
                torch.testing.assert_close(runner.get_inference_policy()(wrapper.get_observations()),
                                           reloaded(env.observation_manager.compute()["policy"]))
            if args.checkpoint:
                policy = reloaded
                if args.stochastic:
                    noise = network.std.detach() if hasattr(network, "std") else network.log_std.detach().exp()
                    policy = lambda x: reloaded(x)+torch.randn(x.shape[0], 12, device=x.device)*noise
            print("PPO, checkpoint reload, JIT parity and ONNX export checks passed", flush=True)
        with torch.inference_mode():
            obs, _ = env.reset()
            # RSL-RL creates simulator caches under inference_mode during learn.
            # Reset/evaluate under the same mode after the PPO smoke.
            command.planner.tick[:] = torch.arange(args.num_envs, device=env.device)+11
            before = command.planner.tick.clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            torch.testing.assert_close(command.planner.tick[1:], before[1:])
            obs, _ = env.reset()
        records, falls, endings, prep_failures = [], 0, 0, 0
        prepared_entries = 0
        reward_sums = torch.zeros(len(env.reward_manager.active_terms), device=env.device)
        reward_samples = 0
        with (args.output/"trace.jsonl").open("w") as stream, torch.inference_mode():
            for step in range(args.steps):
                if args.video_foot:
                    # Evaluation-only selection at normal entry; no state teleport.
                    command.prepare_leg[~command.started] = 0 if args.video_foot in ("FL", "RR") else 1
                reward_mask = command.started & ~command.preparing & (command.core.kind == Skill.STEP)
                old_preparing = command.preparing.clone()
                action = policy(obs["policy"]) if policy else torch.zeros(args.num_envs, 12, device=env.device)
                obs, rewards, terminated, truncated, _ = env.step(action)
                if args.audit_rewards:
                    reward_sums += env.reward_manager._step_reward[reward_mask].sum(0)
                    reward_samples += int(reward_mask.sum())
                if writer is not None:
                    import math
                    body = command.robot.data.root_pos_w[0].cpu().tolist()
                    yaw = float(command.state().yaw[0])
                    x = 1.0 if args.video_foot.startswith("F") else -1.0
                    y = .9 if args.video_foot.endswith("L") else -.9
                    eye = (body[0]+x*math.cos(yaw)-y*math.sin(yaw),
                           body[1]+x*math.sin(yaw)+y*math.cos(yaw), body[2]+.45)
                    env.sim.set_camera_view(eye, (body[0], body[1], body[2]-.12))
                    frame = env.render()
                    repeats = 1
                    if args.diagnostic_video:
                        import numpy as np
                        from PIL import Image, ImageDraw, ImageFont
                        event = command.last_events
                        ended = bool(terminated[0] or truncated[0])
                        active = int(event["kind"][0]) == int(Skill.STEP) and not ended
                        if ended:
                            if bool(event.get("support_lost", torch.zeros(1, device=env.device))[0]):
                                last_video_failure = "required support contact lost"
                            else:
                                last_video_failure = "episode ended (see contact trace)"
                        label = "RESET (physics already reset)" if ended else ("STEP - 0.25x playback" if active else "SETTLING - 1x playback")
                        repeats = 4 if active else 1
                        picture = Image.fromarray(frame)
                        draw = ImageDraw.Draw(picture)
                        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                        font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
                        draw.rectangle((0, 0, picture.width, 110), fill=(12, 16, 22))
                        pair = "FL/RR" if args.video_foot in ("FL", "RR") else "FR/RL"
                        draw.text((15, 10), f"{pair} | {label} | simulation {(step+1)*env.step_dt:.2f}s", font=font, fill="white")
                        draw.text((15, 42), f"Skill elapsed: {float(event['duration'][0]):.2f}s | Last reset: {last_video_failure}", font=font, fill="white")
                        draw.text((15, 76), "Original policy and failure rules; no extra skill time added", font=font, fill=(190,210,230))
                        frame = np.asarray(picture)
                    for _ in range(repeats):
                        writer.append_data(frame)
                assert torch.isfinite(rewards).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert (command.planner.command[:, :2].norm(dim=-1) <= .500001).all()
                torch.testing.assert_close(env.reward_manager._step_reward.sum(-1)*env.step_dt, rewards,
                                           atol=1e-5, rtol=1e-5)
                prepared_entries += int((old_preparing & ~command.preparing & ~terminated & ~truncated).sum())
                data = {key: value.detach().cpu().tolist() for key, value in command.last_events.items()}
                data["joint_position"] = command.robot.data.joint_pos.cpu().tolist()
                data["applied_action"] = env.action_manager.action.cpu().tolist()
                data["preparing"] = command.preparing.cpu().tolist()
                falls += sum(fall and bool(terminated[i] or truncated[i]) for i, fall in enumerate(data.get("fall", [])))
                endings += int((terminated | truncated).sum())
                prep_failures += sum(data.get("preparation_failed", []))
                for i in range(args.num_envs):
                    row = {key: values[i] for key, values in data.items()}
                    row.update(step=step, env=i, episode_end=bool(terminated[i] or truncated[i]))
                    stream.write(json.dumps(row)+"\n")
                    if row.get("event"):
                        records.append(row)
        groups = {}
        for kind in (Skill.STEP, Skill.KICK):
            for entry in ("four-foot", "raised"):
                rows = [r for r in records if r["kind"] == int(kind) and
                        ("raised" if any(r["entry_raised"]) else "four-foot") == entry]
                success = [r for r in rows if r["status"] == int(Status.SUCCESS)]
                transitions = [r for r in rows if r["from_kick"] and kind == Skill.STEP]
                timed = [r for r in rows if r["missing_touchdowns"] == 0 and kind == Skill.STEP]
                groups[f"{kind.name}/{entry}"] = {
                    "trials": len(rows), "successes": len(success),
                    "success_rate": len(success)/len(rows) if rows else None,
                    "kick_to_step_trials": len(transitions),
                    "kick_to_step_success_rate": sum(r["status"] == 1 and r["premature_touchdowns"] == 0 for r in transitions)/len(transitions) if transitions else None,
                    "contact_sequence_correct_rate": sum(r["contact_sequence_correct"] for r in rows)/len(rows) if rows and kind == Skill.STEP else None,
                    "missing_touchdowns": sum(r["missing_touchdowns"] for r in rows) if kind == Skill.STEP else None,
                    "mean_touchdown_timing_error": sum(r["touchdown_timing_error"] for r in timed)/len(timed) if timed else None,
                    **{f"mean_{metric}": sum(r[metric] for r in rows)/len(rows) if rows else None for metric in
                       ("duration", "foot_error", "body_error", "heading_error", "slip", "premature_touchdowns", "support_departures", "timing_error", "dance_timing_error")},
                    **{f"mean_{metric}": sum(r[metric] for r in rows)/len(rows) if rows and kind == Skill.KICK else None
                       for metric in ("kick_height_error", "kick_lateral_error")},
                }
        summary = dict(task=args.task, scenario=args.scenario, steps=args.steps, num_envs=args.num_envs,
                       stochastic=args.stochastic,
                       trained_checkpoint=str(args.checkpoint) if args.checkpoint else None,
                       falls=falls, episode_endings=endings, preparation_failures=prep_failures,
                       prepared_entries=prepared_entries, groups=groups,
                       observation_dimensions=[100, 112], action_dimension=12,
                       note="No checkpoint: zero-action infrastructure smoke only" if not policy else "Controlled policy evaluation")
        if args.audit_rewards:
            audit = dict(active_step_samples=reward_samples, weighted_mean_rates=dict(zip(
                env.reward_manager.active_terms, (reward_sums/max(1,reward_samples)).cpu().tolist())))
            (args.output/"reward_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        (args.output/"status.json").write_text(json.dumps({"status": "passed", "scope": "infrastructure"})+"\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"status.json").write_text(json.dumps({"status": "failed", "error": traceback.format_exc()})+"\n")
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    app.close()
