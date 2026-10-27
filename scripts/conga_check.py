"""Validate configs/rewards or run a bounded Go2 simulation; never starts training.

Examples:
    python scripts/conga_check.py --headless --config-only
    python scripts/conga_check.py --headless --num_envs 8 --steps 220
    python scripts/conga_check.py --headless --checkpoint exported/policy.pt --steps 2000
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/unitree_rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-only", action="store_true")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=220)
parser.add_argument("--task", default="Unitree-Go2-Conga")
parser.add_argument("--checkpoint", type=Path, help="Optional exported TorchScript policy, not an RSL checkpoint")
parser.add_argument(
    "--check-ppo-export", action="store_true", help="Construct PPO and check JIT/ONNX export without learning"
)
parser.add_argument("--output", type=Path, default=Path("logs/conga_check"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.hydra import register_task_to_hydra

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def main():
    for suffix in ("", "-Cadence", "-Bounce", "-Kick", "-Robust"):
        task = f"Unitree-Go2-Conga{suffix}"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        cfg.validate()
        cfg.commands.base_velocity.reference.validate()
        assert cfg.scene.terrain.terrain_type == "plane"
        assert cfg.rewards.joint_pos is None and cfg.rewards.feet_air_time is None
        assert cfg.observations.policy.conga_phase.func is not None
        runner = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        assert runner.experiment_name == "go2_conga"
        hydra_cfg, _ = register_task_to_hydra(task, "rsl_rl_cfg_entry_point")
        hydra_cfg.from_dict({"commands": {"base_velocity": {"reference": {"kick_scale": 0.5}}}})
        assert hydra_cfg.commands.base_velocity.reference.kick_scale == 0.5
        # Independent configuration instances must not share curriculum state.
        assert cfg.commands.base_velocity.reference.kick_scale != 0.5
        print(f"CONFIG PASS: {task}", flush=True)
    # Exercise real Isaac imports and command/reward code using controlled states.
    import pytest

    result = pytest.main(
        [str(Path(__file__).resolve().parents[1] / "tests/test_conga_mdp.py"), "-q", "-p", "no:cacheprovider"]
    )
    if result:
        raise RuntimeError(f"Conga component tests failed: pytest exit {result}")
    if args.config_only:
        return

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        term = env.command_manager.get_term("base_velocity")
        nominal_body, _, _ = term.foot_state()
        expected_y = torch.tensor([1.0, -1.0], device=env.device) * term.cfg.reference.nominal_front_y
        torch.testing.assert_close(nominal_body[:, :2, 1], expected_y.expand(args.num_envs, -1), atol=0.002, rtol=0)
        assert obs["policy"].shape == (args.num_envs, 55)  # original 45 + 10 phase/command features
        assert obs["critic"].shape == (args.num_envs, 70)
        # Explicit asynchronous reset: unaffected environments retain phase.
        env.episode_length_buf[:] = torch.arange(args.num_envs, device=env.device) * 7 + 1
        before = term.reference.phase.clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        assert term.reference.phase[0] == 0
        torch.testing.assert_close(before[1:], term.reference.phase[1:])
        env.reset()
        args.output.mkdir(parents=True, exist_ok=True)
        export_deploy_cfg(env, str(args.output))
        if args.check_ppo_export:
            from rsl_rl.runners import OnPolicyRunner

            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

            runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
            runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=env.device)
            network = runner.alg.policy
            normalizer = getattr(network, "actor_obs_normalizer", None)
            export_dir = str(args.output / "untrained_export")
            export_policy_as_jit(network, normalizer=normalizer, path=export_dir, filename="policy.pt")
            export_policy_as_onnx(network, normalizer=normalizer, path=export_dir, filename="policy.onnx")
            reloaded = torch.jit.load(str(Path(export_dir) / "policy.pt"), map_location=env.device)
            with torch.inference_mode():
                inference = runner.get_inference_policy(device=env.device)
                actual = inference(wrapped.get_observations())
                exported = reloaded(env.observation_manager.compute()["policy"])
                torch.testing.assert_close(actual, exported)
                assert actual.shape == (args.num_envs, 12) and torch.isfinite(actual).all()
            print("PPO/EXPORT PASS: runner, actor forward, JIT parity, ONNX serialization; no learning", flush=True)
        policy = torch.jit.load(str(args.checkpoint), map_location=env.device).eval() if args.checkpoint else None
        # Cover zero, translating, combined turning, and pure-yaw commands.
        cases = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [-0.4, 0.2, 0.5],
                [0.0, 0.0, 1.0],
                [1.25, 0.0, 0.0],
                [0.0, 1.25, 0.0],
                [0.8, 0.8, -0.5],
                [-1.25, 0.0, 0.0],
            ],
            device=env.device,
        )
        command = cases[torch.arange(args.num_envs, device=env.device) % len(cases)]
        total_falls = 0
        with (args.output / "rollout.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "step",
                    "env",
                    "episode_time_s",
                    "speed_command_mps",
                    "yaw_command_radps",
                    "travel_multiplier",
                    "recovery_relaxation",
                    "body_height_error_m",
                    "velocity_xy_error_mps",
                    "velocity_yaw_error_radps",
                    "kick_active",
                    "kick_leg",
                    "kick_y_error_m",
                    "kick_height_error_m",
                    "reference_contact_FL_FR_RL_RR",
                    "actual_contact_FL_FR_RL_RR",
                    "touchdown_FL_FR_RL_RR",
                    "touchdown_distance_FL_FR_RL_RR_s",
                    "slip_mps",
                    "fall",
                ]
            )
            for step in range(args.steps):
                with torch.inference_mode():
                    term.vel_command_b[:] = command
                    term.is_standing_env[:] = False
                    term.time_left[:] = 1e6
                    obs = env.observation_manager.compute()
                    actions = (
                        policy(obs["policy"])
                        if policy
                        else torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=env.device)
                    )
                    obs, rewards, terminated, truncated, _ = env.step(actions)
                    assert torch.isfinite(rewards).all()
                    assert all(torch.isfinite(value).all() for value in obs.values())
                    assert (term.command[:, :2].norm(dim=-1) <= 1.25 + 1e-6).all()
                    total_falls += int(terminated.sum())
                    ref = term.reference
                    body, height, contact = term.foot_state()
                    first = term.sensor.compute_first_contact(env.step_dt)[:, term.sensor_ids]
                    for i in range(args.num_envs):
                        # Auto-reset terminal states are not valid final measurements.
                        # Keep fall rows, but mark post-reset physical values empty.
                        done = bool(terminated[i] or truncated[i])
                        leg = int(ref.kick_leg[i])
                        physical = [
                            float(env.episode_length_buf[i] * env.step_dt),
                            float(command[i, :2].norm()),
                            float(command[i, 2]),
                            float(ref.travel[i]),
                            float(ref.recovery[i]),
                            float(term.robot.data.root_pos_w[i, 2] - env.scene.env_origins[i, 2] - ref.body_height[i]),
                            float((term.command[i, :2] - term.robot.data.root_lin_vel_b[i, :2]).norm()),
                            float((term.command[i, 2] - term.robot.data.root_ang_vel_b[i, 2]).abs()),
                            bool(ref.kick_active[i]),
                            leg,
                            float(body[i, leg, 1] - ref.kick_y[i]),
                            float(height[i, leg] - ref.foot_height[i, leg]),
                            ref.contact[i].tolist(),
                            contact[i].tolist(),
                            first[i].tolist(),
                            ref.touchdown_distance[i].tolist(),
                            float(
                                (term.robot.data.body_lin_vel_w[i, term.foot_ids, :2].norm(dim=-1) * contact[i]).sum()
                                / contact[i].sum().clamp_min(1)
                            ),
                        ]
                        writer.writerow([step, i, *([""] * len(physical) if done else physical), bool(terminated[i])])
        print(
            f"SIMULATION PASS: {args.steps} steps x {args.num_envs} envs; falls={total_falls}; output={args.output}",
            flush=True,
        )
        if policy is None:
            print(
                "Zero-action smoke test only: this does not validate learned locomotion or kick feasibility.",
                flush=True,
            )
    finally:
        env.close()


try:
    main()
finally:
    app.close()
