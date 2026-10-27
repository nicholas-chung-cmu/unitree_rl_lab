import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

for suffix, config in (
    ("", "CongaEnvCfg"),
    ("-Cadence", "CongaCadenceEnvCfg"),
    ("-Bounce", "CongaBounceEnvCfg"),
    ("-Kick", "CongaKickEnvCfg"),
    ("-Robust", "CongaRobustEnvCfg"),
):
    gym.register(
        id=f"Unitree-Go2-Conga{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.conga_env_cfg:{config}",
            "play_env_cfg_entry_point": f"{__name__}.conga_env_cfg:" + ("CongaPlayEnvCfg" if not suffix else config),
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaPPORunnerCfg",
        },
    )
