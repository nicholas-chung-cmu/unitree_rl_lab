import gymnasium as gym

gym.register(
    id="Unitree-Go2-Conga-Cadence-Paired",
    entry_point="unitree_rl_lab.utils.cadence_training:CadenceFadeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_cadence_paired_env_cfg:CongaCadencePairedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_cadence_paired_env_cfg:CongaCadencePairedEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaCadencePairedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-Cadence-Start",
    entry_point="unitree_rl_lab.utils.cadence_training:CadenceFadeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_cadence_start_env_cfg:CongaCadenceStartEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_cadence_start_env_cfg:CongaCadenceStartEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaCadenceStartPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-March-Balanced",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_march_balanced_env_cfg:CongaMarchBalancedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_march_balanced_env_cfg:CongaMarchBalancedEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaMarchBalancedPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-March",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_march_env_cfg:CongaMarchEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_march_env_cfg:CongaMarchEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaMarchPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-Pause",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_pause_env_cfg:CongaPauseEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_pause_env_cfg:CongaPauseEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaPausePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-History",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_history_env_cfg:CongaHistoryEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_history_env_cfg:CongaHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaHistoryPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-Early",
    entry_point="unitree_rl_lab.utils.early_training:EarlyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_early_env_cfg:CongaEarlyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_early_env_cfg:CongaEarlyEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaEarlyPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-Basic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_basic_env_cfg:CongaBasicEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_basic_env_cfg:CongaBasicEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaBasicPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Conga-Walk-Validation",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conga_walk_validation_env_cfg:CongaWalkValidationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.conga_walk_validation_env_cfg:CongaWalkValidationEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaWalkValidationPPORunnerCfg",
    },
)

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


gym.register(
    id='Unitree-Go2-Conga-Duration',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.conga_duration_env_cfg:CongaDurationEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.conga_duration_env_cfg:CongaDurationEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaDurationPPORunnerCfg',
    },
)


gym.register(
    id='Unitree-Go2-Conga-Duration-Strong',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.conga_duration_env_cfg:CongaDurationStrongEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.conga_duration_env_cfg:CongaDurationStrongEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaDurationStrongPPORunnerCfg',
    },
)


gym.register(
    id='Unitree-Go2-Conga-Duration-Average',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.conga_duration_env_cfg:CongaDurationAverageEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.conga_duration_env_cfg:CongaDurationAverageEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaDurationAveragePPORunnerCfg',
    },
)


gym.register(
    id='Unitree-Go2-Conga-Kick-Test',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.conga_kick_test_env_cfg:CongaKickTestEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.conga_kick_test_env_cfg:CongaKickTestEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaBasicPPORunnerCfg',
    },
)


gym.register(
    id='Unitree-Go2-Conga-Kick-Intro',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.conga_kick_train_env_cfg:CongaKickTrainEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.conga_kick_train_env_cfg:CongaKickTrainEnvCfg',
        'rsl_rl_cfg_entry_point':'unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaKickTrainPPORunnerCfg',
    },
)


gym.register(
    id='Unitree-Go2-Conga-Kick-Refine',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.conga_kick_train_env_cfg:CongaKickRefineEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.conga_kick_train_env_cfg:CongaKickRefineEnvCfg',
        'rsl_rl_cfg_entry_point':'unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg:CongaKickTrainPPORunnerCfg',
    },
)

# Separate architecture and checkpoint family; preserve every legacy registration.
for suffix, config in (
    ("Step", "HierarchicalStepEnvCfg"),
    ("Raised-Step", "HierarchicalRaisedStepEnvCfg"),
    ("Kick", "HierarchicalKickEnvCfg"),
    ("Transition", "HierarchicalTransitionEnvCfg"),
    ("Timing", "HierarchicalTimingEnvCfg"),
    ("Dance", "HierarchicalDanceEnvCfg"),
):
    gym.register(
        id=f"Unitree-Go2-Hierarchical-{suffix}",
        entry_point="unitree_rl_lab.tasks.locomotion.skills.adapter:SkillEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hierarchical_env_cfg:{config}",
            "play_env_cfg_entry_point": f"{__name__}.hierarchical_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.hierarchical_ppo_cfg:HierarchicalPPORunnerCfg",
        },
    )

# CPG residual policies have their own reference, observation and checkpoint family.
gym.register(
    id='Unitree-Go2-CPG-Walk',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point':f'{__name__}.cpg_env_cfg:CPGWalkEnvCfg',
        'play_env_cfg_entry_point':f'{__name__}.cpg_env_cfg:CPGWalkEnvCfg',
        'rsl_rl_cfg_entry_point':'unitree_rl_lab.tasks.locomotion.agents.cpg_ppo_cfg:CPGPPORunnerCfg',
    },
)

for suffix,config in [('Timing','CPGTimingEnvCfg'),('Travel','CPGTravelEnvCfg'),('Dance','CPGDanceEnvCfg'),('Kick-Hold','CPGKickHoldEnvCfg'),('Bounce','CPGBounceEnvCfg')]:
    gym.register(
        id=f'Unitree-Go2-CPG-{suffix}',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point':f'{__name__}.cpg_env_cfg:{config}',
            'play_env_cfg_entry_point':f'{__name__}.cpg_env_cfg:{config}',
            'rsl_rl_cfg_entry_point':'unitree_rl_lab.tasks.locomotion.agents.cpg_ppo_cfg:CPGPPORunnerCfg',
        },
    )
