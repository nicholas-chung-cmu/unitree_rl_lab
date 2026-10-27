"""Flat-ground Go2 conga task and checkpoint-compatible curriculum stages."""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.mdp import conga

from .velocity_env_cfg import CommandsCfg, ObservationsCfg, RewardsCfg, RobotEnvCfg


@configclass
class CongaCommandsCfg(CommandsCfg):
    base_velocity = conga.CongaVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.2,
        rel_turning_envs=0.15,
        heading_command=False,
        debug_vis=False,
        ranges=conga.CongaVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.25, 1.25),
            lin_vel_y=(-1.25, 1.25),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class CongaObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        conga_phase = ObsTerm(func=conga.conga_features)

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        conga_phase = ObsTerm(func=conga.conga_features)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CongaRewardsCfg(RewardsCfg):
    # Keep standard velocity tracking, actuator, joint-limit and stability costs.
    base_linear_velocity = RewTerm(func=conga.vertical_velocity_error, weight=-0.5)
    joint_pos = None
    feet_air_time = None
    air_time_variance = None
    conga_contact = RewTerm(func=conga.contact_tracking, weight=1.0)
    conga_kick = RewTerm(func=conga.kick_tracking, weight=1.0, params={"lateral_std": 0.035, "height_std": 0.025})
    conga_height = RewTerm(func=conga.height_tracking, weight=0.75, params={"std": 0.025})
    conga_cadence = RewTerm(func=conga.cadence_tracking, weight=2.0, params={"timing_std": 0.08})
    conga_clearance = RewTerm(func=conga.swing_clearance, weight=0.75, params={"std": 0.015})
    conga_marching = RewTerm(func=conga.marching_in_place, weight=0.5, params={"std": 0.015})


@configclass
class CongaEnvCfg(RobotEnvCfg):
    commands: CongaCommandsCfg = CongaCommandsCfg()
    observations: CongaObservationsCfg = CongaObservationsCfg()
    rewards: CongaRewardsCfg = CongaRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.visual_material = None
        self.scene.height_scanner = None  # analytic local ground is exact on the plane
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0)
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.commands.base_velocity.reference.validate()


@configclass
class CongaCadenceEnvCfg(CongaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.reference.kick_enabled = False
        self.commands.base_velocity.reference.kick_scale = 0.0
        self.commands.base_velocity.reference.bounce_scale = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)


@configclass
class CongaBounceEnvCfg(CongaCadenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.reference.bounce_scale = 1.0


@configclass
class CongaKickEnvCfg(CongaBounceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.reference.kick_enabled = True
        self.commands.base_velocity.reference.kick_scale = 0.25


@configclass
class CongaRobustEnvCfg(CongaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Restore the existing Go2 randomization distributions for the final stage.
        from .velocity_env_cfg import EventCfg

        events = EventCfg()
        self.events.physics_material = events.physics_material
        self.events.add_base_mass = events.add_base_mass
        self.events.push_robot = events.push_robot


@configclass
class CongaPlayEnvCfg(CongaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
