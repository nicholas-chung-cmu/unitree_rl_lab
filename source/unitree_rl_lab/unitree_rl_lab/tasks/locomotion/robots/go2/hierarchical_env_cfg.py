"""Independent hierarchical skill curriculum. No legacy conga task is modified."""
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.skills import adapter
from .velocity_env_cfg import RobotEnvCfg, ObservationsCfg, RewardsCfg, TerminationsCfg


@configclass
class SkillCommandsCfg:
    skill = adapter.SkillCommandCfg()


@configclass
class SkillObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        velocity_commands = None
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        skill_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "skill"})

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        velocity_commands = None
        skill_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "skill"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


def reward(component, weight):
    return RewTerm(func=adapter.skill_reward, weight=weight, params={"component": component})


@configclass
class SkillRewardsCfg(RewardsCfg):
    track_lin_vel_xy = None
    track_ang_vel_z = None
    joint_pos = None
    feet_air_time = None
    air_time_variance = None
    base_linear_velocity = None
    placement = reward("placement", 1.5)
    clearance = reward("clearance", 1.0)
    body_height = reward("body_height", -.1)
    displacement = reward("displacement", 1.0)
    heading = reward("heading", .5)
    support = reward("support", .3)
    skill_contact = reward("contact", .5)
    success = reward("success", 2.0)  # integrated event costs/rewards (divided by dt in adapter)
    failure = reward("failure", -2.0)
    transition = reward("transition", 1.0)
    premature = reward("premature", -1.0)
    kick_touchdown = reward("kick_touchdown", -1.0)
    endpoint_speed = reward("endpoint_speed", -.1)
    touchdown_timing = reward("touchdown_timing", 0.)
    bounce = reward("bounce", 0.0)
    # Explicit fall cost supplements inherited orientation/collision costs.
    fall = RewTerm(func=adapter.physical_fall, weight=-5.0)


@configclass
class SkillTerminationsCfg(TerminationsCfg):
    skill_failure = DoneTerm(func=adapter.skill_failure)
    isolated_complete = DoneTerm(func=adapter.isolated_complete, time_out=True)


@configclass
class HierarchicalStepEnvCfg(RobotEnvCfg):
    commands: SkillCommandsCfg = SkillCommandsCfg()
    observations: SkillObservationsCfg = SkillObservationsCfg()
    rewards: SkillRewardsCfg = SkillRewardsCfg()
    terminations: SkillTerminationsCfg = SkillTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.skill.__post_init__()
        # Stronger grounded-foot sliding cost in hierarchical tasks only.
        self.rewards.feet_slide.weight = -4.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.visual_material = None
        self.scene.height_scanner = None
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0)
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["velocity_range"] = (-.1, .1)
        self.episode_length_s = 20.


@configclass
class HierarchicalRaisedStepEnvCfg(HierarchicalStepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.stage = "B"


@configclass
class HierarchicalKickEnvCfg(HierarchicalStepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.stage = "C"
        self.commands.skill.travel_range = 0.
        self.commands.skill.yaw_range = 0.


@configclass
class HierarchicalTransitionEnvCfg(HierarchicalStepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.stage = "D"


@configclass
class HierarchicalTimingEnvCfg(HierarchicalStepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.stage = "E"
        self.commands.skill.step_duration_range = (.60, .73)
        self.commands.skill.skill.touchdown_tolerance = .06
        self.rewards.touchdown_timing.weight = -.5


@configclass
class HierarchicalDanceEnvCfg(HierarchicalTimingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.stage = "F"
        self.commands.skill.travel_range = .5
        self.commands.skill.yaw_range = .8
        # Explicit opt-in after timing/transition qualification; amplitude starts zero.
        self.rewards.bounce.weight = -.5
