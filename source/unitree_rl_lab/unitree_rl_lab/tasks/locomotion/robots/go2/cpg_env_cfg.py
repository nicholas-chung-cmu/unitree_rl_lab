"""Independent CPG/reference + residual PPO curriculum."""
import math
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.cpg import adapter
from .velocity_env_cfg import RobotEnvCfg, ObservationsCfg, RewardsCfg, TerminationsCfg

@configclass
class CPGCommandsCfg:
    cpg=adapter.CPGCommandCfg()

@configclass
class CPGActionsCfg:
    JointPositionAction=adapter.ResidualPositionActionCfg(asset_name='robot',joint_names=['.*'],scale=.12,use_default_offset=False)

@configclass
class CPGObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        velocity_commands=None
        cpg_reference=ObsTerm(func=mdp.generated_commands,params={'command_name':'cpg'})
    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        velocity_commands=None
        cpg_reference=ObsTerm(func=mdp.generated_commands,params={'command_name':'cpg'})
    policy:PolicyCfg=PolicyCfg()
    critic:CriticCfg=CriticCfg()

def rew(name,weight):return RewTerm(func=adapter.reward,weight=weight,params={'component':name})

@configclass
class CPGRewardsCfg(RewardsCfg):
    track_lin_vel_xy=None
    track_ang_vel_z=None
    joint_pos=None
    feet_air_time=None
    air_time_variance=None
    base_linear_velocity=None
    contact=rew('contact',1.)
    clearance=rew('clearance',2.)
    velocity=rew('velocity',.5)
    yaw=rew('yaw',.25)
    body_height=rew('height',-.1)
    residual=rew('residual',-.05)
    timing=rew('timing',.25)

@configclass
class CPGTerminationsCfg(TerminationsCfg):
    preparation_failed=DoneTerm(func=adapter.cpg_failure)

@configclass
class CPGWalkEnvCfg(RobotEnvCfg):
    equivariant_actor:bool=False
    commands:CPGCommandsCfg=CPGCommandsCfg()
    observations:CPGObservationsCfg=CPGObservationsCfg()
    actions:CPGActionsCfg=CPGActionsCfg()
    rewards:CPGRewardsCfg=CPGRewardsCfg()
    terminations:CPGTerminationsCfg=CPGTerminationsCfg()
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type='plane'
        self.scene.terrain.terrain_generator=None
        self.scene.terrain.max_init_terrain_level=None
        self.scene.terrain.visual_material=None
        self.scene.height_scanner=None
        self.scene.sky_light.spawn=sim_utils.DomeLightCfg(intensity=750.)
        self.curriculum.terrain_levels=None
        self.curriculum.lin_vel_cmd_levels=None
        self.events.physics_material=None
        self.events.add_base_mass=None
        self.events.push_robot=None
        self.events.reset_robot_joints.params['velocity_range']=(0.,0.)
        self.events.reset_base.params['pose_range']={'x':(-.1,.1),'y':(-.1,.1),'yaw':(-.1,.1)}
        angle=math.acos(.30/(2*.213))
        self.scene.robot.init_state.pos=(0.,0.,.325)
        self.scene.robot.init_state.joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':angle,'.*_calf_joint':-2*angle}
        # Geometric IK is corrected for compliant stance load through position
        # targets. Preserve stock motor gains/torque limits rather than stiffness.
        self.scene.robot.actuators['GO2HV'].stiffness=25.
        self.scene.robot.actuators['GO2HV'].damping=.5
        self.commands.cpg.gait.load_compensation=1.
        self.commands.cpg.gait.swing_seconds=.20
        self.commands.cpg.gait.clearance=.06
        self.commands.cpg.gait.integrated_walk_reference=self.commands.cpg.gait.mode=='walk'
        self.commands.cpg.gait.linear_acceleration_limit=.075
        self.commands.cpg.gait.yaw_acceleration_limit=.15
        self.rewards.feet_slide.weight=-.5
        self.rewards.action_rate.weight=-.01
        self.rewards.joint_vel.weight=-.0002
        self.episode_length_s=20.


@configclass
class CPGTimingEnvCfg(CPGWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.cpg.gait.timing_tolerance=.06


@configclass
class CPGTravelEnvCfg(CPGTimingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.cpg.gait.max_speed=.20
        self.commands.cpg.gait.max_yaw=.30


from unitree_rl_lab.tasks.locomotion.cpg.dance_adapter import CPGDanceCommandCfg,dance_reward

@configclass
class CPGDanceCommandsCfg:
    cpg=CPGDanceCommandCfg()

@configclass
class CPGDanceRewardsCfg(CPGRewardsCfg):
    kick_tracking=RewTerm(func=dance_reward,weight=1.,params={'component':'kick_tracking'})
    endpoint_speed=RewTerm(func=dance_reward,weight=-.1,params={'component':'endpoint_speed'})
    hold_quality=RewTerm(func=dance_reward,weight=1.,params={'component':'hold_quality'})

@configclass
class CPGDanceEnvCfg(CPGTimingEnvCfg):
    commands:CPGDanceCommandsCfg=CPGDanceCommandsCfg()
    rewards:CPGDanceRewardsCfg=CPGDanceRewardsCfg()


@configclass
class CPGKickHoldEnvCfg(CPGDanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.cpg.gait.mode='kick_hold'


@configclass
class CPGBounceEnvCfg(CPGDanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.cpg.gait.bounce_amplitude=.0075
