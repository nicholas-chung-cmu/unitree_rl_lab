"""CPG observation/action contracts; run inside the simulator's AppLauncher."""
from types import SimpleNamespace
import torch
from unitree_rl_lab.tasks.locomotion.cpg.core import GaitConfig,ik,nominal_feet
from unitree_rl_lab.tasks.locomotion.cpg.dance import DanceReference
from unitree_rl_lab.tasks.locomotion.cpg.dance_adapter import CPGDanceCommand
from unitree_rl_lab.tasks.locomotion.cpg.adapter import ResidualPositionAction,reward
from unitree_rl_lab.tasks.locomotion.cpg.symmetry import mirror_observation


def test_actual_balanced_preroll_actor_features_reflect():
    cfg=GaitConfig(swing_seconds=.2,balanced_phrases=True)
    command=object.__new__(CPGDanceCommand)
    command._debug_vis_handle=None
    command._env=SimpleNamespace(num_envs=2,device='cpu')
    command.cfg=SimpleNamespace(gait=cfg)
    command.dance=DanceReference(2,'cpu',cfg)
    command.time=command.dance.time.clone()
    command.target_time=command.dance.target_time.clone()
    command.q_ref=ik(nominal_feet(2,'cpu',cfg)).transpose(1,2).flatten(1)
    command.expected=torch.ones(2,4,dtype=torch.bool)
    command.pair=torch.zeros(2,4,dtype=torch.bool)
    command.velocity=torch.zeros(2,3)
    command.ready=torch.ones(2,dtype=torch.bool)
    command.recovery=torch.zeros(2,dtype=torch.bool)
    observation=torch.cat((torch.zeros(2,42),command.command),-1)
    assert observation.shape==(2,75)
    torch.testing.assert_close(mirror_observation(observation[:1]),observation[1:],atol=2e-6,rtol=1e-5)


def test_residual_bound_and_preparation_gate():
    cfg=GaitConfig(residual_scale=.12)
    command=SimpleNamespace(cfg=SimpleNamespace(gait=cfg),q_ref=torch.ones(2,12)*.2,
                            ready=torch.tensor([False,True]))
    action=object.__new__(ResidualPositionAction)
    action._debug_vis_handle=None
    action._raw_actions=torch.zeros(2,12)
    action._env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:command))
    action._asset=SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=torch.tensor([-.3,.3]).expand(2,12,2)))
    action.process_actions(torch.full((2,12),100.))
    torch.testing.assert_close(action._processed_actions[0],command.q_ref[0])
    torch.testing.assert_close(action._processed_actions[1],torch.full((12,),.3))
    assert (action._processed_actions-command.q_ref).abs().max()<=cfg.residual_scale


def test_height_cost_tracks_scheduled_bounce_and_preserves_flat_target():
    cfg=GaitConfig(bounce_amplitude=.0075)
    time=torch.tensor([cfg.beat/4,3*cfg.beat/4])
    position=torch.tensor([[0.,0.,.3075],[0.,0.,.2925]])
    command=SimpleNamespace(cfg=SimpleNamespace(gait=cfg),time=time,
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=position)),
        ready=torch.ones(2,dtype=torch.bool),sync=lambda:None)
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:command),
                        scene=SimpleNamespace(env_origins=torch.zeros(2,3)))
    torch.testing.assert_close(reward(env,'height'),torch.zeros(2),atol=1e-10,rtol=0)
    cfg.bounce_amplitude=0.
    torch.testing.assert_close(reward(env,'height'),torch.full((2,),(.0075/.05)**2),atol=1e-6,rtol=0)
