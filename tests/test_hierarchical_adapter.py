"""Isaac API adapters against controlled states (run after AppLauncher)."""
from types import SimpleNamespace

import pytest
import torch
from unitree_rl_lab.tasks.locomotion.skills import adapter as a
from unitree_rl_lab.tasks.locomotion.skills.core import Skill, Status


@pytest.fixture
def rig():
    n = 3
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[0., 0., .32]]).repeat(n, 1),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3),
        root_ang_vel_b=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        default_joint_pos=torch.zeros(n, 12),
        body_pos_w=torch.tensor([[[.2, .17, .022], [.2, -.17, .022], [-.2, .17, .022], [-.2, -.17, .022]]]).repeat(n, 1, 1),
        body_lin_vel_w=torch.zeros(n, 4, 3),
    )
    find = lambda names, preserve_order: (list(range(4)), list(a.FOOT_NAMES))
    robot = SimpleNamespace(data=data, find_bodies=find)
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(n, 4, 3)), find_bodies=find)
    sensor.data.net_forces_w[..., 2] = 30
    class Scene(dict):
        env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, common_step_counter=0,
                          scene=Scene(robot=robot, contact_forces=sensor))
    cmd = a.SkillCommand(a.SkillCommandCfg(), env)
    env.command_manager = SimpleNamespace(get_term=lambda name: cmd)
    cmd.reset()
    for _ in range(60):
        env.common_step_counter += 1
        cmd.sync()
    return env, cmd, robot, sensor


def test_sync_only_once_and_immutable_terminal_snapshot(rig):
    env, cmd, _, _ = rig
    elapsed = cmd.core.elapsed.clone()
    cmd.sync()
    torch.testing.assert_close(cmd.core.elapsed, elapsed)
    cmd._update_command()
    assert (cmd.core.kind == Skill.STEP).all()
    for _ in range(85):
        env.common_step_counter += 1
        cmd.sync()
    assert (cmd.core.status == Status.TIMEOUT).all()
    snapshot = cmd.last_events["status"].clone()
    cmd.reset(torch.tensor([0]))
    torch.testing.assert_close(cmd.last_events["status"], snapshot)


def test_d_actual_terminal_state_dispatches_step_without_reset(rig):
    env, cmd, robot, sensor = rig
    cmd.cfg.stage = "D"
    cmd.started[:] = True
    ids = torch.arange(3)
    cmd.core.begin(ids, Skill.KICK, (0, 1, 0, 0), torch.zeros(3, 3), .25, cmd.state())
    robot.data.body_pos_w[:, 1, 1] -= .08
    robot.data.body_pos_w[:, 1, 2] += .05
    sensor.data.net_forces_w[:, 1] = 0
    for _ in range(13):
        env.common_step_counter += 1
        cmd.sync()
        if cmd.core.success_event.all():
            break
    assert (cmd.core.status == Status.SUCCESS).all()
    before = robot.data.body_pos_w.clone()
    assert (a.skill_reward(env, "success") == 1/env.step_dt).all()
    cmd._update_command()
    assert (cmd.core.kind == Skill.STEP).all()
    assert cmd.from_kick.all()
    assert cmd.core.entry_raised[:, 1].all()
    torch.testing.assert_close(robot.data.body_pos_w, before)
    assert cmd.core.selected.tolist() == [[False, True, True, False]]*3


def test_stabilize_three_foot_reward_preserves_airborne_foot(rig):
    env, cmd, robot, sensor = rig
    cmd.started[:] = True
    robot.data.body_pos_w[:, 0, 2] += .05
    sensor.data.net_forces_w[:, 0] = 0
    env.common_step_counter += 1
    cmd.sync()
    cmd.core.begin(torch.arange(3), Skill.STABILIZE, cmd.core.confirmed, torch.zeros(3, 3), .2, cmd.state())
    torch.testing.assert_close(a.skill_reward(env, "contact"), torch.ones(3))
    robot.data.body_pos_w[:, 0, 2] -= .05
    assert (a.skill_reward(env, "placement") < 1).all()


def test_goal_schema_same_across_stages_and_export_config_is_independent():
    from unitree_rl_lab.tasks.locomotion.robots.go2.hierarchical_env_cfg import (
        HierarchicalStepEnvCfg, HierarchicalRaisedStepEnvCfg, HierarchicalKickEnvCfg,
        HierarchicalTransitionEnvCfg, HierarchicalTimingEnvCfg, HierarchicalDanceEnvCfg)
    configs = [cls() for cls in (HierarchicalStepEnvCfg, HierarchicalRaisedStepEnvCfg, HierarchicalKickEnvCfg,
                                HierarchicalTransitionEnvCfg, HierarchicalTimingEnvCfg, HierarchicalDanceEnvCfg)]
    assert [cfg.commands.skill.stage for cfg in configs] == list("ABCDEF")
    configs[0].commands.skill.skill.bpm = 80
    assert all(cfg.commands.skill.skill.bpm == 90 for cfg in configs[1:])
    assert all(cfg.rewards.track_ang_vel_z is None for cfg in configs)


def test_operator_updates_do_not_restart_skill_or_clock(rig):
    _, cmd, _, _ = rig
    cmd._update_command()
    old = cmd.core.goal.clone()
    cmd.planner.tick[:] = 45
    cmd.set_velocity(torch.tensor([[1., 1., 1.]]).repeat(3, 1))
    torch.testing.assert_close(cmd.core.goal, old)
    assert (cmd.planner.tick == 45).all()
    assert (cmd.planner.command[:, :2].norm(dim=-1) <= .500001).all()


def test_no_velocity_tracking_and_integrated_terminal_reward(rig):
    env, cmd, _, _ = rig
    cmd.started[:] = True
    cmd.core.success_event[:] = 1
    cmd.core.kind[:] = Skill.STEP
    reward = a.skill_reward(env, "success")
    torch.testing.assert_close(reward*env.step_dt, torch.ones(3))
    cmd.core.failure_event[:] = 1
    torch.testing.assert_close(a.skill_reward(env, "failure")*env.step_dt, torch.ones(3))


def test_support_wrench_balances_three_feet_and_mirrors():
    from unitree_rl_lab.tasks.locomotion.skills.preparation import support_forces
    feet = torch.tensor([[[.2, .17, 0.], [.2, -.17, 0.], [-.2, .17, 0.], [-.2, -.17, 0.]]])
    com = feet[:, 1:].mean(1)
    com[:, 2] = .28
    force = torch.tensor([[0., 0., 140.]])
    torque = torch.zeros(1, 3)
    weights = torch.tensor([[0., 1., 1., 1.]])
    f = support_forces(feet, com, force, torque, weights)
    torch.testing.assert_close(f[:, 0], torch.zeros(1, 3))
    torch.testing.assert_close(f.sum(1), force, atol=.002, rtol=0)
    torch.testing.assert_close(torch.cross(feet-com[:, None], f, dim=-1).sum(1), torque, atol=.002, rtol=0)
    reflected_feet, reflected_com = feet.clone(), com.clone()
    reflected_feet[..., 1] *= -1
    reflected_com[..., 1] *= -1
    mirrored = support_forces(reflected_feet, reflected_com, force, torque, weights)
    expected = f.clone()
    expected[..., 1] *= -1
    torch.testing.assert_close(mirrored, expected)


def test_support_wrench_obeys_unilateral_friction_bounds():
    from unitree_rl_lab.tasks.locomotion.skills.preparation import support_forces
    feet = torch.tensor([[[.2, .17, 0.], [.2, -.17, 0.], [-.2, .17, 0.], [-.2, -.17, 0.]]])
    f = support_forces(feet, torch.tensor([[0., 0., .28]]),
                       torch.tensor([[1000., -1000., 140.]]), torch.tensor([[30., 40., 50.]]),
                       torch.tensor([[0., 1., 1., 1.]]))
    assert torch.isfinite(f).all()
    assert (f[..., 2] >= 0).all() and (f[..., 2] <= 150).all()
    assert (f[..., :2].norm(dim=-1) <= .5*f[..., 2]+1e-5).all()
    assert (f[:, 0] == 0).all()


def test_preparation_requires_completed_lift_and_continuous_stationary_hold(rig):
    env, cmd, robot, sensor = rig
    cmd.cfg.stage = "B"
    cmd.started[:] = True
    cmd.preparing[:] = True
    cmd.prepare_leg[:] = 0
    cmd.prepare_height[:] = .05
    cmd.prepare_lateral[:] = .08
    cmd.prepare_lift_time[:] = 2.
    cmd.prepare_elapsed[:] = 2.
    robot.data.body_pos_w[:, 0, 1] += .08
    robot.data.body_pos_w[:, 0, 2] += .05
    sensor.data.net_forces_w[:, 0] = 0
    def step():
        env.common_step_counter += 1
        cmd.sync()
        cmd._update_command()
    for _ in range(50):
        step()
    assert cmd.preparing.all() and (cmd.prepare_confirm == 0).all()
    cmd.prepare_elapsed[:] = 4.
    for _ in range(30):
        step()
    assert cmd.preparing.all()
    # Translating, yawing, and sliding-support states must each reset their hold.
    robot.data.root_lin_vel_w[0, 0] = .2
    robot.data.root_ang_vel_w[1, 2] = .3
    robot.data.body_lin_vel_w[2, 3, 0] = .1
    step()
    assert (cmd.prepare_confirm == 0).all()
    robot.data.root_lin_vel_w.zero_()
    robot.data.root_ang_vel_w.zero_()
    robot.data.body_lin_vel_w.zero_()
    for _ in range(51):
        step()
    assert not cmd.preparing.any()
    assert (cmd.core.kind == Skill.STEP).all()
    assert cmd.core.entry_raised[:, 0].all()
    # A masked episode reset must not erase another environment's controller state.
    old = cmd.prepare_lift_time.clone()
    cmd.reset(torch.tensor([0]))
    assert cmd.prepare_lift_time[0] == -1
    torch.testing.assert_close(cmd.prepare_lift_time[1:], old[1:])


def test_preparation_waits_for_unloading_and_other_three_contacts(rig):
    env, cmd, _, sensor = rig
    cmd.cfg.stage = "B"
    cmd.started[:] = True
    cmd.preparing[:] = True
    cmd.prepare_leg[:] = 0
    cmd.prepare_elapsed[:] = 2.
    def step():
        env.common_step_counter += 1
        cmd.sync()
        cmd._update_command()
    for _ in range(10):
        step()
    assert (cmd.prepare_lift_time == -1).all()
    sensor.data.net_forces_w[:, 0] = 0
    # One remaining support lost: unloading alone must not start the lift.
    sensor.data.net_forces_w[1, 3] = 0
    for _ in range(4):
        step()
    assert (cmd.prepare_lift_time == -1).all()
    for _ in range(2):
        step()
    assert cmd.prepare_lift_time[0] >= 0 and cmd.prepare_lift_time[2] >= 0
    assert cmd.prepare_lift_time[1] == -1


def test_step_contact_follows_height_reference_through_late_swing(rig):
    _, cmd, robot, sensor = rig
    env = cmd._env
    cmd._update_command()
    cmd.core.selected[:] = torch.tensor([True, False, False, True])
    # At 70 and 80 percent the reference is still airborne: full credit
    # requires the selected pair airborne, with the other two supporting.
    for progress in (.5, .7, .8):
        cmd.core.elapsed[:] = cmd.core.duration*progress
        cmd.core.contact[:] = torch.tensor([False, True, True, False])
        torch.testing.assert_close(a.skill_reward(env, 'contact'), torch.ones(3))
        cmd.core.contact[:] = True
        torch.testing.assert_close(a.skill_reward(env, 'contact'), torch.full((3,), .5))
    # At the endpoint all-four support earns full credit, not airborne feet.
    cmd.core.elapsed[:] = cmd.core.duration
    cmd.core.contact[:] = True
    torch.testing.assert_close(a.skill_reward(env, 'contact'), torch.ones(3))
    cmd.core.contact[:] = torch.tensor([False, True, True, False])
    torch.testing.assert_close(a.skill_reward(env, 'contact'), torch.full((3,), .5))
    # Near ground the contact preference is soft rather than contradictory.
    cmd.core.elapsed[:] = cmd.core.duration*.1
    value = a.skill_reward(env, 'contact')
    assert ((value > .5) & (value < 1.)).all()


def test_settling_rejects_drop_and_restarts_hold_on_motion(rig):
    env, cmd, robot, _ = rig
    cmd.reset()
    for _ in range(60):
        robot.data.root_lin_vel_w[:, 2] = -.3
        env.common_step_counter += 1
        cmd.sync()
        cmd._update_command()
    assert not cmd.started.any()
    assert (cmd.settle_confirm == 0).all()
    robot.data.root_lin_vel_w.zero_()
    cmd.core.status[:] = Status.TIMEOUT  # old short STABILIZE timeout must not block admission
    for _ in range(20):
        env.common_step_counter += 1
        cmd.sync()
        cmd._update_command()
    assert not cmd.started.any()
    robot.data.body_lin_vel_w[0, 0, 0] = .1
    env.common_step_counter += 1
    cmd.sync()
    cmd._update_command()
    assert cmd.settle_confirm[0] == 0
    robot.data.body_lin_vel_w.zero_()
    for _ in range(26):
        env.common_step_counter += 1
        cmd.sync()
        cmd._update_command()
    assert cmd.started.all()
    assert (cmd.core.kind == Skill.STEP).all()


def test_lift_shaping_prefers_clearance_to_grounded_slide(rig):
    env, cmd, robot, _ = rig
    cmd._update_command()
    cmd.core.elapsed[:] = .5*cmd.core.duration
    reference,_ = cmd.core.reference(cmd.state())
    selected=cmd.core.selected
    # Perfect horizontal progress with grounded feet no longer earns full credit.
    robot.data.body_pos_w[:,:,:2] = reference[:,:,:2]
    robot.data.body_pos_w[:,:,2] = .022
    grounded=a.skill_reward(env,'placement')
    torch.testing.assert_close(grounded,torch.zeros(3))
    torch.testing.assert_close(a.skill_reward(env,'clearance'),torch.zeros(3))
    robot.data.body_pos_w[:,:,2] += selected*.02
    half=a.skill_reward(env,'placement')
    torch.testing.assert_close(a.skill_reward(env,'clearance'),torch.full((3,),.5))
    robot.data.body_pos_w[:,:,2] += selected*.02
    full=a.skill_reward(env,'placement')
    assert (full>half).all() and (half>grounded).all()
    torch.testing.assert_close(full,torch.ones(3))
    torch.testing.assert_close(a.skill_reward(env,'clearance'),torch.ones(3))
    robot.data.body_pos_w[:,:,2] += selected*.04
    torch.testing.assert_close(a.skill_reward(env,'clearance'),torch.zeros(3),atol=1e-6,rtol=0)
    # Staying airborne at the landing endpoint is penalized, not rewarded.
    cmd.core.elapsed[:] = cmd.core.duration
    assert (a.skill_reward(env,'clearance')<0).all()


def test_body_height_cost_is_mild_symmetric_and_masked_during_preparation(rig):
    env,cmd,robot,_=rig
    cmd._update_command()
    robot.data.root_pos_w[:,2]=torch.tensor([.32,.27,.37])
    torch.testing.assert_close(a.skill_reward(env,'body_height'),torch.tensor([0.,.64,.64]),atol=1e-6,rtol=0)
    # Runtime weight -.1 gives -.064/s at 5 cm deviation; reward-manager dt
    # scaling remains unchanged. No new posture cost during scripted preparation.
    cmd.preparing[:]=True
    torch.testing.assert_close(a.skill_reward(env,'body_height'),torch.zeros(3))
    torch.testing.assert_close(a.skill_reward(env,'clearance'),torch.zeros(3))


def test_placement_gate_preserves_correct_near_ground_tracking():
    from unitree_rl_lab.tasks.locomotion.skills.shaping import swing_placement_gate
    target=torch.tensor([0.,.001,.004,.008,.04])
    torch.testing.assert_close(swing_placement_gate(target,target),torch.ones_like(target))
    torch.testing.assert_close(swing_placement_gate(torch.zeros_like(target),target),
                               torch.tensor([1.,.875,.5,0.,0.]))
