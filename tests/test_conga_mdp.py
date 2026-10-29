"""Controlled-state adapter tests; run via scripts/conga_check.py after AppLauncher."""

import torch
from types import SimpleNamespace

import pytest

from unitree_rl_lab.tasks.locomotion.mdp import conga
from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import FOOT_NAMES


class Scene(dict):
    env_origins = torch.zeros(3, 3)


@pytest.fixture
def fixture():
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.32]]).repeat(3, 1),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        root_lin_vel_w=torch.zeros(3, 3),
        root_lin_vel_b=torch.zeros(3, 3),
        root_ang_vel_b=torch.zeros(3, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
        body_pos_w=torch.zeros(3, 4, 3),
        body_lin_vel_w=torch.zeros(3, 4, 3),
    )

    def find(names, preserve_order):
        return list(range(4)), list(FOOT_NAMES)

    robot = SimpleNamespace(data=data, find_bodies=find)
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(3, 4, 3)), find_bodies=find)
    sensor.first = torch.zeros(3, 4, dtype=torch.bool)
    sensor.compute_first_contact = lambda dt: sensor.first
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.02, episode_length_buf=torch.tensor([10, 69, 169]))
    env.scene = Scene(robot=robot, contact_forces=sensor)
    cfg = conga.CongaVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        debug_vis=False,
        ranges=conga.CongaVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.25, 1.25), lin_vel_y=(-1.25, 1.25), ang_vel_z=(-1.0, 1.0)
        ),
    )
    term = conga.CongaVelocityCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    return env, term, robot, sensor


def set_perfect_state(env, term, robot, sensor):
    ref = term.reference
    robot.data.root_pos_w[:, 2] = ref.body_height
    robot.data.root_lin_vel_w[:, 2] = ref.body_velocity_z
    robot.data.body_pos_w[:, :, 2] = ref.foot_height + term.cfg.reference.foot_radius
    sensor.data.net_forces_w[:, :, 2] = (ref.contact > 0.5) * 20.0
    rows = torch.arange(3)
    robot.data.body_pos_w[rows, ref.kick_leg, 1] = ref.kick_y


def test_commands_phase_and_reset(fixture):
    env, term, _, _ = fixture
    term.vel_command_b[:] = torch.tensor([[1.25, 1.25, 1.0], [0.5, 0.0, -0.4], [0.0, 0.0, 1.0]])
    assert (term.raw_command[:, :2].norm(dim=-1) <= 1.25 + 1e-6).all()
    assert (term.command[1:] == 0).all()  # paused XY and yaw during either kick
    assert conga.conga_features(env).shape == (3, 10)
    before = term.reference.phase.clone()
    term._resample_command(torch.tensor([0]))
    torch.testing.assert_close(before, term.reference.phase)
    env.episode_length_buf[0] = 0
    assert term.reference.phase[0] == 0
    torch.testing.assert_close(before[1:], term.reference.phase[1:])
    term.reset(slice(None))


def test_reference_reward_optima_and_no_fore_aft_constraint(fixture):
    env, term, robot, sensor = fixture
    set_perfect_state(env, term, robot, sensor)
    torch.testing.assert_close(conga.height_tracking(env, 0.025), torch.ones(3))
    torch.testing.assert_close(conga.vertical_velocity_error(env), torch.zeros(3))
    correct = conga.kick_tracking(env, 0.035, 0.025)
    torch.testing.assert_close(correct, term.reference.kick_weight)
    robot.data.body_pos_w[:, :, 0] = 10.0
    torch.testing.assert_close(correct, conga.kick_tracking(env, 0.035, 0.025))
    robot.data.body_pos_w[:, :, 1] += 0.1
    assert (conga.kick_tracking(env, 0.035, 0.025)[1:] < correct[1:]).all()
    robot.data.root_pos_w[:, 2] += 0.1
    assert (conga.height_tracking(env, 0.025) < 0.01).all()


def test_marching_beats_stationary_feet_and_drift(fixture):
    env, term, robot, sensor = fixture
    set_perfect_state(env, term, robot, sensor)
    marching = conga.marching_in_place(env, 0.015)
    contact = conga.contact_tracking(env)
    assert (marching > 0.9).all()
    robot.data.body_pos_w[:, :, 2] = term.cfg.reference.foot_radius
    sensor.data.net_forces_w[:, :, 2] = 20.0
    assert (conga.marching_in_place(env, 0.015) < marching).all()
    assert (conga.contact_tracking(env) < contact).all()
    set_perfect_state(env, term, robot, sensor)
    robot.data.root_lin_vel_b[:, 0] = 0.5
    assert (conga.marching_in_place(env, 0.015) < 0.01).all()


def test_actual_touchdown_timing_and_missing_events(fixture):
    env, term, _, sensor = fixture
    env.episode_length_buf[:] = torch.tensor([19, 10, 19])  # .38 exact touchdown; .20 wrong
    sensor.first[:, 0] = True
    reward = conga.cadence_tracking(env, 0.08)
    assert reward[0] > 0 and reward[1] < 0
    sensor.first[:] = False
    assert (conga.cadence_tracking(env, 0.08) == 0).all()


def test_metrics_finite_and_partial_reset(fixture):
    env, term, robot, sensor = fixture
    set_perfect_state(env, term, robot, sensor)
    term._update_metrics()
    sensor.first[:, 0] = True
    term._update_metrics()
    assert all(torch.isfinite(value).all() for value in term.metrics.values())
    samples = term._samples.clone()
    result = term.reset(torch.tensor([1]))
    assert "touchdown_error_s" in result
    assert term._samples[1] == 0
    torch.testing.assert_close(term._samples[[0, 2]], samples[[0, 2]])


def test_recovery_neither_forces_stance_nor_rewards_flight(fixture):
    env, term, robot, sensor = fixture
    env.episode_length_buf[:] = 85  # neutral recovery at 1.70 seconds
    set_perfect_state(env, term, robot, sensor)
    assert (conga.swing_clearance(env, 0.015) == 0).all()
    free_contact_score = conga.contact_tracking(env)
    sensor.data.net_forces_w[:, :, 2] = 20.0
    torch.testing.assert_close(free_contact_score, conga.contact_tracking(env))
    sensor.first[:] = True
    assert (conga.cadence_tracking(env, 0.08) == 0).all()


def test_rotated_body_and_local_ground(fixture):
    env, term, robot, sensor = fixture
    set_perfect_state(env, term, robot, sensor)
    # Rotate both the body and feet +90 degrees in yaw: lateral target is invariant.
    robot.data.root_quat_w[:] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    old_y = robot.data.body_pos_w[:, :, 1].clone()
    robot.data.body_pos_w[:, :, 0] = -old_y
    robot.data.body_pos_w[:, :, 1] = 0.0
    torch.testing.assert_close(conga.kick_tracking(env, 0.035, 0.025), term.reference.kick_weight)

    env.scene.env_origins = torch.tensor([[0.0, 0.0, 2.0]]).repeat(3, 1)
    robot.data.body_pos_w[:, :, 2] += 2.0
    robot.data.root_pos_w[:, 2] += 2.0
    torch.testing.assert_close(conga.height_tracking(env, 0.025), torch.ones(3))
    torch.testing.assert_close(conga.kick_tracking(env, 0.035, 0.025), term.reference.kick_weight)


def test_walk_rewards_reject_planted_feet_and_reward_real_swings(fixture):
    from unitree_rl_lab.tasks.locomotion.mdp import conga_walk_rewards as walk

    env, term, robot, sensor = fixture
    term.cfg.reference.kick_enabled = False
    term.cfg.reference.bounce_scale = 0.0
    env.episode_length_buf[:] = 11  # swing midpoint at 0.22 s
    set_perfect_state(env, term, robot, sensor)
    assert (walk.swing_quality(env) > 0.99).all()
    assert (walk.missed_swing(env) == 0).all()
    assert (walk.contact_error(env) == 0).all()
    robot.data.body_pos_w[:, :, 2] = term.cfg.reference.foot_radius
    sensor.data.net_forces_w[:, :, 2] = 20
    assert (walk.swing_quality(env) == 0).all()
    assert (walk.marching(env) == 0).all()
    assert (walk.missed_swing(env) > 0.99).all()
    assert (walk.contact_error(env) == 0.5).all()


def test_walk_event_scaling_chatter_and_reset(fixture):
    from unitree_rl_lab.tasks.locomotion.mdp import conga_walk_rewards as walk

    env, term, _, sensor = fixture
    term.cfg.reference.kick_enabled = False
    sensor.data.last_air_time = torch.full((3, 4), 0.3)
    sensor.first[:, 0] = True
    env.episode_length_buf[:] = 19
    integrated = walk.touchdown_events(env) * env.step_dt
    assert (integrated > 0).all()
    env.step_dt = 0.01
    env.episode_length_buf[:] = 38
    torch.testing.assert_close(integrated, walk.touchdown_events(env) * env.step_dt)
    sensor.data.last_air_time[:] = 0.01
    assert (walk.touchdown_events(env) == 0).all()
    env.episode_length_buf[:] = 20  # off-phase chatter remains penalized
    assert (walk.touchdown_events(env) < 0).all()
    env.episode_length_buf[:] = 0
    assert (walk.touchdown_events(env) == 0).all()


def test_walk_changes_do_not_mutate_final_conga_config():
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_env_cfg import CongaEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_walk_validation_env_cfg import CongaWalkValidationEnvCfg

    walk = CongaWalkValidationEnvCfg()
    final = CongaEnvCfg()
    assert walk.rewards.track_lin_vel_xy.params["std"] == 0.2
    assert final.rewards.track_lin_vel_xy.params["std"] == pytest.approx(0.5)
    assert final.rewards.conga_contact.weight == 1.0
    assert final.commands.base_velocity.reference.kick_enabled


def test_basic_stage_matches_stock_except_added_observations():
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_basic_env_cfg import CongaBasicEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_env_cfg import CongaCadenceEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    from unitree_rl_lab.tasks.locomotion.agents.conga_ppo_cfg import CongaBasicPPORunnerCfg
    from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    from isaaclab_tasks.utils.hydra import register_task_to_hydra

    basic, stock, cadence = CongaBasicEnvCfg(), RobotEnvCfg(), CongaCadenceEnvCfg()
    basic.validate()
    b, s = basic.to_dict(), stock.to_dict()
    for group in ("policy", "critic"):
        b["observations"][group].pop("conga_phase")
        assert list(vars(basic.observations.__dict__[group])) == list(vars(cadence.observations.__dict__[group]))
    b["commands"]["base_velocity"].pop("reference")
    assert b == s  # includes all rewards, terrain, events, terminations and curriculum
    runner, stock_runner = CongaBasicPPORunnerCfg().to_dict(), BasePPORunnerCfg().to_dict()
    for key in ("experiment_name", "max_iterations"):
        runner.pop(key)
        stock_runner.pop(key)
    assert runner == stock_runner
    cfg, _ = register_task_to_hydra("Unitree-Go2-Conga-Basic", "rsl_rl_cfg_entry_point")
    cfg.validate()
    assert not basic.commands.base_velocity.reference.kick_enabled
    assert basic.commands.base_velocity.reference.bounce_scale == 0
    assert cadence.rewards.joint_pos is None  # later-stage specification untouched


def test_basic_phase_matches_later_stage_and_resets_independently(fixture):
    env, term, _, _ = fixture
    term.cfg.reference.kick_enabled = False
    term.cfg.reference.kick_scale = 0.0
    term.cfg.reference.bounce_scale = 0.0
    env.cfg = SimpleNamespace(commands=SimpleNamespace(base_velocity=term.cfg))
    env.command_manager.get_command = lambda name: term.command
    term.vel_command_b[:] = torch.tensor([0.3, 0.1, 0.5])
    torch.testing.assert_close(conga.basic_conga_features(env), conga.conga_features(env))
    before = conga.basic_conga_features(env).clone()
    term._resample_command(torch.tensor([0]))
    torch.testing.assert_close(conga.basic_conga_features(env)[:, :5], before[:, :5])
    env.episode_length_buf[0] = 0
    after = conga.basic_conga_features(env)
    torch.testing.assert_close(after[0, :5], torch.tensor([0., 1., 0., 1., 1.]))
    torch.testing.assert_close(after[1:], before[1:])


def test_cadence_start_speed_cap_and_isolation(fixture):
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_cadence_start_env_cfg import CongaCadenceStartEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_env_cfg import CongaEnvCfg, CongaBounceEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    from isaaclab_tasks.utils.hydra import register_task_to_hydra

    cfg, runner = register_task_to_hydra("Unitree-Go2-Conga-Cadence-Start", "rsl_rl_cfg_entry_point")
    cfg.validate()
    assert runner.num_steps_per_env == 24
    assert cfg.rewards.conga_contact.weight == 0.25
    assert cfg.rewards.conga_clearance.weight == 0.5
    rewards = cfg.rewards.to_dict()
    rewards.pop("conga_contact")
    rewards.pop("conga_clearance")
    assert rewards == RobotEnvCfg().rewards.to_dict()
    assert cfg.events.to_dict() == RobotEnvCfg().events.to_dict()
    assert cfg.curriculum.lin_vel_cmd_levels is None
    assert CongaEnvCfg().commands.base_velocity.reference.max_travel_speed == 1.25
    assert CongaBounceEnvCfg().rewards.conga_contact.weight == 1.0

    env, term, _, _ = fixture
    term.cfg.reference = cfg.commands.base_velocity.reference
    term.vel_command_b[:] = torch.tensor([[0.5, 0.25, 1.0], [-0.5, -0.25, -1.0], [0., 0., 0.]])
    before = term.reference.phase.clone()
    assert (term.command[:, :2].norm(dim=-1) <= 0.5 + 1e-6).all()
    torch.testing.assert_close(term.command[:, 2], torch.tensor([1., -1., 0.]))
    torch.testing.assert_close(term.reference.phase, before)
    torch.testing.assert_close(term.reference.travel, torch.ones(3))
    assert term.reference.foot_height[2].max() > 0  # zero command still has swing targets


def test_cadence_fade_steps_and_resume(monkeypatch):
    from unitree_rl_lab.utils.cadence_training import apply_fade, CadenceRunner, OnPolicyRunner

    terms = {name: SimpleNamespace(weight=w) for name, w in
             {"joint_pos": -0.7, "feet_air_time": 0.1, "air_time_variance": -1.0,
              "conga_contact": 0.25, "conga_clearance": 0.5, "joint_torques": -2e-4}.items()}
    manager = SimpleNamespace(get_term_cfg=lambda n: terms[n], set_term_cfg=lambda n, c: terms.update({n: c}))
    env = SimpleNamespace(common_step_counter=0, reward_manager=manager)
    for iterations, expected in [(0, 1), (50, .75), (100, .5), (200, 0), (1000, 0)]:
        env.common_step_counter = iterations * 24
        assert apply_fade(env) == expected
        assert terms["joint_pos"].weight == pytest.approx(-.7 * expected)
        assert terms["feet_air_time"].weight == pytest.approx(.1 * expected)
        assert terms["air_time_variance"].weight == -expected
        assert terms["conga_contact"].weight == .25
        assert terms["conga_clearance"].weight == .5
        assert terms["joint_torques"].weight == -2e-4
    runner = CadenceRunner.__new__(CadenceRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    runner.current_learning_iteration = 999
    monkeypatch.setattr(OnPolicyRunner, "load", lambda *a: None)
    runner.load("basic")
    assert env.common_step_counter == 0 and runner.current_learning_iteration == 0
    runner.current_learning_iteration = 49
    monkeypatch.setattr(OnPolicyRunner, "load", lambda *a: {"cadence_stage": {
        "control_steps": 1200, "fade_steps": 4800, "rollout_steps": 24}})
    runner.load("cadence")
    assert env.common_step_counter == 1200 and runner.current_learning_iteration == 50
    assert terms["joint_pos"].weight == pytest.approx(-.525)


def test_balanced_march_resets_and_rewards(fixture):
    from unitree_rl_lab.tasks.locomotion.mdp.balanced_phase import BalancedMarchCommand
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_march_env_cfg import CongaMarchEnvCfg
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    cfg,_=register_task_to_hydra('Unitree-Go2-Conga-March-Balanced','rsl_rl_cfg_entry_point')
    cfg.validate()
    assert cfg.rewards.to_dict()==CongaMarchEnvCfg().rewards.to_dict()
    env,_,_,_=fixture
    term=BalancedMarchCommand(cfg.commands.base_velocity,env)
    assert abs(int((term.phase_offset==0).sum())-int((term.phase_offset==.5).sum()))<=1
    before=term.phase_offset.clone();ages=env.episode_length_buf.clone()
    term.reset(torch.tensor([0]))
    assert term.phase_offset[0]==.5-before[0]
    torch.testing.assert_close(term.phase_offset[1:],before[1:])
    torch.testing.assert_close(env.episode_length_buf,ages)
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.phase_offset,before)
    phase=term.reference.phase.clone()
    term._resample_command(torch.arange(env.num_envs))
    torch.testing.assert_close(term.reference.phase,phase)
    env.episode_length_buf[:]=11  # .22 s: center of first scheduled swing
    target=term.reference.contact
    for i in range(env.num_envs):
        expected=torch.tensor([0.,1.,1.,0.]) if term.phase_offset[i]==0 else torch.tensor([1.,0.,0.,1.])
        torch.testing.assert_close(target[i],expected)
    assert torch.all(term.command==0)


def test_march_resume_preserves_zero_commands(monkeypatch):
    from unitree_rl_lab.utils.march_training import MarchRunner,OnPolicyRunner
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_march_env_cfg import CongaMarchEnvCfg
    cfg=CongaMarchEnvCfg()
    env=SimpleNamespace(common_step_counter=0,cfg=cfg)
    runner=MarchRunner.__new__(MarchRunner)
    runner.env=SimpleNamespace(unwrapped=env)
    runner.current_learning_iteration=49
    monkeypatch.setattr(OnPolicyRunner,'load',lambda *a:{'march_stage':{
        'control_steps':1200,'schedule_version':1,'rollout_steps':24}})
    runner.load('march')
    assert runner.current_learning_iteration==50 and env.common_step_counter==1200
    assert cfg.commands.base_velocity.ranges.lin_vel_x==(0.,0.)
    assert cfg.rewards.cadence_phase.weight==1. and cfg.rewards.cadence_pair.weight==.25
    monkeypatch.setattr(OnPolicyRunner,'load',lambda *a:{'early_stage':{}})
    with pytest.raises(ValueError):runner.load('wrong_stage')


def test_march_first_config_and_dense_feedback(fixture):
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    from unitree_rl_lab.tasks.locomotion.mdp import march_rewards as m
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_pause_env_cfg import CongaPauseEnvCfg
    cfg,runner=register_task_to_hydra('Unitree-Go2-Conga-March','rsl_rl_cfg_entry_point')
    cfg.validate()
    assert runner.num_steps_per_env==24
    assert cfg.commands.base_velocity.ranges.lin_vel_x==(0.,0.)
    assert cfg.commands.base_velocity.ranges.lin_vel_y==(0.,0.)
    assert cfg.commands.base_velocity.ranges.ang_vel_z==(0.,0.)
    assert cfg.curriculum.lin_vel_cmd_levels is None and cfg.curriculum.terrain_levels is None
    assert cfg.rewards.cadence_phase.weight==1. and cfg.rewards.cadence_pair.weight==.25
    assert cfg.rewards.march_clearance.weight==.25 and cfg.rewards.march_missed_swing.weight==-.5
    assert cfg.rewards.joint_pos.weight==cfg.rewards.feet_air_time.weight==cfg.rewards.air_time_variance.weight==0
    assert CongaPauseEnvCfg().rewards.cadence_phase.weight==.25
    target=torch.tensor([[0.,1.,1.,0.],[.5,.5,.5,.5]])
    height=torch.tensor([[.03,0,0,.03],[0,0,0,0.]])
    planted=m.raw_terms(target,height,torch.ones_like(target),torch.zeros_like(height))
    assert planted[0,1]==1 and planted[1,1]==0
    one=target.clone();one[0,3]=1
    lifted=height.clone();lifted[0,3]=0
    partial=m.raw_terms(target,height,one,lifted)
    assert partial[0,0]>0 and partial[0,1]==.5
    env,term,robot,sensor=fixture
    term.cfg=cfg.commands.base_velocity
    before=term.reference.phase.clone()
    term._resample_command(torch.arange(env.num_envs))
    torch.testing.assert_close(term.command,torch.zeros_like(term.command))
    torch.testing.assert_close(term.reference.phase,before)
    assert torch.isfinite(m.terms(env)).all()


def test_pause_repeat_accounting_and_reset(fixture):
    from unitree_rl_lab.tasks.locomotion.mdp import pause_rewards as p
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    cfg,runner=register_task_to_hydra('Unitree-Go2-Conga-Pause','rsl_rl_cfg_entry_point')
    cfg.validate()
    assert cfg.rewards.cadence_pair.weight==.1 and cfg.rewards.cadence_phase.weight==.25
    assert cfg.rewards.pause_missing.weight==-.25 and cfg.rewards.repeated_departures.weight==-.025
    assert cfg.commands.base_velocity.reference.cadence_swing_end-cfg.commands.base_velocity.reference.cadence_swing_start==pytest.approx(.2)
    history=p.StepHistory(2,'cpu');counts=torch.zeros(2,dtype=torch.long)
    for i in range(110):
        c=torch.ones((2,4),dtype=torch.bool);h=torch.zeros((2,4))
        feet=([0,3] if any(a<=i<a+6 for a in (5,20,50,65,95)) else
              [1,2] if 35<=i<41 else [1] if 80<=i<86 else [])
        if feet:c[:,feet]=False;h[:,feet]=.01
        _,repeats=history.update(c,h,torch.tensor([i if i<60 else i-60,i]))
        counts+=repeats
    torch.testing.assert_close(counts,torch.tensor([4,6]))
    # Cost is integrated per repeated foot, not accidentally scaled by dt twice.
    assert -.025*(2/.02)*.02==pytest.approx(-.05)
    env,term,robot,sensor=fixture
    env.common_step_counter=10
    term.cfg.reference=cfg.commands.base_velocity.reference
    set_perfect_state(env,term,robot,sensor)
    p.paired_step(env);last=term._pause_history.previous_step.clone()
    p.missing_pause_contacts(env);p.repeated_departures(env)
    torch.testing.assert_close(term._pause_history.previous_step,last)


def test_liftoff_history_matches_offline():
    import runpy
    from pathlib import Path
    import numpy as np
    from unitree_rl_lab.tasks.locomotion.mdp.liftoff_history import LiftoffHistory
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    offline=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/score_liftoff_history.py'))['history_gate']
    cfg, runner = register_task_to_hydra('Unitree-Go2-Conga-History','rsl_rl_cfg_entry_point')
    cfg.validate()
    assert cfg.rewards.cadence_pair.weight == .1 and cfg.rewards.cadence_phase.weight == .25
    assert cfg.rewards.joint_pos.weight == 0 and cfg.rewards.cadence_support is None
    rng=np.random.default_rng(7)
    contact=np.ones((240,4))
    for j,pair in enumerate(([0,3],[0,3],[1,2],[0,3],[1,2],[1,2],[0,3])):
        contact[8+j*30:18+j*30,pair]=0
    contact[100:120]=rng.integers(0,2,(20,4))
    t=np.arange(240)*.02;t[150:]-=3.
    expected,_=offline(t,contact)
    state=LiftoffHistory(2,'cpu');actual=[]
    for i in range(240):
        step=torch.tensor([round(t[i]/.02),i])
        result=state.update(torch.tensor(np.stack([contact[i],contact[i]])).bool(),step)
        actual.append(result[0].numpy().copy())
    np.testing.assert_array_equal(np.array(actual),expected)
    assert state.last.shape==(2,)


def test_early_schedule_and_resume(monkeypatch):
    from unitree_rl_lab.utils.early_training import schedule, apply_schedule, EarlyRunner, OnPolicyRunner
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    cfg, runner_cfg = register_task_to_hydra('Unitree-Go2-Conga-Early', 'rsl_rl_cfg_entry_point')
    cfg.validate()
    assert runner_cfg.num_steps_per_env == 24
    assert cfg.rewards.cadence_support is None
    assert cfg.commands.base_velocity.reference.max_travel_speed == .5
    terms = {n: SimpleNamespace(weight=w) for n,w in schedule(0)[0].items()}
    command = SimpleNamespace(cfg=cfg.commands.base_velocity)
    env = SimpleNamespace(common_step_counter=0,
        reward_manager=SimpleNamespace(get_term_cfg=lambda n: terms[n], set_term_cfg=lambda n,c: terms.update({n:c})),
        command_manager=SimpleNamespace(get_term=lambda n: command))
    for iteration, fraction in [(0,0),(50,0),(125,.5),(200,1),(1000,1)]:
        env.common_step_counter = iteration*24
        weights, vx, vy = apply_schedule(env)
        assert weights['cadence_phase'] == pytest.approx(.05+.2*fraction)
        assert weights['cadence_pair'] == pytest.approx(.1*fraction)
        assert weights['joint_pos'] == pytest.approx(-.7*(1-fraction))
        assert vx == pytest.approx(.1+.4*fraction)
        assert vy == pytest.approx(.1+.15*fraction)
    env.early_frozen_step = 0
    assert apply_schedule(env)[0]['cadence_phase'] == .05
    del env.early_frozen_step
    runner = EarlyRunner.__new__(EarlyRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    runner.current_learning_iteration = 199
    monkeypatch.setattr(OnPolicyRunner, 'load', lambda *a: {'early_stage': {
        'control_steps':4800, 'schedule_version':1, 'rollout_steps':24}})
    runner.load('early')
    assert runner.current_learning_iteration == 200 and env.common_step_counter == 4800
    assert terms['cadence_phase'].weight == pytest.approx(.25)
    monkeypatch.setattr(OnPolicyRunner, 'load', lambda *a: None)
    with pytest.raises(ValueError):
        runner.load('wrong_stage')


def test_paired_cadence_config_and_budget(fixture):
    from unitree_rl_lab.tasks.locomotion.mdp import cadence_pair_scores as scores
    from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import generate_reference
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_env_cfg import CongaEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_cadence_start_env_cfg import CongaCadenceStartEnvCfg
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    cfg, runner = register_task_to_hydra("Unitree-Go2-Conga-Cadence-Paired", "rsl_rl_cfg_entry_point")
    cfg.validate()
    assert runner.num_steps_per_env == 24
    assert cfg.rewards.cadence_support.weight == .1
    assert cfg.rewards.cadence_pair.weight == .5
    assert cfg.rewards.cadence_phase.weight == .5
    assert cfg.rewards.conga_contact is None and cfg.rewards.conga_clearance is None
    assert cfg.commands.base_velocity.reference.max_travel_speed == .5
    assert CongaEnvCfg().commands.base_velocity.reference.cadence_swing_start == .04
    assert CongaCadenceStartEnvCfg().commands.base_velocity.reference.cadence_swing_end == .4
    env, term, robot, sensor = fixture
    term.cfg.reference = cfg.commands.base_velocity.reference
    ref = generate_reference(torch.arange(200)*.02, term.cfg.reference)
    ideal_contact = (ref.contact >= .5).float()
    norm = scores.raw_scores(ref.contact, ref.foot_height, ideal_contact, ref.foot_height).mean(0)
    ideal = scores.normalized_scores(scores.raw_scores(ref.contact, ref.foot_height, ideal_contact, ref.foot_height), norm)
    torch.testing.assert_close(ideal.mean(0), torch.ones(3))
    planted = scores.normalized_scores(scores.raw_scores(ref.contact, ref.foot_height, torch.ones_like(ideal_contact), torch.zeros_like(ref.foot_height)), norm)
    assert planted[:, 1].max() == 0
    assert (ideal.mean(0)*torch.tensor([.1,.5,.5])).sum() > (planted.mean(0)*torch.tensor([.1,.5,.5])).sum()
    set_perfect_state(env, term, robot, sensor)
    assert torch.isfinite(scores.env_scores(env)).all()


def test_duration_walking_configuration_and_clock(fixture):
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_duration_env_cfg import CongaDurationEnvCfg
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_basic_env_cfg import CongaBasicEnvCfg
    from unitree_rl_lab.tasks.locomotion.mdp import duration_walking as d
    cfg=CongaDurationEnvCfg(); base=CongaBasicEnvCfg()
    assert cfg.rewards.duration_air.weight==cfg.rewards.duration_stance.weight==-.05
    assert cfg.rewards.feet_air_time.weight==cfg.rewards.air_time_variance.weight==0
    for name in vars(base.rewards):
        if name not in ('feet_air_time','air_time_variance'):
            a=getattr(base.rewards,name); b=getattr(cfg.rewards,name)
            if hasattr(a,'weight'): assert a.weight==b.weight
    env=fixture[0]
    env.common_step_counter=10
    command=torch.tensor([[0.,0.,0.],[.4,0.,0.],[0.,0.,.5]])
    env.cfg=cfg
    env.command_manager=SimpleNamespace(get_command=lambda name:command)
    a=d.features(env);assert a.shape==(3,10)
    env.common_step_counter+=1;d.features(env)
    torch.testing.assert_close(env._duration_clock,torch.tensor([0.,.02,.02]))
    command[0,0]=.4;command[1,0]=.3
    env.common_step_counter+=1;d.features(env)
    torch.testing.assert_close(env._duration_clock,torch.tensor([0.,.04,.04]))
    command[:]=0;env.common_step_counter+=1;d.features(env)
    assert env._duration_clock.sum()==0
    assert d.state(env).sum()==0


def test_duration_runner_uses_live_command_ranges(monkeypatch):
    from unitree_rl_lab.utils.duration_training import DurationRunner
    from unitree_rl_lab.utils.march_training import MarchRunner
    runner=object.__new__(DurationRunner)
    live=SimpleNamespace(lin_vel_x=(-.4,.4),lin_vel_y=(-.25,.25),ang_vel_z=(-1.,1.))
    term=SimpleNamespace(cfg=SimpleNamespace(ranges=live))
    runner.env=SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:term)))
    saved={}
    monkeypatch.setattr(MarchRunner,'save',lambda self,path,infos: saved.update(infos))
    runner.save('unused')
    assert saved['duration_ranges']['lin_vel_x']==(-.4,.4)
    monkeypatch.setattr(MarchRunner,'load',lambda *args,**kwargs:saved)
    live.lin_vel_x=(-.1,.1)
    runner.load('unused')
    assert live.lin_vel_x==(-.4,.4)


def test_duration_strong_changes_only_two_weights():
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_duration_env_cfg import CongaDurationEnvCfg,CongaDurationStrongEnvCfg
    a=CongaDurationEnvCfg();b=CongaDurationStrongEnvCfg()
    assert b.rewards.duration_air.weight==b.rewards.duration_stance.weight==-.20
    b.rewards.duration_air.weight=a.rewards.duration_air.weight
    b.rewards.duration_stance.weight=a.rewards.duration_stance.weight
    assert a.to_dict()==b.to_dict()


def test_average_velocity_history_reset_commands_and_ring():
    from collections import deque
    from unitree_rl_lab.tasks.locomotion.mdp.velocity_average import VelocityHistory
    h=VelocityHistory(3,'cpu',5)
    queues=[deque(maxlen=5) for _ in range(3)]
    old=torch.zeros(3,3)
    generator=torch.Generator().manual_seed(31)
    for step in range(120):
        cmd=torch.tensor([[.4,0.,0.],[0.,0.,.5],[.2,0.,0.]])
        if 30<=step<50:cmd[0]=0
        if step>=65:cmd[2,0]=-.2
        vel=torch.randn(3,2,generator=generator)*.2
        reset=torch.tensor([step==0,step in (0,70),step==0])
        actual=h.step(vel,cmd,reset)
        expected=[]
        for i in range(3):
            active=cmd[i,:2].norm()>.05 or cmd[i,2].abs()>.05
            if reset[i] or (cmd[i]-old[i]).abs().max()>1e-6 or not active:queues[i].clear()
            error=vel[i]-cmd[i,:2];queues[i].append(error)
            instant=torch.exp(-error.square().sum()/.25)
            avg=torch.exp(-torch.stack(list(queues[i])).mean(0).square().sum()/.25)
            expected.append(.8*avg+.2*instant if active else instant)
        torch.testing.assert_close(actual,torch.stack(expected),atol=2e-6,rtol=2e-6)
        old=cmd.clone()


def test_average_configuration_changes_only_xy_tracking():
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_duration_env_cfg import CongaDurationAverageEnvCfg,CongaDurationStrongEnvCfg
    a=CongaDurationStrongEnvCfg();b=CongaDurationAverageEnvCfg()
    assert b.rewards.track_lin_vel_xy.weight==1.5
    b.rewards.track_lin_vel_xy=a.rewards.track_lin_vel_xy
    assert a.to_dict()==b.to_dict()


def test_isolated_kick_reference_and_configuration():
    from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import CongaReferenceCfg,generate_kick_only_reference
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_kick_test_env_cfg import CongaKickTestEnvCfg
    cfg=CongaReferenceCfg(kick_scale=1.,bounce_scale=0.)
    t=torch.tensor([0.,.2,1.25,1.375,1.5,2.2,3.375,3.5],dtype=torch.float64)
    r=generate_kick_only_reference(t,cfg)
    assert (r.contact[[0,1,2,4,5,7]]==1).all()
    assert r.contact[3].tolist()==[1.,0.,1.,1.]
    assert r.contact[6].tolist()==[0.,1.,1.,1.]
    assert r.foot_height[3,1]==pytest.approx(.05)
    assert r.foot_height[6,0]==pytest.approx(.05)
    assert r.kick_y[3]==pytest.approx(-cfg.nominal_front_y-.08)
    assert r.kick_y[6]==pytest.approx(cfg.nominal_front_y+.08)
    assert (r.travel==0).all() and (r.body_velocity_z==0).all()
    assert (r.body_height==.32).all()
    c=CongaKickTestEnvCfg();c.validate()
    assert c.rewards.conga_cadence is c.rewards.conga_clearance is c.rewards.conga_marching is None
    assert c.commands.base_velocity.ranges.lin_vel_x==(0.,0.)
    assert c.commands.base_velocity.reference.bounce_scale==0


def test_isolated_kick_request_independent_of_command(fixture):
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_kick_test_env_cfg import IsolatedKickCommand
    env,original,*_=fixture
    term=IsolatedKickCommand(original.cfg,env)
    env.episode_length_buf[:]=69
    a=term.reference
    term.vel_command_b[:]=torch.tensor([.4,.2,.5])
    b=term.reference
    torch.testing.assert_close(a.contact,b.contact)
    torch.testing.assert_close(a.kick_y,b.kick_y)
    torch.testing.assert_close(a.foot_height,b.foot_height)
    assert not term.command.any()


def test_kick_intro_targets_weights_and_balanced_reset(fixture):
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_kick_train_env_cfg import CongaKickTrainEnvCfg
    from unitree_rl_lab.tasks.locomotion.mdp.kick_training import KickTrainingCommand
    cfg=CongaKickTrainEnvCfg();cfg.validate()
    assert cfg.commands.base_velocity.reference.kick_lateral==cfg.commands.base_velocity.reference.kick_height==.02
    assert [cfg.rewards.kick_height.weight,cfg.rewards.kick_lateral.weight,cfg.rewards.kick_airborne.weight,cfg.rewards.kick_support.weight]==[2.,2.,.5,-1.]
    env,original,*_=fixture;env.cfg=SimpleNamespace(seed=42)
    term=KickTrainingCommand(original.cfg,env);before=term.phase_offset.clone()
    assert int((before==0).sum()) in [1,2]
    term.reset(torch.tensor([0]));assert term.phase_offset[0]==2-before[0]
    torch.testing.assert_close(term.phase_offset[1:],before[1:])
    assert term.reference.body_velocity_z.sum()==0


def test_kick_components_independent_progress_and_support():
    from unitree_rl_lab.tasks.locomotion.mdp.kick_scores import components
    from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import CongaReferenceCfg,generate_kick_only_reference
    cfg=CongaReferenceCfg(kick_lateral=.02,kick_height=.02,kick_scale=1.,bounce_scale=0.)
    ref=generate_kick_only_reference(torch.tensor([1.375,1.375]),cfg)
    y=torch.zeros(2,4);y[:,1]=-cfg.nominal_front_y
    h=torch.zeros(2,4);h[1,1]=.02
    contact=torch.ones(2,4,dtype=torch.bool);contact[1,1]=False
    v=components(y,h,contact,ref)
    assert v[1,0]>v[0,0] and v[1,1]==v[0,1]
    assert v[1,2]>v[0,2] and v[1,3]==0
    contact[1,2]=False
    assert components(y,h,contact,ref)[1,3]>v[1,3]


def test_refined_kick_lift_bonus_requires_clearance():
    from unitree_rl_lab.tasks.locomotion.mdp.kick_scores import components
    from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import CongaReferenceCfg,generate_kick_only_reference
    ref=generate_kick_only_reference(torch.tensor([1.375]*3),CongaReferenceCfg(kick_lateral=.02,kick_height=.02,bounce_scale=0.))
    y=torch.zeros(3,4);h=torch.zeros(3,4);h[:,1]=torch.tensor([0.,.001,.02])
    c=torch.ones(3,4,dtype=torch.bool);c[:,1]=False
    old=components(y,h,c,ref);new=components(y,h,c,ref,height_aware=True)
    torch.testing.assert_close(old[:,[0,1,3]],new[:,[0,1,3]])
    assert new[0,2]==0 and new[1,2]==pytest.approx(.1) and new[2,2]==1
    from unitree_rl_lab.tasks.locomotion.robots.go2.conga_kick_train_env_cfg import CongaKickRefineEnvCfg
    cfg=CongaKickRefineEnvCfg();assert cfg.rewards.kick_airborne.weight==2.
