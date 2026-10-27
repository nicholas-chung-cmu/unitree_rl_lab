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
