"""Physics-independent contracts, including adversarial completion/transition cases."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

PATH = Path(__file__).resolve().parents[1]/"source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/skills/core.py"
spec = importlib.util.spec_from_file_location("hierarchical_core", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def setup(n=2):
    body = torch.tensor([[0., 0., .32]]).repeat(n, 1)
    feet = torch.tensor([[[.2, .17, 0.], [.2, -.17, 0.], [-.2, .17, 0.], [-.2, -.17, 0.]]]).repeat(n, 1, 1)
    state = m.State(body, torch.zeros(n), feet, torch.full((n, 4), 30.), torch.zeros(n, 4, 3),
                    torch.zeros(n), torch.zeros(n), torch.zeros(n))
    core = m.SkillExecutor(n, "cpu")
    ids = torch.arange(n)
    core.reset(ids, state)
    for _ in range(16):
        core.update(state, .02)
    return core, state, ids


def begin_step(c, s, ids, pair=(1, 0, 0, 1), goal=None):
    c.begin(ids, m.Skill.STEP, pair, torch.zeros(len(ids), 3) if goal is None else goal, 1., s)


def test_exact_rhythm_and_speed_cap():
    p = m.DancePlanner(2, "cpu")
    assert p.cfg.beat == 60/90
    assert 2.5*p.cfg.beat == pytest.approx(5/3)
    p.set_velocity(torch.tensor([[.5, .5, 3.], [0., 0., -3.]]))
    assert p.command[0, :2].norm().item() == pytest.approx(.5)
    assert p.command[1, :2].norm() == 0
    assert p.command[1, 2].item() == pytest.approx(-.8)


def test_standing_cannot_complete_step():
    c, s, ids = setup()
    begin_step(c, s, ids)
    for _ in range(72):
        c.update(s, .02)
    assert (c.status == m.Status.TIMEOUT).all()
    assert not c.success_event.any()
    assert c.metrics["missing_touchdowns"].tolist() == [2, 2]


def test_normal_step_success_once_without_zero_velocity_requirement():
    c, s, ids = setup()
    begin_step(c, s, ids)
    s.force[:, [0, 3]] = 0
    s.feet[:, [0, 3], 2] = .04
    for _ in range(30):
        c.update(s, .02)
    s.feet[:, [0, 3], 2] = 0
    s.force[:] = 30
    for _ in range(6):
        c.update(s, .02)
    assert (c.status == m.Status.SUCCESS).all()
    c.update(s, .02)
    assert not c.success_event.any()


def test_contact_hysteresis_and_confirmation():
    c, s, ids = setup()
    s.force[:] = 3
    c.update(s, .02)
    assert c.confirmed.all()
    s.force[:] = 1
    c.update(s, .02)
    assert not c.contact.any()
    s.force[:] = 3
    c.update(s, .02)
    assert not c.contact.any()
    s.force[:] = 6
    c.update(s, .02)
    assert not c.confirmed.any()
    c.update(s, .02)
    assert c.confirmed.all()


def test_kick_is_airborne_endpoint_and_then_direct_step():
    c, s, ids = setup()
    c.begin(ids, m.Skill.KICK, (0, 1, 0, 0), torch.zeros(2, 3), .25, s)
    s.feet[:, 1, 1] -= .08
    s.feet[:, 1, 2] = .05
    s.force[:, 1] = 0
    for _ in range(13):
        c.update(s, .02)
    assert (c.status == m.Status.SUCCESS).all()
    assert not c.confirmed[:, 1].any()
    before = s.feet.clone()
    begin_step(c, s, ids, (0, 1, 1, 0))
    torch.testing.assert_close(c.reference(s)[0], before)
    assert c.entry_raised[:, 1].all()
    s.force[:, 2] = 0
    s.feet[:, 2, 2] = .04
    for _ in range(35):
        c.update(s, .02)
    s.feet.copy_(c.target_feet)
    s.force[:] = 30
    for _ in range(3):
        c.update(s, .02)
    assert (c.status == m.Status.SUCCESS).all()
    assert not c.premature.any()


def test_wrong_postkick_pair_rejected():
    c, s, ids = setup()
    s.force[:, 1] = 0
    s.feet[:, 1, 2] = .05
    c.update(s, .02)
    with pytest.raises(ValueError, match="entry"):
        begin_step(c, s, ids)


def test_kick_cannot_complete_by_retracting_or_while_moving_fast():
    c, s, ids = setup()
    c.begin(ids, m.Skill.KICK, (1, 0, 0, 0), torch.zeros(2, 3), .25, s)
    for _ in range(15):
        c.update(s, .02)
    assert (c.status == m.Status.RUNNING).all()
    s.feet[:, 0, 1] += .08
    s.feet[:, 0, 2] = .05
    s.force[:, 0] = 0
    s.foot_velocity[:, 0, 1] = 1
    c.update(s, .02)
    assert (c.status == m.Status.RUNNING).all()


def test_partial_liftoff_no_success_and_fall_precedes_success():
    c, s, ids = setup()
    begin_step(c, s, ids)
    s.force[:, 0] = 0
    s.feet[:, 0, 2] = .04
    for _ in range(35):
        c.update(s, .02)
    s.force[:] = 30
    s.feet[:, 0, 2] = 0
    for _ in range(3):
        c.update(s, .02)
    assert (c.status == m.Status.RUNNING).all()
    s.tilt[:] = .8
    c.update(s, .02)
    assert (c.status == m.Status.FAILURE).all()


def test_stabilize_preserves_raised_leg():
    c, s, ids = setup()
    s.force[:, 0] = 0
    s.feet[:, 0, 2] = .05
    c.update(s, .02)
    c.begin(ids, m.Skill.STABILIZE, c.confirmed, torch.zeros(2, 3), .1, s)
    for _ in range(6):
        c.update(s, .02)
    assert (c.status == m.Status.SUCCESS).all()
    torch.testing.assert_close(c.reference(s)[0], s.feet)


def test_masked_reset_does_not_reset_other_skill():
    c, s, ids = setup()
    begin_step(c, s, ids)
    c.update(s, .02)
    age = c.elapsed[1].clone()
    c.reset(torch.tensor([0]), s)
    assert c.kind.tolist() == [m.Skill.STABILIZE, m.Skill.STEP]
    assert c.elapsed[1] == age


def test_kick_reference_has_zero_endpoint_derivative_and_no_retraction():
    c, s, ids = setup()
    c.begin(ids, m.Skill.KICK, (1, 0, 0, 0), torch.zeros(2, 3), .25, s)
    c.elapsed[:] = .25-1e-4
    left = c.reference(s)[0].clone()
    c.elapsed[:] = .25
    end = c.reference(s)[0].clone()
    c.elapsed[:] = .4
    torch.testing.assert_close(c.reference(s)[0], end)
    assert ((end-left)/1e-4).abs().max() < .003
    assert end[0, 0, 2].item() == pytest.approx(.05)


def test_planner_dispatches_one_skill_and_honors_deadline():
    c, s, ids = setup()
    p = m.DancePlanner(2, "cpu")
    p.advance(c, s, .02)
    assert (c.kind == m.Skill.STEP).all()
    assert p.slot.tolist() == [1, 1]
    assert c.duration[0].item() == pytest.approx(2/3-.02)
    p.tick[:] = 90  # missed kick deadline while still running
    p.advance(c, s, .02)
    assert (c.kind == m.Skill.STEP).all()
    assert p.slot.tolist() == [1, 1]


def test_planner_kick_to_correct_pair_and_alternating_phrase():
    c, s, ids = setup()
    p = m.DancePlanner(2, "cpu")
    p.slot[:] = 3
    p.due[:] = 5/3
    p.tick[:] = 116
    p.advance(c, s, .02)
    assert (c.kind == m.Skill.KICK).all()
    s.force[:, 1] = 0
    s.feet[:, 1, 1] -= .08
    s.feet[:, 1, 2] = .05
    for _ in range(13):
        c.update(s, .02)
    p.tick += 12
    p.advance(c, s, .02)
    assert (c.kind == m.Skill.STEP).all()
    assert c.selected.tolist() == [[False, True, True, False]]*2
    assert p.phrase.tolist() == [1, 1]
    assert p.slot.tolist() == [1, 1]
    assert (c.duration >= 2/3-1e-6).all()


def test_heading_rotation_and_displacement_goal():
    c, s, ids = setup()
    s.yaw[:] = torch.pi/2
    c.nominal_xy = m.rotate_xy(s.feet[..., :2]-s.body[:, None, :2], -s.yaw[:, None])
    goal = torch.tensor([[.1, 0, 0.]]).repeat(2, 1)
    begin_step(c, s, ids, goal=goal)
    torch.testing.assert_close(c.target_feet[:, 0, :2]-s.feet[:, 0, :2], torch.tensor([[0., .1]]).repeat(2, 1), atol=1e-6, rtol=0)
    assert c.features(s).shape == (2, 52)


def test_premature_raised_touchdown_recorded():
    c, s, ids = setup()
    s.force[:, 1] = 0
    s.feet[:, 1, 2] = .05
    c.update(s, .02)
    begin_step(c, s, ids, (0, 1, 1, 0))
    s.force[:, 1] = 30
    s.feet[:, 1, 2] = 0
    c.update(s, .02)
    assert c.premature.tolist() == [1, 1]


def test_full_oracle_sequence_preserves_three_steps_and_alternating_kicks():
    c, s, ids = setup()
    c.cfg.touchdown_tolerance = .06
    p = m.DancePlanner(2, "cpu")
    events = []
    for i in range(700):
        # Synthetic kinematic oracle only; this establishes scheduling logic,
        # explicitly not dynamics feasibility or policy performance.
        c.elapsed += .02
        ref, _ = c.reference(s)
        c.elapsed -= .02
        s.foot_velocity = (ref-s.feet)/.02
        s.feet = ref
        s.force = torch.where(s.feet[..., 2] < .002, 30., 0.)
        c.update(s, .02)
        if c.success_event[0] and c.kind[0] != m.Skill.STABILIZE:
            events.append((int(c.kind[0]), c.selected[0].tolist()))
        p.advance(c, s, .02)
    kicks = [e for e in events if e[0] == m.Skill.KICK]
    assert len(kicks) >= 4
    assert [e[1].index(True) for e in kicks[:4]] == [1, 0, 1, 0]
    assert [kind for kind, _ in events[:8]] == [0, 0, 0, 1, 0, 0, 0, 1]


def test_early_step_completion_dispatches_stabilize_until_due():
    c, s, ids = setup()
    p = m.DancePlanner(2, "cpu")
    p.advance(c, s, .02)
    c.status[:] = m.Status.SUCCESS
    c.elapsed[:] = .5
    p.tick[:] = 25
    p.advance(c, s, .02)
    assert (c.kind == m.Skill.STABILIZE).all()
    assert c.selected.all()
    torch.testing.assert_close(c.reference(s)[0], s.feet)


def test_terminal_failures_are_latched():
    c, s, ids = setup()
    begin_step(c, s, ids)
    s.tilt[:] = 1
    c.update(s, .02)
    assert c.failure_event.all()
    s.tilt[:] = 0
    c.update(s, .02)
    assert (c.status == m.Status.FAILURE).all()
    assert not c.failure_event.any()


def test_beats_are_touchdown_targets_and_third_step_has_full_lead():
    c, s, ids = setup()
    c.cfg.touchdown_tolerance = .06
    p = m.DancePlanner(2, "cpu")
    deadlines, durations = [], []
    for _ in range(180):
        c.elapsed += .02
        ref, _ = c.reference(s)
        c.elapsed -= .02
        s.foot_velocity = (ref-s.feet)/.02
        s.feet = ref
        s.force = torch.where(s.feet[..., 2] < .002, 30., 0.)
        c.update(s, .02)
        p.advance(c, s, .02)
        if c.kind[0] == m.Skill.STEP and c.elapsed[0] == 0:
            deadlines.append(float(p.time(.02)[0]+c.duration[0]))
            durations.append(float(c.duration[0]))
    assert deadlines[:4] == pytest.approx([0., 2/3, 4/3, 8/3], abs=1e-6)
    assert min(durations[:3]) > .60
    assert durations[3] > .70  # kick-to-next-beat landing budget


def test_landing_early_then_waiting_does_not_satisfy_beat_timing():
    c, s, ids = setup()
    c.cfg.touchdown_tolerance = .06
    begin_step(c, s, ids)
    s.feet[:, [0, 3], 2] = .04
    s.force[:, [0, 3]] = 0
    for _ in range(10):
        c.update(s, .02)
    s.feet[:, [0, 3], 2] = 0
    s.force[:] = 30
    for _ in range(62):
        c.update(s, .02)
    assert (c.status == m.Status.TIMEOUT).all()
    assert (c.touchdown_time[:, [0, 3]] < .3).all()


def test_per_request_kick_geometry_and_duration_are_observable():
    c, s, ids = setup()
    c.begin(ids, m.Skill.KICK, (1, 0, 0, 0), torch.zeros(2, 3), torch.tensor([.25, .3]), s,
            lateral_displacement=torch.tensor([.08, .06]), foot_height=torch.tensor([.05, .04]))
    c.elapsed.copy_(c.duration)
    ref, _ = c.reference(s)
    torch.testing.assert_close(ref[:, 0, 1]-s.feet[:, 0, 1], torch.tensor([.08, .06]))
    torch.testing.assert_close(ref[:, 0, 2], torch.tensor([.05, .04]))
    assert not torch.equal(c.features(s)[0], c.features(s)[1])


def test_repeated_verified_swing_fails_before_success():
    c,s,ids=setup()
    begin_step(c,s,ids)
    for _ in range(2):
        s.force[:,[0,3]]=0
        s.feet[:,[0,3],2]=.04
        c.update(s,.02)
    s.force[:]=30
    s.feet[:,:,2]=0
    for _ in range(2):c.update(s,.02)
    assert (c.landing_count[:,[0,3]]==1).all()
    s.force[:,[0,3]]=0
    s.feet[:,[0,3],2]=.04
    for _ in range(2):c.update(s,.02)
    assert (c.status==m.Status.FAILURE).all()
    assert (c.swing_count[:,[0,3]]==2).all()


def test_support_loss_confirmed_but_one_sample_chatter_tolerated():
    c,s,ids=setup()
    begin_step(c,s,ids)
    s.force[:,1]=0
    c.update(s,.02)
    assert (c.status==m.Status.RUNNING).all()
    s.force[:,1]=30
    for _ in range(2):c.update(s,.02)
    assert not c.support_lost.any()
    s.force[:,1]=0
    for _ in range(2):c.update(s,.02)
    assert (c.status==m.Status.FAILURE).all()
    assert c.support_lost.all()
