"""Numerical motion contract tests, runnable without launching Isaac Sim."""

import importlib.util
import sys
import torch
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[1] / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/conga_reference.py"
)
spec = importlib.util.spec_from_file_location("conga_reference", PATH)
ref_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref_module
spec.loader.exec_module(ref_module)
Cfg = ref_module.CongaReferenceCfg
generate = ref_module.generate_reference


def at(times, cfg=None):
    return generate(torch.tensor(times, dtype=torch.float64), cfg or Cfg())


def test_timing_and_alternation():
    r = at([1.249, 1.25, 1.375, 1.5, 3.375, 5.375])
    assert r.kick_active.tolist() == [False, True, True, False, True, True]
    assert r.kick_leg.tolist() == [1, 1, 1, 1, 0, 1]
    assert r.beat_index.tolist() == [2, 2, 2, 3, 6, 2]
    torch.testing.assert_close(r.contact[2], torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(r.contact[4], torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64))


def test_kick_geometry_and_endpoint_velocity():
    cfg = Cfg()
    r = at([1.25, 1.375, 1.5, 3.375])
    assert r.kick_y[1].item() == pytest.approx(-cfg.nominal_front_y - 0.08)
    assert r.kick_y[3].item() == pytest.approx(cfg.nominal_front_y + 0.08)
    assert r.foot_height[1, 1].item() == pytest.approx(0.05)
    assert r.foot_height[3, 0].item() == pytest.approx(0.05)
    for t in [1.25, 1.375, 1.5]:
        eps = 1e-6
        x = at([t - eps, t, t + eps])
        assert abs((x.kick_y[2] - x.kick_y[0]).item() / (2 * eps)) < 1e-4
        assert abs((x.foot_height[2, 1] - x.foot_height[0, 1]).item() / (2 * eps)) < 1e-4


def test_three_diagonals_then_next_front_and_recovery():
    r = at([0.2, 0.7, 1.12, 1.375, 1.75, 2.2, 2.7, 3.12, 3.375, 3.75])
    assert (r.contact[:5] < 0.5).tolist() == [
        [True, False, False, True],
        [False, True, True, False],
        [True, False, False, True],
        [False, True, False, False],
        [False] * 4,
    ]
    assert (r.contact[5:8] < 0.5).tolist() == (r.contact[:3] > 0.5).tolist()
    dense = at(torch.linspace(0, 4, 4001).tolist())
    assert (dense.contact.sum(-1) >= 2 - 1e-6).all()
    assert (dense.contact[dense.kick_active].sum(-1) >= 3 - 1e-6).all()


def test_speed_cap_envelope_and_zero_commands():
    cmds = torch.tensor([[1.25, 1.25, 1.0], [-2.0, -3.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    original = cmds.clone()
    capped = ref_module.cap_velocity(cmds, 1.25)
    assert (torch.linalg.vector_norm(capped[:, :2], dim=-1) <= 1.25 + 1e-6).all()
    torch.testing.assert_close(capped[:, 2], cmds[:, 2])
    torch.testing.assert_close(cmds, original)
    r = at([0.0, 1.1, 1.175, 1.25, 1.375, 1.5, 1.6, 1.7, 2.0])
    torch.testing.assert_close(
        r.travel, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0], dtype=torch.float64)
    )
    # Zero speed never changes the reference or removes marching.
    assert at([0.2]).foot_height.max() > 0.02


def test_bounce_magnitude_frequency_and_derivative():
    r = at([0.125, 0.375, 0.625])
    assert (r.body_height[0] - r.body_height[1]).item() == pytest.approx(0.015)
    torch.testing.assert_close(r.body_height[0], r.body_height[2])
    eps = 1e-6
    r = at([0.1 - eps, 0.1, 0.1 + eps])
    assert ((r.body_height[2] - r.body_height[0]) / (2 * eps)).item() == pytest.approx(r.body_velocity_z[1].item())


def test_independent_reset_and_four_second_periodicity():
    t = torch.tensor([0.7, 1.375, 3.375], dtype=torch.float64)
    before = generate(t, Cfg())
    t[1] = 0.0
    after = generate(t, Cfg())
    torch.testing.assert_close(before.contact[[0, 2]], after.contact[[0, 2]])
    torch.testing.assert_close(after.phase_features(), generate(t + 4, Cfg()).phase_features())
    assert after.phase[1] == 0


def test_contact_and_travel_continuity():
    for t in [0.0, 0.04, 0.4, 0.54, 0.9, 1.02, 1.1, 1.23, 1.25, 1.5, 1.7, 2.0, 4.0]:
        r = at([t - 1e-6, t + 1e-6])
        assert (r.contact[0] - r.contact[1]).abs().max() < 1e-3
        assert (r.foot_height[0] - r.foot_height[1]).abs().max() < 1e-3
        assert (r.travel[0] - r.travel[1]).abs() < 1e-3


def test_touchdown_distance_matches_contact_crossings():
    for time, foot in [(0.38, 0), (0.88, 1), (1.21, 3), (1.48, 1), (3.48, 0)]:
        r = at([time])
        assert r.contact[0, foot].item() == pytest.approx(0.5)
        assert r.touchdown_distance[0, foot].item() == pytest.approx(0.0)


def test_recovery_frees_contacts_while_travel_resumes():
    r = at([1.5, 1.7, 1.95, 2.0])
    torch.testing.assert_close(r.contact[1], torch.full((4,), 0.5, dtype=torch.float64))
    assert r.travel[1] == pytest.approx(1.0)
    assert r.recovery[1] == 1 and r.recovery[0] == 0 and r.recovery[3] == 0
    assert (r.contact[2] > 0.5).all()


def test_curriculum_and_validation():
    cfg = Cfg(kick_enabled=False, bounce_scale=0.0)
    r = at([1.7, 3.7], cfg)
    assert not r.kick_active.any()
    assert (r.travel == 1).all()
    assert (r.body_height == cfg.base_height).all()
    assert (r.contact < 0.5).sum().item() == 4
    assert at([1.375], Cfg(kick_scale=0.5)).foot_height[0, 1].item() == pytest.approx(0.04)
    Cfg().validate()
    with pytest.raises(ValueError):
        Cfg(max_travel_speed=1.3).validate()
