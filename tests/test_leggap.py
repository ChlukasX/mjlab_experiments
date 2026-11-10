"""The `-LegGap` task (docs/mount_selfcontact): one new reward term, nothing else changes.

GPU-free: the configs load without a simulation, and the distance runs on a stand-in env.
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_selfcontact_cfg as sc

PREFIX = "Mjlab-Piplus-"
NOTAP = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap"
LEGGAP = NOTAP + "-LegGap"
TERM = "leg_gap"


def _cfg(task, play=False):
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return load_env_cfg(task, play=play)


def test_leggap_task_differs_from_notap_only_in_the_term():
    for play in (False, True):
        a, b = _cfg(NOTAP, play), _cfg(LEGGAP, play)
        assert set(b.rewards) - set(a.rewards) == {TERM}
        assert set(b.curriculum) - set(a.curriculum) == ({TERM} if not play else set())
        for k in a.rewards:
            assert a.rewards[k].weight == b.rewards[k].weight, k
    a, b = _cfg(NOTAP), _cfg(LEGGAP)
    # the model, the clip, the observations, the simulation sizes: unchanged
    assert a.scene.entities["robot"].spec_fn is b.scene.entities["robot"].spec_fn
    assert a.actions["joint_pos"].clip == b.actions["joint_pos"].clip
    assert a.sim.nconmax == b.sim.nconmax and a.sim.njmax == b.sim.njmax
    assert len(a.scene.sensors) == len(b.scene.sensors)
    for group in ("actor", "critic"):
        assert set(a.observations[group].terms) == set(b.observations[group].terms)


def test_weight_is_zero_in_training_and_full_in_play():
    assert _cfg(LEGGAP).rewards[TERM].weight == 0.0  # the curriculum sets it
    assert _cfg(LEGGAP, play=True).rewards[TERM].weight == sc.LEG_GAP_WEIGHT < 0
    assert _cfg(LEGGAP).rewards[TERM].func is sc.leg_gap


def test_no_old_task_has_the_term():
    from mjlab.tasks.registry import list_tasks

    for task in list_tasks():
        if task.startswith(PREFIX) and not task.startswith(LEGGAP):  # -LegGap and its variants
            assert TERM not in _cfg(task).rewards, task


def test_box_and_capsule_distance():
    half = torch.tensor([0.1, 0.2, 0.3])
    p = torch.tensor([[[0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.4]]])
    d = sc._signed_distance(p, True, half)[0]
    assert d[0] == pytest.approx(0.05)
    assert d[1] == pytest.approx(-0.1)  # centre: inside, nearest face 0.1 away
    assert d[2] == pytest.approx(math.sqrt(0.1**2 + 0.1**2 + 0.1**2))
    cap = torch.tensor([0.01, 0.05, 0.0])  # radius, half length
    q = torch.tensor([[[0.03, 0.0, 0.0], [0.0, 0.0, 0.08], [0.0, 0.0, 0.0]]])
    d = sc._signed_distance(q, False, cap)[0]
    assert d[0] == pytest.approx(0.02)
    assert d[1] == pytest.approx(0.03 - 0.01)
    assert d[2] == pytest.approx(-0.01)


def _model():
    xml = """<mujoco><worldbody>
    <body name="r"><freejoint/><geom name="robot/r_calf_link_collision0" type="box" size="0.03 0.03 0.03"/></body>
    <body name="l"><freejoint/><geom name="robot/l_calf_link_collision0" type="box" size="0.03 0.03 0.03"/></body>
    </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def test_leg_distance_of_two_boxes_matches_mujoco():
    model = _model()
    data = mujoco.MjData(model)
    for gap in (-0.01, 0.0, 0.015, 0.05):
        data.qpos[:] = 0
        data.qpos[0:3] = [0, 0, 0]
        data.qpos[7:10] = [0.06 + gap, 0.0, 0.0]
        data.qpos[3] = data.qpos[10] = 1.0
        mujoco.mj_kinematics(model, data)
        fromto = np.zeros(6)
        true = mujoco.mj_geomDistance(model, data, 0, 1, 1.0, fromto)
        xpos = torch.tensor(data.geom_xpos, dtype=torch.float32)[None]
        xmat = torch.tensor(data.geom_xmat, dtype=torch.float32).reshape(1, 2, 3, 3)
        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            sim=SimpleNamespace(
                mj_model=model,
                data=SimpleNamespace(geom_xpos=xpos, geom_xmat=xmat.reshape(1, 2, 9)),
            ),
        )
        assert sc.leg_distance(env).item() == pytest.approx(true, abs=1e-5), gap


def test_term_value_is_zero_at_the_margin_one_at_touch(monkeypatch):
    m = sc.LEG_GAP_MARGIN
    env = SimpleNamespace(num_envs=1, device="cpu")
    cases = ((m, 0.0), (2 * m, 0.0), (0.0, 1.0), (m / 2, 0.5), (-m, 2.0), (-5 * m, sc.LEG_GAP_CAP))
    for d, want in cases:
        monkeypatch.setattr(sc, "leg_distance", lambda e, d=d: torch.tensor([d]))
        assert sc.leg_gap(env, m).item() == pytest.approx(want), d
