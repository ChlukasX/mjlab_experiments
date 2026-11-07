"""The `-NoTap` task (docs/mount_curriculum): two new reward terms, nothing else changes.

GPU-free: the configs load without a simulation, and the terms run on stand-in envs.
"""

from types import SimpleNamespace

import pytest
import torch

from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_curriculum_cfg as cur

PREFIX = "Mjlab-Piplus-"
NOTAP = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap"
NEW_TERMS = cur.NOTAP_TERMS


def _tasks():
    from mjlab.tasks.registry import list_tasks

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return [t for t in list_tasks() if t.startswith(PREFIX)]


def _rewards(task, play=False):
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(task, play=play).rewards


def test_old_tasks_have_neither_term():
    """The default `notap=False` keeps the old reward set of every other task."""
    for task in _tasks():
        if task == NOTAP:
            continue
        for play in (False, True):
            assert not set(NEW_TERMS) & set(_rewards(task, play)), task


def test_old_stage_settings_have_neither_weight():
    for s in range(cur.FINAL_STAGE + 1):
        assert not set(NEW_TERMS) & set(cur.stage_settings(s, 0)["weights"])
        assert not set(NEW_TERMS) & set(cur.ramped_stage_settings(s, 0)["weights"])


def test_notap_task_registers_with_the_two_terms():
    assert NOTAP in _tasks()
    train, play = _rewards(NOTAP), _rewards(NOTAP, play=True)
    assert set(NEW_TERMS) <= set(train)
    assert play["free_sole_floor"].weight == cur.TAP_PENALTY < 0
    assert play["stance_duration"].weight == cur.STANCE_BONUS > 0
    assert train["free_sole_floor"].func is cur.free_sole_floor_contact
    assert train["stance_duration"].func is cur.stance_duration


def test_notap_task_has_the_old_terms_unchanged():
    old = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess"
    for play in (False, True):
        a, b = _rewards(old, play), _rewards(NOTAP, play)
        assert set(b) - set(a) == set(NEW_TERMS)
        assert all(a[k].weight == b[k].weight for k in a)


def test_notap_curriculum_term_only_in_the_new_task():
    from mjlab.tasks.registry import load_env_cfg

    assert "notap" in load_env_cfg(NOTAP).curriculum
    assert "notap" not in load_env_cfg(NOTAP.removesuffix("-NoTap")).curriculum


def test_weights_are_zero_before_stage_8_and_ramp_in_at_stage_8():
    for s in range(cur.FINAL_STAGE):
        w = cur.ramped_stage_settings(s, 1e6, notap=True)["weights"]
        assert w["free_sole_floor"] == w["stance_duration"] == 0.0
    ramp = cur.STAGE_RAMP_ITERS[cur.FINAL_STAGE]
    start = cur.ramped_stage_settings(cur.FINAL_STAGE, 0, notap=True)["weights"]
    mid = cur.ramped_stage_settings(cur.FINAL_STAGE, ramp / 2, notap=True)["weights"]
    end = cur.ramped_stage_settings(cur.FINAL_STAGE, ramp, notap=True)["weights"]
    assert start["free_sole_floor"] == start["stance_duration"] == 0.0
    assert cur.TAP_PENALTY < mid["free_sole_floor"] < 0
    assert 0 < mid["stance_duration"] < cur.STANCE_BONUS
    assert end["free_sole_floor"] == cur.TAP_PENALTY
    assert end["stance_duration"] == cur.STANCE_BONUS


def test_run_length_counts_breaks_and_restarts():
    t, f = torch.tensor(True), torch.tensor(False)
    n = torch.zeros(1)
    seen = []
    for active, fresh in ((t, f), (t, f), (f, f), (t, f), (t, t), (t, f)):
        n = cur.run_length(n, active.view(1), fresh.view(1))
        seen.append(int(n))
    assert seen == [1, 2, 0, 1, 1, 2]


def _fake_env(stance_seq, n=2):
    """Env stand-in whose `single_leg_stance` term returns the next row of `stance_seq`."""
    rows = iter(stance_seq)
    term = SimpleNamespace(func=lambda env, **kw: next(rows), params={})
    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.arange(1, n + 1),
        reward_manager=SimpleNamespace(get_term_cfg=lambda name: term),
    )


def test_stance_duration_grows_with_the_stance_and_caps():
    steps = 120  # 2.4 s at 0.02 s: past the 2 s cap
    seq = [torch.tensor([1.0, 0.0])] * steps
    env = _fake_env(seq)
    out = []
    for _ in range(steps):
        env.episode_length_buf = env.episode_length_buf + 1
        out.append(cur.stance_duration(env, cap_s=2.0))
    out = torch.stack(out)
    assert out[0, 0] == pytest.approx(0.02 / 2.0)
    assert out[49, 0] == pytest.approx(0.5)
    assert out[-1, 0] == 1.0
    assert (out[:, 1] == 0).all()
    assert (out[1:, 0] >= out[:-1, 0]).all()


def test_one_long_stance_pays_more_than_many_short_ones():
    """The same 60 stance steps: one stance against six of 10 steps with a break between."""

    def total(rows):
        env = _fake_env([torch.tensor([r]) for r in rows], n=1)
        env.episode_length_buf = torch.tensor([5])
        return sum(float(cur.stance_duration(env, cap_s=2.0)) for _ in rows)

    long_one = [1.0] * 60
    short_ones = ([1.0] * 10 + [0.0]) * 6
    assert total(long_one) > 3 * total(short_ones)


def test_stance_duration_restarts_with_the_episode():
    env = _fake_env([torch.tensor([1.0])] * 4, n=1)
    env.episode_length_buf = torch.tensor([5])
    for _ in range(3):
        cur.stance_duration(env, cap_s=2.0)
    assert float(env._stance_len) == 3
    env.episode_length_buf = torch.tensor([1])  # first step after a reset
    cur.stance_duration(env, cap_s=2.0)
    assert float(env._stance_len) == 1
