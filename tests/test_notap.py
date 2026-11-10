"""The `-NoTap` task (docs/mount_curriculum): two new reward terms, nothing else changes.

GPU-free: the configs load without a simulation, and the terms run on stand-in envs.
"""

from types import SimpleNamespace

import pytest
import torch

from mjlab_piplus.common.rewards import stance
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
        if task.startswith(NOTAP):  # -NoTap and -MidThigh
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
    assert play["free_foot_contact"].weight == cur.TAP_PENALTY < 0
    assert play["stance_duration"].weight == cur.STANCE_BONUS > 0
    assert train["free_foot_contact"].func is cur.free_foot_contact
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
        assert w["free_foot_contact"] == w["stance_duration"] == 0.0
    ramp = cur.STAGE_RAMP_ITERS[cur.FINAL_STAGE]
    start = cur.ramped_stage_settings(cur.FINAL_STAGE, 0, notap=True)["weights"]
    mid = cur.ramped_stage_settings(cur.FINAL_STAGE, ramp / 2, notap=True)["weights"]
    end = cur.ramped_stage_settings(cur.FINAL_STAGE, ramp, notap=True)["weights"]
    assert start["free_foot_contact"] == start["stance_duration"] == 0.0
    assert cur.TAP_PENALTY < mid["free_foot_contact"] < 0
    assert 0 < mid["stance_duration"] < cur.STANCE_BONUS
    assert end["free_foot_contact"] == cur.TAP_PENALTY
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


def _contact_env(
    *, latched=True, sup=(False, False), clearance=(0.2, 0.2), ball=(False, False), n=1
):
    """Env stand-in for `free_foot_contact`: support flags, sole clearances (m), foot-ball contact."""
    env = SimpleNamespace(num_envs=n, device="cpu")
    if latched is not None:
        env._on_ball_latch = torch.full((n,), latched)
    env._sup = torch.tensor([sup]).expand(n, 2)
    env._clr = torch.tensor([clearance]).expand(n, 2)
    env._ball = torch.tensor([ball]).expand(n, 2)
    return env


@pytest.fixture
def contact(monkeypatch):
    """`free_foot_contact` with its three helpers replaced by the stand-in env's values.

    The function lives in `common/rewards/stance.py`, so the helpers are patched there."""
    monkeypatch.setattr(stance, "_support_foot", lambda env, f, b, r, flat: env._sup)
    monkeypatch.setattr(stance, "_sole_clearance", lambda env, f: env._clr)
    monkeypatch.setattr(stance, "_ball_contact", lambda env: env._ball)
    return lambda env: float(cur.free_foot_contact(env, None, None)[0])


def test_floor_contact_is_penalized_without_a_support_foot(contact):
    """A tap disturbs the support foot (not flat, or off the ball): the term must still fire."""
    low = cur.FLOOR_TOUCH_CLEARANCE / 2
    assert contact(_contact_env(sup=(False, False), clearance=(low, 0.2))) == 1.0
    assert contact(_contact_env(sup=(False, False), clearance=(0.2, low))) == 1.0
    assert contact(_contact_env(sup=(True, False), clearance=(0.2, low))) == 1.0
    assert contact(_contact_env(sup=(False, True), clearance=(low, 0.2))) == 1.0
    assert contact(_contact_env(sup=(False, False), clearance=(low, low))) == 1.0


def test_free_foot_on_the_ball_is_penalized_in_single_support(contact):
    assert contact(_contact_env(sup=(True, False), ball=(False, True))) == 1.0
    assert contact(_contact_env(sup=(False, True), ball=(True, False))) == 1.0


def test_the_support_foot_is_not_penalized(contact):
    """Ball contact of the support foot, and a clean stance, give 0."""
    assert contact(_contact_env(sup=(True, False), ball=(True, False))) == 0.0
    assert contact(_contact_env(sup=(False, True), ball=(False, True))) == 0.0
    assert contact(_contact_env(sup=(True, False), ball=(False, False))) == 0.0


def test_both_feet_on_the_ball_or_neither_gives_no_ball_penalty(contact):
    """The ball branch needs exactly one support foot (two feet on the ball is the mount, not a tap)."""
    assert contact(_contact_env(sup=(True, True), ball=(True, True))) == 0.0
    assert contact(_contact_env(sup=(False, False), ball=(True, True))) == 0.0


def test_nothing_is_penalized_when_not_latched(contact):
    low = cur.FLOOR_TOUCH_CLEARANCE / 2
    assert (
        contact(
            _contact_env(latched=False, clearance=(low, low), sup=(True, False), ball=(False, True))
        )
        == 0.0
    )
    env = _contact_env(latched=None, clearance=(low, low))
    assert not hasattr(env, "_on_ball_latch")
    assert contact(env) == 0.0


def test_free_foot_contact_is_per_env(contact):
    env = _contact_env(n=3)
    env._clr = torch.tensor([[0.005, 0.2], [0.2, 0.2], [0.2, 0.2]])
    env._sup = torch.tensor([[False, False], [True, False], [True, False]])
    env._ball = torch.tensor([[False, False], [False, True], [True, False]])
    assert cur.free_foot_contact(env, None, None).tolist() == [1.0, 1.0, 0.0]


MIDTHIGH = NOTAP + "-MidThigh"


def test_midthigh_limits_the_thigh_and_pays_no_heading():
    """`-MidThigh`: thigh range and clip +-MID_THIGH_LIMIT, heading weight 0 in train and play."""
    from mjlab.tasks.registry import load_env_cfg

    for play in (False, True):
        cfg = load_env_cfg(MIDTHIGH, play=play)
        spec = cfg.scene.entities["robot"].spec_fn()
        ranges = {j.name: list(j.range) for j in spec.joints if "thigh" in j.name}
        assert ranges == {n: [-cur.MID_THIGH_LIMIT, cur.MID_THIGH_LIMIT] for n in ranges} and ranges
        clip = cfg.actions["joint_pos"].clip
        assert all(clip[n] == (-cur.MID_THIGH_LIMIT, cur.MID_THIGH_LIMIT) for n in ranges)
        assert cfg.rewards["support_foot_heading"].weight == 0.0
    assert "heading_off" in load_env_cfg(MIDTHIGH).curriculum
    assert "heading_off" not in load_env_cfg(NOTAP).curriculum


def test_midthigh_differs_from_notap_only_in_thigh_and_heading():
    from mjlab.tasks.registry import load_env_cfg

    a, b = load_env_cfg(NOTAP), load_env_cfg(MIDTHIGH)
    assert set(a.rewards) == set(b.rewards)
    assert all(a.rewards[k].weight == b.rewards[k].weight for k in a.rewards)  # stage 0 weights
    assert set(b.curriculum) - set(a.curriculum) == {"heading_off"}


def test_heading_off_curriculum_sets_the_weight_to_zero():
    term = SimpleNamespace(weight=1.0)
    env = SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=lambda k: term))
    cur.heading_off_curriculum(env, None)
    assert term.weight == 0.0
