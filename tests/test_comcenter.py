"""The `-ComCenter` task (docs/mount_com_center): two reward terms and one curriculum term differ
from `-ActFit`.

GPU-free: the configs load without a simulation, and the terms run on a stand-in env.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_comcenter_cfg as cc

PREFIX = "Mjlab-Piplus-"
ACTFIT = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit"
COMCENTER = ACTFIT + "-ComCenter"
NEW_REWARDS = ["com_center", "com_still", "early_end"]


def _cfg(task: str, play: bool = False):
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return load_env_cfg(task, play=play)


@pytest.mark.parametrize("play", [False, True])
def test_only_the_new_terms_differ(play: bool) -> None:
    a, b = _cfg(ACTFIT, play), _cfg(COMCENTER, play)
    assert list(b.rewards) == list(a.rewards) + NEW_REWARDS
    for k in a.rewards:
        assert a.rewards[k].weight == b.rewards[k].weight, k
    for group in ("actor", "critic"):
        assert list(a.observations[group].terms) == list(b.observations[group].terms)
    assert list(a.events) == list(b.events)
    assert list(a.terminations) == list(b.terminations)
    assert list(b.curriculum) == list(a.curriculum) + ([] if play else ["com_gate"])
    weights = (cc.COM_CENTER_WEIGHT, cc.COM_STILL_WEIGHT) if play else (0.0, 0.0)
    weights += (0.0,)  # early_end is off unless the command line sets it
    assert tuple(b.rewards[k].weight for k in NEW_REWARDS) == weights
    gate = cc.COM_GATE_LEVELS[-1] if play else cc.COM_GATE_LEVELS[0]
    assert b.rewards["single_leg_stance"].params["com_max"] == gate
    assert a.rewards["single_leg_stance"].params["com_max"] == cc.COM_GATE_LEVELS[0]  # base intact


def test_gate_levels_only_tighten() -> None:
    assert list(cc.COM_GATE_LEVELS) == sorted(cc.COM_GATE_LEVELS, reverse=True)
    assert cc.COM_GATE_LEVELS[0] == 0.12  # COM_OVER_BALL: the first level is the -ActFit gate


def test_next_gate_level() -> None:
    ramp, wait, share = cc.STAGE_RAMP_ITERS[cc.FINAL_STAGE], cc.COM_GATE_MIN_ITERS, 0.5
    assert cc.next_gate_level(0, 7, 1e6, share) == 0  # not before the final stage
    assert cc.next_gate_level(0, 8, ramp - 1, share) == 0  # level 0 waits for the stage ramp
    assert cc.next_gate_level(0, 8, ramp, share) == 1
    assert cc.next_gate_level(1, 8, wait - 1, share) == 1
    assert cc.next_gate_level(1, 8, wait, cc.COM_GATE_SHARE - 0.01) == 1  # share too low
    assert cc.next_gate_level(1, 8, wait, share) == 2
    last = len(cc.COM_GATE_LEVELS) - 1
    assert cc.next_gate_level(last, 8, 1e6, 1.0) == last


def test_com_speed_is_a_one_step_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    rel = torch.zeros(2, 2)
    monkeypatch.setattr(cc, "com_to_ball", lambda *a: rel.clone())
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        common_step_counter=5,
        episode_length_buf=torch.tensor([9, 9]),
    )
    assert cc.com_speed(env, None, None).tolist() == [0.0, 0.0]  # no earlier call
    rel[:, 0] = 0.01
    env.common_step_counter, env.episode_length_buf = 6, torch.tensor([10, 1])
    v = cc.com_speed(env, None, None)
    assert v[0].item() == pytest.approx(0.5)  # 0.01 m in 0.02 s
    assert v[1].item() == 0.0  # first step of an episode: the last position is of the old one
    env.common_step_counter = 8  # one call skipped (weight 0)
    assert cc.com_speed(env, None, None).tolist() == [0.0, 0.0]


def _gate_env(stage_t0: int, now: int) -> SimpleNamespace:
    return SimpleNamespace(
        common_step_counter=now,
        _cur_stage=cc.FINAL_STAGE,
        _cur_t0=stage_t0,
        _cur_share=0.0,
        reward_manager=MagicMock(),
    )


def test_gate_defaults_start_at_level_0_with_the_ramp() -> None:
    env = _gate_env(stage_t0=1000, now=1000)
    out = cc.com_gate_curriculum(env, None)
    assert env._cur_t0 == 1000 and env._gate_level == 0
    assert out["com_gate"].item() == pytest.approx(cc.COM_GATE_LEVELS[0])
    assert out["com_weight"].item() == 0.0  # start of the ramp
    assert _cfg(COMCENTER).curriculum["com_gate"].params == {"start_level": 0, "skip_ramp": False}


def test_gate_resume_skips_the_ramp_once_and_keeps_the_level() -> None:
    env = _gate_env(stage_t0=1000, now=1000)
    ramp_steps = cc.STAGE_RAMP_ITERS[cc.FINAL_STAGE] * cc.STEPS_PER_ITER
    for _ in range(2):  # the clock moves once, not at each call
        out = cc.com_gate_curriculum(env, None, start_level=5, skip_ramp=True)
        assert env._cur_t0 == 1000 - ramp_steps
        assert env._gate_level == 5
        assert out["com_gate"].item() == pytest.approx(cc.COM_GATE_LEVELS[5])
        assert out["com_weight"].item() == cc.COM_CENTER_WEIGHT


def test_early_end_is_one_event_per_termination() -> None:
    env = SimpleNamespace(
        step_dt=0.02, termination_manager=SimpleNamespace(terminated=torch.tensor([True, False]))
    )
    assert (cc.early_end(env) * env.step_dt).tolist() == [1.0, 0.0]
