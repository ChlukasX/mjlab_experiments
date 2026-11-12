"""The `-TapLimit` task (docs/mount_tap_limit): only three training terms differ from `-LegGap`.

GPU-free: the configs load without a simulation, and the curriculum runs on a stand-in env.
"""

from types import SimpleNamespace

import torch

from mjlab_piplus.tasks.ball_mount_min import piplus_ball_mount_taplimit_cfg as tl

PREFIX = "Mjlab-Piplus-"
LEGGAP = PREFIX + "Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap"
TAPLIMIT = LEGGAP + "-TapLimit"


def _cfg(task: str, play: bool = False):
    from mjlab.tasks.registry import load_env_cfg

    import mjlab_piplus.tasks  # noqa: F401  (registers the tasks)

    return load_env_cfg(task, play=play)


def test_training_adds_only_the_three_terms() -> None:
    a, b = _cfg(LEGGAP), _cfg(TAPLIMIT)
    assert list(b.terminations) == list(a.terminations) + ["tap_limit"]
    assert list(b.rewards) == list(a.rewards) + ["early_end"]
    assert list(b.curriculum) == list(a.curriculum) + ["tap_rate"]
    assert b.rewards["early_end"].weight == tl.EARLY_END_PENALTY < 0
    for k in a.rewards:
        assert a.rewards[k].weight == b.rewards[k].weight, k
    for group in ("actor", "critic"):
        assert list(a.observations[group].terms) == list(b.observations[group].terms)
    assert list(a.events) == list(b.events)


def test_play_equals_leggap() -> None:
    a, b = _cfg(LEGGAP, True), _cfg(TAPLIMIT, True)
    assert list(a.terminations) == list(b.terminations)
    assert list(a.rewards) == list(b.rewards)
    assert list(a.curriculum) == list(b.curriculum)


def _env(n: int = 8, stage: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        common_step_counter=1,
        _cur_stage=stage,
        _tap_ended=torch.zeros(n, dtype=torch.bool),
    )


def test_no_budget_before_the_final_stage() -> None:
    env = _env(stage=7)
    tl.tap_rate_curriculum(env, torch.arange(8))
    assert env._tap_rate == tl.NO_LIMIT


def _advance(env: SimpleNamespace, iters: int) -> None:
    env.common_step_counter += iters * tl.STEPS_PER_ITER


def test_warmup_ends_on_survival_after_min_iters() -> None:
    env = _env()
    ids = torch.arange(8)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.NO_LIMIT  # warm-up: no termination
    env._tap_survival = 1.0
    env.reset_time_outs = torch.ones(8, dtype=torch.bool)
    _advance(env, tl.TAP_WARMUP_MIN_ITERS - 1)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.NO_LIMIT  # good survival, but too early
    _advance(env, 1)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.TAP_RATES[0]


def test_warmup_ends_after_max_iters_without_survival() -> None:
    env = _env()
    ids = torch.arange(8)
    env.reset_time_outs = torch.zeros(8, dtype=torch.bool)  # every episode ends early
    tl.tap_rate_curriculum(env, ids)
    _advance(env, tl.TAP_WARMUP_MIN_ITERS)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.NO_LIMIT
    _advance(env, tl.TAP_WARMUP_MAX_ITERS - tl.TAP_WARMUP_MIN_ITERS)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.TAP_RATES[0]


def test_budget_steps_down_on_survival_not_on_the_tap_share() -> None:
    env = _env()
    ids = torch.arange(8)
    tl.tap_rate_curriculum(env, ids)
    _advance(env, tl.TAP_WARMUP_MAX_ITERS)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.TAP_RATES[0]
    # tl1: no episode ends by a tap (they fall), no episode survives: no step down
    env.reset_time_outs = torch.zeros(8, dtype=torch.bool)
    env._tap_share, env._tap_survival = 0.0, 0.0
    _advance(env, tl.TAP_MIN_ITERS)
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.TAP_RATES[0]
    env._tap_survival = tl.TAP_ADVANCE_SURVIVAL  # good survival
    env.reset_time_outs[:] = True
    tl.tap_rate_curriculum(env, ids)
    assert env._tap_rate == tl.TAP_RATES[1] and env._tap_survival == 0.0
    tl.tap_rate_curriculum(env, ids)  # the new level waits TAP_MIN_ITERS
    assert env._tap_rate == tl.TAP_RATES[1]
    for level in range(2, len(tl.TAP_RATES) + 1):
        _advance(env, tl.TAP_MIN_ITERS)
        env._tap_survival = 1.0
        tl.tap_rate_curriculum(env, ids)
        assert env._tap_rate == tl.TAP_RATES[min(level, len(tl.TAP_RATES) - 1)]
    assert env._tap_rate == 0.0


def test_early_end_charges_every_termination_only_while_the_budget_acts() -> None:
    env = _env()
    terminated = torch.zeros(8, dtype=torch.bool)
    terminated[[2, 5]] = True  # for example a fall and a tap
    env.termination_manager = SimpleNamespace(terminated=terminated)
    env._tap_rate = tl.NO_LIMIT  # warm-up
    assert float(tl.early_end(env).abs().sum()) == 0.0
    env._tap_rate = 3
    r = tl.early_end(env) * env.step_dt  # the reward manager multiplies by step_dt
    assert torch.allclose(r, terminated.float())


def test_tap_limit_follows_the_rate_with_the_grace_time(monkeypatch) -> None:
    env = _env(n=4)
    monkeypatch.setattr(tl, "tap_state", lambda *a, **k: None)
    seconds = torch.tensor([0.0, 0.0, 7.0, 7.0])  # time since the mount
    env._tap_since = (seconds / env.step_dt).round().long()
    env._tap_count = torch.tensor([4, 5, 14, 15])
    env._tap_rate = 14.0  # allowed: 1.4 * (t + 3) = 4.2 at t 0, 14 at t 7
    assert tl.tap_limit(env, None, None).tolist() == [False, True, False, True]
    env._tap_rate = 0.0  # the first tap ends the episode
    env._tap_count = torch.tensor([0, 1, 0, 1])
    assert tl.tap_limit(env, None, None).tolist() == [False, True, False, True]
    env._tap_rate = tl.NO_LIMIT  # warm-up
    env._tap_count = torch.tensor([100, 100, 100, 100])
    assert not tl.tap_limit(env, None, None).any()
