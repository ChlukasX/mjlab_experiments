"""`-TapLimit` task: the episode ends when the free foot taps the floor too often after the first
mount (docs/mount_tap_limit).

Base: `piplus_ball_mount_selfcontact_env_cfg` (the `-LegGap` task). The model, the observations,
the actions, and the network are the same, so the `cur4c` actor and critic load.

The best policies tap the floor about 13 times per 10 s (`sg1-leggap`, `im1-imunoise`). The tap
penalty of `-NoTap` (`free_foot_contact`, -8 per second of contact) costs less than 1 per tap, so a
tap stays cheap. Here a tap after the first mount ends the episode once the episode taps faster
than the tap rate limit. The limit goes down to 0, so the policy must then get the balance right on
the first mount.

Mount: the first step of the episode in a gated single support (`single_support` with
`gated=True`: one level sole on the ball, CoM within COM_OVER_BALL of the ball, the free sole more
than GATE_FREE_CLEARANCE above the floor). Tap: after the mount, a step where a sole corner comes
within FLOOR_TOUCH_CLEARANCE of the floor and was not within it the step before (the tap test of
`eval_mountbalance.py --gait`). After the mount the support foot is on the ball, so the foot on
the floor is the free foot.

Tap rate limit: after the mount, the episode may have at most rate / 10 * (t + TAP_GRACE_S) taps,
with rate in taps per 10 s and t the time since the mount (s). The tap above that number ends the
episode. The grace time keeps one early tap from ending the episode. At rate 0 the first tap ends
the episode. The limit acts only in the final stage (stage 8) of the stage curriculum. Stage 8
starts with a warm-up without the termination. The warm-up ends when it has had
TAP_WARMUP_MIN_ITERS and the running survival (share of finished episodes that end by `time_out`)
is at least TAP_WARMUP_SURVIVAL, or at the latest after TAP_WARMUP_MAX_ITERS. Then the rate starts
at TAP_RATES[0] (14, about the tap rate of the policies) and goes down to 0. It goes down when the
level has had TAP_MIN_ITERS and the running survival is at least TAP_ADVANCE_SURVIVAL. The actor
and the critic do not see the rate or the tap count.

Why a rate (tl4) and not a fixed number of taps per episode (tl1-tl3): a time-out needs 20 s with
at most the budget of taps. With budget 6 (tl2) and 10 (tl3) no episode reached the time-out (about
5 and 6 s per episode), so the survival never rose and the budget never went down. The first rate
fits the current policy, so the survival starts above 0.

Early end: while the limit acts, every termination except `time_out` costs EARLY_END_PENALTY
(reward `early_end`), a fall as much as a tap. tl1 had the penalty on `tap_limit` only and stepped
the budget down on the share of episodes that ended by `tap_limit`: the policy learned to fall
before its taps ran out (`fell_over` 43 per iteration, episode 1.7 s), and the falls also pushed
the tap share below the threshold, so the budget went down to 0. Survival cannot be gamed by a
fall.
"""

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_piplus.common.ball_stance import FLOOR_TOUCH_CLEARANCE, sole_clearance
from mjlab_piplus.common.rewards.stance import single_support
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_curriculum_cfg import (
    FINAL_STAGE,
    STEPS_PER_ITER,
)
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_selfcontact_cfg import (
    piplus_ball_mount_selfcontact_env_cfg,
)
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mountmin_env_cfg import _FEET

# Warm-up in stage 8 without the termination: the stage-8 rewards ramp in over 500 iterations
# (STAGE_RAMP_ITERS[8]), and the smoke test with budget 3 from the first step cut 85% of the
# episodes to about 2.8 s (taplimit-smoke).
TAP_WARMUP_MIN_ITERS = 500  # iterations of warm-up before the survival test can end it
TAP_WARMUP_SURVIVAL = 0.7  # running share of finished episodes that end by time_out
TAP_WARMUP_MAX_ITERS = 1500  # the warm-up ends after this many iterations in any case
# Tap rate limit (taps per 10 s after the mount), one entry per level. The policies tap 12-13.5
# times per 10 s (eval of sg1-leggap and tl2).
TAP_RATES = (14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0)
TAP_GRACE_S = 3.0  # s added to the time since the mount: at rate 14, 4 taps right after the mount
TAP_MIN_ITERS = 300  # iterations at one level before it can go down
# Running survival at which the rate goes down. The warm-up of tl1 (no tap limit) held 0.5.
TAP_ADVANCE_SURVIVAL = 0.3
TAP_EMA_EPISODES = 2000  # time constant of the running share, in finished episodes
EARLY_END_PENALTY = -50.0  # reward per termination other than time_out (one event, not per second)
NO_LIMIT = float("inf")  # rate before the final stage and in the warm-up: no termination


def tap_state(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> None:
    """Update the per-episode tap state on `env`: `_tap_mounted` [B] bool (gated single support
    reached once), `_tap_count` [B] (taps after the mount), `_tap_since` [B] (steps since the
    mount), `_tap_floor` [B] bool (a sole near the floor in the last step). A new episode
    (`episode_length_buf <= 1`) starts from zero."""
    if not hasattr(env, "_tap_count"):
        env._tap_mounted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._tap_floor = torch.zeros_like(env._tap_mounted)
        env._tap_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._tap_since = torch.zeros_like(env._tap_count)
    fresh = env.episode_length_buf <= 1
    env._tap_mounted &= ~fresh
    env._tap_floor &= ~fresh
    env._tap_count[fresh] = 0
    env._tap_since[fresh] = 0
    stance, _ = single_support(env, foot_cfg, ball_cfg, gated=True)
    floor = (sole_clearance(env, foot_cfg) < FLOOR_TOUCH_CLEARANCE).any(dim=-1)
    env._tap_count += (env._tap_mounted & floor & ~env._tap_floor).long()
    env._tap_since += env._tap_mounted.long()
    env._tap_mounted |= stance
    env._tap_floor = floor


def tap_limit(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B] bool: the episode has more taps after the first mount than the rate limit allows,
    rate / 10 * (t + TAP_GRACE_S) with t the time since the mount (s) and rate
    `env._tap_rate` (taps per 10 s, NO_LIMIT until the curriculum sets it). Also stores the result
    in `env._tap_ended` for the curriculum."""
    tap_state(env, foot_cfg, ball_cfg)
    rate = getattr(env, "_tap_rate", NO_LIMIT)
    if rate == NO_LIMIT:
        env._tap_ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return env._tap_ended
    allowed = rate / 10.0 * (env._tap_since * env.step_dt + TAP_GRACE_S)
    env._tap_ended = env._tap_count > allowed
    return env._tap_ended


def early_end(env) -> torch.Tensor:
    """[B] 1 / step_dt in a step that ends the episode by a termination other than `time_out`,
    while the tap rate limit acts (after the warm-up of stage 8), else 0. The reward manager
    multiplies by step_dt, so the weight is the reward per event (give it a negative weight)."""
    if getattr(env, "_tap_rate", NO_LIMIT) == NO_LIMIT:
        return torch.zeros(env.num_envs, device=env.device)
    return env.termination_manager.terminated.float() / env.step_dt


def _ema(old: float, new: torch.Tensor, n: int) -> float:
    """Running mean over TAP_EMA_EPISODES finished episodes, updated with a batch of `n`."""
    alpha = 1.0 - (1.0 - 1.0 / TAP_EMA_EPISODES) ** n
    return old + alpha * (float(new.float().mean()) - old)


def tap_rate_curriculum(env, env_ids) -> dict[str, torch.Tensor]:
    """Set `env._tap_rate`: NO_LIMIT before the final stage and during the warm-up, then
    TAP_RATES[level]. The warm-up ends after TAP_WARMUP_MIN_ITERS once the running survival is
    at least TAP_WARMUP_SURVIVAL, or after TAP_WARMUP_MAX_ITERS. A level goes up when it has had
    TAP_MIN_ITERS and the running survival is at least TAP_ADVANCE_SURVIVAL. It runs after
    `stage_curriculum` and reads its stage. It runs before the termination manager resets, so
    `env._tap_ended` and `env.reset_time_outs` hold the last step. Level -1 is the warm-up."""
    if not hasattr(env, "_tap_level"):
        env._tap_level, env._tap_t0, env._tap_share = -1, None, 1.0
        env._tap_survival, env._tap_mount_share = 0.0, 0.0
    if getattr(env, "_cur_stage", 0) < FINAL_STAGE or env.common_step_counter == 0:
        env._tap_rate = NO_LIMIT
        zero = torch.tensor(0.0)
        return {
            "tap_rate": torch.tensor(-1.0),
            "tap_end_share": zero,
            "survival": zero,
            "mount_share": zero,
        }
    if env._tap_t0 is None:  # the level clock starts with the final stage (or the resume)
        env._tap_t0 = env.common_step_counter
    if env_ids is not None and len(env_ids):
        n = len(env_ids)
        ended = getattr(env, "_tap_ended", None)
        if ended is not None:
            env._tap_share = _ema(env._tap_share, ended[env_ids], n)
        time_outs = getattr(env, "reset_time_outs", None)
        if time_outs is not None:
            env._tap_survival = _ema(env._tap_survival, time_outs[env_ids], n)
        mounted = getattr(env, "_tap_mounted", None)
        if mounted is not None:
            env._tap_mount_share = _ema(env._tap_mount_share, mounted[env_ids], n)
    iters = (env.common_step_counter - env._tap_t0) / STEPS_PER_ITER
    if env._tap_level < 0:
        good = iters >= TAP_WARMUP_MIN_ITERS and env._tap_survival >= TAP_WARMUP_SURVIVAL
        advance = good or iters >= TAP_WARMUP_MAX_ITERS
    else:
        advance = (
            env._tap_level < len(TAP_RATES) - 1
            and iters >= TAP_MIN_ITERS
            and env._tap_survival >= TAP_ADVANCE_SURVIVAL
        )
    if advance:
        # The running means restart for the new level, so the next level waits for real data.
        env._tap_level += 1
        env._tap_t0, env._tap_share, env._tap_survival = env.common_step_counter, 1.0, 0.0
    env._tap_rate = NO_LIMIT if env._tap_level < 0 else TAP_RATES[env._tap_level]
    return {
        "tap_rate": torch.tensor(env._tap_rate if env._tap_level >= 0 else -1.0),
        "tap_end_share": torch.tensor(env._tap_share),
        "survival": torch.tensor(env._tap_survival),
        "mount_share": torch.tensor(env._tap_mount_share),
    }


def piplus_ball_mount_taplimit_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """`-LegGap` task plus the termination `tap_limit`, the reward `early_end`, and the curriculum
    term `tap_rate`. `kwargs` go to `piplus_ball_mount_selfcontact_env_cfg`. Play has none of
    the three: play and eval keep the episode to its end, as in `-LegGap`."""
    cfg = piplus_ball_mount_selfcontact_env_cfg(play=play, **kwargs)
    if play:
        return cfg
    feet, ball = SceneEntityCfg("robot", body_names=_FEET), SceneEntityCfg("ball")
    cfg.terminations["tap_limit"] = TerminationTermCfg(
        func=tap_limit, params={"foot_cfg": feet, "ball_cfg": ball}
    )
    cfg.rewards["early_end"] = RewardTermCfg(func=early_end, weight=EARLY_END_PENALTY)
    cfg.curriculum["tap_rate"] = CurriculumTermCfg(func=tap_rate_curriculum)
    return cfg
