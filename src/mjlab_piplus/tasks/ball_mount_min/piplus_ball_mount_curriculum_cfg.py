"""Mount + stance from a cold start in one run: a staged curriculum (docs/mount_curriculum).

The mount was only ever learned from scratch on an easy task (mm1: no latency, ball visible,
two-foot stance allowed); every robustness feature after it came by a warm start. Cold starts
with everything at once (bl2, st2, st3) never mounted. Here one run walks the same path: all
terms, sensors and inputs exist from step 0, and each stage only changes weights, parameters
and ranges, so the network never changes. A stage advances once the running share of
episode steps in the target stance passes its threshold and the stage has had MIN_ITERS.

Thigh (yaw) joints limited to +-0.6 rad from the start (`straight`; cur1 parked with a foot on
the ball top and never mounted, as st1/st3), or the full range (`straight=False`, -FullThigh).
No start hold, no mount deadline.
"""

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import BALL_SIZES
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mountmin_env_cfg import (
    ANKLE_ROLL_SOFT,
    BIG_BALL_MIN_RADIUS,
    LAG_FINAL,
    LAG_START,
    NOISE_START,
    ON_BALL_BASE_HEIGHT,
    _FEET,
    ankle_roll_excess,
    piplus_ball_mountmin_env_cfg,
    piplus_ball_mountmin_ppo_runner_cfg,
    set_lag,
    set_noise_scale,
    support_foot_heading,
)

FINAL_STAGE = 8
STEPS_PER_ITER = 64  # bounded runner's rollout length
MIN_ITERS = 300  # iterations in a stage before it can advance
# Share of episode steps in the target stance (running mean) needed to advance, per stage.
THRESHOLD = (0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4)
EMA_EPISODES = 2000  # time constant of the running share, in finished episodes
LAG_RAMP_ITERS = 300  # stage 4: lag range ramps from LAG_START to LAG_FINAL
NOISE_RAMP_ITERS = 500  # stage 6: actor noise ramps from NOISE_START to 1x
# Ramp (-Ramp): a stage's new weights and upright std move from the previous stage's values over
# STAGE_RAMP_ITERS[stage] instead of jumping, slowly at first (RAMP_SHAPE: 9% of the way at half
# the ramp, 31% at three quarters). cur2: stage 3 switched action_excess to -4 at once; the policy's
# mean targets lie far past the joint limits (action_excess -90 per second at full weight, against
# +8 for the stance), reward +248 -> -1650 per episode, and it learned to fall at once (episodes
# of 6 steps) and never recovered. In the hand-made chain action_excess came in over three runs
# (-0.2, -1, -4).
STAGE_RAMP_ITERS = (0, 500, 500, 2000, 500, 1000, 500, 500, 500)
RAMP_SHAPE = 100.0  # f = (RAMP_SHAPE**p - 1) / (RAMP_SHAPE - 1), p = progress / ramp length
# Upper bound on the actor's action noise std (-Ramp): cur2's std grew 1 -> 11 over stages 0-2
# (raw actions clipped to +-7, so noise past the clip cost nothing and the entropy bonus paid).
MAX_ACTION_STD = 1.0
NO_LATCH = 1e9  # height threshold that never latches (ball visible until stage 7)
STAGE_NAMES = ("mount", "clear6", "clear10", "smooth", "robust", "posture", "noise", "blind", "freeleg")


# Early action_excess (-Excess): stages 0-2 already pay this weight. cur2's raw outputs drifted far
# past the runner's +-7 clip during stages 0-2 (stage-2 policy: median |a| 45, 70% past the clip),
# where no penalty has a gradient, so the stage-3 penalties could only make it fall (cur3).
EARLY_ACTION_EXCESS = -1.0


def stage_settings(s: int, progress_iters: float, excess0: float = 0.0) -> dict:
    """Everything a stage sets; cumulative (stage s includes all earlier stages). `excess0`:
    action_excess weight of stages 0-2 (0, or EARLY_ACTION_EXCESS for -Excess)."""
    w = {"single_leg_stance": 8.0, "com_over_ball": 2.0, "action_rate": -0.01, "upright": 2.0,
         "action_excess": excess0, "base_yaw_rate": 0.0, "torque_saturation": 0.0, "torque_demand": 0.0,
         "ankle_roll_excess": 0.0, "free_foot_ball_clear": 0.0, "support_hip_vel": 0.0,
         "support_foot_heading": 0.0}
    clearance, upright_std = 0.03, 0.2236
    if s >= 1:
        clearance = 0.06
    if s >= 2:
        clearance, w["single_leg_stance"] = 0.10, 16.0
    if s >= 3:
        w.update(action_excess=-4.0, base_yaw_rate=-0.1, torque_saturation=-2.0, torque_demand=-0.5,
                 action_rate=-0.02, com_over_ball=0.0)
        upright_std = 0.5
    if s >= 5:
        w["ankle_roll_excess"] = -20.0
    if s >= 6:
        w["upright"] = 6.0
    if s >= 8:
        w.update(free_foot_ball_clear=2.0, support_hip_vel=-0.01, support_foot_heading=1.0)
    f_lag = 0.0 if s < 4 else (min(progress_iters / LAG_RAMP_ITERS, 1.0) if s == 4 else 1.0)
    f_noise = 0.0 if s < 6 else (min(progress_iters / NOISE_RAMP_ITERS, 1.0) if s == 6 else 1.0)
    return {
        "weights": w,
        "clearance": clearance,
        "upright_std": upright_std,
        "big_ball": s >= 4,
        "lag": tuple(round(a + f_lag * (b - a)) for a, b in zip(LAG_START, LAG_FINAL)),
        "level": s >= 5,  # free foot off the ball and level support sole
        "noise": NOISE_START + f_noise * (1.0 - NOISE_START),
        "latch": s >= 7,
        "release": s >= 8,
    }


def ramped_stage_settings(s: int, progress_iters: float, excess0: float = 0.0) -> dict:
    """stage_settings, with the weights and upright std of stage s ramped in from stage s-1's
    over STAGE_RAMP_ITERS[s] (the rest, e.g. clearance and switches, change at once)."""
    st = stage_settings(s, progress_iters, excess0)
    R = STAGE_RAMP_ITERS[s]
    if s == 0 or progress_iters >= R:
        return st
    prev = stage_settings(s - 1, progress_iters, excess0)
    f = (RAMP_SHAPE ** (progress_iters / R) - 1.0) / (RAMP_SHAPE - 1.0)
    st["weights"] = {k: prev["weights"][k] + f * (v - prev["weights"][k]) for k, v in st["weights"].items()}
    st["upright_std"] = prev["upright_std"] + f * (st["upright_std"] - prev["upright_std"])
    return st


def _apply_rewards(get, names, st: dict) -> None:
    """Reward weights and params; `get(name)` returns the term cfg (manager's or the cfg's)."""
    for k, v in st["weights"].items():
        get(k).weight = v
    get("single_leg_stance").params["sole_clearance"] = st["clearance"]
    get("upright").params["std"] = st["upright_std"]
    get("posture").params["release"] = st["release"]
    for k in ("single_leg_stance", "free_foot_lift"):
        get(k).params["free_off_ball"] = st["level"]
    for k in names:
        p = get(k).params
        if p.get("flat") in (True, "level"):
            p["flat"] = "level" if st["level"] else True
    get("sole_flat").params["level"] = st["level"]


def stage_curriculum(env, env_ids, start_stage: int = 0, ramp: bool = False, excess0: float = 0.0,
                     late_threshold: float = THRESHOLD[6]):
    """Advance the stage when the running stance share passes THRESHOLD[stage] after MIN_ITERS,
    and apply the stage's settings (ramps inside a stage follow its iteration count)."""
    rm = env.reward_manager
    if not hasattr(env, "_cur_stage"):
        env._cur_stage, env._cur_t0, env._cur_share = start_stage, None, 0.0
    if env._cur_t0 is None and env.common_step_counter > 0:
        # The stage clock starts at the first call during training: the first call runs at env
        # creation, before a resume restores the step counter (cur3, first try: the clock started
        # at 0, so MIN_ITERS and the ramps counted as done at once).
        env._cur_t0 = env.common_step_counter
    t0 = env.common_step_counter if env._cur_t0 is None else env._cur_t0
    # Share of this batch of finished episodes' steps in the target stance (curriculum runs
    # before the reward manager resets its sums).
    if env_ids is not None and len(env_ids):
        scale = env.step_dt if rm._scale_by_dt else 1.0
        n = env.episode_length_buf[env_ids].clamp_min(1)
        share = rm._episode_sums["single_leg_stance"][env_ids] / (rm.get_term_cfg("single_leg_stance").weight * scale) / n
        alpha = 1.0 - (1.0 - 1.0 / EMA_EPISODES) ** len(env_ids)
        env._cur_share += alpha * (float(share.mean()) - env._cur_share)
    iters = (env.common_step_counter - t0) / STEPS_PER_ITER
    s = env._cur_stage
    # With the ramp, a stage also waits for its ramp to finish: the next stage ramps from this
    # stage's full values, so advancing mid-ramp would be a jump again.
    min_iters = max(MIN_ITERS, STAGE_RAMP_ITERS[s]) if ramp else MIN_ITERS
    # `late_threshold`: stages 6-7 (full noise, blind on the ball); cur4 levelled off at a share of
    # ~0.35 in stage 6 against 0.4 and was resumed at stage 7 with 0.3.
    # (cur4b crashed indexing THRESHOLD[8] right after advancing to the final stage, which has none.)
    threshold = late_threshold if s in (6, 7) else THRESHOLD[s] if s < FINAL_STAGE else None
    if threshold is not None and iters >= min_iters and env._cur_share > threshold:
        env._cur_stage, env._cur_t0, env._cur_share, iters = s + 1, env.common_step_counter, 0.0, 0.0
        s += 1
    st = (ramped_stage_settings if ramp else stage_settings)(s, iters, excess0)
    _apply_rewards(rm.get_term_cfg, rm.active_terms, st)
    set_lag(env, *st["lag"])
    set_noise_scale(env, st["noise"])
    env.observation_manager.get_term_cfg("actor", "ball_pos").params["height_threshold"] = (
        ON_BALL_BASE_HEIGHT if st["latch"] else NO_LATCH)
    if st["big_ball"] and env_ids is not None and len(env_ids):
        # Ball size is a startup event: from stage 4 on, resample it (and the mass) per reset
        # from the big-ball range.
        em = env.event_manager
        for k in ("ball_size", "ball_mass_from_size"):
            ev = em.get_term_cfg(k)
            if k == "ball_size":
                ev.params["ranges"] = (BIG_BALL_MIN_RADIUS / BALL_SIZES[5][0], 1.0)
            ev.func(env, env_ids, **ev.params)
    return {"stage": torch.tensor(float(s)), "stance_share": torch.tensor(env._cur_share),
            "lag_max": torch.tensor(float(st["lag"][1])), "noise_scale": torch.tensor(st["noise"])}


def curriculum_stage_obs(env) -> torch.Tensor:
    """Critic: curriculum stage / FINAL_STAGE [B, 1] (play: the final stage)."""
    s = getattr(env, "_cur_stage", FINAL_STAGE)
    return torch.full((env.num_envs, 1), s / FINAL_STAGE, device=env.device)


def piplus_ball_mount_curriculum_env_cfg(play: bool = False, start_stage: int = 0, straight: bool = True,
                                        ramp: bool = False, excess0: float = 0.0) -> ManagerBasedRlEnvCfg:
    cfg = piplus_ball_mountmin_env_cfg(play, bounded_actions=True, blind_on_ball=True, latency=True,
        one_foot=True, latch_fix=True, kin_ball=True, free_leg=True, straight=straight, gate=True)
    r = cfg.rewards
    if not straight:  # full thigh range: the heading term (stage 8) still asks to untwist the support foot
        r["support_foot_heading"] = RewardTermCfg(func=support_foot_heading, weight=0.0, params={
            "foot_cfg": SceneEntityCfg("robot", body_names=_FEET), "ball_cfg": SceneEntityCfg("ball"), "gated": True})
    cfg.curriculum.pop("latency", None)  # the stage curriculum sets the lag
    r["ankle_roll_excess"] = RewardTermCfg(func=ankle_roll_excess, weight=0.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_roll_joint",)), "limit": ANKLE_ROLL_SOFT})
    r["sole_flat"].params = {**r["sole_flat"].params, "level": False}  # was shared with foot_lift
    r["posture"].params["release"] = False
    cfg.observations["critic"].terms["curriculum_stage"] = ObservationTermCfg(func=curriculum_stage_obs)
    if play:
        # Final stage, static: eval and play set lag and noise themselves (latency=True sizes
        # the play actuators for the final 10-40 ms).
        st = stage_settings(FINAL_STAGE, 0, excess0)
        _apply_rewards(r.__getitem__, list(r), st)
        cfg.events["ball_size"].params["ranges"] = (BIG_BALL_MIN_RADIUS / BALL_SIZES[5][0], 1.0)
        cfg.observations["actor"].terms["ball_pos"].params["height_threshold"] = ON_BALL_BASE_HEIGHT
    else:
        st = stage_settings(start_stage, 0, excess0)
        _apply_rewards(r.__getitem__, list(r), st)
        cfg.observations["actor"].terms["ball_pos"].params["height_threshold"] = (
            ON_BALL_BASE_HEIGHT if st["latch"] else NO_LATCH)
        cfg.curriculum["stage"] = CurriculumTermCfg(func=stage_curriculum, params={"start_stage": start_stage, "ramp": ramp, "excess0": excess0,
            "late_threshold": THRESHOLD[6]})
    return cfg


def piplus_ball_mount_curriculum_ppo_runner_cfg(suffix: str = "", max_std: float | None = None):
    cfg = piplus_ball_mountmin_ppo_runner_cfg("_curriculum" + suffix, bounded_actions=True)
    if max_std is not None:  # -Ramp: clamp the action noise std (see MAX_ACTION_STD)
        cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "std_range": (1e-6, max_std)}
    return cfg
