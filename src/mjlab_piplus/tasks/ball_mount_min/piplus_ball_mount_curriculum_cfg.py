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

from dataclasses import replace
from functools import partial

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.robot.piplus_constants import get_spec_with_arms
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    BALL_SIZES,
    _ball_contact,
    _sole_clearance,
    _support_foot,
)
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mountmin_env_cfg import (
    ANKLE_ROLL_SOFT,
    BIG_BALL_MIN_RADIUS,
    LAG_FINAL,
    LAG_START,
    NOISE_START,
    ON_BALL_BASE_HEIGHT,
    _FEET,
    _THIGHS,
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

# Stop the tap (-NoTap, docs/mount_curriculum). cur4c taps: the free sole touches the floor 14.8
# times per 10 s and a stance lasts 0.36 s. Both terms act from stage 8 and ramp in with it.
TAP_PENALTY = -8.0  # weight of free_foot_contact, per second of free-foot contact with the floor or the ball
STANCE_BONUS = 16.0  # weight of stance_duration, per second at the full bonus (the stance weight)
STANCE_BONUS_CAP_S = 2.0  # s of one unbroken stance after which the bonus stops growing
FLOOR_TOUCH_CLEARANCE = 0.01  # m, sole corner above the floor: the tap test of eval_mountbalance --gait


def free_foot_contact(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Free-foot indicator [B] (0 or 1 per step; give it a negative weight, per second). 1 while the
    robot is latched on the ball and either (a) a sole's lowest corner is within
    FLOOR_TOUCH_CLEARANCE (m) of the floor (the foot on the floor is never the support foot), or
    (b) exactly one foot is the (flat) support foot and the other foot touches the ball."""
    latched = getattr(env, "_on_ball_latch", None)
    if latched is None:
        return torch.zeros(env.num_envs, device=env.device)
    floor = (_sole_clearance(env, foot_cfg) < FLOOR_TOUCH_CLEARANCE).any(dim=-1)
    sup = _support_foot(env, foot_cfg, ball_cfg, None, True)
    on_ball = (sup.sum(dim=-1) == 1) & (_ball_contact(env) & ~sup).any(dim=-1)
    return (latched & (floor | on_ball)).float()


def run_length(prev: torch.Tensor, active: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    """Steps [B] that `active` has held without a break: prev + 1 while active, else 0. `fresh`
    (first step of an episode) starts the count from 0."""
    return (torch.where(fresh, torch.zeros_like(prev), prev) + 1.0) * active


def stance_duration(env, cap_s: float) -> torch.Tensor:
    """Stance bonus [B] in [0, 1] per step (give it a positive weight, per second): the length of
    the current unbroken stance divided by `cap_s` (s), at most 1. The stance is the live
    `single_leg_stance` term (same params as in the stage). The count is kept in `env._stance_len`
    (steps) and restarts with each episode. The reward manager skips a term at weight 0, so the
    count can be stale for the first steps of the ramp; the first step without a stance resets it."""
    cfg = env.reward_manager.get_term_cfg("single_leg_stance")
    stance = cfg.func(env, **cfg.params) > 0
    prev = getattr(env, "_stance_len", None)
    if prev is None:
        prev = torch.zeros(env.num_envs, device=env.device)
    env._stance_len = run_length(prev, stance, env.episode_length_buf <= 1)
    return (env._stance_len * env.step_dt / cap_s).clamp(max=1.0)


def stage_settings(s: int, progress_iters: float, excess0: float = 0.0, notap: bool = False) -> dict:
    """Everything a stage sets; cumulative (stage s includes all earlier stages). `excess0`:
    action_excess weight of stages 0-2 (0, or EARLY_ACTION_EXCESS for -Excess). `notap`: also the
    weights of free_foot_contact and stance_duration (0 before stage 8)."""
    w = {"single_leg_stance": 8.0, "com_over_ball": 2.0, "action_rate": -0.01, "upright": 2.0,
         "action_excess": excess0, "base_yaw_rate": 0.0, "torque_saturation": 0.0, "torque_demand": 0.0,
         "ankle_roll_excess": 0.0, "free_foot_ball_clear": 0.0, "support_hip_vel": 0.0,
         "support_foot_heading": 0.0}
    if notap:
        w.update(free_foot_contact=0.0, stance_duration=0.0)
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
        if notap:
            w.update(free_foot_contact=TAP_PENALTY, stance_duration=STANCE_BONUS)
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


def ramped_stage_settings(s: int, progress_iters: float, excess0: float = 0.0, notap: bool = False) -> dict:
    """stage_settings, with the weights and upright std of stage s ramped in from stage s-1's
    over STAGE_RAMP_ITERS[s] (the rest, e.g. clearance and switches, change at once)."""
    st = stage_settings(s, progress_iters, excess0, notap)
    R = STAGE_RAMP_ITERS[s]
    if s == 0 or progress_iters >= R:
        return st
    prev = stage_settings(s - 1, progress_iters, excess0, notap)
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


NOTAP_TERMS = ("free_foot_contact", "stance_duration")


def notap_curriculum(env, env_ids) -> dict[str, torch.Tensor]:
    """Set the weights of the NOTAP_TERMS with the ramp of the current stage. It runs after
    `stage_curriculum` (same dict, later key) and reads its stage and clock. `stage_curriculum`
    itself is not changed: a change in its source changes the fingerprint of the old tasks."""
    t0 = env.common_step_counter if getattr(env, "_cur_t0", None) is None else env._cur_t0
    iters = (env.common_step_counter - t0) / STEPS_PER_ITER
    w = ramped_stage_settings(getattr(env, "_cur_stage", 0), iters, notap=True)["weights"]
    for k in NOTAP_TERMS:
        env.reward_manager.get_term_cfg(k).weight = w[k]
    return {k: torch.tensor(w[k]) for k in NOTAP_TERMS}


# Middle thigh range (-MidThigh, docs/mount_curriculum). cur6 has the full thigh range, but the
# heading term (stage 8) pays for a straight support foot, and the policy stays straight (-20 deg).
# Older policies hold a stance better with the support foot turned by 109-138 deg. This task
# limits the thigh joint to MID_THIGH_LIMIT in the model and in the action clip, and sets the
# heading weight to 0 so that the turn is allowed, but not paid.
MID_THIGH_LIMIT = 1.2  # rad, about 69 deg


def _spec_thigh(limit: float):
    spec = get_spec_with_arms()
    for j in spec.joints:
        if j.name in _THIGHS:
            j.range = [-limit, limit]
    return spec


def heading_off_curriculum(env, env_ids) -> dict[str, torch.Tensor]:
    """Set the weight of support_foot_heading to 0. It runs after `stage_curriculum`, which sets
    the stage weight (1.0 from stage 8) in the same step."""
    env.reward_manager.get_term_cfg("support_foot_heading").weight = 0.0
    return {"heading_weight": torch.tensor(0.0)}


def curriculum_stage_obs(env) -> torch.Tensor:
    """Critic: curriculum stage / FINAL_STAGE [B, 1] (play: the final stage)."""
    s = getattr(env, "_cur_stage", FINAL_STAGE)
    return torch.full((env.num_envs, 1), s / FINAL_STAGE, device=env.device)


def piplus_ball_mount_curriculum_env_cfg(play: bool = False, start_stage: int = 0, straight: bool = True,
                                        ramp: bool = False, excess0: float = 0.0,
                                        notap: bool = False,
                                        thigh_limit: float | None = None) -> ManagerBasedRlEnvCfg:
    """`notap` (-NoTap): add free_foot_contact and stance_duration (weights 0 before stage 8).
    `thigh_limit` (-MidThigh, needs `straight=False`): thigh range +-thigh_limit (rad) in the model
    and the action clip; the heading weight stays 0."""
    cfg = piplus_ball_mountmin_env_cfg(play, bounded_actions=True, blind_on_ball=True, latency=True,
        one_foot=True, latch_fix=True, kin_ball=True, free_leg=True, straight=straight, gate=True)
    r = cfg.rewards
    if not straight:  # full thigh range: the heading term (stage 8) still asks to untwist the support foot
        r["support_foot_heading"] = RewardTermCfg(func=support_foot_heading, weight=0.0, params={
            "foot_cfg": SceneEntityCfg("robot", body_names=_FEET), "ball_cfg": SceneEntityCfg("ball"), "gated": True})
    if thigh_limit is not None:
        assert not straight, "thigh_limit replaces the +-0.6 rad limit of straight"
        cfg.scene.entities["robot"] = replace(cfg.scene.entities["robot"], spec_fn=partial(_spec_thigh, thigh_limit))
        clip = cfg.actions["joint_pos"].clip
        cfg.actions["joint_pos"].clip = {**clip, **{n: (-thigh_limit, thigh_limit) for n in _THIGHS}}
    cfg.curriculum.pop("latency", None)  # the stage curriculum sets the lag
    r["ankle_roll_excess"] = RewardTermCfg(func=ankle_roll_excess, weight=0.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_roll_joint",)), "limit": ANKLE_ROLL_SOFT})
    r["sole_flat"].params = {**r["sole_flat"].params, "level": False}  # was shared with foot_lift
    r["posture"].params["release"] = False
    cfg.observations["critic"].terms["curriculum_stage"] = ObservationTermCfg(func=curriculum_stage_obs)
    if notap:
        r["free_foot_contact"] = RewardTermCfg(func=free_foot_contact, weight=0.0, params={
            "foot_cfg": SceneEntityCfg("robot", body_names=_FEET), "ball_cfg": SceneEntityCfg("ball")})
        r["stance_duration"] = RewardTermCfg(func=stance_duration, weight=0.0, params={"cap_s": STANCE_BONUS_CAP_S})
    if play:
        # Final stage, static: eval and play set lag and noise themselves (latency=True sizes
        # the play actuators for the final 10-40 ms).
        st = stage_settings(FINAL_STAGE, 0, excess0, notap)
        _apply_rewards(r.__getitem__, list(r), st)
        if thigh_limit is not None:
            r["support_foot_heading"].weight = 0.0
        cfg.events["ball_size"].params["ranges"] = (BIG_BALL_MIN_RADIUS / BALL_SIZES[5][0], 1.0)
        cfg.observations["actor"].terms["ball_pos"].params["height_threshold"] = ON_BALL_BASE_HEIGHT
    else:
        st = stage_settings(start_stage, 0, excess0, notap)
        _apply_rewards(r.__getitem__, list(r), st)
        cfg.observations["actor"].terms["ball_pos"].params["height_threshold"] = (
            ON_BALL_BASE_HEIGHT if st["latch"] else NO_LATCH)
        cfg.curriculum["stage"] = CurriculumTermCfg(func=stage_curriculum, params={"start_stage": start_stage, "ramp": ramp, "excess0": excess0,
            "late_threshold": THRESHOLD[6]})
        if notap:
            cfg.curriculum["notap"] = CurriculumTermCfg(func=notap_curriculum)
        if thigh_limit is not None:
            cfg.curriculum["heading_off"] = CurriculumTermCfg(func=heading_off_curriculum)
    return cfg


def piplus_ball_mount_curriculum_ppo_runner_cfg(suffix: str = "", max_std: float | None = None):
    cfg = piplus_ball_mountmin_ppo_runner_cfg("_curriculum" + suffix, bounded_actions=True)
    if max_std is not None:  # -Ramp: clamp the action noise std (see MAX_ACTION_STD)
        cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "std_range": (1e-6, max_std)}
    return cfg
