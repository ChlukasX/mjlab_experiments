"""`-ComCenter` task: dense rewards for a centred and still centre of mass on the ball, and a
stance gate that tightens (docs/mount_com_center).

Base: `piplus_ball_mount_actfit_env_cfg` (the `-ActFit` task). The model, the observations, the
actions and the network are the same, so the `af1` actor and critic load. The change is in the
rewards and the curriculum only.

Why: every policy up to `af1` holds the centre of mass (CoM) about 5 cm from the ball centre and
taps the floor with the free foot about 13 times per 10 s. The stance reward is binary and passes
up to COM_OVER_BALL (0.12 m), and `com_over_ball` has weight 0 from stage 3 on, so no term pays a
CoM nearer to the ball centre. On a free ball only a CoM over the ball centre is an equilibrium.

- `com_center`: linear in the horizontal CoM-to-ball-centre distance, with a slope down to 0 m.
- `com_still`: linear in the horizontal speed of the CoM relative to the ball centre.
- `com_gate`: curriculum term. It ramps the two weights in with stage 8 and steps the `com_max`
  of `single_leg_stance` down through COM_GATE_LEVELS on the running stance share.
- `early_end`: cost of a fall, off by default (weight 0). `cc1b` stopped the taps but its survival
  fell from 98% to 77%: a fall cost only the rest of the episode. Set the weight on the command
  line (`--env.rewards.early-end.weight -50`).
"""

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.common.ball_stance import com_to_ball, support_foot
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_actfit_cfg import (
    piplus_ball_mount_actfit_env_cfg,
)
from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mount_curriculum_cfg import (
    FINAL_STAGE,
    STAGE_RAMP_ITERS,
    STEPS_PER_ITER,
    ramped_stage_settings,
)

# Ranges from af1 on its play config (512 envs, on the ball, after 3 s): distance median 0.050 m,
# p90 0.089 m; relative speed median 0.21 m/s, p90 0.44 m/s. Each range is near the p95 of af1, so
# af1 starts with a slope on almost every step.
COM_CENTER_RANGE = 0.10  # m, com_center is 0 from this distance on
COM_SPEED_RANGE = 0.5  # m/s, com_still is 0 from this speed on
COM_CENTER_WEIGHT = 8.0  # per second at distance 0: half of the single_leg_stance weight
COM_STILL_WEIGHT = 4.0  # per second at speed 0
# Stance gate (m): `com_max` of single_leg_stance, and through it of stance_duration. af1 holds
# 81% of its on-ball steps in the stance at 0.12 m, 47% at 0.05 m, 32% at 0.04 m.
COM_GATE_LEVELS = (0.12, 0.10, 0.08, 0.06, 0.05, 0.04)
COM_GATE_MIN_ITERS = 300  # iterations at a level before the next one
COM_GATE_SHARE = 0.4  # running stance share (of the stage curriculum) at which the gate steps down


def _on_ball(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B] bool: a level sole is on the ball."""
    return support_foot(env, foot_cfg, ball_cfg, None, "level").any(dim=-1)


def com_center(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, linear_range: float
) -> torch.Tensor:
    """With a level sole on the ball [B], 0 to 1 per step (positive weight, per second):
    1 - d / `linear_range`, d = horizontal distance (m) from the CoM to the ball centre."""
    d = com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1)
    return _on_ball(env, foot_cfg, ball_cfg).float() * (1.0 - d / linear_range).clamp(0.0, 1.0)


def com_speed(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B] horizontal speed (m/s) of the CoM relative to the ball centre, as a difference over one
    control step. 0 in the first step of an episode and when the last call was not one step ago
    (the reward manager skips a term at weight 0)."""
    rel = com_to_ball(env, foot_cfg, ball_cfg)
    step = env.common_step_counter
    if getattr(env, "_com_rel_step", None) == step - 1:
        v = (rel - env._com_rel).norm(dim=-1) / env.step_dt
        v = torch.where(env.episode_length_buf <= 1, torch.zeros_like(v), v)
    else:
        v = torch.zeros(env.num_envs, device=env.device)
    env._com_rel, env._com_rel_step = rel.clone(), step
    return v


def com_still(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, speed_range: float
) -> torch.Tensor:
    """With a level sole on the ball [B], 0 to 1 per step (positive weight, per second):
    1 - v / `speed_range`, v = `com_speed` (m/s)."""
    v = com_speed(env, foot_cfg, ball_cfg)
    return _on_ball(env, foot_cfg, ball_cfg).float() * (1.0 - v / speed_range).clamp(0.0, 1.0)


def early_end(env) -> torch.Tensor:
    """[B] 1 / step_dt in a step that ends the episode by a termination other than `time_out`,
    else 0. The reward manager multiplies by step_dt, so the weight is the reward per event (give
    it a negative weight)."""
    return env.termination_manager.terminated.float() / env.step_dt


def next_gate_level(level: int, stage: int, iters: float, share: float) -> int:
    """Gate level after this step. A level goes down in the final stage only, after the stage-8
    ramp (level 0) or COM_GATE_MIN_ITERS (later levels) and with a stance share of at least
    COM_GATE_SHARE. `iters`: iterations since the start of the level."""
    wait = STAGE_RAMP_ITERS[FINAL_STAGE] if level == 0 else COM_GATE_MIN_ITERS
    last = level == len(COM_GATE_LEVELS) - 1
    if stage < FINAL_STAGE or last or iters < wait or share < COM_GATE_SHARE:
        return level
    return level + 1


def com_gate_curriculum(
    env, env_ids, start_level: int = 0, skip_ramp: bool = False
) -> dict[str, torch.Tensor]:
    """Set the weights of `com_center` and `com_still` with the ramp of stage 8 and the stance gate
    of its level. It runs after `stage_curriculum`, which stays unchanged (a change in its source
    changes the fingerprints of the old tasks), and reads its stage, clock and stance share. The
    ramp factor is the ramped weight of `support_foot_heading` (0 before stage 8, 1.0 at its end),
    as in `leg_gap_curriculum`.

    For the resume of a run that is past the ramp: `start_level` is the first gate level, and
    `skip_ramp` moves the clock of the final stage back by its ramp length, once, so that the
    stage-8 weights of all curriculum terms are full from the second step on. Without them a
    resume repeats the ramp (the stage-8 weights restart at the stage-7 values) and the gate."""
    stage = getattr(env, "_cur_stage", 0)
    stage_t0 = getattr(env, "_cur_t0", None)
    if not hasattr(env, "_gate_level"):
        env._gate_level, env._gate_t0 = start_level, None
    if env._gate_t0 is None and stage == FINAL_STAGE and stage_t0 is not None:
        if skip_ramp:
            # stage_curriculum reads env._cur_t0 at each step and has no parameter for this.
            stage_t0 = env._cur_t0 = stage_t0 - STAGE_RAMP_ITERS[FINAL_STAGE] * STEPS_PER_ITER
        env._gate_t0 = stage_t0  # level 0 starts with the final stage
    now = env.common_step_counter
    iters = 0.0 if env._gate_t0 is None else (now - env._gate_t0) / STEPS_PER_ITER
    level = next_gate_level(env._gate_level, stage, iters, getattr(env, "_cur_share", 0.0))
    if level != env._gate_level:
        env._gate_level, env._gate_t0 = level, now
    stage_iters = 0.0 if stage_t0 is None else (now - stage_t0) / STEPS_PER_ITER
    f = ramped_stage_settings(stage, stage_iters)["weights"]["support_foot_heading"]
    rm = env.reward_manager
    rm.get_term_cfg("com_center").weight = COM_CENTER_WEIGHT * f
    rm.get_term_cfg("com_still").weight = COM_STILL_WEIGHT * f
    rm.get_term_cfg("single_leg_stance").params["com_max"] = COM_GATE_LEVELS[level]
    return {
        "com_gate": torch.tensor(COM_GATE_LEVELS[level]),
        "com_weight": torch.tensor(COM_CENTER_WEIGHT * f),
    }


def piplus_ball_mount_comcenter_env_cfg(play: bool = False, **kwargs) -> ManagerBasedRlEnvCfg:
    """`-ActFit` task plus the reward terms `com_center` and `com_still` (weight 0 before stage 8),
    the reward term `early_end` (weight 0: off) and the curriculum term `com_gate`. Play: full
    weights and the last gate level. `kwargs` go to `piplus_ball_mount_actfit_env_cfg`."""
    cfg = piplus_ball_mount_actfit_env_cfg(play=play, **kwargs)
    feet = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
    ball = SceneEntityCfg("ball")
    cfg.rewards["com_center"] = RewardTermCfg(
        func=com_center,
        weight=COM_CENTER_WEIGHT if play else 0.0,
        params={"foot_cfg": feet, "ball_cfg": ball, "linear_range": COM_CENTER_RANGE},
    )
    cfg.rewards["com_still"] = RewardTermCfg(
        func=com_still,
        weight=COM_STILL_WEIGHT if play else 0.0,
        params={"foot_cfg": feet, "ball_cfg": ball, "speed_range": COM_SPEED_RANGE},
    )
    cfg.rewards["early_end"] = RewardTermCfg(func=early_end, weight=0.0)
    if play:
        cfg.rewards["single_leg_stance"].params["com_max"] = COM_GATE_LEVELS[-1]
    else:
        cfg.curriculum["com_gate"] = CurriculumTermCfg(
            func=com_gate_curriculum, params={"start_level": 0, "skip_ramp": False}
        )
    return cfg
