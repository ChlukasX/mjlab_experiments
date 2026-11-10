from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mount_curriculum_cfg import (
  piplus_ball_mount_curriculum_env_cfg,
  EARLY_ACTION_EXCESS,
  MAX_ACTION_STD,
  MID_THIGH_LIMIT,
  piplus_ball_mount_curriculum_ppo_runner_cfg,
)
from .piplus_ball_mount_imunoise_cfg import piplus_ball_mount_imunoise_env_cfg
from .piplus_ball_mount_selfcontact_cfg import piplus_ball_mount_selfcontact_env_cfg
from .piplus_ball_mountmin_env_cfg import (
  piplus_ball_mountmin_env_cfg,
  piplus_ball_mountmin_ppo_runner_cfg,
)

# Each variant adds to the one before it (see piplus_ball_mountmin_env_cfg for the flags).
_BLIND = dict(bounded_actions=True, blind_on_ball=True)
_LAT = dict(_BLIND, latency=True)
_BIG = dict(_LAT, big_ball=True)
_ONE = dict(_BIG, one_foot=True)
_LEVEL = dict(_ONE, level=True)
_RAMP = dict(_LEVEL, noise_ramp=True)
for suffix, kw in (
  ("", {}),
  ("-Clip", dict(bounded_actions=True)),
  ("-Locked", dict(locked=True)),
  ("-BlindOnBall", _BLIND),
  ("-BlindOnBall-Latency", _LAT),
  ("-BlindOnBall-Latency-Big", _BIG),
  ("-BlindOnBall-Latency-Big-OneFoot", _ONE),
  ("-BlindOnBall-Latency-Big-OneFoot-Level", _LEVEL),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp", _RAMP),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold", dict(_RAMP, latch_fix=True, start_hold=True)),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall", dict(_RAMP, latch_fix=True, start_hold=True, kin_ball=True)),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg", dict(_RAMP, latch_fix=True, start_hold=True, kin_ball=True, free_leg=True)),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg-Straight", dict(_RAMP, latch_fix=True, start_hold=True, kin_ball=True, free_leg=True, straight=True)),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg-Straight-Gate-Deadline", dict(_RAMP, latch_fix=True, start_hold=True, kin_ball=True, free_leg=True, straight=True, gate=True, deadline=True)),
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-Deadline", dict(_RAMP, latch_fix=True, start_hold=True, deadline=True)),
):
  env_cfg = partial(piplus_ball_mountmin_env_cfg, **kw)
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-MountMin{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mountmin_ppo_runner_cfg(suffix.lower().replace("-", "_"), kw.get("bounded_actions", False)),
  )

# One cold start through all stages (mount first, then the robustness features); docs/mount_curriculum.
register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-MountMin-Curriculum",
  env_cfg=piplus_ball_mount_curriculum_env_cfg(),
  play_env_cfg=piplus_ball_mount_curriculum_env_cfg(play=True),
  rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg(),
)
# Same with the full thigh range (cur1 with +-0.6 rad never mounted).
register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh",
  env_cfg=piplus_ball_mount_curriculum_env_cfg(straight=False),
  play_env_cfg=piplus_ball_mount_curriculum_env_cfg(play=True, straight=False),
  rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh"),
)
# Same, with the stage weights ramped in and the action noise std capped (cur2 collapsed at stage 3).
register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp",
  env_cfg=piplus_ball_mount_curriculum_env_cfg(straight=False, ramp=True),
  play_env_cfg=piplus_ball_mount_curriculum_env_cfg(play=True, straight=False, ramp=True),
  rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp", max_std=MAX_ACTION_STD),
)
# Same, with action_excess from stage 0 (cur2's outputs drifted past the clip, cur3 collapsed).
register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess",
  env_cfg=piplus_ball_mount_curriculum_env_cfg(straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS),
  play_env_cfg=piplus_ball_mount_curriculum_env_cfg(play=True, straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS),
  rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp_excess", max_std=MAX_ACTION_STD),
)
# Same, with the tap penalised (free sole on the floor while on the ball) and a stance-duration bonus
# (cur4c taps: 14.8 floor taps per 10 s, stances of 0.36 s). Reward change only; warm start from cur4c.
register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap",
    env_cfg=piplus_ball_mount_curriculum_env_cfg(straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True),
    play_env_cfg=piplus_ball_mount_curriculum_env_cfg(
        play=True, straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True),
    rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp_excess_notap", max_std=MAX_ACTION_STD),
)
# Same, with the thigh limited to +-1.2 rad and no heading reward: a support foot turned by up to
# ~69 deg is allowed, not paid (cur6 stays straight and taps). Warm start from cur4c.
register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-MidThigh",
    env_cfg=piplus_ball_mount_curriculum_env_cfg(
        straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True, thigh_limit=MID_THIGH_LIMIT),
    play_env_cfg=piplus_ball_mount_curriculum_env_cfg(
        play=True, straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True, thigh_limit=MID_THIGH_LIMIT),
    rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp_excess_notap_midthigh", max_std=MAX_ACTION_STD),
)
# Same as -NoTap (full thigh range), plus a reward for a support foot turned by ~100 deg from the
# torso (kb1 +109 deg holds the stance; cur6 and cur7 stay at -20 deg and tap). Warm start from cur4c.
register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-Turn",
    env_cfg=piplus_ball_mount_curriculum_env_cfg(
        straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True, turn=True),
    play_env_cfg=piplus_ball_mount_curriculum_env_cfg(
        play=True, straight=False, ramp=True, excess0=EARLY_ACTION_EXCESS, notap=True, turn=True),
    rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp_excess_notap_turn", max_std=MAX_ACTION_STD),
)
# Same as -NoTap (full thigh range), plus a penalty for a right leg that is close to the left leg
# (term leg_gap from stage 8: a GPU distance between the leg collision primitives; docs/mount_selfcontact).
register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap",
    env_cfg=piplus_ball_mount_selfcontact_env_cfg(),
    play_env_cfg=piplus_ball_mount_selfcontact_env_cfg(play=True),
    rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp_excess_notap_leggap", max_std=MAX_ACTION_STD),
)
# Same as -LegGap, with half the actor noise on the two IMU inputs (projected gravity, angular
# velocity); the IMU is better than the mjlab default assumes. Warm start from cur4c; docs/mount_imu_noise.
register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise",
    env_cfg=piplus_ball_mount_imunoise_env_cfg(),
    play_env_cfg=piplus_ball_mount_imunoise_env_cfg(play=True),
    rl_cfg=piplus_ball_mount_curriculum_ppo_runner_cfg("_fullthigh_ramp_excess_notap_leggap_imunoise", max_std=MAX_ACTION_STD),
)
