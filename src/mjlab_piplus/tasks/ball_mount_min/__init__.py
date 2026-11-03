from functools import partial

from mjlab.tasks.registry import register_mjlab_task

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
  ("-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-Deadline", dict(_RAMP, latch_fix=True, start_hold=True, deadline=True)),
):
  env_cfg = partial(piplus_ball_mountmin_env_cfg, **kw)
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-MountMin{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mountmin_ppo_runner_cfg(suffix.lower().replace("-", "_"), kw.get("bounded_actions", False)),
  )
