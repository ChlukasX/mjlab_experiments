from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mountmin_env_cfg import (
  piplus_ball_mountmin_env_cfg,
  piplus_ball_mountmin_ppo_runner_cfg,
)

# (suffix, bounded actions, locked ball, blind on the ball, latency, big balls only)
for suffix, bounded, locked, blind, latency, big in (
  ("", False, False, False, False, False),
  ("-Clip", True, False, False, False, False),
  ("-Locked", False, True, False, False, False),
  ("-BlindOnBall", True, False, True, False, False),
  ("-BlindOnBall-Latency", True, False, True, True, False),
  ("-BlindOnBall-Latency-Big", True, False, True, True, True),
):
  env_cfg = partial(
    piplus_ball_mountmin_env_cfg, bounded_actions=bounded, locked=locked, blind_on_ball=blind,
    latency=latency, big_ball=big,
  )
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-MountMin{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mountmin_ppo_runner_cfg(suffix.lower().replace("-", "_"), bounded),
  )
