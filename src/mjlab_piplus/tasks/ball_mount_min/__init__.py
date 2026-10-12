from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mountmin_env_cfg import (
  piplus_ball_mountmin_env_cfg,
  piplus_ball_mountmin_ppo_runner_cfg,
)

# (suffix, bounded actions, locked ball)
for suffix, bounded, locked in (("", False, False), ("-Clip", True, False), ("-Locked", False, True)):
  env_cfg = partial(piplus_ball_mountmin_env_cfg, bounded_actions=bounded, locked=locked)
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-MountMin{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mountmin_ppo_runner_cfg(suffix.lower().replace("-", "_"), bounded),
  )
