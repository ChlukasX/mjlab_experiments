from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mountcurr_env_cfg import (
  piplus_ball_mountcurr_env_cfg,
  piplus_ball_mountcurr_ppo_runner_cfg,
)

# (suffix, ball lock, potential-based reward)
for suffix, lock, potential in (
  ("Release", "release", False),
  ("ReleasePot", "release", True),
  ("Pot", "none", True),
  ("Locked", "locked", True),
):
  env_cfg = partial(piplus_ball_mountcurr_env_cfg, lock=lock, potential=potential)
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-MountCurr-{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mountcurr_ppo_runner_cfg(suffix.lower()),
  )
