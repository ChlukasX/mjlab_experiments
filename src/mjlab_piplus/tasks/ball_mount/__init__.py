from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mountbalance_env_cfg import (
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)
from .piplus_ball_mount_env_cfg import (
    piplus_ball_balance_env_cfg,
    piplus_ball_balance_ppo_runner_cfg,
    piplus_ball_mount_env_cfg,
    piplus_ball_mount_ppo_runner_cfg,
)

# (suffix, flat_sole, ball_size): plain/flat sole requirement, FIFA size 5 / 1.
for suffix, flat, size in (
  ("", False, 5),
  ("-Flat", True, 5),
  ("-Size1", False, 1),
  ("-Size1-Flat", True, 1),
):
  env_cfg = partial(piplus_ball_mount_env_cfg, flat_sole=flat, ball_size=size)
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-Mount{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mount_ppo_runner_cfg(flat_sole=flat, ball_size=size),
  )

# Drop onto the ball and balance (no stepping up); flat sole required.
register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-Balance-Size1",
  env_cfg=piplus_ball_balance_env_cfg(),
  play_env_cfg=piplus_ball_balance_env_cfg(play=True),
  rl_cfg=piplus_ball_balance_ppo_runner_cfg(),
)

# One policy for both skills on a randomized ball (size 1-5, mass, friction).
# Suffix = share of resets that start beside the ball (rest are dropped onto it).
for pct in (50, 30, 70):
  env_cfg = partial(piplus_ball_mountbalance_env_cfg, mount_fraction=pct / 100)
  register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-MountBalance" + ("" if pct == 50 else f"-Mix{pct}"),
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_mountbalance_ppo_runner_cfg(pct),
  )
