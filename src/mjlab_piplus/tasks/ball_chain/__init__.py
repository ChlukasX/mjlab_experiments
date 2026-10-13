from mjlab.tasks.registry import register_mjlab_task

from mjlab_piplus.tasks.ball_mount_min.piplus_ball_mountmin_env_cfg import (
  piplus_ball_mountmin_env_cfg,
  piplus_ball_mountmin_ppo_runner_cfg,
)


def _play_cfg():
  cfg = piplus_ball_mountmin_env_cfg(play=True, bounded_actions=True)
  del cfg.commands["spawn_mode"]  # the chain always starts standing beside the ball
  return cfg


# The mount task's scene (standing beside a randomized ball, bounded joint targets), played
# by two policies with a switch: scripts/play_chain.py. `uv run play` on this task would
# run a single policy.
register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-Chain",
  env_cfg=piplus_ball_mountmin_env_cfg(bounded_actions=True),
  play_env_cfg=_play_cfg(),
  rl_cfg=piplus_ball_mountmin_ppo_runner_cfg("_clip", bounded_actions=True),
)
