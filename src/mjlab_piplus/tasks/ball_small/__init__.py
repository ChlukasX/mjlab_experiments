from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_small_env_cfg import (
    piplus_ball_small_env_cfg,
    piplus_ball_small_ppo_runner_cfg,
    piplus_ball_small_turf_env_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-Small",
  env_cfg=piplus_ball_small_env_cfg(),
  play_env_cfg=piplus_ball_small_env_cfg(play=True),
  rl_cfg=piplus_ball_small_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Piplus-Ball-Small-Turf",
  env_cfg=piplus_ball_small_turf_env_cfg(),
  play_env_cfg=piplus_ball_small_turf_env_cfg(play=True),
  rl_cfg=piplus_ball_small_ppo_runner_cfg(),
)
