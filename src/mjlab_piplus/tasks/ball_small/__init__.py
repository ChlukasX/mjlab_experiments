from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_small_env_cfg import (
    piplus_ball_small_env_cfg,
    piplus_ball_small_ppo_runner_cfg,
    piplus_ball_small_turf_env_cfg,
)

# Task suffix -> environment config factory. Both tasks share one runner config.
VARIANTS = {
    "": piplus_ball_small_env_cfg,
    "-Turf": piplus_ball_small_turf_env_cfg,
}

for suffix, env_cfg in VARIANTS.items():
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-Small{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_small_ppo_runner_cfg(),
    )
