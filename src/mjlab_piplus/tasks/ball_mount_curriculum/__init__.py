from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mountcurr_env_cfg import (
    piplus_ball_mountcurr_env_cfg,
    piplus_ball_mountcurr_ppo_runner_cfg,
)

# Task suffix -> flags: ball lock mode, potential-based reward.
VARIANTS = {
    "Release": dict(lock="release", potential=False),
    "ReleasePot": dict(lock="release", potential=True),
    "Pot": dict(lock="none", potential=True),
    "Locked": dict(lock="locked", potential=True),
}

for suffix, flags in VARIANTS.items():
    env_cfg = partial(piplus_ball_mountcurr_env_cfg, **flags)
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-MountCurr-{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_mountcurr_ppo_runner_cfg(suffix.lower()),
    )
