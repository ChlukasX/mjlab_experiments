from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_mount_env_cfg import (
    piplus_ball_balance_env_cfg,
    piplus_ball_balance_ppo_runner_cfg,
    piplus_ball_mount_env_cfg,
    piplus_ball_mount_ppo_runner_cfg,
)
from .piplus_ball_mountbalance_env_cfg import (
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)

# Task suffix -> flags of piplus_ball_mount_env_cfg: flat sole required, FIFA ball size (5 or 1).
MOUNT_VARIANTS = {
    "": dict(flat_sole=False, ball_size=5),
    "-Flat": dict(flat_sole=True, ball_size=5),
    "-Size1": dict(flat_sole=False, ball_size=1),
    "-Size1-Flat": dict(flat_sole=True, ball_size=1),
}

for suffix, flags in MOUNT_VARIANTS.items():
    env_cfg = partial(piplus_ball_mount_env_cfg, **flags)
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-Mount{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_mount_ppo_runner_cfg(**flags),
    )

# Drop onto the ball and balance (no stepping up); flat sole required.
register_mjlab_task(
    task_id="Mjlab-Piplus-Ball-Balance-Size1",
    env_cfg=piplus_ball_balance_env_cfg(),
    play_env_cfg=piplus_ball_balance_env_cfg(play=True),
    rl_cfg=piplus_ball_balance_ppo_runner_cfg(),
)

# One policy for both skills on a randomized ball (size 1-5, mass, friction).
# Task suffix -> (mount share of the resets in percent, bounded actions).
# -MixN = share of resets that start beside the ball (rest are dropped onto it).
# -Clip = joint position targets clipped to the joint limits + lower entropy bonus.
MOUNTBALANCE_VARIANTS = {
    "": (50, False),
    "-Mix30": (30, False),
    "-Mix70": (70, False),
    "-Mix0": (0, False),
    "-Mix100": (100, False),
    "-Clip": (50, True),
    "-Mix100-Clip": (100, True),
    "-Mix0-Clip": (0, True),
}

for suffix, (pct, bounded) in MOUNTBALANCE_VARIANTS.items():
    env_cfg = partial(
        piplus_ball_mountbalance_env_cfg, mount_fraction=pct / 100, bounded_actions=bounded
    )
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-MountBalance{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_mountbalance_ppo_runner_cfg(pct, bounded),
    )
