from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_balancefrommount_env_cfg import (
    piplus_ball_balancefrommount_env_cfg,
    piplus_ball_balancefrommount_ppo_runner_cfg,
)

# Task suffix -> flags: share of resets that start in a mount hand-off state (rest: drops),
# bounded actions.
VARIANTS = {
    "": dict(state_fraction=1.0, bounded_actions=False),
    "-Mix50": dict(state_fraction=0.5, bounded_actions=False),
    "-Clip": dict(state_fraction=1.0, bounded_actions=True),
}

for suffix, flags in VARIANTS.items():
    env_cfg = partial(piplus_ball_balancefrommount_env_cfg, **flags)
    register_mjlab_task(
        task_id=f"Mjlab-Piplus-Ball-BalanceFromMount{suffix}",
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=piplus_ball_balancefrommount_ppo_runner_cfg(
            suffix.lower().replace("-", "_"), flags["bounded_actions"]
        ),
    )
