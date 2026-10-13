from functools import partial

from mjlab.tasks.registry import register_mjlab_task

from .piplus_ball_balancefrommount_env_cfg import (
  piplus_ball_balancefrommount_env_cfg,
  piplus_ball_balancefrommount_ppo_runner_cfg,
)

# (suffix, share of resets starting in a mount hand-off state (rest: drops), bounded actions)
for suffix, fraction, bounded in (("", 1.0, False), ("-Mix50", 0.5, False), ("-Clip", 1.0, True)):
  env_cfg = partial(
    piplus_ball_balancefrommount_env_cfg, state_fraction=fraction, bounded_actions=bounded
  )
  register_mjlab_task(
    task_id=f"Mjlab-Piplus-Ball-BalanceFromMount{suffix}",
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=piplus_ball_balancefrommount_ppo_runner_cfg(suffix.lower().replace("-", "_"), bounded),
  )
