"""Balance on the ball, starting from the mount policy's hand-off states.

The drop-balance policy (MountBalance-Mix0) cannot take over from the mount policy: it
only knows the drop spawn (standing tall, at rest). Here the episodes start in recorded
states of the mount policy at the moment of the hand-off (single-leg stance on the ball,
crouched, moving). Same randomized ball, observations, rewards and unbounded actions as
MountBalance-Mix0. See docs/balance_from_mount/balance_from_mount.md.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_piplus.tasks.ball_mount.events import reset_from_mount_states
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import FREE_FOOT_LIFT
from mjlab_piplus.tasks.ball_mount.piplus_ball_mountbalance_env_cfg import (
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)

# Recorded with scripts/record_mount_states.py from mm3-c10-s16 (not in git: regenerate).
MOUNT_STATES = "logs/mount_states/mm3-c10-s16.pt"


def piplus_ball_balancefrommount_env_cfg(
    play: bool = False, state_fraction: float = 1.0
) -> ManagerBasedRlEnvCfg:
    """`state_fraction`: share of resets that start in a mount hand-off state (rest: drops)."""
    cfg = piplus_ball_mountbalance_env_cfg(play, mount_fraction=0.0, flat_sole=True)
    # After spawn_mode (the drop), which it overrides for the chosen envs.
    cfg.events["mount_states"] = EventTermCfg(
        func=reset_from_mount_states,
        mode="reset",
        params={"path": MOUNT_STATES, "fraction": state_fraction},
    )
    # Lift measured at the sole, as in the mount tasks (the ankle height passes on tiptoe).
    cfg.rewards["single_leg_stance"].params["sole_clearance"] = FREE_FOOT_LIFT
    cfg.rewards["free_foot_lift"].params["sole"] = True
    return cfg


def piplus_ball_balancefrommount_ppo_runner_cfg(suffix: str = "") -> RslRlOnPolicyRunnerCfg:
    cfg = piplus_ball_mountbalance_ppo_runner_cfg(0)  # the Mix0 runner (entropy 0.01, unbounded)
    cfg.experiment_name = "piplus_ball_balancefrommount" + suffix
    return cfg
