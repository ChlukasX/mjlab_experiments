"""Balance on the ball, starting from the mount policy's hand-off states.

The drop-balance policy (MountBalance-Mix0) cannot take over from the mount policy: it
only knows the drop spawn (standing tall, at rest). Here the episodes start in recorded
states of the mount policy at the moment of the hand-off (single-leg stance on the ball,
crouched, moving). Same randomized ball, observations, rewards and unbounded actions as
MountBalance-Mix0. See docs/balance_from_mount/balance_from_mount.md.
"""

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_piplus.tasks.ball_mount.events import reset_from_mount_states
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import FREE_FOOT_LIFT
from mjlab_piplus.tasks.ball_mount.piplus_ball_mountbalance_env_cfg import (
    _joint_limit_clip,
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)

# Recorded with scripts/record_mount_states.py from mm3-c10-s16 (not in git: regenerate).
MOUNT_STATES = "logs/mount_states/mm3-c10-s16.pt"


def action_excess(env) -> torch.Tensor:
    """[B] rad by which the commanded joint targets lie outside the joint ranges, summed
    over the joints (before any clipping). The balance policies command targets tens of
    rad beyond the limits (bang-bang through the torque limit); this is 0 for any target
    inside the range, so it does not pull a well-behaved policy toward the home pose."""
    term = env.action_manager.get_term("joint_pos")
    if not hasattr(env, "_joint_range"):
        limits = _joint_limit_clip()
        env._joint_range = torch.tensor([limits[n] for n in term.target_names], device=env.device)
    lo, hi = env._joint_range[:, 0], env._joint_range[:, 1]
    target = env.action_manager.action * term._scale + term._offset
    return ((lo - target).clamp(min=0.0) + (target - hi).clamp(min=0.0)).sum(dim=-1)


def piplus_ball_balancefrommount_env_cfg(
    play: bool = False, state_fraction: float = 1.0, bounded_actions: bool = False
) -> ManagerBasedRlEnvCfg:
    """`state_fraction`: share of resets that start in a mount hand-off state (rest: drops).
    `bounded_actions`: joint targets clipped to the joint ranges (smoothing runs)."""
    cfg = piplus_ball_mountbalance_env_cfg(
        play, mount_fraction=0.0, flat_sole=True, bounded_actions=bounded_actions
    )
    # After spawn_mode (the drop), which it overrides for the chosen envs.
    cfg.events["mount_states"] = EventTermCfg(
        func=reset_from_mount_states,
        mode="reset",
        params={"path": MOUNT_STATES, "fraction": state_fraction},
    )
    # Lift measured at the sole, as in the mount tasks (the ankle height passes on tiptoe).
    cfg.rewards["single_leg_stance"].params["sole_clearance"] = FREE_FOOT_LIFT
    cfg.rewards["free_foot_lift"].params["sole"] = True
    # Off by default; the smoothing runs enable it (about -0.02).
    cfg.rewards["action_excess"] = RewardTermCfg(func=action_excess, weight=0.0)
    return cfg


def piplus_ball_balancefrommount_ppo_runner_cfg(
    suffix: str = "", bounded_actions: bool = False
) -> RslRlOnPolicyRunnerCfg:
    cfg = piplus_ball_mountbalance_ppo_runner_cfg(0)  # the Mix0 runner (entropy 0.01, unbounded)
    cfg.experiment_name = "piplus_ball_balancefrommount" + suffix
    if bounded_actions:
        cfg.algorithm.entropy_coef = 0.003  # as the mount runs; 0.01 grows the action noise
    return cfg
