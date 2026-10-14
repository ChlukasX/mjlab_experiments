"""Mount with a pruned reward set and a staged stance target.

Mounts only, randomized ball (0.07 to 0.11 m). The reward terms that earlier rounds showed
to have no effect are removed, the free foot's lift is measured at the sole (the ankle
height is passed on tiptoe), and the stance reward starts at a small clearance that is
raised in later stages. See docs/ball_mount_rewards/ball_mount_rewards.md.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_piplus.tasks.ball_mount.events import ball_rolling_resistance
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import ball_pos_until_on_ball
from mjlab_piplus.tasks.ball_mount.piplus_ball_mountbalance_env_cfg import (
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)
from mjlab_piplus.tasks.ball_mount_curriculum.piplus_ball_mountcurr_env_cfg import (
    MU_ROLL_LOCKED,
    piplus_ball_mountcurr_ppo_runner_cfg,
)

# No effect in mb3 (momentum), mb5 (penalties); foot_on_ball is contained in
# support_on_ball; xy_centering had weight 0.
_REMOVED = (
    "foot_on_ball", "com_toward_ball_velocity", "push_up_velocity",
    "straddle_penalty", "termination_penalty", "xy_centering",
)
COM_LINEAR_RANGE = 0.30  # m: com_over_ball falls linearly to 0 at the mount spawn distance
# Blind on the ball: base height above which the robot counts as on the ball (standing on
# the floor 0.39 m, up to ~0.43 on tiptoe; on a size 1 ball from ~0.42 crouched, 0.46 tall).
ON_BALL_BASE_HEIGHT = 0.42
ON_BALL_MAX_TILT = 0.35  # rad (20 deg): only an upright robot counts as on the ball
BALL_OBS_BIAS = (0.03, 0.03, 0.01)  # m, per-episode error of the ball filter's estimate


def piplus_ball_mountmin_env_cfg(
    play: bool = False,
    bounded_actions: bool = False,
    lifted_clearance: float = 0.03,
    locked: bool = False,
    blind_on_ball: bool = False,
) -> ManagerBasedRlEnvCfg:
    """`lifted_clearance`: m the free sole must clear the floor for the stance reward
    (stage 1: 0.03, final: 0.10). `locked`: ball held by rolling resistance all training
    (feasibility probe; play and eval use a free ball). `blind_on_ball`: the actor sees
    the ball position only until it is on the ball (base height), with a per-episode
    perception error, and balances on proprioception from then on (as on the robot)."""
    cfg = piplus_ball_mountbalance_env_cfg(
        play, mount_fraction=1.0, flat_sole=True, bounded_actions=bounded_actions
    )
    r = cfg.rewards
    for k in _REMOVED:
        del r[k]
    r["support_on_ball"].weight = 2.0
    r["com_over_ball"].params["linear_range"] = COM_LINEAR_RANGE
    r["single_leg_stance"].weight = 8.0
    r["single_leg_stance"].params["sole_clearance"] = lifted_clearance
    r["free_foot_lift"].weight = 3.0
    r["free_foot_lift"].params["sole"] = True

    if blind_on_ball:
        term = cfg.observations["actor"].terms["ball_pos"]
        term.func = ball_pos_until_on_ball
        term.params = {**term.params, "height_threshold": ON_BALL_BASE_HEIGHT,
            "bias": BALL_OBS_BIAS, "max_tilt": ON_BALL_MAX_TILT}
        # ponytail: the per-step noise (term.noise, +-2 cm) also hits the zeros once on the
        # ball; harmless as a "no ball" code, since a real value has z ~ -0.3 m.

    if locked and not play:
        cfg.events["ball_lock"] = EventTermCfg(
            func=ball_rolling_resistance,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "mu_roll_max": MU_ROLL_LOCKED,
                "steps_per_iter": 1,
                "hold_iters": 0,
                "ramp_iters": 0,  # never released
            },
        )
    return cfg


def piplus_ball_mountmin_ppo_runner_cfg(
    suffix: str = "", bounded_actions: bool = False
) -> RslRlOnPolicyRunnerCfg:
    """Unbounded: the runner of the earlier mount runs (entropy 0.01, 24 steps), the only
    setting that learned to balance on the ball. Bounded: the mount curriculum's runner
    (entropy 0.003, 64 steps, gamma 0.995)."""
    if bounded_actions:
        cfg = piplus_ball_mountcurr_ppo_runner_cfg("")
    else:
        cfg = piplus_ball_mountbalance_ppo_runner_cfg(100)
    cfg.experiment_name = "piplus_ball_mountmin" + suffix
    return cfg
