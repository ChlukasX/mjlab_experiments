"""Mount curriculum: learn the mount on a ball that is first held still, with progress-only reward.

Mounts only, randomized ball (0.07 to 0.11 m), bounded actions. Round 1 of the mount
curriculum experiment; see docs/ball_mount_curriculum/ball_mount_curriculum.md.

Variants (`lock`, `potential`):
  - lock "release": the ball has a large rolling resistance (it cannot roll away under the
    robot), released over training.
  - lock "locked": the ball stays locked all training (feasibility probe).
  - potential: the standing rewards that pay for resting in the straddle are replaced by a
    potential-based progress reward; the single-leg stance reward stays.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_piplus.tasks.ball_mount.events import ball_rolling_resistance
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import mount_potential
from mjlab_piplus.tasks.ball_mount.piplus_ball_mountbalance_env_cfg import (
    piplus_ball_mountbalance_env_cfg,
    piplus_ball_mountbalance_ppo_runner_cfg,
)

STEPS_PER_ITER = 64  # num_steps_per_env of the runner below
GAMMA = 0.995
POTENTIAL_WEIGHT = 2000.0  # rewards are scaled by dt = 0.02: a full mount (delta phi ~0.6) is ~+24
MU_ROLL_LOCKED = 0.2  # m: rolling-friction coefficient; cuts ball travel under a dropped robot by ~60%
LOCK_HOLD_ITERS, LOCK_RAMP_ITERS = 1000, 3000  # locked, then released linearly (free at it 4000 of 6000)

# Standing terms that pay for resting in the straddle; the potential replaces them.
_FARMED = (
    "support_on_ball", "com_over_ball", "stand_tall", "free_foot_lift", "foot_on_ball",
    "foot_to_ball_top", "foot_lift", "sole_flat", "com_toward_ball_velocity", "push_up_velocity",
)


def piplus_ball_mountcurr_env_cfg(
    play: bool = False, lock: str = "none", potential: bool = False
) -> ManagerBasedRlEnvCfg:
    assert lock in ("none", "release", "locked")
    cfg = piplus_ball_mountbalance_env_cfg(
        play, mount_fraction=1.0, flat_sole=True, bounded_actions=True
    )
    r = cfg.rewards
    # Rebalanced weights of the mb4 runs.
    r["support_on_ball"].weight = 2.0
    r["single_leg_stance"].weight = 8.0
    r["free_foot_lift"].weight = 3.0

    if potential:
        ref = r["support_on_ball"].params  # foot_cfg, ball_cfg, ball_radius (None = per env), flat, com_max
        params = {k: ref[k] for k in ("foot_cfg", "ball_cfg", "ball_radius")} | {"gamma": GAMMA}
        for k in _FARMED:
            r[k].weight = 0.0
        r["mount_potential"] = RewardTermCfg(func=mount_potential, weight=POTENTIAL_WEIGHT, params=params)

    if lock != "none":
        cfg.events["ball_lock"] = EventTermCfg(
            func=ball_rolling_resistance,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "mu_roll_max": MU_ROLL_LOCKED,
                "steps_per_iter": STEPS_PER_ITER,
                "hold_iters": LOCK_HOLD_ITERS,
                "ramp_iters": LOCK_RAMP_ITERS if lock == "release" else 0,
            },
        )
        if play:
            del cfg.events["ball_lock"]  # evaluate and play on a free ball
    return cfg


def piplus_ball_mountcurr_ppo_runner_cfg(variant: str) -> RslRlOnPolicyRunnerCfg:
    """Longer rollouts and discounting for a 2 to 3 s maneuver (was 24 steps, gamma 0.99)."""
    cfg = piplus_ball_mountbalance_ppo_runner_cfg(100, bounded_actions=True)
    cfg.experiment_name = f"piplus_ball_mountcurr_{variant}"
    cfg.num_steps_per_env = STEPS_PER_ITER
    cfg.algorithm.gamma = GAMMA
    cfg.algorithm.lam = 0.97
    cfg.algorithm.entropy_coef = 0.003
    cfg.max_iterations = 6000  # with 2048 envs x 64 steps: as many samples as 8000 it of 4096 envs x 24
    return cfg
