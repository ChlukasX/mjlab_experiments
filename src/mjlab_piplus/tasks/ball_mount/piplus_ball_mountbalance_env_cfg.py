"""Piplus mount+balance on a domain-randomized ball: one policy for both skills.

Each env starts either standing beside the ball (mount, then balance) or dropped
onto it (balance), chosen per reset with probability `mount_fraction`. The ball
is randomized per env: radius 0.07-0.11 m (FIFA size 1 to 5), mass tied to the
radius, ball and ground friction. The actor does not see the ball size; the
critic does. Rewards are the flat-sole Ball-Mount ones with per-env radius.

See docs/ball_mount.md.
"""

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import is_terminated
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_piplus.robot.piplus_constants import get_spec_with_arms
from mjlab_piplus.tasks.ball_mount.events import (
    SpawnModeCommandCfg,
    ball_mass_from_size,
    ground_friction,
    reset_mount_or_drop,
)
from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    BALL_SIZES,
    COM_OVER_BALL,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
    _ball_radius,
    com_over_ball,
    com_toward_ball_velocity,
    free_foot_lift,
    free_foot_on_floor,
    piplus_ball_mount_env_cfg,
    push_up_velocity,
    stand_tall_on_ball,
    support_on_ball,
)
from mjlab_piplus.tasks.ball_small.piplus_ball_small_env_cfg import (
    piplus_ball_small_ppo_runner_cfg,
)

RADIUS_RANGE = (BALL_SIZES[1][0], BALL_SIZES[5][0])  # 0.07-0.11 m
MASS_RANGE = (BALL_SIZES[1][1], BALL_SIZES[5][1])  # 0.14-0.43 kg
FRICTION_RANGE = (0.3, 1.2)


def _joint_limit_clip() -> dict[str, tuple[float, float]]:
    """Joint name -> (lo, hi) from the robot spec, for clipping position targets."""
    spec = get_spec_with_arms()
    return {
        j.name: (float(j.range[0]), float(j.range[1]))
        for j in spec.joints
        if j.type == mujoco.mjtJoint.mjJNT_HINGE
    }


def ball_privileged_obs(env, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 4] exact ball radius, mass, ball friction and ground friction (critic only)."""
    model = env.sim.model
    ball = env.scene[ball_cfg.name]
    body_id = ball.indexing.body_ids[ball.body_names.index("ball_body")]
    geom_id = ball.indexing.geom_ids[ball.geom_names.index("ball_geom")]
    terrain = env.scene.terrain
    ground_id = terrain.indexing.geom_ids[terrain.geom_names.index("terrain")]
    n = env.num_envs
    return torch.stack(
        [
            _ball_radius(env, ball_cfg, None),
            model.body_mass[:, body_id].expand(n),
            model.geom_friction[:, geom_id, 0].expand(n),
            model.geom_friction[:, ground_id, 0].expand(n),
        ],
        dim=-1,
    )


def piplus_ball_mountbalance_env_cfg(
    play: bool = False,
    mount_fraction: float = 0.5,
    flat_sole: bool = True,
    bounded_actions: bool = False,
) -> ManagerBasedRlEnvCfg:
    """`mount_fraction`: share of resets that start beside the ball (rest are drops).

    `bounded_actions`: clip the joint position targets to the joint limits. Without
    it the action is unbounded (target = 0.5 * action rad); the earlier runs ended up
    with mean |action| of 3 to 4 and the noise std grown to ~3, i.e. targets several
    rad past the limits.
    """
    # Size-5 geometry is the reference the geom_size scale is applied to.
    cfg = piplus_ball_mount_env_cfg(play, flat_sole=flat_sole, ball_size=5)
    ball_cfg = SceneEntityCfg("ball")

    # Per-env radius: rewards read it from the model instead of a fixed value.
    for term in cfg.rewards.values():
        if "ball_radius" in term.params:
            term.params["ball_radius"] = None

    # Replace the fixed-size ball events; keep the rest of the Ball-Small DR.
    del cfg.events["ball_spawn_offset"], cfg.events["ball_mass"]
    cfg.events["ball_friction"].params["ranges"] = FRICTION_RANGE
    cfg.events["ball_size"] = EventTermCfg(
        func=dr.geom_size,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
            "ranges": (RADIUS_RANGE[0] / RADIUS_RANGE[1], 1.0),
            "operation": "scale",
            "axes": [0],
        },
    )
    # Must come after ball_size (reads the per-env radius).
    cfg.events["ball_mass_from_size"] = EventTermCfg(
        func=ball_mass_from_size,
        mode="startup",
        params={
            "ball_cfg": ball_cfg,
            "radius_range": RADIUS_RANGE,
            "mass_range": MASS_RANGE,
            "jitter": (0.8, 1.2),
        },
    )
    cfg.events["ground_friction"] = EventTermCfg(
        func=ground_friction,
        mode="startup",
        params={"friction_range": FRICTION_RANGE},
    )
    # Must come after reset_scene_to_default (first in the dict).
    cfg.events["spawn_mode"] = EventTermCfg(
        func=reset_mount_or_drop,
        mode="reset",
        params={"mount_fraction": mount_fraction},
    )

    # Balance quality: stand tall (not a deep crouch) and really lift the free foot.
    # The stance success test also demands the full clearance.
    ball_params = {
        k: v for k, v in cfg.rewards["single_leg_stance"].params.items() if k != "flat"
    }
    cfg.rewards["single_leg_stance"].params["lifted_z"] = STAND_FOOT_Z + FREE_FOOT_LIFT
    cfg.rewards["single_leg_stance"].params["com_max"] = COM_OVER_BALL
    cfg.rewards["stand_tall"] = RewardTermCfg(
        func=stand_tall_on_ball, weight=2.0, params={**ball_params, "flat": flat_sole}
    )
    cfg.rewards["free_foot_lift"] = RewardTermCfg(
        func=free_foot_lift, weight=1.0, params={**ball_params, "flat": flat_sole}
    )

    # Weight on the ball, not a foot propped on it: a foot resting on the ball while
    # standing on the floor beside it only gets the small touch bonus now.
    cfg.rewards["foot_on_ball"].weight = 1.0
    cfg.rewards["support_on_ball"] = RewardTermCfg(
        func=support_on_ball,
        weight=4.0,
        params={**ball_params, "flat": flat_sole, "com_max": COM_OVER_BALL},
    )
    cfg.rewards["com_over_ball"] = RewardTermCfg(
        func=com_over_ball, weight=2.0, params={**ball_params, "flat": flat_sole}
    )
    # Momentum for the mount: lean toward the ball, then push up. (Set weight 0 to ablate.)
    cfg.rewards["com_toward_ball_velocity"] = RewardTermCfg(
        func=com_toward_ball_velocity,
        weight=1.0,
        params={**ball_params, "flat": flat_sole, "com_max": COM_OVER_BALL},
    )
    cfg.rewards["push_up_velocity"] = RewardTermCfg(
        func=push_up_velocity, weight=0.5, params={**ball_params, "flat": flat_sole}
    )
    # Off by default (weight 0, so earlier runs stay reproducible); enable per run:
    #   termination_penalty: -200 is about -4 per failure (rewards are scaled by dt=0.02);
    #     without it a mount spawn with net-negative step rewards rewards ending the
    #     episode early (mb4 warm starts: 0% of mounts survived).
    #   straddle_penalty: -2 to -6, cancels the reward for resting with one foot on the
    #     floor and one on the ball (mb4-w4 sat there).
    cfg.rewards["termination_penalty"] = RewardTermCfg(func=is_terminated, weight=0.0)
    cfg.rewards["straddle_penalty"] = RewardTermCfg(
        func=free_foot_on_floor,
        weight=0.0,
        params={**ball_params, "flat": flat_sole, "com_max": COM_OVER_BALL},
    )
    # Centering on the env origin works against moving over the ball.
    cfg.rewards["xy_centering"].weight = 0.0

    # Play only: viewer dropdown to pick the spawn mode.
    if play:
        cfg.commands["spawn_mode"] = SpawnModeCommandCfg()

    # Critic only: exact ball radius, mass and friction.
    cfg.observations["critic"].terms["ball_privileged"] = ObservationTermCfg(
        func=ball_privileged_obs, params={"ball_cfg": ball_cfg}
    )

    if bounded_actions:
        cfg.actions["joint_pos"].clip = _joint_limit_clip()

    cfg.episode_length_s = 20.0 if not play else 1e9
    return cfg


def piplus_ball_mountbalance_ppo_runner_cfg(
    mix_pct: int = 50, bounded_actions: bool = False
) -> RslRlOnPolicyRunnerCfg:
    """`mix_pct`: mount-spawn share in percent; 50 is the base task, others get their own dir.

    `bounded_actions` also lowers the entropy bonus 0.01 -> 0.001: the Gaussian std grew
    from 1.0 to 2.4 to 3.9 in every earlier ball-balancing run.
    """
    cfg = piplus_ball_small_ppo_runner_cfg()
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.experiment_name = (
        "piplus_ball_mountbalance"
        + ("" if mix_pct == 50 else f"_mix{mix_pct}")
        + ("_clip" if bounded_actions else "")
    )
    cfg.max_iterations = 8000
    if bounded_actions:
        cfg.algorithm.entropy_coef = 0.001
    return cfg
