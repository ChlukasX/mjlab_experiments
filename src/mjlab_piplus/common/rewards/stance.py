"""Single-support reward functions and helpers of the ball mount tasks.

Moved without change from `tasks/ball_mount` (`single_leg_stance_lifted`) and
`tasks/ball_mount_min` (`free_foot_contact`, `single_support`, `heading`). The old modules
import the names from here. `single_support` was `_single_support` and `heading` was `_yaw`.
The helpers of `ball_stance` keep their old private names here, because check C2 hashes the
source of the reward functions. See docs/mount_curriculum/mount_curriculum.md.
"""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from mjlab_piplus.common.ball_stance import (
    COM_OVER_BALL,
    FLOOR_TOUCH_CLEARANCE,
    GATE_FREE_CLEARANCE,
    LIFTED_FOOT_Z,
)
from mjlab_piplus.common.ball_stance import ball_contact as _ball_contact
from mjlab_piplus.common.ball_stance import com_to_ball as _com_to_ball
from mjlab_piplus.common.ball_stance import sole_clearance as _sole_clearance
from mjlab_piplus.common.ball_stance import support_foot as _support_foot


def single_leg_stance_lifted(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
    lifted_z: float = LIFTED_FOOT_Z,
    com_max: float | None = None,
    sole_clearance: float | None = None,
    free_off_ball: bool = False,
) -> torch.Tensor:
    """One foot on the ball (sole flat if `flat`), the other clearly off the floor.

    Ball-Small's version of this check passes at floor height, so it never
    required the second foot to lift. `sole_clearance` (m) tests the free sole's lowest
    corner instead of the ankle height `lifted_z`, which a foot on tiptoe passes.
    `free_off_ball`: the free foot must not touch the ball (needs ball_contact_sensors);
    without it a foot resting tilted on the ball counts as free (two-foot stance).
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat)
    if sole_clearance is None:
        foot_z = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2]
        lifted = foot_z > lifted_z
    else:
        lifted = _sole_clearance(env, foot_cfg) > sole_clearance
    if free_off_ball:
        lifted = lifted & ~_ball_contact(env)
    other_lifted = (on_ball.flip(-1) & ~on_ball) & lifted
    stance = on_ball.any(dim=-1) & other_lifted.any(dim=-1)
    if com_max is not None:
        stance = stance & (_com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) < com_max)
    return stance.float()


def free_foot_contact(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Free-foot indicator [B] (0 or 1 per step; give it a negative weight, per second). 1 while the
    robot is latched on the ball and either (a) a sole's lowest corner is within
    FLOOR_TOUCH_CLEARANCE (m) of the floor (the foot on the floor is never the support foot), or
    (b) exactly one foot is the (flat) support foot and the other foot touches the ball."""
    latched = getattr(env, "_on_ball_latch", None)
    if latched is None:
        return torch.zeros(env.num_envs, device=env.device)
    floor = (_sole_clearance(env, foot_cfg) < FLOOR_TOUCH_CLEARANCE).any(dim=-1)
    sup = _support_foot(env, foot_cfg, ball_cfg, None, True)
    on_ball = (sup.sum(dim=-1) == 1) & (_ball_contact(env) & ~sup).any(dim=-1)
    return (latched & (floor | on_ball)).float()


def single_support(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, gated: bool = False):
    """([B] bool single support with a level sole, [B] index of the support foot, 0 = right).
    `gated`: also CoM within COM_OVER_BALL of the ball centre and the free sole above
    GATE_FREE_CLEARANCE (a real stance, not a sole resting on the ball from the floor)."""
    sup = _support_foot(env, foot_cfg, ball_cfg, None, "level")
    single, si = sup.sum(-1) == 1, sup[:, 1].long()
    if gated:
        free_clear = _sole_clearance(env, foot_cfg)[
            torch.arange(env.num_envs, device=env.device), 1 - si
        ]
        single = (
            single
            & (_com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) < COM_OVER_BALL)
            & (free_clear > GATE_FREE_CLEARANCE)
        )
    return single, si


def heading(q: torch.Tensor) -> torch.Tensor:
    """Heading (rad) of the body x axis in the world xy plane."""
    x = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[:-1] + (3,)))
    return torch.atan2(x[..., 1], x[..., 0])
