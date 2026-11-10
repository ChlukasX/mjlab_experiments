"""Ball-stance geometry and contact helpers for the ball mount tasks and their scripts.

Moved from `tasks/ball_mount/piplus_ball_mount_env_cfg.py`, where they had private names
(`_ball_radius`, `_support_foot`, ...). Only the names changed: the old module imports each
helper under its old name, so `from <old module> import _support_foot` still works.
`ball_radii` was `_ball_radius`; the others lost their leading underscore. Check C2 hashes
the callers of these helpers: change a body here only with a new task variant (rule S6).
`foot_on_ball` is not the private `_foot_on_ball` of `tasks/ball_small`: that one has
another height window and a fixed radius. See AGENTS.md (rules S7 and S15) and
docs/mount_feasibility/mount_feasibility.md.
"""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

# Ankle height above the ball top for a foot standing on it, measured on trained
# policies: 0.033-0.05 m (0.049 with the sole flat). Window is centre +- half.
ON_BALL_Z_CENTER = 0.045
ON_BALL_Z_HALF = 0.03
FOOT_ON_BALL_XY_MARGIN = 0.05  # m, horizontal margin of a foot on the ball
MAX_SOLE_TILT = 0.3  # rad (~17 deg): sole counts as flat on the ball below this

# Sole corners in the ankle_roll_link frame (the four sole capsules: 16 cm long, 5 cm wide,
# bottom 5 cm below the link origin).
SOLE_CORNERS = [(x, y, -0.05) for x in (-0.10, 0.065) for y in (-0.035, 0.035)]

# Contact sensors between each ankle roll link and the ball (r, l: the foot_cfg order), all
# physics steps of a control step. Added by ball_contact_sensors() in
# tasks/ball_mount/piplus_ball_mount_env_cfg.py; read by ball_contact().
FOOT_BALL_SENSORS = ("r_foot_ball", "l_foot_ball")
BALL_CONTACT_FORCE = 1.0  # N

# Stance geometry and thresholds. Moved from `tasks/ball_mount/piplus_ball_mount_env_cfg.py` and
# `tasks/ball_mount_min`; the old modules import them from here.
STAND_FOOT_Z = 0.05  # ankle link height when standing on flat floor (measured)
LIFTED_FOOT_Z = STAND_FOOT_Z + 0.03  # the non-support foot must clear the floor by this
FREE_FOOT_LIFT = 0.10  # m the free foot should clear the floor (trained policies: ~0.045)
# Standing tall on the ball: base height above the ball top with the leg at its
# loaded home pose (0.34 m unloaded, ~5 cm of PD sag). The trained crouch sat at ~0.19 m.
BASE_ABOVE_TOP_TARGET = 0.30
# Weight on the ball: the whole-robot centre of mass must be over the ball, not
# beside it with a foot resting on top. Trained policies sat 0.25 m away.
COM_OVER_BALL = 0.12  # m horizontal CoM-to-ball-centre distance that counts as supported
COM_SIGMA = 0.25  # m, width of the dense com_over_ball reward
GATE_FREE_CLEARANCE = (
    0.03  # m, free sole's lowest corner above the floor (stage 1 stance clearance)
)
FLOOR_TOUCH_CLEARANCE = (
    0.01  # m, sole corner above the floor: the tap test of eval_mountbalance --gait
)


def ball_radii(env, ball_cfg: SceneEntityCfg, ball_radius: float | None) -> torch.Tensor:
    """[B] ball radius: the fixed value, or each env's own (randomized) geom size."""
    if ball_radius is not None:
        return torch.full((env.num_envs,), ball_radius, device=env.device)
    ball = env.scene[ball_cfg.name]
    geom_id = ball.indexing.geom_ids[ball.geom_names.index("ball_geom")]
    return env.sim.model.geom_size[:, geom_id, 0].expand(env.num_envs)


def foot_on_ball(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None
) -> torch.Tensor:
    """[B, 2] bool: which ankle_roll links are on top of the ball.

    Like Ball-Small's test, but the height window is centred on where the ankle
    really sits (top + ON_BALL_Z_CENTER), so a foot hovering beside a small
    ball does not count.
    """
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    foot_pos = robot.data.body_link_pose_w[:, foot_cfg.body_ids, :3]
    ball_pos = ball.data.root_link_pos_w
    radius = ball_radii(env, ball_cfg, ball_radius)
    ankle_z = ball_pos[:, 2] + radius + ON_BALL_Z_CENTER
    z_ok = (foot_pos[:, :, 2] - ankle_z.unsqueeze(-1)).abs() < ON_BALL_Z_HALF
    horiz = (foot_pos[:, :, :2] - ball_pos[:, :2].unsqueeze(1)).norm(dim=-1)
    return z_ok & (horiz < (radius + FOOT_ON_BALL_XY_MARGIN).unsqueeze(-1))


def sole_tilt(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 2] angle (rad) between each sole normal and the ball surface normal.

    The ankle_roll_link z axis is the sole normal (identity quat when flat on
    the floor). The ball surface normal under the foot is the direction from
    the ball centre to the foot.
    """
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    pose = robot.data.body_link_pose_w[:, foot_cfg.body_ids]  # [B, 2, 7]
    foot_pos, quat = pose[..., :3], pose[..., 3:7]
    up = torch.tensor([0.0, 0.0, 1.0], device=pose.device).expand_as(foot_pos)
    normal = quat_apply(quat, up)
    to_foot = foot_pos - ball.data.root_link_pos_w.unsqueeze(1)
    to_foot = to_foot / to_foot.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.acos((normal * to_foot).sum(dim=-1).clamp(-1.0, 1.0))


def sole_level_tilt(env, foot_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 2] angle (rad) between each sole normal and the world vertical."""
    quat = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 3:7]
    up = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*quat.shape[:-1], 3)
    return torch.acos(quat_apply(quat, up)[..., 2].clamp(-1.0, 1.0))


def sole_clearance(env, foot_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 2] height of each sole's lowest corner above the floor (z = 0).

    The ankle link height alone is not a lift test: on tiptoe the link rises ~4.5 cm
    with the toes still on the floor (mountbalance-mix50-v1 passed the 3 cm test that way).
    """
    pose = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids]  # [B, 2, 7]
    corners = torch.tensor(SOLE_CORNERS, device=pose.device)  # [4, 3]
    quat = pose[..., None, 3:7].expand(-1, -1, 4, -1)
    z = pose[..., None, 2] + quat_apply(quat, corners.expand_as(quat[..., :3]))[..., 2]
    return z.min(dim=-1).values


def ball_contact(env) -> torch.Tensor:
    """[B, 2] bool: foot touched the ball in any physics step of the last control step."""
    return torch.stack(
        [
            env.scene[n].data.force_history[:, 0].norm(dim=-1).amax(dim=-1) > BALL_CONTACT_FORCE
            for n in FOOT_BALL_SENSORS
        ],
        dim=1,
    )


def support_foot(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool | str,
) -> torch.Tensor:
    """[B, 2] bool: foot on the ball (and the sole flat against it, if `flat`).

    `flat="level"` also requires the sole within MAX_SOLE_TILT of horizontal, i.e. the foot
    near the ball top: flat against the ball alone passes on the flank of the ball (of2-clip:
    foot 23 deg down the side, sole 25 deg from horizontal, ankle roll at its stop)."""
    on_ball = foot_on_ball(env, foot_cfg, ball_cfg, ball_radius)
    if flat:
        on_ball = on_ball & (sole_tilt(env, foot_cfg, ball_cfg) < MAX_SOLE_TILT)
    # ponytail: str flag on a bool param, saves threading a new arg through 10 terms
    if flat == "level":
        on_ball = on_ball & (sole_level_tilt(env, foot_cfg) < MAX_SOLE_TILT)
    return on_ball


def com_to_ball(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 2] horizontal vector from the whole-robot centre of mass to the ball centre."""
    robot = env.scene[foot_cfg.name]
    com = env.sim.data.subtree_com[:, robot.indexing.root_body_id]
    return env.scene[ball_cfg.name].data.root_link_pos_w[:, :2] - com[:, :2]


def ball_contact_sensors(decimation: int) -> tuple:
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    return tuple(
        ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="body", pattern=f"{name[0]}_ankle_roll_link", entity="robot"),
            secondary=ContactMatch(mode="body", pattern=".*", entity="ball"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=decimation,
        )
        for name in FOOT_BALL_SENSORS
    )
