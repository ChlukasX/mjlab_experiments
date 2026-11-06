"""Big ball (radius 0.45 m) constants, builder, observation, reward, and terminations.

Moved without change from `tasks/ball` and `tasks/ball_arms`. The function names,
signatures, and bodies stay equal to the old copies, because check C2 hashes them.
`ROBOT_SPAWN_Z` and `BASE_HEIGHT_TARGET` here are the big-ball values. `tasks/platform`
keeps its own, different values under the same names. The football builder and its
sizes stay in `tasks/ball_small`. See AGENTS.md (rule S7).
"""

import mujoco
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

BALL_RADIUS = 0.45
BALL_MASS = 5.0
BALL_TOP = BALL_RADIUS * 2  # = 0.90 m
ROBOT_SPAWN_Z = BALL_TOP + 0.5  # = 1.40 m  (0.5 m free-fall onto ball)
BASE_HEIGHT_TARGET = BALL_TOP + 0.35  # = 1.25 m


def get_ball_spec() -> mujoco.MjSpec:
    """Free-floating sphere.  freejoint makes it a physics body, not mocap."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball_body", pos=[0, 0, BALL_RADIUS])
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS],
        mass=BALL_MASS,
        rgba=[0.85, 0.35, 0.10, 1.0],
        contype=1,
        conaffinity=1,
        condim=3,
        friction=[0.9, 0.02, 0.001],
    )
    return spec


def ball_state_obs(env, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball position relative to robot base + ball linear velocity [B, 5]."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel_pos = ball.data.root_link_pos_w - robot.data.root_link_pos_w  # [B, 3]
    ball_vel = ball.data.root_link_vel_w[:, :2]  # [B, 2] xy only
    return torch.cat([rel_pos, ball_vel], dim=-1)


def ball_control(env, std: float, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward ball moving slowly — penalises the robot destabilising the ball."""
    ball = env.scene[ball_cfg.name]
    vel_sq = ball.data.root_link_vel_w[:, :3].pow(2).sum(dim=-1)
    return torch.exp(-vel_sq / (std * std))


def ball_escaped(env, max_dist: float, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when ball rolls more than max_dist from env origin."""
    ball = env.scene[ball_cfg.name]
    dist_sq = (ball.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).pow(2).sum(dim=-1)
    return dist_sq > max_dist**2


def robot_off_ball(
    env,
    min_height_above_ball: float,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when robot base drops below ball top + margin (fell off ball)."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    ball_top_z = ball.data.root_link_pos_w[:, 2] + BALL_RADIUS
    robot_z = robot.data.root_link_pos_w[:, 2]
    return robot_z < ball_top_z + min_height_above_ball
