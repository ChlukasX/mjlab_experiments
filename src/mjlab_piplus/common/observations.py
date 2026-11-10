"""Ball observation functions of the ball mount tasks.

Moved without change from `tasks/ball_mount/piplus_ball_mount_env_cfg.py`. The old module
imports the names from here. `_ball_radius` is the old name of `ball_radii` in
`common/ball_stance.py`: the callers keep the old name, because check C2 hashes their source.
See docs/mount_feasibility/mount_feasibility.md and docs/mount_latency/mount_latency.md.
"""

import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mjlab_piplus.common.ball_stance import ON_BALL_Z_CENTER
from mjlab_piplus.common.ball_stance import ball_radii as _ball_radius


def ball_pos_base_obs(env, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball position relative to the base, in the base frame [B, 3]."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel)


def ball_pos_until_on_ball(
    env,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    height_threshold: float,
    bias: tuple[float, float, float],
    max_tilt: float = 0.35,
    step_noise: float = 0.0,
    kinematic: SceneEntityCfg | None = None,
    kin_bias: float = 0.0,
) -> torch.Tensor:
    """Ball position in the base frame [B, 3] as the robot's camera pipeline would give it,
    until the robot is on the ball; zeros from then on (never a real value: the ball is
    always below the base).

    On the robot the head camera cannot see a ball in front of the feet, and the ball is
    stationary, so before the mount the position comes from the ball filter's estimate of
    earlier sightings: true position plus a perception error that is constant over an
    episode (`bias`, uniform +- per axis, m; per-step noise comes from the term's noise
    config). "On the ball" is decided as on the robot: base (IMU) height above
    `height_threshold` while the torso is upright (tilt below `max_tilt`, rad), so a
    robot falling sideways does not count; latched for the rest of the episode.
    `step_noise` (m): per-step uniform noise added here, before the zeros, so they stay exact
    as on the robot (instead of the term's noise config, which also hits the zeros).
    `kinematic` (the two ankle roll links): once on the ball, the kinematic ball estimate
    (`kinematic_ball_pos`) instead of zeros, with its own per-episode error (+- `kin_bias`, m).
    """
    true = ball_pos_base_obs(env, asset_cfg, ball_cfg)
    fresh = env.episode_length_buf == 0
    if not hasattr(env, "_ball_obs_bias"):
        env._ball_obs_bias = torch.zeros(env.num_envs, 3, device=env.device)
        env._on_ball_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        fresh = torch.ones_like(fresh)
    if not hasattr(env, "_blind_episode"):
        env._blind_episode = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if fresh.any():
        b = torch.tensor(bias, device=env.device)
        env._ball_obs_bias[fresh] = (torch.rand(int(fresh.sum()), 3, device=env.device) * 2 - 1) * b
        if kinematic is not None:
            if not hasattr(env, "_kin_bias"):
                env._kin_bias = torch.zeros(env.num_envs, 3, device=env.device)
            env._kin_bias[fresh] = (
                torch.rand(int(fresh.sum()), 3, device=env.device) * 2 - 1
            ) * kin_bias
        env._on_ball_latch[fresh] = False
        # Blindness curriculum: only this share of episodes goes blind on the ball (1 unless set).
        env._blind_episode[fresh] = torch.rand(int(fresh.sum()), device=env.device) < getattr(
            env, "blind_prob", 1.0
        )
    robot = env.scene[asset_cfg.name]
    upright = -robot.data.projected_gravity_b[:, 2] > math.cos(max_tilt)
    env._on_ball_latch |= (robot.data.root_link_pos_w[:, 2] > height_threshold) & upright
    out = true + env._ball_obs_bias
    if step_noise:
        out += (torch.rand_like(out) * 2 - 1) * step_noise
    blind = env._on_ball_latch & env._blind_episode
    if kinematic is None:
        out[blind] = 0.0
    else:
        kin = kinematic_ball_pos(env, asset_cfg, kinematic, ball_cfg) + env._kin_bias
        if step_noise:
            kin += (torch.rand_like(kin) * 2 - 1) * step_noise
        out = torch.where(blind.unsqueeze(-1), kin, out)
    return out


def kinematic_ball_pos(
    env, asset_cfg: SceneEntityCfg, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Ball centre in the base frame [B, 3] as the robot can compute it standing on the ball,
    without the camera: below the higher foot (higher in the gravity frame, i.e. from the IMU), along
    its sole normal, by the ankle's height above the sole contact (ON_BALL_Z_CENTER) plus
    the ball radius (known on the robot: the node is given the ball size).

    ponytail: uses the true ankle pose, not FK of the biased encoders; the encoder error is
    modelled as the per-episode `kin_bias` of ball_pos_until_on_ball (0.03 rad over the ~0.5 m
    leg ~ 1.5 cm). Real FK if the sim2sim check shows the two differ."""
    robot = env.scene[asset_cfg.name]
    pose = robot.data.body_link_pose_w[:, foot_cfg.body_ids]  # [B, 2, 7]
    # Support foot: the higher one (the free foot hangs beside the ball, below the ball top).
    # Probe (nr1-up6, stance steps): right 99.3%; nearest under the base 93%, under the CoM 81%.
    foot = pose[
        torch.arange(env.num_envs, device=env.device), pose[:, :, 2].argmax(dim=-1)
    ]  # [B, 7]
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3)
    normal = quat_apply(foot[:, 3:7], up)
    radius = _ball_radius(env, ball_cfg, None)
    centre = foot[:, :3] - (radius + ON_BALL_Z_CENTER).unsqueeze(-1) * normal
    return quat_apply_inverse(robot.data.root_link_quat_w, centre - robot.data.root_link_pos_w)
