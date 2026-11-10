"""Reward functions for the approach to the ball and the step onto it.

Moved without change from `tasks/ball_mount/piplus_ball_mount_env_cfg.py`, which imports
them from here. The helpers of `ball_stance` keep their old private names here, because check C2
hashes the source of the reward functions. See docs/mount_potential/mount_potential.md.
"""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.common.ball_stance import (
    BASE_ABOVE_TOP_TARGET,
    COM_SIGMA,
    FREE_FOOT_LIFT,
    STAND_FOOT_Z,
)
from mjlab_piplus.common.ball_stance import ball_radii as _ball_radius
from mjlab_piplus.common.ball_stance import com_to_ball as _com_to_ball


def foot_to_ball_top(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None, std: float
) -> torch.Tensor:
    """exp(-d^2/std^2) of the nearest foot's distance to the ball top point."""
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    foot = robot.data.body_link_pose_w[:, foot_cfg.body_ids, :3]  # [B, 2, 3]
    top = ball.data.root_link_pos_w.clone()
    top[:, 2] += _ball_radius(env, ball_cfg, ball_radius)
    d2 = (foot - top.unsqueeze(1)).pow(2).sum(dim=-1).min(dim=-1).values
    return torch.exp(-d2 / std**2)


def _mount_phi(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None
) -> torch.Tensor:
    """Progress potential in [0, 1] along the mount: foot on the ball top, then CoM over the
    ball, then base height and the free foot's clearance (only counted with the CoM over
    the ball AND a foot on the ball top: without that gate, draping the body over the ball
    with both feet off it scored 0.45 and the search converged on it)."""
    robot = env.scene[foot_cfg.name]
    top_z = env.scene[ball_cfg.name].data.root_link_pos_w[:, 2] + _ball_radius(
        env, ball_cfg, ball_radius
    )
    p_foot = foot_to_ball_top(env, foot_cfg, ball_cfg, ball_radius, 0.15)
    c = torch.exp(-(_com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) / COM_SIGMA).pow(2))
    h = ((robot.data.root_link_pos_w[:, 2] - top_z) / BASE_ABOVE_TOP_TARGET).clamp(0.0, 1.0)
    foot_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    lift = ((foot_z.min(dim=-1).values - STAND_FOOT_Z) / FREE_FOOT_LIFT).clamp(0.0, 1.0)
    return 0.15 * p_foot + 0.85 * p_foot * c * (0.3 + 0.35 * h + 0.35 * lift)


def mount_potential(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    gamma: float,
) -> torch.Tensor:
    """Potential-based shaping gamma * phi(s') - phi(s): pays for progress along the mount
    and nothing for resting in the straddle (so the straddle is not a place to farm
    reward). Zero on the first step after a reset.

    The potential of a terminal state is 0 (a fall costs -phi(s)). Without that the shaped
    return telescopes to gamma^T phi(s_T) - phi(s_0), so diving at the ball and dying in a
    high-potential state keeps the whole gain: the first version of this reward did exactly
    that (episodes of 22 steps, 90% ending in fell_over)."""
    phi = _mount_phi(env, foot_cfg, ball_cfg, ball_radius)
    prev = getattr(env, "_mount_phi_prev", None)
    if prev is None or prev.shape != phi.shape:
        prev = phi.clone()
    phi_next = torch.where(env.termination_manager.terminated, torch.zeros_like(phi), phi)
    reward = torch.where(
        env.episode_length_buf <= 1, torch.zeros_like(phi), gamma * phi_next - prev
    )
    env._mount_phi_prev = phi.detach().clone()
    return reward
