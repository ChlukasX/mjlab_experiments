"""Reward functions shared by the stand, platform, and ball tasks.

Moved without change from `tasks/ball`, `tasks/ball_arms`, `tasks/noninertial`,
`tasks/platform`, and `tasks/upright`. The function names, signatures, and bodies stay
equal to the old copies, because check C2 hashes them. See AGENTS.md (rule S7).
"""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg


def base_height_gauss(
    env, target_height: float, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Gaussian reward on the root height error [m]. Bounded [0, 1], 1 at the target."""
    asset = env.scene[asset_cfg.name]
    err = asset.data.root_link_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(err) / (std * std))


def foot_flatness(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean over the feet of the world-z component of the foot z-axis. Bounded [0, 1]."""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_com_quat_w[:, asset_cfg.body_ids, :]  # [B, 2, 4]
    # `z` stays unused: the body must equal the old copies for check C2.
    _, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]  # noqa: F841
    foot_up_z = 1 - 2 * (x * x + y * y)  # [B, 2]
    return foot_up_z.clamp(0, 1).mean(dim=-1)


def xy_centering(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise robot drifting from env origin in XY. Value is the negative squared drift [m^2]."""
    asset = env.scene[asset_cfg.name]
    drift = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return -drift.pow(2).sum(dim=-1)


def xy_drift_gauss(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for staying near the spawn point. Bounded [0, 1], safe when fallen."""
    asset = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_link_pos_w[:, :2]
    if not hasattr(env, "_spawn_xy"):
        env._spawn_xy = pos_xy.clone()
    dist_sq = torch.sum((pos_xy - env._spawn_xy) ** 2, dim=-1)
    return torch.exp(-dist_sq / (std * std))
