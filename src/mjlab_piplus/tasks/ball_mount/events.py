"""Events for the randomized-ball mount+balance task."""

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.tasks.ball_mount.piplus_ball_mount_env_cfg import (
    BALL_GAP_X,
    RIGHT_FOOT_Y,
    STAND_Z,
    _ball_radius,
    drop_base_z,
)
from mjlab_piplus.tasks.ball_small.piplus_ball_small_env_cfg import ROBOT_SPAWN_X

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _all_envs(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def ball_mass_from_size(
    env,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    radius_range: tuple[float, float],
    mass_range: tuple[float, float],
    jitter: tuple[float, float],
) -> None:
    """Startup: ball mass follows radius (linear between the two sizes) times a jitter.

    Inertia is set to a solid sphere's 0.4 m r^2, matching the compiled model.
    Run after the ball_size (geom_size) event.
    """
    env_ids = _all_envs(env, env_ids)
    radius = _ball_radius(env, ball_cfg, None)[env_ids]
    frac = ((radius - radius_range[0]) / (radius_range[1] - radius_range[0])).clamp(0, 1)
    mass = mass_range[0] + frac * (mass_range[1] - mass_range[0])
    mass = mass * (jitter[0] + torch.rand_like(mass) * (jitter[1] - jitter[0]))
    ball = env.scene[ball_cfg.name]
    body_id = ball.indexing.body_ids[ball.body_names.index("ball_body")]
    env.sim.model.body_mass[env_ids, body_id] = mass
    env.sim.model.body_inertia[env_ids, body_id] = (0.4 * mass * radius**2).unsqueeze(-1)


@requires_model_fields("geom_friction")
def ground_friction(
    env, env_ids: torch.Tensor | None, friction_range: tuple[float, float]
) -> None:
    """Startup: per-env sliding friction of the ground plane.

    MuJoCo mixes the two geoms' friction with a max unless priorities differ, so
    this mainly sets ball-ground rolling (the robot's geoms have priority 1 and
    keep their own friction against the ground).
    """
    env_ids = _all_envs(env, env_ids)
    terrain = env.scene.terrain
    geom_id = terrain.indexing.geom_ids[terrain.geom_names.index("terrain")]
    lo, hi = friction_range
    env.sim.model.geom_friction[env_ids, geom_id, 0] = lo + torch.rand(
        len(env_ids), device=env.device
    ) * (hi - lo)


def reset_mount_or_drop(
    env,
    env_ids: torch.Tensor | None,
    mount_fraction: float,
    ball_xy_noise: tuple[float, float] = (0.03, 0.02),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset: each env starts either beside the ball (mount) or dropped onto it.

    mount: robot standing, ball just ahead of the right foot (+ small noise).
    drop:  robot just above the ball (a few cm of clearance) with the right foot over it.
    Both use the env's own ball radius. Must run after reset_scene_to_default.
    The choice is kept in ``env.spawn_is_mount`` for evaluation.
    """
    env_ids = _all_envs(env, env_ids)
    n = len(env_ids)
    dev = env.device
    if not hasattr(env, "spawn_is_mount"):
        env.spawn_is_mount = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    # Viewer toggle (SpawnModeCommand): "random" uses mount_fraction.
    override = getattr(env, "spawn_mode_override", "random")
    if override == "mount":
        mount = torch.ones(n, dtype=torch.bool, device=dev)
    elif override == "drop":
        mount = torch.zeros(n, dtype=torch.bool, device=dev)
    else:
        mount = torch.rand(n, device=dev) < mount_fraction
    env.spawn_is_mount[env_ids] = mount

    radius = _ball_radius(env, ball_cfg, None)[env_ids]
    zeros = torch.zeros(n, device=dev)
    noise = (torch.rand(n, 2, device=dev) * 2 - 1) * torch.tensor(ball_xy_noise, device=dev)

    robot_xyz = torch.stack(
        [
            torch.where(mount, zeros, zeros + ROBOT_SPAWN_X),
            zeros,
            torch.where(mount, zeros + STAND_Z, drop_base_z(radius)),
        ],
        dim=-1,
    )
    ball_xyz = torch.stack(
        [
            torch.where(mount, BALL_GAP_X + radius + noise[:, 0], zeros),
            zeros + RIGHT_FOOT_Y + torch.where(mount, noise[:, 1], zeros),
            radius,
        ],
        dim=-1,
    )
    quat = torch.tensor(_IDENTITY_QUAT, device=dev).expand(n, 4)
    origins = env.scene.env_origins[env_ids]
    env.scene[robot_cfg.name].write_root_link_pose_to_sim(
        torch.cat([robot_xyz + origins, quat], dim=-1), env_ids=env_ids
    )
    env.scene[ball_cfg.name].write_root_link_pose_to_sim(
        torch.cat([ball_xyz + origins, quat], dim=-1), env_ids=env_ids
    )


_SPAWN_MODES = {"Random mix": "random", "Mount": "mount", "Drop": "drop"}


class SpawnModeCommand(CommandTerm):
    """Play-only: viewer dropdown choosing the spawn mode (random mix / mount / drop).

    Holds no command; it only hosts the GUI. Changing the dropdown sets
    ``env.spawn_mode_override`` (read by ``reset_mount_or_drop``) and resets.
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        dropdown = server.gui.add_dropdown(
            "Spawn mode", options=tuple(_SPAWN_MODES), initial_value="Random mix"
        )

        @dropdown.on_update
        def _(_ev) -> None:
            self._env.spawn_mode_override = _SPAWN_MODES[dropdown.value]
            if request_action is not None:
                request_action("RESET")

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass


@dataclass(kw_only=True)
class SpawnModeCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env) -> SpawnModeCommand:
        return SpawnModeCommand(self, env)
