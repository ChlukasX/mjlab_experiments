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


@requires_model_fields("geom_solref")
def contact_stiffness(
    env,
    env_ids: torch.Tensor | None,
    timeconst_range: tuple[float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Startup: per-env contact time constant (`solref[0]`) of the ball and of both feet.

    All geoms had MuJoCo's default 0.02 s, which lets the foot sink ~3 mm into the ball
    and the ball ~3 mm into the floor at rest (up to 2 cm when the policy pushes;
    docs/collision_check). How much a real ball gives is not measured, so one value per
    env is drawn from the range. The feet have contact priority 1, so their value governs
    foot-ball and foot-floor contact; ball-floor mixes the ball's value with the floor's.
    The lower limit is 2 x the simulation step (0.01 s at 5 ms).
    """
    env_ids = _all_envs(env, env_ids)
    lo, hi = timeconst_range
    tc = lo + torch.rand(len(env_ids), device=env.device) * (hi - lo)
    robot, ball = env.scene[robot_cfg.name], env.scene[ball_cfg.name]
    geom_ids = [ball.indexing.geom_ids[ball.geom_names.index("ball_geom")]] + [
        robot.indexing.geom_ids[i] for i, n in enumerate(robot.geom_names) if "ankle_roll_link_collision" in n
    ]
    for g in geom_ids:
        env.sim.model.geom_solref[env_ids, g, 0] = tc


@requires_model_fields("geom_friction")
def ball_rolling_resistance(
    env,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    mu_roll_max: float,
    steps_per_iter: int,
    hold_iters: float,
    ramp_iters: float,
) -> None:
    """Reset: give the ball a large rolling-friction coefficient so it barely rolls under the
    robot's weight, then release it over training.

    Rolling friction (the third `geom_friction` entry, torque <= mu_roll * normal force) is a
    constraint, so it stays stable; joint damping on a 0.12 to 0.5 kg ball blew up the
    simulation (damping * dt / mass > 1). The coefficient is `base + level * (mu_roll_max -
    base)` with level 1 for the first `hold_iters` PPO iterations, then falling linearly to 0
    over `ramp_iters` (`ramp_iters` <= 0 keeps the ball locked all training). With the
    robot's weight on the ball (hundreds of N) 0.05 m resists several N m, far more than a
    foot can apply, so the ball stays put while the mount is learned.
    """
    env_ids = _all_envs(env, env_ids)
    it = env.common_step_counter / max(steps_per_iter, 1)
    level = 1.0 if ramp_iters <= 0 else float(min(1.0, max(0.0, 1.0 - (it - hold_iters) / ramp_iters)))
    ball = env.scene[ball_cfg.name]
    geom_id = ball.indexing.geom_ids[ball.geom_names.index("ball_geom")]
    if not hasattr(env, "_ball_mu_roll_base"):
        env._ball_mu_roll_base = float(env.sim.model.geom_friction[0, geom_id, 2])
    base = env._ball_mu_roll_base
    env.sim.model.geom_friction[env_ids, geom_id, 2] = base + level * (mu_roll_max - base)
    env.ball_lock_level = level


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


def reset_from_mount_states(
    env,
    env_ids: torch.Tensor | None,
    path: str,
    fraction: float = 1.0,
    neighbours: int = 64,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset: start `fraction` of the envs in a recorded hand-off state of the mount policy
    (scripts/record_mount_states.py): robot in the single-leg stance on the ball, moving.

    Each env takes a random one of the recorded states whose ball radius is closest to its
    own. Must run after the spawn event it overrides; the other envs keep that spawn.
    """
    env_ids = _all_envs(env, env_ids)
    if not hasattr(env, "_mount_states"):
        data = torch.load(path, map_location=env.device)
        env._mount_states, env._mount_states_nj = data["states"], data["num_joints"]
    states, nj = env._mount_states, env._mount_states_nj
    env_ids = env_ids[torch.rand(len(env_ids), device=env.device) < fraction]
    if len(env_ids) == 0:
        return
    if hasattr(env, "spawn_is_mount"):
        env.spawn_is_mount[env_ids] = True
    # ponytail: nearest-radius lookup, no height correction; with tens of thousands of
    # states the radius mismatch is well below 1 mm. Correct z by the difference if the
    # file is small.
    radius = _ball_radius(env, ball_cfg, None)[env_ids]
    idx = torch.searchsorted(states[:, 0].contiguous(), radius)
    idx = idx + torch.randint(-neighbours, neighbours + 1, idx.shape, device=env.device)
    s = states[idx.clamp(0, len(states) - 1)]
    origin = torch.zeros(len(env_ids), 13, device=env.device)
    origin[:, :3] = env.scene.env_origins[env_ids]
    robot = env.scene[robot_cfg.name]
    robot.write_root_state_to_sim(s[:, 1:14] + origin, env_ids=env_ids)
    robot.write_joint_state_to_sim(s[:, 14 : 14 + nj], s[:, 14 + nj : 14 + 2 * nj], env_ids=env_ids)
    env.scene[ball_cfg.name].write_root_state_to_sim(s[:, 14 + 2 * nj : 27 + 2 * nj] + origin, env_ids=env_ids)
    if s.shape[1] > 27 + 2 * nj:
        # The mount policy's last action, for `last_action_restored` (the action manager is
        # reset after the events, so it cannot be written here).
        if not hasattr(env, "_spawn_action"):
            env._spawn_action = torch.zeros(env.num_envs, s.shape[1] - 27 - 2 * nj, device=env.device)
            env._spawn_action_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._spawn_action[env_ids] = s[:, 27 + 2 * nj :]
        env._spawn_action_pending[env_ids] = True


def last_action_restored(env) -> torch.Tensor:
    """The "last action" observation, with the mount policy's last action restored after a
    spawn from a recorded hand-off state.

    At the real hand-off the balance policy's first observation holds the mount policy's
    last action, and the action-rate penalty is measured from it. A spawn would start with
    zeros instead; the smoother the balance policy got, the more that difference cost
    (bs2-exc50: 91% from recorded states, 16% in the real hand-off).
    """
    pending = getattr(env, "_spawn_action_pending", None)
    if pending is not None and pending.any():
        am = env.action_manager
        am._action[pending] = env._spawn_action[pending]
        am._prev_action[pending] = env._spawn_action[pending]
        pending[:] = False
    return env.action_manager.action


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
