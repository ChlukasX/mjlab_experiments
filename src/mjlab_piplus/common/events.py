"""Event functions that two or more task packages use.

Moved without change from `tasks/ball_mount/events.py`, which imports them from here.
`all_envs` was `_all_envs`; `_all_envs` stays as an alias here, because check C2 hashes the
source of the callers. See docs/ball_mount/ball_mount.md and
docs/balance_from_mount/balance_from_mount.md.
"""

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.common.ball_stance import ball_radii as _ball_radius


def all_envs(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    """The env ids to act on: all envs when `env_ids` is None."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device)


_all_envs = all_envs


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
    level = (
        1.0 if ramp_iters <= 0 else float(min(1.0, max(0.0, 1.0 - (it - hold_iters) / ramp_iters)))
    )
    ball = env.scene[ball_cfg.name]
    geom_id = ball.indexing.geom_ids[ball.geom_names.index("ball_geom")]
    if not hasattr(env, "_ball_mu_roll_base"):
        env._ball_mu_roll_base = float(env.sim.model.geom_friction[0, geom_id, 2])
    base = env._ball_mu_roll_base
    env.sim.model.geom_friction[env_ids, geom_id, 2] = base + level * (mu_roll_max - base)
    env.ball_lock_level = level


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
    env.scene[ball_cfg.name].write_root_state_to_sim(
        s[:, 14 + 2 * nj : 27 + 2 * nj] + origin, env_ids=env_ids
    )
    if s.shape[1] > 27 + 2 * nj:
        # The mount policy's last action, for `last_action_restored` (the action manager is
        # reset after the events, so it cannot be written here).
        if not hasattr(env, "_spawn_action"):
            env._spawn_action = torch.zeros(
                env.num_envs, s.shape[1] - 27 - 2 * nj, device=env.device
            )
            env._spawn_action_pending = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
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
