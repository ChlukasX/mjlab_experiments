"""Piplus ball mount (feasibility spike): step from the floor onto a football.

Robot spawns standing at the home pose, ball just ahead of the right foot.
Same 20 DOF, rewards and randomisation as Ball-Small, plus a dense
foot-to-ball-top reward. The actor additionally sees the ball position in the
base frame (privileged for now; real perception comes in a later phase).

See docs/mount_feasibility/mount_feasibility.md.
"""

from functools import partial

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_piplus.tasks.ball_small.piplus_ball_small_env_cfg import (
    FOOT_ON_BALL_XY_MARGIN,
    ROBOT_SPAWN_X,
    get_ball_spec,
    piplus_ball_small_env_cfg,
    piplus_ball_small_ppo_runner_cfg,
)

# FIFA ball sizes: (radius m, mass kg). Size 5 is the Ball-Small football.
# ponytail: size 1 is a mini ball, ~44 cm circumference, ~0.14 kg; check against the real ball.
BALL_SIZES = {5: (0.11, 0.43), 1: (0.07, 0.14)}

# Home-pose standing: base at 0.39 m, right ankle link ~0.05 m ahead of base
# and 0.0815 m to the right. Ball centre sits 0.19 m + radius in front of the
# base (0.30 m for size 5), on the right-foot line.
STAND_Z = 0.39
RIGHT_FOOT_Y = -0.0815
BALL_GAP_X = 0.19
# Ankle height above the ball top for a foot standing on it, measured on trained
# policies: 0.033-0.05 m (0.049 with the sole flat). Window is centre +- half.
ON_BALL_Z_CENTER = 0.045
ON_BALL_Z_HALF = 0.03
STAND_FOOT_Z = 0.05  # ankle link height when standing on flat floor (measured)
LIFTED_FOOT_Z = STAND_FOOT_Z + 0.03  # the non-support foot must clear the floor by this
MAX_SOLE_TILT = 0.3  # rad (~17 deg): sole counts as flat on the ball below this

# Drop spawn: the robot starts just above the ball at the home pose, so it
# settles onto it instead of falling onto it. Right ankle is ~0.29 m below the
# base at the home pose and ~0.05 m above the ball top with the sole flat.
HOME_BASE_ABOVE_ANKLE = 0.29
FLAT_ANKLE_ABOVE_TOP = 0.05
DROP_CLEARANCE = 0.03  # m between sole and ball top at spawn

# Standing tall on the ball: base height above the ball top with the leg at its
# loaded home pose (0.34 m unloaded, ~5 cm of PD sag). The trained crouch sat at ~0.19 m.
BASE_ABOVE_TOP_TARGET = 0.30
BASE_ABOVE_TOP_STD = 0.10
FREE_FOOT_LIFT = 0.10  # m the free foot should clear the floor (trained policies: ~0.045)


# Weight on the ball: the whole-robot centre of mass must be over the ball, not
# beside it with a foot resting on top. Trained policies sat 0.25 m away.
COM_OVER_BALL = 0.12  # m horizontal CoM-to-ball-centre distance that counts as supported
COM_SIGMA = 0.25  # m, width of the dense com_over_ball reward
COM_VEL_CAP = 0.5  # m/s cap for the momentum rewards


def drop_base_z(radius):
    """Base height of the drop spawn for a ball of this radius (float or tensor)."""
    return 2 * radius + FLAT_ANKLE_ABOVE_TOP + HOME_BASE_ABOVE_ANKLE + DROP_CLEARANCE


def _ball_radius(env, ball_cfg: SceneEntityCfg, ball_radius: float | None) -> torch.Tensor:
    """[B] ball radius: the fixed value, or each env's own (randomized) geom size."""
    if ball_radius is not None:
        return torch.full((env.num_envs,), ball_radius, device=env.device)
    ball = env.scene[ball_cfg.name]
    geom_id = ball.indexing.geom_ids[ball.geom_names.index("ball_geom")]
    return env.sim.model.geom_size[:, geom_id, 0].expand(env.num_envs)


def _foot_on_ball(
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
    radius = _ball_radius(env, ball_cfg, ball_radius)
    ankle_z = ball_pos[:, 2] + radius + ON_BALL_Z_CENTER
    z_ok = (foot_pos[:, :, 2] - ankle_z.unsqueeze(-1)).abs() < ON_BALL_Z_HALF
    horiz = (foot_pos[:, :, :2] - ball_pos[:, :2].unsqueeze(1)).norm(dim=-1)
    return z_ok & (horiz < (radius + FOOT_ON_BALL_XY_MARGIN).unsqueeze(-1))


def ball_pos_base_obs(env, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball position relative to the base, in the base frame [B, 3]."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel)


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


def foot_lift(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None
) -> torch.Tensor:
    """Highest foot's lift from standing height, scaled to 1 at the ball top [B]."""
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    foot_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2].max(dim=-1).values
    top_z = ball.data.root_link_pos_w[:, 2] + _ball_radius(env, ball_cfg, ball_radius)
    return ((foot_z - STAND_FOOT_Z) / (top_z - STAND_FOOT_Z)).clamp(0.0, 1.0)


def _sole_tilt(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
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


def _support_foot(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None, flat: bool
) -> torch.Tensor:
    """[B, 2] bool: foot on the ball (and the sole flat against it, if `flat`)."""
    on_ball = _foot_on_ball(env, foot_cfg, ball_cfg, ball_radius)
    if flat:
        on_ball = on_ball & (_sole_tilt(env, foot_cfg, ball_cfg) < MAX_SOLE_TILT)
    return on_ball


def foot_on_ball_reward(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None, flat: bool
) -> torch.Tensor:
    return _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1).float()


def _com_to_ball(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 2] horizontal vector from the whole-robot centre of mass to the ball centre."""
    robot = env.scene[foot_cfg.name]
    com = env.sim.data.subtree_com[:, robot.indexing.root_body_id]
    return env.scene[ball_cfg.name].data.root_link_pos_w[:, :2] - com[:, :2]


def single_leg_stance_lifted(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
    lifted_z: float = LIFTED_FOOT_Z,
    com_max: float | None = None,
) -> torch.Tensor:
    """One foot on the ball (sole flat if `flat`), the other clearly off the floor.

    Ball-Small's version of this check passes at floor height, so it never
    required the second foot to lift.
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat)
    foot_z = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    other_lifted = (on_ball.flip(-1) & ~on_ball) & (foot_z > lifted_z)
    stance = on_ball.any(dim=-1) & other_lifted.any(dim=-1)
    if com_max is not None:
        stance = stance & (_com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) < com_max)
    return stance.float()


def support_on_ball(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
    com_max: float,
) -> torch.Tensor:
    """1 if a foot is on the ball and the centre of mass is over it (weight actually on the ball)."""
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1)
    over = _com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) < com_max
    return (on_ball & over).float()


def com_over_ball(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
) -> torch.Tensor:
    """With a foot on the ball: exp(-(d / COM_SIGMA)^2), d = horizontal CoM-to-ball distance.

    Dense signal for shifting the weight onto the ball (broad: still has a gradient at 0.25 m).
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1)
    d = _com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1)
    return on_ball.float() * torch.exp(-(d / COM_SIGMA).pow(2))


def com_toward_ball_velocity(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
    com_max: float,
) -> torch.Tensor:
    """Momentum: with a foot on the ball but the CoM not yet over it, speed toward the ball (0 to 1 at COM_VEL_CAP)."""
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1)
    to_ball = _com_to_ball(env, foot_cfg, ball_cfg)
    dist = to_ball.norm(dim=-1)
    vel = env.scene[foot_cfg.name].data.root_com_lin_vel_w[:, :2]
    toward = (vel * to_ball / dist.clamp_min(1e-6).unsqueeze(-1)).sum(dim=-1)
    gate = on_ball & (dist > 0.5 * com_max)
    return gate.float() * (toward / COM_VEL_CAP).clamp(0.0, 1.0)


def push_up_velocity(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
) -> torch.Tensor:
    """Pushing up: with a foot on the ball and the base still below the standing-tall height, upward speed (0 to 1 at COM_VEL_CAP)."""
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1)
    robot = env.scene[foot_cfg.name]
    top_z = env.scene[ball_cfg.name].data.root_link_pos_w[:, 2] + _ball_radius(env, ball_cfg, ball_radius)
    below = (robot.data.root_link_pos_w[:, 2] - top_z) < BASE_ABOVE_TOP_TARGET - 0.03
    return (on_ball & below).float() * (robot.data.root_com_lin_vel_w[:, 2] / COM_VEL_CAP).clamp(0.0, 1.0)


def free_foot_on_floor(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
    com_max: float,
) -> torch.Tensor:
    """Straddle indicator: foot on the ball, CoM over it, but the other foot still on the floor.

    Meant as a penalty (negative weight) so the two-foot straddle stops being a
    comfortable resting place on the way to a single-leg stance.
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat)
    foot_z = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    free_on_floor = (on_ball.flip(-1) & ~on_ball) & (foot_z < STAND_FOOT_Z + 0.03)
    over = _com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) < com_max
    return (free_on_floor.any(dim=-1) & over).float()


def _mount_phi(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None
) -> torch.Tensor:
    """Progress potential in [0, 1] along the mount: foot on the ball top, then CoM over the
    ball, then base height and the free foot's clearance (only counted with the CoM over
    the ball)."""
    robot = env.scene[foot_cfg.name]
    top_z = env.scene[ball_cfg.name].data.root_link_pos_w[:, 2] + _ball_radius(env, ball_cfg, ball_radius)
    p_foot = foot_to_ball_top(env, foot_cfg, ball_cfg, ball_radius, 0.15)
    c = torch.exp(-(_com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1) / COM_SIGMA).pow(2))
    h = ((robot.data.root_link_pos_w[:, 2] - top_z) / BASE_ABOVE_TOP_TARGET).clamp(0.0, 1.0)
    foot_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    lift = ((foot_z.min(dim=-1).values - STAND_FOOT_Z) / FREE_FOOT_LIFT).clamp(0.0, 1.0)
    return 0.15 * p_foot + 0.85 * c * (0.3 + 0.35 * h + 0.35 * lift)


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
    reward = torch.where(env.episode_length_buf <= 1, torch.zeros_like(phi), gamma * phi_next - prev)
    env._mount_phi_prev = phi.detach().clone()
    return reward


def stand_tall_on_ball(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
) -> torch.Tensor:
    """With a foot on the ball: exp(-((h - target) / std)^2), h = base height above the ball top.

    Rewards standing up on the ball instead of holding a deep crouch.
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1)
    ball = env.scene[ball_cfg.name]
    top_z = ball.data.root_link_pos_w[:, 2] + _ball_radius(env, ball_cfg, ball_radius)
    h = env.scene[foot_cfg.name].data.root_link_pos_w[:, 2] - top_z
    err = (h - BASE_ABOVE_TOP_TARGET) / BASE_ABOVE_TOP_STD
    return on_ball.float() * torch.exp(-err.pow(2))


def free_foot_lift(
    env,
    foot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    ball_radius: float | None,
    flat: bool,
) -> torch.Tensor:
    """With one foot on the ball: how far the other foot is lifted, 0 to 1 at FREE_FOOT_LIFT."""
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat)
    foot_z = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    lift = ((foot_z - STAND_FOOT_Z) / FREE_FOOT_LIFT).clamp(0.0, 1.0)
    free = on_ball.flip(-1) & ~on_ball
    return (free.float() * lift).amax(dim=-1)


def sole_flat(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None
) -> torch.Tensor:
    """Dense: (1 + cos(tilt)) / 2 for a foot that is on the ball.

    Linear in cos(tilt), so it still has a gradient when the foot is rolled
    far over (an exp(-tilt^2/std^2) term is ~0 there and gave no signal).
    """
    on_ball = _foot_on_ball(env, foot_cfg, ball_cfg, ball_radius)
    flat = 0.5 * (1.0 + torch.cos(_sole_tilt(env, foot_cfg, ball_cfg)))
    return (on_ball.float() * flat).max(dim=-1).values


def _use_ball_size_rewards(cfg: ManagerBasedRlEnvCfg, ball_params: dict, flat_sole: bool) -> None:
    """Swap Ball-Small's on-ball and single-leg rewards for the size-aware ones.

    Ball-Small's assume the size-5 ball; single-leg also makes the other foot
    really lift. `flat_sole` adds the flat-sole requirement and its dense reward.
    """
    cfg.rewards["foot_on_ball"].func = foot_on_ball_reward
    cfg.rewards["foot_on_ball"].params = {**ball_params, "flat": flat_sole}
    cfg.rewards["single_leg_stance"].func = single_leg_stance_lifted
    cfg.rewards["single_leg_stance"].params = {**ball_params, "flat": flat_sole}
    if flat_sole:
        cfg.rewards["sole_flat"] = RewardTermCfg(func=sole_flat, weight=2.0, params=ball_params)


def piplus_ball_mount_env_cfg(
    play: bool = False, flat_sole: bool = False, ball_size: int = 5
) -> ManagerBasedRlEnvCfg:
    """`flat_sole` also requires the sole flat on the ball; `ball_size` is the FIFA size."""
    radius, mass = BALL_SIZES[ball_size]
    cfg = piplus_ball_small_env_cfg(play)
    robot_cfg = SceneEntityCfg("robot")
    ball_cfg = SceneEntityCfg("ball")
    foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
    ball_params = {"foot_cfg": foot_cfg, "ball_cfg": ball_cfg, "ball_radius": radius}

    # Spawn standing on the floor, ball ahead of the right foot.
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, STAND_Z)
    ball = cfg.scene.entities["ball"]
    ball.spec_fn = partial(get_ball_spec, radius=radius, mass=mass)
    ball.init_state.pos = (BALL_GAP_X + radius, RIGHT_FOOT_Y, radius)
    cfg.events["ball_spawn_offset"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": ball_cfg,
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02)},
            "velocity_range": {},
        },
    )

    # Actor needs the ball position to find it; velocity stays critic-only.
    cfg.observations["actor"].terms["ball_pos"] = ObservationTermCfg(
        func=ball_pos_base_obs,
        params={"asset_cfg": robot_cfg, "ball_cfg": ball_cfg},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    )
    cfg.observations["critic"].terms["ball_pos"] = ObservationTermCfg(
        func=ball_pos_base_obs,
        params={"asset_cfg": robot_cfg, "ball_cfg": ball_cfg},
    )

    # Dense shaping toward the ball; loosen posture so the leg can lift.
    cfg.rewards["foot_to_ball_top"] = RewardTermCfg(
        func=foot_to_ball_top, weight=2.0, params={**ball_params, "std": 0.15}
    )
    # foot_to_ball_top is flat for the first stretch of lift; this is not.
    cfg.rewards["foot_lift"] = RewardTermCfg(func=foot_lift, weight=1.0, params=ball_params)
    _use_ball_size_rewards(cfg, ball_params, flat_sole)
    cfg.rewards["posture"].weight = 0.3

    cfg.episode_length_s = 10.0 if not play else 1e9
    return cfg


def piplus_ball_balance_env_cfg(
    play: bool = False, flat_sole: bool = True, ball_size: int = 1
) -> ManagerBasedRlEnvCfg:
    """Ball-Small's drop-onto-the-ball balance task for a FIFA size `ball_size` ball.

    Actor stays proprioception-only (no ball position). Ball-Small's spawn puts
    the right foot 0.08 m off the ball centre, which misses a small ball, so the
    ball is placed under the right ankle.
    """
    radius, mass = BALL_SIZES[ball_size]
    cfg = piplus_ball_small_env_cfg(play)
    foot_cfg = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))
    ball_params = {"foot_cfg": foot_cfg, "ball_cfg": SceneEntityCfg("ball"), "ball_radius": radius}

    cfg.scene.entities["robot"].init_state.pos = (ROBOT_SPAWN_X, 0.0, drop_base_z(radius))
    ball = cfg.scene.entities["ball"]
    ball.spec_fn = partial(get_ball_spec, radius=radius, mass=mass)
    ball.init_state.pos = (0.0, RIGHT_FOOT_Y, radius)
    _use_ball_size_rewards(cfg, ball_params, flat_sole)
    return cfg


def piplus_ball_balance_ppo_runner_cfg(ball_size: int = 1) -> RslRlOnPolicyRunnerCfg:
    cfg = piplus_ball_small_ppo_runner_cfg()
    cfg.experiment_name = f"piplus_ball_balance_size{ball_size}"
    return cfg


def piplus_ball_mount_ppo_runner_cfg(
    flat_sole: bool = False, ball_size: int = 5
) -> RslRlOnPolicyRunnerCfg:
    cfg = piplus_ball_small_ppo_runner_cfg()
    # Matches mjx's task-id -> log-dir slug.
    cfg.experiment_name = (
        "piplus_ball_mount"
        + ("" if ball_size == 5 else f"_size{ball_size}")
        + ("_flat" if flat_sole else "")
    )
    return cfg
