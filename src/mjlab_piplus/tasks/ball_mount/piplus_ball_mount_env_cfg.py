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
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# The ball-stance helpers moved to mjlab_piplus.common.ball_stance (public names). The old
# names stay here: other modules and scripts import them, and the callers' source (hashed by
# check C2) calls them. The `X as X` names are only re-exported.
from mjlab_piplus.common.ball_stance import (
    BALL_CONTACT_FORCE as BALL_CONTACT_FORCE,
    BASE_ABOVE_TOP_TARGET,
    COM_OVER_BALL as COM_OVER_BALL,
    COM_SIGMA,
    FOOT_BALL_SENSORS as FOOT_BALL_SENSORS,
    FREE_FOOT_LIFT,
    LIFTED_FOOT_Z as LIFTED_FOOT_Z,
    STAND_FOOT_Z,
    FOOT_ON_BALL_XY_MARGIN as FOOT_ON_BALL_XY_MARGIN,
    MAX_SOLE_TILT as MAX_SOLE_TILT,
    ON_BALL_Z_CENTER as ON_BALL_Z_CENTER,
    ON_BALL_Z_HALF as ON_BALL_Z_HALF,
    ball_contact as _ball_contact,
    ball_contact_sensors as ball_contact_sensors,
    ball_radii as _ball_radius,
    com_to_ball as _com_to_ball,
    foot_on_ball as _foot_on_ball,
    sole_clearance as _sole_clearance,
    sole_level_tilt as _sole_level_tilt,
    sole_tilt as _sole_tilt,
    support_foot as _support_foot,
)
from mjlab_piplus.common.football import BALL_SIZES, ROBOT_SPAWN_X, get_ball_spec
from mjlab_piplus.common.observations import (
    ball_pos_base_obs,
    ball_pos_until_on_ball as ball_pos_until_on_ball,
    kinematic_ball_pos as kinematic_ball_pos,
)
from mjlab_piplus.common.rewards.mounting import foot_to_ball_top, mount_potential as mount_potential
from mjlab_piplus.common.rewards.stance import single_leg_stance_lifted
from mjlab_piplus.tasks.ball_small.piplus_ball_small_env_cfg import (
    piplus_ball_small_env_cfg,
    piplus_ball_small_ppo_runner_cfg,
)

# Home-pose standing: base at 0.39 m, right ankle link ~0.05 m ahead of base
# and 0.0815 m to the right. Ball centre sits 0.19 m + radius in front of the
# base (0.30 m for size 5), on the right-foot line.
STAND_Z = 0.39
RIGHT_FOOT_Y = -0.0815
BALL_GAP_X = 0.19

# Drop spawn: the robot starts just above the ball at the home pose, so it
# settles onto it instead of falling onto it. Right ankle is ~0.29 m below the
# base at the home pose and ~0.05 m above the ball top with the sole flat.
HOME_BASE_ABOVE_ANKLE = 0.29
FLAT_ANKLE_ABOVE_TOP = 0.05
DROP_CLEARANCE = 0.03  # m between sole and ball top at spawn

# Standing tall on the ball: base height above the ball top with the leg at its
# loaded home pose (0.34 m unloaded, ~5 cm of PD sag). The trained crouch sat at ~0.19 m.
BASE_ABOVE_TOP_STD = 0.10  # BASE_ABOVE_TOP_TARGET is in common.ball_stance


# Weight on the ball: COM_OVER_BALL and COM_SIGMA are in common.ball_stance.
COM_VEL_CAP = 0.5  # m/s cap for the momentum rewards


def drop_base_z(radius):
    """Base height of the drop spawn for a ball of this radius (float or tensor)."""
    return 2 * radius + FLAT_ANKLE_ABOVE_TOP + HOME_BASE_ABOVE_ANKLE + DROP_CLEARANCE


def foot_lift(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None
) -> torch.Tensor:
    """Highest foot's lift from standing height, scaled to 1 at the ball top [B]."""
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    foot_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2].max(dim=-1).values
    top_z = ball.data.root_link_pos_w[:, 2] + _ball_radius(env, ball_cfg, ball_radius)
    return ((foot_z - STAND_FOOT_Z) / (top_z - STAND_FOOT_Z)).clamp(0.0, 1.0)


# Contact sensors between each ankle roll link and the ball: FOOT_BALL_SENSORS in
# mjlab_piplus.common.ball_stance, read by ball_contact() there.


def foot_on_ball_reward(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None, flat: bool
) -> torch.Tensor:
    return _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1).float()


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
    linear_range: float | None = None,
) -> torch.Tensor:
    """With a foot on the ball: exp(-(d / COM_SIGMA)^2), d = horizontal CoM-to-ball distance.

    Dense signal for shifting the weight onto the ball (broad: still has a gradient at 0.25 m).
    The exp kernel is already at 0.89 in the straddle (d = 0.087 m; a real stance has d =
    0.011 m), so the last stretch pays almost nothing. `linear_range` (m) uses
    1 - d / linear_range instead: the same slope all the way to the ball centre.
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1)
    d = _com_to_ball(env, foot_cfg, ball_cfg).norm(dim=-1)
    if linear_range is not None:
        return on_ball.float() * (1.0 - d / linear_range).clamp(0.0, 1.0)
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
    sole: bool = False,
    free_off_ball: bool = False,
) -> torch.Tensor:
    """With one foot on the ball: how far the other foot is lifted, 0 to 1 at FREE_FOOT_LIFT.

    `sole` measures the free sole's lowest corner instead of the ankle height (tiptoe pays
    up to ~0.45 of this term with the ankle height). `free_off_ball`: 0 while the free foot
    touches the ball (see single_leg_stance_lifted)."""
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat)
    if sole:
        height = _sole_clearance(env, foot_cfg)
    else:
        height = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2] - STAND_FOOT_Z
    lift = (height / FREE_FOOT_LIFT).clamp(0.0, 1.0)
    free = on_ball.flip(-1) & ~on_ball
    if free_off_ball:
        free = free & ~_ball_contact(env)
    return (free.float() * lift).amax(dim=-1)


def sole_flat(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float | None, level: bool = False
) -> torch.Tensor:
    """Dense: (1 + cos(tilt)) / 2 for a foot that is on the ball.

    Linear in cos(tilt), so it still has a gradient when the foot is rolled
    far over (an exp(-tilt^2/std^2) term is ~0 there and gave no signal).
    `level`: tilt is the larger of the tilt against the ball surface and against horizontal.
    """
    on_ball = _foot_on_ball(env, foot_cfg, ball_cfg, ball_radius)
    tilt = _sole_tilt(env, foot_cfg, ball_cfg)
    if level:
        tilt = torch.maximum(tilt, _sole_level_tilt(env, foot_cfg))
    flat = 0.5 * (1.0 + torch.cos(tilt))
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
