"""Piplus ball mount (feasibility spike): step from the floor onto a football.

Robot spawns standing at the home pose, ball just ahead of the right foot.
Same 20 DOF, rewards and randomisation as Ball-Small, plus a dense
foot-to-ball-top reward. The actor additionally sees the ball position in the
base frame (privileged for now; real perception comes in a later phase).

See docs/mount_feasibility.md.
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


def _foot_on_ball(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float
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
    ankle_z = ball_pos[:, 2] + ball_radius + ON_BALL_Z_CENTER
    z_ok = (foot_pos[:, :, 2] - ankle_z.unsqueeze(-1)).abs() < ON_BALL_Z_HALF
    horiz = (foot_pos[:, :, :2] - ball_pos[:, :2].unsqueeze(1)).norm(dim=-1)
    return z_ok & (horiz < ball_radius + FOOT_ON_BALL_XY_MARGIN)


def ball_pos_base_obs(env, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball position relative to the base, in the base frame [B, 3]."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel)


def foot_to_ball_top(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float, std: float
) -> torch.Tensor:
    """exp(-d^2/std^2) of the nearest foot's distance to the ball top point."""
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    foot = robot.data.body_link_pose_w[:, foot_cfg.body_ids, :3]  # [B, 2, 3]
    top = ball.data.root_link_pos_w.clone()
    top[:, 2] += ball_radius
    d2 = (foot - top.unsqueeze(1)).pow(2).sum(dim=-1).min(dim=-1).values
    return torch.exp(-d2 / std**2)


def foot_lift(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float
) -> torch.Tensor:
    """Highest foot's lift from standing height, scaled to 1 at the ball top [B]."""
    robot = env.scene[foot_cfg.name]
    ball = env.scene[ball_cfg.name]
    foot_z = robot.data.body_link_pose_w[:, foot_cfg.body_ids, 2].max(dim=-1).values
    top_z = ball.data.root_link_pos_w[:, 2] + ball_radius
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
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float, flat: bool
) -> torch.Tensor:
    """[B, 2] bool: foot on the ball (and the sole flat against it, if `flat`)."""
    on_ball = _foot_on_ball(env, foot_cfg, ball_cfg, ball_radius)
    if flat:
        on_ball = on_ball & (_sole_tilt(env, foot_cfg, ball_cfg) < MAX_SOLE_TILT)
    return on_ball


def foot_on_ball_reward(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float, flat: bool
) -> torch.Tensor:
    return _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat).any(dim=-1).float()


def single_leg_stance_lifted(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float, flat: bool
) -> torch.Tensor:
    """One foot on the ball (sole flat if `flat`), the other clearly off the floor.

    Ball-Small's version of this check passes at floor height, so it never
    required the second foot to lift.
    """
    on_ball = _support_foot(env, foot_cfg, ball_cfg, ball_radius, flat)
    foot_z = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    other_lifted = (on_ball.flip(-1) & ~on_ball) & (foot_z > LIFTED_FOOT_Z)
    return (on_ball.any(dim=-1) & other_lifted.any(dim=-1)).float()


def sole_flat(
    env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, ball_radius: float
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

    cfg.scene.entities["robot"].init_state.pos = (ROBOT_SPAWN_X, 0.0, 2 * radius + 0.5)
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
