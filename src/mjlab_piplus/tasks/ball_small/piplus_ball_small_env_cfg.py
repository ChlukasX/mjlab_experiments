"""Piplus small-ball balancing — two-stage curriculum.

Robot spawns directly above ball and drops onto it (same pattern as large-ball
task).  No approach phase — straight to balance training.

Stage 1 (step 0–12000, ~iter 0–500): foot on ball, stay upright.
Stage 2 (step 12000+,  ~iter 500+): single-leg stance — other foot lifted.

Ball: standard football (radius=0.11 m, mass=0.43 kg).
Arms enabled from start — critical for single-leg counterbalancing.
"""

import math

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import illegal_contact, upright
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_piplus.robot.piplus_constants import (
    FULL_COLLISION,
    HOME_KEYFRAME_WITH_ARMS,
    PIPLUS_ARTICULATION_WITH_ARMS,
    get_spec_with_arms,
)
from mjlab_piplus.terrains import AstroturfTerrainCfg, ground_softness

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALL_RADIUS = 0.11
BALL_MASS   = 0.43
BALL_TOP    = BALL_RADIUS * 2          # = 0.22 m
ROBOT_SPAWN_Z = BALL_TOP + 0.5        # = 0.72 m — drops onto ball
# Offset right foot over ball top; left foot lands on floor beside ball.
# Right ankle_roll is ~0.05 m lateral from base centre — shift base by same.
ROBOT_SPAWN_X = -0.05                 # base shifted left so right foot is over ball

# Base height when standing on flat floor is ~0.39 m; below this the robot is
# sitting/lying (e.g. slumped against the ball) even if feet stay on the ball.
MIN_BASE_HEIGHT = 0.30

FOOT_ON_BALL_Z_MARGIN  = 0.06
FOOT_ON_BALL_XY_MARGIN = 0.05

STAGE2_STEPS = 0   # single_leg active immediately — spawn already places one foot on ball


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball_body", pos=[0, 0, BALL_RADIUS])
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS],
        mass=BALL_MASS,
        rgba=[0.2, 0.6, 0.2, 1.0],
        contype=1,
        conaffinity=1,
        condim=6,
        friction=[0.8, 0.01, 0.001],
    )
    return spec


def get_piplus_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(ROBOT_SPAWN_X, 0, ROBOT_SPAWN_Z),
            joint_pos=HOME_KEYFRAME_WITH_ARMS.joint_pos,
            joint_vel=HOME_KEYFRAME_WITH_ARMS.joint_vel,
        ),
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec_with_arms,
        articulation=PIPLUS_ARTICULATION_WITH_ARMS,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _foot_on_ball(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """[B, 2] bool: which ankle_roll links are on top of the ball."""
    robot = env.scene[foot_cfg.name]
    ball  = env.scene[ball_cfg.name]
    foot_pos  = robot.data.body_link_pose_w[:, foot_cfg.body_ids, :3]  # [B, 2, 3]
    ball_pos  = ball.data.root_link_pos_w                               # [B, 3]
    ball_top_z = ball_pos[:, 2] + BALL_RADIUS                          # [B]

    z_ok  = (foot_pos[:, :, 2] - ball_top_z.unsqueeze(-1)).abs() < FOOT_ON_BALL_Z_MARGIN
    horiz = (foot_pos[:, :, :2] - ball_pos[:, :2].unsqueeze(1)).norm(dim=-1)
    xy_ok = horiz < (BALL_RADIUS + FOOT_ON_BALL_XY_MARGIN)
    return z_ok & xy_ok


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def ball_rel_obs(env, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball pos relative to robot base + ball xy vel [B, 5]."""
    robot = env.scene[asset_cfg.name]
    ball  = env.scene[ball_cfg.name]
    rel_pos  = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    ball_vel = ball.data.root_link_vel_w[:, :2]
    return torch.cat([rel_pos, ball_vel], dim=-1)


def foot_ball_dist_obs(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """XYZ offset from each foot to ball centre [B, 6]."""
    robot    = env.scene[foot_cfg.name]
    ball     = env.scene[ball_cfg.name]
    foot_pos = robot.data.body_link_pose_w[:, foot_cfg.body_ids, :3]  # [B, 2, 3]
    diff     = ball.data.root_link_pos_w.unsqueeze(1) - foot_pos
    return diff.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def foot_on_ball_reward(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for at least one foot on the ball (active from step 0)."""
    on_ball = _foot_on_ball(env, foot_cfg, ball_cfg)
    return on_ball.any(dim=-1).float()


def single_leg_stance(env, foot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Stage 2: one foot on ball, other foot off ground."""
    if env.common_step_counter < STAGE2_STEPS:
        return torch.zeros(env.num_envs, device=env.device)
    on_ball    = _foot_on_ball(env, foot_cfg, ball_cfg)
    foot_z     = env.scene[foot_cfg.name].data.body_link_pose_w[:, foot_cfg.body_ids, 2]
    off_ground = foot_z > 0.02
    one_on_ball   = on_ball.any(dim=-1)
    other_lifted  = (on_ball ^ off_ground).any(dim=-1)
    return (one_on_ball & other_lifted).float()


def xy_centering(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    drift = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return -drift.pow(2).sum(dim=-1)


def ball_escaped(env, max_dist: float, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    dist_sq = (
        (ball.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2])
        .pow(2).sum(dim=-1)
    )
    return dist_sq > max_dist ** 2


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

def piplus_ball_small_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = SceneEntityCfg("robot")
    ball_cfg  = SceneEntityCfg("ball")
    foot_cfg  = SceneEntityCfg("robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link"))

    # Actor: only what the real Pi Plus measures (joint encoders + IMU).
    # Noise levels follow mjlab's velocity task.
    actor_terms = {
        "joint_pos":         ObservationTermCfg(func=envs_mdp.joint_pos_rel,     noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel":         ObservationTermCfg(func=envs_mdp.joint_vel_rel,     noise=Unoise(n_min=-1.5,  n_max=1.5)),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "base_ang_vel":      ObservationTermCfg(func=envs_mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2)),
        "actions":           ObservationTermCfg(func=envs_mdp.last_action),
    }
    # Critic: privileged — also base lin vel and ball state (sim only).
    critic_terms = {
        **actor_terms,
        "base_lin_vel":      ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "ball_rel":          ObservationTermCfg(
            func=ball_rel_obs,
            params={"asset_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
        "foot_ball_dist":    ObservationTermCfg(
            func=foot_ball_dist_obs,
            params={"foot_cfg": foot_cfg, "ball_cfg": ball_cfg},
        ),
    }
    observations = {
        # History lets the proprio-only actor infer ball motion from its own dynamics.
        "actor":  ObservationGroupCfg(terms=actor_terms, enable_corruption=not play, history_length=5),
        "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    rewards = {
        "foot_on_ball": RewardTermCfg(
            func=foot_on_ball_reward,
            weight=4.0,
            params={"foot_cfg": foot_cfg, "ball_cfg": ball_cfg},
        ),
        "single_leg_stance": RewardTermCfg(
            func=single_leg_stance,
            weight=4.0,
            params={"foot_cfg": foot_cfg, "ball_cfg": ball_cfg},
        ),
        "upright": RewardTermCfg(
            func=upright,
            weight=2.0,
            params={
                "std": math.sqrt(0.05),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        ),
        "posture": RewardTermCfg(
            func=envs_mdp.posture,
            weight=1.0,
            params={
                "std": {
                    ".*_hip_roll_joint":       0.10,
                    ".*_hip_pitch_joint":      0.40,
                    ".*_thigh_joint":          0.40,
                    ".*_calf_joint":           0.40,
                    ".*_ankle_pitch_joint":    0.20,
                    ".*_ankle_roll_joint":     0.15,
                    ".*_shoulder_pitch_joint": 0.40,
                    ".*_shoulder_roll_joint":  0.20,
                    ".*_upper_arm_joint":      0.50,
                    ".*_elbow_joint":          0.30,
                },
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_thigh_joint", ".*_calf_joint",
                    ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                    ".*_upper_arm_joint", ".*_elbow_joint",
                )),
            },
        ),
        "xy_centering":     RewardTermCfg(func=xy_centering,              weight=0.3,  params={"asset_cfg": robot_cfg}),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate":      RewardTermCfg(func=envs_mdp.action_rate_l2,   weight=-0.01),
        "joint_vel":        RewardTermCfg(func=envs_mdp.joint_vel_l2,     weight=-0.005),
    }

    terminations = {
        "time_out":     TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over":    TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0)},
        ),
        # Any robot body other than the feet touching the floor = fallen.
        # Catches falls where feet stay on the ball and orientation stays < 60°.
        "body_on_floor": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": "body_floor_contact"},
        ),
        "base_too_low": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum,
            params={"minimum_height": MIN_BASE_HEIGHT},
        ),
        "ball_escaped": TerminationTermCfg(
            func=ball_escaped,
            params={"max_dist": 3.0, "ball_cfg": ball_cfg},
        ),
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        # --- Sim2real domain randomisation -------------------------------
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_ankle_roll_link_collision[0-9]$",)),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "ball_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_geom",)),
                "operation": "abs",
                "ranges": (0.4, 1.2),
            },
        ),
        # pseudo_inertia scales mass and inertia together by exp(2*alpha).
        "ball_mass": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("ball", body_names=("ball_body",)),
                "alpha_range": (math.log(0.8) / 2, math.log(1.2) / 2),  # mass x0.8-1.2
            },
        ),
        "robot_mass": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link", "torso_link")),
                "alpha_range": (math.log(0.9) / 2, math.log(1.1) / 2),  # mass x0.9-1.1
            },
        ),
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        "pd_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
            },
        ),
        # Gentler than locomotion pushes — single-leg on a 22 cm ball.
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 5.0),
            params={
                "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2),
                    "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.3, 0.3),
                },
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_piplus_cfg(),
                "ball":  EntityCfg(
                    spec_fn=get_ball_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, BALL_RADIUS)),
                ),
            },
            sensors=(
                ContactSensorCfg(
                    name="body_floor_contact",
                    primary=ContactMatch(
                        mode="body",
                        pattern=".*",
                        entity="robot",
                        exclude=(r".*_ankle_roll_link",),
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found",),
                    reduce="netforce",
                    num_slots=1,
                ),
            ),
            num_envs=1,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=3.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005), njmax=200, nconmax=100),
        decimation=4,
        episode_length_s=20.0 if not play else 1e9,
    )


def piplus_ball_small_turf_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Ball-small on astroturf; turf softens from firm to soft over ~2000 iters."""
    cfg = piplus_ball_small_env_cfg(play)
    cfg.scene.terrain = AstroturfTerrainCfg()
    cfg.events["ground_softness"] = EventTermCfg(
        func=ground_softness,
        mode="reset",
        # ponytail: solref timeconst range is a guess; calibrate against real foot-sink tests.
        params={"timeconst_range": (0.02, 0.06), "ramp_steps": 2000 * 24},
    )
    return cfg


def piplus_ball_small_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="piplus_ball_small",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=3000,
    )
