"""Piplus ball-balancing with full arm actuation (20 DOF).

Identical to Mjlab-Piplus-Ball but uses get_piplus_robot_with_arms_cfg().
Arm swing can contribute to angular momentum management on the ball.
"""

import math

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import upright
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_lukas.robot.piplus_constants import (
    HOME_KEYFRAME_WITH_ARMS,
    get_piplus_robot_with_arms_cfg,
    FULL_COLLISION,
)

BALL_RADIUS = 0.45
BALL_MASS = 5.0
BALL_TOP = BALL_RADIUS * 2
ROBOT_SPAWN_Z = BALL_TOP + 0.5
BASE_HEIGHT_TARGET = BALL_TOP + 0.35


def get_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball_body", pos=[0, 0, BALL_RADIUS])
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS],
        mass=BALL_MASS,
        rgba=[0.85, 0.35, 0.10, 1.0],
        contype=1,
        conaffinity=1,
        condim=3,
        friction=[0.9, 0.02, 0.001],
    )
    return spec


def get_piplus_on_ball_with_arms_cfg() -> EntityCfg:
    spawn_state = EntityCfg.InitialStateCfg(
        pos=(0, 0, ROBOT_SPAWN_Z),
        joint_pos=HOME_KEYFRAME_WITH_ARMS.joint_pos,
        joint_vel=HOME_KEYFRAME_WITH_ARMS.joint_vel,
    )
    from mjlab_lukas.robot.piplus_constants import PIPLUS_ARTICULATION_WITH_ARMS, get_spec_with_arms
    return EntityCfg(
        init_state=spawn_state,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec_with_arms,
        articulation=PIPLUS_ARTICULATION_WITH_ARMS,
    )


def ball_state_obs(env, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel_pos = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    ball_vel = ball.data.root_link_vel_w[:, :2]
    return torch.cat([rel_pos, ball_vel], dim=-1)


def base_height_gauss(env, target_height: float, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    err = asset.data.root_link_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(err) / (std * std))


def ball_control(env, std: float, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    vel_sq = ball.data.root_link_vel_w[:, :3].pow(2).sum(dim=-1)
    return torch.exp(-vel_sq / (std * std))


def ball_escaped(env, max_dist: float, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    dist_sq = (
        (ball.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2])
        .pow(2).sum(dim=-1)
    )
    return dist_sq > max_dist ** 2


def xy_centering(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    drift = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return -drift.pow(2).sum(dim=-1)


def robot_off_ball(
    env,
    min_height_above_ball: float,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    ball_top_z = ball.data.root_link_pos_w[:, 2] + BALL_RADIUS
    robot_z = robot.data.root_link_pos_w[:, 2]
    return robot_z < ball_top_z + min_height_above_ball


def piplus_ball_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = SceneEntityCfg("robot")
    ball_cfg = SceneEntityCfg("ball")

    obs_terms = {
        "joint_pos":         ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel":         ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "actions":           ObservationTermCfg(func=envs_mdp.last_action),
        "ball_state":        ObservationTermCfg(
            func=ball_state_obs,
            params={"asset_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
    }
    observations = {
        "actor":  ObservationGroupCfg(terms=obs_terms, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=obs_terms, enable_corruption=False),
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
        "upright": RewardTermCfg(
            func=upright,
            weight=1.5,
            params={
                "std": math.sqrt(0.05),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        ),
        "base_height": RewardTermCfg(
            func=base_height_gauss,
            weight=1.5,
            params={
                "target_height": BASE_HEIGHT_TARGET,
                "std": 0.15,
                "asset_cfg": robot_cfg,
            },
        ),
        "posture": RewardTermCfg(
            func=envs_mdp.posture,
            weight=3.0,
            params={
                "std": {
                    ".*_hip_roll_joint":       0.05,
                    ".*_hip_pitch_joint":      0.40,
                    ".*_thigh_joint":          0.40,
                    ".*_calf_joint":           0.40,
                    ".*_ankle_pitch_joint":    0.20,
                    ".*_ankle_roll_joint":     0.25,
                    # arms loose — let them find useful poses
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
        "ball_control":     RewardTermCfg(func=ball_control,     weight=3.0,  params={"std": 0.5, "ball_cfg": ball_cfg}),
        "xy_centering":     RewardTermCfg(func=xy_centering,    weight=0.5,  params={"asset_cfg": robot_cfg}),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate":      RewardTermCfg(func=envs_mdp.action_rate_l2,   weight=-0.05),
        "joint_vel":        RewardTermCfg(func=envs_mdp.joint_vel_l2,     weight=-0.005),
    }

    terminations = {
        "time_out":     TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over":    TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "ball_escaped": TerminationTermCfg(
            func=ball_escaped,
            params={"max_dist": 3.0, "ball_cfg": ball_cfg},
        ),
        "robot_off_ball": TerminationTermCfg(
            func=robot_off_ball,
            params={"min_height_above_ball": 0.3, "asset_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "randomize_yaw": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"z": (ROBOT_SPAWN_Z, ROBOT_SPAWN_Z), "yaw": (-math.pi, math.pi)},
                "velocity_range": {},
                "asset_cfg": robot_cfg,
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_piplus_on_ball_with_arms_cfg(),
                "ball":  EntityCfg(
                    spec_fn=get_ball_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=(0, 0, BALL_RADIUS)),
                ),
            },
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
            distance=4.0,
            elevation=-20.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005), njmax=200, nconmax=100),
        decimation=4,
        episode_length_s=20.0 if not play else 1e9,
    )


def piplus_ball_arms_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 128),
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
        experiment_name="piplus_ball_arms",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=1000,
    )
