"""Piplus upright-only environment: the sole reward is staying upright."""

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import upright
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_lukas.robot.piplus_constants import get_piplus_robot_cfg


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def foot_flatness(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  # body_quat_w: [B, num_bodies, 4] as (w, x, y, z)
  quat = asset.data.body_com_quat_w[:, asset_cfg.body_ids, :]  # [B, 2, 4]
  _, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  # z-component of the body's local z-axis in world frame (R[2,2] = 1 - 2(x²+y²))
  # equals 1 when foot sole is flat, decreases with tilt
  foot_up_z = 1 - 2 * (x * x + y * y)  # [B, 2]
  return foot_up_z.clamp(0, 1).mean(dim=-1)  # [B]


def piplus_upright_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  obs_terms = {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=obs_terms, enable_corruption=not play),
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
      weight=1.0,
      params={
        "std": math.sqrt(0.05),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "base_height": RewardTermCfg(
      func=base_height_l2,
      weight=-2.0,
      params={"target_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-1.0,
    ),
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=2.0,
      params={
        "std": {
          ".*_hip_roll_joint": 0.05,
          ".*_hip_pitch_joint": 0.25,
          ".*_thigh_joint": 0.25,
          ".*_calf_joint": 0.25,
          ".*_ankle_pitch_joint": 0.05,
          ".*_ankle_roll_joint": 0.25,
        },
        "asset_cfg": SceneEntityCfg("robot", joint_names=(
          ".*_hip_roll_joint",
          ".*_hip_pitch_joint",
          ".*_thigh_joint",
          ".*_calf_joint",
          ".*_ankle_pitch_joint",
          ".*_ankle_roll_joint",
        )),
      },
    ),
    "foot_flatness": RewardTermCfg(
      func=foot_flatness,
      weight=1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=["r_ankle_roll_link", "l_ankle_roll_link"])},
    ),
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.005,
    ),
    "joint_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.01,
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
  }

  events = {
    "push": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 10.0),
      params={
        "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_piplus_robot_cfg()},
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
      distance=2.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005), njmax=200),
    decimation=4,
    episode_length_s=20.0 if not play else 1e9,
  )


def piplus_upright_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="piplus_upright",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
