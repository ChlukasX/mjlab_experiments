"""Piplus upright-only environment: the sole reward is staying upright."""

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import upright
from mjlab.terrains import TerrainEntityCfg

from mjlab_piplus.common.env_parts import (
    DECIMATION,
    EPISODE_LENGTH_S,
    PLAY_EPISODE_LENGTH_S,
    SIM_TIMESTEP,
    joint_pos_action,
    ppo_runner_cfg,
    proprio_obs_terms,
    robot_viewer,
    symmetric_obs_groups,
)
from mjlab_piplus.common.rewards import foot_flatness
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg

HIDDEN_DIMS = (128, 128)
MAX_ITERATIONS = 1000


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def piplus_upright_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    obs_terms = proprio_obs_terms()
    observations = symmetric_obs_groups(obs_terms, play)

    actions = joint_pos_action(scale=0.5)

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
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        ".*_hip_roll_joint",
                        ".*_hip_pitch_joint",
                        ".*_thigh_joint",
                        ".*_calf_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                    ),
                ),
            },
        ),
        "foot_flatness": RewardTermCfg(
            func=foot_flatness,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["r_ankle_roll_link", "l_ankle_roll_link"]
                )
            },
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
        viewer=robot_viewer(distance=2.5, elevation=-10.0),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=SIM_TIMESTEP), njmax=200),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S if not play else PLAY_EPISODE_LENGTH_S,
    )


def piplus_upright_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return ppo_runner_cfg("piplus_upright", HIDDEN_DIMS, MAX_ITERATIONS)
