"""Piplus locomotion: track commanded (vx, vy, ω_z) on flat terrain.

Prerequisite for all higher-level tasks (ball reception, platform locomotion).
Velocity commands are resampled every 5–10 s; 10% of envs receive zero command
so the policy also learns to stand still.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import (
    feet_air_time,
    track_angular_velocity,
    track_linear_velocity,
    upright,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommandCfg,
)
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
from mjlab_piplus.common.locomotion import FOOT_SENSOR_NAME, VEL_CMD_NAME
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg

HIDDEN_DIMS = (256, 128)
MAX_ITERATIONS = 3000


def piplus_locomotion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = SceneEntityCfg("robot")

    obs_terms = {
        **proprio_obs_terms(),
        "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": VEL_CMD_NAME},
        ),
    }
    observations = symmetric_obs_groups(obs_terms, play)

    actions = joint_pos_action(scale=0.5)

    commands: dict[str, CommandTermCfg] = {
        VEL_CMD_NAME: UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
            ),
        ),
    }

    rewards = {
        "track_lin_vel": RewardTermCfg(
            func=track_linear_velocity,
            weight=2.0,
            params={"std": 0.25, "command_name": VEL_CMD_NAME},
        ),
        "track_ang_vel": RewardTermCfg(
            func=track_angular_velocity,
            weight=0.5,
            params={"std": 0.25, "command_name": VEL_CMD_NAME},
        ),
        "upright": RewardTermCfg(
            func=upright,
            weight=1.5,
            params={
                "std": math.sqrt(0.05),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        ),
        "feet_air_time": RewardTermCfg(
            func=feet_air_time,
            weight=1.5,
            params={
                "sensor_name": FOOT_SENSOR_NAME,
                "threshold_min": 0.05,
                "threshold_max": 0.30,
                "command_name": VEL_CMD_NAME,
                "command_threshold": 0.1,
            },
        ),
        "posture": RewardTermCfg(
            func=envs_mdp.posture,
            weight=1.5,
            params={
                "std": {
                    ".*_hip_roll_joint": 0.10,
                    ".*_hip_pitch_joint": 0.40,
                    ".*_thigh_joint": 0.40,
                    ".*_calf_joint": 0.40,
                    ".*_ankle_pitch_joint": 0.20,
                    ".*_ankle_roll_joint": 0.15,
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
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.005),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0)},
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
                "pose_range": {"yaw": (-math.pi, math.pi)},
                "velocity_range": {},
                "asset_cfg": robot_cfg,
            },
        ),
        "push": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 10.0),
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                "asset_cfg": robot_cfg,
            },
        ),
    }

    foot_sensor = ContactSensorCfg(
        name=FOOT_SENSOR_NAME,
        primary=ContactMatch(
            mode="body",
            pattern=".*_ankle_roll_link",
            entity="robot",
        ),
        fields=("found",),
        track_air_time=True,
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_piplus_robot_cfg()},
            sensors=(foot_sensor,),
            num_envs=1,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        viewer=robot_viewer(distance=3.0, elevation=-15.0),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=SIM_TIMESTEP), njmax=200, nconmax=100),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S if not play else PLAY_EPISODE_LENGTH_S,
    )


def piplus_locomotion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return ppo_runner_cfg("piplus_locomotion", HIDDEN_DIMS, MAX_ITERATIONS)
