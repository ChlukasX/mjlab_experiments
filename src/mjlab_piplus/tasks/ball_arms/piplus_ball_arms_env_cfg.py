"""Piplus ball-balancing with full arm actuation (20 DOF).

Identical to Mjlab-Piplus-Ball but uses get_piplus_robot_with_arms_cfg().
Arm swing can contribute to angular momentum management on the ball.
"""

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import upright
from mjlab.terrains import TerrainEntityCfg

from mjlab_piplus.common.balls import (
    BALL_MASS,  # noqa: F401  (re-export)
    BALL_RADIUS,
    BALL_TOP,  # noqa: F401  (re-export)
    BASE_HEIGHT_TARGET,
    ROBOT_SPAWN_Z,
    ball_control,
    ball_escaped,
    ball_state_obs,
    get_ball_spec,
    robot_off_ball,
)
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
from mjlab_piplus.common.rewards import base_height_gauss, xy_centering
from mjlab_piplus.robot.piplus_constants import (
    FULL_COLLISION,
    HOME_KEYFRAME_WITH_ARMS,
)

HIDDEN_DIMS = (128, 128)
MAX_ITERATIONS = 1000


def get_piplus_on_ball_with_arms_cfg() -> EntityCfg:
    spawn_state = EntityCfg.InitialStateCfg(
        pos=(0, 0, ROBOT_SPAWN_Z),
        joint_pos=HOME_KEYFRAME_WITH_ARMS.joint_pos,
        joint_vel=HOME_KEYFRAME_WITH_ARMS.joint_vel,
    )
    from mjlab_piplus.robot.piplus_constants import (
        PIPLUS_ARTICULATION_WITH_ARMS,
        get_spec_with_arms,
    )

    return EntityCfg(
        init_state=spawn_state,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec_with_arms,
        articulation=PIPLUS_ARTICULATION_WITH_ARMS,
    )


def piplus_ball_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = SceneEntityCfg("robot")
    ball_cfg = SceneEntityCfg("ball")

    obs_terms = {
        **proprio_obs_terms(),
        "ball_state": ObservationTermCfg(
            func=ball_state_obs,
            params={"asset_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
    }
    observations = symmetric_obs_groups(obs_terms, play)

    actions = joint_pos_action(scale=0.5)

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
                    ".*_hip_roll_joint": 0.05,
                    ".*_hip_pitch_joint": 0.40,
                    ".*_thigh_joint": 0.40,
                    ".*_calf_joint": 0.40,
                    ".*_ankle_pitch_joint": 0.20,
                    ".*_ankle_roll_joint": 0.25,
                    # arms loose — let them find useful poses
                    ".*_shoulder_pitch_joint": 0.40,
                    ".*_shoulder_roll_joint": 0.20,
                    ".*_upper_arm_joint": 0.50,
                    ".*_elbow_joint": 0.30,
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
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_upper_arm_joint",
                        ".*_elbow_joint",
                    ),
                ),
            },
        ),
        "ball_control": RewardTermCfg(
            func=ball_control, weight=3.0, params={"std": 0.5, "ball_cfg": ball_cfg}
        ),
        "xy_centering": RewardTermCfg(
            func=xy_centering, weight=0.5, params={"asset_cfg": robot_cfg}
        ),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
        "joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.005),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
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
                "ball": EntityCfg(
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
        viewer=robot_viewer(distance=4.0, elevation=-20.0),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=SIM_TIMESTEP), njmax=200, nconmax=100),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S if not play else PLAY_EPISODE_LENGTH_S,
    )


def piplus_ball_arms_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return ppo_runner_cfg("piplus_ball_arms", HIDDEN_DIMS, MAX_ITERATIONS)
