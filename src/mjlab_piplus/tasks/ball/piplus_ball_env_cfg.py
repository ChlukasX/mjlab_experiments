"""Piplus ball-balancing experiment.

The robot stands on top of a free-floating sphere.  Its foot forces transmit into
the ball and cause it to roll — the robot must learn to counteract its own momentum
or it slides off.  Instability is entirely self-generated (no injected forces or
platform motion).

Ball geometry:
    radius = 0.45 m  (roughly an exercise ball relative to robot scale)
    mass   = 5.0 kg  (light enough to respond clearly to foot forces)
"""

import math

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
from mjlab_piplus.common.rewards import base_height_gauss, xy_centering
from mjlab_piplus.robot.piplus_constants import (
    ACTUATOR_5036,
    FULL_COLLISION,
    HOME_KEYFRAME,
    PIPLUS_ARTICULATION,
    get_spec,
)

# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_piplus_on_ball_cfg() -> EntityCfg:
    spawn_state = EntityCfg.InitialStateCfg(
        pos=(0, 0, ROBOT_SPAWN_Z),
        joint_pos=HOME_KEYFRAME.joint_pos,
        joint_vel=HOME_KEYFRAME.joint_vel,
    )
    return EntityCfg(
        init_state=spawn_state,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=PIPLUS_ARTICULATION,
    )


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

def piplus_ball_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "posture": RewardTermCfg(
            func=envs_mdp.posture,
            weight=3.0,
            params={
                "std": {
                    ".*_hip_roll_joint":    0.05,
                    ".*_hip_pitch_joint":   0.40,
                    ".*_thigh_joint":       0.40,
                    ".*_calf_joint":        0.40,
                    ".*_ankle_pitch_joint": 0.20,
                    ".*_ankle_roll_joint":  0.25,
                },
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*_hip_roll_joint", ".*_hip_pitch_joint",
                    ".*_thigh_joint", ".*_calf_joint",
                    ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                )),
            },
        ),
        "ball_control": RewardTermCfg(
            func=ball_control,
            weight=3.0,
            params={"std": 0.5, "ball_cfg": ball_cfg},
        ),
        "xy_centering": RewardTermCfg(
            func=xy_centering,
            weight=0.5,
            params={"asset_cfg": robot_cfg},
        ),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
        "joint_vel":   RewardTermCfg(func=envs_mdp.joint_vel_l2,   weight=-0.005),
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
                "robot": get_piplus_on_ball_cfg(),
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


def piplus_ball_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
        experiment_name="piplus_ball",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=1000,
    )
