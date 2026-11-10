"""Env-config parts that several task families share.

Boilerplate has a default here. A design choice (action scale, network size, iteration count)
is an argument that the task states in its own module. A function returns a fresh object on
every call, so two tasks never share an instance. Each part reproduces the literal that the
first families wrote, so the fingerprint (check C2) of every old task stays equal.
"""

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.viewer import ViewerConfig

SIM_TIMESTEP = 0.005
DECIMATION = 4
EPISODE_LENGTH_S = 20.0
PLAY_EPISODE_LENGTH_S = 1e9


def proprio_obs_terms() -> dict[str, ObservationTermCfg]:
    """Joint position, joint velocity, projected gravity, and the last action."""
    return {
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }


def symmetric_obs_groups(
    terms: dict[str, ObservationTermCfg], play: bool
) -> dict[str, ObservationGroupCfg]:
    """Actor and critic read the same terms. Only the actor sees noise (off when playing)."""
    return {
        "actor": ObservationGroupCfg(terms=terms, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=terms, enable_corruption=False),
    }


def joint_pos_action(scale: float) -> dict[str, ActionTermCfg]:
    """Joint position targets around the default pose, all joints."""
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=scale,
            use_default_offset=True,
        )
    }


def robot_viewer(distance: float, elevation: float) -> ViewerConfig:
    """Viewer that follows the robot base from the side (azimuth 90 deg)."""
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="base_link",
        distance=distance,
        elevation=elevation,
        azimuth=90.0,
    )


def ppo_runner_cfg(
    experiment_name: str, hidden_dims: tuple[int, ...], max_iterations: int
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner of the early families: elu MLPs, adaptive KL schedule, 24 steps per env."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=hidden_dims,
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
        experiment_name=experiment_name,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )
