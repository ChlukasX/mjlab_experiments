"""Piplus moving-platform environment (Phase 2 non-inertial).

A physical 3m×3m platform body is driven kinematically each control step with
sinusoidal roll, pitch, and heave — simulating boat/ship motion.  The robot must
learn to stand on the tilting surface and resist being thrown off.

Platform motion (3 DOF):
    roll  = A_r · sin(ω_r · t + φ_r)   [rad]
    pitch = A_p · sin(ω_p · t + φ_p)   [rad]
    heave = A_h · sin(ω_h · t + φ_h)   [m]

Curriculum (common_step_counter):
    step     0 : no motion      (learn to stand on platform)
    step  6000 : gentle swell   (roll ≤ 0.10 rad, heave ≤ 0.10 m)
    step 12000 : moderate       (roll ≤ 0.20 rad, heave ≤ 0.20 m)
    step 20000 : full ship      (roll ≤ 0.30 rad, heave ≤ 0.25 m)
"""

import math

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from mjlab_lukas.robot.piplus_constants import (
    ACTUATOR_5036,
    FULL_COLLISION,
    HOME_KEYFRAME,
    PIPLUS_ARTICULATION,
    get_spec,
)

# Platform geometry
PLATFORM_Z = 0.2        # center height — bottom at z=0.15, top at z=0.25
PLATFORM_HALF_H = 0.05  # half-thickness
PLATFORM_HALF_XY = 1.5  # half-width/length (3m × 3m platform)

# Geometry-derived spawn height.
# In HOME_KEYFRAME at base_z=0.69: ankle body z=0.3985, base-to-ankle=0.2915.
# Lowest collision geom: capsule at pos z=-0.04 + radius 0.01 → 0.05 below ankle.
# base-to-lowest = 0.2915 + 0.05 = 0.3415
# Settle measurement (200 steps, zero action): base settles to ~0.615.
_BASE_TO_LOWEST = 0.3415
PLATFORM_TOP = PLATFORM_Z + PLATFORM_HALF_H            # = 0.25
ROBOT_SPAWN_Z = PLATFORM_TOP + 0.5                     # = 0.75  (0.5 m above platform)
BASE_HEIGHT_TARGET = PLATFORM_TOP + 0.35               # = 0.60  (matches noninertial convention)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def get_platform_spec() -> mujoco.MjSpec:
    """Flat box platform — no joints, auto-wrapped as mocap by mjlab.

    Uses BOX geom for collision.  The robot's feet (ankle_roll_link_collision0)
    are BOX type, so BOX-BOX contact generates multiple contact points and is
    fully supported by MuJoCo Warp (PLANE geoms on non-terrain bodies hang).
    The earlier failure was from njmax overflow (200 → 512 fixed it).
    """
    spec = mujoco.MjSpec()
    # Set the body pos in the MJCF so mocap_pos is at PLATFORM_Z immediately
    # after mj_resetData — before any entity-level reset events run.
    body = spec.worldbody.add_body(name="platform_body", pos=[0, 0, PLATFORM_Z])
    body.add_geom(
        name="platform_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PLATFORM_HALF_XY, PLATFORM_HALF_XY, PLATFORM_HALF_H],
        rgba=[0.25, 0.45, 0.65, 1.0],
        contype=1,
        conaffinity=1,
        condim=3,
        friction=[0.8, 0.02, 0.001],
    )
    return spec


def get_piplus_on_platform_cfg() -> EntityCfg:
    """Robot config with spawn height adjusted for the elevated platform."""
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
# Helper: quaternion from roll+pitch (MuJoCo [w,x,y,z] convention)
# ---------------------------------------------------------------------------

def _roll_pitch_to_quat(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Returns [B, 4] quaternion [w,x,y,z] for roll-then-pitch rotation."""
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # q = q_pitch * q_roll
    w = cr * cp
    x = sr * cp
    y = cr * sp
    z = -sr * sp
    return torch.stack([w, x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# Custom obs / reward / event functions
# ---------------------------------------------------------------------------

def base_height_gauss(env, target_height: float, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    err = asset.data.root_link_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(err) / (std * std))


def foot_flatness(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_com_quat_w[:, asset_cfg.body_ids, :]
    _, x, y, _ = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    foot_up_z = 1 - 2 * (x * x + y * y)
    return foot_up_z.clamp(0, 1).mean(dim=-1)


def xy_drift_gauss(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for staying near platform center (same env-local-frame trick)."""
    asset = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_link_pos_w[:, :2]
    if not hasattr(env, "_spawn_xy"):
        env._spawn_xy = pos_xy.clone()
    dist_sq = torch.sum((pos_xy - env._spawn_xy) ** 2, dim=-1)
    return torch.exp(-dist_sq / (std * std))


def platform_state_obs(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged obs: [roll, pitch, roll_rate, pitch_rate, heave_vel] [B, 5]."""
    if not hasattr(env, "_plat_amp"):
        return torch.zeros(env.num_envs, 5, device=env.device)
    a = env._plat_amp    # [B, 3]  (roll, pitch, heave)
    ph = env._plat_phase  # [B, 3]
    fr = env._plat_freq   # [B, 3]
    roll       = a[:, 0] * torch.sin(ph[:, 0])
    pitch      = a[:, 1] * torch.sin(ph[:, 1])
    roll_rate  = a[:, 0] * torch.cos(ph[:, 0]) * fr[:, 0]
    pitch_rate = a[:, 1] * torch.cos(ph[:, 1]) * fr[:, 1]
    heave_vel  = a[:, 2] * torch.cos(ph[:, 2]) * fr[:, 2]
    return torch.stack([roll, pitch, roll_rate, pitch_rate, heave_vel], dim=-1)


def reset_platform_motion(
    env,
    env_ids: torch.Tensor,
    amp_max: tuple[float, float, float],
    freq_range: tuple[float, float],
    platform_cfg: SceneEntityCfg,
) -> None:
    """Reset-mode event: randomize sinusoidal motion params + set platform pose."""
    device = env.device
    n = len(env_ids)

    if not hasattr(env, "_plat_amp"):
        B = env.num_envs
        env._plat_amp   = torch.zeros(B, 3, device=device)
        env._plat_phase = torch.zeros(B, 3, device=device)
        env._plat_freq  = torch.zeros(B, 3, device=device)

    r_max, p_max, h_max = amp_max
    f_lo, f_hi = freq_range
    env._plat_amp[env_ids, 0] = torch.rand(n, device=device) * r_max
    env._plat_amp[env_ids, 1] = torch.rand(n, device=device) * p_max
    env._plat_amp[env_ids, 2] = torch.rand(n, device=device) * h_max
    env._plat_phase[env_ids] = torch.rand(n, 3, device=device) * 2 * math.pi
    env._plat_freq[env_ids] = (
        torch.rand(n, 3, device=device) * (f_hi - f_lo) + f_lo
    )

    # Place platform at rest pose for reset env_ids.
    platform = env.scene[platform_cfg.name]
    pose = torch.zeros(n, 7, device=device)
    pose[:, :2] = env.scene.env_origins[env_ids, :2]
    pose[:, 2] = PLATFORM_Z
    pose[:, 3] = 1.0  # identity quaternion (w=1)
    platform.write_mocap_pose_to_sim(pose, env_ids=env_ids)


class ApplyPlatformMotion:
    """Step-mode event: drive platform pose each control step + debug vis slider."""

    def __init__(self, cfg=None, env=None):
        self._slider_added = False

    def __call__(self, env, env_ids, platform_cfg: SceneEntityCfg) -> None:
        self._env = env
        self._platform_cfg = platform_cfg
        if not hasattr(env, "_plat_amp"):
            return

        platform = env.scene[platform_cfg.name]
        env._plat_phase += env._plat_freq * env.step_dt

        # Allow play-mode amplitude override via slider.
        amp = getattr(env, "_plat_amp_override", env._plat_amp)

        roll  = amp[:, 0] * torch.sin(env._plat_phase[:, 0])
        pitch = amp[:, 1] * torch.sin(env._plat_phase[:, 1])
        heave = amp[:, 2] * torch.sin(env._plat_phase[:, 2])

        all_ids = torch.arange(env.num_envs, device=env.device)
        pose = torch.zeros(env.num_envs, 7, device=env.device)
        pose[:, :2] = env.scene.env_origins[:, :2]
        pose[:, 2]  = PLATFORM_Z + heave
        pose[:, 3:7] = _roll_pitch_to_quat(roll, pitch)
        platform.write_mocap_pose_to_sim(pose, env_ids=all_ids)

    def reset(self, env_ids=None) -> None:
        pass

    def debug_vis(self, visualizer) -> None:
        if not hasattr(self, "_env"):
            return
        env = self._env
        if not hasattr(env, "_plat_amp"):
            return

        if not self._slider_added and hasattr(visualizer, "server"):
            self._slider_added = True
            with visualizer.server.gui.add_folder("Platform Motion"):
                s_roll = visualizer.server.gui.add_slider(
                    "Roll amp (rad)", min=0.0, max=0.35, step=0.01, initial_value=0.0,
                )
                s_pitch = visualizer.server.gui.add_slider(
                    "Pitch amp (rad)", min=0.0, max=0.25, step=0.01, initial_value=0.0,
                )
                s_heave = visualizer.server.gui.add_slider(
                    "Heave amp (m)", min=0.0, max=0.30, step=0.01, initial_value=0.0,
                )

                def _on_change(_ev, _r=s_roll, _p=s_pitch, _h=s_heave):
                    override = torch.zeros(env.num_envs, 3, device=env.device)
                    override[:, 0] = float(_r.value)
                    override[:, 1] = float(_p.value)
                    override[:, 2] = float(_h.value)
                    env._plat_amp_override = override

                s_roll.on_update(_on_change)
                s_pitch.on_update(_on_change)
                s_heave.on_update(_on_change)

        # Draw tilt normal arrow for first visualized env.
        amp = getattr(env, "_plat_amp_override", env._plat_amp)
        for env_idx in visualizer.get_env_indices(env.num_envs):
            roll  = float(amp[env_idx, 0] * torch.sin(env._plat_phase[env_idx, 0]))
            pitch = float(amp[env_idx, 1] * torch.sin(env._plat_phase[env_idx, 1]))
            heave = float(amp[env_idx, 2] * torch.sin(env._plat_phase[env_idx, 2]))
            if abs(roll) < 0.01 and abs(pitch) < 0.01:
                continue
            platform = env.scene[self._platform_cfg.name]
            center = platform.data.root_link_pos_w[env_idx].cpu().numpy().copy()
            center[2] += PLATFORM_HALF_H + heave
            # Surface normal direction from roll/pitch.
            nx = -math.sin(pitch)
            ny = math.sin(roll)
            nz = math.cos(roll) * math.cos(pitch)
            arrow_end = center + np.array([nx, ny, nz]) * 0.8
            visualizer.add_arrow(
                start=center, end=arrow_end, color=(0.2, 0.8, 0.4, 0.9), width=0.05
            )


def platform_motion_curriculum(
    env,
    env_ids: torch.Tensor,
    stages: list[dict],
) -> dict[str, torch.Tensor]:
    """Ramp up platform motion amplitude at step thresholds."""
    roll_max = pitch_max = heave_max = 0.0
    for stage in stages:
        if env.common_step_counter >= stage["step"]:
            roll_max  = stage["roll_max"]
            pitch_max = stage["pitch_max"]
            heave_max = stage["heave_max"]

    cfg = env.event_manager.get_term_cfg("reset_platform")
    cfg.params["amp_max"] = (roll_max, pitch_max, heave_max)
    return {"roll_max": torch.tensor(roll_max), "heave_max": torch.tensor(heave_max)}


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

def piplus_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = SceneEntityCfg("robot")
    platform_cfg = SceneEntityCfg("platform")

    obs_terms = {
        "joint_pos":         ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel":         ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "actions":           ObservationTermCfg(func=envs_mdp.last_action),
        "platform_state":    ObservationTermCfg(
            func=platform_state_obs,
            params={"asset_cfg": robot_cfg},
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
                "std": 0.15,  # wider: heave shifts the robot's absolute z
                "asset_cfg": robot_cfg,
            },
        ),
        "joint_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "posture": RewardTermCfg(
            func=envs_mdp.posture,
            weight=3.0,
            params={
                "std": {
                    ".*_hip_roll_joint":    0.05,
                    ".*_hip_pitch_joint":   0.25,
                    ".*_thigh_joint":       0.25,
                    ".*_calf_joint":        0.25,
                    ".*_ankle_pitch_joint": 0.05,
                    ".*_ankle_roll_joint":  0.25,
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
        "xy_drift": RewardTermCfg(
            func=xy_drift_gauss,
            weight=0.5,
            params={"std": 2.0, "asset_cfg": robot_cfg},
        ),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.015),
        "joint_vel":   RewardTermCfg(func=envs_mdp.joint_vel_l2,   weight=-0.005),
    }

    terminations = {
        "time_out":  TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "push": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 12.0),
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                "asset_cfg": robot_cfg,
            },
        ),
        "reset_platform": EventTermCfg(
            func=reset_platform_motion,
            mode="reset",
            params={
                "amp_max": (0.0, 0.0, 0.0),  # ramped by curriculum
                "freq_range": (0.3, 1.2),     # rad/s ≈ 0.05–0.19 Hz (ship swell)
                "platform_cfg": platform_cfg,
            },
        ),
        "platform_motion": EventTermCfg(
            func=ApplyPlatformMotion,
            mode="step",
            params={"platform_cfg": platform_cfg},
        ),
    }

    curriculum = {} if play else {
        "platform_motion": CurriculumTermCfg(
            func=platform_motion_curriculum,
            params={
                "stages": [
                    {"step":     0, "roll_max": 0.00, "pitch_max": 0.00, "heave_max": 0.00},
                    {"step":  6000, "roll_max": 0.10, "pitch_max": 0.08, "heave_max": 0.10},
                    {"step": 12000, "roll_max": 0.20, "pitch_max": 0.15, "heave_max": 0.20},
                    {"step": 20000, "roll_max": 0.30, "pitch_max": 0.20, "heave_max": 0.25},
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot":    get_piplus_on_platform_cfg(),
                "platform": EntityCfg(
                    spec_fn=get_platform_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=(0, 0, PLATFORM_Z)),
                    collisions=(CollisionCfg(
                        geom_names_expr=("platform_geom",),
                        contype=1,
                        conaffinity=1,
                        condim=3,
                        priority=1,
                        friction=(0.8,),
                    ),),
                ),
            },
            num_envs=1,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=3.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005), njmax=400, nconmax=160),
        decimation=4,
        episode_length_s=20.0 if not play else 1e9,
    )


def piplus_platform_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
        experiment_name="piplus_platform",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=1000,
    )
