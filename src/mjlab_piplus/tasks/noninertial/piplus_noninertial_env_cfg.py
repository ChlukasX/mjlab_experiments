"""Piplus non-inertial environment: stand upright on a simulated moving platform.

Platform motion is modelled by injecting sinusoidal forces on the root body each
control step (fictitious-force approximation).  The policy receives the applied
force / robot_mass as a privileged observation so it can feed-forward compensate.

Automatic curriculum (via mjlab CurriculumManager, steps = common_step_counter):

  step       0 :  amplitude = 0 N   (learn to stand, ~250 iters)
  step    6000 :  amplitude ≤ 10 N  (~250 iters)
  step   12000 :  amplitude ≤ 20 N  (~333 iters)
  step   20000 :  amplitude ≤ 40 N  (~166 iters)

With max_iterations=1000 and num_steps_per_env=24, common_step_counter reaches
24 000 at the end of training.
"""

import math

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.viewer import ViewerConfig

from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg

# Pi+ total body mass from MJCF (sum of all body_mass values)
ROBOT_MASS = 13.02


# ---------------------------------------------------------------------------
# Custom reward / observation / event functions
# ---------------------------------------------------------------------------

def base_height_gauss(env, target_height: float, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  err = asset.data.root_link_pos_w[:, 2] - target_height
  return torch.exp(-torch.square(err) / (std * std))


def foot_flatness(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  quat = asset.data.body_com_quat_w[:, asset_cfg.body_ids, :]  # [B, 2, 4]
  _, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  foot_up_z = 1 - 2 * (x * x + y * y)  # [B, 2]
  return foot_up_z.clamp(0, 1).mean(dim=-1)


def xy_drift_gauss(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Reward for staying near spawn origin. Bounded [0,1] — safe when fallen."""
  asset = env.scene[asset_cfg.name]
  pos_xy = asset.data.root_link_pos_w[:, :2]
  if not hasattr(env, "_spawn_xy"):
    env._spawn_xy = pos_xy.clone()
  dist_sq = torch.sum((pos_xy - env._spawn_xy) ** 2, dim=-1)
  return torch.exp(-dist_sq / (std * std))


def platform_accel_obs(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Privileged obs: applied force / robot_mass → platform acceleration [B, 3]."""
  asset = env.scene[asset_cfg.name]
  if not hasattr(env, "_plat_phase"):
    return torch.zeros(env.num_envs, 3, device=env.device)
  force = asset.data.body_external_wrench[:, 0, :3]
  return force / ROBOT_MASS


def reset_platform_state(
  env,
  env_ids: torch.Tensor,
  amplitude_range: tuple[float, float],
  freq_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset-mode event: randomize per-env sinusoidal platform parameters."""
  device = env.device
  n = len(env_ids)

  if not hasattr(env, "_plat_phase"):
    B = env.num_envs
    env._plat_phase = torch.zeros(B, 2, device=device)
    env._plat_freq = torch.zeros(B, 2, device=device)
    env._plat_amp = torch.zeros(B, device=device)

  a_lo, a_hi = amplitude_range
  f_lo, f_hi = freq_range
  env._plat_phase[env_ids] = torch.rand(n, 2, device=device) * 2 * math.pi
  env._plat_freq[env_ids] = torch.rand(n, 2, device=device) * (f_hi - f_lo) + f_lo
  env._plat_amp[env_ids] = torch.rand(n, device=device) * (a_hi - a_lo) + a_lo


class ApplyPlatformForces:
  """Step-mode event: sinusoidal platform forces with debug vis + play slider."""

  def __init__(self, cfg=None, env=None):
    self._slider_added = False

  def __call__(self, env, env_ids, asset_cfg: SceneEntityCfg) -> None:
    self._env = env
    self._asset_cfg = asset_cfg
    if not hasattr(env, "_plat_phase"):
      return

    asset = env.scene[asset_cfg.name]
    env._plat_phase += env._plat_freq * env.step_dt

    # In play mode the slider sets _force_amp_override; else use per-env amp.
    amp = getattr(env, "_force_amp_override", env._plat_amp)

    forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    forces[:, 0, 0] = amp * torch.sin(env._plat_phase[:, 0])
    forces[:, 0, 1] = amp * torch.sin(env._plat_phase[:, 1])
    asset.write_external_wrench_to_sim(forces, torch.zeros_like(forces), body_ids=[0])

  def reset(self, env_ids=None) -> None:
    pass

  def debug_vis(self, visualizer) -> None:
    if not hasattr(self, "_env"):
      return
    env = self._env
    asset = env.scene[self._asset_cfg.name]

    # Add amplitude slider once when running under viser.
    if not self._slider_added and hasattr(visualizer, "server"):
      self._slider_added = True
      with visualizer.server.gui.add_folder("Platform Forces"):
        slider = visualizer.server.gui.add_slider(
          "Amplitude (N)", min=0.0, max=40.0, step=1.0, initial_value=0.0,
        )
        def _on_change(_ev, _s=slider):
          env._force_amp_override = torch.full(
            (env.num_envs,), float(_s.value), device=env.device
          )
        slider.on_update(_on_change)

    # Draw force arrow for each visualized env.
    if not hasattr(env, "_plat_phase"):
      return
    wrench = asset.data.body_external_wrench   # [B, nbody, 6]
    root_pos = asset.data.root_link_pos_w      # [B, 3]
    for env_idx in visualizer.get_env_indices(env.num_envs):
      force = wrench[env_idx, 0, :3].cpu().numpy()
      mag = float(np.linalg.norm(force))
      if mag < 0.5:
        continue
      start = root_pos[env_idx].cpu().numpy() + np.array([0.0, 0.0, 0.5])
      end = start + force * 0.05  # 40 N → 2 m arrow
      visualizer.add_arrow(start=start, end=end, color=(1.0, 0.3, 0.1, 0.9), width=0.04)


def platform_amplitude_curriculum(
  env,
  env_ids: torch.Tensor,
  stages: list[dict],
) -> dict[str, torch.Tensor]:
  """Curriculum: ramp up platform force amplitude at step thresholds."""
  amp_max = 0.0
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      amp_max = stage["amplitude_max"]

  event_cfg = env.event_manager.get_term_cfg("reset_platform")
  a_lo = event_cfg.params["amplitude_range"][0]
  event_cfg.params["amplitude_range"] = (a_lo, amp_max)

  return {"amplitude_max": torch.tensor(float(amp_max))}


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

def piplus_noninertial_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  robot_cfg = SceneEntityCfg("robot")

  obs_terms = {
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "platform_accel": ObservationTermCfg(
      func=platform_accel_obs,
      params={"asset_cfg": robot_cfg},
    ),
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
      weight=1.5,
      params={
        "std": math.sqrt(0.05),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "base_height": RewardTermCfg(
      func=base_height_gauss,
      weight=1.5,
      params={"target_height": 0.35, "std": 0.08, "asset_cfg": robot_cfg},
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
    "xy_drift": RewardTermCfg(
      func=xy_drift_gauss,
      weight=0.5,
      params={"std": 2.0, "asset_cfg": robot_cfg},
    ),
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.015,
    ),
    "joint_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.005,
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
      interval_range_s=(2.0, 8.0),
      params={
        "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        "asset_cfg": robot_cfg,
      },
    ),
    "reset_platform": EventTermCfg(
      func=reset_platform_state,
      mode="reset",
      params={
        "amplitude_range": (0.0, 0.0),  # starts at zero; ramped by curriculum
        "freq_range": (0.1, 2.0),
        "asset_cfg": robot_cfg,
      },
    ),
    "platform_forces": EventTermCfg(
      func=ApplyPlatformForces,
      mode="step",
      params={"asset_cfg": robot_cfg},
    ),
  }

  curriculum = {} if play else {
    "platform_amplitude": CurriculumTermCfg(
      func=platform_amplitude_curriculum,
      params={
        "stages": [
          {"step":     0, "amplitude_max":  0.0},
          {"step":  6000, "amplitude_max": 10.0},
          {"step": 12000, "amplitude_max": 20.0},
          {"step": 20000, "amplitude_max": 40.0},
        ],
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
    curriculum=curriculum,
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


def piplus_noninertial_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="piplus_noninertial",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
