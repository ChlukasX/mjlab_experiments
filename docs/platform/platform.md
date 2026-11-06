# Mjlab-Piplus-Platform

Task ID: `Mjlab-Piplus-Platform`

The robot learns to balance on a physical 3 m × 3 m platform. The platform moves with sinusoidal roll, pitch, and heave. This simulates the motion of a ship deck or a boat deck. Unlike the NonInertial task, the platform is a real collision body. The robot must keep foot contact with a tilting surface. It does not only resist injected forces.

## Version History

| Version | Key changes |
|---------|------------|
| v1 | Initial setup: platform body, spawn height, njmax/nconmax sizing, basic rewards |
| v2 | Reward tuning: loosened posture standard deviations (std) (hip/knee), added `foot_grounding`, raised `foot_flatness` weight |
| v3 | Tightened `foot_grounding` (std 0.05→0.02, weight 2→5), increased `action_rate` penalty, `PushEventWithVis` with play-mode slider |
| **v4** | `off_platform` termination, random yaw at spawn, platform resets to flat (phase=0) on episode reset |

## Platform Geometry

```
PLATFORM_Z      = 0.20 m   (center height)
PLATFORM_HALF_H = 0.05 m   (half-thickness; top at 0.25 m)
PLATFORM_HALF_XY= 1.50 m   (3 m × 3 m surface)
ROBOT_SPAWN_Z   = 0.75 m   (0.5 m above platform top — avoids spawn clipping)
BASE_HEIGHT_TARGET = 0.60 m
```

`get_platform_spec()` (in `tasks/platform/piplus_platform_env_cfg.py`) builds the platform as a body without joints and with a BOX geom. mjlab wraps the body as a motion capture (mocap) entity.

Platform motion: 3 degrees of freedom (DOF), sinusoids, driven through the mocap pose at each control step:

```
roll  = A_r · sin(ω_r · t)   [rad]
pitch = A_p · sin(ω_p · t)   [rad]
heave = A_h · sin(ω_h · t)   [m]
```

The phase resets to 0 at the start of each episode. The platform is then always flat when the robot spawns. The task randomizes amplitude and frequency per episode per environment. The curriculum ramps the amplitudes.

## Observations

| Term | Shape | Content |
|------|-------|---------|
| `joint_pos` | [B, n_j] | Joint positions relative to home pose |
| `joint_vel` | [B, n_j] | Joint velocities relative to default |
| `projected_gravity` | [B, 3] | Gravity in base frame — encodes tilt |
| `actions` | [B, n_j] | Last policy output |
| `platform_state` | [B, 5] | `[roll, pitch, roll_rate, pitch_rate, heave_vel]` — privileged signal |

Training applies noise corruption to the actor observations. Play and the critic have noise corruption disabled.

## Actions

A single `JointPositionActionCfg` targets all leg actuators (`ACTUATOR_5036`).  
The scale is `0.5` rad. The action is added as an offset to the home pose (`use_default_offset=True`).

## Reward Formulation

The environment sums all rewards at 50 Hz. The sum is discounted with γ=0.99.

### Upright (weight +1.5)

```
r = exp(−‖g_proj‖² / 0.05)
```

The reward uses the gravity vector projected onto the `torso_link` frame. It penalizes torso tilt.

### Base Height (weight +1.5)

```
r = exp(−(z_root − 0.60)² / 0.0225)    std = 0.15
```

The std (0.15 m) is wide compared with the upright task. This accommodates the heave, which shifts the absolute robot height.

### Joint Position Limits (weight −1.0)

This is a built-in penalty. It applies when a joint exceeds its soft limit (`soft_joint_pos_limit_factor = 0.9`).

### Posture (weight +3.0)

```
r = exp(−mean_j(err_j² / std_j²))
```

The per-joint std values let the hips and knees absorb the platform motion. They keep the robot upright:

| Joint group | std | Rationale |
|-------------|-----|-----------|
| `.*_hip_roll_joint` | 0.05 | Tight — prevent lateral sway |
| `.*_ankle_roll_joint` | 0.25 | Moderate |
| `.*_ankle_pitch_joint` | 0.20 | Some ankle compensation allowed |
| `.*_hip_pitch_joint` | 0.40 | Loose — hip flex absorbs pitch |
| `.*_thigh_joint` | 0.40 | Loose — knee flex absorbs motion |
| `.*_calf_joint` | 0.40 | Loose — knee flex absorbs motion |

<!-- STE-deviation: table cell "Some ankle compensation allowed" keeps its fragment; no number in source -->

### Foot Flatness (weight +2.0)

```
r = mean([R_right[2,2], R_left[2,2]]).clamp(0, 1)
```

This term rewards feet that stay parallel to the platform surface. `R[2,2] = 1 − 2(x² + y²)` is the z-component of the local z-axis of the ankle body in the world frame.

### Foot Grounding (weight +5.0)

```
foot_bottom = ankle_z − 0.05          (capsule geom 5 cm below ankle body)
lift = max(0, foot_bottom − platform_top)
r = exp(−(lift_left + lift_right) / 0.02)
```

This term strongly penalizes lifting the feet off the platform. `std=0.02` means that a lift of 2 cm per foot leaves about 37% of the reward. A lift of 5 cm leaves about 8%. This is the dominant reward term. It discourages tapping.

### XY Drift (weight +0.5)

```
r = exp(−‖pos_xy − spawn_xy‖² / 4.0)    std = 2.0
```

This term encourages the robot to stay near the platform center.

### Action Rate (weight −0.05)

```
r = −‖a_t − a_{t−1}‖²
```

This is an L2 penalty on action changes. It discourages oscillatory joint commands. These commands produce tapping.

### Joint Velocity (weight −0.005)

```
r = −‖q̇‖²
```

This term suppresses unnecessary joint movement.

## Terminations

| Condition | Trigger |
|-----------|---------|
| `time_out` | Episode > 20 s (infinite in play mode) |
| `fell_over` | Torso tilt > 70° from vertical |
| `off_platform` | Robot xy > 2.0 m from platform center (1.5 m boundary + 0.5 m margin) |

`off_platform` stops the robot from accumulating rewards by standing on the floor after it falls off the platform edge.

## Events

| Event | Mode | Description |
|-------|------|-------------|
| `reset_scene_to_default` | reset | Writes `init_state.pos` (spawn height) to qpos — must be first |
| `randomize_yaw` | reset | Uniform yaw ∈ [−π, π] at spawn via `reset_root_state_uniform` |
| `push` (`PushEventWithVis`) | interval 4–12 s | ±0.5 m/s x/y velocity kick; disabled by default in play mode (slider at 0) |
| `reset_platform` | reset | Randomizes amplitude/frequency; sets phase = 0 (platform flat at spawn) |
| `platform_motion` (`ApplyPlatformMotion`) | step | Drives mocap platform pose each control step; exposes roll/pitch/heave sliders in play mode |

Random yaw ensures that the policy sees all orientations relative to the tilt axes of the platform. This prevents the policy from learning asymmetric hip_pitch versus hip_roll specializations.

## Curriculum

The amplitudes ramp automatically with `common_step_counter`. The counter increments by 1 per `env.step()` call. This gives 24 per Proximal Policy Optimization (PPO) iteration at `num_steps_per_env=24`:

| Step threshold | PPO iter (~) | Roll max | Pitch max | Heave max |
|----------------|-------------|----------|-----------|-----------|
| 0 | 0 | 0.00 rad | 0.00 rad | 0.00 m |
| 6 000 | 250 | 0.10 rad | 0.08 rad | 0.10 m |
| 12 000 | 500 | 0.20 rad | 0.15 rad | 0.20 m |
| 20 000 | 833 | 0.30 rad | 0.20 rad | 0.25 m |

The frequency range is 0.3–1.2 rad/s (about 0.05–0.19 Hz, the ship swell band).

## Sim Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 0.005 s |
| Decimation | 4 (control at 50 Hz) |
| Episode length | 20 s |
| Environments | 4096 (graphics processing unit, GPU) / 64 (central processing unit, CPU) |
| njmax | 400 |
| nconmax | 160 |

`njmax=400, nconmax=160` fit the BOX-BOX foot/platform contact at 4096 environments (observed peaks: 372/138 + buffer). PLANE geoms on non-terrain mocap bodies hang in mjwarp. The platform therefore uses BOX only.

## Play / Video

```bash
# Interactive viser (with platform and push sliders)
uv run play Mjlab-Piplus-Platform \
  --checkpoint-file logs/rsl_rl/piplus_platform/<run>/model_N.pt \
  --viewer viser --num-envs 9

# Record video (offscreen)
uv run play Mjlab-Piplus-Platform \
  --checkpoint-file logs/rsl_rl/piplus_platform/<run>/model_N.pt \
  --viewer viser --num-envs 9 --video True --video-length 500

# Native MuJoCo viewer (no GUI panels)
uv run play Mjlab-Piplus-Platform \
  --checkpoint-file logs/rsl_rl/piplus_platform/<run>/model_N.pt \
  --viewer native --num-envs 1
```

`--env.viewer.max-extra-envs N` sets the number of neighboring environments that render around the primary environment.
