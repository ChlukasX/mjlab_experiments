# Mjlab-Piplus-Platform

Task ID: `Mjlab-Piplus-Platform`

The robot learns to balance on a physical 3 m × 3 m platform undergoing sinusoidal roll, pitch, and heave — simulating ship/boat deck motion. Unlike the NonInertial task, the platform is a real collision body; the robot must maintain foot contact with a tilting surface rather than simply resist injected forces.

## Version History

| Version | Key changes |
|---------|------------|
| v1 | Initial setup: platform body, spawn height, njmax/nconmax sizing, basic rewards |
| v2 | Reward tuning: loosened posture stds (hip/knee), added `foot_grounding`, raised `foot_flatness` weight |
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

Platform motion (3 DOF sinusoids driven via mocap pose each control step):

```
roll  = A_r · sin(ω_r · t)   [rad]
pitch = A_p · sin(ω_p · t)   [rad]
heave = A_h · sin(ω_h · t)   [m]
```

Phase is reset to 0 at episode start so the platform is always flat when the robot spawns. Amplitude and frequency are randomized per episode per environment; amplitudes are ramped by curriculum.

## Observations

| Term | Shape | Content |
|------|-------|---------|
| `joint_pos` | [B, n_j] | Joint positions relative to home pose |
| `joint_vel` | [B, n_j] | Joint velocities relative to default |
| `projected_gravity` | [B, 3] | Gravity in base frame — encodes tilt |
| `actions` | [B, n_j] | Last policy output |
| `platform_state` | [B, 5] | `[roll, pitch, roll_rate, pitch_rate, heave_vel]` — privileged signal |

Actor observations have noise corruption during training; disabled for play and critic.

## Actions

Single `JointPositionActionCfg` targeting all leg actuators (`ACTUATOR_5036`).  
Scale `0.5` rad added as offset to home pose (`use_default_offset=True`).

## Reward Formulation

All rewards summed at 50 Hz, discounted with γ=0.99.

### Upright (weight +1.5)

```
r = exp(−‖g_proj‖² / 0.05)
```

Gravity projected onto `torso_link` frame. Penalizes torso tilt.

### Base Height (weight +1.5)

```
r = exp(−(z_root − 0.60)² / 0.0225)    std = 0.15
```

Wide std (0.15 m) relative to the upright task to accommodate heave shifting absolute robot height.

### Joint Position Limits (weight −1.0)

Built-in penalty when joints exceed soft limits (`soft_joint_pos_limit_factor = 0.9`).

### Posture (weight +3.0)

```
r = exp(−mean_j(err_j² / std_j²))
```

Per-joint stds tuned to let hips and knees absorb platform motion while keeping the robot upright:

| Joint group | std | Rationale |
|-------------|-----|-----------|
| `.*_hip_roll_joint` | 0.05 | Tight — prevent lateral sway |
| `.*_ankle_roll_joint` | 0.25 | Moderate |
| `.*_ankle_pitch_joint` | 0.20 | Some ankle compensation allowed |
| `.*_hip_pitch_joint` | 0.40 | Loose — hip flex absorbs pitch |
| `.*_thigh_joint` | 0.40 | Loose — knee flex absorbs motion |
| `.*_calf_joint` | 0.40 | Loose — knee flex absorbs motion |

### Foot Flatness (weight +2.0)

```
r = mean([R_right[2,2], R_left[2,2]]).clamp(0, 1)
```

Rewards feet staying parallel to the platform surface. `R[2,2] = 1 − 2(x² + y²)` is the z-component of the ankle body's local z-axis in world frame.

### Foot Grounding (weight +5.0)

```
foot_bottom = ankle_z − 0.05          (capsule geom 5 cm below ankle body)
lift = max(0, foot_bottom − platform_top)
r = exp(−(lift_left + lift_right) / 0.02)
```

Strongly penalises lifting feet off the platform. `std=0.02` means a 2 cm lift per foot yields ~37% reward remaining; 5 cm → 8%. Dominant reward term to discourage tapping.

### XY Drift (weight +0.5)

```
r = exp(−‖pos_xy − spawn_xy‖² / 4.0)    std = 2.0
```

Encourages staying near the platform center.

### Action Rate (weight −0.05)

```
r = −‖a_t − a_{t−1}‖²
```

L2 penalty on action changes. Discourages oscillatory joint commands that produce tapping.

### Joint Velocity (weight −0.005)

```
r = −‖q̇‖²
```

Suppresses unnecessary joint movement.

## Terminations

| Condition | Trigger |
|-----------|---------|
| `time_out` | Episode > 20 s (infinite in play mode) |
| `fell_over` | Torso tilt > 70° from vertical |
| `off_platform` | Robot xy > 2.0 m from platform center (1.5 m boundary + 0.5 m margin) |

`off_platform` prevents the robot accumulating rewards by standing on the floor after falling off the platform edge.

## Events

| Event | Mode | Description |
|-------|------|-------------|
| `reset_scene_to_default` | reset | Writes `init_state.pos` (spawn height) to qpos — must be first |
| `randomize_yaw` | reset | Uniform yaw ∈ [−π, π] at spawn via `reset_root_state_uniform` |
| `push` (`PushEventWithVis`) | interval 4–12 s | ±0.5 m/s x/y velocity kick; disabled by default in play mode (slider at 0) |
| `reset_platform` | reset | Randomizes amplitude/frequency; sets phase = 0 (platform flat at spawn) |
| `platform_motion` (`ApplyPlatformMotion`) | step | Drives mocap platform pose each control step; exposes roll/pitch/heave sliders in play mode |

Random yaw ensures the policy sees all orientations relative to the platform's tilt axes, preventing it from learning asymmetric hip_pitch vs hip_roll specialisations.

## Curriculum

Amplitudes ramp automatically via `common_step_counter` (increments 1 per `env.step()` call = 24 per PPO iteration at `num_steps_per_env=24`):

| Step threshold | PPO iter (~) | Roll max | Pitch max | Heave max |
|----------------|-------------|----------|-----------|-----------|
| 0 | 0 | 0.00 rad | 0.00 rad | 0.00 m |
| 6 000 | 250 | 0.10 rad | 0.08 rad | 0.10 m |
| 12 000 | 500 | 0.20 rad | 0.15 rad | 0.20 m |
| 20 000 | 833 | 0.30 rad | 0.20 rad | 0.25 m |

Frequency range: 0.3–1.2 rad/s (≈ 0.05–0.19 Hz, ship swell band).

## Sim Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 0.005 s |
| Decimation | 4 (control at 50 Hz) |
| Episode length | 20 s |
| Environments | 4096 (GPU) / 64 (CPU) |
| njmax | 400 |
| nconmax | 160 |

`njmax=400, nconmax=160` sized for BOX-BOX foot/platform contact at 4096 envs (observed peaks: 372/138 + buffer). PLANE geoms on non-terrain mocap bodies hang in mjwarp — platform uses BOX only.

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

`--env.viewer.max-extra-envs N` controls how many neighbouring envs render around the primary one.
