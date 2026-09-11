# Mjlab-Piplus-Upright

Task ID: `Mjlab-Piplus-Upright`

The robot learns to stand upright on flat ground with flat feet, resisting random velocity disturbances.

## Observations

| Term | Content |
|------|---------|
| `joint_pos` | Joint positions relative to default pose |
| `joint_vel` | Joint velocities relative to default |
| `projected_gravity` | Gravity vector projected into base frame — encodes tilt |
| `actions` | Last action (policy output) |

Actor observations have noise corruption enabled during training; disabled for play and critic.

## Actions

Single `JointPositionActionCfg` targeting all leg actuators (`ACTUATOR_5036`).  
Scale `0.5` rad, added as offset to the home pose (`use_default_offset=True`).

## Reward Formulation

All rewards are summed each control step (50 Hz) and discounted with γ=0.99.

### Upright (weight +1.0)

```
r = exp(−‖g_proj‖² / std²),  std = √0.05 ≈ 0.224
```

`g_proj` = gravity vector projected onto the `torso_link` body frame. Zero when perfectly upright, falls off as the torso tilts. Gaussian width chosen to stay near 1.0 for small tilts and sharply penalize large ones.

### Base Height (weight −2.0)

```
r = −(z_root − 0.35)²
```

L2 penalty on CoM height deviation from 0.35 m. Prevents the robot from cheating by crouching or rising onto toes (toe-standing raises CoM above target).

### Joint Position Limits (weight −1.0)

Built-in penalty when joints exceed their soft limits (`soft_joint_pos_limit_factor = 0.9`). Prevents hyperextension and keeps joints in safe operating range.

### Posture (weight +2.0)

```
r = exp(−mean_j(err_j² / std_j²))
```

Encourages joint positions to stay close to the home keyframe. Per-joint std controls tightness:

| Joint group | std | Rationale |
|-------------|-----|-----------|
| `.*_hip_roll_joint` | 0.05 | Tight — prevents split pose |
| `.*_ankle_pitch_joint` | 0.05 | Tight — prevents toe-standing |
| `.*_hip_pitch_joint` | 0.25 | Loose — needed for balance |
| `.*_thigh_joint` | 0.25 | Loose |
| `.*_calf_joint` | 0.25 | Loose |
| `.*_ankle_roll_joint` | 0.25 | Loose |

Tight std collapses the reward to near zero even at 0.2 rad deviation; loose std allows ±0.25 rad with ~37% reward remaining.

### Foot Flatness (weight +1.0)

```
r = mean([R_right[2,2], R_left[2,2]]).clamp(0, 1)
```

`R[2,2] = 1 − 2(x² + y²)` is the z-component of the foot body's local z-axis in world frame.  
Equals 1.0 when the sole is flat (local z points up), decreases as the foot tilts.  
Replaces a binary "foot in contact" reward which also fired during toe-standing.

### Action Rate (weight −0.005)

```
r = −‖a_t − a_{t−1}‖²
```

L2 penalty on action changes. Smooths joint commands and discourages jitter.

### Joint Velocity (weight −0.01)

```
r = −‖q̇‖²
```

L2 penalty on all joint velocities. Suppresses toe-tapping and fidgeting.

## Termination

| Condition | Trigger |
|-----------|---------|
| `time_out` | Episode length > 20 s (infinite in play mode) |
| `fell_over` | Torso tilt > 70° from vertical |

## Disturbances (Push Events)

Random velocity pushes applied to the root body in interval mode:

- **Interval:** 1–10 s (uniform random per env per episode)  
- **Magnitude:** x ∈ [−2.0, 2.0] m/s, y ∈ [−2.0, 2.0] m/s independently  
- **Implementation:** `push_by_setting_velocity` — directly sets root linear velocity, not a force/impulse

2.0 m/s ≈ fast-walking speed applied sideways. The wide interval range ensures some environments see frequent pushes and others see rare but harder ones in the same training batch.

## Sim Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 0.005 s |
| Decimation | 4 (control at 50 Hz) |
| Episode length | 20 s |
| Environments | 4096 (GPU) / 64 (CPU) |
| njmax | 200 |

## Robot Config

Home keyframe (Pi+ leg joints, symmetric signs for mirrored axes):

| Joint | Value (rad) |
|-------|------------|
| r/l_hip_pitch | ±0.5 |
| r/l_hip_roll | 0.0 |
| r/l_thigh | 0.0 |
| r/l_calf | ±1.0 |
| r/l_ankle_pitch | ±0.5 |
| r/l_ankle_roll | 0.0 |

Arms are defined in the MJCF but excluded from actuators and home keyframe.

## Curriculum (Planned)

Training is robust enough to reach stable standing quickly, but strong pushes create a sparse recovery problem — the robot may learn to fall gracefully rather than recover.

Planned two-phase approach:
1. **Phase 1**: Train to convergence (~400 iters) with mild pushes (±0.5 m/s, 5–10 s interval)
2. **Phase 2**: Load Phase 1 checkpoint, retrain with current harsh pushes (±2.0 m/s, 1–10 s interval)
