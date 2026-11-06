# Mjlab-Piplus-Upright

Task ID: `Mjlab-Piplus-Upright`

The robot learns to stand upright on flat ground with flat feet. It resists random velocity disturbances.

## Observations

| Term | Content |
|------|---------|
| `joint_pos` | Joint positions relative to default pose |
| `joint_vel` | Joint velocities relative to default |
| `projected_gravity` | Gravity vector projected into base frame — encodes tilt |
| `actions` | Last action (policy output) |

Training enables noise corruption for the actor observations. Play and the critic have noise corruption disabled.

## Actions

A single `JointPositionActionCfg` targets all leg actuators (`ACTUATOR_5036`).  
The scale is `0.5` rad. The action is added as an offset to the home pose (`use_default_offset=True`).

## Reward Formulation

The environment sums all rewards at each control step (50 Hz). The sum is discounted with γ=0.99.

### Upright (weight +1.0)

```
r = exp(−‖g_proj‖² / std²),  std = √0.05 ≈ 0.224
```

`g_proj` is the gravity vector projected onto the `torso_link` body frame. It is zero when the torso is perfectly upright. The reward falls off as the torso tilts. The Gaussian width keeps the reward near 1.0 for small tilts. It penalizes large tilts sharply. Here std is the standard deviation (the width) of the Gaussian.

### Base Height (weight −2.0)

```
r = −(z_root − 0.35)²
```

This is an L2 penalty on the deviation of the center of mass (CoM) height from 0.35 m. It stops the robot from cheating by crouching or by rising onto the toes. Toe-standing raises the CoM above the target.

### Joint Position Limits (weight −1.0)

This is a built-in penalty. It applies when a joint exceeds its soft limit (`soft_joint_pos_limit_factor = 0.9`). It prevents hyperextension. It keeps the joints in the safe operating range.

### Posture (weight +2.0)

```
r = exp(−mean_j(err_j² / std_j²))
```

The reward encourages the joint positions to stay close to the home keyframe. The std of each joint controls the tightness:

| Joint group | std | Rationale |
|-------------|-----|-----------|
| `.*_hip_roll_joint` | 0.05 | Tight — prevents split pose |
| `.*_ankle_pitch_joint` | 0.05 | Tight — prevents toe-standing |
| `.*_hip_pitch_joint` | 0.25 | Loose — needed for balance |
| `.*_thigh_joint` | 0.25 | Loose |
| `.*_calf_joint` | 0.25 | Loose |
| `.*_ankle_roll_joint` | 0.25 | Loose |

A tight std collapses the reward to near zero even at a deviation of 0.2 rad. A loose std allows ±0.25 rad with about 37% of the reward remaining.

### Foot Flatness (weight +1.0)

```
r = mean([R_right[2,2], R_left[2,2]]).clamp(0, 1)
```

`R[2,2] = 1 − 2(x² + y²)` is the z-component of the foot body's local z-axis in world frame.  
It equals 1.0 when the sole is flat (local z points up). It decreases as the foot tilts.  
It replaces a binary "foot in contact" reward. That reward also fired during toe-standing.

### Action Rate (weight −0.005)

```
r = −‖a_t − a_{t−1}‖²
```

This is an L2 penalty on action changes. It smooths the joint commands. It discourages jitter.

### Joint Velocity (weight −0.01)

```
r = −‖q̇‖²
```

This is an L2 penalty on all joint velocities. It suppresses toe-tapping and fidgeting.

## Termination

| Condition | Trigger |
|-----------|---------|
| `time_out` | Episode length > 20 s (infinite in play mode) |
| `fell_over` | Torso tilt > 70° from vertical |

## Disturbances (Push Events)

Random velocity pushes act on the root body in interval mode:

- **Interval:** 1–10 s (uniform random per environment per episode)  
- **Magnitude:** x ∈ [−2.0, 2.0] m/s, y ∈ [−2.0, 2.0] m/s independently  
- **Implementation:** `push_by_setting_velocity` sets the root linear velocity directly. It does not apply a force or an impulse.

2.0 m/s is about the speed of fast walking, applied sideways. The wide interval range ensures that some environments see frequent pushes. Other environments see rare but harder pushes. Both kinds are in the same training batch. <!-- STE-deviation: no number in source -->

## Sim Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 0.005 s |
| Decimation | 4 (control at 50 Hz) |
| Episode length | 20 s |
| Environments | 4096 (graphics processing unit, GPU) / 64 (central processing unit, CPU) |
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

The MuJoCo XML format (MJCF) file defines the arms. The actuators and the home keyframe do not include the arms.

## Curriculum (Planned)

Training reaches stable standing quickly. Strong pushes create a sparse recovery problem. The robot may learn to fall gracefully instead of recovering.

The planned approach has two phases:
1. **Phase 1**: Train to convergence (about 400 iterations) with mild pushes (±0.5 m/s, 5–10 s interval)
2. **Phase 2**: Load the Phase 1 checkpoint. Retrain with the current harsh pushes (±2.0 m/s, 1–10 s interval)
