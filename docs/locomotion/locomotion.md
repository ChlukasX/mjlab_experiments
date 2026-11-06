# Mjlab-Piplus-Locomotion / Mjlab-Piplus-Locomotion-Arms

Task IDs: `Mjlab-Piplus-Locomotion`, `Mjlab-Piplus-Locomotion-Arms`

The robot tracks a commanded base velocity `(vx, vy, ω_z)` on flat terrain. The task is the prerequisite for higher-level tasks (walking to a ball, walking on a platform).

`Locomotion-Arms` is the same task with actuated arm joints (20 degrees of freedom (DOF) instead of 12). The arm posture is loose. This allows a natural counterbalancing swing. `Locomotion-Arms` is a separate task. This allows a direct comparison of the leg-only results and the arm results.

Config: `tasks/locomotion/piplus_locomotion_env_cfg.py`, `tasks/locomotion_arms/piplus_locomotion_arms_env_cfg.py`.

## Velocity Commands

`UniformVelocityCommandCfg` resamples the command every 5–10 s:

| Component | Range |
|-----------|-------|
| `lin_vel_x` | [−1.0, 1.0] m/s |
| `lin_vel_y` | [−0.5, 0.5] m/s |
| `ang_vel_z` | [−1.0, 1.0] rad/s |

10% of the environments receive a zero command. The policy then also learns to stand still. Heading commands are off.

## Observations

| Term | Content |
|------|---------|
| `joint_pos` | Joint positions relative to home pose |
| `joint_vel` | Joint velocities relative to default |
| `projected_gravity` | Gravity in base frame |
| `actions` | Last policy output |
| `base_lin_vel` | Base linear velocity (base frame) |
| `base_ang_vel` | Base angular velocity (base frame) |
| `velocity_commands` | Current `(vx, vy, ω_z)` command |

The actor and the critic see the same terms. `base_lin_vel` is ground truth from the simulator. The real Pi Plus has no sensor for it. It needs a state estimator. The configuration sets no noise on any term, so `enable_corruption` currently has no effect.

## Actions

One `JointPositionActionCfg` acts on all actuators. The scale is 0.5 rad. The action is an offset from the home pose.

| Task | Actuators | DOF |
|------|-----------|-----|
| Locomotion | legs (`ACTUATOR_5036`) | 12 |
| Locomotion-Arms | legs + arms (`ACTUATOR_4438`) | 20 |

## Reward Formulation

In this section, std means standard deviation.

| Term | Weight | Purpose |
|------|--------|---------|
| `track_lin_vel` | +2.0 | Track commanded xy velocity, std 0.25 |
| `track_ang_vel` | +0.5 | Track commanded yaw rate, std 0.25 |
| `upright` | +1.5 | Torso upright, std √0.05 |
| `feet_air_time` | +1.5 | Reward swing phases of 0.05–0.30 s; only when ‖command‖ > 0.1 |
| `posture` | +1.5 | Stay near home pose (per-joint std below) |
| `joint_pos_limits` | −1.0 | Soft joint limit penalty |
| `action_rate` | −0.01 | L2 on action change |
| `joint_vel` | −0.005 | L2 on joint velocity |

Posture std (rad):

| Joint | Std |
|-------|-----|
| hip roll | 0.10 |
| hip pitch, thigh, calf | 0.40 |
| ankle pitch | 0.20 |
| ankle roll | 0.15 |
| shoulder pitch (Arms only) | 0.40 |
| shoulder roll (Arms only) | 0.20 |
| upper arm (Arms only) | 0.50 |
| elbow (Arms only) | 0.30 |

## Sensors

`foot_contact`: contact sensor on both `*_ankle_roll_link` bodies with air-time tracking. Used by `feet_air_time`.

## Terminations

| Term | Condition |
|------|-----------|
| `time_out` | 20 s episode |
| `fell_over` | Base tilt > 60° |

## Events

| Event | Mode | Effect |
|-------|------|--------|
| `reset_scene_to_default` | reset | Places the robot at its spawn pose. Must be first. |
| `randomize_yaw` | reset | Random yaw in [−π, π] at spawn |
| `push` | interval 4–10 s | ±0.5 m/s velocity kick in x and y |

The task has no domain randomization.

## Sim Parameters

```
timestep    = 0.005 s
decimation  = 4          (50 Hz control)
episode     = 20 s
njmax       = 200
nconmax     = 100
```

## Training

| | Locomotion | Locomotion-Arms |
|--|------------|-----------------|
| Network (actor and critic) | 256 × 128, exponential linear unit (ELU) | 256 × 128, ELU |
| `max_iterations` | 3000 | 3000 |
| Experiment dir | `logs/rsl_rl/piplus_locomotion` | `logs/rsl_rl/piplus_locomotion_arms` |

```bash
bash scripts/train.sh Mjlab-Piplus-Locomotion
bash scripts/train.sh Mjlab-Piplus-Locomotion-Arms
```

## Known Limitations

- **Arm collisions changed.** `get_spec_with_arms()` now resizes the torso and arm collision geoms. They match the visual meshes (see `ball_small.md`). `Locomotion-Arms` checkpoints that were trained before this change may rely on arms that pass through the torso.
- **Not sim2real-ready:** the actor reads `base_lin_vel`. The task has no observation noise and no domain randomization. `ball_small.md` describes the setup that closes this gap (proprioception-only actor, privileged critic, noise, randomization).
