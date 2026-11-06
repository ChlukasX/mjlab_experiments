# Mjlab-Piplus-Ball / Mjlab-Piplus-Ball-Arms

Task IDs: `Mjlab-Piplus-Ball`, `Mjlab-Piplus-Ball-Arms`

The robot stands with both feet on top of a large free-rolling sphere. The foot forces of the robot make the ball roll. The instability is therefore entirely self-generated. The task has no injected forces and no platform motion. The robot must keep the ball still underneath it. Otherwise it slides off.

`Ball-Arms` is the same task with actuated arm joints (20 degrees of freedom (DOF) instead of 12). The arm swing can then help manage angular momentum. It is a separate task. This allows a direct comparison of the two tasks.

Config: `tasks/ball/piplus_ball_env_cfg.py`, `tasks/ball_arms/piplus_ball_arms_env_cfg.py`.

## Ball Geometry

```
BALL_RADIUS        = 0.45 m   (roughly an exercise ball at robot scale)
BALL_MASS          = 5.0 kg   (light enough to respond clearly to foot forces)
BALL_TOP           = 0.90 m
ROBOT_SPAWN_Z      = 1.40 m   (0.5 m free-fall onto the ball)
BASE_HEIGHT_TARGET = 1.25 m   (ball top + 0.35 m)
```

The ball is a sphere with a free joint (a physics body, not a motion capture (mocap) body). It has `condim=3` and friction `(0.9, 0.02, 0.001)`. It sits on a flat plane.

## Observations

| Term | Content |
|------|---------|
| `joint_pos` | Joint positions relative to home pose |
| `joint_vel` | Joint velocities relative to default |
| `projected_gravity` | Gravity in base frame (encodes tilt) |
| `actions` | Last policy output |
| `ball_state` | Ball position relative to robot base (3) + ball xy velocity (2) |

The actor and the critic see the same terms. `ball_state` is ground truth from the simulator. This policy is therefore not deployable as is, because a real robot has no ball sensor. See `ball_small.md` for the proprioception-only setup.

The configuration sets no noise on any term, so `enable_corruption` currently has no effect.

## Actions

One `JointPositionActionCfg` acts on all actuators. The scale is 0.5 rad. The action is an offset from the home pose.

| Task | Actuators | DOF |
|------|-----------|-----|
| Ball | legs (`ACTUATOR_5036`) | 12 |
| Ball-Arms | legs + arms (`ACTUATOR_4438`) | 20 |

## Reward Formulation

In this section, std means standard deviation.

| Term | Weight | Formula / purpose |
|------|--------|-------------------|
| `upright` | +1.5 | Torso upright, std √0.05 |
| `base_height` | +1.5 | `exp(−(z − 1.25)² / 0.15²)` |
| `posture` | +3.0 | Stay near home pose (per-joint std below) |
| `ball_control` | +3.0 | `exp(−‖v_ball‖² / 0.5²)` — keep the ball slow |
| `xy_centering` | +0.5 | `−‖base_xy − origin_xy‖²` |
| `joint_pos_limits` | −1.0 | Soft joint limit penalty |
| `action_rate` | −0.05 | L2 on action change |
| `joint_vel` | −0.005 | L2 on joint velocity |

Posture std (rad):

| Joint | Std |
|-------|-----|
| hip roll | 0.05 |
| hip pitch, thigh, calf | 0.40 |
| ankle pitch | 0.20 |
| ankle roll | 0.25 |
| shoulder pitch (Arms only) | 0.40 |
| shoulder roll (Arms only) | 0.20 |
| upper arm (Arms only) | 0.50 |
| elbow (Arms only) | 0.30 |

The arm std values are loose on purpose. The arms can then find useful counterbalancing poses.

## Terminations

| Term | Condition |
|------|-----------|
| `time_out` | 20 s episode |
| `fell_over` | Base tilt > 70° |
| `ball_escaped` | Ball more than 3 m from environment origin |
| `robot_off_ball` | Base lower than ball top + 0.3 m |

## Events

| Event | Mode | Effect |
|-------|------|--------|
| `reset_scene_to_default` | reset | Places robot and ball at their spawn poses. Must be first. |
| `randomize_yaw` | reset | Random yaw in [−π, π] at spawn |

The task has no pushes and no domain randomization.

## Sim Parameters

```
timestep    = 0.005 s
decimation  = 4          (50 Hz control)
episode     = 20 s
njmax       = 200
nconmax     = 100
```

## Training

| | Ball | Ball-Arms |
|--|------|-----------|
| Network (actor and critic) | 128 × 128, exponential linear unit (ELU) | 128 × 128, ELU |
| `max_iterations` | 1000 | 1000 |
| Experiment dir | `logs/rsl_rl/piplus_ball` | `logs/rsl_rl/piplus_ball_arms` |

```bash
bash scripts/train.sh Mjlab-Piplus-Ball
bash scripts/train.sh Mjlab-Piplus-Ball-Arms
```

## Known Limitations

- **Fall detection uses orientation and height only.** `Ball-Small` had a failure mode in which the robot fell and kept its feet on the ball. `robot_off_ball` covers most of these cases here. The task does not check whether non-foot bodies touch the ball or the floor. <!-- STE-deviation: no number in source ("most of") -->
- **Arm collisions changed.** `get_spec_with_arms()` now resizes the torso and arm collision geoms. They match the visual meshes (see `ball_small.md`). `Ball-Arms` checkpoints that were trained before this change may rely on arms that pass through the torso.
- **Not sim2real-ready:** the actor reads the privileged ball state. The task has no noise and no randomization.
