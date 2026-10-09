# Mjlab-Piplus-Ball / Mjlab-Piplus-Ball-Arms

Task IDs: `Mjlab-Piplus-Ball`, `Mjlab-Piplus-Ball-Arms`

The robot stands with both feet on top of a large free-rolling sphere. Its own foot forces make the ball roll, so the instability is entirely self-generated: there are no injected forces and no platform motion. The robot must keep the ball still underneath it or it slides off.

`Ball-Arms` is the same task with the arm joints actuated (20 DOF instead of 12), so arm swing can help manage angular momentum. It is a separate task so the two can be compared directly.

Config: `tasks/ball/piplus_ball_env_cfg.py`, `tasks/ball_arms/piplus_ball_arms_env_cfg.py`.

## Ball Geometry

```
BALL_RADIUS        = 0.45 m   (roughly an exercise ball at robot scale)
BALL_MASS          = 5.0 kg   (light enough to respond clearly to foot forces)
BALL_TOP           = 0.90 m
ROBOT_SPAWN_Z      = 1.40 m   (0.5 m free-fall onto the ball)
BASE_HEIGHT_TARGET = 1.25 m   (ball top + 0.35 m)
```

The ball is a sphere with a free joint (a physics body, not mocap), `condim=3`, friction `(0.9, 0.02, 0.001)`. It sits on a flat plane.

## Observations

| Term | Content |
|------|---------|
| `joint_pos` | Joint positions relative to home pose |
| `joint_vel` | Joint velocities relative to default |
| `projected_gravity` | Gravity in base frame (encodes tilt) |
| `actions` | Last policy output |
| `ball_state` | Ball position relative to robot base (3) + ball xy velocity (2) |

Actor and critic see the same terms. `ball_state` is ground truth from the simulator, so this policy is not deployable as-is: a real robot has no ball sensor. See `ball_small.md` for the proprioception-only setup.

No noise is configured on any term, so `enable_corruption` currently has no effect.

## Actions

One `JointPositionActionCfg` over all actuators, scale 0.5 rad, offset from the home pose.

| Task | Actuators | DOF |
|------|-----------|-----|
| Ball | legs (`ACTUATOR_5036`) | 12 |
| Ball-Arms | legs + arms (`ACTUATOR_4438`) | 20 |

## Reward Formulation

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

The arm stds are deliberately loose so the arms can find useful counterbalancing poses.

## Terminations

| Term | Condition |
|------|-----------|
| `time_out` | 20 s episode |
| `fell_over` | Base tilt > 70° |
| `ball_escaped` | Ball more than 3 m from env origin |
| `robot_off_ball` | Base lower than ball top + 0.3 m |

## Events

| Event | Mode | Effect |
|-------|------|--------|
| `reset_scene_to_default` | reset | Places robot and ball at their spawn poses. Must be first. |
| `randomize_yaw` | reset | Random yaw in [−π, π] at spawn |

No pushes and no domain randomisation.

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
| Network (actor and critic) | 128 × 128, ELU | 128 × 128, ELU |
| `max_iterations` | 1000 | 1000 |
| Experiment dir | `logs/rsl_rl/piplus_ball` | `logs/rsl_rl/piplus_ball_arms` |

```bash
bash scripts/train.sh Mjlab-Piplus-Ball
bash scripts/train.sh Mjlab-Piplus-Ball-Arms
```

## Known Limitations

- **Fall detection is orientation- and height-only.** `Ball-Small` had a failure mode where the robot fell while keeping its feet on the ball. `robot_off_ball` covers most of that here, but there is no check for non-foot bodies touching the ball or floor.
- **Arm collisions changed.** `get_spec_with_arms()` now resizes the torso and arm collision geoms to match the visual meshes (see `ball_small.md`). `Ball-Arms` checkpoints trained before that change may rely on arms passing through the torso.
- **Not sim2real-ready:** privileged ball state in the actor, no noise, no randomisation.
