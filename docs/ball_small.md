# Mjlab-Piplus-Ball-Small / Mjlab-Piplus-Ball-Small-Turf

Task IDs: `Mjlab-Piplus-Ball-Small`, `Mjlab-Piplus-Ball-Small-Turf`

The robot balances on one leg on top of a football, with the other foot lifted. Arms are actuated (20 DOF) for counterbalancing. This is the task being prepared for the real Pi Plus, so it carries the sim2real setup: the actor only sees what the real robot can measure, and physics is randomised.

`Ball-Small-Turf` is the same task on an astroturf surface (RoboCup-style artificial grass) instead of a hard plane.

Config: `tasks/ball_small/piplus_ball_small_env_cfg.py`. Terrain: `terrains/astroturf.py`.

## Version History

| Version | Key changes |
|---------|-------------|
| v1 | Football balance, both feet, single-leg reward switched on after 12 000 steps |
| v2 | Spawn with right foot over the ball; single-leg reward active from step 0 |
| v3 | Fall terminations: non-foot body on floor, base height below 0.30 m |
| v4 | Sim2real: proprioception-only actor with noise and history, privileged critic, domain randomisation, pushes |
| v5 | Torso/arm collision geoms fitted to the visual meshes |
| **v6** | Mass randomisation via `pseudo_inertia`; astroturf terrain and `-Turf` task |

## Ball Geometry

```
BALL_RADIUS   = 0.11 m    (standard football)
BALL_MASS     = 0.43 kg
BALL_TOP      = 0.22 m
ROBOT_SPAWN_Z = 0.72 m    (0.5 m free-fall onto the ball)
ROBOT_SPAWN_X = −0.05 m   (base shifted so the right foot is over the ball)
```

The ball is a sphere with a free joint, `condim=6`, friction `(0.8, 0.01, 0.001)`. After the drop the robot lands with the right foot on the ball and the left foot on the floor (base at about 0.50 m).

## Observations

Asymmetric actor-critic: the critic is only used during training, so it may see simulator ground truth.

| Term | Actor | Critic | Actor noise (uniform ±) |
|------|:-----:|:------:|-------------------------|
| `joint_pos` | ✓ | ✓ | 0.01 rad |
| `joint_vel` | ✓ | ✓ | 1.5 rad/s |
| `projected_gravity` | ✓ | ✓ | 0.05 |
| `base_ang_vel` | ✓ | ✓ | 0.2 rad/s |
| `actions` | ✓ | ✓ | — |
| `base_lin_vel` | | ✓ | |
| `ball_rel` (ball pos rel. to base + ball xy vel) | | ✓ | |
| `foot_ball_dist` (each foot to ball centre) | | ✓ | |

- **Actor:** 66 values × 5-step history = 330 inputs. Joint terms come from the Pi Plus joint state perceptor, gravity and angular velocity from the IMU. The history lets the policy infer ball motion from its own dynamics.
- **Critic:** 80 inputs, no noise.
- Noise levels follow mjlab's reference velocity task. Noise is off in play mode.

Base linear velocity and ball state are excluded from the actor because the real robot cannot measure them.

## Actions

One `JointPositionActionCfg` over all 20 actuators (legs + arms), scale 0.5 rad, offset from the home pose.

## Reward Formulation

| Term | Weight | Purpose |
|------|--------|---------|
| `foot_on_ball` | +4.0 | 1 if at least one foot is on top of the ball |
| `single_leg_stance` | +4.0 | 1 if one foot is on the ball and the other is lifted |
| `upright` | +2.0 | Torso upright, std √0.05 |
| `posture` | +1.0 | Stay near home pose |
| `xy_centering` | +0.3 | `−‖base_xy − origin_xy‖²` |
| `joint_pos_limits` | −1.0 | Soft joint limit penalty |
| `action_rate` | −0.01 | L2 on action change |
| `joint_vel` | −0.005 | L2 on joint velocity |

A foot counts as "on the ball" when its `ankle_roll_link` is within 0.06 m vertically of the ball top and within `BALL_RADIUS + 0.05 m` horizontally of the ball centre.

Posture std (rad): hip roll 0.10; hip pitch, thigh, calf 0.40; ankle pitch 0.20; ankle roll 0.15; shoulder pitch 0.40; shoulder roll 0.20; upper arm 0.50; elbow 0.30.

## Terminations

| Term | Condition |
|------|-----------|
| `time_out` | 20 s episode (1000 steps) |
| `fell_over` | Base tilt > 60° |
| `body_on_floor` | Any robot body except the two `*_ankle_roll_link` feet touches the floor |
| `base_too_low` | Base height < 0.30 m |
| `ball_escaped` | Ball more than 3 m from env origin |

`body_on_floor` and `base_too_low` were added because tilt alone missed a failure mode: the robot slumped or lay against the ball with its feet still on it, staying under 60°. For reference, the base is about 0.39 m high when standing on flat floor.

`body_on_floor` uses mjlab's `illegal_contact` with the `body_floor_contact` sensor (25 non-foot bodies against the terrain body).

## Events

| Event | Mode | Effect |
|-------|------|--------|
| `reset_scene_to_default` | reset | Places robot and ball at spawn. Must be first. |
| `foot_friction` | startup | Foot sliding friction 0.3–1.2, shared across foot geoms |
| `ball_friction` | startup | Ball sliding friction 0.4–1.2 |
| `ball_mass` | startup | Ball mass × 0.8–1.2 (inertia scaled with it) |
| `robot_mass` | startup | Base and torso mass × 0.9–1.1 (inertia scaled with it) |
| `base_com` | startup | Base centre of mass ±2.5 cm in x/y, ±3 cm in z |
| `encoder_bias` | startup | Joint encoder bias ±0.015 rad |
| `pd_gains` | startup | Actuator kp and kd × 0.8–1.2 |
| `push_robot` | interval 2–5 s | Velocity kick: ±0.2 m/s x/y, ±0.2 rad/s roll/pitch, ±0.3 rad/s yaw |
| `ground_softness` | reset | **Turf task only.** Per-env turf softness, see below |

Mass uses `dr.pseudo_inertia` (mass scales by `exp(2·alpha)`), which keeps mass and inertia consistent. Pushes are much gentler than the locomotion task because the robot stands on one leg on a 22 cm ball. Pushes and randomisation stay on in play mode; only observation noise is disabled.

## Robot Collisions

`get_spec_with_arms()` calls `_fit_arm_collisions()` to fix collision geoms that the exported MJCF undersizes, which let actuated arms pass through the torso. The vendored `piplus.xml` is unchanged.

| Part | Change |
|------|--------|
| Torso | Box enlarged to x ±0.098, y ±0.10, z −0.01 to 0.20 (was ±0.078 all round) |
| Upper arm | Cylinder shifted down to cover z −0.06 to 0.04 |
| Elbow | New 5.6 cm box (had no collision geom) |
| Forearm | 8 mm rod replaced by a 2 cm-radius capsule |

This affects every task that uses `get_spec_with_arms()`: Ball-Small, Ball-Arms and Locomotion-Arms.

## Astroturf Terrain (`-Turf` task)

Two stacked planes, built by an `AstroturfTerrain` subclass of mjlab's `TerrainEntity`:

- **Soft top plane** at z = 0. Its contact `solref` time constant sets how far a foot or the ball sinks in. Contact priority 2, above the robot's geoms (priority 1), so the turf's parameters govern the contact.
- **Hard base plane** at z = −0.02 m. A rigid backstop: soft for the first 2 cm, then firm.

mjlab's `Scene` constructs `TerrainEntity` by name with no class hook, so importing `mjlab_piplus.terrains` rebinds that name to the subclass. Other terrain types are unaffected.

`ground_softness` samples the top-plane time constant per env on each reset from `[0.02, 0.02 + p · 0.04]`, where `p` ramps from 0 to 1 over the first 2000 iterations (48 000 env steps). Early training sees firm ground; later training sees the full 0.02–0.06 range.

Known gaps in the turf model:

- **No ball rolling resistance.** The top plane uses `condim=3` and wins every contact, so the ball rolls as freely as on a hard floor. On real turf, rolling resistance is likely the largest effect on the ball.
- **Softness range is an estimate**, not measured against real foot-sink tests.
- **Foot friction randomisation is overridden on turf**, because the top plane's friction (0.6) wins the contact.

## Sim Parameters

```
timestep    = 0.005 s
decimation  = 4          (50 Hz control)
episode     = 20 s
njmax       = 200
nconmax     = 100
```

## Training

Network: actor and critic 256 × 128, ELU. Both tasks log to `logs/rsl_rl/piplus_ball_small`. wandb projects are named after the task ID.

```bash
bash scripts/train.sh Mjlab-Piplus-Ball-Small

# Continue from a checkpoint (max-iterations is the number of additional iterations)
bash scripts/train.sh Mjlab-Piplus-Ball-Small-Turf \
  --agent.resume True \
  --agent.load-run <run_dir_name> \
  --agent.load-checkpoint model_N.pt \
  --agent.max-iterations 5000
```

Both tasks have the same observation and action sizes, so a flat-floor checkpoint can be continued on turf.

### Flat-floor results (v5, 4096 envs)

One run continued across four sessions, ending at `2026-10-01_09-30-11_sim2real-armcol-cont3/model_20996.pt`.

| Iteration | Mean reward | Mean episode length (of 1000) |
|-----------|-------------|-------------------------------|
| 3 000 | 19.6 | 224 |
| 9 000 | 51.3 | 525 |
| 15 000 | 64.6 | 610 |
| 20 000 | 71.2 | 658 |

Progress slowed from about 50 steps of episode length per 1000 iterations to about 10, so the run was treated as plateaued. Roughly 43% of episodes reached the 20 s limit.

That flat-trained checkpoint, evaluated on turf without retraining (32 envs, 20 s):

| Turf | Survived 20 s | Ball sink, mean / max |
|------|---------------|-----------------------|
| Firm (time constant 0.02) | 31 / 32 | 1.7 / 6.5 mm |
| Full softness range (0.02–0.06) | 26 / 32 | 6.6 / 20.8 mm |

## Known Issues

- **The policy spins.** The flat-trained policy rotates clockwise on top of the ball at about −1.4 rad/s (one turn every ~4.6 s) in every environment, while the ball itself barely spins. It persists with torsional friction enabled at the foot contact, so it is learned behaviour, not a contact artefact. No reward term penalises yaw rate. Planned fix: a small penalty on base yaw angular velocity.
- **Not yet modelled:** action latency, motor torque limits checked against real Pi Plus specs, ball size variation.
- **Old checkpoints:** anything trained before v4 has an 80-input actor and will not load.
