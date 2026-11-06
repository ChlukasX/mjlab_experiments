# Mjlab-Piplus-NonInertial

Task ID: `Mjlab-Piplus-NonInertial`

The robot learns to stand upright and to resist drift. Persistent sinusoidal platform accelerations act on the robot. The task models a ship deck, a vehicle, or an elevator. These surfaces accelerate, oscillate, and jolt without warning.

## Platform Simulation

The scene has no physical platform body. The task approximates the moving-platform effect. At each control step, it injects sinusoidal forces directly onto the root body:

```
F(t) = A · [sin(ωₓ·t + φₓ),  sin(ω_y·t + φ_y),  0]
```

The task randomizes the parameters per episode per environment:

| Parameter | Range | Notes |
|-----------|-------|-------|
| Amplitude A | [0, A_max] N | curriculum-controlled |
| Frequency ω | [0.1, 2.0] rad/s | 0.016–0.32 Hz, comparable to maritime swell |
| Phase φ | [0, 2π] | uniform random, independent x/y |

At a robot mass of 13 kg, 10 N gives 0.77 m/s² and 40 N gives 3.1 m/s² (about 0.3g).

The implementation uses the `mode="step"` event of mjlab (`apply_platform_forces`). This event fires at every control step for all environments. The platform state (phase, frequency, amplitude) is stored as per-environment tensors on the environment object. A `mode="reset"` event (`reset_platform_state`) resets the platform state at each episode.

## Observations

The task inherits all upright observations and adds:

| Term | Content |
|------|---------|
| `platform_accel` | Applied force / robot_mass → `[B, 3]` acceleration estimate |

This is a privileged signal during training. The policy gets the ground-truth platform acceleration. For real deployment, a proprioceptive state estimator must estimate it from the history of the inertial measurement unit (IMU).

## Reward Formulation

The task inherits all upright rewards unchanged and adds:

### XY Drift (weight −0.5)

```
r = −‖pos_xy − spawn_xy‖²
```

The reward penalizes horizontal drift from the spawn origin of the environment. On a real platform, the robot must stay near the center of the platform.

Full reward list: `upright`, `base_height`, `joint_pos_limits`, `posture`, `foot_flatness`, `xy_drift`, `action_rate`, `joint_vel`.

## Events

| Event | Mode | Parameters |
|-------|------|-----------|
| `push` | interval 2–8 s | ±1.0 m/s x/y velocity kick |
| `reset_platform` | reset | randomizes A, ω, φ per env |
| `platform_forces` | step (every control step) | applies F(t) to root body |

The push magnitude is ±1.0 m/s (upright: ±2.0 m/s). The platform forces already provide a continuous disturbance.

## Curriculum

The curriculum has three stages. A person advances the stages manually from checkpoint to checkpoint. Each stage takes about 500 iterations:

| Stage | A_max | Axes | A_max in g |
|-------|-------|------|-----------|
| 1 | 10 N | x only | 0.08g |
| 2 | 20 N | x + y | 0.16g |
| 3 | 40 N | x + y | 0.31g |

To run a stage with a given amplitude and given axes:

```python
# Stage 1 — edit in piplus_noninertial_env_cfg.py:
piplus_noninertial_env_cfg(amplitude_range=(0.0, 10.0), freq_range=(0.1, 2.0))
```

## Research Basis

The design follows **LAS-MP** (arXiv:2602.03367, 2026): active stabilization of a quadruped on a moving platform with 6 degrees of freedom (6-DOF), with curriculum learning and privileged platform state observations. The bipedal implementation adapts it as follows:

- The task injects forces instead of using a physical platform body (simpler, sufficient for Phase 1).
- The task uses the privileged `platform_accel` observation (LAS-MP uses the platform velocity and angular velocity).
- The curriculum increases the force amplitude instead of the number of trajectory waypoints.
- The xy_drift reward replaces the positional centering reward of LAS-MP.

## Phase 2 (Future)

Add a kinematic tilting platform body to the MuJoCo XML format (MJCF) model with `get_spec()`. Drive it with a trajectory event under proportional-derivative (PD) control. This enables contact-geometry tilt effects: foot slip on an incline and variation of the ground normal. LAS-MP reports roll and pitch of ±0.7 rad and yaw of ±2.6 rad for maritime conditions.
