# Mjlab-Piplus-NonInertial

Task ID: `Mjlab-Piplus-NonInertial`

The robot learns to stand upright and resist drift while experiencing persistent sinusoidal platform accelerations. Models a ship deck, vehicle, or elevator — surfaces that accelerate, oscillate, and jolt without warning.

## Platform Simulation

No physical platform body in the scene. The moving-platform effect is approximated by injecting sinusoidal forces directly onto the root body each control step:

```
F(t) = A · [sin(ωₓ·t + φₓ),  sin(ω_y·t + φ_y),  0]
```

Parameters are randomized per episode per environment:

| Parameter | Range | Notes |
|-----------|-------|-------|
| Amplitude A | [0, A_max] N | curriculum-controlled |
| Frequency ω | [0.1, 2.0] rad/s | 0.016–0.32 Hz, comparable to maritime swell |
| Phase φ | [0, 2π] | uniform random, independent x/y |

At 13 kg robot mass, 10 N → 0.77 m/s², 40 N → 3.1 m/s² (≈ 0.3g).

The implementation uses mjlab's `mode="step"` event (`apply_platform_forces`) which fires every control step for all environments. Platform state (phase, frequency, amplitude) is stored as per-env tensors on the environment object and reset per-episode via a `mode="reset"` event (`reset_platform_state`).

## Observations

Inherits all upright observations plus:

| Term | Content |
|------|---------|
| `platform_accel` | Applied force / robot_mass → `[B, 3]` acceleration estimate |

This is a privileged signal during training — the policy gets ground-truth platform acceleration. For real deployment it would need to be estimated from IMU history (proprioceptive state estimator).

## Reward Formulation

Inherits all upright rewards unchanged. Adds:

### XY Drift (weight −0.5)

```
r = −‖pos_xy − spawn_xy‖²
```

Penalizes horizontal drift from the env spawn origin. On a real platform, staying near the center of the platform is critical.

Full reward list: `upright`, `base_height`, `joint_pos_limits`, `posture`, `foot_flatness`, `xy_drift`, `action_rate`, `joint_vel`.

## Events

| Event | Mode | Parameters |
|-------|------|-----------|
| `push` | interval 2–8 s | ±1.0 m/s x/y velocity kick |
| `reset_platform` | reset | randomizes A, ω, φ per env |
| `platform_forces` | step (every control step) | applies F(t) to root body |

Push magnitude is reduced to ±1.0 m/s (vs ±2.0 in upright) since platform forces already provide continuous disturbance.

## Curriculum

Three stages (manual checkpoint progression, ~500 iters each):

| Stage | A_max | Axes | A_max in g |
|-------|-------|------|-----------|
| 1 | 10 N | x only | 0.08g |
| 2 | 20 N | x + y | 0.16g |
| 3 | 40 N | x + y | 0.31g |

To run a stage with specific amplitude and axes:

```python
# Stage 1 — edit in piplus_noninertial_env_cfg.py:
piplus_noninertial_env_cfg(amplitude_range=(0.0, 10.0), freq_range=(0.1, 2.0))
```

## Research Basis

Design follows **LAS-MP** (arXiv:2602.03367, 2026): quadruped active stabilization on 6-DOF moving platform using curriculum learning and privileged platform state observations. Key adaptations for this bipedal implementation:

- Force injection instead of physical platform body (simpler, sufficient for Phase 1)
- Privileged `platform_accel` observation (LAS-MP uses platform velocity/angular velocity)
- Curriculum escalates force amplitude instead of trajectory waypoint count
- xy_drift reward replaces LAS-MP's positional centering reward

## Phase 2 (Future)

Add a kinematic tilting platform body to the MJCF via `get_spec()`. Drive it with a PD-controlled trajectory event. This enables contact-geometry tilt effects: foot slip on incline, ground normal variation. LAS-MP reports roll/pitch ±0.7 rad, yaw ±2.6 rad for maritime conditions.
