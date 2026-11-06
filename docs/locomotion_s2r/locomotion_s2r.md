# Locomotion S2R (phase 1)

Planned branch: `feat/locomotion-s2r` (not created yet) — **placeholder, not started**

This is Locomotion-Arms v2 with the same simulation-to-real (sim2real) setup as Ball-Small. S2R in the title means sim2real. The result can serve as the frozen low-level skill under the ball approach policy.

## Intent

- The actor uses proprioception only, with a 5-step history. The critic is privileged (`base_lin_vel`).
- The task adds observation noise and domain randomization (friction, mass, center of mass (CoM), encoder bias, proportional-derivative (PD) gains).
- The scene contains the ball as an obstacle.
- The reward penalizes the base yaw rate when no yaw is commanded.
- The action space has the same 20 degrees of freedom (DOF) as Ball-Small.

## Open questions

- Does the training start on the flat floor and continue on turf (same recipe as Ball-Small)?
