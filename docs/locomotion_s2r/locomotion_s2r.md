# Locomotion S2R (phase 1)

Branch: `feat/locomotion-s2r` — **placeholder, not started**

Locomotion-Arms v2 with the same sim2real setup as Ball-Small, so it can serve as the frozen low-level skill under the ball approach policy.

## Intent

- Proprioception-only actor with 5-step history; privileged critic (`base_lin_vel`).
- Observation noise and domain randomisation (friction, mass, COM, encoder bias, PD gains).
- Ball present in the scene as an obstacle.
- Penalty on base yaw rate where no yaw is commanded.
- Same 20 DOF action space as Ball-Small.

## Open questions

- Flat floor first, then continue on turf (same recipe as Ball-Small)?
