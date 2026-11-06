# Ball Perception (phase 2)

Planned branch: `feat/ball-perception` (not created yet) — **placeholder, not started**

This is the simulation stand-in for a camera ball detector. It defines the interface that the detector of the real robot must fill.

## Intent

- Observation: the ball position in the robot base frame (range and bearing, or xyz).
- Corruption: Gaussian noise, random dropout, latency, and a field-of-view limit.
- The observation is the privileged simulation state with realistic corruption. The task has no rendered camera. A rendered camera is too slow for 4096 environments.
- The actor gets no ball velocity. The real robot cannot measure it.
