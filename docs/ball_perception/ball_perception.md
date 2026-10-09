# Ball Perception (phase 2)

Branch: `feat/ball-perception` — **placeholder, not started**

Sim stand-in for a camera ball detector. Defines the interface the real robot's detector must fill.

## Intent

- Observation: ball position in the robot base frame (range/bearing or xyz).
- Corruption: Gaussian noise, random dropout, latency, field-of-view limit.
- Privileged sim state with realistic corruption; no rendered camera (too slow for 4096 envs).
- No ball velocity on the actor: the real robot cannot measure it.
