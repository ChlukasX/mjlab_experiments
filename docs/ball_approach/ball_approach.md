# Ball Approach (phase 3)

Planned branch: `feat/ball-approach` (not created yet) — **placeholder, not started**

This is the high-level policy. It walks the robot to the ball.

## Intent

- Action: a `(vx, vy, ωz)` command. The frozen Locomotion S2R (simulation-to-real, sim2real) policy receives it ([locomotion_s2r.md](../locomotion_s2r/locomotion_s2r.md)).
- Observation: the noisy ball position from [ball_perception.md](../ball_perception/ball_perception.md) and the proprioception.
- Reward: the robot closes the distance, faces the ball, and stops at the stand-off pose with the mounting foot toward the ball.
- The task logs the end states. They seed the spawn distribution of [ball_mount.md](../ball_mount/ball_mount.md).

## Open questions

- Does mjlab support an action term that wraps a frozen policy? If not, write one.
