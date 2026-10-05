# Ball Approach (phase 3)

Branch: `feat/ball-approach` — **placeholder, not started**

High-level policy that walks the robot to the ball.

## Intent

- Action: `(vx, vy, ωz)` command fed to the frozen Locomotion S2R policy ([locomotion_s2r.md](locomotion_s2r.md)).
- Observation: noisy ball position from [ball_perception.md](ball_perception.md) plus proprioception.
- Reward: close the distance, face the ball, stop at the stand-off pose with the mounting foot toward the ball.
- End states are logged to seed the spawn distribution of [ball_mount.md](ball_mount.md).

## Open questions

- Does mjlab support an action term that wraps a frozen policy? If not, write one.
