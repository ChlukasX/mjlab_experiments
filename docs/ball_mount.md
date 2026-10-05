# Ball Mount + Balance (phase 4)

Branch: `feat/ball-mount` — **placeholder, not started**

Single policy that sets the stance, steps onto the ball and balances. Builds on the result of [mount_feasibility.md](mount_feasibility.md).

## Intent

- Warm start from Ball-Small. Actor gains noisy `ball_rel`; first-layer weights are expanded with zeros so the checkpoint still loads.
- Spawn curriculum: drop onto ball → standing next to ball → approach end states from [ball_approach.md](ball_approach.md).
- Success: 1 s of single-leg stance on the ball.
