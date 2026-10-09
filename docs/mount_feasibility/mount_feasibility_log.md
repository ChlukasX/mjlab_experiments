# Mount feasibility (size 5 ball): run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design and findings: [mount_feasibility.md](mount_feasibility.md). Shared conventions (launching, resuming, metrics, eval): [../README.md](../README.md).

Branch `feat/mount-feasibility`. Tasks `Mjlab-Piplus-Ball-Mount[-Flat]`: robot starts standing, ball 0.30 m ahead of the right foot, 10 s episodes. Spike runs v1 to v4 had wandb disabled.

Task `Mjlab-Piplus-Ball-Mount` (and `-Flat`): robot starts standing, ball 0.30 m ahead of the right foot, 10 s episodes. Spike runs v1 to v4 had wandb disabled.

| Run | Machine | Setup | Result | Takeaway |
|-----|---------|-------|--------|----------|
| `mount-spike-v1` | local, 2048 envs | From scratch, 400 it. Dense `foot_to_ball_top` reward. | Reward 22, episode 500/500, `foot_on_ball` 0 | Learns to stand, never steps on. |
| `mount-spike-v2-lift` | local, 2048 envs | Resumed v1 `model_399`, + `foot_lift` reward measured from 0.10 m. Stopped at 1600 it. | `foot_lift` 0.0000 throughout | Bug: standing ankle height is 0.050 m, so lifts below 5 cm paid nothing. |
| `mount-spike-v3-lift-fix` | local, 2048 envs | Resumed v2 `model_1600`, `STAND_FOOT_Z` = 0.05. To `model_4599`. | Mounting starts ~it 2000; peak reward 84 at it 2421, `foot_on_ball` ~3.0 (of 4); drifts to jerky control (reward 44, `action_rate` −2.8 at it 3756) | Mounting is feasible in sim. `single_leg_stance` was identical to `foot_on_ball` (Ball-Small check never needs a lifted foot). |
| `mount-spike-v4/v5-flat-sole` | local, 2048 envs | Resumed v3 `model_2400`, added flat-sole requirement and `sole_flat = exp(-tilt²/0.2²)`. v4 killed after 2 min; v5 (`r32i1cnm`) 3000 it. | v5: `foot_on_ball` (flat) 0, `foot_lift` 0.97 | Foot reaches the ball but the sole is rolled over. |
| `mount-flat-sole-scratch` | cl06, 4096 | Same flat shaping, from scratch, 5000 it (`7se9n1gp`). | Reward 40, `sole_flat` 0. Probe: median sole tilt 112°, 0 of 44672 samples under 17°. | `exp(-tilt²/0.2²)` ≈ 0 at that tilt: no gradient. |
| `mount-flat-sole-v2-scratch` | cl06, 4096 | `sole_flat = (1 + cos tilt)/2`, from scratch, 5000 it (`jm04uwmd`). | Reward 96, episode 483, `foot_on_ball` 3.72, `single_leg_stance` 3.57, `sole_flat` 1.88 | Flat sole learned from scratch. |
| `mount-balance-v1` | cl06, 4096 | Plain `Ball-Mount`, resumed v3 `model_2400`, 3000 it (`ioq6yazr`); free foot must clear the floor by 3 cm. | Reward 56, episode 437, `foot_on_ball` 3.37, `single_leg_stance` 3.25, `action_rate` −3.34 | Mount + balance without the flat requirement. Jerky. |
| `mount-flat-v1` | cl06, 4096 | `Ball-Mount-Flat`, from scratch, 5000 it (`w3mq6c0w`). | Reward 100, episode 487, `foot_on_ball` 3.74, `single_leg_stance` 3.61, `sole_flat` 1.89, `action_rate` −2.53 | Reproduces `v2-scratch`. Flat is better than plain on every metric; mean action change per step 0.47 vs 0.85. |
| `mount-smooth-v1` | cl06, 4096 | Plain `Ball-Mount`, from scratch, `action_rate` −0.05, `joint_vel` −0.015, 5000 it (`z2x62wty`). | Reward 25, `foot_on_ball` 0, `foot_lift` 0.39 | 5× smoothness from scratch never mounts. Smooth an existing policy instead. |

Measurements on the finished policies (probe scripts, not in the repo): on-ball ankle sits 0.049 m above the ball top with a flat sole (flat run) and 0.037 m (plain run, tilted sole).
