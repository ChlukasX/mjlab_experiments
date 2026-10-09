# Size 1 ball (mount and balance): run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design and findings: [ball_size1.md](ball_size1.md). Shared conventions (launching, resuming, metrics, eval): [../README.md](../README.md).

Tasks `Mjlab-Piplus-Ball-Mount-Size1[-Flat]` and `Mjlab-Piplus-Ball-Balance-Size1`.

| Run | Machine | Setup | Result | Takeaway |
|-----|---------|-------|--------|----------|
| `mount-size1-v1` | cl06, 4096 | `Ball-Mount-Size1`, from scratch, 5000 it (`iwc06dqm`). | Reward 58, episode 431, `foot_on_ball` 3.37, `single_leg_stance` 3.28, `action_rate` −2.94 | Mounts a size 1 ball. |
| `mount-size1-flat-v1` | cl06, 4096 | `Ball-Mount-Size1-Flat`, from scratch, 5000 it (`47ya5hv8`). | Reward 93, episode 484, `foot_on_ball` 3.65, `single_leg_stance` 3.51, `sole_flat` 1.88, `action_rate` −2.61 | |
| `mount-size1-flat-v2-cont` | local, 4096 | Resumed `size1-flat-v1` `model_4999`, 5000 more (to `model_9998`; `qt3bwrlm`). Rewards were still rising. | Reward 115, episode 496, `foot_on_ball` 3.83, `single_leg_stance` 3.78, `sole_flat` 1.94, `action_rate` −1.58 | Continuing helped: reward 93 → 115 and smoother actions. |
| `balance-size1-flat-v1` | local, 4096 | `Ball-Balance-Size1` (drop onto the size 1 ball, flat sole), from scratch, 10000 it (`20doj5z7`). Drop spawn: base 0.5 m above the ball top. | Reward 163, episode 950/1000, `foot_on_ball` 3.74, `single_leg_stance` 3.72, `sole_flat` 1.89, `action_rate` −1.91 | Drop-and-balance is learnable on a fixed size 1 ball even with the higher drop. |

## Open

- Real size 1 ball radius and mass (0.07 m and 0.14 kg are assumptions).
