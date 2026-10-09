# Ball Mount Feasibility (phase 0)

Branch: `feat/mount-feasibility` — **in progress**

Spike to answer one question before building the rest of the ball chain: **can the Pi Plus step from standing on the floor onto a 22 cm football and hold single-leg stance, in sim?**

The Ball-Small task hides this step by dropping the robot onto the ball.

Run-by-run record with launch commands: [mount_feasibility_log.md](mount_feasibility_log.md).

## Tasks

| Task | Success condition | Log dir |
|------|-------------------|---------|
| `Mjlab-Piplus-Ball-Mount` | Foot on the ball, other foot lifted | `logs/rsl_rl/piplus_ball_mount` |
| `Mjlab-Piplus-Ball-Mount-Flat` | Same, and the sole within 0.3 rad (~17°) of flat on the ball, plus a dense `sole_flat` reward | `logs/rsl_rl/piplus_ball_mount_flat` |
| `Mjlab-Piplus-Ball-Mount-Size1` | Plain, FIFA size 1 ball (radius 0.07 m, 0.14 kg assumed) | `logs/rsl_rl/piplus_ball_mount_size1` |
| `Mjlab-Piplus-Ball-Mount-Size1-Flat` | Flat sole, size 1 ball | `logs/rsl_rl/piplus_ball_mount_size1_flat` |

Both: 20 DOF, Ball-Small sim2real actor plus a noisy ball position in the base frame, robot spawns standing with the ball 0.30 m ahead of the right foot, 10 s episodes. Config: `tasks/ball_mount/piplus_ball_mount_env_cfg.py` (`flat_sole` and `ball_size` switches). The ball is placed 0.19 m + radius ahead of the base.

Extra rewards over Ball-Small: `foot_to_ball_top` (2.0), `foot_lift` (1.0), `posture` weight 0.3. `single_leg_stance` uses a real lift check (`LIFTED_FOOT_Z` = 8 cm).

Runs made before the split are all in `piplus_ball_mount`, including the flat ones (`*flat-sole*`).

## On-ball test

The mount tasks use their own size-aware test (Ball-Small's assumes the size-5 ball): the ankle link must be within `ON_BALL_Z_CENTER` ± `ON_BALL_Z_HALF` (0.045 ± 0.03 m) above the ball top, and within radius + 0.05 m horizontally. The window comes from measuring trained policies: the on-ball ankle sits 0.033–0.05 m above the top (0.049 with the sole flat). Ball-Small's ±6 cm window around the top would count a foot hovering beside a size 1 ball. Both finished size-5 policies are on the ball with the other foot lifted 99.7% of the time under this test.

## Findings

- **Mounting is feasible in sim.** The ball does not roll away from the stepping foot, so no stopper or extra friction was needed.
- **Dead reward zone:** standing ankle height is 0.050 m, not 0.10 m. A lift reward measured from 0.10 m paid nothing for the first 5 cm and the policy never lifted (v2).
- **Ball-Small's `single_leg_stance` never requires a lifted foot.** It tests `foot_z > 0.02`, but the ankle is 0.05 m high on the floor, so it equals `foot_on_ball`. `Ball-Small` is unchanged; the mount tasks use their own check.
- **Stronger smoothness penalties from scratch failed.** `action_rate` −0.05 and `joint_vel` −0.015 (5× / 3×) on the plain task: after 5000 iterations the policy never mounted (`foot_on_ball` ≈ 0, `foot_lift` 0.39). Smooth a policy that already mounts instead (resume, or ramp the weights).
- **Plain is much jerkier than flat.** Mean |Δaction| per step 0.85 (plain) vs 0.47 (flat); ankle sits 0.037 m above the ball top (plain, tilted sole) vs 0.049 m (flat).
- **Flat sole needs a shaping term with gradient.** `exp(-tilt²/0.2²)` is ~0 at 112° tilt, so the first flat run stayed rolled over with the sole facing inward. `(1 + cos(tilt)) / 2` fixes that.

## Results (4096 envs, from scratch, flat variant)

`mount-flat-sole-v2-scratch`, 5000 iterations, final logged values:

| `foot_lift` | `foot_on_ball` | `single_leg_stance` | `sole_flat` |
|---|---|---|---|
| 0.96 | 3.72 | 3.57 | 1.88 (max 2.0) |

Plain variant: `mount-balance-v1` (resumed from `model_2400` of the earlier plain run, 5400 iterations): reward ~55, episode length ~430/500, `foot_on_ball` 3.3, `single_leg_stance` 3.2, `action_rate` −3.3.
Flat from scratch under the Flat task: `mount-flat-v1` reproduces `v2-scratch` (reward ~100, episode length ~488, `foot_on_ball` 3.74, `single_leg_stance` 3.62, `sole_flat` 1.89).

Size 1 ball: `mount-size1-v1` and `mount-size1-flat-v1` in progress, from scratch.

## Open

- The flat variant's numbers look good but the behaviour has not been signed off visually (reported issues while playing).
- Policy gets jerky late in training (`action_rate`, `joint_vel` penalties grow); stronger penalties from scratch did not learn to mount.
- Size 1: sole is longer than the ball, so "flat" there means sole normal along the ball's surface normal, not a flat contact patch.
