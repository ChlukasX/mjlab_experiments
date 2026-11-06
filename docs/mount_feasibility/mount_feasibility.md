# Ball Mount Feasibility (phase 0)

Branch: `feat/mount-feasibility` (deleted). Commits: `git log --oneline --grep '^ball-mount:'`. — **done; the main finding below was wrong**

> **Correction (Oct 10 2026).** The policies recorded here as mounting the ball did not mount it. They stand on the toes of the left foot on the floor with the right foot resting on the ball (centre of mass 0.30 m from the ball). The stance test measured the height of the ankle link. Tiptoe raises the link by 4.5 cm. This is enough to pass the 3 cm test. Therefore these runs do not establish "Mounting is feasible in sim" (sim: simulation) or the 99.7% figures below. The measurement, the corrected test (lowest corner of the free sole), and the first real mount are in [ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md).

The spike answers one question before the rest of the ball chain is built: **can the Pi Plus step from standing on the floor onto a 22 cm football and hold single-leg stance, in sim?**

The Ball-Small task hides this step by dropping the robot onto the ball.

Run-by-run record with launch commands: [mount_feasibility_log.md](mount_feasibility_log.md).

## Tasks

| Task | Success condition | Log dir |
|------|-------------------|---------|
| `Mjlab-Piplus-Ball-Mount` | Foot on the ball, other foot lifted | `logs/rsl_rl/piplus_ball_mount` |
| `Mjlab-Piplus-Ball-Mount-Flat` | Same, and the sole within 0.3 rad (~17°) of flat on the ball, plus a dense `sole_flat` reward | `logs/rsl_rl/piplus_ball_mount_flat` |
| `Mjlab-Piplus-Ball-Mount-Size1` | Plain, FIFA size 1 ball (radius 0.07 m, 0.14 kg assumed) | `logs/rsl_rl/piplus_ball_mount_size1` |
| `Mjlab-Piplus-Ball-Mount-Size1-Flat` | Flat sole, size 1 ball | `logs/rsl_rl/piplus_ball_mount_size1_flat` |

<!-- STE-deviation: FIFA is an acronym of a proper name; its first use is in the table above and the table stays unchanged -->

Both tasks use these settings. They have 20 degrees of freedom (DOF). The actor is the Ball-Small sim2real actor plus a noisy ball position in the base frame. The robot spawns standing with the ball 0.30 m ahead of the right foot. Episodes last 10 s. <!-- STE-deviation: "Both" kept from the source; the table lists 4 tasks and the source does not say which tasks it means --> Config: `tasks/ball_mount/piplus_ball_mount_env_cfg.py` (`flat_sole` and `ball_size` switches). The ball is placed 0.19 m + radius ahead of the base.

The tasks add the rewards `foot_to_ball_top` (2.0) and `foot_lift` (1.0) to Ball-Small. The `posture` weight is 0.3. `single_leg_stance` uses a real lift check (`LIFTED_FOOT_Z` = 8 cm).

All runs made before the split are in `piplus_ball_mount`, including the flat ones (`*flat-sole*`).

## On-ball test

The mount tasks use their own size-aware test. The Ball-Small test assumes the size-5 ball. The ankle link must be within `ON_BALL_Z_CENTER` ± `ON_BALL_Z_HALF` (0.045 ± 0.03 m) above the ball top. The ankle link must also be within radius + 0.05 m horizontally. The window comes from measurements of trained policies. The on-ball ankle sits 0.033–0.05 m above the top (0.049 with the sole flat). The ±6 cm window of Ball-Small around the top would count a foot that hovers beside a size 1 ball. Under this test, both finished size-5 policies are on the ball with the other foot lifted 99.7% of the time.

## Findings

- **Mounting is feasible in sim.** The ball does not roll away from the stepping foot. No stopper or extra friction was needed.
- **Dead reward zone:** The standing ankle height is 0.050 m, not 0.10 m. A lift reward measured from 0.10 m paid nothing for the first 5 cm. The policy never lifted (v2).
- **Ball-Small's `single_leg_stance` never requires a lifted foot.** It tests `foot_z > 0.02`. The ankle is 0.05 m high on the floor. Therefore the test equals `foot_on_ball`. `Ball-Small` is unchanged. The mount tasks use their own check.
- **Stronger smoothness penalties from scratch failed.** The weights were `action_rate` −0.05 and `joint_vel` −0.015 (5× / 3×) on the plain task. After 5000 iterations the policy never mounted (`foot_on_ball` ≈ 0, `foot_lift` 0.39). Smooth a policy that already mounts instead (resume, or ramp the weights).
- **Plain is much jerkier than flat.** Mean |Δaction| per step is 0.85 (plain) versus 0.47 (flat). The ankle sits 0.037 m above the ball top (plain, tilted sole) versus 0.049 m (flat).
- **Flat sole needs a shaping term with gradient.** `exp(-tilt²/0.2²)` is ~0 at 112° tilt. Therefore the first flat run stayed rolled over with the sole facing inward. `(1 + cos(tilt)) / 2` fixes this.

## Results (4096 envs, from scratch, flat variant)

Final logged values of `mount-flat-sole-v2-scratch` (5000 iterations):

| `foot_lift` | `foot_on_ball` | `single_leg_stance` | `sole_flat` |
|---|---|---|---|
| 0.96 | 3.72 | 3.57 | 1.88 (max 2.0) |

Plain variant: `mount-balance-v1` (resumed from `model_2400` of the earlier plain run, 5400 iterations): reward ~55, episode length ~430/500, `foot_on_ball` 3.3, `single_leg_stance` 3.2, `action_rate` −3.3.
Flat from scratch under the Flat task: `mount-flat-v1` reproduces `v2-scratch` (reward ~100, episode length ~488, `foot_on_ball` 3.74, `single_leg_stance` 3.62, `sole_flat` 1.89).

Size 1 ball: `mount-size1-v1` and `mount-size1-flat-v1` finished (see [ball_size1](../ball_size1/ball_size1.md)). The same tiptoe correction applies to them.

## Open

- The numbers of the flat variant look good. The behaviour is not signed off visually. Issues were reported during play.
- The policy becomes jerky late in training (the `action_rate` and `joint_vel` penalties grow). Stronger penalties from scratch did not learn to mount.
- Size 1: The sole is longer than the ball. Therefore "flat" means that the sole normal is along the ball's surface normal. It does not mean a flat contact patch.
