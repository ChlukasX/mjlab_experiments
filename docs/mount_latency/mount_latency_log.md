# mount_latency log

Design: [mount_latency.md](mount_latency.md).

## Round 1 (`lat1`): latency curriculum from `bob2-up05`

Resume `bob2-up05` `model_39988` (copied to `piplus_ball_mountmin_blindonball_latency/2026-10-29_00-00-00_init-from-bob2-up05`, step counter zeroed), 4000 iterations, 2048 envs, seed 42, all `bob2-up05` weights passed as overrides (the task defaults differ): `single_leg_stance` 16 with `sole_clearance` 0.1, `upright` std 0.5, `action_rate` −0.02, `com_over_ball` 0, `action_excess` −4, `base_yaw_rate` −0.1, `torque_saturation` −2, `torque_demand` −0.5. Local 4090 (idle; cl06's 5090 had 1 GB more free but was 59% busy with another user's run), 2.0 s per iteration. Launched Oct 29 2026 in tmux `mjlab-lat1`.

Two false starts (aborted at the first iterations, run dirs removed; wandb `ksv4gooc` and one more left in the project): the curriculum read the resumed step counter (2.56 M) and started at the final 10-40 ms (episode length fell to ~60).

| Run | wandb | Outcome |
|-----|-------|---------|
| `lat1` | `8p9v4v82` | Finished Oct 29 2026 (`model_43987`, 2.2 h). Range at 10-40 ms from iteration ~41500; reward 80-110 and episode length 360-430 from then on, flat (plateau, not still learning). |

### Eval (Oct 29 2026)

`scripts/eval_mountbalance.py`, task `MountMin-BlindOnBall-Latency` for both checkpoints (same conditions), 2048 envs, seed 0, **observation noise on, encoder bias ±0.03 rad read by the actor** in every row unless stated; `--act-lag k k` = fixed latency. Success by ball size 1-2 / 3 / 4-5, then overall success and survive.

| Condition | `lat1` | `bob2-up05` |
|---|---|---|
| latency 0 ms | 21 / 54 / 71% → 48%, 62% | 55 / 79 / 83% → 72%, 79% |
| latency 10 ms | 50 / 88 / 93% → **77%**, 89% | 65 / 88 / 91% → **81%**, 88% |
| latency 20 ms | 45 / 88 / 93% → **75%**, 90% | 1 / 4 / 13% → **6%**, 10% |
| latency 30 ms | 32 / 78 / 86% → **65%**, 83% | 0% |
| latency 40 ms | 3 / 28 / 50% → 27%, 45% | 0% |
| latency 10-40 ms, jitter | 24 / 73 / 81% → 59%, 77% | 0% |
| 20 ms sensor delay + 10 ms | 24 / 67 / 82% → 58%, 76% | 0% |
| 20 ms sensor delay + 20 ms | 1 / 3 / 9% → 4%, 10% | 0% |
| 20 ms, RoboCup ball (r 0.077) | 53% → 53%, 86% | 0%, 2% |
| 20 ms, latch +2 cm | 46 / 89 / 91% → 75%, 89% | 1 / 4 / 13% → 6%, 10% |
| 20 ms, kp/kd ×0.5-0.8 | 25 / 66 / 79% → 56%, 78% | 9 / 40 / 62% → 36%, 48% |
| 20 ms, bias ±0.015 (trained by `bob2`) | 48 / 91 / 95% → 78%, 92% | |
| 20 ms, bias ±0.03, **noise off** | 69 / 98 / 97% → 88%, 96% | |
| 20 ms, no bias, noise off | 71 / 99 / 100% → 90%, 98% | |

**Findings**
- **Latency robustness gained:** at 20 ms 6% → 75%, at 30 ms 0% → 65%, at 10 ms on par (77% against 81%). It costs at 0 ms (48%), which is outside the trained range and not a real-robot condition. 40 ms and a 20 ms sensor delay on top of 20 ms (40 ms in total) are still out of reach: the useful range is up to ~30 ms.
- **Target (≥90% from 10 to 40 ms) not reached overall; reached on size 3-5 balls at 10-20 ms** (88-95% with noise and bias). The size 1-2 balls are the weak bin everywhere (≤53% with noise, ≤71% without), including the RoboCup ball at 20 ms (53%).
- **The observation noise is the largest remaining cost** (20 ms: 88% without, 75% with), more than the encoder bias (90% → 88%). The training noise levels are mjlab's velocity-task defaults (joint velocity ±1.5 rad/s, gyro ±0.2 rad/s, gravity ±0.05, uniform), not measured on the robot; if the real signals are cleaner, these rows are pessimistic.
- More iterations of the same setup would not help (training plateaued once the full range was in).

**Next (open):** measure the robot's loop latency and signal noise (rosbag of IMU, joint states and motor commands); then either narrow the latency range to the measured value and margin, or restrict the ball range to size 3-5 if the deployment ball is size 4-5, so the policy does not spend capacity on the smallest balls.

## Round 2 (`lat2-big`): size 3-5 balls only

Deployment ball decided: size 4-5. Task `MountMin-BlindOnBall-Latency-Big` (`big_ball=True`): ball radius 0.083-0.11 m (size 3-5, the eval's bins from 0.083 m on) instead of 0.07-0.11; mass and inertia keep following the radius on the same size line. Resume `lat1` `model_43987` (copied to `piplus_ball_mountmin_blindonball_latency_big/2026-10-29_00-00-00_init-from-lat1`; step counter kept, so the latency range is the final 10-40 ms from the start), 3000 iterations, 2048 envs, seed 42, the `lat1` overrides. Local 4090 (idle; cl06 56% busy), 2.0 s per iteration. Launched Oct 29 2026 in tmux `mjlab-lat2`.

| Run | wandb | Outcome |
|-----|-------|---------|
| `lat2-big` | `52vo6epy` | Finished Oct 29 2026 (`model_46986`, 1.7 h). Reward 130-160, episode length 470-580 (`lat1` on all sizes: 80-110, 360-430); level from ~45500 on. 16 iterations with a negative mean reward (down to −4900, from ~45500 on), each recovered within a few iterations: single envs with a huge penalty, not a collapse. |

### Eval (Oct 29 2026)

`scripts/eval_mountbalance.py`, task `MountMin-BlindOnBall-Latency-Big` (size 3-5) for both checkpoints, 2048 envs, seed 0, observation noise on, encoder bias ±0.03 rad read by the actor, unless stated. Success by ball size 3 / 4-5, then overall success and survive; `r` = all balls at that radius (size 4 ≈ 0.10 m, size 5 = 0.11 m).

| Condition | `lat2-big` | `lat1` |
|---|---|---|
| latency 10 ms | 91 / 94% → **92%**, 94% | 89 / 92% → 90%, 92% |
| latency 20 ms | 92 / 94% → **93%**, 95% | 87 / 92% → 89%, 92% |
| latency 30 ms | 79 / 89% → **84%**, 90% | 76 / 85% → 80%, 87% |
| latency 40 ms | 36 / 56% → 46%, 62% | 27 / 45% → 35%, 52% |
| latency 10-40 ms, jitter | 74 / 84% → 79%, 86% | 69 / 82% → 75%, 83% |
| 20 ms sensor delay + 10 ms | 77 / 86% → 81%, 88% | 68 / 82% → 74%, 83% |
| 20 ms, r 0.10 | **94%**, 96% | 91%, 93% |
| 30 ms, r 0.10 | **87%**, 91% | 83%, 87% |
| 20 ms, r 0.11 | **95%**, 96% | 93%, 93% |
| 20 ms, latch +2 cm | 92 / 94% → 93%, 95% | 87 / 93% → 90%, 93% |
| 20 ms, kp/kd ×0.5-0.8 | 64 / 79% → 71%, 85% | 70 / 77% → 73%, 85% |
| 20 ms, noise off | 98 / 98% → 98%, 99% | 97 / 98% → 97%, 98% |

**Findings**
- **Target reached at 10-20 ms on the deployment balls:** 94-95% on size 4-5 with noise and bias (92-93% over size 3-5). At 30 ms 87-89% on size 4-5, just under. 40 ms stays out of reach (56%).
- Dropping size 1-2 helped every latency row by 2-11 points over `lat1` on the same balls, most where it was weakest (40 ms +11, sensor delay + 10 ms +7, 30 ms +4). Weak gains at 10 ms (+2), where `lat1` was already close.
- The noise cost on these balls is small now (20 ms: 93% with, 98% without).
- Weaker actuators (kp/kd ×0.5-0.8) did not improve (71% against 73%): the one row where `lat2-big` does not lead; weak gains are not in training (×0.8-1.2).
- Latch +2 cm: no effect.

**Next (open):** measure the real loop latency; if it is ≤30 ms, `lat2-big` is the export candidate (`scripts/export_mount.py`, then sim2sim). Real kp/kd unknown; if the motors are much softer than the model, widen the gain randomization.

## Finding: the "single-leg" stance is a two-foot stance (Oct 29 2026)

Prompted by review comments on a `lat1` video (one foot tilted, "seems to avoid the sole"). Probe over the last 3 s of a 20 s episode, steps where `single_leg_stance` pays, 1024 envs, seed 0, noise on, bias ±0.03, MuJoCo contact forces between each ankle roll link and the ball (all 4 physics steps):

| Policy (latency) | free foot on the ball by position | free foot tilt vs ball normal (median) | free foot touching the ball | free foot's share of the foot-ball force |
|---|---|---|---|---|
| `bob2-up05` (0 ms) | 100% | 54° | 73-82% | 44% |
| `lat1` (20 ms) | 100% | 70° | 70-78% | 41% |
| `lat2-big` (20 ms) | 99% | 73° | 68-77% | 38% |

**Cause:** `single_leg_stance_lifted` (and `free_foot_lift`) call a foot free when it is not on the ball **with a flat sole**. A foot on the ball tilted beyond `MAX_SOLE_TILT` (0.3 rad) therefore counts as free, and its sole is above the 10 cm clearance anyway, since it rests on the ball. The policies hold the second foot on the side of the ball, tilted ~55-75°, carrying ~40% of the load. The eval's success test uses the same function, so every success rate in this and the mount_single log measures this two-foot stance, not a single-leg stance. No reward term looks at contacts.

**Also found:** the floor-contact termination (`body_floor_contact`) has `history_length=0`, so it sees only the last of the 4 physics steps per control step; floor touches of 5-15 ms between control steps go unseen. Not checked whether the policies use this.

### Strict eval (Oct 30 2026)

`eval_mountbalance.py` now also reports **strict success**: the stance test plus no contact (> 1 N) between the free foot and the ball in any of the 4 physics steps (contact sensors `r_foot_ball`, `l_foot_ball`, `history_length=4`). Contact only, not position: a lifted foot hovering beside the ball at ball-top height is a valid stance but passes the position window (`_foot_on_ball`, horizontal r + 0.05 m against a hip width of 0.1-0.15 m). `--floor-history` makes the floor termination check all 4 physics steps.

20 ms latency, noise on, bias ±0.03, 2048 envs, seed 0:

| Policy | success (old test) | strict success | with `--floor-history` |
|---|---|---|---|
| `lat2-big` | 92% | **0%** (also 0% at 50% hold) | 92% / 0% |
| `lat1` | 75% | **0%** | |

- No policy holds a single-leg stance; all success rates in this log and the mount_single log are two-foot stances.
- Checking every physics step for floor contact changes nothing: the policies do not use short floor touches between control steps.

## Round 3 (`of1`): free foot off the ball

Task `MountMin-BlindOnBall-Latency-Big-OneFoot` (`one_foot=True`): `single_leg_stance` and `free_foot_lift` count a foot as free only if it does not touch the ball in any physics step (`free_off_ball`, contact sensors from `ball_contact_sensors`). Otherwise as `lat2-big`. Resume `lat2-big` `model_46986` (copied to `piplus_ball_mountmin_blindonball_latency_big_onefoot/2026-10-29_00-00-00_init-from-lat2-big`), 3000 iterations, 2048 envs, seed 42, the `lat1` overrides. Local 4090 (cl06 89% busy). Launched Oct 30 2026 in tmux `mjlab-of1`.

First iterations: mean reward 150 → 80 and stance reward per episode 6.9 → 3.0 (the two-foot stance no longer pays), episode length unchanged (~520).

| Run | wandb | Outcome |
|-----|-------|---------|
| `of1` | `fodiwmf2` | **Collapsed** at iteration 48197. Until then good: reward back to ~160 by 47600, stance reward per episode 5.9-6.7 (more than `lat2-big` paid for the two-foot stance), episode length ~590. Then `action_rate` went from ~−0.3 to −577 000 per episode within 3 iterations: the raw policy output grew without bound (`action_rate` squares the raw output, before any clip), episodes fell to ~9 steps and never recovered (final `model_49985`: 0% in every eval row). Spikes before it: −276 at 48125; `lat2-big`'s negative mean rewards (down to −4900) were the same mechanism, recovered there. |
| `of2-clip` | `vaxckr6f` | Finished Oct 30 2026 (`model_49949`, 1.2 h). Resume `of1` `model_47950` (copied to `.../2026-10-30_00-00-00_init-from-of1-47950`), 2000 iterations, raw policy output clamped to ±7 (`RAW_ACTION_CLIP`, runner `clip_actions`; 7 is above the largest offset the joint clip lets through, hip pitch 6.7, so no target is lost and the deployed policy is unaffected). Stable: no negative mean reward (lowest 1.4; `action_rate` ≥ −0.91 per episode). Reward 160 → 179, stance reward per episode 6.0 → 6.7, episode length 580 → 615, still rising slowly at the end. |

**Eval `of1` `model_47950`** (before the collapse; 2048 envs, seed 0, noise on, bias ±0.03; success by size 3 / 4-5, then overall; strict = free foot never touching the ball):

| Latency | success | strict | survive |
|---|---|---|---|
| 10 ms | 80 / 86% → 83% | 79 / 86% → **82%** | 89% |
| 20 ms | 83 / 93% → 88% | 82 / 92% → **87%** | 94% |
| 30 ms | 68 / 86% → 76% | 65 / 83% → **74%** | 88% |

- The fix works: strict success 0% → 87% at 20 ms after ~1000 iterations, and old and strict success now agree (≤3 points apart): the policy stands on one foot.
- Below `lat2-big`'s old-test numbers (92 / 93 / 84%), since one foot is harder than two; size 3 costs most.

**Eval `of2-clip` `model_49949`**: the round 2 sweep (task `-Latency-Big`, 2048 envs, seed 0, noise on, bias ±0.03 unless stated). Success by size 3 / 4-5 as old test / strict, then overall success, survive and strict; `r` = all balls at that radius.

| Condition | size 3 | size 4-5 | success, survive | **strict** | `lat2-big` (old test, two feet) |
|---|---|---|---|---|---|
| latency 10 ms | 83 / 82% | 89 / 88% | 86%, 91% | **84%** | 92% |
| latency 20 ms | 86 / 86% | 93 / 93% | 89%, 95% | **89%** | 93% |
| latency 30 ms | 72 / 69% | 86 / 85% | 79%, 89% | **77%** | 84% |
| latency 40 ms | 23 / 19% | 46 / 41% | 34%, 58% | **29%** | 46% |
| latency 10-40 ms, jitter | 58 / 54% | 80 / 77% | 68%, 84% | **65%** | 79% |
| 20 ms sensor delay + 10 ms | 67 / 64% | 87 / 85% | 76%, 88% | **74%** | 81% |
| 20 ms, r 0.10 | | 92 / 92% | 92%, 95% | **92%** | 94% |
| 30 ms, r 0.10 | | 85 / 83% | 85%, 91% | **83%** | 87% |
| 20 ms, r 0.11 | | 93 / 93% | 93%, 94% | **93%** | 95% |
| 20 ms, latch +2 cm | 86 / 85% | 93 / 93% | 89%, 94% | **89%** | 93% |
| 20 ms, kp/kd ×0.5-0.8 | 46 / 43% | 69 / 65% | 57%, 81% | **53%** | 71% |
| 20 ms, noise off | 94 / 94% | 96 / 96% | 95%, 96% | **95%** | 98% |

**Findings**
- **Real single-leg stance on the deployment balls:** strict 92-93% at 20 ms on size 4-5 (r 0.10 / 0.11), 83% at 30 ms. Old and strict test within 5 points everywhere: the free foot stays off the ball.
- Against `model_47950`: +2 to +3 points (20 ms 87 → 89%, 30 ms 74 → 77%); training still rising slowly.
- The cost of one foot against the two-foot `lat2-big`: 1-4 points on size 4-5 at 10-30 ms, more under hard conditions (jitter −14, 40 ms −17, soft motors −18). Soft motors (kp/kd ×0.5-0.8) are the weakest row (53%): one leg has less margin than two.
- Size 3 is the weak bin (82-86% at 10-20 ms).

**Next (open):** measure the real loop latency and motor gains. If latency ≤ 20 ms: export `of2-clip` (`scripts/export_mount.py`, sim2sim with the strict test). Otherwise, or for the soft-motor row: continue training (still rising), widen the kp/kd randomization if the real gains are softer.

### Finding: `of2-clip` stands on the flank of the ball with the ankle rolled to its limit (Oct 30 2026)

Review of an `of2-clip` video: "balances on the side of the foot instead of the sole". Probe (`logs/probes/side_probe.py`, gitignored): last 3 s of 1024 episodes, stance steps, 20 ms, noise, bias ±0.03, contact force per foot geom with the ball over all 4 physics steps.

| Measure | `of2-clip` | `lat2-big` |
|---|---|---|
| support-foot contact: sole capsules / ankle box | 90% / 0% of steps | 90% / 0% |
| edge only (outer capsule, no middle one) | 2% | |
| outer capsules' share of the support force | 22% | |
| sole tilt vs the ball surface (median) | 5.8° | 6.2° |
| **sole tilt vs world vertical** (median, p10-p90) | **24.7°** (20-30°) | |
| **foot position, angle down from the ball top** (median) | **23°** | |
| **support ankle roll** (median; limit ±0.648 rad) | **+0.60 rad (34°), at the limit 26%** | +0.62 rad, 36% |
| base tilt (median) | 30° | |

- In physics the load is on the sole (the visual mesh matches the collision sole within 3 mm), not on the edge.
- But the foot stands ~23° down the side of the ball, its sole tangent to the ball there (25° from horizontal), and the whole robot leans 30° toward the ball. The leg reaches it with the ankle roll at its mechanical limit (34° against the shin): from outside, the robot stands on the side of a rolled foot.
- The rewards allow it: "flat" is measured against the ball surface under the foot (`_sole_tilt`), not against vertical, so a foot on the flank counts as flat; `joint_pos_limits` costs only the 0.02 rad past the soft limit (0.583 rad). Inherited from `lat2-big` (same ankle posture), not caused by the one-foot fix.
- Not deployable: the ankle at its stop all the time, and a stance on the flank depends on friction and pushes the ball sideways.

## Round 4 (`lv1`): level sole, ankle roll off its stop

Task `MountMin-BlindOnBall-Latency-Big-OneFoot-Level` (`level=True`), on top of `-OneFoot`:
1. **Level sole:** every stance term with `flat=True` gets `flat="level"`: the support sole must be within `MAX_SOLE_TILT` (0.3 rad) of the ball surface **and** of horizontal, so the foot must stand near the ball top. `sole_flat` (dense) uses the larger of the two tilts.
2. **Ankle roll:** new term `ankle_roll_excess`, weight −20: Σ over both ankle roll joints of |q| beyond 0.45 rad (`ANKLE_ROLL_SOFT`; stop 0.648 rad). At `of2-clip`'s 0.60-0.70 rad it costs 3-5 per step against 16 for the stance.

The eval adds **level success**: strict plus the level sole, and prints the support ankle roll. `of2-clip` on it (20 ms, noise, bias): success 89%, strict 88%, **level 0%**, ankle roll at the stop (> 0.62 rad) in 97% of stance steps (median max |q| 0.70 rad).

Resume `of2-clip` `model_49949` (copied to `piplus_ball_mountmin_blindonball_latency_big_onefoot_level/2026-10-30_00-00-00_init-from-of2-clip`), 3000 iterations, 2048 envs, seed 42, the `lat1` overrides, clip ±7. Local 4090. Launched Oct 30 2026 in tmux `mjlab-lv1`. First iterations: mean reward 179 → 39, stance reward per episode 6.7 → 2.9, `ankle_roll_excess` −1.6 per episode, episode length unchanged (~595).

| Run | wandb | Outcome |
|-----|-------|---------|
| `lv1` | `94oc64wb` | Finished Oct 30 2026 (`model_52948`). Negative mean rewards only in the first ~40 iterations (lowest −57, the switch to the new rewards), none after. Reward 120 → 220 by ~51400, then flat (plateau); stance reward per episode 4.9 → 7.2; `ankle_roll_excess` −1.3 → −0.45 per episode. |

**Eval `lv1` `model_52948`** (task `-Level`, 2048 envs, seed 0, noise on, bias ±0.03 unless stated). Success by size 3 / 4-5 as old test / strict / level; then overall; ankle roll = median of the larger |ankle roll| in stance steps.

| Condition | size 3 | size 4-5 | overall success / strict / **level**, survive | ankle roll |
|---|---|---|---|---|
| latency 10 ms | 61 / 57 / 54% | 81 / 75 / 73% | 70 / 65 / **63%**, 94% | 0.44 |
| latency 20 ms | 53 / 51 / 47% | 77 / 74 / 70% | 64 / 61 / **58%**, 95% | 0.43 |
| latency 30 ms | 36 / 32 / 25% | 61 / 55 / 46% | 48 / 43 / **35%**, 93% | 0.43 |
| latency 40 ms | 7 / 5 / 3% | 21 / 16 / 9% | 13 / 10 / **6%**, 66% | 0.43 |
| latency 10-40 ms, jitter | 28 / 25 / 19% | 52 / 47 / 38% | 40 / 35 / **28%**, 90% | 0.43 |
| 20 ms sensor delay + 10 ms | 32 / 27 / 21% | 58 / 53 / 43% | 44 / 39 / **31%**, 92% | 0.43 |
| 20 ms, r 0.10 | | 75 / 72 / 69% | **69%**, 97% | 0.44 |
| 20 ms, r 0.11 | | 81 / 78 / 75% | **75%**, 94% | 0.44 |
| 20 ms, kp/kd ×0.5-0.8 | 11 / 10 / 7% | 27 / 24 / 15% | 19 / 16 / **11%**, 85% | 0.42 |
| 20 ms, noise off | 82 / 82 / 82% | 89 / 89 / 88% | 85 / 85 / **85%**, 95% | 0.44 |

Foot probe (20 ms, noise, bias; `of2-clip` in brackets): support sole 10.6° from horizontal (24.7°), foot 12° from the ball top (23°), ankle roll median +0.27 rad, at the stop 0% (+0.60, 26%), load on the sole capsules (83% of steps, ankle box 0%, edge only 2%), base tilt 32° (30°).

**Findings**
- **The posture is fixed:** the foot stands near the top with a level sole, the ankle roll is off its stop (0% of steps, from 97%). The three tests now agree within 3-5 points.
- **The cost is large:** level success at 20 ms 58% (size 4-5: 70-75%), against 88% strict for the flank stance of `of2-clip`. The flank stance had been easier: the leaning robot rests against the ball's side.
- **Observation noise is the main cost** (20 ms: 85% without, 58% with), more than for any earlier policy (`of2-clip`: 95 against 89%). Soft motors fall to 11%; 30-40 ms and jitter lose most.
- Survival stays at 93-97%: the failures are mostly no clean stance in the last 3 s, not falls.
- The base still tilts 32°: `upright` (std 0.5) has not changed the lean in any round.
- Training plateaued by ~51400: more iterations of the same setup are unlikely to help.

**Next (open):** the level stance is right but not robust. Options: (a) noise levels measured on the robot instead of mjlab's velocity-task defaults (the largest single cost); (b) a curriculum on the noise; (c) address the 32° lean (stronger `upright`), which leaves a smaller margin on top of the ball.

## Round 5 (`nr1-up6`): noise curriculum, stronger `upright`

From `lv1`'s next steps (b) and (c). Task `MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp` (`noise_ramp=True`), on top of `-Level`:
- **Noise curriculum** (`noise_curriculum`): every uniform actor noise range is scaled from 0.25× (`NOISE_START`) to 1× over 1500 iterations (`NOISE_RAMP_STEPS`), counted from `t0` = the resumed checkpoint's step counter (829 760, passed as `--env.curriculum.noise.params.t0`; the latency curriculum stays at its final 10-40 ms). Full noise for the last 1500 iterations.
- **`upright` weight 2 → 6** (std 0.5; override): at the 32° lean it pays 0.33 per step, at 15° 0.83.

Resume `lv1` `model_52948` (copied to `piplus_ball_mountmin_blindonball_latency_big_onefoot_level_noiseramp/2026-10-30_00-00-00_init-from-lv1`), 3000 iterations, 2048 envs, seed 42, the `lat1` overrides except `upright`, clip ±7. Local 4090. Launched Oct 30 2026 in tmux `mjlab-nr1`. Checked: noise scale 0.286 at iteration 72 (as computed), latency at 10-40 ms.

| Run | wandb | Outcome |
|-----|-------|---------|
| `nr1-up6` | `0qfx9y03` | Finished Oct 31 2026 (`model_55947`). No negative mean reward (lowest 4.6). Reward fell with the noise ramp (367 at 0.25× → 215 at 1×, iteration ~54450), then rose to 230-250 at full noise, still rising slowly at the end. Stance reward per episode 11.0 → 6.7 → 7.2-7.7 (`lv1` at its end: 7.1 at full noise). |

**Eval `nr1-up6` `model_55947`** (task `-Level`, full noise unless stated, bias ±0.03, 2048 envs, seed 0). Level success (strict + level sole) by size 3 / 4-5, overall; `lv1` overall in brackets.

| Condition | size 3 | size 4-5 | overall level (`lv1`), survive |
|---|---|---|---|
| latency 10 ms | 56% | 75% | **65%** (63%), 92% |
| latency 20 ms | 57% | 76% | **66%** (58%), 95% |
| latency 30 ms | 27% | 54% | **40%** (35%), 92% |
| latency 40 ms | | | **7%** (6%), 66% |
| latency 10-40 ms, jitter | | | **33%** (28%), 88% |
| 20 ms sensor delay + 10 ms | | | **37%** (31%), 93% |
| 20 ms, r 0.10 | | 75% | **75%** (69%), 96% |
| 20 ms, r 0.11 | | 83% | **83%** (75%), 94% |
| 20 ms, kp/kd ×0.5-0.8 | 10% | 24% | **16%** (11%), 82% |
| 20 ms, noise off | 86% | 91% | **88%** (85%), 96% |

Ankle roll at the stop 0-1% in every row. Foot probe (20 ms, `lv1` in brackets): sole 10.3° from horizontal (10.6°), foot 12° from the top (12°), ankle roll +0.29 rad (+0.27), **base tilt 28.5° (32°)**.

**Findings**
- Gains everywhere, but small: +2 to +8 points level success (20 ms 58 → 66%, size 4-5 at r 0.11 75 → 83%), noise off 85 → 88%.
- The noise gap remains: 20 ms 88% without, 66% with (lv1: 85 / 58%). The ramp made the policy no more noise-robust in proportion; it mainly improved the base policy.
- `upright` ×3 lowered the lean only 32° → 28.5°.
- The level posture held (ankle off its stop, foot near the top).
- Still rising slowly at the end (reward 230 → 250 over the last 1500 iterations at full noise).

**Next (open):** the noise is the remaining limit, and the training noise is mjlab's velocity-task default, not measured: a rosbag (IMU, joint states) gives the real levels; joint velocity ±1.5 rad/s is the likely largest term (to check: eval with one noise term off at a time). The lean needs more than `upright` (e.g. a CoM-over-foot term, or a narrower `upright` std).

### Noise-term check on `nr1-up6` (Nov 1 2026)

`eval_mountbalance.py --noise-terms`: only the listed actor terms keep their training noise. 20 ms, bias ±0.03, 2048 envs, seed 0, `nr1-up6` `model_55947`. Strict success by size 3 / 4-5, then level success overall. Training noise (uniform ±): joint position 0.01 rad, joint velocity 1.5 rad/s, gravity 0.05, gyro 0.2 rad/s, ball position 0.02 m (per step, on top of the per-episode bias).

| Noise | size 3 / 4-5 (strict) | level | cost against no noise |
|---|---|---|---|
| all terms | 62 / 77% | **67%** | −21 |
| all but joint position | 61 / 79% | 67% | −21 |
| all but joint velocity | 70 / 84% | 76% | −12 |
| all but gravity | 69 / 83% | 75% | −13 |
| all but gyro | 65 / 83% | 72% | −16 |
| all but ball position | 73 / 85% | 78% | −10 |
| joint position alone | 85 / 91% | 88% | 0 |
| joint velocity alone | 81 / 89% | 84% | −4 |
| gravity alone | 83 / 91% | 87% | −1 |
| gyro alone | 84 / 92% | 88% | 0 |
| ball position alone | 79 / 90% | 84% | −4 |
| none | 85 / 91% | 88% | |

- **No single dominant term:** alone, each costs at most 4 points; together 21. The costs compound. Taking out one term gains 5-11 points; the largest are ball position (+11), joint velocity (+9) and gravity (+8). Joint position noise costs nothing.
- ~~**Ball position noise after the latch is not the cause:**~~ Wrong, see the latch finding below: the latch never fired, so there were no zeros; the ball noise acted through the whole balance.
- (original:) in sim the term's noise also hit the zeros the actor gets once on the ball (the robot sends exact zeros). With the noise moved before the zeros (`ball_pos_until_on_ball(step_noise=...)`, eval `--ball-noise-clean-zeros`): 64 / 66 / 41% level at 10 / 20 / 30 ms, unchanged. The ball noise costs during the mount (foot placement), before the latch.
- The per-step ball noise (iid ±2 cm every 20 ms) is likely pessimistic: on the robot the ball position before the mount is the ball filter's estimate of a stationary ball, smooth from step to step; its error is the per-episode bias, which stays in.
- What a rosbag has to give: the real joint velocity, gravity and gyro noise (the three terms after the ball that cost most), and the ball filter's step-to-step jitter.

## Finding: the robot was never blind on the ball (Nov 2 2026)

Question from a play session: why does the policy have a ball input on the ball? The latch ("on the ball" → ball input zeros) needs base height > 0.42 m **and** base tilt < 0.35 rad (20°). Probe `logs/probes/latch_probe.py`, 1024 envs, 20 s:

| Policy | envs ever latched | base tilt in the stance (median) | base height (median) |
|---|---|---|---|
| `bob2-up05` | 0% | 28° | 0.503 m |
| `lat1` | 0% | 30° | 0.500 m |
| `nr1-up6` | 0% | 28° | 0.498 m |

- The height passes, the tilt never: every policy since blind-on-ball (mount_single round 5) balances with the base 28-32° tilted, so none ever latched. They all saw the true ball position (plus bias and noise) through the whole balance. The robot node uses the same rule (`deploy/bitbots/mount_obs.py`), so the sim2sim check had the same gap; on the robot the camera cannot see a ball under the feet.
- All success rates since mount_single round 5 are with the ball visible on the ball.
- Correction to the noise-term check: the ball noise acted through the whole balance, not only during the mount (no zeros were ever sent).
- `nr1-up6` with the latch fixed (45°, below): **0% success, 0% survival** (20 ms, with and without noise): it depends entirely on the ball input.

## Round 6 (`bl1`): latch at 45°, 1 s start hold

Task `MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold`, on top of `-NoiseRamp`:
- **Latch** (`latch_fix`): base tilt below 0.79 rad (45°, `LATCH_MAX_TILT`) instead of 0.35; a robot falling sideways still does not count. Checked: with `nr1-up6`, 64% of envs latch within 8 s (then fall).
- **Start hold** (`start_hold`): for 1 s (`START_HOLD_S`, fixed) after each reset the robot holds the home pose: `HoldJointPositionAction` sets the raw action to 0 and also the action manager's copy, so the policy's last-action input reads 0, as on the robot after activation. Checked: action and last action 0 for the first 50 steps, the policy acts from step 51. `scripts/export_mount.py` writes `start_hold_s` to the YAML for the node.
- **Critic input `start_hold`:** seconds left in the hold. It changes the critic's input size, so (rule in AGENTS.md) the task is a new experiment: the warm start loads only the actor from `nr1-up6` (unchanged inputs) and starts the critic and optimizer fresh (`logs/probes/make_actor_init.py`; checked: actor identical, critic untouched, step counter kept).

Actor from `nr1-up6` `model_55947`, fresh critic (`piplus_ball_mountmin_blindonball_latency_big_onefoot_level_noiseramp_latch45_hold/2026-11-02_00-00-00_init-actor-from-nr1-up6`), 4000 iterations, 2048 envs, seed 42, the round 5 overrides (`upright` 6), noise ramp restarted from 0.25× (`t0` 1 021 760). Local 4090. Launched Nov 2 2026 in tmux `mjlab-bl2`. First iterations: episode length ~70 steps (falls once blind; the fresh critic still learning), stance reward ~0.

| Run | wandb | Outcome |
|-----|-------|---------|
| `bl1` | `lkn5u020` | Stopped after ~100 iterations, run dir removed: trained without the critic input (dropped so the full warm start would load); replaced by `bl2`. |
| `bl2` | `9ihbeq9e` | Finished Nov 2 2026 (`model_59946`). **Never mounts:** 0% success in every condition, survival 100% (58-83% soft motors / noise off), 0% of envs ever latched, stance reward 0 for the whole run. It stands on the floor with a foot resting on the ball top (end, per episode: `upright` 5.7, `foot_to_ball_top` 1.6, `sole_flat` 1.3, `foot_lift` 0.9). Two causes: going up meant going blind, which meant falling; and the fresh critic broke the actor at the start (see `bl3`, first try). |
| `bl3-blindcurr` (first try) | | Stopped after ~300 iterations, run dir removed. Blindness curriculum (below) with the fresh critic of `bl2`: from iteration ~40 every episode ended in `base_too_low` after ~63 steps while blind p was still 0.02, so the fresh critic wrecked the warm-started actor in the first updates, not the blindness. |
| `bl3-blindcurr` | `imymhfx1` | Finished Nov 3 2026 (`model_59946`). **Parked once fully blind:** stance reward per episode 6.6 (p 0.2) → 0.9 (p 0.8) → 0.26 (p 1, falls 9.8 per log interval) → 0 from iteration ~58300, episode length back to ~900 on the floor. Sweep: 0% success in every condition, survival 99-100% (57% soft motors), 0% latched. The curriculum kept the mount while sighted but blind balance was never learned: the stance reward dropped about as fast as the blind share rose. |
| `bl4-deadline` | `woxp2cxd` | **Stopped** at iteration 58303 (`model_58300`): stopped mounting once fully blind. Stance reward per episode 7.7 (p 0.2) → 1.8 (0.6) → 0.32 (0.9, falls 13.3 per log interval) → 0 (p 1); episodes end at the deadline (length ~147 = 3 s, `mount_deadline` 12.8 per log interval, falls 0.7). 3 s on the floor is worth more than mounting and falling at once. Sweep not run. |

**`bl4-deadline`** (option 2 on top of 1): task `...-Latch45-Hold-Deadline` (`deadline=True`, training only): the episode ends if the robot is not latched on the ball (the robot's criterion: base height > 0.42 m, tilt < 45°) 3 s after its start (`MOUNT_DEADLINE_S`: 1 s hold + 2 s; the policies are on the ball ~0.6 s after starting). Parking on the floor then earns ~3 s of reward instead of ~165 per episode. A termination only, no input change; own variant so `bl3` stays reproducible. Checked: terminates exactly the unlatched envs from step 150. Same start as `bl3` (padded `nr1-up6` critic), blindness curriculum and noise ramp as in `bl3`, 4000 iterations. First 60 iterations: as `bl3` (episode length 740, stance reward 6.9 at p 0.03), deadline ends ~2% of episodes.

**`bl3-blindcurr`:**
- **Blindness curriculum** (`blind_curriculum`, in `-Latch45-Hold` training only): each episode goes blind after the latch with probability p, ramped 0 → 1 over 2000 iterations (`BLIND_RAMP_STEPS`, `t0` 1 021 760); unset (play, eval, all other tasks) p = 1. Checked: blind share and zeroed inputs follow p (0 / 0.5 / 1).
- **Padded critic:** `nr1-up6`'s critic with a zero-weight column for `start_hold` (normalizer mean 0, std 1; Adam moments padded), so it starts exactly as before (checked: values identical, max diff 0.0) and the actor's advantages stay meaningful (`logs/probes/make_padded_init.py`; init dir `2026-11-02_00-00-00_init-padded-from-nr1-up6`). Otherwise as `bl2` (4000 iterations, noise ramp from 0.25×).
- First 90 iterations: episode length 53 → 830, stance reward per episode 8.6 (blind p 0.04): mounting kept.
- Plan: if it parks on the floor again as p rises, add a mount deadline (option 2).
- Mid-ramp (iteration 57203, p 0.63): no parking: stance reward 9.2 (p 0.05) → 4.3 (0.30) → 1.7 (0.63), episode length 853 → 332, `base_too_low` terminations up (0.5 → 4.5 per log interval), timeouts down. It still mounts and falls once blind; the stance reward falls about as fast as the blind share rises, so blind balance is not learned yet. Mount deadline not needed so far.

Eval when finished: the round 5 sweep (task `-Latch45-Hold`), the foot probe and the latch probe.

**Node (open):** `deploy/bitbots/mount_obs.py` and the bitbots_main node: latch tilt from the YAML (`on_ball.max_tilt`, 0.79 for this task; the class default is still 0.35), and the 1 s hold (`start_hold_s`): send the home pose and a last-action input of 0 for 1 s after activation.

**Round 6 conclusion (Nov 3 2026):** blind balance on the ball was never learned, by any policy. `bl2` (blind at once), `bl3` (blindness curriculum) and `bl4` (curriculum + mount deadline) all keep the stance only while the ball input is there; once blind they fall, and then avoid the ball. The earlier "blind on the ball" results never tested it (the latch never fired). Open: is blind balance feasible with this actor input (5 frames = 0.1 s of proprioceptive history)? Next options: drop spawns (start on the ball, blind from step 1) to train and test blind balance without the mount risk; a longer history (input size change: new variant, actor padded); or a deployment-realistic ball input on the ball (the ball filter's estimate frozen at the latch instead of zeros).

## Round 7 (`kb1`): kinematic ball estimate on the ball

From the round 6 conclusion. Every policy depends on the ball input in the stance, and the zeros after the latch carry nothing. Standing on the ball, though, the robot knows where the ball is without the camera: the centre lies below the support sole, along the sole normal, by the ankle's height above the contact (0.045 m, `ON_BALL_Z_CENTER`) plus the ball radius (known: the ball in use). Task `...-Latch45-Hold-KinBall` (`kin_ball=True`), on top of `-Latch45-Hold`:
- **Ball input after the latch** (`kinematic_ball_pos`, in `ball_pos_until_on_ball(kinematic=...)`): that estimate in the base frame instead of zeros, plus a per-episode error ±1.5 cm per axis (`KIN_BALL_BIAS`: encoder bias 0.03 rad over the ~0.5 m leg) and the term's ±2 cm per-step noise (pessimistic for forward kinematics). Same input size and meaning as before the latch, so the `nr1-up6` actor warm-starts on it unchanged; a new variant because an actor input's content changes (AGENTS.md).
- **Support foot:** the higher ankle (gravity frame). Probe on `nr1-up6`'s stance steps: right 99.4%; nearest under the base 96%, under the CoM 81%; the lower ankle (first try) is the free foot, error 25 cm.
- No blindness curriculum: the input never goes away.
- ponytail: the sim uses the true ankle pose, not forward kinematics of the biased encoders; the encoder error is the per-episode bias above.

Probe `logs/probes/kin_ball_probe.py` (`nr1-up6` `model_55947` zero-shot, 20 ms, noise on, 1024 envs): estimate error in stance steps (without the bias and noise) median 2.1 cm, p95 4.3 cm, mean (+0.7, −0.1, +0.2) cm. Alive at 4 s 13% (with zeros: 0-4%), 27k stance steps (zeros: ~1k).

**Node:** `deploy/bitbots/mount_obs.py` computes the same estimate (`kinematic_ball`) from the ankle roll links' poses in the IMU frame (forward kinematics / tf) when `kin_ball_radius` is set; `scripts/export_mount.py` writes `on_ball.kin_ball_radius` (size 5 by default: set to the ball in use; null for the zero-input tasks) and `ankle_above_contact`. Sim2sim (`scripts/sim2sim_onnx.py`, export of `nr1-up6` on the task): ball input robot-side against training max diff 6e-7. The other inputs still differ (max 3.9), because the node does not do the 1 s start hold yet (open, round 6).

Start: the padded `nr1-up6` init of `bl3` (copied to `piplus_ball_mountmin_blindonball_latency_big_onefoot_level_noiseramp_latch45_hold_kinball/2026-11-03_00-00-00_init-padded-from-nr1-up6`), 4000 iterations, 2048 envs, seed 42, the round 5 overrides (`upright` 6), noise ramp from 0.25× (`t0` 1 021 760), as `bl3`.

| Run | wandb | Outcome |
|-----|-------|---------|
| `kb1` | `9gf1yy24` | Finished Nov 3 2026 (`model_59946`). Episode length 62 → ~500 by iteration ~57000, then flat (500-600) to the end; stance reward per episode 0.05 → 3.3-3.6 (`nr1-up6` with the ball visible: 7.2-7.7); falls 11 → 2.5 per log interval. |

**Eval `kb1` `model_59946`** (task `-KinBall`, noise on, bias ±0.03, 2048 envs, seed 0; `logs/probes/sweep_kb1.txt`). Success (old test / strict / level) by size 3 / 4-5, then overall level success and survival.

| Condition | size 3 | size 4-5 | overall level, survive |
|---|---|---|---|
| latency 10 ms | 6 / 5 / 5% | 16 / 13 / 12% | **8%**, 93% |
| latency 20 ms | 9 / 8 / 6% | 12 / 11 / 10% | **8%**, 93% |
| latency 30 ms | 3 / 3 / 2% | 6 / 5 / 5% | **4%**, 86% |
| latency 40 ms | 0% | 1 / 1 / 0% | **0%**, 59% |
| latency 10-40 ms, jitter | 1% | 5 / 4 / 3% | **2%**, 81% |
| 20 ms sensor delay + 10 ms | 3 / 3 / 2% | 7 / 6 / 5% | **3%**, 86% |
| 20 ms, r 0.10 | | 11 / 9 / 9% | **9%**, 95% |
| 20 ms, r 0.11 | | 16 / 15 / 14% | **14%**, 93% |
| 20 ms, kp/kd ×0.5-0.8 | 0% | 2 / 2 / 1% | **1%**, 55% |
| 20 ms, noise off | 28 / 28 / 27% | 36 / 36 / 36% | **31%**, 96% |

Latch probe (20 ms, noise): latched among alive 100%, alive 97% at 10 s, 93% at 19 s. Kin-ball probe: estimate error in stance median 1.45 cm, p95 3.3 cm; higher-ankle pick right 99.998%. Foot probe: sole 8.7° from horizontal, foot 9.9° from the top, ankle roll +0.32 rad (at the stop 0%), **base tilt 20.7°** (`nr1-up6`: 28.5°). Stance probe (latched, alive, after 5 s, noise on): one level foot on the ball 90% of steps (none 10%), free sole ≥10 cm clear 85% (median 13.4 cm), CoM within 12 cm of the ball centre 99% (median 2.7 cm), full stance test 77%.

**Findings**
- **First policy that balances on the ball without the camera:** 93% survive 20 s at 10-20 ms, all of them latched (blind). `bl2`-`bl4` and `nr1-up6` with zeros: 0% on the ball.
- **The success rate undercounts it:** success needs the stance test in ≥90% of the last 3 s; the policy is in it 77% of the time (single leg, CoM 2.7 cm off the centre), dropping out on the level sole (10%) and the 10 cm clearance (15%). Survival is the measure of the blind balance; the success rate is of a clean, held stance.
- Noise costs most again (20 ms: 31% level success without, 8% with), and 30-40 ms, jitter and soft motors cost survival (59-86%).
- Lean 28.5° → 20.7°, the first real change in the lean: the kinematic input does not reward leaning on the ball.
- Training plateaued by ~57000 of 59946.

**Next (open):** (a) a looser success test to compare with deployment (e.g. on the ball, single leg, ≥70% of the last 3 s), or reward terms for the steps that drop out (level sole, clearance); (b) the node: ankle poses from tf, start hold, latch tilt 0.79, `kin_ball_radius` of the real ball; (c) the real noise levels (rosbag), still the largest cost.

## Round 8 (`fl1`): free leg as a counterweight, ankle balance

From a play session of `kb1`: the free foot seems to keep reaching for the ball and the balance is inconsistent. Probe `logs/probes/free_leg_probe.py` (`kb1`, 20 ms, noise, 1024 envs, latched single-support steps after 5 s):
- the free foot touches the ball in 2% of the steps, but hovers over its edge: free ankle 0.99 r horizontally from the centre (88% within 1.5 r), 3.2 cm out from the support foot (p90 9 cm), sole 13.4 cm above the floor; it moves a lot (calf 2.8, thigh 2.4 rad/s), tucked in;
- the support leg balances with all its joints: share of mean |q̇| hip pitch 22%, thigh 17%, calf 17%, ankle pitch 19%, hip roll 13%, ankle roll 13% (ankle 32%).
- What pins the free leg: `posture`'s hip roll std 0.10 rad, the narrowest of all joints.

Task `...-KinBall-FreeLeg` (`free_leg=True`, rewards only, so the inputs and network are those of `-KinBall`):
- **`posture` without the free leg** (`posture_free_leg`): in single support (one level sole on the ball, `_support_foot(flat="level")`) the mean runs over the support leg and the arms only; both legs otherwise. Weight 0.3 as before.
- **`free_foot_ball_clear`, +2.0:** in single support, the free ankle origin's distance to the ball surface, 0 at 3 cm and 1 from 10 cm (`FREE_CLEAR_NEAR/FAR`); `kb1` ~2.5 cm.
- **`support_hip_vel`, −0.01:** in single support, Σ q̇² of the support leg's hip roll, hip pitch, thigh and calf (ankle exempt), on top of `joint_vel` (−0.005 over all 20).
- Check `logs/probes/test_free_leg.py` (`kb1`, 256 envs, 8 s): the posture mask removes the leg away from the support foot, hip joints ordered right then left, both terms 0 outside single support; `kb1` collects 0.36 of `free_foot_ball_clear` and hip q̇² 39 (rad/s)² per single-support step.

Resume `kb1` `model_59946` (actor and critic, unchanged inputs; copied to `piplus_ball_mountmin_blindonball_latency_big_onefoot_level_noiseramp_latch45_hold_kinball_freeleg/2026-11-03_00-00-00_init-from-kb1`), 3000 iterations, 2048 envs, seed 42, the round 5 overrides, full noise (no ramp: `t0` 0). Local 4090. Launched Nov 3 2026 in tmux `mjlab-fl1` (`logs/probes/run_fl1.sh`; window `sweep` runs `after_fl1.sh`: the round 5 sweep and the side, latch, kin-ball and free-leg probes into `logs/probes/sweep_fl1.txt`).

| Run | wandb | Outcome |
|-----|-------|---------|
| `fl1` | `fp5ae5ov` | Finished Nov 3 2026 (`model_62945`). Back at `kb1`'s level within 30 iterations, then flat for the whole run: episode length 520-630, stance reward per episode 3.4-4.1 (`kb1` 3.1-3.6), `free_foot_ball_clear` 0.29 → 0.35-0.39, `support_hip_vel` −0.13 to −0.16, falls 1.8-2.5 per log interval. |

**Eval `fl1` `model_62945`** (as `kb1`; `logs/probes/sweep_fl1.txt`). Level success overall and survival, `kb1` in brackets.

| Condition | level success | survive |
|---|---|---|
| latency 10 ms | 3% (8%) | **96%** (93%) |
| latency 20 ms | 2% (8%) | **97%** (93%) |
| latency 30 ms | 0% (4%) | **92%** (86%) |
| latency 40 ms | 0% (0%) | **65%** (59%) |
| latency 10-40 ms, jitter | 1% (2%) | **88%** (81%) |
| 20 ms sensor delay + 10 ms | 0% (3%) | **91%** (86%) |
| 20 ms, r 0.10 / 0.11 | 1 / 2% (9 / 14%) | 97 / 97% (95 / 93%) |
| 20 ms, kp/kd ×0.5-0.8 | 0% (1%) | **61%** (55%) |
| 20 ms, noise off | 8% (31%) | 98% (96%) |

Probes (20 ms, noise; `kb1` in brackets):
- Latch: alive at 19 s 97% (93%), all latched. Kin-ball estimate error median 1.4 cm (1.45).
- Foot: base tilt **14.5°** (20.7°), sole 6.8° from horizontal (8.7°), foot 8.1° from the top (9.9°), ankle roll +0.37 rad (+0.32), at the stop 0%.
- Free leg (single-support latched steps): touches the ball 1% (2%); free ankle 1.10 r from the ball centre horizontally (0.99 r), within 1.5 r 77% (88%); out from the support foot median 2.4 cm (3.2), p90 15.9 cm (9.0); sole 12.8 cm above the floor (13.4).
- Support leg share of mean |q̇|: ankle pitch 22% + ankle roll 13% = 35% (32%); hip pitch 20% (22%), thigh 13% (17%).
- Stance (`logs/probes/stance_why_probe.py`, latched, alive, after 5 s): one level foot on the ball **96%** (90%), free sole ≥10 cm **74%** (85%), CoM within 12 cm 99% (median 3.9 cm, 2.7), full stance test 71% (77%).

**Findings**
- **Steadier on the ball, but the held-stance success fell:** survival up 3-7 points in every condition, the lean 20.7° → 14.5°, the support sole more level and nearer the top (level foot 90 → 96%). Success 8 → 2% at 20 ms because the free foot dips below the 10 cm clearance more often (85 → 74% of steps): out of `posture`, the free leg swings lower, and `free_foot_lift` stops paying at 10 cm, so there is no margin above the test.
- **The free leg moved out only a little:** median 2.4 cm out, though the p90 grew 9 → 16 cm; it swings out at times rather than holding out. `free_foot_ball_clear` paid 0.29 → 0.39 per episode, weak against the stance terms.
- **Ankle share up only 32 → 35%:** `support_hip_vel` at −0.01 costs ~0.15 per episode, too small to change the strategy.
- Training flat from the first ~100 iterations: the warm start went straight to a nearby optimum.

**Next (open):** (a) a clearance margin: `free_foot_lift` up to 15 cm (and/or the test at 10 cm kept), so the free foot stays clear of the 10 cm line; (b) stronger `free_foot_ball_clear` / `support_hip_vel` if the posture change is wanted; (c) whether the 90%-of-3 s test is the right deployment criterion: survival on the ball (97%) is what the robot needs, and the test counts every dip of the free foot below 10 cm as a failure.
