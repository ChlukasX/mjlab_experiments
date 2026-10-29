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
| `lat2-big` | `52vo6epy` | Finished Oct 29 2026 (`model_46987`, 1.7 h). Reward 130-160, episode length 470-580 (`lat1` on all sizes: 80-110, 360-430); level from ~45500 on. 16 iterations with a negative mean reward (down to −4900, from ~45500 on), each recovered within a few iterations: single envs with a huge penalty, not a collapse. |

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

**Fix (open):** test the free foot by position or contact, not by flat sole: free = not on the ball by position (`_foot_on_ball`) and no contact with the ball over the 4 physics steps; apply the same test in the eval. Contact sensor with `history_length=4` for the floor termination.
