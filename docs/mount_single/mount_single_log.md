# One policy for mount and stance: run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design and findings: [mount_single.md](mount_single.md). Shared conventions: [../README.md](../README.md).

## Round 1 (`ms1-*`): spin and torque saturation

Resume `mm7-exc200-ar2` `model_23993`, 3000 iterations, 2048 envs, task `MountMin-Clip`, cl06, with its weights (`single_leg_stance` 16 at 10 cm clearance, `com_over_ball` 0, `action_excess` −4.0, `action_rate` −0.02). Launched Oct 14 2026 on the working tree of `feat/mount-single` (on top of `746982e`).

| Run | wandb | `base_yaw_rate` | `torque_saturation` | Question | Outcome |
|-----|-------|-----------------|---------------------|----------|---------|
| `ms1-yaw` | `8gx3yvla` | −0.05 | 0 | Can the policy unlearn the spin (−2 per step at 6 rad/s) and keep the stance? | Reward 183, episode 612. 86%, spin 1.6 rad/s. |
| `ms1-yaw-lo` | `0fb6izf8` | −0.01 | 0 | Same with a fifth of the weight, in case the first collapses. | Reward 263, episode 798. 86%, spin 5.0 rad/s: too weak. |
| `ms1-sat` | `1l6w28w3` | 0 | −2.0 | Does the saturation share come down without losing the stance? | Reward 286, episode 793. **94%**, spin 5.1 rad/s; saturation share barely moved. |
| `ms1-yaw-sat` | `vg8jziyh` | −0.05 | −2.0 | Both together. | **Best.** Reward 211, episode 698. 93%, spin 1.9 rad/s, 80% under the changed friction model. |

All four finished (`model_26992`). Eval: success = stance with the free sole 10 cm clear for 90% of the last 3 s (1024 envs, 20 s); "friction changed" = same with an elliptic cone and torsional friction 0.02 at the feet; audit = `scripts/audit_policy.py`.

| Run | Success (by ball size 1-2 / 3 / 4-5) | Survive | Spin (base / ball) | Success, friction changed (spin) | mean \|action\| (limit 1.0) | Targets outside limits (2%) | Action change per step (0.30) | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|
| `mm7-exc200-ar2` (start) | 87% (74 / 94 / 94%) | 99% | +6.3 / +5.0 rad/s | 48% (+5.8) | 1.24 | 3.8% | 0.26 | 17% |
| `ms1-yaw` | 86% (66 / 94 / 97%) | 97% | +1.6 / +1.0 | 73% (+0.9) | 1.13 | 5.3% | 0.41 | 18% |
| `ms1-yaw-lo` | 86% (67 / 92 / 98%) | 99% | +5.0 / +3.7 | 56% (+4.2) | 1.27 | 3.5% | 0.28 | 18% |
| `ms1-sat` | **94%** (85 / 98 / 99%) | 100% | +5.1 / +3.5 | 61% (+4.5) | 1.19 | **2.0%** | **0.24** | 16% |
| `ms1-yaw-sat` | 93% (86 / 97 / 96%) | 98% | **+1.9 / +0.9** | **80%** (+1.0) | 1.12 | 3.5% | 0.36 | 16% |

**Findings**
- **The yaw penalty at −0.05 cuts the spin from 6.3 to 1.6 to 1.9 rad/s without losing the stance,** and makes the policy much less sensitive to the friction model (48% -> 73 to 80% with the elliptic cone and torsional friction). −0.01 is too weak. The spin keeps its direction.
- **`torque_saturation` raised the success** (87% -> 93 to 94%, small balls 74% -> 85%) **but not the saturation share** (17% -> 16%). It sees the applied force, which stops at the limit, so it gives no gradient for how far past the limit the policy asks. New term `torque_demand` (the demanded PD torque past the limit, with each env's gains) for round 2.
- The yaw penalty raises the action change per step (0.24 to 0.28 -> 0.36 to 0.41).

## Round 2 (`ms2-*`)

Resume `ms1-yaw-sat` `model_26992`, 3000 iterations, same base weights plus `torque_saturation` −2.0. Launched Oct 14 2026 on the working tree of `feat/mount-single`.

| Run | wandb | `base_yaw_rate` | `action_rate` | `torque_demand` | Question | Outcome |
|-----|-------|-----------------|---------------|-----------------|----------|---------|
| `ms2-yaw10` | `jnxoh5s8` | −0.1 | −0.02 | 0 | Does a doubled yaw penalty stop the remaining spin? | Reward 197, episode 663. 94%, spin +0.6 rad/s. |
| `ms2-yaw10-ar3` | `xdu93rlo` | −0.1 | −0.03 | 0 | Does a larger rate penalty bring the action change back under 0.30? | Reward 175, episode 579. **97%**, spin +0.9 rad/s, action change still 0.37. |
| `ms2-yaw10-demand` | `thy5bg7p` | −0.1 | −0.02 | −0.5 | Does penalising the demanded torque bring the saturation share down? | **Best overall.** Reward 209, episode 639. 95%, no systematic spin, leg saturation 9%, within or at the audit limits. |

All three finished (`model_29991`). Same evaluation as round 1.

| Run | Success (by ball size 1-2 / 3 / 4-5) | Survive | Spin, base mean (median \|.\|, share clockwise) | Success, friction changed | mean \|action\| (limit 1.0) | Targets outside limits (2%) | Action change per step (0.30) | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|
| `ms1-yaw-sat` (start) | 93% (86 / 97 / 96%) | 98% | +1.9 rad/s (1.7, 0%) | 80% | 1.12 | 3.5% | 0.36 | 16% |
| `ms2-yaw10` | 94% (87 / 99 / 95%) | 98% | +0.6 (0.55, 18%) | 80% | 1.17 | 4.8% | 0.36 | 15% |
| `ms2-yaw10-ar3` | **97%** (92 / 99 / 99%) | 99% | +0.9 (0.90, 2%) | **90%** | 1.09 | 4.0% | 0.37 | 15% |
| `ms2-yaw10-demand` | 95% (90 / 99 / 97%) | 98% | **+0.3 (0.46, 35%)** | **90%** | **1.01** | **2.8%** | **0.27** | **9%** |

**Findings**
- **`torque_demand` does what `torque_saturation` could not:** the leg actuators are at their limit 9% of the time (16% before), and with it the actions got smaller and calmer (mean |action| 1.01, action change 0.27, targets outside the limits 2.8%). It also has the smallest spin, and the spin no longer has one direction (35% of the envs turn clockwise): a residual wobble, not a learned rotation.
- **The doubled yaw penalty removes the systematic spin** in all three runs (0.3 to 0.9 rad/s mean, from 6.3 two rounds ago) and the success went up, not down (94 to 97%). Under the changed friction model 80 to 90% (48% before the yaw penalty).
- A larger `action_rate` alone does not lower the action change (0.37); the demanded-torque penalty does.

## Status (Oct 14 2026)

Best single policy: `piplus_ball_mountmin_clip/*_ms2-yaw10-demand/model_29991.pt`. From standing beside a randomized ball (size 1 to 5): mounts in 0.5 s and holds the single-leg stance with the free sole 10 cm clear in 95% of the episodes (90% on the smallest balls), no systematic spin, 90% under a different friction model, and at or within the proposed audit limits on two of three measures (mean |action| 1.01 against 1.0, targets outside the limits 2.8% against 2%), leg torque at the limit 9% of the time.

Weights of this policy (all on `Mjlab-Piplus-Ball-MountMin-Clip`): `single_leg_stance` 16 at 10 cm sole clearance, `com_over_ball` 0, `action_excess` −4.0, `action_rate` −0.02, `torque_saturation` −2.0, `torque_demand` −0.5, `base_yaw_rate` −0.1.

## Round 3 (`ms3-*`): hollow-shell ball

The ball's inertia is now that of a thin hollow shell, 2/3 m r^2 (`ball_mass_from_size`, `inertia_factor`), instead of a solid sphere's 0.4 m r^2: a football resists rolling and spinning about 1.7 times more than the simulated ball did. `ms2-yaw10-demand` evaluated on it without retraining: 94% (88 / 98 / 97%), spin +0.3 rad/s, against 95% on the solid ball.

Resume `ms2-yaw10-demand` `model_29991`, 3000 iterations, 2048 envs, all its weights (`single_leg_stance` 16 at 10 cm, `com_over_ball` 0, `action_excess` −4.0, `action_rate` −0.02, `torque_saturation` −2.0, `torque_demand` −0.5, `base_yaw_rate` −0.1), cl06. Launched Oct 14 2026 on the working tree of `feat/mount-single` (on top of `951d5ca`).

| Run | wandb | Seed | Outcome |
|-----|-------|------|---------|
| `ms3-shell` | `59vdjpnr` | 42 | **Best.** Reward 252, episode 715. 98%, leg saturation 5%. |
| `ms3-shell-s2` | `0k2528u2` | 2 | Reward 240, episode 690. 98%; the only one within the outside-limits threshold (1.6%). |

Both finished (`model_32990`). Same evaluation as before, on the hollow-shell ball.

| Run | Success (by ball size 1-2 / 3 / 4-5) | Survive | Spin, base mean (median \|.\|, share clockwise) | Success, friction changed | mean \|action\| (limit 1.0) | Targets outside limits (2%) | Action change per step (0.30) | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|
| `ms2-yaw10-demand` (start, zero-shot on the shell ball) | 94% (88 / 98 / 97%) | 98% | +0.3 rad/s (0.45, 35%) | | 1.01 | 2.8% | 0.27 | 9% |
| `ms3-shell` | **98%** (93 / 99 / 100%) | 100% | +0.3 (0.46, 29%) | **93%** | **1.04** | 2.2% | 0.24 | **5%** |
| `ms3-shell-s2` | **98%** (94 / 100 / 99%) | 99% | +0.6 (0.53, 8%) | 92% | 1.16 | **1.6%** | **0.23** | **5%** |

**Findings**
- **98% success with both seeds,** 93 to 94% on the smallest balls, 92 to 93% under the changed friction model. The two seeds agree within a point on success and on every audit measure except mean |action|.
- **The resume on the shell ball did more than adapt:** leg torque at the limit 9% -> 5%, action change 0.27 -> 0.23, targets outside the limits 2.8% -> 1.6 to 2.2%. The audit limits are met or missed by a hair (mean |action| 1.04 against 1.0, outside 2.2% against 2% for seed 42); seed 2 meets the outside-limits threshold.
- Spin stays small (0.3 to 0.6 rad/s); seed 2 keeps a slight preferred direction (8% clockwise).

## Status (Oct 14 2026)

Best single policy: `piplus_ball_mountmin_clip/*_ms3-shell/model_32990.pt` (seed 42), with `ms3-shell-s2` as an equal alternative. Hollow-shell ball, fitted head box, randomized contact stiffness; weights as listed for round 3.

## Round 4 (`bob1-*`): blind on the ball

Task `MountMin-BlindOnBall` (see the design doc). Resume `ms3-shell` `model_32990` (copied to `piplus_ball_mountmin_blindonball/2026-10-14_00-00-00_init-from-ms3-shell`), 4000 iterations, 2048 envs, all `ms3` weights, cl06. Launched Oct 14 2026 on the working tree of `feat/mount-single` (on top of `0e517f7`). Episode length at the start of the resume: 41 to 47 steps (the policy falls once its ball input goes blank, as expected).

| Run | wandb | Seed | Outcome |
|-----|-------|------|---------|
| `bob1` | `ntpnn4k3` | 42 | **Best.** Reward 203, episode 641/1000. 95% with the ball input blank on the ball. |
| `bob1-s2` | `edxjs5mc` | 2 | Reward 177, episode 625. 93%. |

A third run with a larger perception error (±5 cm) did not start: tuple parameters cannot be overridden from the command line. That case is tested at evaluation instead.

Both finished (`model_36989`). After the switch to the blind input the episodes first collapsed (`bob1`: 47 -> 13 steps at about 280 iterations) and then recovered over the 4000 iterations. Eval on `MountMin-BlindOnBall` (the play config keeps the masking and the per-episode ball error; 1024 envs, 20 s); "ball error" = per-episode error of the ball estimate (`--ball-obs-bias`, training used ±3 / ±3 / ±1 cm).

| Run | Success (by ball size 1-2 / 3 / 4-5) | Survive | Spin, base mean (share clockwise) | Ball error ±5 cm | Ball error ±8 cm | Friction changed | mean \|action\| (1.0) | Targets outside limits (2%) | Action change (0.30) | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|------|------|
| `ms3-shell` (start, sees the ball throughout) | 0% on this task | 0% | | | | | | | | |
| `bob1` | **95%** (87 / 99 / 100%) | 98% | +0.16 rad/s (33%) | 90% | 66% | 83% | 1.35 | 2.5% | 0.26 | **4%** |
| `bob1-s2` | 93% (88 / 96 / 97%) | 97% | −0.01 (49%) | 90% | 67% | 89% | 1.31 | 2.1% | 0.28 | 6% |

**Findings**
- **The policy mounts and balances with the ball input blank from the moment it is on the ball:** 93 to 95%, 100% on size 4-5 for `bob1`. The stance is as before (CoM 0.6 cm from the ball centre, base 0.32 m above the top, free sole 13 cm up, never on the floor).
- **Ball-estimate error:** unchanged up to the trained ±3 cm, 90% at ±5 cm, 66% at ±8 cm. The robot's estimate of a stationary ball before the step must be within about 5 cm.
- No systematic spin; leg torque at the limit 4 to 6%; audit as before (action change inside, outside-limits 2.1 to 2.5%, mean |action| 1.3).
- Smallest balls (radius below 0.083 m) are the weakest bin (87 to 88%). The RoboCup ball the Bit-Bots perception is configured for (diameter 0.153 m, radius 0.077 m) falls in that bin.

## Status (Oct 15 2026)

Deployment candidate: `piplus_ball_mountmin_blindonball/*_bob1/model_36989.pt`. Best on a size 4 to 5 ball (radius 0.10 to 0.11 m).

## Round 5 (`bob2-*`): wider upright reward

`upright` (torso orientation) paid 0.01 on the ball: its kernel (std 0.22) is flat at the 28° lean the policy stands with. Resume `bob1` `model_36989`, 3000 iterations, all `bob1` weights, `upright` std 0.22 -> 0.5, cl06 (shared with another user's training, 4.6 s per iteration). Launched Oct 15 2026.

| Run | wandb | `upright` weight | Outcome |
|-----|-------|------------------|---------|
| `bob2-up05` | `je24u82p` | 2 | Reward term 0.47 of 2 at the end (was 0.02). 97%. |
| `bob2-up05-w4` | `tdy6v4a1` | 4 | 0.89 of 4. 96%. |

Eval (Oct 21 2026; same procedure as round 4, `bob1` re-run alongside):

| Run | Success | Survive | Base tilt on the ball | Spin, mean (share clockwise) | Ball error ±5 cm | Friction changed | mean \|action\| | Targets outside limits | Action change | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|------|------|
| `bob1` | 93% | 98% | 28.1° | +0.15 rad/s (36%) | 91% | 83% | 1.36 | 2.5% | 0.26 | 4% |
| `bob2-up05` | **97%** | 99% | 27.7° | +0.43 (15%) | **93%** | **91%** | 1.36 | **1.7%** | 0.26 | 4% |
| `bob2-up05-w4` | 96% | 99% | 27.3° | +0.24 (28%) | 92% | 85% | 1.30 | 2.5% | 0.28 | 5% |

**Findings**
- **The lean did not change** (27 to 28° in all three): the wider kernel now pays and has a gradient, but the policy keeps the posture. The lean seems to be part of how it stands on one leg over the ball, not a missing incentive.
- **`bob2-up05` is better on everything else:** 97% (re-run of `bob1`: 93%; earlier 95%, so about ±2 points of evaluation spread), more robust to the friction model (91% against 83%) and to a ±5 cm ball error (93%), and within the outside-limits threshold (1.7%). Likely the 3000 extra iterations more than the upright term. Its remaining spin has a slight preferred direction (+0.43 rad/s, 15% clockwise).
