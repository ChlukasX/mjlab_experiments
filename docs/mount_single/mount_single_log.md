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
