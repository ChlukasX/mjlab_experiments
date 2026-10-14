# Ball mount, pruned rewards: run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design and the review of earlier results: [ball_mount_rewards.md](ball_mount_rewards.md). Shared conventions: [../README.md](../README.md).

## Stage 1 (`mm1-*`): free sole 3 cm clear

From scratch, mounts only. Launched Oct 10 2026 on the working tree of `feat/ball-mount-rewards` (uncommitted, on top of `c3f37ba`).

| Run | Task | Where | wandb | Envs, it | Override | Question | Outcome |
|-----|------|-------|-------|----------|----------|----------|---------|
| `mm1-min` | `MountMin` | local | `q6ka0v80` | 4096, 8000 | none | Does the pruned set with the sole test mount? | Reward 67, episode 597/1000, `single_leg_stance` 2.31 of 8, `action_rate` −3.76. Eval: survive 96%, stance held 68% of the last 3 s. |
| `mm1-nocom` | `MountMin` | local | `91dhqkw7` | 4096, 8000 | `--env.rewards.com_over_ball.weight 0.0` | Is the dense centre-of-mass term needed? | Reward 77, episode 698, `single_leg_stance` 2.89, `action_rate` −3.87. Eval: survive 96%, stance held 71%. Same as `mm1-min`: the dense centre-of-mass term is not needed. |
| `mm1-min-clip` | `MountMin-Clip` | cl06 | `ob1tqroz` | 2048, 6000 | none | Same rewards with bounded actions and low entropy. | **Best.** Reward 243, episode 922, `single_leg_stance` 4.16, `action_rate` −1.61. Eval: survive 99%, stance held 73%. |
| `mm1-locked` | `MountMin-Locked` | cl06 | `ipzsk0rj` | 4096, 8000 | none | Does the ball leg take the load when the ball cannot roll? | Reward 93, episode 701, `single_leg_stance` 3.23 on the locked ball (like the free-ball runs). Eval on a free ball: survive 8% (never trained on it). |

All four finished. Eval: `scripts/eval_mountbalance.py --free-foot-clearance 0.03 --probe`, 1024 envs, 20 s, free ball; probe values are means over the last 3 s of the envs that survived.

| Run | Success (stance >= 90% of the last 3 s) | Stance held, mean (3 cm) | Envs holding > 50% | Stance held at 10 cm | First stance | CoM to ball | Base above top | Free sole above floor | Free sole on floor |
|-----|------|------|------|------|------|------|------|------|------|
| `mm1-min` | 0% | 68% | 100% | 5% | 0.4 to 0.5 s | 0.057 m | 0.223 m | 0.048 m | 22% |
| `mm1-nocom` | 0% | 71% | 100% | | 0.4 to 0.5 s | 0.056 m | 0.213 m | 0.054 m | 20% |
| `mm1-min-clip` | 0% | 73% (61% at 6 cm) | 100% | 9% | 0.5 s | 0.045 m | 0.221 m | 0.057 m | 21% |
| `mm1-locked` (8% survive) | 0% | | | | 0.5 s | 0.056 m | 0.210 m | 0.041 m | 24% |

**Findings**
- **First real single-leg support on the ball from a standing start.** In all four runs the robot is on the ball 0.5 s after the start with the other sole off the floor and the centre of mass 4.5 to 6 cm from the ball centre (straddle of the earlier rounds: 8.7 cm, sole on the floor 69% of the time). Every env holds the stance more than half of the last 3 s.
- **It is not a steady stance yet:** the free foot comes back to the floor about 20% of the time (a tap), so no env holds the stance for 90% of the last 3 s and the strict success is 0%. The posture is still a crouch (base 0.21 to 0.22 m above the ball top; target 0.30, the drop policy stands at 0.32).
- **Measuring the lift at the sole was the change that mattered.** `mm1-nocom` (no dense centre-of-mass term) equals `mm1-min`, and the locked ball trains like the free one.
- **Bounded actions with entropy 0.003 are the better setting here** (survive 99%, smoother, more stance), unlike in the drop-balance runs.
- The ball leg now carries the load and the knee is not at its limit (`mm1-min-clip`: right knee 5% of steps at the force limit, right hip pitch 88%), so the torque concern (cause 5 in the design doc) does not block the mount in simulation. The saturated hip pitch is unexplained.

## Stage 2 (`mm2-*`): higher clearance

Resume `mm1-min-clip` `model_5999`, 3000 more iterations, 2048 envs, task `MountMin-Clip`. Launched Oct 10 2026 on the same working tree. The clearance is a command-line override: `--env.rewards.single-leg-stance.params.sole-clearance <m>`.

| Run | Where | wandb | Stance clearance | Other | Question | Outcome |
|-----|-------|-------|------------------|-------|----------|---------|
| `mm2-c06` | local | `7hlwrywh` | 0.06 | | Does the next stage stop the tapping and raise the foot? | Reward 223, episode 883, `single_leg_stance` 4.17, `stand_tall` 1.00, `free_foot_lift` 1.55. |
| `mm2-c10` | local | `71i2j9c4` | 0.10 | | Is the intermediate stage needed? | Reward 184, episode 824, `single_leg_stance` 2.91. No better than the 6 cm stage at any clearance. |
| `mm2-c06-nocom` | cl06 | `2asbc6ji` | 0.06 | `com_over_ball` 0 | Can the centre-of-mass term be dropped here too? | **Best.** Reward 206, episode 903, `single_leg_stance` 4.25. |
| `mm2-c03` | cl06 | `cajpdzys` | 0.03 | | Control: more training at the stage-1 clearance. | Reward 206, episode 814, `single_leg_stance` 4.33. More training alone also helps at 3 cm, not at 10 cm. |

All four finished (`model_8998`). Eval as in stage 1; "held" = mean share of the last 3 s in the stance, in brackets the strict success (>= 90% held).

| Run | Held at 3 cm | Held at 6 cm | Held at 10 cm | Survive | CoM to ball | Base above top | Free sole above floor | Free sole on floor |
|-----|------|------|------|------|------|------|------|------|
| `mm1-min-clip` (start) | 73% (0%) | 61% | 9% (0%) | 99% | 0.045 m | 0.221 m | 0.057 m | 21% |
| `mm2-c06` | 81% (8%) | 77% (2%) | 59% (1%) | 98% | 0.029 m | 0.252 m | 0.083 m | 14% |
| `mm2-c10` | 76% (0%) | 72% (0%) | 63% (0%) | 99% | 0.033 m | 0.258 m | 0.082 m | 20% |
| `mm2-c06-nocom` | **84% (17%)** | **80% (7%)** | 59% (3%) | 99% | 0.029 m | 0.257 m | 0.088 m | 12% |
| `mm2-c03` (control) | 83% (7%) | 76% (2%) | 36% (0%) | 99% | 0.031 m | 0.241 m | 0.077 m | 12% |

Strict success by ball size for `mm2-c06-nocom` at 3 cm: 2% (size 1-2), 18% (size 3), 31% (size 4-5). Small balls are the hard ones.

**Findings**
- **Every measure moved toward the real stance** (drop policy: CoM 0.011 m, base 0.32 m, sole 0.17 m): CoM 0.045 -> 0.029 m, base 0.22 -> 0.26 m, free sole 5.7 -> 8.8 cm, floor contact 21% -> 12%.
- **The remaining failure is the tap:** the free foot still touches the floor 12 to 20% of the time, so few envs hold 90% of the last 3 s. Below the 50% target at every clearance.
- **Raising the clearance in one step (10 cm) is no better than 6 cm** and taps more (20%). The stage matters less than expected: the control at 3 cm improved almost as much, except at 10 cm.
- **`com_over_ball` can go** (the run without it is the best one): 12 terms left.

### Hand-off test (Oct 10 2026)

`scripts/eval_handoff.py`, 1024 envs, 20 s: mount policy `mm2-c06-nocom` `model_8998`, then the drop-balance policy `mb3-mix0` `model_7999`. New option `--switch-clearance`: switch on a real stance (free sole that far off the floor) instead of on the straddle test.

| Switch condition | Switched | Success | Survived |
|---|---|---|---|
| foot on ball, CoM within 0.12 m, for 1.0 s | 100% (1.5 s) | 0% | 0% |
| same, for 0.3 s | 100% (0.7 s) | 0% | 0% |
| stance with the free sole 6 cm clear, for 0.2 s | 100% (0.7 s) | 0% | 0% |
| stance with the free sole 10 cm clear, for 0.1 s | 98% (1.2 s) | 0% | 2% |

**The existing balance policy cannot take over even from a real single-leg stance.** It was trained only from the drop spawn (standing tall, at rest) and runs unbounded bang-bang targets; the mount policy's stance (base 6 cm lower, moving) is outside what it has seen. So either the mount policy holds the stance itself, or a balance policy is trained from the mount policy's end states.

## Stage 3 (`mm3-*`): 10 cm

Resume `mm2-c06-nocom` `model_8998`, 3000 more iterations, 2048 envs, stance clearance 0.10, `com_over_ball` 0. Launched Oct 10 2026.

| Run | Where | wandb | Other | Question | Outcome |
|-----|-------|-------|-------|----------|---------|
| `mm3-c10` | local | `anfpadk1` | | Does the last stage reach the 10 cm stance and stop the tap? | Reward 135, episode 722, `single_leg_stance` 2.95 of 8, `stand_tall` 0.92, `action_rate` −2.71. |
| `mm3-c10-s16` | cl06 | `xbp3hhtx` | `single_leg_stance` weight 16 | Does a larger stance reward make the tap cost more? | Reward 190, episode 708, `single_leg_stance` 6.09 of 16, `stand_tall` 0.88, `action_rate` −3.35 (jerkier). |

Both finished (`model_11997`). Eval as before; in brackets the strict success.

| Run | Held at 3 cm | Held at 6 cm | Held at 10 cm | Survive | CoM to ball | Base above top | Free sole above floor | Free sole on floor |
|-----|------|------|------|------|------|------|------|------|
| `mm2-c06-nocom` (start) | 84% (17%) | 80% (7%) | 59% (3%) | 99% | 0.029 m | 0.257 m | 0.088 m | 12% |
| `mm3-c10` | 83% (14%) | 79% (3%) | 71% (3%) | 98% | 0.031 m | 0.267 m | 0.088 m | 13% |
| `mm3-c10-s16` | 84% (18%) | 81% (7%) | **74% (5%)** | 97% | 0.030 m | 0.269 m | 0.092 m | 12% |

Strict success at 10 cm by ball size (`mm3-c10-s16`): 0% (size 1-2), 2% (size 3), 13% (size 4-5).

**Findings**
- **The 10 cm stage raised the time in the 10 cm stance from 59% to 71 to 74%,** with 92 to 95% of the envs above half. The 3 and 6 cm numbers did not move: the policy has plateaued there.
- **The tap is unchanged** (free sole on the floor 12 to 13% of the time), also with the stance weight doubled, which only made the actions jerkier. Strict success stays at 3 to 5% at 10 cm, far below the 50% target, and is close to 0 on the small balls.

## Status (Oct 10 2026)

The mount now exists: from standing, the robot is on the ball in 0.5 s with the other foot really off the floor, survives 97 to 99% of 20 s episodes and is in the 10 cm stance about three quarters of the time. It is not a held stance: about every eighth step the free foot is back on the floor. The existing balance policy cannot take over from it. Open, in order of expected value:
1. Train the balance policy from the mount policy's end states (spawn from `mm3` rollouts instead of the drop), then repeat the hand-off test.
2. Find out what the tap is for (a periodic balance correction, or one foot placement per ball roll) before adding a reward against it; it survived a doubled stance reward.
3. Small balls (radius < 0.083 m) separately: they have almost no strict success at any stage.

## Stage 4 (`mm4-*`): new robot and contact model, smoothing

From Oct 13 2026 the randomized-ball tasks have the head collision box fitted to the head mesh and a randomized contact time constant (0.01 to 0.03 s) on the ball and the feet ([collision_check](../collision_check/collision_check.md)), and an `action_excess` term (weight 0 by default; rad of commanded joint target outside the joint ranges, [balance_smooth](../balance_smooth/balance_smooth.md)). Audit of `mm3-c10-s16` before: mean |action| 7.9, 57% of the targets outside the joint limits, action change 0.98 per step.

Resume `mm3-c10-s16` `model_11997`, 3000 iterations, 2048 envs, cl06, same weights (clearance 0.10, `single_leg_stance` 16, `com_over_ball` 0). Launched Oct 13 2026 on the working tree of `feat/balance-smooth` (on top of `dbd8667`).

| Run | wandb | `action_excess` | Question | Outcome |
|-----|-------|-----------------|----------|---------|
| `mm4-dr` | `eahx3412` | 0 | Does the mount adapt to the new head box and contact range? | Reward 171, episode 693, `single_leg_stance` 7.04 of 16, `action_rate` −4.28. Eval: **62% success at 10 cm**. |
| `mm4-dr-exc` | `3ufjn5na` | −0.2 | Can the mount policy's commands be brought inside the joint limits at the same time? | Reward 178, episode 705, `single_leg_stance` 8.15, `action_rate` −2.44. Eval: **56% success**, actions half the size. |

Both finished (`model_14996`). Eval on the new model: `scripts/eval_mountbalance.py --probe` (1024 envs, 20 s, success = stance with the free sole 10 cm clear for 90% of the last 3 s) and `scripts/audit_policy.py`.

| Run | Success at 10 cm | Held at 10 cm / 6 cm | Survive | CoM to ball | Base above top | Free sole above floor | Free sole on floor | mean \|action\| | Targets outside limits | Action change per step | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|------|------|------|
| `mm3-c10-s16` (start, old model) | 5% | 74% / 81% | 97% | 0.030 m | 0.269 m | 0.092 m | 12% | 7.9 | 57% | 0.98 | 18% |
| `mm4-dr` | **62%** | 89% / 93% | 96% | 0.018 m | 0.285 m | 0.103 m | 3% | 8.3 | 60% | 1.13 | 17% |
| `mm4-dr-exc` | 56% | 88% / 92% | 96% | 0.019 m | 0.285 m | 0.106 m | 3% | **3.9** | 47% | **0.67** | 16% |

**Findings**
- **The mount policy now holds the stance by itself:** 56 to 62% strict success at 10 cm (target 50%, before 5%). The tap is nearly gone (free sole on the floor 3% of the time, was 12%), the centre of mass is 1.8 cm from the ball centre and the base 1.6 cm higher. What changed is 3000 more iterations on the model with the fitted head box and the randomized contact stiffness; which of the three did it is not separated.
- **`action_excess` halves the mount policy's actions at a small cost** (56% against 62%). Still far outside the audit limits.

Next: `mm5-exc-ar2` (wandb `baiq45bn`, cl06): resume `mm4-dr-exc`, 3000 iterations, `action_excess` −0.2, `action_rate` −0.02.

`mm5-exc-ar2` finished (`model_17995`): reward 199, episode 715, `single_leg_stance` 9.14 of 16.

| Run | Success at 10 cm (by ball size 1-2 / 3 / 4-5) | Held at 10 cm | Survive | CoM to ball | Base above top | Free sole on floor | mean \|action\| | Targets outside limits | Action change per step | Leg torque saturated |
|-----|------|------|------|------|------|------|------|------|------|------|
| `mm4-dr-exc` (start) | 56% | 88% | 96% | 0.019 m | 0.285 m | 3% | 3.9 | 47% | 0.67 | 16% |
| `mm5-exc-ar2` | **65%** (51 / 73 / 71%) | 91% | 97% | 0.016 m | 0.289 m | 3% | 3.2 | 41% | 0.52 | 16% |

Smoother and better at the same time: a doubled `action_rate` on top of `action_excess` did not hurt the mount. Still far from the audit limits.

Next (`mm6-*`, cl06, resume `mm5-exc-ar2`, 3000 iterations): `mm6-exc50-ar2` (wandb `ebo2pjrr`: `action_excess` −1.0, `action_rate` −0.02) and `mm6-exc25-ar3` (`f91wfcza`: −0.5, −0.03).

`mm6-*` finished (`model_20994`).

| Run | Success at 10 cm (by ball size 1-2 / 3 / 4-5) | Held at 10 cm | Survive | CoM to ball | Base above top | Free sole on floor | mean \|action\| | Targets outside limits | Action change per step | Leg torque saturated | Hand-off with `bs3-exc25-ar2` |
|-----|------|------|------|------|------|------|------|------|------|------|------|
| `mm5-exc-ar2` (start) | 65% (51 / 73 / 71%) | 91% | 97% | 0.016 m | 0.289 m | 3% | 3.2 | 41% | 0.52 | 16% | 95% |
| `mm6-exc50-ar2` | 77% (61 / 82 / 86%) | 94% | 96% | 0.014 m | 0.299 m | 2% | **1.84** | **25%** | **0.41** | 17% | 96% |
| `mm6-exc25-ar3` | **78%** (60 / 87 / 89%) | 94% | 96% | 0.015 m | 0.296 m | 2% | 2.37 | 29% | 0.43 | 17% | 96% |

Again smoother and better together: 77 to 78% of the mounts hold the 10 cm stance without any balance policy, standing at the target height (base 0.30 m above the ball top). A quarter of the commanded targets are still outside the joint limits and the leg torque saturation has not moved (17%).

Next (`mm7-*`, cl06, resume `mm6-exc50-ar2`, 3000 iterations): `mm7-exc200-ar2` (wandb `579sgsy6`: `action_excess` −4.0, `action_rate` −0.02) and `mm7-exc100-ar3` (`5h0ai077`: −2.0, −0.03).

`mm7-*` finished (`model_23993`). Hand-off with balance policy `bs4-exc100-ar2`.

| Run | Success at 10 cm (by ball size 1-2 / 3 / 4-5) | Held at 10 cm | Survive | CoM to ball | Base above top | Free sole on floor | mean \|action\| (limit 1.0) | Targets outside limits (limit 2%) | Action change per step (limit 0.30) | Leg torque saturated | Hand-off |
|-----|------|------|------|------|------|------|------|------|------|------|------|
| `mm6-exc50-ar2` (start) | 77% (61 / 82 / 86%) | 94% | 96% | 0.014 m | 0.299 m | 2% | 1.84 | 25% | 0.41 | 17% | 94% |
| `mm7-exc200-ar2` | **87%** (74 / 94 / 94%) | 94% | 99% | 0.014 m | 0.312 m | 2% | **1.24** | **3.8%** | **0.26** | 17% | 91% (83 / 96 / 94%) |
| `mm7-exc100-ar3` | 83% (71 / 89 / 89%) | 94% | 99% | 0.014 m | 0.304 m | 2% | 1.54 | 14.5% | 0.28 | 17% | 94% (88 / 97 / 97%) |

**Findings**
- **The mount policy alone now reaches 87%** (stance with the free sole 10 cm clear held for 90% of the last 3 s, from the standing start), survives 99% of the 20 s episodes and is close to the audit limits (targets outside the joint limits 3.8%, action change 0.26). Over four resumes (`mm4` to `mm7`) on the corrected model every step of `action_excess` and `action_rate` made it both smoother and more successful: 5% -> 62 / 56% -> 65% -> 77% -> 87%.
- **The balance policy adds little now:** 91% with the hand-off against 87% without (94% against 83% for `mm7-exc100-ar3`), and its hand-off states were recorded from the mount policy four stages ago. The difference left is on the small balls (83 to 88% against 71 to 74%).
- Leg torque saturation is unchanged at 17% through all stages: none of the terms acts on it.

## Status (Oct 14 2026)

`piplus_ball_mountmin_clip/*_mm7-exc200-ar2/model_23993.pt` is one policy that mounts the ball and holds the stance in 87% of the episodes. Open: the small balls (74%), the torque saturation, and whether a separate balance policy is still worth keeping (if so: record the hand-off states again from this policy and resume `bs4-exc100-ar2`).
