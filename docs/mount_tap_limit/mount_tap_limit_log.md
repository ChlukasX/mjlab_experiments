# mount_tap_limit run log

Design: [mount_tap_limit.md](mount_tap_limit.md). Run ID prefix `tl`. Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_taplimit`.

Start (all runs): a copy of `cur4c` `model_24450.pt` (SHA-256 `83ec928c…`, byte-identical to the other init copies) as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_taplimit/2026-11-10_00-00-00_init-from-cur4c-24450`.

## `taplimit-smoke`

Smoke run with the first design (budget 3 from the first step of stage 8, no warm-up), 40 iterations (24450 → 24490), tensorboard (no wandb), run directory `2026-11-10_22-19-50_taplimit-smoke`. The loop runs, no error, no NaN. Last iteration: mean reward −10.8, episode length 140, `tap_budget` 3, `tap_end_share` 0.88, about 85% of the terminations by `tap_limit`, value loss 469. This result led to the warm-up and the budgets 6, 4, 3, 2, 1, 0.

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --env.scene.num-envs 4096 --env.scene.env-spacing 1.0 --agent.logger tensorboard --agent.run-name taplimit-smoke --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `taplimit-smoke-warmup`

Smoke run with the warm-up, 40 iterations (24450 → 24490), tensorboard (no wandb). The loop runs, no error. Last iteration: mean reward 191.7, episode length 737, `tap_budget` −1 (warm-up), `tap_limit` 0, `mount_share` 0.98, `survival` 0.50 (the running mean starts at 0).

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --env.scene.num-envs 4096 --env.scene.env-spacing 1.0 --agent.logger tensorboard --agent.run-name taplimit-smoke-warmup --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `tl1-taplimit`

Warm start from `cur4c` `model_24450`, stage 8, 4000 iterations (24450 → 28450), seed 42, 4096 envs. Recipe of `sg1-leggap`; only the tap termination differs. Wandb `u75t0h0y`, run directory `2026-11-10_22-37-34_tl1-taplimit`, started Nov 10 2026 22:37 in the tmux session `mjlab-train-tl1` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-tap-limit`, uncommitted (the diff is in the run directory, `git/`).

Code of this run: the first version of the curriculum, before the change of `tl2` (reward `tap_end` on `tap_limit` only, budget steps down on a `tap_limit` share below 0.25). The diff is in the run directory (`git/`). The task ID is the same; the code of the task changed after this run, before the first commit.

**Outcome:** finished 4000 iterations (24450 → 28450, last checkpoint `model_28449`, finished Nov 11 2026 03:04). Reward hacking: the policy learned to fall before its taps ran out.

| Iteration | Phase | Mean reward | Episode length | `tap_limit` / iteration | `base_too_low` / iteration | `fell_over` / iteration | Survival | Mount share |
|---|---|---|---|---|---|---|---|---|
| ~25000 | warm-up | 200.6 | 767 | 0 | 1.9 | — | 0.56 | 0.98 |
| 25795 | warm-up | 190.5 | 732 | 0 | 2.5 | — | 0.49 | 0.99 |
| 25963 | budget 6 | 11.4 | 225 | 8.6 | 2.3 | 0.09 | 0.0001 | 0.98 |
| 26248 | budget 6 | 16.7 | 165 | 4.8 | 18.3 | — | 0.0 | 0.94 |
| 28449 | budget 0 | 30.3 | 87 | 0.6 | 5.3 | 43.4 | 0.0 | 1.00 |

The survival never reached 0.7, so the warm-up ended at its limit (iteration 25962). With budget 6, 88% of the episodes ended by `tap_limit`. Within 300 iterations the falls replaced the taps: `tap_limit` had the −50 penalty, a fall had none. The falls also pushed the `tap_limit` share below 0.25, so the budget went down to 0. The last policy falls after 1.7 s. No episode reaches the time-out. The policy is not usable. Fix in `tl2`: the penalty on every early end and the budget step on survival.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --name tl1-taplimit --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 4000 --agent.seed 42
```

## `taplimit-smoke-earlyend`

Smoke run with the `tl2` code (reward `early_end` on every early end, budget step on survival), 40 iterations (24450 → 24490), tensorboard (no wandb). The loop runs, no error. Last iteration: mean reward 171.3, episode length 655, `tap_budget` −1 (warm-up), `early_end` 0 (warm-up), `mount_share` 0.97, `survival` 0.44.

Live check of the active path (no file kept): `sg1-leggap` `model_27449`, 256 envs, stage 8, warm-up skipped by hand (budget 6), 600 steps: 482 early ends (462 `tap_limit`, 21 `base_too_low`, 1 `body_on_floor`, 1 `fell_over`), `early_end` −49.9 per early end.

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --env.scene.num-envs 4096 --env.scene.env-spacing 1.0 --agent.logger tensorboard --agent.run-name taplimit-smoke-earlyend --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `tl2-earlyend`

Same start and recipe as `tl1-taplimit`. Changes: the reward `early_end` (−50 on every termination other than `time_out` while the budget acts) replaces `tap_end`, and a budget level goes down when the running survival is at least 0.3 (`TAP_ADVANCE_SURVIVAL`). Wandb `xya94e32`, run directory `2026-11-11_13-25-18_tl2-earlyend`, started Nov 11 2026 13:25 in the tmux session `mjlab-train-tl2` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-tap-limit`, uncommitted (the diff is in the run directory, `git/`).

**Outcome:** finished 4000 iterations (24450 → 28450, last checkpoint `model_28449`). The warm-up ended at its limit (iteration 25952, survival below 0.7). The budget stayed at 6 to the end: the running survival never reached 0.3 (0.005 at the end). No collapse into falls as in `tl1`, but a slow drift from taps to falls.

| Iteration | Phase | Mean reward | Episode length | `tap_limit` / iteration | `base_too_low` / iteration | `fell_over` / iteration | Survival | Mount share |
|---|---|---|---|---|---|---|---|---|
| 25950 | budget 6 (first iteration) | 7.9 | 223 | 48.2 | 1.0 | 0.05 | 0.0 | 0.98 |
| 26308 | budget 6 | −10.8 | 226 | 12.1 | 5.4 | 0.3 | 0.0 | 0.98 |
| 26938 | budget 6 | −7.7 | 239 | 12.0 | 5.4 | 0.3 | 0.0 | — |
| 27432 | budget 6 | −14.8 | 230 | 10.0 | 5.7 | 0.9 | 0.0 | 0.97 |
| 28449 | budget 6 | — | — | 10.9 | — | 0.6 | 0.005 | 0.97 |

Evaluation of `model_28449` with `scripts/eval_mountbalance.py --obs-noise --gait --probe --audit` (1024 envs, default seed, task `-LegGap`: old noise, no tap termination, 20 s per env). `sg1-leggap` row: same conditions, from the log of [mount_imu_noise](../mount_imu_noise/mount_imu_noise_log.md).

| Policy | Survived | Stance share | Stance held (last 3 s) | Taps per 10 s | Stance segment median | Base tilt | CoM − ball x |
|---|---|---|---|---|---|---|---|
| `sg1-leggap` | 88% | 58% | 59% | 13.5 | 0.38 s | 10.9° | −4.7 cm |
| `tl2-earlyend` | 73% | 47% | 48% | 12.0 | 0.36 s | 11.9° | −5.2 cm |

Success by the strict definition is 0%. `r_hip_pitch` is at the force limit in 58% of the steps (`sg1`: 47-53%). Targets outside the joint limits 7.7%. Reading: the tap limit at budget 6 does not teach the balance. The policy taps 11% less, but it survives 15 points less, and its stance is shorter. Next: `tl3` with the first budget 10.

```
uv run python scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap --checkpoint logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_taplimit/2026-11-11_13-25-18_tl2-earlyend/model_28449.pt --obs-noise --gait --probe --audit --num-envs 1024
```

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --name tl2-earlyend --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 4000 --agent.seed 42
```

## `taplimit-smoke-budget10`

Smoke run with the `tl3` code (first budget 10: `TAP_BUDGETS` 10, 8, 6, 4, 2, 1, 0), 40 iterations (24450 → 24490), tensorboard (no wandb). The loop runs, no error. Last iteration: mean reward 180.2, episode length 706, `tap_budget` −1 (warm-up).

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --env.scene.num-envs 4096 --env.scene.env-spacing 1.0 --agent.logger tensorboard --agent.run-name taplimit-smoke-budget10 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `tl3-budget10`

Same start and recipe as `tl2-earlyend`. Change: the first budget is 10 (`TAP_BUDGETS` 10, 8, 6, 4, 2, 1, 0) instead of 6. Wandb `mzeia4er`, run directory `2026-11-11_18-12-13_tl3-budget10`, started Nov 11 2026 18:12 in the tmux session `mjlab-train-tl3` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-tap-limit`, uncommitted (the diff is in the run directory, `git/`).

**Outcome:** stopped by decision of the user at about iteration 27550 (last checkpoint `model_27550`), not finished. The warm-up ended at its limit (iteration 25950). The budget stayed at 10: no episode reached the time-out after the first budget iteration.

| Iteration | Phase | Mean reward | Episode length | `tap_limit` / iteration | `base_too_low` / iteration | `fell_over` / iteration | Survival | Mount share |
|---|---|---|---|---|---|---|---|---|
| 25950 | budget 10 (first iteration) | 17.4 | 276 | 38.3 | 1.8 | 0.09 | 0.0003 | 0.99 |
| 26282 | budget 10 | −8.7 | 290 | 8.9 | 5.7 | 0.28 | 0 | 0.97 |
| 26614 | budget 10 | −8.2 | 289 | 7.1 | 6.4 | 0.41 | 0 | 0.98 |
| 26946 | budget 10 | −3.9 | 315 | 8.2 | 4.6 | 0.50 | 0 | 0.97 |
| 27115 | budget 10 | −9.9 | 305 | 8.3 | 5.4 | 0.38 | 0 | 0.97 |

Reading: the episodes stayed at about 6 s (`tl2`: 4.6 s). A fixed budget per episode needs a tap rate of budget / 2 per 10 s for a time-out, far below the 13 per 10 s of the policy, so the survival never rose. Next: `tl4` with a tap rate limit. No evaluation (run not finished).


```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --name tl3-budget10 --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 4000 --agent.seed 42
```

## `taplimit-smoke-rate`

Smoke run with the `tl4` code (tap rate limit 14, 12, 10, 8, 6, 4, 2, 0 taps per 10 s, grace time 3 s; curriculum term `tap_rate`), 40 iterations (24450 → 24490), tensorboard (no wandb). The loop runs, no error. Last iteration: mean reward 170.0, episode length 682, `tap_rate` −1 (warm-up).

Live check of the active path (no file kept): `sg1-leggap` `model_27449`, 512 envs, stage 8, warm-up skipped by hand (rate 14), 1500 steps: 625 finished episodes, 72% by `time_out`, 3% by `tap_limit`, 22% by `base_too_low`. The first level fits the policy, so the survival can pass 0.3.

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --env.scene.num-envs 4096 --env.scene.env-spacing 1.0 --agent.logger tensorboard --agent.run-name taplimit-smoke-rate --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `tl4-taprate`

Same start and recipe as `tl3-budget10`. Change: a tap rate limit replaces the fixed budget per episode. The episode ends when the taps after the mount exceed rate / 10 × (time since the mount + 3 s), rate 14 → 12 → 10 → 8 → 6 → 4 → 2 → 0 taps per 10 s. Wandb `gskv5akf`, run directory `2026-11-11_21-50-48_tl4-taprate`, started Nov 11 2026 21:50 in the tmux session `mjlab-train-tl4` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-tap-limit`, uncommitted (the diff is in the run directory, `git/`).

**Outcome:** finished 4000 iterations (24450 → 28450, last checkpoint `model_28449`). The warm-up ended at its limit (iteration 25950). Rate 14 held for about 620 iterations (the survival stayed at 0.24 to 0.28 and passed 0.3 at about iteration 26570), then rate 12 to the end. Final survival 0.10. No collapse into falls: `base_too_low` stayed at 3 to 5 per iteration at both levels.

| Iteration | Rate | Mean reward | Episode length | `time_out` / iteration | `tap_limit` / iteration | `base_too_low` / iteration | Survival | Mount share |
|---|---|---|---|---|---|---|---|---|
| 26011 | 14 | — | 604 | 1.57 | 1.32 | 3.6 | 0.28 | — |
| 26437 | 14 | — | 580 | 1.80 | 0.98 | 4.2 | 0.25 | — |
| 26560 | 14 | 84.3 | 619 | 1.75 | 0.89 | 3.4 | 0.28 | 0.98 |
| 26607 | 12 | 21.4 | 410 | 0.03 | 6.0 | 3.5 | 0.003 | 0.97 |
| 26795 | 12 | 21.6 | 444 | 0.25 | 4.1 | 4.7 | 0.02 | 0.97 |
| 26935 | 12 | 29.5 | 472 | 0.34 | 3.6 | 4.7 | 0.05 | 0.97 |
| 28449 | 12 | — | — | — | — | 3.6 | 0.10 | 0.97 |

Evaluation of `model_28449` with `scripts/eval_mountbalance.py --obs-noise --gait --probe --audit` (1024 envs, default seed, task `-LegGap`: old noise, no tap termination, 20 s per env):

| Policy | Survived | Stance share | Stance held (last 3 s) | Taps per 10 s | Stance segment median | Base tilt | CoM − ball x | Base yaw rate |
|---|---|---|---|---|---|---|---|---|
| `sg1-leggap` | 88% | 58% | 59% | 13.5 | 0.38 s | 10.9° | −4.7 cm | — |
| `tl2-earlyend` | 73% | 47% | 48% | 12.0 | 0.36 s | 11.9° | −5.2 cm | −0.07 rad/s |
| `tl4-taprate` | 88% | 53% | 54% | 12.9 | 0.38 s | 12.4° | −5.0 cm | +1.27 rad/s |

Success by the strict definition is 0%. Envs with the stance held above 50% of the last 3 s: 77%. `r_hip_pitch` at the force limit in 56% of the steps. Targets outside the joint limits 8.0%. The robot turns on the ball: mean base yaw rate +1.27 rad/s (about 73° per second), 100% of the envs turn counter-clockwise; ball yaw rate +0.32 rad/s.

Reading: `tl4` keeps the survival of `sg1-leggap` (88%), unlike `tl2` (73%). The tap rate in the evaluation is only 4% lower (12.9 against 13.5 per 10 s), and the stance share is 5 points lower. The tap limit has reached only rate 12, about the rate of the old policies, so a large drop in taps is not expected yet. The constant turn is new: no reward term pays it, and no earlier evaluation in this log shows it. Next: continue at rate 12 (resume), and check if the turn stays.

```
uv run python scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap --checkpoint logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_taplimit/2026-11-11_21-50-48_tl4-taprate/model_28449.pt --obs-noise --gait --probe --audit --num-envs 1024
```

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit --name tl4-taprate --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 4000 --agent.seed 42
```
