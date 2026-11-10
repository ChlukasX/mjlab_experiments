# mount_curriculum run log

Design: [mount_curriculum.md](mount_curriculum.md).

## `cur1`: cold start through all stages

Task `Mjlab-Piplus-Ball-MountMin-Curriculum`, cold start, 20000 iterations, 2048 envs, seed 42, weights from the task's stages (no overrides; `logs/probes/run_cur1.sh`). Local 4090, 2.15 s per iteration (~12 h for 20000). Launched Nov 4 2026 in tmux `mjlab-cur1` (window `sweep` runs `after_cur1.sh` when it ends: the mount_latency sweep and the side, latch, kin-ball, free-leg and foot-yaw probes on the stage-8 play config, into `logs/probes/sweep_cur1.txt`).

First 65 iterations: stage 0, falls 42 → 1 per log interval, episode length 34 → 542, `foot_lift` 0.007 → 0.08 per episode, stance reward and share 0 (as `mm1` early on), lag 0-15 ms, noise 0.25×, no NaN.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur1` | `ef6lg6t0` | **Stopped** at iteration 3072 (`model_3000`), never left stage 0: **parked like `st1`**, a foot level on the ball top and the weight on the floor foot. Per episode at ~2930: `sole_flat` 1.92 (of 2; 0.02 at ~1000), `foot_to_ball_top` 1.64, `foot_lift` 0.98, `com_over_ball` 0.43, `free_foot_lift` 0.001, `single_leg_stance` and `support_on_ball` 0 throughout, stance share 0; episode length ~970 (timeouts). `mm1` (full thigh range) mounted by ~2000. With `st1` and `st3`, the third run with the thigh at ±0.6 rad to park there: the limit, not the curriculum, blocks the mount. |

## `cur2`: same curriculum, full thigh range

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh` (`straight=False`): the curriculum of `cur1` with the MJCF thigh range (−2.77 to 2.45 rad). `support_foot_heading` is still added (gated, weight 0 until stage 8), so the last stage asks the support foot to point like the torso; with the full range that is a pull to untwist, not a hard limit. Experiment `piplus_ball_mountmin_curriculum_fullthigh`. Check `logs/probes/test_curriculum.py <task>` passes for both tasks (thigh range per task, stage settings 0-8, advance rule).

Cold start, 20000 iterations, 2048 envs, seed 42, no overrides (`logs/probes/run_cur2.sh`). Local 4090, 2.2 s per iteration (3.0 while a viewer ran beside it; ~12 h). Launched Nov 4 2026 in tmux `mjlab-cur2` (window `sweep` runs `after_cur2.sh` when it ends, into `logs/probes/sweep_cur2.txt`). First 60 iterations: stage 0, falls 43 → 0.7 per log interval, episode length 34 → 565, `foot_lift` 0.007 → 0.09 per episode, no NaN.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur2` | `g9ud6tzg` | Finished (`model_19999`), **collapsed at stage 3**. Stages 0-2 worked: stage 0 (3 cm) to iteration ~7430 (weight on the ball from ~2100, free foot up from ~4000, share 0.50), stage 1 (6 cm) to ~12070, stage 2 (10 cm) to ~15220 (`single_leg_stance` 7.1 of 16, share 0.50). Stage 3 switched the smoothing penalties on at once: reward +248 → −1650 per episode (`action_excess` alone −90), and within ~400 iterations it learned to fall at once (episodes of 6 steps, 339 falls per log interval) and never recovered. The action noise std had grown 1 → 10.9 over stages 0-2 (raw actions clipped to ±7; noise past the clip costs nothing, the entropy bonus pays), but the policy's mean targets also lie far past the joint limits (`cur3` first try: `action_excess` −90 at std 0.87). The sweep ran on the collapsed checkpoint (0% everywhere, not meaningful). |

## `cur3`: resume `cur2` at stage 3 with ramped stage weights and a std cap

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp` (`ramp=True`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp`):
- **Ramped stages** (`ramped_stage_settings`): a stage's new weights and `upright` std move from the previous stage's values over `STAGE_RAMP_ITERS` (stage 3: 2000 iterations, stage 5: 1000, others 500), slowly at first: f = (100^p − 1) / 99, p = progress / ramp length (9% of the way at half the ramp, 31% at three quarters). Switches and the clearance still change at once.
- **A stage advances only after its ramp** (`max(MIN_ITERS, ramp length)`): the next stage ramps from this stage's full values, so advancing mid-ramp would be a jump again.
- **Action noise std capped at 1.0** (`MAX_ACTION_STD`, rsl_rl `std_range`), and the init copy of `cur2` `model_15200` (end of stage 2) has its std clamped to 1.0 (`logs/probes/clamp_std_init.py`; was mean 10.9, max 18.3), so it can still fall below the cap.
- **Stage clock fix:** the clock starts at the first curriculum call with a non-zero step counter (the first call runs at env creation, before a resume restores the counter).
- Checks: `logs/probes/test_ramp.py` (ramp values per stage, std cap in the runner cfg, old tasks unchanged), `test_curriculum.py` on the task.

Resume `model_15200` (std clamped; `piplus_ball_mountmin_curriculum_fullthigh_ramp/2026-11-05_00-00-00_init-from-cur2-15200`), `start_stage 3`, 15000 iterations, 2048 envs, seed 42 (`logs/probes/run_cur3.sh`). Local 4090. Tmux `mjlab-cur3` (window `sweep` runs `after_cur3.sh` when it ends, into `logs/probes/sweep_cur3.txt`).

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur3` try 1 | `e1xfsvx3` | Stopped after ~130 iterations (dir `*_cur3-try1-clockbug`): the stage clock started at 0, before the resume restored the step counter, so MIN_ITERS and the ramp counted as done: `action_excess` −92 at once, stage 3 → 4 after ~45 iterations. |
| `cur3` try 2 | `9fvrik9s` | Stopped at ~150 iterations (dir `*_cur3-try2-advance-midramp`): ramp working (reward ~+300, `action_excess` 0 → −0.36, falls unchanged, stance share 0.57), but the share had passed 0.5 and the stage would have advanced at 300 iterations, mid-ramp. Fixed by the ramp-complete rule. |
| `cur3` | `zrb014gg` | **Stopped** at iteration ~19800, **collapsed in stage 3** at ~1400-1600 iterations into the ramp, as `cur2`. The ramp did what it should (weight −0.02 → −0.9 by iteration 1400 of 2000), but the overshoot past the joint limits did not fall: `action_excess` per second / weight stayed at ~22-24 rad at every weight. Reward +280 → +26 (iteration 1200) → −99 (1400), then falls at once (episodes of 6 steps, ~340 falls per log interval) for the last ~2500 iterations. |

**Cause of both collapses** (`logs/probes/action_mean_probe.py` on `cur2` `model_15200`, the end of stage 2, 256 envs): the policy's raw outputs have median |a| **45**, **70%** past the runner's ±7 clip (66% past ±14); per joint up to 60-70 (ankle pitch, hip roll, elbows). The runner clips the raw output before the env, so moving an output from 45 to 20 changes nothing in the simulation and no penalty has a gradient there: the policy is bang-bang, and ramping the penalties cannot pull it back. The outputs drifted out during stages 0-2, where nothing held them inside the clip (no `action_excess`, noise std free to grow to ~11). The hand-made chain did not hit this because its early tasks had no raw clip, so `action_excess` saw the whole output.

## `cur4`: cold start with `action_excess` from stage 0

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess` (`excess0=EARLY_ACTION_EXCESS`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess`): as `-Ramp` (ramped stages, ramp-complete rule, std cap 1.0), plus `action_excess` −1 in stages 0-2 so the outputs never leave the range where it has a gradient; stage 3 ramps it −1 → −4. `mm4`-`mm7` mounted with `action_excess` on. Checks: `test_ramp.py` (−1 in stages 0-2, −1 → −4 over stage 3, std cap; `-Ramp` unchanged), `test_curriculum.py` on the task.

Cold start, 30000 iterations, 2048 envs, seed 42 (`logs/probes/run_cur4.sh`), 2.2 s per iteration (~18 h). Tmux `mjlab-cur4` (window `sweep` runs `after_cur4.sh` when it ends, into `logs/probes/sweep_cur4.txt`). Plan: `action_mean_probe.py` at ~iteration 3000 to check that the outputs stay inside ±7.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur4` | `n90vngh3` | **Stopped** at iteration 12816 (`model_12800`), stuck in stage 6. First 60 iterations: stage 0, falls 43 → 1.9 per log interval, episode length 34 → 450, `foot_lift` 0.007 → 0.08, `action_excess` −0.09 per second, std 1.0 → 0.80. Weight on the ball from ~2250 (`support_on_ball` 1.83), free foot up from ~3750; **stage 0 → 1 at ~4566** (`cur2`: ~7430). Stage 1 at ~5100: share 0.48, `single_leg_stance` 3.3, `free_foot_lift` 1.24. **Outputs in range** (`action_mean_probe.py`, `model_5150`): raw |a| median 0.96, mean 1.35, 0% past the ±7 clip (`cur2` stage 2: median 45, 70%); largest per joint l upper arm 3.9, r hip pitch 2.7. Std 0.67. Stages: 0 → 4566, 1 → 5254 (688), 2 → 6268 (1014), 3 → 8267 (1999, the full ramp, **no collapse**: share 0.52 → 0.67, reward +301 → +311 with all smoothing penalties at full weight, `action_excess` −1.9 per second ≈ 0.5 rad overshoot), 4 → 8766 (499; latency to 40 ms: share 0.68 → 0.45 → 0.55, falls up to 1.6 per log interval), 5 → 9765 (999; one-foot + level: share 0.01 → 0.41, `ankle_roll_excess` −0.13 at full weight), 6 from 9765 (noise 0.25 → 1×: share 0.41 → 0.30, then 0.35 by ~12000 and flat to 12800 against the 0.4 threshold; reward +200-220, falls 0.7-1.1). |

### `cur4b`: resume `cur4` at stage 7 with a lower late threshold

`stage_curriculum` gets `late_threshold` (default 0.4 = `THRESHOLD[6]`, so `cur4` is reproducible): the advance threshold of stages 6 and 7. Resume `cur4` `model_12800` with `start_stage 7` (blind on the ball: kinematic ball estimate after the latch) and `late_threshold 0.3`, 17200 iterations (to 30000), same experiment dir, run name `cur4b` (`logs/probes/run_cur4b.sh`; tmux `mjlab-cur4b`, window `sweep` runs `after_cur4b.sh`, into `logs/probes/sweep_cur4b.txt`). Stage 6 counts as done: its noise ramp had ended ~2500 iterations before.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur4b` | `k6nfwd6c` | **Crashed** at iteration 21605 (last checkpoint `model_21600`), at the stage 7 → 8 hand-over: `stage_curriculum` looked up `THRESHOLD[8]` (the final stage has none; the `late_threshold` change had dropped the `s < FINAL_STAGE` guard, and `test_curriculum.py` was not rerun after it). Fixed; the test now covers the 7 → 8 hand-over. Stage 7 ran 8806 iterations: episode length 63 → ~480 (iteration 1000) → 630-690, share 0.22 → 0.29 → past 0.3 at the end, falls 7-17 → 1.6-2.1 per log interval, reward +108 → +167. |

**Sweep of `cur4b` `model_21600`** (end of stage 7, on the stage-8 play config: blind on the ball, sizes 3-5; `logs/probes/sweep_cur4b.txt`): success 0% in every condition (the held-stance test: ≥ 90% of the last 3 s), **survival 96 / 98 / 93 / 68%** at 10 / 20 / 30 / 40 ms, 90% with 10-40 ms jitter, 94% with soft motors, 98% without noise; on the ball after 0.6 s; latched (blind) 100% of the alive envs. Kinematic estimate error median 1.3 cm. Foot probe: **base tilt 6.1°** (`fl1` 14.5°, `kb1` 20.7°, `nr1-up6` 28.5°), sole 6.7° from horizontal, foot 8.6° from the top, ankle roll −0.27 rad (at the stop 0%). Free leg: touches the ball 7%, free ankle 1.6 r from the ball centre, 6.2 cm out from the support foot (`fl1` 2.4 cm). Support leg share: hip pitch 27%, calf 26%, ankle 29%. **Foot yaw: support foot −10° from the torso** (p10 −21°, p90 +7°), thigh +26°: no twist, with the full thigh range (`mm1`-`fl1`: 120-160°).

### `cur4c`: resume `cur4b` at stage 8

Resume `cur4b` `model_21600` with `start_stage 8` (free leg), `late_threshold 0.3`, 8400 iterations (to 30000), run name `cur4c` (`logs/probes/run_cur4c.sh`; tmux `mjlab-cur4c`, window `sweep` runs `after_cur4c.sh`, into `logs/probes/sweep_cur4c.txt`).

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur4c` | `3z7h5zxk` | **Stopped** at iteration 24452 (`model_24450`) on Nov 6 2026, ~2850 iterations into stage 8: stance share 0.31 throughout (0.306 at ~22300, 0.315 at ~24400), `single_leg_stance` 3.3-3.9 (of 16), `free_foot_ball_clear` 0.42-0.49 (of 2), `support_foot_heading` 0.24-0.29 (of 1), `support_hip_vel` −0.13 to −0.17, episode length 590-700, falls 1.6-2.4 per log interval. The free-leg terms did not change the tap rhythm (below). No sweep (stopped before the end). |

**Free leg behind: a tap-and-recover rhythm, not a held stance** (Nov 6 2026, from a play session of `cur4c` `model_23150`: the free leg is always behind the support leg). Probe `logs/probes/free_foot_place_probe.py` (512 envs, 20 s, latched on-ball steps after 3 s; free ankle relative to the support ankle and CoM relative to the ball centre in the torso heading frame, x forward):

| Condition | alive at 20 s | in stance | free ankle x / y, in stance | behind (x < −3 cm) | CoM − ball x | free-sole floor taps | stance segments, median / p90 |
|---|---|---|---|---|---|---|---|
| 20 ms, noise on | 98% | 55% | **−14.5** / +5.2 cm | **100%** | −5.2 cm | **14.8 per 10 s** | **0.36 / 0.46 s** |
| no latency, noise off | 95% | 53% | −14.3 / +4.2 cm | 100% | −6.3 cm | 18.5 per 10 s | 0.28 / 0.40 s |

- The free foot is behind the support foot in every stance step (median 14.5 cm back, 5 cm out), and the CoM sits ~5 cm behind the ball centre: the robot leans back over the ball and catches itself on the rear foot, ~1.5 times per second; no stance lasts longer than ~0.5 s.
- **Not forced by noise or latency:** without either, the rhythm is the same or faster (18.5 taps per 10 s, segments 0.28 s). It is a learned strategy, not the limit the sensing/delay sets.
- **Why it pays:** a tap only loses the stance reward for those steps (no penalty, no termination), while a fall ends the episode; the tap keeps the robot on the ball (survival 96-98%), and a stance share of ~0.3-0.55 passes the late-stage thresholds (0.3). Holding the stance continuously would need closed-loop balance on a point contact; tapping resets the lean every ~0.4 s.
- **Physical limit not shown:** the earlier warm-started policies held long stances (`ms3-shell` 98% success with the ball visible and 0-15 ms; `kb1` blind with latency: full stance 77% of the steps). A test: reward/penalize against the tap and see whether stance segments lengthen.
- Options for a next run (not started): penalize free-sole floor contact while latched; a stance-duration bonus (pay more the longer the current stance lasts); and only then a placement term (free foot to the side/front), since without the tap incentive the rear placement may change by itself.

## `cur5`: warm start from `cur4c` with the `-NoTap` terms

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap` (flag `notap`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap`). It adds two reward terms at stage 8: `free_sole_floor` (penalty, weight −8) and `stance_duration` (bonus, weight +16, cap 2 s). The terms are in the Design section of [mount_curriculum.md](mount_curriculum.md). Observations and network are unchanged, so the `cur4c` actor and critic load (checked: the loaded actor equals the actor of `model_24450`).

Start: a copy of `cur4c` `model_24450.pt` (byte-identical) in the new log group, as `2026-11-07_00-00-00_init-from-cur4c-24450`. `--env.curriculum.stage.params.start_stage 8` starts the stage clock at stage 8. The three old stage-8 terms (`free_foot_ball_clear`, `support_hip_vel`, `support_foot_heading`) therefore ramp in again over 500 iterations, together with the new terms.

4096 envs need `--env.scene.env-spacing 1.0`. With the default 2.0, a smoke run crashed at iteration 5 with NaN in the actor observation. The NaN envs lie far from the world origin (the grid reaches ±64 m at 4096 envs). Spacing 1.0 halves the extent.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur5-notap` | `3zudj4wb` | **Killed** at iteration 24986 (last checkpoint `model_24950`) after 31 min. The log has no error. The process ended with the agent session (`setsid nohup`, not tmux). |
| `cur5b-notap` | `xz7581wc` | **Finished** 3000 iterations (24450 → 27450, last checkpoint `model_27449`) in 2 h 53 min, 3.45 s per iteration. Restart of `cur5-notap` from the same init copy, in tmux `mjlab-train`. |

Launch command of `cur5-notap`:

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap --name cur5-notap --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-07_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

Launch command of `cur5b-notap`:

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap --name cur5b-notap --tmux --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-07_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

**Training (`cur5b-notap`).** The run is stable: no NaN, no divergence.

| Iteration | Mean reward | Stance share | `single_leg_stance` (of 16) | `free_sole_floor` | `stance_duration` |
|---|---|---|---|---|---|
| 24700 | 202 | 0.35 | 4.6 | −0.10 | +0.03 |
| 24950 | 227 | 0.36 | 4.8 | −0.84 | +0.39 |
| 26200 | 211 | 0.38 | 4.8 | −0.33 | +0.44 |
| 27449 | 224 | 0.40 | 5.3 | −0.22 | +0.52 |

The `free_sole_floor` value fell by 74% from iteration 24950 to 27449. The eval below shows that the taps did not fall by the same amount.

**Gait eval** (`scripts/eval_mountbalance.py --task <task> --checkpoint <model.pt> --gait`, 1024 envs, mount spawn, ball sizes 3-5):

| Metric | `cur4c` `model_24450` | `cur5-notap` `model_24950` | `cur5b-notap` `model_27449` |
|---|---|---|---|
| Floor taps per 10 s | 13.3 | 12.9 | **11.9** |
| Stance share of the on-ball steps | 57% | 59% | 61% |
| Stance segment median / p90 | 0.42 / 0.58 s | 0.44 / 0.62 s | 0.48 / 0.70 s |
| Survived | 94% | 95% | 97% |
| Free foot behind the support foot | 100% | 100% | 100% |

The held-stance success is 0% for all three checkpoints (the test needs the stance in at least 90% of the last 3 s). The policy still taps.

**Free-foot state** (final policy `model_27449` against `cur4c` `model_24450`; latched, alive, after 3 s; the numbers come from a patched copy of the `--gait` loop that is not kept in the repository):

| Within single support | `cur4c` | `cur5b` |
|---|---|---|
| Free foot touches the ball | 7.4% (13.6 events per 10 s) | 3.1% (7.1 events per 10 s) |
| Free foot on the floor | 14.4% (13.6 events per 10 s) | 13.2% (12.0 events per 10 s) |
| Free foot in the air | 78.2% | 83.7% |

- The ball touches fell by about half. The stance bonus needs the free foot off the ball, so this follows from `stance_duration`.
- The floor taps stayed at about the same rate.
- **`free_sole_floor` is blind.** On `model_27449` a sole is on the floor in 25.9% of the on-ball steps. The term is active in 1.5% of them. In 94% of the floor steps the term is zero, because it needs a flat and level support foot on the ball (`_support_foot` with `"level"`). A tap disturbs the support foot, so the gate fails when the free foot is on the floor. The fall of the term in training shows that the policy avoids the narrow case that the term penalizes. It does not show fewer taps.
- **The price of a tap is low.** A floor contact lasts about 0.11 s. At weight −8 it costs about 0.9 reward. A fall ends the episode, so the tap stays the better choice.

**Conclusion.** `stance_duration` and the stance share moved in the right direction (stance segment median 0.42 → 0.48 s, survival 94% → 97%). The floor taps did not stop (13.3 → 11.9 per 10 s), because the penalty term does not see most of them. Next run: penalize any sole on the floor, and the free foot on the ball, while latched. See Next steps in [mount_curriculum.md](mount_curriculum.md#next-steps).

## `cur6`: warm start from `cur4c` with the widened `-NoTap` penalty

`cur5b` showed that `free_sole_floor` was almost never active (section `cur5`). The task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap` now carries the term `free_foot_contact` in its place. The task changed in place: it exists only on the unmerged branch, and the naming convention allows one new suffix per experiment. The earlier definition is in the section `cur5` and in commit `28730cc`. The term is in the Design section of [mount_curriculum.md](mount_curriculum.md).

Definition of `free_foot_contact` (value 1 or 0 per step, weight −8, from stage 8): the robot is latched on the ball, and either (a) the lowest corner of any sole is within 1 cm of the floor, or (b) exactly one foot is the support foot (flat sole on the ball) and the other foot touches the ball. There is no level test and no support-foot test in (a). The weights stay as in `cur5`: `free_foot_contact` −8, `stance_duration` +16 (cap 2 s). Observations and network are unchanged. Check: the `cur4c` `model_24450` actor and critic load strictly into the `-Excess` task and into the `-NoTap` task, and the actor outputs of the two tasks are equal (largest difference 0.0 on the real observations and on random observations).

Start: the same init copy as `cur5`, `2026-11-07_00-00-00_init-from-cur4c-24450`, a fresh start of the `-NoTap` task (`cur6`).

**Size of the terms.** Measured on the start policy (`cur4c` `model_24450`) with the play config of the task (stage 8, full weights), 256 envs, seed 3, 20 s episodes, steps from 3 s to 15 s, mean over all envs of the weighted value per second in the reward manager (a one-off measurement, no script is kept):

| Term | Weight | Value per second | Remark |
|---|---|---|---|
| `free_foot_contact` | −8 | −2.32 | active on 29% of the steps |
| `stance_duration` | +16 | +1.13 | mean factor 0.07 |
| `single_leg_stance` | +16 | +9.22 | stance share 0.58 |
| All terms | | +21.26 | |

The penalty is 25% of the `single_leg_stance` value and 11% of the total. The first definition was −0.22 per second at iteration 27449 of `cur5b`. The penalty does not dominate the stance reward, so the weights stay at −8 and +16.

**Smoke test** `notap2-smoke` (wandb `717pzj3s`, 20 iterations from 24450 to 24470, same flags as below with `--agent.max-iterations 20`, no tmux): no NaN. The curriculum logs `Curriculum/notap/free_foot_contact` and `Curriculum/notap/stance_duration`, and the reward logs show both terms (at iteration 24469: `free_foot_contact` −0.0028, `stance_duration` +0.0004 per episode step, the weights are in the ramp).

**Eval script check.** `scripts/eval_mountbalance.py --gait` now prints the free-foot state. Run on `cur5b-notap` `model_27449` (1024 envs): single support 79% of the on-ball steps, neither foot on the ball 21%. In single support the free foot touches the ball in 3.1%, is on the floor in 13.2%, and is in the air in 83.7%. Ball-contact events 7.1 per 10 s, floor events 12.0 per 10 s. The new `free_foot_contact` is active on 26.7% of the on-ball steps; a sole is on the floor on 25.9%. The tracked script reproduces the free-foot numbers of the section `cur5`.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur6-notap` | `pr33kxoj` | **Finished** 3000 iterations (24450 → 27450, last checkpoint `model_27449`) in 2 h 54 min, 3.5 s per iteration. Run directory `2026-11-07_18-42-03_cur6-notap`. Stable: no NaN, no divergence. The taps did not stop (eval below). |

**Training end values** (iteration 27449): mean reward 203.7, mean episode length 731, stance share 0.39, `single_leg_stance` 4.86, `stance_duration` 0.48, `free_foot_contact` −1.66 (logged episode values).

**Gait eval** (`scripts/eval_mountbalance.py --task <task> --checkpoint <model.pt> --gait --probe --audit`, 1024 envs, mount spawn, ball sizes 3-5) of `cur6-notap` `model_27449`, against `cur5b-notap` `model_27449`:

| Metric | `cur5b-notap` | `cur6-notap` |
|---|---|---|
| Floor taps per 10 s | 11.9 | 12.3 |
| Stance share of the on-ball steps | 61% | 63% |
| Stance segment median / p90 | 0.48 / 0.70 s | 0.48 / 0.68 s |
| Survived | 97% | 94% |
| Free foot behind the support foot | 100% | 100% |
| Free foot touches the ball (single support) | 3.1% (7.1 events per 10 s) | 2.4% (5.2 events per 10 s) |
| Free foot on the floor, no ball (single support) | 13.2% (12.0 events per 10 s) | 12.6% (12.2 events per 10 s) |
| `free_foot_contact` active / any sole on the floor | 26.7% / 25.9% | 23.7% / 23.5% |

The held-stance success is 0%. Other values of `cur6-notap`: base tilt 9.7°, CoM 4.3 cm behind the ball centre, support foot heading −20° from the torso, raw |a| median 1.12 (0% past the clip), targets outside the joint limits 5.8%, `r_hip_pitch` at the force limit in 43% of the steps.

**Conclusion.** The widened penalty is active as designed (it covers the floor steps), but the taps did not change: 12.3 against 11.9 per 10 s. The penalty of about 8 per second of contact does not outweigh what the tap gives the policy (recovery of the balance). A larger weight is risky (no termination penalty, see Design). The cause is open: the next probe looks at the motion of the support foot on the ball.

Launch command of `cur6-notap`. The session `mjlab-train` held two idle shells from `cur5b`, so the command ran inside the tmux session `mjlab-train-cur6`, without `--tmux`:

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap --name cur6-notap --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-07_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

## `cur7`: warm start from `cur4c` with a middle thigh range and no heading reward

**Reason.** `cur6` keeps the support foot straight (−20° from the torso) and taps. Four policies, all measured with `scripts/eval_mountbalance.py --gait --probe --audit` (1024 envs, mount spawn):

| Policy | Held-stance success | Taps per 10 s | Segment p90 | `r_hip_pitch` at the force limit | CoM behind the ball centre | Support foot vs torso |
|---|---|---|---|---|---|---|
| `cur6` `model_27449` | 0% | 12.3 | 0.68 s | 43% | 4.3 cm | −20° |
| `cur4b` `model_21600` | 0% | 14.3 | 0.52 s | 58% | 5.0 cm | −15° |
| `kb1` | 16% | 4.9 | 1.60 s | 3% | 1.2 cm | +109° |
| `ms3-shell` | 98%, but the free foot rests on the ball in 94% of the single-support steps | 0 | – | 25% | 1.4 cm | +138° |

Other one-off checks on `cur6` `model_27449` (the scripts are not kept): more actuator force (up to 4×) leaves the gait unchanged, and a fixed rule for the support foot in the kinematic ball estimate leaves it unchanged. So force limit and ball estimate do not cause the taps. The posture is the open candidate. This is a correlation across four policies, not an intervention.

**Definition.** Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-MidThigh` (`thigh_limit=1.2`). The `-NoTap` task already has the full thigh range, so a limit alone does not change the policy: the heading term (weight 1.0 from stage 8) pays for a straight foot. The new task limits the two thigh joints to ±1.2 rad (about 69°) in the model and in the action clip, and the curriculum term `heading_off_curriculum` sets the weight of `support_foot_heading` to 0 (it runs after `stage_curriculum`). All other terms and weights equal `cur6`: `free_foot_contact` −8, `stance_duration` +16. The observations and the network are unchanged.

Start: a copy of `cur4c` `model_24450` in the log group of the new task, `2026-11-09_00-00-00_init-from-cur4c-24450`. Same seed and flags as `cur6`. The change against `cur6` is the task only.

**Checks.** `uv run pytest`: 78 passed, 2 deselected. C1: 50 tasks, 0 problems. C2: 0 differences after the new baseline entry; the other 49 entries are equal. C3: 351 checkpoints, 10 known history-only, 0 problems.

**Smoke test** `midthigh-smoke` (20 iterations from 24450 to 24470, no tmux): no NaN. `support_foot_heading` is 0.0000 and `Curriculum/heading_off/heading_weight` is 0.0000. Both `-NoTap` terms are logged.

**Run.** `cur7-midthigh`, wandb `dyk3z52j`, run directory `2026-11-09_09-22-56_cur7-midthigh`, started Nov 9 2026 in the tmux session `mjlab-train-cur7`. Finished 3000 iterations (24450 to 27450, last checkpoint `model_27449`) in 2 h 46 min. Training end values at iteration 27449: mean reward 188.3, mean episode length 717, stance share 0.40, `single_leg_stance` 4.95, `stance_duration` 0.47, `free_foot_contact` −1.79. The `r_thigh` joint is at the force limit in 2% of the steps.

Eval of `model_27449` (`--gait --probe --audit`, 1024 envs, mount spawn), against `cur6`:

| Metric | `cur6` | `cur7` |
|---|---|---|
| Held-stance success | 0% | 0% |
| Survived | 94% | 96% |
| Taps per 10 s | 12.3 | 12.8 |
| Stance share | 63% | 61% |
| Segment median / p90 | 0.48 / 0.68 s | 0.46 / 0.64 s |
| Support foot vs torso | −20° | −23° |
| CoM behind the ball centre | 4.3 cm | 4.5 cm |
| `r_hip_pitch` at the force limit | 43% | 47% |

**Conclusion.** The policy did not leave the straight posture. Without the heading reward and with the thigh range up to ±1.2 rad, the support foot still points forward (−23°), and the gait equals `cur6`. A warm start from `cur4c` stays in the straight basin. The test does not show whether a turned foot would hold better, only that this warm start does not find it.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-MidThigh --name cur7-midthigh --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-09_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

## `cur8`: warm start from `cur4c` with a reward for a turned support foot

**Reason.** `cur7` allowed a turned support foot (thigh ±1.2 rad, heading reward 0), and the policy stayed straight (−23°). The test needs a push toward the turn. Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-Turn` (flag `turn`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_turn`). It builds on `-NoTap` (full thigh range, not `-MidThigh`) and adds the reward `support_foot_turn`. The heading reward `support_foot_heading` has weight 0. This is the third new suffix in `mount_curriculum`; the naming convention allows one per experiment (the user knows). The term is in the Design section of [mount_curriculum.md](mount_curriculum.md).

**Definition.** Per step, in the gated single support (as `support_foot_heading`): 0.5 × (1 + cos(e)), e = d − 100°, d = yaw(support foot) − yaw(torso). This is the number that `scripts/eval_mountbalance.py --gait` prints as "support foot heading - torso" (wrapped to ±180°). The sign of d is mirrored for the left foot. The value is 1 at the target, 0.5 at 90° from it, and 0 at 180° from it.

**Side and target (measured).** Script: a copy of the `eval_mountbalance.py` loop that records d per support foot (not kept). 1024 envs, mount spawn, steps after 3 s, gated single support:

| Policy | Right foot is support foot | d, median (p10 / p90) |
|---|---|---|
| `kb1` `model_59946` | 100% | +109° (+99° / +119°) |
| `cur4c` `model_24450` | 100% (3 steps of 526064 left) | −15° (−23° / −6°) |
| `cur6` `model_27449` | 100% | −20° (−27° / −10°) |

No measured policy uses the left foot as support foot. The sign for the left foot is the mirror image and is not tested. The target is +100° (about the p10 of `kb1`, and a round number inside the "about 100°" of the decision).

**Width.** A clamped cos around +100° is 0 for every heading from −170° to +10° (more than 90° from the target). The start policy is at −15°, 115° from the target, so it would get value 0 and no gradient. The raised cosine has a gradient everywhere except at 180° from the target. Its value on `cur4c` is 0.291 when active (p10 0.227, p90 0.363).

**Size of the term.** Measured on the play config of the task (stage 8, full weights), 256 envs, seed 3, 20 s episodes, steps from 3 s to 15 s, mean over all envs of the weighted value per second in the reward manager (as in the section `cur6`; a one-off script, not kept). The measurement used weight 4. The values for weight 8 are scaled (the term is linear in the weight):

| Policy | `support_foot_turn` at weight 1, mean per step | Active (gated single support) | Mean when active | All terms per second | Std of the total reward per second across envs |
|---|---|---|---|---|---|
| `cur4c` `model_24450` | 0.182 | 62.5% | 0.291 | 21.28 (w 4) | 1.84 |
| `cur6` `model_27449` | 0.166 | 65.7% | 0.253 | 23.07 (w 4) | 1.92 |
| `kb1` `model_59946` | 0.817 | 83.0% | 0.985 (p10 0.974) | 17.86 (no turn term) | 1.82 |

The term is high on `kb1` (0.985 when active). Weight choice, with the numbers of `cur4c`:
- Weight 8 gives +1.46 per second on the start policy (6.6% of the total 22.0), against +9.18 for `single_leg_stance`. The largest possible value is 8 × 0.83 = 6.6 per second, below the stance value.
- The gain from the start posture to a turned foot at the same active share is 8 × 0.625 × (0.985 − 0.291) = 3.5 per second. This is 1.9 times the std across envs (1.84), 38% of the stance value, and 1.5 times the tap penalty (2.33 per second, which did not change the gait in `cur6`). A turn of 10° from −15° is worth 0.4 per second.
- Weight 4 would give a gain of 1.7 per second, 0.94 times the std. This is not clearly above the noise. The weight stays below the stance weight (16), as `free_foot_contact` (−8, half of it).

**Checks.** `uv run pytest`: 92 passed, 2 deselected (14 new tests in `tests/test_turn.py`, stand-in env). C1: 51 tasks, 50 log groups, 0 problems. C2: 51 tasks compared, 0 differences after the new baseline entry (the 50 old fingerprints are equal). C3: 414 checkpoints, 10 known history-only, 0 problems. Strict load of `cur4c` `model_24450` (actor, critic and optimizer, `strict=True`) into the `-NoTap` task and the `-Turn` task: both load, the largest difference to the checkpoint is 0.0 for actor and critic, and the actor outputs of the two tasks on random observations are equal (largest difference 0.0). The actor takes 345 inputs, the critic 88.

**Smoke test** `turn-smoke` (wandb `ubzmhdlg`, run directory `2026-11-09_13-38-48_turn-smoke`, 20 iterations from 24450 to 24470): no NaN. `Curriculum/turn/turn_weight` is 0.0159 at iteration 24469 (the ramp: 8 × 0.002), `support_foot_heading` is 0.0000.

Start: a copy of `cur4c` `model_24450.pt` (SHA-256 `83ec928c…`, byte-identical to the other init copies) as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_turn/2026-11-09_00-00-00_init-from-cur4c-24450`.

| Run | wandb | Outcome |
|-----|-------|---------|
| `cur8-turn` | `i86jdjb0` | **Finished** 3000 iterations (24450 → 27450, last checkpoint `model_27449`) in 2 h 53 min, 3.5 s per iteration. Run directory `2026-11-09_13-40-49_cur8-turn`, tmux session `mjlab-train-cur8`. Stable: no NaN, no divergence. The support foot did not turn (eval below). |

**Training** (mean of the log entries within 20 iterations of the given iteration; the turn weight reaches 8 at about iteration 24950):

| Iteration | Mean reward | Stance share | `single_leg_stance` (of 16) | `free_foot_contact` | `support_foot_turn` (weighted) |
|---|---|---|---|---|---|
| 24650 | 204 | 0.35 | 4.6 | −0.11 | 0.04 |
| 24950 | 205 | 0.37 | 4.8 | −1.87 | 0.81 |
| 25950 | 207 | 0.38 | 4.9 | −1.83 | 0.89 |
| 27449 | 217 | 0.40 | 5.2 | −1.76 | 0.92 |

At the last iteration: mean reward 240.9 (one log entry, with the turn term in it, so not comparable to `cur6` and `cur7`), episode length 805, stance share 0.41, `stance_duration` 0.51, `support_foot_turn` 0.96. The turn term rose by 14% after the ramp ended.

**Eval** of `model_27449` (`--gait --probe --audit`, 1024 envs, seed 0, mount spawn, ball sizes 3-5), against `cur6`, `cur7` and `kb1`. The `kb1` column is a new run of the same command on `kb1` `model_59946` (task `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall`); the taps value is 4.8, where the section `cur7` has 4.9:

| Metric | `cur6` | `cur7` | `cur8` | `kb1` |
|---|---|---|---|---|
| Held-stance success (strict) | 0% | 0% | **0%** | 16% |
| Survived | 94% | 96% | **94%** | 92% |
| Taps per 10 s | 12.3 | 12.8 | **12.2** | 4.8 |
| Stance share of the on-ball steps | 63% | 61% | **63%** | 78% |
| Segment median / p90 | 0.48 / 0.68 s | 0.46 / 0.64 s | **0.48 / 0.70 s** | 0.38 / 1.62 s |
| Support foot vs torso (p10 / p90) | −20° | −23° | **−13° (−18° / −6°)** | +109° (+99° / +116°) |
| CoM behind the ball centre | 4.3 cm | 4.5 cm | **4.0 cm** | 1.2 cm |
| `r_hip_pitch` at the force limit | 43% | 47% | **46%** | 3% |
| Free foot behind the support foot | 100% | 100% | 100% | – |

Other values of `cur8`: time to the stance 0.6 s, stance held 64% of the last 3 s, base tilt 10.3°, `r_thigh` at the force limit 4%, raw |a| median 1.01 (0% past the clip), targets outside the joint limits 5.6%, free foot in the air 84.4% / on the floor 12.7% / on the ball 2.9% of the single-support steps.

**Conclusion.** The support foot did not turn: −13° from the torso (the start was −15°). The gait equals `cur6` and `cur7` in every column: 12.2 taps per 10 s, segments 0.48 / 0.70 s, held-stance success 0%, `r_hip_pitch` at the limit in 46% of the steps, CoM 4 cm behind the ball. Survival is unchanged (94%), so the reward cost nothing, and it also gained nothing. A pull of about 1.5 per second at the start (3.5 per second for the full turn) is not enough to move a warm start out of the straight basin in 3000 iterations, with and without a heading reward (`cur7`). The test still does not show whether a turned foot would hold the stance better: the correlation across `kb1` and `ms3-shell` stays a correlation. Not tested: a larger weight (the term must stay below the stance weight 16), a target heading that moves from −15° to +100° during the run, the turn term from the cold start (stage 5 or earlier, before the straight posture is learned), and a start from a turned posture (reset event or distillation from `kb1`).

Launch command of `turn-smoke` (without tmux):

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-Turn --name turn-smoke --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-09_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 20 --agent.seed 42
```

Launch command of `cur8-turn` (inside the tmux session `mjlab-train-cur8`, without `--tmux`):

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-Turn --name cur8-turn --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-09_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

## Conclusion (Nov 6 2026)

- **A cold start reaches the final stage** with ramped stages, a std cap and `action_excess` from stage 0 (`cur4` → `cur4b` → `cur4c`). The two earlier collapses (`cur2`, `cur3`) came from raw outputs drifting past the ±7 clip in stages 0-2, where no penalty has a gradient; `action_excess` from stage 0 kept them inside (median |a| ~1 against 45).
- **Best robustness and posture so far** (end of stage 7, blind on the ball): survival 96-98% at 10-20 ms, 94% with soft motors; base tilt 6°; support foot facing forward (−10°) with the full thigh range.
- **But no held stance:** a tap-and-recover rhythm (free foot behind, ~1.5 floor taps per second, stances of ~0.36 s), the same without noise or latency. The reward lets a tap cost only a few steps of stance reward.
- Next steps: see [mount_curriculum.md](mount_curriculum.md#next-steps) (penalize the tap and reward stance duration from `model_24450`; then placement; then a physical-limit check).

*Cleanup (Nov 6 2026):* the `logs/probes/` scripts named above (run/sweep wrappers, checks, probes) and the sweep outputs are deleted; the results are in this log. The placement/tap probe and the action probe are now `scripts/eval_mountbalance.py --gait` and `--audit` (`cur4c` `model_24450` with both, 512 envs, 20 ms, noise and bias on: raw |a| median 1.06, 0% past the clip, targets outside the joint limits 7.3%, action change 0.34 per step; free foot −14.5 cm behind in 100% of stance steps, 14.3 floor taps per 10 s, stance segments median 0.40 s, support foot −15° from the torso). Only each run's last checkpoint is kept: `cur2` `model_15200` is gone (its std-clamped copy remains in the `-Ramp` init dir); `cur4b` `model_21600` and `cur4c` `model_24450` remain.
