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

## Conclusion (Nov 6 2026)

- **A cold start reaches the final stage** with ramped stages, a std cap and `action_excess` from stage 0 (`cur4` → `cur4b` → `cur4c`). The two earlier collapses (`cur2`, `cur3`) came from raw outputs drifting past the ±7 clip in stages 0-2, where no penalty has a gradient; `action_excess` from stage 0 kept them inside (median |a| ~1 against 45).
- **Best robustness and posture so far** (end of stage 7, blind on the ball): survival 96-98% at 10-20 ms, 94% with soft motors; base tilt 6°; support foot facing forward (−10°) with the full thigh range.
- **But no held stance:** a tap-and-recover rhythm (free foot behind, ~1.5 floor taps per second, stances of ~0.36 s), the same without noise or latency. The reward lets a tap cost only a few steps of stance reward.
- Next steps: see [mount_curriculum.md](mount_curriculum.md#next-steps) (penalize the tap and reward stance duration from `model_24450`; then placement; then a physical-limit check).
