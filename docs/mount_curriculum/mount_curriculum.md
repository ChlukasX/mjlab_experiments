# mount_curriculum: mount and stance from a cold start in one staged run

Branch: `feat/mount-curriculum` (off `feat/mount-straight`). Tasks `Mjlab-Piplus-Ball-MountMin-Curriculum[-FullThigh[-Ramp[-Excess]]]`, config `tasks/ball_mount_min/piplus_ball_mount_curriculum_cfg.py`. Run log: [mount_curriculum_log.md](mount_curriculum_log.md). **Status (Nov 6 2026): stopped after `cur4c`; see Result and Next steps below.**

## Why

The mount was learned from scratch once: `mm1-min-clip` ([ball_mount_rewards](../ball_mount_rewards/ball_mount_rewards.md)), on an easy task (no latency, ball visible, sizes 1-5, a free foot touching the ball allowed, sole flat but not level, full thigh range). Every robustness feature after it came by a warm start, about 18 in a row (`mm2` → … → `kb1` → `fl1`). Cold starts with everything at once never mounted: `bl2` (blind at once, [mount_latency](../mount_latency/mount_latency_log.md)), `st2` and `st3` ([mount_straight](../mount_straight/mount_straight_log.md)). Here one cold start walks the same path, so the result is reproducible from scratch, and it keeps the thigh (yaw) joint at ±0.6 rad from the start (the chain's mount twisted it to −117 to −135°).

## Design

All terms, sensors and inputs exist from step 0 (built from `piplus_ball_mountmin_env_cfg` with blind-on-ball + kinematic ball, latency, one-foot sensors, free-leg and straight terms, the gate); a stage changes only weights, parameters and ranges, so the network never changes. No start hold, no mount deadline. Bounded actions with the chain's runner (entropy 0.003, 64 steps, gamma 0.995). Critic input `curriculum_stage` (stage / 8).

| Stage | Adds (cumulative) | From |
|---|---|---|
| 0 mount | `mm1` recipe: the 13 `MountMin` terms at the `mm1-min-clip` weights (`single_leg_stance` 8, clearance 0.03 m, `com_over_ball` 2, `action_rate` −0.01, `upright` 2 std 0.22), ball sizes 1-5 (fixed per env), ball visible, latency 0-15 ms, actor noise 0.25×, encoder bias ±0.03 seen by the actor | `mm1` |
| 1 clear6 | clearance 0.06 m | `mm2` |
| 2 clear10 | clearance 0.10 m, `single_leg_stance` 16 | `mm3` |
| 3 smooth | `action_excess` −4, `base_yaw_rate` −0.1, `torque_saturation` −2, `torque_demand` −0.5, `action_rate` −0.02, `com_over_ball` 0, `upright` std 0.5 | `mm4`-`ms2`, `bob2` |
| 4 robust | ball sizes 3-5 (resampled per reset from here on, mass following), latency ramped 0-15 → 10-40 ms over 300 iterations | `lat1`, `lat2-big` |
| 5 posture | free foot off the ball (`free_off_ball`), level support sole (`flat="level"`, `sole_flat` level), `ankle_roll_excess` −20 | `of2`, `lv1` |
| 6 noise | actor noise 0.25 → 1× over 500 iterations, `upright` 6 | `nr1` |
| 7 blind | on-ball latch on (0.42 m, 45°): the ball input becomes the kinematic estimate | `kb1` |
| 8 freeleg | free leg out of `posture`, `free_foot_ball_clear` 2, `support_hip_vel` −0.01, `support_foot_heading` 1 (all gated on the real stance) | `fl1`, `st2` |

**Gate:** at each reset, the share of the finished episodes' steps in the target stance (`single_leg_stance` episode sum / (weight × dt × length)) feeds a running mean over ~2000 episodes. A stage advances when the mean passes its threshold (0.5 for stages 0-3, 0.4 for 4-7) and the stage has run at least 300 iterations; the running mean restarts at 0 in the new stage. Logged as `Curriculum/stage/{stage,stance_share,lag_max,noise_scale}`. A resume sets the stage with `--env.curriculum.stage.params.start_stage N` (it is not in the checkpoint).

**Decisions:** `com_over_ball` at 2 in stages 0-2 as in `mm1-min-clip` (`mm1-nocom` showed it is not needed; off from stage 3 as in the chain). `support_foot_heading` only in stage 8: the ±0.6 rad thigh limit already keeps the foot within ~34° of the torso. The encoder bias and its actor reading are on from stage 0 (an input meaning that must not change mid-run). Play/eval config = stage 8, static (no curriculum term; eval and play set lag and noise themselves).

**Judgement:** if stage 0 has not advanced by ~3000 iterations, the straight-leg mount is the blocker, not the curriculum: rerun with the full thigh range.

**Tap stop (`-NoTap`, `notap=True`; added Nov 7 2026).** `cur4c` taps: the free sole touches the floor 14.8 times per 10 s and a stance lasts 0.36 s. The task adds two terms to the `-Excess` reward set. Observations, network and actions stay the same, so the `cur4c` actor and critic load. Both terms exist from stage 0 at weight 0. They get their weight in stage 8, through the curriculum term `notap_curriculum`, which uses the ramp of stage 8 (500 iterations, slow start). `stage_curriculum` stays unchanged, so the fingerprints of the old tasks stay equal. A warm start with `start_stage 8` restarts the stage clock: all stage-8 terms, the old ones too, ramp in again from 0.

| Term | Weight | Value per step |
|---|---|---|
| `free_sole_floor` | −8 | 1 while the robot is latched on the ball, one foot is on the ball (level sole), and the lowest corner of the other sole is within 1 cm of the floor (the tap test of `eval_mountbalance.py --gait`). Else 0 |
| `stance_duration` | +16 | min(length of the current unbroken stance / 2 s, 1). The stance is the live `single_leg_stance` term with its stage params. The count `env._stance_len` restarts at each episode start and at each step without stance |

Why these weights:
- `free_sole_floor` −8 is half of the `single_leg_stance` weight (16). A tap already loses the stance reward of its steps. The penalty makes each second of floor contact cost 24 instead of 16. A larger weight is risky: the task has no termination penalty, so a penalty that the policy cannot avoid makes ending the episode attractive. Strong penalties that came in at once also collapsed `cur2` and `cur3` at stage 3.
- `stance_duration` +16 equals the stance weight: an unbroken stance of 2 s or more doubles the stance reward. The bonus grows with the square of the stance length (about L²/4 for a stance of L seconds below the cap), so one long stance pays more than several short ones of the same total length. The cap of 2 s is about 4× the longest `cur4c` stances (p90 0.46 s), so the gradient exists over the whole range that the policy can reach now.
- Both weights are constants in `piplus_ball_mount_curriculum_cfg.py` (`TAP_PENALTY`, `STANCE_BONUS`, `STANCE_BONUS_CAP_S`). The curriculum sets the weights at each step, so a command line override of `--env.rewards.<term>.weight` does not hold: change the constants for a new run.

Check (a one-off script, not kept): forcing each stage sets the expected weights, params, ranges, lag, noise and latch in the running managers; the ball is resampled from size 3 up on reset in stage 4; the advance rule moves at share 0.6 after 300 iterations, not before, not at 0.3; thigh range ±0.6.

## Variants (added after the first runs)

| Task | Change | Why |
|---|---|---|
| `-FullThigh` (`straight=False`) | MJCF thigh range (−2.77 to 2.45 rad); `support_foot_heading` kept (stage 8, a pull, not a limit) | `cur1` at ±0.6 rad parked with a foot on the ball top and never mounted, as `st1`/`st3` |
| `-Ramp` (`ramp=True`) | a stage's new weights and `upright` std ramp from the previous stage's values over `STAGE_RAMP_ITERS` (stage 3: 2000, stage 5: 1000, others 500), slow start f = (100^p − 1)/99; a stage advances only after its ramp; action noise std capped at 1.0 (`MAX_ACTION_STD`, rsl_rl `std_range`) | `cur2` collapsed when stage 3 switched the smoothing penalties on at once |
| `-Excess` (`excess0=-1`) | `action_excess` −1 from stage 0 (ramped to −4 in stage 3) | `cur2`'s raw outputs had drifted past the ±7 clip (median \|a\| 45) in stages 0-2, where no penalty has a gradient; `cur3` (ramp only) collapsed the same way |
| `late_threshold` param | advance threshold of stages 6-7 (default 0.4) | `cur4` levelled off at ~0.35 in stage 6; resumed with 0.3 |

Also fixed during the runs: the stage clock starts at the first call with a non-zero step counter (a resume restores the counter after env creation), and the final stage has no threshold lookup (`cur4b` crashed at the 7 → 8 hand-over).

## Result (`cur4` → `cur4b` → `cur4c`, task `-FullThigh-Ramp-Excess`)

**The curriculum works from a cold start: one run reached the final stage** (iterations 0-24450; stages 0 → 4566, 1 → 5254, 2 → 6268, 3 → 8267, 4 → 8766, 5 → 9765, 6 → 12800 (resumed), 7 → 21605, 8 to 24450). Stage 3 (the smoothing penalties, where `cur2`/`cur3` collapsed) passed without a dip in reward; the raw outputs stayed inside the clip (median |a| ~1). Sweep of the end-of-stage-7 policy (`cur4b` `model_21600`, blind on the ball, sizes 3-5, noise on):

- **Survival 96 / 98 / 93 / 68%** at 10 / 20 / 30 / 40 ms latency, 90% with 10-40 ms jitter, 94% with soft motors (best of all policies; `fl1` 96 / 97 / 92 / 65%, soft motors 61%); on the ball after 0.6 s.
- **Posture:** base tilt **6°** (`fl1` 14.5°, `nr1-up6` 28.5°); **support foot −10° from the torso heading, thigh +26°, without any thigh limit** (every warm-started policy from `mm1` to `fl1`: 120-160° twist).
- **Held-stance success 0%** (the stance in ≥ 90% of the last 3 s): the policy does not hold the stance, it **taps**. Free foot always behind the support foot (14.5 cm back, 5 cm out), CoM ~5 cm behind the ball centre, the free sole touches the floor ~1.5 times per second, stance segments median 0.36 s (p90 0.46 s). The same or faster without noise and latency (0.28 s): a learned strategy, not forced by the sensing or delay. A tap only costs the stance reward of those steps; a fall ends the episode.
- Stage 8 (free-leg terms) changed nothing in ~2900 iterations: stance share flat at ~0.31, `free_foot_ball_clear` ~0.45 of 2.

## Next steps

1. **Stop the tap (implemented and run Nov 7 2026 as `cur5-notap` and `cur5b-notap`, see the run log; the taps did not stop because `free_sole_floor` is active in 1.5% of the on-ball steps; task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap`, terms in Design):** warm-start from `cur4c` `model_24450` (or `cur4b` `model_21600`) at stage 8 with (a) a penalty on free-sole floor contact while latched on the ball, and (b) a stance-duration bonus that grows with the length of the current stance (so one long stance beats many short ones). Reward-only change: same inputs and network, actor and critic both load. Measure with `scripts/eval_mountbalance.py --gait` (taps per 10 s, stance segment length, free-foot placement) and the held-stance success of the sweep.
2. **Only if the free foot stays behind:** a placement term for the free foot to the side or front of the support foot. Without the tap incentive the rear placement may change by itself.
3. **Physical limit check:** if stances stay short with tapping penalized, test whether a still stance on a size 3-5 ball is possible with the 5036 ankle torque (e.g. a drop spawn onto the ball with the same rewards, or a higher ankle torque limit as a probe).
4. **Real thigh range:** the policy uses +26° (p90 +42°); check this against the Pi Plus thigh joint before deploying.
5. **Robot node:** the kinematic ball estimate (`kin_ball_radius`, ankle poses from tf), latch tilt 0.79 rad, no start hold in this task (the node's hold is for the `-Hold` tasks only).
