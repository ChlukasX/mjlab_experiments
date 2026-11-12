# mount_tap_limit: end the episode on free-foot taps after the first mount

Status: active (Nov 10 2026).
Code: branch `feat/mount-tap-limit` (off `main` `e4f36d3`). Files: `tasks/ball_mount_min/piplus_ball_mount_taplimit_cfg.py`, `tests/test_taplimit.py`.
Tasks: `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit`.
Log: [mount_tap_limit_log.md](mount_tap_limit_log.md). Run ID prefix: `tl`.

## Why

The best policies tap the floor with the free foot about 13 times per 10 s, and a stance lasts about 0.4 s (`sg1-leggap`, `im1-imunoise`, see [mount_imu_noise](../mount_imu_noise/mount_imu_noise.md)). The held-stance success is 0%. The tap penalty of `-NoTap` (`free_foot_contact`, −8 per second of contact) costs less than 1 per tap, against an episode return of about 190. A tap is therefore cheap, and the policy uses the floor to correct its balance.

Request of the user: after the first mount, a tap of the free foot on the floor ends the episode. A curriculum first allows 3 taps, then 2, then 1, then none. The policy must get the balance on the support foot right at the first mount.

## Design

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-TapLimit` (flag `TapLimit`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_taplimit`). The factory `piplus_ball_mount_taplimit_env_cfg` calls the `-LegGap` factory and adds three terms. Base: `-LegGap`, the best policy so far (`-ImuNoise` was not better).

**Mount.** The first step of the episode in a gated single support (`single_support` with `gated=True`): one level sole on the ball, the CoM (center of mass) within `COM_OVER_BALL` (0.12 m) of the ball center, the free sole more than `GATE_FREE_CLEARANCE` (0.03 m) above the floor. The flag stays set until the episode ends.

**Tap.** After the mount, a step in which a sole corner comes within `FLOOR_TOUCH_CLEARANCE` (0.01 m) of the floor, and no sole corner was within it in the step before. This is the tap test of `eval_mountbalance.py --gait`. After the mount the support foot is on the ball, so the foot near the floor is the free foot. A contact of the free foot with the ball is not a tap (the reward `free_foot_contact` still costs it).

Terms (training only; the play config equals `-LegGap`, so play and eval keep the episode to its end):

| Term | Kind | What it does |
|---|---|---|
| `tap_limit` | termination | Ends the episode when the tap count after the mount is above rate / 10 × (t + `TAP_GRACE_S`), with the tap rate limit in taps per 10 s, t the time since the mount, and `TAP_GRACE_S` 3 s. A terminal state, not a time-out (the value is not bootstrapped). `tl1` to `tl3` used a fixed tap budget per episode instead of a rate |
| `early_end` | reward | `EARLY_END_PENALTY` (−50) once, in the step of each termination other than `time_out` (`tap_limit`, `fell_over`, `base_too_low`, `body_on_floor`, `ball_escaped`), while the tap limit acts. 0 in the warm-up |
| `tap_rate` | curriculum | Sets the tap rate limit, see below (`tl1` to `tl3`: `tap_budget`, the tap budget) |

**Why every early end costs the same (after `tl1`).** The first version had the penalty `tap_end` on `tap_limit` only, and a budget level went down when fewer than 25% of the finished episodes ended by `tap_limit`. In `tl1` the policy learned to fall before its taps ran out: a fall had no penalty, and it also pushed the `tap_limit` share below the threshold. The budget went down to 0 on falls, not on fewer taps (log section `tl1-taplimit`). Now a fall costs as much as a tap, and a level goes down on survival, which a fall cannot raise.

**Tap limit curriculum.** Before stage 8 of the stage curriculum there is no limit (`NO_LIMIT`; `NO_BUDGET` in `tl1` to `tl3`), so a cold start learns the mount first. Stage 8 starts with a warm-up without the termination, because the stage-8 rewards ramp in over 500 iterations and because the smoke test with budget 3 from the first step cut 85% of the episodes to about 2.8 s (`taplimit-smoke`). The warm-up ends when both conditions are true:

- It has had `TAP_WARMUP_MIN_ITERS` (500) iterations.
- The running survival is at least `TAP_WARMUP_SURVIVAL` (0.7). Survival is the share of finished episodes that end by `time_out`.

The warm-up ends after `TAP_WARMUP_MAX_ITERS` (1500) iterations in any case. Then the tap rate limit starts at 14 and goes 14 → 12 → 10 → 8 → 6 → 4 → 2 → 0 taps per 10 s (`TAP_RATES`). At 0 the first tap after the mount ends the episode. History: `tl1` and `tl2` had a fixed budget per episode of 6 → 4 → 3 → 2 → 1 → 0 taps, `tl3` 10 → 8 → 6 → 4 → 2 → 1 → 0. A level goes down when it has had `TAP_MIN_ITERS` (300) iterations and the running survival is at least `TAP_ADVANCE_SURVIVAL` (0.3; the warm-up of `tl1` without a tap limit held about 0.5). All running shares are exponential means over `TAP_EMA_EPISODES` (2000) finished episodes. The survival starts at 0 and the `tap_limit` share at 1 for each new level. Wandb logs `Curriculum/tap_rate/tap_rate` (−1 in the warm-up; `Curriculum/tap_budget/tap_budget` in `tl1` to `tl3`), `.../tap_end_share`, `.../survival`, and `.../mount_share` (share of finished episodes that reached the mount; a fall of this share shows a policy that avoids the mount to avoid the termination).

**Penalty size.** −50 is a first guess. The termination itself costs the rest of the episode (up to about 190). The fixed penalty also acts near the end of an episode, where little future reward is left.

**Softer start (after `tl2`).** In `tl2` the budget stayed at 6: no episode reached the time-out in 1480 iterations, and the episodes stayed at about 4.6 s. Without the limit the policy taps about 26 times in a 20 s episode, so a budget of 10 ends an episode at about 8 s and gives the survival a chance to rise above 0. `tl3` starts at 10.

**Tap rate instead of a budget (after `tl3`).** In `tl3` the budget stayed at 10 as well: episodes stayed at about 6 s, and no episode reached the time-out in about 1160 iterations. A time-out with a fixed budget of b taps needs a tap rate of b / 2 per 10 s or less (20 s episodes). The policy taps about 13 per 10 s, so every budget up to about 20 ends every episode before the time-out, and the survival stays 0. A small improvement of the tap rate gives no signal. A rate limit fits the current policy at the first level (14 per 10 s), so a part of the episodes reaches the time-out, and each level asks for a lower rate by 2 per 10 s. Grace time `TAP_GRACE_S` (3 s): without it one tap right after the mount would end the episode at any rate. `tl4` uses the rate limit.

**What stays the same.** The model, the observations, the actions, the network, and all other rewards, events, and curriculum terms of `-LegGap`. The actor and the critic do not see the limit or the tap count. The critic therefore cannot tell how many taps are left; a critic input for it would change the critic size (padding with `scripts/make_padded_init.py`). This is a possible next step if the value loss stays high.

**Live check (Nov 10 2026).** `sg1-leggap` `model_27449` on the new task, 256 envs, stage 8, budget 3, 1000 steps: 99% of the finished episodes ended by `tap_limit`. The current policies tap far too often for the first level; the curriculum stays at budget 3 until the share falls below 0.25.

**Run.** Warm start from `cur4c` `model_24450` (copy as `init-from-cur4c-24450` in the log group of the new task), `--env.curriculum.stage.params.start_stage 8`, seed 42, 4096 envs. This is the recipe of `sg1-leggap`, so the two runs are an A/B pair. The warm-up (500 to 1500 iterations) and 8 rate levels (at least 2400 iterations) need at least 2900 iterations; plan 4000 and resume if the budget has not reached 0. Smoke test first with `--agent.logger tensorboard` (no wandb).

**How the result is measured.** `scripts/eval_mountbalance.py --obs-noise --gait --probe --audit` (1024 envs, task `-LegGap`, so the episode does not end on a tap) for each `tl` run and `sg1-leggap`: survival at 20 s, taps per 10 s, stance segment median, stance held in the last 3 s, held-stance success.
