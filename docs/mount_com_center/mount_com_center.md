# mount_com_center: dense reward for a centred, still centre of mass on the ball

Status: finished (Nov 14 2026). Selected policy: `cc2-fall50` `model_36649` (protected checkpoint). The history of all experiments that led to it is in [mount_lineage](../mount_lineage/mount_lineage.md).
Code: branch `feat/mount-com-center` (off `main` `9d42400`), first commit `77e3f88`. Files: `tasks/ball_mount_min/piplus_ball_mount_comcenter_cfg.py`, `tests/test_comcenter.py`, `scripts/eval_policy.py` (flags `--mount-fraction`, `--no-foot-floor`).
Tasks: `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter`.
Log: [mount_com_center_log.md](mount_com_center_log.md). Run ID prefix: `cc`.

## Why

No policy holds a still stance on the ball. The best policy (`af1-actfit`, [mount_actuator_fit](../mount_actuator_fit/mount_actuator_fit.md)) taps the floor with the free foot 12 to 13 times per 10 s, and a stance lasts about 0.5 s. The tap penalty (`-NoTap`) and the tap termination ([mount_tap_limit](../mount_tap_limit/mount_tap_limit.md)) did not stop the taps. A tighter tap limit changed taps into falls.

The reward does not pay for balance. Three facts:

- The stance reward `single_leg_stance` is 0 or 1. Its test of the centre of mass (CoM) passes up to 0.12 m from the ball centre (`COM_OVER_BALL`), on a ball of radius 0.08 to 0.11 m.
- The dense term `com_over_ball` has weight 0 from stage 3 on. No term pays a CoM nearer to the ball centre.
- `af1` holds the CoM 0.050 m (median) from the ball centre, 4 to 5 cm behind it in the torso heading frame. The free foot is behind the support foot in 100% of the stance steps.

On a free ball, only a CoM over the ball centre is an equilibrium. At 0.045 m the weight of the robot (128 N) gives a torque of about 6 Nm that tips the robot. The tap gait is a repeated fall to the rear that the free foot catches. A penalty on the tap removes the catch, not the fall.

`af1` earns 0.64 of 16 per second from `stance_duration` and 6.6 of 16 from `single_leg_stance`. A still stance pays about 50 per second against about 20 now. The incentive exists, but no term gives a slope from the tap gait to the still stance.

**Zero-shot probe (Nov 12 2026).** `scripts/eval_policy.py --mount-fraction 0 --no-foot-floor` drops the robot onto the ball and lets the feet pass through the floor. Task `-ActFit`, 512 envs, seed 0:

| Policy | Floor for the feet | Survived | Floor taps per 10 s |
|---|---|---|---|
| `af1` `model_27449` | on | 15% | 13.1 |
| `af1` `model_27449` | off | 0% | — |
| `cur4c` `model_24450` | on | 13% | 16.0 |
| `cur4c` `model_24450` | off | 0% | — |

The drop spawn is outside the training distribution (survival 15% with the floor, 98% from the mount spawn), so the probe does not show if a still stance is possible. It shows that both policies need the floor.

## Design

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter` (flag `ComCenter`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter`). The factory `piplus_ball_mount_comcenter_env_cfg` calls the `-ActFit` factory and adds two reward terms and one curriculum term. The model, the observations, the actions, the events, the terminations, and the network are the same, so the `af1` actor and critic load without padding. The actor reads none of the new terms.

| Term | Kind | Weight | Value per step |
|---|---|---|---|
| `com_center` | reward | +8 | With a level sole on the ball: 1 − d / 0.10, at least 0. d is the horizontal distance (m) from the CoM to the ball centre |
| `com_still` | reward | +4 | With a level sole on the ball: 1 − v / 0.5, at least 0. v is the horizontal speed (m/s) of the CoM relative to the ball centre, as a difference over one control step |
| `early_end` | reward | 0 (off) | 1 per termination other than `time_out`. See "Cost of a fall" |
| `com_gate` | curriculum | — | Ramps the two weights in with stage 8. Steps the `com_max` of `single_leg_stance` down: 0.12, 0.10, 0.08, 0.06, 0.05, 0.04 m |

**Ranges.** Measured on `af1` (play config of `-ActFit`, 512 envs, on the ball, after 3 s): distance median 0.050 m, p90 0.089 m; relative speed median 0.21 m/s, p90 0.44 m/s. Each range (`COM_CENTER_RANGE` 0.10 m, `COM_SPEED_RANGE` 0.5 m/s) is near the p95 of `af1`. Both terms are linear, so the slope is the same down to 0.

**Weights.** `com_center` +8 is half of the `single_leg_stance` weight. `af1` earns about 4 of 8 at the start. `com_still` +4: `af1` earns about 2 of 4. Both terms are positive and act only with a foot on the ball, so they do not pay for leaving the ball. The weights follow the ramp of stage 8 (500 iterations), as `leg_gap` does.

**Gate.** `single_leg_stance` and, through it, `stance_duration` use the gate. The gate does not change `support_on_ball` and the gated single support of the free-leg terms (0.12 m). Share of the on-ball steps of `af1` in the stance at each level: 81% at 0.12 m, 79% at 0.10 m, 71% at 0.08 m, 58% at 0.06 m, 47% at 0.05 m, 32% at 0.04 m. A level goes down when it has had 300 iterations (level 0: the 500 iterations of the stage-8 ramp) and the running stance share of the stage curriculum (`Curriculum/stage/stance_share`) is at least 0.4 (`COM_GATE_SHARE`). The share is measured with the gate of the current level, so each level needs a better policy. No termination is added: a level that the policy cannot reach only stops the curriculum.

**Resume.** The curriculum state is not in a checkpoint. A resume with only `start_stage 8` repeats the stage-8 ramp (the stage-8 weights of all curriculum terms restart at the stage-7 values for 500 iterations) and the gate from 0.12 m. `com_gate` has two parameters for the resume of a run that is past the ramp: `start_level` (first gate level, 0 to 5; default 0) and `skip_ramp` (default `False`; `True` moves the clock of the final stage back by the ramp length, so the weights are full from the second step on). The defaults give the behavior of `cc1`.

**Cost of a fall (after `cc1b`).** `cc1b` stopped the taps (12.4 → 1.1 per 10 s, held-stance success 62%), but the survival fell from 98% to 77% (log). Without the tap, a lost balance ends in a fall, and a fall cost only the reward of the rest of the episode. The reward term `early_end` gives a fall a cost of its own: its weight is the reward per termination other than `time_out` (`fell_over`, `base_too_low`, `body_on_floor`, `ball_escaped`), one event, not per second. The weight in the task is 0 (off), so the task behaves as in `cc1` and `cc1b`. A run sets the weight on the command line: `--env.rewards.early-end.weight -50`. −50 is the value of `EARLY_END_PENALTY` in [mount_tap_limit](../mount_tap_limit/mount_tap_limit.md). For scale: the reward is about 0.35 per step, so with gamma 0.995 the future reward that a fall loses is worth about 70. The curriculum does not set this weight, so the command line value holds.

**Play.** Full weights and the last gate level (0.04 m). `scripts/eval_policy.py` has its own stance test (0.12 m), so the evaluation numbers compare with the earlier runs.

**Not in this task.** The tap termination. A term for the place of the free foot. A critic input for the gate level (the value target changes at each level; add it with `scripts/make_padded_init.py` if the value loss rises at a level change).

Wandb: `Curriculum/com_gate/com_gate` (m), `Curriculum/com_gate/com_weight`, `Episode_Reward/com_center`, `Episode_Reward/com_still`, `Episode_Reward/early_end`.

**How the result is measured.** `scripts/eval_policy.py --num-envs 512 --seed 0 --gait --probe` on the play config of `-ActFit` (the gap check of `af1`): survival, floor taps per 10 s, stance segment median, CoM − ball centre, held-stance success.

## Result

The tap gait is gone. The selected policy `cc2-fall50` `model_36649` holds the single-leg stance on the ball in 75% of the episodes and survives in 82% (play config of `-ActFit`, mount spawn, ball sizes 3-5). The free foot taps the floor 0.4 times per 10 s (`af1`: 12.4). The robot either holds the stance or falls. The numbers of each run are in the [log](mount_com_center_log.md).

| Step | Change | Effect |
|---|---|---|
| `cc1` | `com_center`, `com_still` and the stance gate (0.12 → 0.04 m), warm start from `af1` | CoM over the ball centre, taps down by about a quarter, first held stances |
| `cc1b` | resume of `cc1` at the last gate level without the ramp | taps nearly gone, held-stance success up, survival down: a lost balance ends in a fall |
| `cc2-fall50` | `early_end` −50 per fall, warm start from `cc1b` `model_33650` | the survival stops falling, the held-stance success rises further |

What made the difference: the dense, linear CoM terms. They give a slope from the tap gait to a still stance. The tap penalty and the tap termination of the earlier experiments had no such slope. The fall cost only stopped the trade of survival for stance; it did not remove the falls.

The user accepted the result after play (Nov 14 2026): a survival near 82% is acceptable for this task.

## Next steps

- Export `cc2-fall50` `model_36649` and test it in the Bit-Bots stack. The work continues in [deploy_mount](../deploy_mount/deploy_mount.md).
- Measure the cause of the falls (termination, time, hip torque, CoM offset, ball state before each fall). `r_hip_pitch` is at its force limit in about 60% of the steps.
- The free foot hangs behind the support foot in every stance step. No term sets its place.
- The stance segments are short (median 0.26 s) at a stance share of 95%. The cause of the short breaks is not measured.

## Re-run

Copy the init checkpoint first (`af1-actfit` `model_27449.pt`) to `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter/2026-11-12_00-00-00_init-from-af1-27449/`. The launch commands and the run IDs are in the [log](mount_com_center_log.md).

Runs, in order: `cc1-comcenter` (warm start from `af1`), `cc1b` (resume of `cc1` `model_31448`), `cc2-fall50` (warm start from `cc1b` `model_33650`, `--env.rewards.early-end.weight -50`). Each log section has the full launch command. The selected checkpoint is `cc2-fall50` `model_36649`. Code: commits `77e3f88` (eval flags) and `734aaa9` (task) on `feat/mount-com-center`.
