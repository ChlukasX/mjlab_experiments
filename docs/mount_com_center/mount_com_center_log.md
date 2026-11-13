# mount_com_center run log

Design: [mount_com_center.md](mount_com_center.md). Run ID prefix `cc`. Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter`.

Start of all runs: a copy of `af1-actfit` `model_27449.pt` (SHA-256 `1397028a0461fa0b…`, byte-identical to the file in the `af1-actfit` run directory) as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter/2026-11-12_00-00-00_init-from-af1-27449`.

## `cc0-smoke`

Smoke run, 40 iterations (27449 → 27489), warm start from `af1` `model_27449`, stage 8, 64 environments, tensorboard (no wandb). The loop runs, no error. Last iteration: mean reward 265.6, episode length 792, `Curriculum/com_gate/com_gate` 0.12, `Curriculum/com_gate/com_weight` 0.035 (ramp), `Episode_Reward/com_center` 0.0056, `Episode_Reward/com_still` 0.0041.

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter --env.scene.num-envs 64 --agent.logger tensorboard --agent.run-name cc0-smoke --agent.max-iterations 40 --agent.resume True --agent.load-run 2026-11-12_00-00-00_init-from-af1-27449 --agent.load-checkpoint model_27449.pt --env.curriculum.stage.params.start_stage 8
```

## `cc1-comcenter`

Warm start from `af1-actfit` `model_27449`, stage 8, 4000 iterations (27449 → 31449), seed 42, 4096 envs. Recipe of `af1-actfit`; only the two CoM rewards and the stance gate differ. Wandb `g9luigv2`, run directory `2026-11-12_16-27-21_cc1-comcenter`, started Nov 12 2026 16:27 in the tmux session `mjlab-train-cc1` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-com-center`, uncommitted (the diff is in the run directory, `git/`). The GPU was shared with an unrelated training run (`booster_mjlab`, 4096 environments): about 7 s per iteration at the start.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter --name cc1-comcenter --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-12_00-00-00_init-from-af1-27449 --agent.load-checkpoint model_27449.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 4000 --agent.seed 42
```

**Outcome:** finished 4000 iterations (27449 → 31449, last checkpoint `model_31448`). Stable: no NaN, no collapse. The gate reached its last level (0.04 m) at about iteration 29850. Mean of 10 iterations at each point:

| Iteration | Gate (m) | Mean reward | Episode length | Stance share | `com_center` | `com_still` | `single_leg_stance` | `stance_duration` | `free_foot_contact` |
|---|---|---|---|---|---|---|---|---|---|
| 27849 | 0.12 | 304.7 | 828 | 0.47 | 0.64 | 0.41 | 6.68 | 0.22 | −0.62 |
| 28249 | 0.10 | 342.8 | 807 | 0.47 | 2.19 | 1.23 | 6.56 | 0.62 | −1.59 |
| 28649 | 0.06 | 303.1 | 760 | 0.39 | 2.51 | 1.13 | 5.26 | 0.47 | −1.50 |
| 29049 | 0.05 | 306.5 | 778 | 0.36 | 2.63 | 1.19 | 4.88 | 0.45 | −1.46 |
| 29849 | 0.04 | 313.0 | 790 | 0.33 | 2.95 | 1.24 | 4.66 | 0.44 | −1.37 |
| 30649 | 0.04 | 327.7 | 799 | 0.35 | 2.99 | 1.23 | 4.91 | 0.50 | −1.28 |
| 31448 | 0.04 | 355.8 | 828 | 0.39 | 3.20 | 1.29 | 5.47 | 0.65 | −1.22 |

The stance share is measured with the gate of its row, so the rows do not compare directly. At the last level the share, the reward, and `stance_duration` still rise at the end: the run has not converged.

**Evaluation.** Same command as the gap check of `af1` (mount spawn, play config of `-ActFit`, 512 envs, seed 0, stance test with the 0.12 m limit). The `af1` row is from the log of [mount_actuator_fit](../mount_actuator_fit/mount_actuator_fit_log.md):

| Policy | Survived | Success (held stance) | Stance share (on ball) | Floor taps per 10 s | Stance segment median | Single support |
|---|---|---|---|---|---|---|
| `af1` `model_27449` | 98% | 0% | 69% | 12.4 | 0.48 s | 86% |
| `cc1` `model_31448` | 96% | 4% | 75% | 8.9 | 0.54 s | 93% |

More numbers of `cc1`: CoM − ball centre x −1.2 cm (heading frame; `af1` on the drop spawn −4.2 cm), mean CoM-to-ball distance 0.027 m in the last 3 s, stance segments p90 1.02 s, free foot behind the support foot in 100% of the stance steps (median −22.1 cm, +14.2 cm to the side), base yaw rate +1.26 rad/s (all envs counter-clockwise), `r_hip_pitch` at the force limit in 55% of the steps, base tilt 11.6°.

Reading: the CoM moved over the ball centre, and the taps fell by 28%. The success is above 0% for the first time in this task family. The gait is still a tap gait: the free foot stays behind, and a stance lasts about 0.5 s. Next: resume at the last level, because the run has not converged.

```
uv run python scripts/eval_policy.py --task Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit --checkpoint logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter/2026-11-12_16-27-21_cc1-comcenter/model_31448.pt --num-envs 512 --seed 0 --gait --probe
```

## `cc0-smoke-resume`

Smoke run of the resume parameters of `com_gate`, 20 iterations (31448 → 31468), from `cc1-comcenter` `model_31448`, 64 environments, tensorboard (no wandb). The loop runs, no error. From the first iteration: `Curriculum/com_gate/com_gate` 0.04, `Curriculum/com_gate/com_weight` 8.0, `Curriculum/notap/free_foot_contact` −8.0, `Curriculum/leg_gap/leg_gap_weight` −4.0 (no ramp).

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter --env.scene.num-envs 64 --agent.logger tensorboard --agent.run-name cc0-smoke-resume --agent.max-iterations 20 --agent.resume True --agent.load-run 2026-11-12_16-27-21_cc1-comcenter --agent.load-checkpoint model_31448.pt --env.curriculum.stage.params.start_stage 8 --env.curriculum.com-gate.params.start-level 5 --env.curriculum.com-gate.params.skip-ramp True
```

## `cc1b`

Resume of `cc1-comcenter` `model_31448` at the last gate level (0.04 m) without the stage-8 ramp (`start_level 5`, `skip_ramp True`), 3000 iterations (31448 → 34448), seed 42, 4096 envs. Wandb `7t6c5zmc`, run directory `2026-11-13_15-13-34_cc1b`, started Nov 13 2026 15:13 in the tmux session `mjlab-train-cc1b` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-com-center`, uncommitted (the diff is in the run directory, `git/`). The code differs from `cc1` only in the two resume parameters of `com_gate`. The GPU was shared with an unrelated training run (`booster_mjlab`): about 7 s per iteration at the start.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter --name cc1b --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-12_16-27-21_cc1-comcenter --agent.load-checkpoint model_31448.pt --env.curriculum.stage.params.start_stage 8 --env.curriculum.com-gate.params.start-level 5 --env.curriculum.com-gate.params.skip-ramp True --agent.max-iterations 3000 --agent.seed 42
```

**Outcome:** finished 3000 iterations (31448 → 34448, last checkpoint `model_34447`). Stable: no NaN. The gate stayed at 0.04 m and the weights were full from the first iteration. Mean of 20 iterations at each point:

| Iteration | Mean reward | Episode length | Stance share (0.04 m gate) | `com_center` | `com_still` | `single_leg_stance` | `stance_duration` | `free_foot_contact` | `time_out` / iteration | `base_too_low` / iteration |
|---|---|---|---|---|---|---|---|---|---|---|
| 31948 | 346.6 | 793 | 0.39 | 3.17 | 1.25 | 5.52 | 0.66 | −1.16 | 3.32 | 1.64 |
| 32948 | 354.9 | 754 | 0.43 | 3.13 | 1.15 | 5.89 | 0.84 | −0.84 | 3.14 | 1.97 |
| 33948 | 352.7 | 707 | 0.47 | 3.06 | 1.08 | 6.22 | 1.08 | −0.57 | 3.02 | 2.38 |
| 34447 | 347.1 | 700 | 0.49 | 3.12 | 1.10 | 6.47 | 1.12 | −0.51 | 2.92 | 2.17 |

The mean reward is flat. The stance terms rise and the tap penalty falls, but the episodes get shorter: the falls (`base_too_low`) rise.

**Evaluation.** Same command as for `cc1` (mount spawn, play config of `-ActFit`, 512 envs, seed 0), for three checkpoints of this run. The `af1` and `cc1` rows are from the sections above:

| Policy | Survived | Success (held stance) | Stance share (on ball) | Floor taps per 10 s | Stance segment median / p90 | CoM − ball centre x | Base yaw rate | Base tilt | `r_hip_pitch` at the force limit |
|---|---|---|---|---|---|---|---|---|---|
| `af1` `model_27449` | 98% | 0% | 69% | 12.4 | 0.48 s / — | — | — | — | — |
| `cc1` `model_31448` | 96% | 4% | 75% | 8.9 | 0.54 s / 1.02 s | −1.2 cm | +1.26 rad/s | 11.6° | 55% |
| `cc1b` `model_32250` | 95% | 17% | 80% | 6.9 | 0.44 s / 1.30 s | −0.7 cm | +1.10 rad/s | 12.9° | 57% |
| `cc1b` `model_33650` | 90% | 53% | 90% | 2.3 | 0.28 s / 1.74 s | −0.3 cm | +0.60 rad/s | 13.9° | 59% |
| `cc1b` `model_34447` | 77% | 62% | 93% | 1.1 | 0.26 s / 1.96 s | −0.1 cm | +0.12 rad/s | 14.9° | 61% |

By ball size, `model_34447`: size 3 success 58%, survived 74%; size 4-5 success 67%, survived 82%. The free foot is behind the support foot in 100% of the stance steps (median −24.2 cm, +16.0 cm to the side) and in the air in 98.8% of the single-support steps.

Reading: the tap gait is gone. The floor taps fell from 12.4 to 1.1 per 10 s, and the held-stance success rose from 0% to 62%. The cost is the survival: 98% → 77%. Without the tap, a lost balance ends in a fall, and a fall has no cost of its own in this task (only the lost future reward). The trade continues to the end of the run: the last 800 iterations gave 9 points of success for 13 points of survival. The base turn went away without a term for it (41% of the envs turn clockwise at the end). The stance segment median is short (0.26 s) at a stance share of 93%; the cause of the short breaks is not measured.

```
uv run python scripts/eval_policy.py --task Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit --checkpoint logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit_comcenter/2026-11-13_15-13-34_cc1b/model_34447.pt --num-envs 512 --seed 0 --gait --probe
```

## `cc0-smoke-fall`

Smoke run of the reward `early_end` at −50, 20 iterations (33650 → 33670), from `cc1b` `model_33650`, 64 environments, tensorboard (no wandb). The loop runs, no error. Last iteration: `Episode_Reward/early_end` −1.67, `Curriculum/com_gate/com_gate` 0.04, `Curriculum/com_gate/com_weight` 8.0.

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter --env.scene.num-envs 64 --agent.logger tensorboard --agent.run-name cc0-smoke-fall --agent.max-iterations 20 --agent.resume True --agent.load-run 2026-11-13_15-13-34_cc1b --agent.load-checkpoint model_33650.pt --env.curriculum.stage.params.start_stage 8 --env.curriculum.com-gate.params.start-level 5 --env.curriculum.com-gate.params.skip-ramp True --env.rewards.early-end.weight -50
```

## `cc2-fall50`

Warm start from `cc1b` `model_33650` (survived 90%, success 53%), last gate level (0.04 m), no stage-8 ramp, 3000 iterations (33650 → 36650), seed 42, 4096 envs. The one setting that differs from `cc1b`: `early_end` −50 per termination other than `time_out`. Wandb `th6csvw8`, run directory `2026-11-13_21-53-14_cc2-fall50`, started Nov 13 2026 21:53 in the tmux session `mjlab-train-cc2` (the command ran inside the session, without `--tmux`). Code: branch `feat/mount-com-center`, uncommitted (the diff is in the run directory, `git/`). The GPU was shared with an unrelated training run (`booster_mjlab`): about 7 s per iteration at the start.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit-ComCenter --name cc2-fall50 --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-13_15-13-34_cc1b --agent.load-checkpoint model_33650.pt --env.curriculum.stage.params.start_stage 8 --env.curriculum.com-gate.params.start-level 5 --env.curriculum.com-gate.params.skip-ramp True --env.rewards.early-end.weight -50 --agent.max-iterations 3000 --agent.seed 42
```

**Outcome:** running.
