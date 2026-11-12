# mount_actuator_fit run log

Design: [mount_actuator_fit.md](mount_actuator_fit.md). Run ID prefix `af`. Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit`.

## `af0-smoke`

Smoke run, 5 iterations, cold start, 64 environments, tensorboard (no wandb). The loop runs. Stage 0, `Curriculum/lag_cap/lag_max` 3. No error.

```
uv run train Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit --env.scene.num-envs 64 --agent.logger tensorboard --agent.run-name af0-smoke --agent.max-iterations 5
```

Start (`af1` and later warm starts): a copy of `cur4c` `model_24450.pt` (SHA-256 `83ec928c…`, byte-identical to the other init copies) as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit/2026-11-12_00-00-00_init-from-cur4c-24450`.

## `af1-actfit`

Warm start from `cur4c` `model_24450`, stage 8, 3000 iterations (24450 → 27450). Same recipe and seed as `sg1-leggap`; only the actuator dynamics, the PD gain scales, and the lag cap (10-25 ms) differ. Wandb `i3ggq5j3`, run directory `2026-11-12_09-03-48_af1-actfit`, started Nov 12 2026 09:03 in the tmux session `mjlab-train-af1` (the command ran inside the session, without `--tmux`). The GPU was shared with an unrelated training run (`booster_mjlab`, 4096 environments).

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit --name af1-actfit --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-12_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

**Outcome:** finished 3000 iterations (24450 → 27450, last checkpoint `model_27449`, finished Nov 12 2026 about 14:55). Stable: no NaN, no divergence. Last training iteration: mean reward 285.1, episode length 805, stance share 0.479 (`sg1-leggap`: 206, 756, 0.39; each run trained under its own dynamics, so these numbers do not compare the policies).

**Gap check.** `scripts/eval_policy.py --num-envs 512 --seed 0 --gait`, mount spawn, play configs (actuator lag: `-LegGap` 10-40 ms, `-ActFit` 10-25 ms). Same command for each row, with `--task` and `--checkpoint` changed:

```
uv run python scripts/eval_policy.py --task Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ActFit --checkpoint logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_actfit/2026-11-12_09-03-48_af1-actfit/model_27449.pt --num-envs 512 --seed 0 --gait
```

| Policy | Play config | Survived | Stance share (on ball) | Floor taps per 10 s | Stance segment median | Single support |
|---|---|---|---|---|---|---|
| `cur4c` `model_24450` | `-LegGap` | 94% | 57% | 13.4 | 0.42 s | 81% |
| `cur4c` `model_24450` | `-ActFit` | 70% | 53% | 15.6 | 0.32 s | 76% |
| `af1` `model_27449` | `-ActFit` | 98% | 69% | 12.4 | 0.48 s | 86% |
| `af1` `model_27449` | `-LegGap` | 63% | 52% | 14.6 | 0.30 s | 76% |

Success (held stance) is 0% in all four rows. The `cur4c` policy loses 24 points of survival on the fitted dynamics, although the lag is lower. `af1` recovers them and holds the stance longer than `cur4c` on its own task. `af1` loses 35 points on the `-LegGap` model (high damping, lag up to 40 ms): the policy is fit to the new model and does not cover the old one.
