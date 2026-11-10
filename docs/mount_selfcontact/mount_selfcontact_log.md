# mount_selfcontact run log

Design: [mount_selfcontact.md](mount_selfcontact.md). Run ID prefix `sg`. Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap`.

Start (all runs): a copy of `cur4c` `model_24450.pt` (SHA-256 `83ec928c…`, byte-identical to the other init copies) as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap/2026-11-09_00-00-00_init-from-cur4c-24450`.

## `leggap-smoke`

Smoke run, 40 iterations (24450 → 24490), wandb `z8fsj1nb`, run directory `2026-11-09_23-35-27_leggap-smoke`. The loop runs; `Episode_Reward/leg_gap` and `Curriculum/leg_gap/leg_gap_weight` are logged; the weight ramps from 0 (−0.018 at iteration 24489, the slow start of the ramp). No error.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap --name leggap-smoke --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-09_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `sg1-leggap`

Warm start from `cur4c` `model_24450`, stage 8, 3000 iterations (24450 → 27450). Wandb `bcpkh32q`, run directory `2026-11-09_23-38-45_sg1-leggap`, started Nov 9 2026 23:38 in the tmux session `mjlab-train-sg1` (the command ran inside the session, without `--tmux`). Expected duration: 3 h (3.5 s per iteration in `cur6` to `cur8`).

**Outcome:** finished 3000 iterations (24450 → 27450, last checkpoint `model_27449`). Stable: no NaN, no divergence. The `leg_gap` weight reached −4 at about iteration 25000. The `leg_gap` term peaked at −0.67 per episode and fell to −0.19 at the end. Mean reward 206, episode length 756, stance share 0.39 (`cur6`: 204, 731, 0.39). Self-contact probe (`scripts/eval_selfcontact.py`, 256 envs, seed 3, `--every 5`) and gait eval (`scripts/eval_mountbalance.py --gait --probe --audit`, 1024 envs) of `model_27449`, against `cur6`:

| Metric | `sg1-leggap` | `cur6-notap` |
|---|---|---|
| Foot-foot contact in phase A (`r_ankle_roll`\|`l_ankle_roll`) | 0.7% | 36.8% |
| Median foot-foot distance in phase A | 50.7 mm | 4.3 mm |
| Foot-foot distance below 2 cm in phase A | 9% | 68% |
| Feet and calves in contact, F > 10 N (steps, t ≥ 3 s) | 2.1% | 34.5% |
| `torso`\|`r_thigh` contact share | 31.1% | 30.1% |
| `torso`\|`l_wrist` contact share | 4.5% | 0.8% |
| Survived / taps per 10 s / stance share | 94% / 12.8 / 62% | 94% / 12.3 / 63% |
| Support foot heading against the torso | −14° | −20° |
| `r_hip_pitch` at the force limit | 42% | 44% |
| Held-stance success | 0% | 0% |

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap --name sg1-leggap --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-09_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

To read after the run:

- Training curves: `leg_gap` (weighted reward per second), `stance_share`, `single_leg_stance`, `free_foot_contact`, mean episode length, against `cur6` (same warm start, same task without the term).
- `scripts/eval_selfcontact.py` on the last checkpoint, with the five policies of the measurement as reference (section Re-run of the design doc).
- `scripts/eval_mountbalance.py --gait --probe --audit` on the last checkpoint (taps, stance share, survival, held-stance success, `r_hip_pitch` at the force limit).
