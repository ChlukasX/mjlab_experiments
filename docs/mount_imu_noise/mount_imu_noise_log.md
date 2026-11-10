# mount_imu_noise run log

Design: [mount_imu_noise.md](mount_imu_noise.md). Run ID prefix `im`. Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise`, log group `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_imunoise`.

Start (all runs): a copy of `cur4c` `model_24450.pt` (SHA-256 `83ec928c…`, byte-identical to the other init copies) as `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_imunoise/2026-11-10_00-00-00_init-from-cur4c-24450`.

## `imunoise-smoke`

Smoke run, 40 iterations (24450 → 24490), wandb `d7x75rh9`, run directory `2026-11-10_09-43-22_imunoise-smoke`. The loop runs. Stage 8, noise scale 1.0, `leg_gap` weight ramps from 0 (−0.017 at iteration 24489). Mean reward 197.8, episode length 718. No error.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise --name imunoise-smoke --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 40 --agent.seed 42
```

## `im1-imunoise`

Warm start from `cur4c` `model_24450`, stage 8, 3000 iterations (24450 → 27450). Same recipe and seed as `sg1-leggap`; only the IMU noise differs (`projected_gravity` ±0.025, `base_ang_vel` ±0.1 rad/s). Wandb `2lbvyb01`, run directory `2026-11-10_09-46-37_im1-imunoise`, started Nov 10 2026 09:46 in the tmux session `mjlab-train-im1` (the command ran inside the session, without `--tmux`). Expected duration: 3 h.

**Outcome:** finished 3000 iterations (24450 → 27450, last checkpoint `model_27449`, finished Nov 10 2026 13:05). Stable: no NaN, no divergence. Last training iteration: mean reward 188.5, episode length 691, stance share 0.403 (`sg1-leggap`: 206, 756, 0.39; each run trained under its own noise, so these numbers do not compare the policies).

Evaluation of `model_27449` of each policy with `scripts/eval_mountbalance.py --obs-noise --gait --probe --audit` (1024 envs, default seed, one 20 s episode per env, play config, noise on at scale 1.0, pushes on). The task ID sets the noise: `-LegGap` is the old noise (±0.05, ±0.2 rad/s), `-ImuNoise` is the low noise (±0.025, ±0.1 rad/s). `cur6-notap` uses `-NoTap` for the old noise (same actor inputs). The networks are equal in all tasks. Probe values are means over the last 3 s of the envs that survived.

| Policy | Eval noise | Survived | Stance share | Stance held (last 3 s) | Taps per 10 s | Stance segment median | Base tilt | CoM − ball x |
|---|---|---|---|---|---|---|---|---|
| `cur6-notap` | old | 90% | 59% | 60% | 12.9 | 0.42 s | 10.1° | −4.5 cm |
| `cur6-notap` | low | 92% | 60% | 61% | 12.7 | 0.44 s | 10.0° | −4.4 cm |
| `sg1-leggap` | old | 88% | 58% | 59% | 13.5 | 0.38 s | 10.9° | −4.7 cm |
| `sg1-leggap` | low | 91% | 59% | 60% | 13.3 | 0.40 s | 10.7° | −4.6 cm |
| `im1-imunoise` | old | 86% | 58% | 59% | 13.0 | 0.38 s | 11.8° | −4.3 cm |
| `im1-imunoise` | low | 88% | 60% | 61% | 12.7 | 0.42 s | 11.7° | −4.2 cm |

Success by the strict definition (stance held at least 50% of the last 3 s with the free foot off the ball) is 0% for all six rows, as before. The share of steps at the force limit of `r_hip_pitch` is 47-53% in all rows.

Reading (details in the design doc): the low noise raises survival by 2 to 3 points for every policy, whichever policy it is. `im1-imunoise` is not better than `sg1-leggap` or `cur6-notap` under either noise level. It survives 2 to 3 points less than `sg1-leggap`, and its base tilt is about 1° higher. Stance share, stance length, taps, and the CoM offset are equal within the spread of the other policies. One evaluation seed and one training seed each.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise --name im1-imunoise --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

Self-contact probe (`scripts/eval_selfcontact.py`, stage-8 play config, observation noise off, 256 envs, seed 3, `--every 5`; this script has no noise flag). `sg1-leggap` and `cur6-notap` are the results of the experiment `mount_selfcontact`, the same conditions:

| Metric | `cur6-notap` | `sg1-leggap` | `im1-imunoise` |
|---|---|---|---|
| Survived (noise off) | 93% | 94% | 93% |
| Foot-foot contact (`r_ankle_roll`\|`l_ankle_roll`) | 24.5% | 0.5% | 0.7% |
| `torso`\|`r_thigh` contact | 30.1% | 31.1% | 30.9% |
| `l_wrist`\|`head_pitch` contact | 7.5% | 3.4% | 12.5% |
| Taps per 10 s / stance share | 12.1 / 63% | 12.6 / 62% | 12.1 / 62% |
| Held-stance success | 0% | 0% | 0% |

The `leg_gap` effect stays (no foot-foot prop). The left wrist touches the head more often (12.5%, mean force 16 N). Gait and survival with the noise off equal the other two policies.

Evaluation command (one row of the table; `<task>` and `<model.pt>` as in the text above):

```
uv run python scripts/eval_mountbalance.py --task <task> --checkpoint <model.pt> --obs-noise --gait --probe --audit --num-envs 1024
```

Checkpoints: `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_imunoise/2026-11-10_09-46-37_im1-imunoise/model_27449.pt`, `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap/2026-11-09_23-38-45_sg1-leggap/model_27449.pt`, `logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap/2026-11-07_18-42-03_cur6-notap/model_27449.pt`.
