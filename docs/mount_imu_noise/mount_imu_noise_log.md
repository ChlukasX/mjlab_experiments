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

**Outcome:** in progress.

```
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise --name im1-imunoise --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

To read after the run:

- Training curves against `sg1-leggap` (same warm start, same seed): mean reward, episode length, `stance_share`, `single_leg_stance`, `free_foot_contact`, `leg_gap`.
- Evaluation under the old noise and the low noise for `im1-imunoise`, `sg1-leggap`, and `cur6` (section Design of the design doc).
