# mount_imu_noise: lower actor noise on the IMU inputs

Status: active (Nov 10 2026).
Code: branch `feat/mount-imu-noise` (off `feat/mount-selfcontact`). Files: `tasks/ball_mount_min/piplus_ball_mount_imunoise_cfg.py`, `tests/test_imunoise.py`.
Tasks: `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise`.
Log: [mount_imu_noise_log.md](mount_imu_noise_log.md). Run ID prefix: `im`.

## Why

The IMU (inertial measurement unit) of the HighTorque Pi Plus is good. The training noise on the two IMU inputs of the actor is larger than the IMU needs. The ranges come from the velocity task of mjlab, not from a measurement on the Pi Plus. Hypothesis of the user: less noise on the IMU gives a better long-term balance on the ball.

Why the hypothesis is plausible. The balance on the ball needs a fast and fine estimate of the tilt and the tilt rate of the torso. The actor reads `projected_gravity` and `base_ang_vel` with 5 steps of history. Noise hides small tilt changes and costs the actor margin for its feedback gain. Why it may not help: in the best policy so far (`sg1-leggap`) the taps (12.8 per 10 s) and the 0% held-stance success come from the gait, not from the noise (they are the same in `cur6` to `cur8`, which all have the same noise). The result of this experiment must show which case holds.

Noise in the actor of the `-LegGap` task (uniform, new draw each control step, from stage 6 at full size; `set_noise_scale`):

| Actor term | Source | Range (±) | Standard deviation | Changed here |
|---|---|---|---|---|
| `projected_gravity` | IMU | 0.05 | 0.029 | yes: 0.025 (std 0.0144) |
| `base_ang_vel` | IMU | 0.2 rad/s | 0.115 rad/s | yes: 0.1 rad/s (std 0.058 rad/s) |
| `joint_pos` | encoder | 0.01 rad | 0.006 rad | no |
| `joint_vel` | encoder | 1.5 rad/s | 0.87 rad/s | no |
| `ball_pos` | ball input, until on the ball | 0.02 m | 0.012 m | no |

The simulation has no IMU bias, no IMU drift, and no IMU delay (the actor terms have a delay of 0 steps). The IMU error in the simulation is white noise only. The encoder bias (±0.03 rad) and the loop latency (10-40 ms) stay as they are.

## Design

Task `Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise` (flag `ImuNoise`, experiment `piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_imunoise`). It builds on `-LegGap` (the best policy so far). The factory `piplus_ball_mount_imunoise_env_cfg` calls the `-LegGap` factory and then replaces the noise of the two IMU terms. The module `piplus_ball_mount_imunoise_cfg.py` holds the amounts as constants `IMU_NOISE_GRAVITY` (0.025) and `IMU_NOISE_ANG_VEL` (0.1).

**Amount: half of the base value, a first guess.** The user gave no number. Half is a clear step that a run can show, and it is still far above the white noise of a typical IMU at 50 Hz. The amount is not derived from the data of the real IMU. It is changed after the datasheet or a recording of the real IMU is known (section Sim2real).

**What stays the same.** The model, the observations (names, sizes, order, 5 steps of history), the actions, the network, the rewards, the curriculum, the events, the terminations, and the critic. The test `tests/test_imunoise.py` compares the config with `-LegGap` in training and in play. The critic group holds the same term objects as the actor group in the base module. The factory therefore replaces the actor term by a copy, so the critic terms keep the old ranges (the critic reads no noise).

**Noise curriculum.** `set_noise_scale` stores the range of a term when it first runs and scales from that range. The new range is therefore the full range (scale 1.0) from stage 6. A live check (8 envs, stage 8, 3 steps) showed the ranges 0.025 and 0.1 in the observation manager of the new task and 0.05 and 0.2 in `-LegGap`.

**Old noise in play and eval.** `play=True` turns the actor noise off. `scripts/eval_mountbalance.py` keeps it off unless `--obs-noise` is set. With `--obs-noise` it uses the ranges of the task at scale 1.0 (it does not call `set_noise_scale`). `scripts/eval_selfcontact.py` always has the noise off. A policy is therefore evaluated under noise of a given size by choosing the task ID: the network is the same in both tasks.

**Run.** Warm start from `cur4c` `model_24450` (a byte-identical copy as `2026-11-10_00-00-00_init-from-cur4c-24450` in the log group of the new task), `--env.curriculum.stage.params.start_stage 8`, 3000 iterations, seed 42, 4096 envs. This is the recipe of `sg1-leggap`. Only the noise differs, so the two runs are an A/B pair. One seed gives no estimate of the spread between seeds. A difference of a few points of survival or stance share between the two runs is not a result by itself (the runs `cur6` to `cur8` and `sg1-leggap` show the spread of the same recipe: survival 93-95%, stance share 61-64%).

**How the hypothesis is measured.** After the run, with `scripts/eval_mountbalance.py --gait --probe --audit` (1024 envs) and `scripts/eval_selfcontact.py` (256 envs, seed 3), for `im1-imunoise`, `sg1-leggap`, and `cur6`:

- Each policy under the old noise (task `-LegGap`, `--obs-noise`) and under the low noise (task `-ImuNoise`, `--obs-noise`). This separates "the policy works better with a cleaner IMU" from "the policy was trained for a cleaner IMU".
- Survival at 20 s, stance share, stance length, taps per 10 s, base tilt, and the offset between the center of mass and the ball.

## Sim2real

A policy that is trained with less noise reacts to smaller signals. If the real IMU is noisier than the training noise, the policy can show more jitter in the joint targets or lose the balance. The training noise is a safety margin. Lower it only to the level that the real IMU needs.

Numbers needed from the real robot (not known to this repository, not assumed here):

- The gyro noise density and the bias stability from the datasheet of the IMU, and the output rate and the filter bandwidth that the driver uses.
- The attitude error of the filter that produces the projected gravity (the vector of gravity in the base frame). Its error is not white: it has a lag and a bias during accelerations. The simulation has none of these.
- A recording of the real IMU at rest (a few minutes) and during a balance on the ball. From it: the standard deviation of `base_ang_vel` and `projected_gravity` at 50 Hz.

A uniform range of ±a has a standard deviation of a divided by 1.73. Compare the measured standard deviation with 0.058 rad/s and 0.0144 in this task. Keep the training noise above the measured value.

## Result

Open. The run `im1-imunoise` is in progress (log).

## Next steps

- Evaluate the run as described in the section Design.
- Set the noise amounts from the data of the real IMU.
- Add an IMU bias and a small IMU delay to the simulation if the real IMU has them (a new task variant, not an edit of this one).

## Re-run

Training (about 3 h for 3000 iterations at 4096 envs). Copy the init checkpoint first:

```
mkdir -p logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_imunoise/2026-11-10_00-00-00_init-from-cur4c-24450
cp logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap/2026-11-09_00-00-00_init-from-cur4c-24450/model_24450.pt logs/rsl_rl/piplus_ball_mountmin_curriculum_fullthigh_ramp_excess_notap_leggap_imunoise/2026-11-10_00-00-00_init-from-cur4c-24450/
bash scripts/train.sh Mjlab-Piplus-Ball-MountMin-Curriculum-FullThigh-Ramp-Excess-NoTap-LegGap-ImuNoise --name im1-imunoise --num-envs 4096 --env.scene.env-spacing 1.0 --agent.resume True --agent.load-run 2026-11-10_00-00-00_init-from-cur4c-24450 --agent.load-checkpoint model_24450.pt --env.curriculum.stage.params.start_stage 8 --agent.max-iterations 3000 --agent.seed 42
```

Evaluation (one policy under one noise level; the task ID sets the noise):

```
uv run python scripts/eval_mountbalance.py --task <task> --checkpoint <model.pt> --obs-noise --gait --probe --audit --num-envs 1024
uv run python scripts/eval_selfcontact.py --task <task> --checkpoint <model.pt> --label <label> --num-envs 256 --seed 3 --every 5 --dist-envs 256 --mesh-every 10 --mesh-envs 64 --out <dir>/<label>
```

Commit: the first commit of the branch `feat/mount-imu-noise` after `feat/mount-selfcontact` adds the task. The commit that adds the result follows when the run ends.
