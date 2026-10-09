# Experiment docs

Here you will find the docs for the experiments.

## Experiments

| Experiment | Task IDs | Docs |
|---|---|---|
| upright | `Mjlab-Piplus-Upright` | [upright](upright/upright.md) |
| noninertial | `Mjlab-Piplus-NonInertial` | [noninertial](noninertial/noninertial.md) |
| platform | `Mjlab-Piplus-Platform` | [platform](platform/platform.md) |
| locomotion | `Mjlab-Piplus-Locomotion[-Arms]` | [locomotion](locomotion/locomotion.md) |
| ball | `Mjlab-Piplus-Ball[-Arms]` | [ball](ball/ball.md) |
| ball_small | `Mjlab-Piplus-Ball-Small[-Turf]` | [ball_small](ball_small/ball_small.md) |
| mount_feasibility | `Mjlab-Piplus-Ball-Mount[-Flat]` | [spec](mount_feasibility/mount_feasibility.md), [log](mount_feasibility/mount_feasibility_log.md) |
| ball_size1 | `Mjlab-Piplus-Ball-Mount-Size1[-Flat]`, `Mjlab-Piplus-Ball-Balance-Size1` | [spec](ball_size1/ball_size1.md), [log](ball_size1/ball_size1_log.md) |
| ball_mount | `Mjlab-Piplus-Ball-MountBalance[-MixN][-Clip]` | [spec](ball_mount/ball_mount.md), [log](ball_mount/ball_mount_log.md) |
| ball_mount_curriculum | `Mjlab-Piplus-Ball-MountCurr-{Release,ReleasePot,Pot,Locked}` | [spec](ball_mount_curriculum/ball_mount_curriculum.md), [log](ball_mount_curriculum/ball_mount_curriculum_log.md) |
| ball_mount_rewards | `Mjlab-Piplus-Ball-MountMin[-Clip\|-Locked]` | [spec](ball_mount_rewards/ball_mount_rewards.md), [log](ball_mount_rewards/ball_mount_rewards_log.md) |
| balance_from_mount | `Mjlab-Piplus-Ball-BalanceFromMount[-Mix50]` | [spec](balance_from_mount/balance_from_mount.md), [log](balance_from_mount/balance_from_mount_log.md) |
| balance_deploy | audit and retraining of the balance policy for the real robot | [spec](balance_deploy/balance_deploy.md), [log](balance_deploy/balance_deploy_log.md) |
| balance_smooth | `Mjlab-Piplus-Ball-BalanceFromMount-Clip` | [spec](balance_smooth/balance_smooth.md), [log](balance_smooth/balance_smooth_log.md) |
| collision_check | (robot model; `scripts/check_collisions.py`) | [findings](collision_check/collision_check.md) |
| mount_single | `Mjlab-Piplus-Ball-MountMin-Clip` (one policy) | [spec](mount_single/mount_single.md), [log](mount_single/mount_single_log.md) |
| mount_latency | `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency[-Big[-OneFoot[-Level[-NoiseRamp[-Latch45-Hold[-Deadline/-KinBall[-FreeLeg]]]]]]]` | [spec](mount_latency/mount_latency.md), [log](mount_latency/mount_latency_log.md) |
| mount_straight | `...-KinBall-FreeLeg-Straight[-Gate-Deadline]` (cold start, thigh limited) | [spec](mount_straight/mount_straight.md), [log](mount_straight/mount_straight_log.md) |
| mount_curriculum | `Mjlab-Piplus-Ball-MountMin-Curriculum[-FullThigh[-Ramp[-Excess]]]` (one cold start through staged mount and robustness stages; thigh ±0.6 rad, -FullThigh: full range) | [spec](mount_curriculum/mount_curriculum.md), [log](mount_curriculum/mount_curriculum_log.md) |
| deploy_mount | export of the ball-mount policy for the Bit-Bots stack, robustness sweep | [plan](deploy_mount/deploy_mount.md), [log](deploy_mount/deploy_mount_log.md) |
| locomotion_s2r, ball_perception, ball_approach, ball_chain | placeholders for the ball chain | [locomotion_s2r](locomotion_s2r/locomotion_s2r.md), [ball_perception](ball_perception/ball_perception.md), [ball_approach](ball_approach/ball_approach.md), [ball_chain](ball_chain/ball_chain.md) |

## Conventions (shared by all experiments)

- **Launch:** `bash scripts/train.sh <task> --name <run-name> [overrides]` (wandb project = task id, default 4096 envs). Parallel runs: stagger launches by ~20 s.
- **Logs / checkpoints:** `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/model_N.pt`. The experiment dir comes from the task (see the task table in `AGENTS.md`).
- **Config actually used:** `logs/rsl_rl/<experiment>/<run>/params/env.yaml` (reward weights, events) and `agent.yaml`.
- **Resume:** `--agent.resume True --agent.load-run <run dir name> --agent.load-checkpoint model_N.pt --agent.max-iterations <additional>`. The iteration counter continues from N.
- **Reward overrides on the command line:** `--env.rewards.<term>.weight <value>`.
- **Metrics quoted** are the last logged iteration (`Mean reward`, `Mean episode length`, `Episode_Reward/*` = reward-weighted per-episode terms; max about 4 for `foot_on_ball` and `single_leg_stance`, 2 for `sole_flat`). They are training numbers, with pushes and noise on. The log's episode length only counts finished episodes, so it is low right after a resume.
- **Eval:** `uv run python scripts/eval_mountbalance.py --checkpoint <model.pt>` (success by spawn mode and ball size; `--probe` posture and saturation, `--audit` action size against the clip and the joint limits, `--gait` free-foot placement, floor taps, stance length and foot heading). **Play:** `uv run mjx --task <task> play [--run <prefix>] [--ckpt N]`; MountBalance has a Spawn mode dropdown.
- **Scripts:** only what training, evaluation and deployment need: `train.sh`, `eval_mountbalance.py`, `export_mount.py`, `sim2sim_onnx.py`, `make_padded_init.py`. One-off checks and probes are not kept as files (put a lasting diagnostic into `eval_mountbalance.py` as a flag instead).
- **Cleanup (Nov 6 2026):** removed `audit_policy.py` (now `eval_mountbalance.py --audit`), `check_collisions.py`, `eval_handoff.py`, `record_mount_states.py`, `play_chain.py`, `mount_cem.py`, `replay_cem.py` (in git history; the two-policy chain is tagged `two-policy-chain`). The probe and run scripts under `logs/probes/` that older logs refer to were never committed and are gone; their results are in the logs. `logs/rsl_rl` keeps only the last checkpoint of each run, so checkpoints named in older logs (other than a run's last) no longer exist. `logs/mount_states/` is gone: `BalanceFromMount` needs `scripts/record_mount_states.py` (git history) to record its states again.
- **Ball sizes:** size 5 = radius 0.11 m, 0.43 kg; size 1 = radius 0.07 m, 0.14 kg (assumed mini ball, check against the real one).
