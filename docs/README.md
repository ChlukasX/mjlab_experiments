# Experiment docs

Here you will find the docs for the experiments. Each experiment has one directory with a design doc and, when it has runs, a run log. The index below is the one map from experiment to status, tasks, code, run IDs, and docs.

Commands (train, resume, evaluate, play) are in [AGENTS.md](../AGENTS.md#commands). The list of tasks is in [AGENTS.md](../AGENTS.md#registered-tasks). Every training run is listed in [runs_index.md](runs_index.md) (generated).

## Experiments

`Status` is one of `planned`, `active`, `finished`, `parked`, `superseded`. `Code package` is a directory under `src/mjlab_piplus/tasks/` (or `robot` for the robot model). `Run ID prefix` lists the prefixes of the run names that the experiment used; `—` means the runs have legacy names or no name.

| Experiment | Status | Task IDs | Code package | Run ID prefix | Docs |
|---|---|---|---|---|---|
| upright | finished | `Mjlab-Piplus-Upright` | `upright` | — | [upright](upright/upright.md) |
| noninertial | finished | `Mjlab-Piplus-NonInertial` | `noninertial` | — | [noninertial](noninertial/noninertial.md) |
| platform | finished | `Mjlab-Piplus-Platform` | `platform` | — | [platform](platform/platform.md) |
| locomotion | finished | `Mjlab-Piplus-Locomotion[-Arms]` | `locomotion`, `locomotion_arms` | — | [locomotion](locomotion/locomotion.md) |
| ball | finished | `Mjlab-Piplus-Ball[-Arms]` | `ball`, `ball_arms` | — | [ball](ball/ball.md) |
| ball_small | finished | `Mjlab-Piplus-Ball-Small[-Turf]` | `ball_small` | — | [ball_small](ball_small/ball_small.md) |
| mount_feasibility | superseded | `Mjlab-Piplus-Ball-Mount[-Flat]` | `ball_mount` | — | [spec](mount_feasibility/mount_feasibility.md), [log](mount_feasibility/mount_feasibility_log.md) |
| ball_size1 | superseded | `Mjlab-Piplus-Ball-Mount-Size1[-Flat]`, `Mjlab-Piplus-Ball-Balance-Size1` | `ball_mount` | — | [spec](ball_size1/ball_size1.md), [log](ball_size1/ball_size1_log.md) |
| ball_mount | finished | `Mjlab-Piplus-Ball-MountBalance[-MixN][-Clip]` | `ball_mount` | `mb` | [spec](ball_mount/ball_mount.md), [log](ball_mount/ball_mount_log.md) |
| mount_potential (formerly ball_mount_curriculum) | finished | `Mjlab-Piplus-Ball-MountCurr-{Release,ReleasePot,Pot,Locked}` | `ball_mount_curriculum` | `mc` | [spec](mount_potential/mount_potential.md), [log](mount_potential/mount_potential_log.md) |
| ball_mount_rewards | finished | `Mjlab-Piplus-Ball-MountMin[-Clip\|-Locked]` | `ball_mount_min` | `mm` | [spec](ball_mount_rewards/ball_mount_rewards.md), [log](ball_mount_rewards/ball_mount_rewards_log.md) |
| balance_from_mount | finished | `Mjlab-Piplus-Ball-BalanceFromMount[-Mix50]` | `ball_balance_from_mount` | `bfm` | [spec](balance_from_mount/balance_from_mount.md), [log](balance_from_mount/balance_from_mount_log.md) |
| balance_deploy | superseded | audit and retraining of the balance policy for the real robot | `ball_balance_from_mount` | — | [spec](balance_deploy/balance_deploy.md), [log](balance_deploy/balance_deploy_log.md) |
| balance_smooth | parked | `Mjlab-Piplus-Ball-BalanceFromMount-Clip` | `ball_balance_from_mount` | `bs` | [spec](balance_smooth/balance_smooth.md), [log](balance_smooth/balance_smooth_log.md) |
| collision_check | finished | (robot model; the checker `check_collisions.py` is in git history) | `robot` | — | [findings](collision_check/collision_check.md) |
| mount_single | superseded | `Mjlab-Piplus-Ball-MountMin-Clip` (one policy) | `ball_mount_min` | `ms`, `bob` | [spec](mount_single/mount_single.md), [log](mount_single/mount_single_log.md) |
| mount_latency | finished | `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency[-Big[-OneFoot[-Level[-NoiseRamp[-Latch45-Hold[-Deadline/-KinBall[-FreeLeg]]]]]]]` | `ball_mount_min` | `bl`, `lat`, `of`, `lv`, `nr`, `kb`, `fl` | [spec](mount_latency/mount_latency.md), [log](mount_latency/mount_latency_log.md) |
| mount_straight | finished | `Mjlab-Piplus-Ball-MountMin-BlindOnBall-…-KinBall-FreeLeg-Straight[-Gate-Deadline]` (cold start, thigh limited) | `ball_mount_min` | `st` | [spec](mount_straight/mount_straight.md), [log](mount_straight/mount_straight_log.md) |
| mount_curriculum | active | `Mjlab-Piplus-Ball-MountMin-Curriculum[-FullThigh[-Ramp[-Excess[-NoTap]]]]` (one cold start through staged mount and robustness stages; thigh ±0.6 rad, -FullThigh: full range) | `ball_mount_min` | `cur` | [spec](mount_curriculum/mount_curriculum.md), [log](mount_curriculum/mount_curriculum_log.md) |
| deploy_mount | parked | export of the ball-mount policy for the Bit-Bots stack, robustness sweep | — | — | [plan](deploy_mount/deploy_mount.md), [log](deploy_mount/deploy_mount_log.md) |
| locomotion_s2r | planned | — | — | — | [locomotion_s2r](locomotion_s2r/locomotion_s2r.md) |
| ball_perception | planned | — | — | — | [ball_perception](ball_perception/ball_perception.md) |
| ball_approach | planned | — | — | — | [ball_approach](ball_approach/ball_approach.md) |
| ball_chain | planned | none (the task `Mjlab-Piplus-Ball-Chain` was dropped; the two-policy chain is in tag `two-policy-chain`) | — | — | [ball_chain](ball_chain/ball_chain.md) |

## Conventions (shared by all experiments)

- **Branches named in the docs** are deleted. Their commits are on `main`, and the subject of each starts with the branch name without `feat/`; each doc gives the `git log --grep` command. A branch marked "planned" is not created yet.
- **Logs / checkpoints:** `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/model_N.pt`. The experiment dir comes from the task (`experiment_name` of its runner config; see the task table in `AGENTS.md`).
- **Config actually used:** `logs/rsl_rl/<experiment>/<run>/params/env.yaml` (reward weights, events) and `agent.yaml`. The code state at launch: `git/mjlab_experiments.diff` (commit, status, diff).
- **Metrics quoted** are the last logged iteration (`Mean reward`, `Mean episode length`, `Episode_Reward/*` = reward-weighted per-episode terms; max about 4 for `foot_on_ball` and `single_leg_stance`, 2 for `sole_flat`). They are training numbers, with pushes and noise on. The log's episode length only counts finished episodes, so it is low right after a resume.
- **Scripts:** only what training, evaluation, checks, and deployment need: `train.sh`, `eval_mountbalance.py`, `check_tasks.py`, `make_runs_index.py`, `make_mount_states.py`, `export_mount.py`, `sim2sim_onnx.py`, `make_padded_init.py`. One-off checks and probes are not kept as files (put a lasting diagnostic into `eval_mountbalance.py` as a flag instead). Removed scripts are in git history (all removed in commit `c6e4c30`; read one with `git show c6e4c30^:scripts/<name>.py`): `audit_policy.py` (now `eval_mountbalance.py --audit`), `check_collisions.py`, `eval_handoff.py`, `record_mount_states.py`, `play_chain.py`, `mount_cem.py`, `replay_cem.py`. (`record_mount_states.py` is back as `make_mount_states.py`.) The two-policy chain is tagged `two-policy-chain`.
- **Lost files:** the probe and run scripts under `logs/probes/` that older logs refer to were never committed and are gone; their results are in the logs, and the launch arguments of most runs are recovered in [runs_index.md](runs_index.md). `BalanceFromMount` needs the file `data/mount_states/mm4-dr-exc.pt` (not in git); `scripts/make_mount_states.py` makes it (see Re-run in [balance_from_mount](balance_from_mount/balance_from_mount.md)).
- **Checkpoints:** `logs/rsl_rl` keeps only the last checkpoint of each run, so checkpoints named in older logs (other than a run's last) are not there. Look in `wandb/run-*/files/` or in the wandb cloud.
- **Ball sizes:** size 5 = radius 0.11 m, 0.43 kg; size 1 = radius 0.07 m, 0.14 kg (assumed mini ball, check against the real one).
