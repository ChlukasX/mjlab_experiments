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
| locomotion_s2r, ball_perception, ball_approach, ball_chain | placeholders for the ball chain | [locomotion_s2r](locomotion_s2r/locomotion_s2r.md), [ball_perception](ball_perception/ball_perception.md), [ball_approach](ball_approach/ball_approach.md), [ball_chain](ball_chain/ball_chain.md) |

## Conventions (shared by all experiments)

- **Launch:** `bash scripts/train.sh <task> --name <run-name> [overrides]` (wandb project = task id, default 4096 envs). Parallel runs: stagger launches by ~20 s.
- **Logs / checkpoints:** `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/model_N.pt`. The experiment dir comes from the task (see the task table in `AGENTS.md`).
- **Config actually used:** `logs/rsl_rl/<experiment>/<run>/params/env.yaml` (reward weights, events) and `agent.yaml`.
- **Resume:** `--agent.resume True --agent.load-run <run dir name> --agent.load-checkpoint model_N.pt --agent.max-iterations <additional>`. The iteration counter continues from N.
- **Reward overrides on the command line:** `--env.rewards.<term>.weight <value>`.
- **Metrics quoted** are the last logged iteration (`Mean reward`, `Mean episode length`, `Episode_Reward/*` = reward-weighted per-episode terms; max about 4 for `foot_on_ball` and `single_leg_stance`, 2 for `sole_flat`). They are training numbers, with pushes and noise on. The log's episode length only counts finished episodes, so it is low right after a resume.
- **Eval:** `uv run python scripts/eval_mountbalance.py --checkpoint <model.pt>` (success by spawn mode and ball size). **Play:** `uv run mjx --task <task> play [--run <prefix>] [--ckpt N]`; MountBalance has a Spawn mode dropdown.
- **Ball sizes:** size 5 = radius 0.11 m, 0.43 kg; size 1 = radius 0.07 m, 0.14 kg (assumed mini ball, check against the real one).
