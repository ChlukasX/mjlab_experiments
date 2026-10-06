# Experiment protocol: ball mount and balance

Chronological record of the training runs for the ball mount / balance work (Oct 2026). One row per run; add a row when a run is launched and fill in the result when it finishes. Design and findings are in [mount_feasibility.md](mount_feasibility.md) and [ball_mount.md](ball_mount.md); this file is the run-by-run log and the commands to reproduce them.

## Conventions

- **Launch:** `bash scripts/train.sh <task> --name <run-name> [overrides]` (wandb project = task id, default 4096 envs). Parallel runs: stagger launches by ~20 s.
- **Where:** `local` = the 4090 on this machine, `cl06` = the 5090 (shared home, so same code and logs).
- **Logs / checkpoints:** `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/model_N.pt`. The experiment dir comes from the task (see the task table in `AGENTS.md`).
- **Config actually used:** `logs/rsl_rl/<experiment>/<run>/params/env.yaml` (reward weights, events) and `agent.yaml`.
- **Resume:** `--agent.resume True --agent.load-run <run dir name> --agent.load-checkpoint model_N.pt --agent.max-iterations <additional>`. The iteration counter continues from N.
- **Reward overrides on the command line:** `--env.rewards.<term>.weight <value>`.
- **Metrics quoted** are the last logged iteration (`Mean reward`, `Mean episode length`, `Episode_Reward/*` = reward-weighted per-episode terms; max about 4 for `foot_on_ball` and `single_leg_stance`, 2 for `sole_flat`). They are training numbers, with pushes and noise on. The log's episode length only counts finished episodes, so it is low right after a resume.
- **Eval:** `uv run python scripts/eval_mountbalance.py --checkpoint <model.pt>` (success by spawn mode and ball size). **Play:** `uv run mjx --task <task> play [--run <prefix>] [--ckpt N]`; MountBalance has a Spawn mode dropdown.
- **Ball sizes:** size 5 = radius 0.11 m, 0.43 kg; size 1 = radius 0.07 m, 0.14 kg (assumed mini ball, check against the real one).

## 1. Mount feasibility: size 5 ball (branch `feat/mount-feasibility`)

Task `Mjlab-Piplus-Ball-Mount` (and `-Flat`): robot starts standing, ball 0.30 m ahead of the right foot, 10 s episodes. Spike runs v1 to v4 had wandb disabled.

| Run | Machine | Setup | Result | Takeaway |
|-----|---------|-------|--------|----------|
| `mount-spike-v1` | local, 2048 envs | From scratch, 400 it. Dense `foot_to_ball_top` reward. | Reward 22, episode 500/500, `foot_on_ball` 0 | Learns to stand, never steps on. |
| `mount-spike-v2-lift` | local, 2048 envs | Resumed v1 `model_399`, + `foot_lift` reward measured from 0.10 m. Stopped at 1600 it. | `foot_lift` 0.0000 throughout | Bug: standing ankle height is 0.050 m, so lifts below 5 cm paid nothing. |
| `mount-spike-v3-lift-fix` | local, 2048 envs | Resumed v2 `model_1600`, `STAND_FOOT_Z` = 0.05. To `model_4599`. | Mounting starts ~it 2000; peak reward 84 at it 2421, `foot_on_ball` ~3.0 (of 4); drifts to jerky control (reward 44, `action_rate` −2.8 at it 3756) | Mounting is feasible in sim. `single_leg_stance` was identical to `foot_on_ball` (Ball-Small check never needs a lifted foot). |
| `mount-spike-v4/v5-flat-sole` | local, 2048 envs | Resumed v3 `model_2400`, added flat-sole requirement and `sole_flat = exp(-tilt²/0.2²)`. v4 killed after 2 min; v5 (`r32i1cnm`) 3000 it. | v5: `foot_on_ball` (flat) 0, `foot_lift` 0.97 | Foot reaches the ball but the sole is rolled over. |
| `mount-flat-sole-scratch` | cl06, 4096 | Same flat shaping, from scratch, 5000 it (`7se9n1gp`). | Reward 40, `sole_flat` 0. Probe: median sole tilt 112°, 0 of 44672 samples under 17°. | `exp(-tilt²/0.2²)` ≈ 0 at that tilt: no gradient. |
| `mount-flat-sole-v2-scratch` | cl06, 4096 | `sole_flat = (1 + cos tilt)/2`, from scratch, 5000 it (`jm04uwmd`). | Reward 96, episode 483, `foot_on_ball` 3.72, `single_leg_stance` 3.57, `sole_flat` 1.88 | Flat sole learned from scratch. |
| `mount-balance-v1` | cl06, 4096 | Plain `Ball-Mount`, resumed v3 `model_2400`, 3000 it (`ioq6yazr`); free foot must clear the floor by 3 cm. | Reward 56, episode 437, `foot_on_ball` 3.37, `single_leg_stance` 3.25, `action_rate` −3.34 | Mount + balance without the flat requirement. Jerky. |
| `mount-flat-v1` | cl06, 4096 | `Ball-Mount-Flat`, from scratch, 5000 it (`w3mq6c0w`). | Reward 100, episode 487, `foot_on_ball` 3.74, `single_leg_stance` 3.61, `sole_flat` 1.89, `action_rate` −2.53 | Reproduces `v2-scratch`. Flat is better than plain on every metric; mean action change per step 0.47 vs 0.85. |
| `mount-smooth-v1` | cl06, 4096 | Plain `Ball-Mount`, from scratch, `action_rate` −0.05, `joint_vel` −0.015, 5000 it (`z2x62wty`). | Reward 25, `foot_on_ball` 0, `foot_lift` 0.39 | 5× smoothness from scratch never mounts. Smooth an existing policy instead. |

Measurements on the finished policies (probe scripts, not in the repo): on-ball ankle sits 0.049 m above the ball top with a flat sole (flat run) and 0.037 m (plain run, tilted sole).

## 2. Size 1 ball

| Run | Machine | Setup | Result | Takeaway |
|-----|---------|-------|--------|----------|
| `mount-size1-v1` | cl06, 4096 | `Ball-Mount-Size1`, from scratch, 5000 it (`iwc06dqm`). | Reward 58, episode 431, `foot_on_ball` 3.37, `single_leg_stance` 3.28, `action_rate` −2.94 | Mounts a size 1 ball. |
| `mount-size1-flat-v1` | cl06, 4096 | `Ball-Mount-Size1-Flat`, from scratch, 5000 it (`47ya5hv8`). | Reward 93, episode 484, `foot_on_ball` 3.65, `single_leg_stance` 3.51, `sole_flat` 1.88, `action_rate` −2.61 | |
| `mount-size1-flat-v2-cont` | local, 4096 | Resumed `size1-flat-v1` `model_4999`, 5000 more (to `model_9998`; `qt3bwrlm`). Rewards were still rising. | Reward 115, episode 496, `foot_on_ball` 3.83, `single_leg_stance` 3.78, `sole_flat` 1.94, `action_rate` −1.58 | Continuing helped: reward 93 → 115 and smoother actions. |
| `balance-size1-flat-v1` | local, 4096 | `Ball-Balance-Size1` (drop onto the size 1 ball, flat sole), from scratch, 10000 it (`20doj5z7`). Drop spawn: base 0.5 m above the ball top. | Reward 163, episode 950/1000, `foot_on_ball` 3.74, `single_leg_stance` 3.72, `sole_flat` 1.89, `action_rate` −1.91 | Drop-and-balance is learnable on a fixed size 1 ball even with the higher drop. |

## 3. One policy for mount + balance on a randomized ball (branch `feat/ball-mount-balance`)

Task `Mjlab-Piplus-Ball-MountBalance`: per reset 50% mount / 50% drop, ball radius 0.07 to 0.11 m, mass tied to radius, ball and ground friction 0.3 to 1.2, actor does not see the size, network 512-256-128, 20 s episodes, flat sole. Details in [ball_mount.md](ball_mount.md).

| Run | Machine | Setup | Result | Takeaway |
|-----|---------|-------|--------|----------|
| `mountbalance-mix50-v1` | local, 4096 | From scratch, 8000 it (`a620eevt`). Code before the changes below: drop spawn base 0.5 m above the ball top; 3 cm free-foot test. | Training: reward 112, episode 518/1000, `foot_on_ball` 2.06, `sole_flat` 1.04. Eval (1024 envs): **mount spawns 100% success on all sizes, drop spawns 0% (dead within ~2 s).** | Mount spawns reach a stance 0.26 s after the start and hold 20 s, but as a deep crouch: base 0.19 m above the ball top (upright is ~0.29 to 0.30 m) with the free foot only 4.5 cm off the floor, which passed the 3 cm test. Drops collapse after landing. |

Changes made after this run (commit `a69a457` and the working tree of the runs below): drop spawn lowered to the sole 3 cm above the ball top (base at 2r + 0.37 m); `stand_tall` (+2.0) and `free_foot_lift` (+1.0) rewards; free-foot clearance required for `single_leg_stance` raised from 3 cm to 10 cm (ankle above 0.15 m); Spawn mode dropdown in play.

Caveat on the drop height: the old drop was not a 0.5 m free fall. The base started 0.5 m above the ball top, which puts the sole about 0.16 m above its contact height (impact ~1.8 m/s). `balance-size1-flat-v1` learned that drop on a fixed ball. So the failed drops in `mix50-v1` come from the harder combination (randomized size and mass, size not observed, mixed spawns), and the lower drop is an easier curriculum choice, not a proven fix.

### `mb2-*`: reward shaping comparison (in progress)

Four runs launched in parallel on the local GPU, 4096 envs, 8000 it, from scratch, same code (`a69a457`), task `Mjlab-Piplus-Ball-MountBalance`, 50/50 mix. Started Oct 6 2026, ~2.2 s per iteration with four jobs.

| Run | wandb | Override relative to base | Question |
|-----|-------|---------------------------|----------|
| `mb2-base` | `euzp8n4n` | none (`stand_tall` 2.0, `free_foot_lift` 1.0, `action_rate` −0.01, `joint_vel` −0.005) | Do the new rewards fix the crouch and the drops? |
| `mb2-tall4` | `qavv1qgh` | `--env.rewards.stand_tall.weight 4.0 --env.rewards.free_foot_lift.weight 2.0` | Is stronger balance shaping better? |
| `mb2-nostand` | `44kjb2d7` | `--env.rewards.stand_tall.weight 0.0` | Is `stand_tall` needed, or is free-foot clearance enough? |
| `mb2-smooth` | `n5vkhzk2` | `--env.rewards.action_rate.weight -0.02 --env.rewards.joint_vel.weight -0.01` | Do milder smoothness penalties (2×) cost mounting? |

Launch commands (base shown; add the overrides above):

```bash
bash scripts/train.sh Mjlab-Piplus-Ball-MountBalance --name mb2-base
bash scripts/train.sh Mjlab-Piplus-Ball-MountBalance --name mb2-tall4 \
  --env.rewards.stand_tall.weight 4.0 --env.rewards.free_foot_lift.weight 2.0
```

Results (all four finished 8000 it; training numbers at the last iteration, eval = `scripts/eval_mountbalance.py`, 1024 envs, 20 s, success = stance with 10 cm free-foot clearance, flat sole):

| Run | Reward | Episode | `foot_on_ball` | `single_leg_stance` | `stand_tall` | `free_foot_lift` | `action_rate` | Eval success |
|-----|--------|---------|----------------|---------------------|--------------|------------------|---------------|--------------|
| `mb2-base` | 92 | 572 | 1.83 | 0.07 | 0.31 | 0.12 | −1.03 | 0% |
| `mb2-tall4` | 55 | 334 | 1.60 | 0.07 | 0.73 | 0.38 | −1.30 | 0% |
| `mb2-nostand` | 92 | 539 | 2.18 | 0.06 | n/a | 0.12 | −0.70 | 0% |
| `mb2-smooth` | 88 | 468 | 1.89 | 0.15 | 0.42 | 0.19 | −0.65 | 0% |

Eval detail (all four alike): mount spawns survive 99 to 100% but never reach the stance; drop spawns survive 0%.

**Diagnosis (probes on `mb2-base` `model_7999`, 256 envs):**
- **Mount spawns never put the weight on the ball.** The whole-robot centre of mass stays 0.25 to 0.26 m horizontally from the ball centre for the full episode, i.e. where it started (ball 0.26 to 0.30 m ahead). The robot stands on its left foot on the floor with the right foot propped on the ball top; `single_leg_stance` stays ~0. `foot_on_ball` still paid ~46% of its maximum for the prop, and nothing rewarded moving the centre of mass over the ball. `xy_centering` (distance from the env origin) even penalised moving toward it.
- **Drop spawns land fine, then topple sideways.** Centre of mass is over the ball at 0.22 s (offset 0.035 m, torso tilt 3°), then drifts to the robot's left (offset 0.13 m at 0.5 s, 0.31 m at 1.0 s, torso tilt 45°) and all die by ~1.5 s. Lateral balance on the ball was never learned, and the mount half of the batch dominated.
- The reward-shaping variants only changed how deep the crouch is, not the missing weight transfer: `stand_tall` / `free_foot_lift` assume the weight is already on the ball.

### Next: centre-of-mass rewards (`mb3-*`)

Changes to `Mjlab-Piplus-Ball-MountBalance` (not yet trained):

| Term | Weight | What |
|------|--------|------|
| `support_on_ball` | +4.0 | Foot flat on the ball **and** whole-robot CoM within 0.12 m (horizontal) of the ball centre. |
| `foot_on_ball` | 4.0 → 1.0 | Now only a touch bonus. |
| `com_over_ball` | +2.0 | With a foot on the ball: `exp(-(d/0.25)²)`, d = CoM-to-ball distance. Broad, so it has a gradient at 0.25 m. |
| `com_toward_ball_velocity` | +1.0 | Momentum: with a foot on the ball but CoM not yet over it (d > 0.06 m), CoM speed toward the ball, 0 to 1 at 0.5 m/s. |
| `push_up_velocity` | +0.5 | Pushing up: with a foot on the ball and base below the standing-tall height, upward CoM speed, 0 to 1 at 0.5 m/s. |
| `xy_centering` | → 0 | Centering on the env origin works against moving over the ball. |
| `single_leg_stance` test | | Now also needs CoM within 0.12 m of the ball centre. |

Also registered `-Mix0` (drops only) and `-Mix100` (mounts only) as diagnostic tasks to separate the two skills. Check on the old `mb2-base` mount spawns: `foot_on_ball` 1.00 (now ×1.0), `support_on_ball` 0, `com_over_ball` 0.34, `stance` 0.

Planned runs (4 parallel, from scratch, 8000 it): `mb3-com` (momentum weights 0), `mb3-mom` (defaults), `mb3-mix100` (mount only), `mb3-mix0` (drop only). Results: _TBD._

## Open questions

- Which mount / drop ratio works best (`-Mix30`, `-Mix70` are registered, not run).
- Whether the 5× smoothness result changes when applied by resuming a mounted policy.
- Real size 1 ball radius and mass.
