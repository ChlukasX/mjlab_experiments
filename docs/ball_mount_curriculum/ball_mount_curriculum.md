# Ball mount curriculum

Branch: `feat/ball-mount-curriculum` — **in progress**

Goal: a mount policy that takes the robot from standing beside the ball to a single-leg stance on top of it, on a randomized ball (radius 0.07 to 0.11 m). It is followed by a balance policy (see [ball_mount](../ball_mount/ball_mount.md)), so the mount only has to reach a state the balance policy can hold. Runs and results: [ball_mount_curriculum_log.md](ball_mount_curriculum_log.md).

## Why

Eight rounds of reward changes ([ball_mount_log](../ball_mount/ball_mount_log.md)) never produced a mount. The best policy reaches the straddle (foot flat on the ball, CoM over it, 100% of episodes within 1.5 s) and stays there: the second foot hovers 2.5 cm off the floor (needs 10 cm) and the base is 9 cm too low. A hand-off test showed the existing balance policy cannot take over from the straddle (0% survive). Diagnosis: the straddle is a stable plateau that already earns every dense reward, leaving it means single support on a rolling ball, and rollouts of 24 steps (0.5 s) are short for a 2 to 3 s maneuver.

## Design

All tasks: mounts only, randomized ball, bounded joint targets, entropy 0.003, `num_steps_per_env` 64, `gamma` 0.995, `lam` 0.97, 6000 iterations (use 2048 envs: the same number of samples as 8000 iterations of 4096 envs at 24 steps). Weights from the `mb4` runs: `support_on_ball` 2, `single_leg_stance` 8, `free_foot_lift` 3.

- **Ball lock then release.** The ball's rolling-friction coefficient is raised to 0.2 m (torque <= mu * normal force), so it barely rolls under the robot's weight; it is held for 1000 iterations, then released linearly over 3000 (free for the last 2000). Rolling friction is a constraint and stays stable; joint damping on the light ball blew up the simulation. With random actions on a dropped robot the lock cuts ball travel from 0.142 m to about 0.06 m. `ball_rolling_resistance` in `tasks/ball_mount/events.py`. Play and eval configs have no lock (free ball).
- **Potential-based progress reward.** `mount_potential` pays `gamma * phi(s') - phi(s)` (weight 2000, rewards are scaled by dt = 0.02), so resting in the straddle pays nothing and only progress pays. `phi = 0.15 * p_foot + 0.85 * c * (0.3 + 0.35 * h + 0.35 * lift)` with `p_foot` = closeness of the nearest foot to the ball top, `c` = closeness of the CoM to the ball centre (`exp(-(d / 0.25)^2)`), `h` = base height above the ball top over 0.30 m, `lift` = clearance of the lower foot over 10 cm. In this variant the standing terms that pay for the straddle are set to 0 (`support_on_ball`, `com_over_ball`, `stand_tall`, `free_foot_lift`, `foot_on_ball`, `foot_to_ball_top`, `foot_lift`, `sole_flat`, the two velocity terms); `single_leg_stance` (8) stays as the stance reward.
- Not in round 1: reset from straddle states (reverse curriculum), RND exploration bonus, symmetry augmentation; a model-based reference trajectory is the fallback if two rounds stall.

## Tasks

| Task | Ball lock | Potential reward | Log dir |
|------|-----------|------------------|---------|
| `Mjlab-Piplus-Ball-MountCurr-Release` | locked, then released | no | `logs/rsl_rl/piplus_ball_mountcurr_release` |
| `Mjlab-Piplus-Ball-MountCurr-ReleasePot` | locked, then released | yes | `logs/rsl_rl/piplus_ball_mountcurr_releasepot` |
| `Mjlab-Piplus-Ball-MountCurr-Pot` | none | yes | `logs/rsl_rl/piplus_ball_mountcurr_pot` |
| `Mjlab-Piplus-Ball-MountCurr-Locked` | locked all training (feasibility probe) | yes | `logs/rsl_rl/piplus_ball_mountcurr_locked` |

Config: `tasks/ball_mount_curriculum/piplus_ball_mountcurr_env_cfg.py`. Evaluate with `scripts/eval_mountbalance.py --task <task>` (success = stance with CoM over the ball, flat sole, free foot 10 cm clear, held over the last 3 s).

## Success criterion

At least 50% success on mount spawns on the free ball. If `Locked` fails too, the stance is not reachable with these rewards even on a fixed ball, and the next step is the model-based search for a reference mount motion.
