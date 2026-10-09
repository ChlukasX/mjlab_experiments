# Ball mount curriculum: run log

One row per run, newest last; add a row at launch and fill in the result when it finishes. Design: [ball_mount_curriculum.md](ball_mount_curriculum.md). Shared conventions: [../README.md](../README.md).

## Round 1

From scratch, mounts only, 2048 envs, 6000 iterations, settings as in the [design doc](ball_mount_curriculum.md) (`num_steps_per_env` 64, `gamma` 0.995, entropy 0.003, bounded actions, rebalanced weights). Launched Oct 9 2026 at commit `7290a9a`; 2 runs on the local GPU, 2 on cl06. All four share the log dirs `logs/rsl_rl/piplus_ball_mountcurr_<variant>`.

| Run | Task | Where | wandb | Outcome |
|-----|------|-------|-------|---------|
| `mc1-release` | `MountCurr-Release` | local | `g3448ra4` | In progress. At it 849 (ball still locked): reward 200, episode 965/1000, `support_on_ball` 1.13, `com_over_ball` 1.45, `single_leg_stance` 0: the straddle again, even on a locked ball. |
| `mc1-releasepot` | `MountCurr-ReleasePot` | local | `0yz48kxc` | **Stopped at it ~1120** (degenerate, see below). |
| `mc1-pot` | `MountCurr-Pot` | cl06 | `kbj6ti7w` | **Stopped at it ~1840** (degenerate). |
| `mc1-locked` | `MountCurr-Locked` | cl06 | `6pzgviha` | **Stopped at it ~1820** (degenerate). |

**The three potential-reward runs learned an exploit.** Episode length 22 steps (0.45 s), `upright` 0.02, reward flat at ~29 from iteration ~300, and ~90 of ~100 episodes per iteration ended in `fell_over`. The policy throws itself at the ball, collects the potential gain on the way and dies. This is the known flaw of potential-based shaping with early termination: the shaped return telescopes to `gamma^T phi(s_T) - phi(s_0)`, so ending the episode in a high-potential state keeps the gain at no cost; the correct form needs `phi(terminal) = 0`. Fix in `mount_potential` (commit below): the potential of a terminated state is 0, so a fall costs `-phi(s)`; a test with violent random actions gives about −4.8 on terminating steps against −0.3 otherwise.

### Round 1b: potential reward with terminal potential 0 (stopped)

Relaunched the three potential runs as `mc1b-*` (same tasks and settings, fixed reward), Oct 9 2026, commit `37ca427`.

| Run | Task | Where | wandb | Outcome |
|-----|------|-------|-------|---------|
| `mc1b-releasepot` | `MountCurr-ReleasePot` | local | `yfy78x6p` | **Stopped at it ~1500**: stands still, never goes to the ball |
| `mc1b-pot` | `MountCurr-Pot` | cl06 | `dpumn1k9` | **Stopped at it ~1560**: same |
| `mc1b-locked` | `MountCurr-Locked` | cl06 | `7arp8izu` | **Stopped at it ~1520**: same, so it did not test the locked-ball stance |

With the terminal potential fixed the dive-and-die exploit is gone (`fell_over` ~0, episode length 1000), but all three converge by it ~400 to a **do-nothing optimum**: reward flat at 36 (almost all from `upright`), `mount_potential` slightly negative (-0.4 to -0.5, the `gamma < 1` leak of standing still), `support_on_ball`, `com_over_ball` and `single_leg_stance` 0. A fall now costs the full potential and standing is safe and pays, so the policy never explores toward the ball. This is a stall, not a hint about feasibility.

`mc1-release` (no potential, rebalanced rewards) at it 1956, ball releasing (level 0.68): reward 221, episode 946/1000, `support_on_ball` 1.73 of 2, `com_over_ball` 1.65 of 2, `single_leg_stance` 0.000 in every window, including the first 1000 iterations with the ball locked. The straddle persists on a locked ball, so the lock alone does not get the second foot off the floor.

**Takeaway.** Neither the lock nor the potential reward gets PPO to the stance, and the locked-ball feasibility question is still open. Next: a model-based search for a mount trajectory that does not depend on PPO exploration (`scripts/mount_cem.py`), on a fixed ball, free and locked. See [Model-based search](#model-based-search).

## Model-based search

`scripts/mount_cem.py`: cross-entropy search over open-loop joint-target splines (16 knots over 4 s, 320 parameters), 4096 identical worlds per iteration (mount spawn, size 5 ball, no randomization or noise, no pushes), scored with the task's own reward (progress potential with terminal potential 0, plus the stance reward). It needs no PPO exploration, so it answers whether a mount exists for this robot and ball. `--lock` uses the ball with large rolling resistance. `scripts/replay_cem.py` replays the best trajectory and prints the state along it. Success = some sample holds the single-leg stance (CoM over the ball, flat sole, free foot 10 cm clear) for 1 s.

| Search | Ball | Settings | Result |
|--------|------|----------|--------|
| v1 (Oct 9 2026) | free, locked | 120 it, init std 0.4, no settling | No stance. Best score 23.2 (free) and 23.7 (locked); mean score -5.9 -> +9; samples alive at the end 0% -> 65%. **The best trajectory is an exploit:** replay shows the robot 0.40 m from the ball with both feet on the floor for 3 s, then in the last half second its body swings over the ball (CoM 0.017 m from the ball centre, potential 0.54) with neither foot on it; the horizon ends before the fall, so it is never penalised. |
| v2 | locked (free stopped at it ~95) | 300 it, init std 0.4, the last action held for 1.5 s after the motion | No stance. Locked ball: best score 19.1 (plateau from it ~180), 28% of samples alive at the end, longest stance 0.00 s. **Another exploit:** the robot stays 0.40 m from the ball for 4 s, then in the last second leans its body over the ball (CoM 0.021 m from the ball centre, base 0.086 m above the ball top, potential 0.45) with neither foot on it; the pose triggers no termination (tilt < 60 deg, base > 0.30 m). Cause: the potential's CoM, height and lift terms did not require a foot on the ball. The free-ball search was stopped at it ~95 with the same trajectory type. |
| v3 | free, locked | 300 it, init std 0.4, settling 2.5 s, **potential gated by a foot on the ball** (`phi = 0.15 p + 0.85 p c (0.3 + 0.35 h + 0.35 l)`, `p` = closeness of the nearest foot to the ball top): the v2 trajectory now scores 0.00 to 0.01 | _TBD_ |

