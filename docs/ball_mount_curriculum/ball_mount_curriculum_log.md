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

### Round 1b: potential reward with terminal potential 0

Relaunched the three potential runs as `mc1b-*` (same tasks and settings, fixed reward). Results: _TBD._
