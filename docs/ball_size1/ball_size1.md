# Size 1 ball (mount and balance)

Branch: `feat/mount-feasibility` (deleted; shared with [mount_feasibility](../mount_feasibility/mount_feasibility.md)). Commits: `git log --oneline --grep '^ball-mount:'`. — **done**

> **Correction (Oct 15 2026).** The size 1 mount runs did not mount the ball either. The run `mount-size1-flat-v2-cont` `model_9998` was re-evaluated with the sole-based test (`scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-Mount-Size1-Flat --free-foot-clearance 0.03 --probe`). It has 0% success. The centre of mass is 0.235 m from the ball. The other sole is on the floor 97% of the time. The robot stands on tiptoe beside the ball, as in [mount_feasibility](../mount_feasibility/mount_feasibility.md). The drop-balance run (`balance-size1-flat-v1`) is not affected.

This document describes the mount and balance tasks of [mount_feasibility](../mount_feasibility/mount_feasibility.md) on a FIFA (International Federation of Association Football) size 1 football instead of a size 5 football. The radius 0.07 m and the mass 0.14 kg are assumptions for a mini ball (about 44 cm circumference). Check them against the real ball (`BALL_SIZES` in `tasks/ball_mount/piplus_ball_mount_env_cfg.py`).

| Task | Description | Log dir |
|------|-------------|---------|
| `Mjlab-Piplus-Ball-Mount-Size1` | Step from the floor onto the ball, hold single-leg stance | `logs/rsl_rl/piplus_ball_mount_size1` |
| `Mjlab-Piplus-Ball-Mount-Size1-Flat` | Same, sole must be flat on the ball | `logs/rsl_rl/piplus_ball_mount_size1_flat` |
| `Mjlab-Piplus-Ball-Balance-Size1` | Drop onto the ball and balance (flat sole); the drop starts with the base 0.5 m above the ball top in the first run, now with the sole 3 cm above | `logs/rsl_rl/piplus_ball_balance_size1` |

The ball is easier to step onto (top at 0.14 m instead of 0.22 m). The ball is harder to balance on. A flat sole means that the sole normal follows the ball surface normal. It does not mean a flat contact patch, because the sole is longer than the ball.

Runs and results: [ball_size1_log.md](ball_size1_log.md).
