# Size 1 ball (mount and balance)

Branch: `feat/mount-feasibility` (deleted; shared with [mount_feasibility](../mount_feasibility/mount_feasibility.md)). Commits: `git log --oneline --grep '^ball-mount:'`. — **done**

> **Correction (Oct 15 2026).** The size 1 mount runs did not mount the ball either: `mount-size1-flat-v2-cont` `model_9998`, re-evaluated with the sole-based test (`scripts/eval_mountbalance.py --task Mjlab-Piplus-Ball-Mount-Size1-Flat --free-foot-clearance 0.03 --probe`), has 0% success, the centre of mass 0.235 m from the ball and the other sole on the floor 97% of the time: tiptoe beside the ball, as in [mount_feasibility](../mount_feasibility/mount_feasibility.md). The drop-balance run (`balance-size1-flat-v1`) is not affected.

The mount and balance tasks of [mount_feasibility](../mount_feasibility/mount_feasibility.md) on a FIFA size 1 football instead of size 5. Radius 0.07 m and mass 0.14 kg are assumptions for a mini ball (about 44 cm circumference); check them against the real ball (`BALL_SIZES` in `tasks/ball_mount/piplus_ball_mount_env_cfg.py`).

| Task | Description | Log dir |
|------|-------------|---------|
| `Mjlab-Piplus-Ball-Mount-Size1` | Step from the floor onto the ball, hold single-leg stance | `logs/rsl_rl/piplus_ball_mount_size1` |
| `Mjlab-Piplus-Ball-Mount-Size1-Flat` | Same, sole must be flat on the ball | `logs/rsl_rl/piplus_ball_mount_size1_flat` |
| `Mjlab-Piplus-Ball-Balance-Size1` | Drop onto the ball and balance (flat sole); the drop starts with the base 0.5 m above the ball top in the first run, now with the sole 3 cm above | `logs/rsl_rl/piplus_ball_balance_size1` |

The ball is easier to step onto (top at 0.14 m instead of 0.22 m) and harder to balance on; a flat sole means the sole normal follows the ball surface normal, not a flat contact patch, because the sole is longer than the ball.

Runs and results: [ball_size1_log.md](ball_size1_log.md).
