# Ball Chain (phases 5–6)

Branches: `feat/ball-chain` (eval task) and `docs/ball-chain` (final docs). Both branches are deleted. Commits: `git log --oneline --grep '^ball-chain:'`.

**Status (Nov 6 2026): planned.** The task `Mjlab-Piplus-Ball-Chain` was dropped from `main`. The tag `two-policy-chain` (commit `746982e`) holds the mount -> balance chain, its player `scripts/play_chain.py`, and the task. To run them, check out the tag (`git switch --detach two-policy-chain`). Work on approach and locomotion has not started.

Full pipeline: find ball → walk → set stance → mount → balance.

```
perception (noisy ball pos in base frame)
   ├─► Approach policy ──vx,vy,ωz──► frozen Locomotion policy ──► 20 joints
   └─► Mount+Balance policy ──► 20 joints
Switcher: dist-to-ball < d_stand → hand off from walk to mount
```

## Mount -> balance in the viewer (Oct 13 2026)

This section describes the state of the tag `two-policy-chain`. The task and the script are not on `main`.

The viewer shows the last two links of the chain. The task `Mjlab-Piplus-Ball-Chain` is the mount scene: the robot stands beside a randomized ball and the joint targets are clipped. The script `scripts/play_chain.py` plays the task with two policies and a switch:

```bash
uv run python scripts/play_chain.py                 # viser on http://localhost:8080, 1 env
uv run python scripts/play_chain.py --num-envs 16   # more balls sizes at once
uv run python scripts/play_chain.py --steps 1000 --num-envs 512   # no viewer: counts switches and falls
```

- The script runs the mount policy (`mm3-c10-s16`) until the robot holds the stance for 0.2 s. The stance means: free sole 6 cm clear, centre of mass (CoM) over the ball, flat sole. Then the script runs the balance policy (`bfm1-warm`) until the environment (env) resets. A reset happens after a fall, or when you press Reset in the viewer. The script prints the switch and the reset for env 0. The options `--mount-ckpt`, `--balance-ckpt`, `--hold`, and `--switch-clearance` change the pair and the switch.
- At the switch of an env, the script lifts the clip on the joint targets of that env, because the balance policy was trained unbounded. At the reset of the env, the script restores the clip.
- The command `uv run play Mjlab-Piplus-Ball-Chain` runs a single policy. Use the script.
- Measured success of this pair: 70% (`scripts/eval_handoff.py` (removed; [git history](../README.md#conventions-shared-by-all-experiments)), see [balance_from_mount](../balance_from_mount/balance_from_mount_log.md)). Headless check: 512 envs, 20 s, with resets after falls. Result: 834 switches, 324 falls after a switch.

Update Oct 14 2026: The defaults are now the mount policy `mm7-exc200-ar2` and the balance policy `bs4-exc100-ar2`. Both have clipped joint targets, a fitted head box, and randomized contact stiffness. The hand-off success is 91%. The mount policy alone reaches 87%. The command `uv run play Mjlab-Piplus-Ball-MountMin-Clip --checkpoint-file <mm7-exc200-ar2 model>` shows the mount policy without the switch. Use `--balance-unbounded` for the older balance policies trained without the clip (`bfm1-*`).

Approach and locomotion are not part of the script yet. The script spawns the robot at the stand-off pose.

## Intent

- The eval task uses a switcher that runs approach → mount/balance. The old `Mjlab-Piplus-Ball-Chain` was dropped. New work starts a new task.
- The metric is the full-chain success rate under pushes, turf, and ball randomisation. Hand-off failures show up in this metric.
- The final write-up replaces these placeholders with measured results.

Components: [locomotion_s2r.md](../locomotion_s2r/locomotion_s2r.md), [ball_perception.md](../ball_perception/ball_perception.md), [ball_approach.md](../ball_approach/ball_approach.md), [ball_mount.md](../ball_mount/ball_mount.md), [mount_feasibility.md](../mount_feasibility/mount_feasibility.md).
