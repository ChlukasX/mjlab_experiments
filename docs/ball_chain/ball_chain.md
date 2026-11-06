# Ball Chain (phases 5–6)

Branches `feat/ball-chain` (eval task) and `docs/ball-chain` (final docs), both deleted. Commits: `git log --oneline --grep '^ball-chain:'`.

**Status (Nov 6 2026): planned.** The task `Mjlab-Piplus-Ball-Chain` was dropped from `main`. The mount -> balance chain, its player `scripts/play_chain.py` and the task are in the tag `two-policy-chain` (commit `746982e`); check it out to run them (`git switch --detach two-policy-chain`). Approach and locomotion are not started.

Full pipeline: find ball → walk → set stance → mount → balance.

```
perception (noisy ball pos in base frame)
   ├─► Approach policy ──vx,vy,ωz──► frozen Locomotion policy ──► 20 joints
   └─► Mount+Balance policy ──► 20 joints
Switcher: dist-to-ball < d_stand → hand off from walk to mount
```

## Mount -> balance in the viewer (Oct 13 2026)

This section describes the state of the tag `two-policy-chain`. The task and the script are not on `main`.

The last two links of the chain can be watched now. Task `Mjlab-Piplus-Ball-Chain` is the mount scene (standing beside a randomized ball, joint targets clipped); `scripts/play_chain.py` plays it with two policies and a switch:

```bash
uv run python scripts/play_chain.py                 # viser on http://localhost:8080, 1 env
uv run python scripts/play_chain.py --num-envs 16   # more balls sizes at once
uv run python scripts/play_chain.py --steps 1000 --num-envs 512   # no viewer: counts switches and falls
```

- Mount policy (`mm3-c10-s16`) until the stance (free sole 6 cm clear, CoM over the ball, flat sole) has been held for 0.2 s, then the balance policy (`bfm1-warm`) until the env resets (a fall, or Reset in the viewer). The switch and the reset are printed for env 0. `--mount-ckpt`, `--balance-ckpt`, `--hold`, `--switch-clearance` change the pair and the switch.
- The clip on the joint targets is lifted for an env at its switch (the balance policy was trained unbounded) and restored at its reset.
- `uv run play Mjlab-Piplus-Ball-Chain` would run a single policy; use the script.
- Measured success of this pair: 70% (`scripts/eval_handoff.py` (removed; [git history](../README.md#conventions-shared-by-all-experiments)), see [balance_from_mount](../balance_from_mount/balance_from_mount_log.md)). Headless check, 512 envs, 20 s with resets after falls: 834 switches, 324 falls after a switch.

Update Oct 14 2026: the defaults are now mount `mm7-exc200-ar2` and balance `bs4-exc100-ar2` (both with clipped joint targets, fitted head box, randomized contact stiffness): 91% hand-off success; the mount policy alone reaches 87% (`uv run play Mjlab-Piplus-Ball-MountMin-Clip --checkpoint-file <mm7-exc200-ar2 model>` shows it without the switch). `--balance-unbounded` is for the older balance policies trained without the clip (`bfm1-*`).

Approach and locomotion are not part of it yet: the robot is spawned at the stand-off pose.

## Intent

- Eval task (the old `Mjlab-Piplus-Ball-Chain` was dropped; new work starts a new task) with a switcher running approach → mount/balance.
- Metric: full-chain success rate under pushes, turf and ball randomisation. Hand-off failures show up here.
- Final write-up replaces these placeholders with measured results.

Components: [locomotion_s2r.md](../locomotion_s2r/locomotion_s2r.md), [ball_perception.md](../ball_perception/ball_perception.md), [ball_approach.md](../ball_approach/ball_approach.md), [ball_mount.md](../ball_mount/ball_mount.md), [mount_feasibility.md](../mount_feasibility/mount_feasibility.md).
