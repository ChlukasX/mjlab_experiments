# Ball Chain (phases 5–6)

Branch: `feat/ball-chain` (eval task), `docs/ball-chain` (final docs) — **placeholder, not started**

Full pipeline: find ball → walk → set stance → mount → balance.

```
perception (noisy ball pos in base frame)
   ├─► Approach policy ──vx,vy,ωz──► frozen Locomotion policy ──► 20 joints
   └─► Mount+Balance policy ──► 20 joints
Switcher: dist-to-ball < d_stand → hand off from walk to mount
```

## Intent

- Eval task `Mjlab-Piplus-Ball-Chain` with a switcher running approach → mount/balance.
- Metric: full-chain success rate under pushes, turf and ball randomisation. Hand-off failures show up here.
- Final write-up replaces these placeholders with measured results.

Components: [locomotion_s2r.md](../locomotion_s2r/locomotion_s2r.md), [ball_perception.md](../ball_perception/ball_perception.md), [ball_approach.md](../ball_approach/ball_approach.md), [ball_mount.md](../ball_mount/ball_mount.md), [mount_feasibility.md](../mount_feasibility/mount_feasibility.md).
