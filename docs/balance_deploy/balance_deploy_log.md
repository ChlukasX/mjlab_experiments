# Balance policy for the real robot: run log

One row per audit or run; add a row at launch and fill in the result when it finishes. Design: [balance_deploy.md](balance_deploy.md). Shared conventions: [../README.md](../README.md).

## Action audit (Oct 9 2026)

`scripts/audit_policy.py`, 256 envs, 10 s, play config (observation noise off, pushes on), deterministic policy.

| Policy | Checkpoint | Survived 10 s | mean \|action\| | 95th pct | max | targets outside joint limits | PD torque saturated (legs / all) | mean \|action change\| per step |
|--------|-----------|---------------|----------------|----------|-----|------------------------------|-----------------------------------|-------------------------------|
| `Ball-Small` (size 5) | `piplus_ball_small/2026-10-01_09-30-11_sim2real-armcol-cont3/model_20996` | 99% | 27.5 | 115 | 192 | 55% | 57% / 59% | 0.48 |
| `Ball-Small-Turf` | `piplus_ball_small/2026-10-01_11-52-17_turf-from-flat/model_25995` | 97% | 55.6 | 221 | 278 | 61% | 56% / 63% | 0.51 |
| `Ball-Balance-Size1` | `piplus_ball_balance_size1/2026-10-05_11-47-46_balance-size1-flat-v1/model_9999` | 87% | 26.0 | 97 | 119 | 63% | 54% / 62% | 0.37 |

(The size 1 policy was trained with the older, higher drop and is audited on the current low-drop spawn.)

All three are far above the proposed limits: targets tens of radians from the home pose, torque saturated in more than half of the joint-steps. The same pattern was seen in the saved action-noise parameters of every ball-balancing policy (std 2.4 to 3.9, against 0.62 for locomotion).

## Next

- Drops-only ablation for a bounded-action balance policy (see the plan in the design doc). Not launched yet: both GPUs are busy with the mount curriculum round 1 ([ball_mount_curriculum_log.md](../ball_mount_curriculum/ball_mount_curriculum_log.md)).
