# mount_straight: mount and stance without the twisted support leg

Branch: `feat/mount-straight` (deleted; created from `feat/mount-free-leg`). Commits: `git log --oneline --grep '^mount-straight:'`. Task `Mjlab-Piplus-Ball-MountMin-BlindOnBall-Latency-Big-OneFoot-Level-NoiseRamp-Latch45-Hold-KinBall-FreeLeg-Straight`. Run log: [mount_straight_log.md](mount_straight_log.md). Predecessor: [mount_latency](../mount_latency/mount_latency.md) (rounds 7-8, `kb1`, `fl1`).

## Why

In play, the support foot of `fl1` points away from the front of the torso. Every mount policy since Oct 10 descends from one cold start, `mm1-min-clip`. The chain has ~18 warm starts in a row: `mm1` → `mm2`-`mm7` → `ms1`-`ms3` → `bob1`/`bob2` → `lat1` → `lat2-big` → `of2` → `lv1` → `nr1` → `kb1` → `fl1`. The probe `logs/probes/foot_yaw_probe.py` counts single-support steps on the ball after 5 s. The old checkpoints run in today's environment, so they make few stance steps.

| Policy | support foot | foot yaw − torso yaw (median) | support thigh (yaw) joint |
|---|---|---|---|
| `mm1-min-clip` (the cold start) | right | −161° | −135° |
| `mm7-exc200-ar2` (6 rad/s spin era) | right | +138° | −135° |
| `ms3-shell` | right | +132° | −124° |
| `fl1` | right, always | +120° | −117° |

The first cold start already turned the right leg ~135° at the thigh (yaw) joint. The later rounds only unwound it a little. <!-- STE-deviation: the source gives no number in this sentence --> The MJCF allows −2.77 to +2.45 rad (−159° to +140°). Only the weak `posture` term (weight 0.3, thigh std 0.40) resists this range. A warm start cannot unlearn a 120° leg rotation without loss of the stance. Therefore this task uses a cold start. Open question: can the real Pi Plus turn the thigh that far at all? The possible limits are cables and hard stops.

## Changes against `-FreeLeg` (`straight=True`)

- **Thigh (yaw) range ±0.6 rad** (`THIGH_LIMIT`; home is 0) in the model of this task. `_spec_straight` narrows `r_thigh_joint` / `l_thigh_joint` in the spec. The vendored MJCF and every other task stay unchanged. The target clip of the bounded actions follows the new range. `joint_pos_limits` uses the range of the model (soft limit 90%: ±0.54 rad). `action_excess` now reads the clip of the action term when the task has one (same values as before for the other bounded tasks). `scripts/export_mount.py` writes the joint limits from the model of the task instead of the MJCF. An export of this task carries ±0.6.
- **`support_foot_heading`, +1.0:** in single support (one level sole on the ball), the term is cos of the yaw between the support foot (ankle roll link x axis in the world xy plane) and the torso. The term is clamped to [0, 1].
- **Cold start**, with the staged stance clearance of `mm1`. `single_leg_stance` uses 3 cm sole clearance first. Later, the run resumes at 6 cm and at 10 cm. The latency curriculum (0-15 → 10-40 ms) and the noise curriculum (0.25× → 1×) ramp over the first 1500 iterations from step 0. The other terms are the round 5 weights of mount_latency (`single_leg_stance` 16, `upright` 6 with std 0.5, `action_rate` −0.02, `com_over_ball` 0, `action_excess` −4, `base_yaw_rate` −0.1, `torque_saturation` −2, `torque_demand` −0.5) and the `-FreeLeg` terms.

Check `logs/probes/test_straight.py`: The thigh range is ±0.6 in the model (hip pitch unchanged). A 1.5 rad target is clipped to 0.60. `action_excess` is 0.90 for this target. `support_foot_heading` is 0 outside single support. Foot and torso are aligned at home (cos ≥ 0.994).

## Evaluation

Run the mount_latency sweep (`logs/probes/after_st1.sh`: latency 10-40 ms, jitter, sensor delay, ball radius, soft motors, noise off; side, latch, kin-ball, free-leg and foot-yaw probes) against `fl1`. Target: the foot is within ~35° of the torso (the joint limit). The robot keeps the mount.

Risk: the mount was hard to learn from scratch. `mm1` needed the pruned rewards and ~2000 iterations before it mounted. Here the policy must learn the mount under latency and noise ramps. If the policy never mounts, use a short start without latency and at low noise. Then run the full task.
